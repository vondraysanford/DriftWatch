"""Metrics API behind the dashboard.

Aggregates the prediction log (the same records the drift monitor reads) and the monitoring
outputs the workflows publish to blob storage under ``monitoring/``: drift verdicts, champion vs
challenger results, and deployment records. Read through ``monitoring.logs`` so the local Postgres
sink works too; the monitoring feeds are simply empty there.

Everything is cached in memory for a short TTL. The log is a few thousand JSONL records today,
and the dashboard polls; a blob scan per request would be silly and would slow ``/predict``'s
neighbour for nothing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd
from fastapi import APIRouter, Query, Request

from data.schema import regime_of
from monitoring import logs
from serving.config import Settings

log = logging.getLogger("driftwatch.serving.metrics")

router = APIRouter(prefix="/api", tags=["metrics"])
TTL_SECONDS = 60
MAX_WINDOW_HOURS = 24 * 30
HISTOGRAM_BINS = 10
MONITORING_PREFIX = "monitoring"
MONITORING_KINDS = ("verdicts", "challenges", "deployments")


class MetricsStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, Any]] = {}

    def _cached(self, key: str, loader: Callable[[], Any]) -> Any:
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < TTL_SECONDS:
                return hit[1]
        value = loader()
        with self._lock:
            self._cache[key] = (time.monotonic(), value)
        return value

    def records(self) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(hours=MAX_WINDOW_HOURS)
        if self.settings.sink == "blob":
            return self._cached("records", lambda: logs.read_blob(
                self.settings.storage_account, self.settings.container, self.settings.prefix, since))  # type: ignore[arg-type]
        return self._cached("records", lambda: logs.read_postgres(self.settings.database_url, since))  # type: ignore[arg-type]

    def monitoring(self, kind: str) -> list[dict]:
        """JSON documents the workflows uploaded under monitoring/<kind>/, oldest first."""
        if self.settings.sink != "blob":
            return []

        def load() -> list[dict]:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient

            service = BlobServiceClient(f"https://{self.settings.storage_account}.blob.core.windows.net",
                                        credential=DefaultAzureCredential())
            container = service.get_container_client(self.settings.container)
            docs = []
            for blob in container.list_blobs(name_starts_with=f"{MONITORING_PREFIX}/{kind}/"):
                try:
                    doc = json.loads(container.get_blob_client(blob.name).download_blob().readall())
                except (ValueError, OSError) as exc:
                    log.warning("skipping %s: %s", blob.name, exc)
                    continue
                doc["_published_at"] = blob.last_modified.isoformat()
                doc["_name"] = blob.name.rsplit("/", 1)[-1]
                docs.append(doc)
            docs.sort(key=lambda d: d["_published_at"])
            return docs

        return self._cached(f"monitoring:{kind}", load)


def _outputs(records: list[dict], hours: float) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["prediction_id", "timestamp", "unit", "cycle", "probability", "label", "model_version", "regime"])
    _, _, outputs = logs.to_frames(records)
    outputs["regime"] = outputs["unit"].map(regime_of)
    outputs["model_version"] = outputs["model_version"].astype(str)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    return outputs[outputs["timestamp"] >= since]


def _latest(docs: list[dict]) -> dict | None:
    return docs[-1] if docs else None


@router.get("/summary")
def summary(request: Request, hours: float = Query(24 * 7, gt=0, le=MAX_WINDOW_HOURS)) -> dict:
    store: MetricsStore = request.app.state.metrics
    info = request.app.state.model.info
    outputs = _outputs(store.records(), hours)
    verdict, challenge, deployment = (_latest(store.monitoring(k)) for k in MONITORING_KINDS)
    return {
        "champion": {"name": info.name, "version": info.version, "run_name": info.run_name, "model_kind": info.model_kind,
                     "threshold": info.threshold, "test_roc_auc": info.metrics.get("test_roc_auc"), "data_version": info.data_version},
        "window_hours": hours,
        "predictions": {
            "total": int(len(outputs)),
            "engines": int(outputs["unit"].nunique()) if len(outputs) else 0,
            "by_regime": {k: int(v) for k, v in outputs["regime"].value_counts().items()},
            "by_model_version": {k: int(v) for k, v in outputs["model_version"].value_counts().items()},
            "positive_rate": round(float(outputs["label"].mean()), 4) if len(outputs) else None,
            "first": outputs["timestamp"].min().isoformat() if len(outputs) else None,
            "last": outputs["timestamp"].max().isoformat() if len(outputs) else None,
        },
        "latest_verdict": None if not verdict else {
            "generated_at": verdict.get("generated_at"), "drift": verdict.get("drift"), "reason": verdict.get("reason"),
            "reference_set": verdict.get("reference_set"), "records": (verdict.get("current") or {}).get("records")},
        "latest_challenge": None if not challenge else {
            "published_at": challenge["_published_at"], "champion_version": (challenge.get("champion") or {}).get("version"),
            "champion_roc_auc": (challenge.get("champion") or {}).get("roc_auc"),
            "challenger_roc_auc": (challenge.get("challenger") or {}).get("roc_auc"),
            "registered_version": challenge.get("registered_version")},
        "latest_deployment": None if not deployment else {k: deployment.get(k) for k in ("timestamp", "model_version", "image_tag", "promotion", "run_url")},
        "sink": store.settings.sink,
    }


@router.get("/predictions")
def predictions(request: Request, hours: float = Query(24, gt=0, le=MAX_WINDOW_HOURS), recent: int = Query(25, ge=0, le=200)) -> dict:
    store: MetricsStore = request.app.state.metrics
    outputs = _outputs(store.records(), hours)
    if outputs.empty:
        return {"window_hours": hours, "per_hour": [], "histogram": [], "recent": []}

    hourly = outputs.assign(hour=outputs["timestamp"].dt.floor("h"))
    pivot = hourly.pivot_table(index="hour", columns="regime", values="prediction_id", aggfunc="count", fill_value=0)
    per_hour = [{"hour": idx.isoformat(), **{str(c): int(row[c]) for c in pivot.columns}, "total": int(row.sum())}
                for idx, row in pivot.iterrows()]

    edges = [i / HISTOGRAM_BINS for i in range(HISTOGRAM_BINS + 1)]
    counts = pd.cut(outputs["probability"], bins=edges, include_lowest=True).value_counts().sort_index()
    histogram = [{"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "low": edges[i], "count": int(counts.iloc[i])} for i in range(HISTOGRAM_BINS)]

    latest = outputs.sort_values("timestamp", ascending=False).head(recent)
    recent_rows = [{"timestamp": r.timestamp.isoformat(), "unit": int(r.unit), "cycle": int(r.cycle), "regime": r.regime,
                    "probability": round(float(r.probability), 4), "label": int(r.label), "model_version": r.model_version,
                    "latency_ms": r.latency_ms} for r in latest.itertuples()]
    return {"window_hours": hours, "per_hour": per_hour, "histogram": histogram, "recent": recent_rows}


@router.get("/drift")
def drift(request: Request) -> dict:
    store: MetricsStore = request.app.state.metrics
    verdicts = []
    for v in store.monitoring("verdicts"):
        parts = [{"regime": p.get("regime"), "compared_to": p.get("compared_to"), "records": p.get("records"),
                  "raw_share": (p.get("raw") or {}).get("drifted_share"), "raw_count": (p.get("raw") or {}).get("drifted_count"),
                  "raw_columns": (p.get("raw") or {}).get("columns_monitored"),
                  "features_share": (p.get("features") or {}).get("drifted_share"),
                  "settings_drifted": p.get("settings_drifted", []), "drift": p.get("drift"), "skipped": p.get("skipped")}
                 for p in v.get("parts", [])]
        perf = v.get("performance") or {}
        verdicts.append({
            "generated_at": v.get("generated_at"), "published_at": v["_published_at"], "drift": v.get("drift"),
            "insufficient_data": v.get("insufficient_data", False), "reference_set": v.get("reference_set"),
            "records": (v.get("current") or {}).get("records"), "window_hours": (v.get("current") or {}).get("window_hours"),
            "reason": v.get("reason"), "parts": parts,
            "roc_auc": perf.get("roc_auc_current"), "roc_auc_reference": perf.get("roc_auc_reference"),
            "roc_auc_by_regime": {k: s.get("roc_auc_current") for k, s in (perf.get("by_regime") or {}).items()},
        })
    return {"verdicts": verdicts}


@router.get("/performance")
def performance(request: Request) -> dict:
    store: MetricsStore = request.app.state.metrics
    labeled = []
    for v in store.monitoring("verdicts"):
        perf = v.get("performance") or {}
        for regime, s in (perf.get("by_regime") or {}).items():
            if s.get("roc_auc_current") is not None:
                labeled.append({"generated_at": v.get("generated_at"), "regime": regime, "roc_auc": s["roc_auc_current"],
                                "precision": s.get("precision_current"), "recall": s.get("recall_current"),
                                "records": s.get("labeled_records"), "reference_set": v.get("reference_set")})
    challenges = [{
        "published_at": c["_published_at"], "champion_version": (c.get("champion") or {}).get("version"),
        "champion_roc_auc": (c.get("champion") or {}).get("roc_auc"),
        "champion_by_regime": {k[len("roc_auc_"):]: v for k, v in (c.get("champion") or {}).items() if k.startswith("roc_auc_")},
        "challenger_run": (c.get("challenger") or {}).get("run_name"), "challenger_roc_auc": (c.get("challenger") or {}).get("roc_auc"),
        "challenger_by_regime": {k[len("roc_auc_"):]: v for k, v in (c.get("challenger") or {}).items() if k.startswith("roc_auc_")},
        "gain": c.get("gain"), "margin": c.get("margin"), "registered_version": c.get("registered_version"),
    } for c in store.monitoring("challenges")]
    deployments = [{k: d.get(k) for k in ("timestamp", "model_version", "image_tag", "promotion", "run_url")}
                   for d in store.monitoring("deployments")]
    return {"labeled": labeled, "challenges": challenges, "deployments": deployments}
