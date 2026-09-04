"""Read prediction-log records into the frames the drift monitor compares.

The serving sinks write one record per prediction (see serving/sinks.py): raw input cycles,
computed features, output, threshold, model version, timestamp. This module reads them back
from wherever they landed, Blob Storage on Azure or Postgres locally, and splits each record into
three aligned frames keyed by prediction_id:

- raw:      the last cycle of the window as received, all 26 raw columns (settings included)
- features: the 99 computed features the model actually saw
- outputs:  timestamp, unit, cycle, probability, label, threshold, model version
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data.schema import RAW_COLUMNS

log = logging.getLogger("driftwatch.monitoring.logs")

OUTPUT_FIELDS = ("prediction_id", "timestamp", "model_name", "model_version", "unit", "cycle",
                 "probability", "label", "threshold", "latency_ms")


def since_hours(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def read_blob(account: str, container: str, prefix: str, since: datetime | None = None) -> list[dict]:
    """Every record under prefix/dt=.../hour=.../*.jsonl written at or after `since`."""
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient(f"https://{account}.blob.core.windows.net", credential=DefaultAzureCredential())
    client = service.get_container_client(container)
    records: list[dict] = []
    n_blobs = 0
    for blob in client.list_blobs(name_starts_with=f"{prefix.strip('/')}/"):
        if since is not None and blob.last_modified < since:
            continue
        n_blobs += 1
        text = client.get_blob_client(blob.name).download_blob().readall().decode("utf-8")
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))
    log.info("blob: %d records from %d file(s) under %s/%s", len(records), n_blobs, container, prefix)
    return _filter_since(records, since)


def read_postgres(dsn: str, since: datetime | None = None) -> list[dict]:
    """Records from the local Postgres sink (docker compose), same shape as the JSONL."""
    import psycopg

    query = "SELECT prediction_id, ts, model_name, model_version, unit, cycle, probability, label, threshold, latency_ms, raw, features FROM predictions"
    params: tuple = ()
    if since is not None:
        query += " WHERE ts >= %s"
        params = (since,)
    records = []
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(query + " ORDER BY ts", params)
        for row in cur.fetchall():
            pid, ts, name, version, unit, cycle, p, label, thr, latency, raw, features = row
            records.append({"prediction_id": str(pid), "timestamp": ts.isoformat(), "model_name": name,
                            "model_version": version, "unit": unit, "cycle": cycle, "probability": p,
                            "label": label, "threshold": thr, "latency_ms": latency, "raw": raw, "features": features})
    log.info("postgres: %d records", len(records))
    return records


def read_local(path: Path, since: datetime | None = None) -> list[dict]:
    """Records from a local JSONL file or a directory of them (e.g. a downloaded artifact)."""
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    records = []
    for file in files:
        for line in file.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    log.info("local: %d records from %d file(s) under %s", len(records), len(files), path)
    return _filter_since(records, since)


def _filter_since(records: list[dict], since: datetime | None) -> list[dict]:
    if since is None:
        return records
    kept = [r for r in records if datetime.fromisoformat(r["timestamp"]) >= since]
    if len(kept) != len(records):
        log.info("kept %d of %d records with timestamp >= %s", len(kept), len(records), since.isoformat())
    return kept


def to_frames(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split records into (raw last-cycle rows, features, outputs), aligned by prediction_id."""
    if not records:
        empty = pd.DataFrame()
        return empty, empty, empty
    raw = pd.DataFrame([{"prediction_id": r["prediction_id"], **{c: r["raw"][-1][c] for c in RAW_COLUMNS}} for r in records])
    features = pd.DataFrame([{"prediction_id": r["prediction_id"], **r["features"]} for r in records])
    outputs = pd.DataFrame([{k: r.get(k) for k in OUTPUT_FIELDS} for r in records])
    outputs["timestamp"] = pd.to_datetime(outputs["timestamp"], utc=True)
    return raw, features, outputs
