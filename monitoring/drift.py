"""Compare what the model is seeing in production with what it was trained on, and say whether
that difference is drift.

    python -m monitoring.drift --logs-from blob --since-hours 24 --labels-from data/interim/fd002_train.parquet
    python -m monitoring.drift --logs-from postgres --dsn postgresql://driftwatch@localhost/driftwatch
    python -m monitoring.drift --logs-from local --path monitoring/out/logs/

Reference: the champion's training engines (raw cycles and computed features). ``--reference-set``
follows the champion's ``data`` tag in the registry: ``fd001`` until a model retrained on the
replayed regime is promoted, ``fd001+fd002`` after.

The comparison is made per regime. Traffic is split by the regime its unit numbers identify
(FD001 units 1..100, replayed FD002 units 1001..1260), and each part is compared against the
champion's training engines for that regime. A regime the champion never trained on is compared
against everything it did train on, which is exactly when drift should be declared. Comparing a
pure-FD002 window against a mixed FD001+FD002 reference would flag the composition difference
even though the retrained model handles both regimes; that false alarm is what this avoids.

For each part, two Evidently reports are written (raw inputs: operating settings plus the sensors
the model uses; and the 99 features) and:

- drift is declared when the share of drifted raw-input columns reaches --drift-share, OR any
  operating setting drifts (constant within a regime, so any movement is a regime change);
- sensors that are constant in the reference but vary in production are listed separately,
  because Evidently has no reference distribution to compare them against and they are the
  loudest possible signal;
- where labels exist (replayed engines run to failure, so labels are derivable by unit and cycle),
  the model's ROC-AUC on that traffic is reported next to the reference number.

No verdict is issued on too little data: a part below --min-records or --min-engines is reported
but does not drive the verdict (25 repeats of two windows once produced a false alarm). Exit code
is 0 in every outcome. The verdict file is the contract with monitoring.retrain_trigger; this
script never dispatches anything itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from data.features import DROPPED_SENSORS, KEPT_SENSORS, LABEL_HORIZON, feature_columns
from data.schema import CYCLE, REPLAY_UNIT_OFFSET, RUL, SETTING_COLUMNS, UNIT, regime_of
from monitoring import logs

log = logging.getLogger("driftwatch.monitoring.drift")

REFERENCE_FILES = {
    "fd001": (Path("data/interim/fd001_train.parquet"), Path("data/processed/train.parquet")),
    "fd002": (Path("data/interim/fd002_train.parquet"), Path("data/processed/fd002_train.parquet")),
}
REFERENCE_SETS = {"fd001": ("fd001",), "fd001+fd002": ("fd001", "fd002")}
RAW_MONITORED: tuple[str, ...] = SETTING_COLUMNS + KEPT_SENSORS
P_VALUE_METHODS = ("p_value", "p-value")


def is_drifted(method: str, value: float, threshold: float) -> bool:
    """Evidently's convention: p-value tests drift below the threshold, distances drift above it."""
    if value is None or value != value:  # None or NaN
        return False
    if any(token in method.lower() for token in P_VALUE_METHODS):
        return bool(value < threshold)
    return bool(value >= threshold)  # plain bool: numpy's is not JSON-serialisable


def run_report(reference: pd.DataFrame, current: pd.DataFrame, drift_share: float, column_threshold: float, html_out: Path) -> dict:
    """Run DataDriftPreset and reduce Evidently's output to share, count, and per-column results.

    column_threshold is the per-column cut for numeric columns (normed Wasserstein distance, in
    reference standard deviations). Evidently's default of 0.1 is below ordinary engine-to-engine
    variation: 20 held-out FD001 engines scored up to 0.15 on single sensors. At 0.2, in-regime
    traffic stays clear while a new regime scores in the hundreds to thousands.
    """
    report = Report([DataDriftPreset(drift_share=drift_share, num_threshold=column_threshold)])
    with warnings.catch_warnings():
        # Evidently computes correlations for its HTML; columns constant in one frame divide by zero there.
        warnings.simplefilter("ignore", RuntimeWarning)
        snapshot = report.run(current_data=current, reference_data=reference)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_out))

    share = count = None
    columns: dict[str, dict] = {}
    for metric in snapshot.dict()["metrics"]:
        name, config, value = metric["metric_name"], metric.get("config", {}), metric["value"]
        if name.startswith("DriftedColumnsCount"):
            share, count = float(value["share"]), int(value["count"])
        elif name.startswith("ValueDrift(") and "column" in config:
            drifted = is_drifted(config.get("method", ""), value, float(config.get("threshold", 0.1)))
            columns[config["column"]] = {"method": config.get("method"), "score": None if value is None else round(float(value), 4),
                                         "threshold": config.get("threshold"), "drifted": drifted}
    drifted_columns = sorted(c for c, r in columns.items() if r["drifted"])
    if share is None:
        count = len(drifted_columns)
        share = count / len(columns) if columns else 0.0
    return {"columns_monitored": len(columns), "drifted_count": count, "drifted_share": round(share, 4),
            "drifted_columns": drifted_columns, "per_column": columns, "report": str(html_out)}


def constant_columns_now_varying(reference_raw: pd.DataFrame, current_raw: pd.DataFrame) -> list[str]:
    """Sensors with one value in the reference (dropped from features) that show more than one value now."""
    return [s for s in DROPPED_SENSORS
            if s in reference_raw and s in current_raw and reference_raw[s].nunique() <= 1 and current_raw[s].nunique() > 1]


def decide(raw: dict, features: dict, now_varying: list[str], drift_share: float) -> tuple[bool, list[str], str]:
    settings_drifted = [c for c in raw["drifted_columns"] if c in SETTING_COLUMNS]
    reasons = []
    if raw["drifted_share"] >= drift_share:
        reasons.append(f"{raw['drifted_count']} of {raw['columns_monitored']} raw input columns drifted "
                       f"(share {raw['drifted_share']:.2f} >= {drift_share:.2f})")
    if settings_drifted:
        reasons.append(f"operating settings drifted: {', '.join(settings_drifted)}")
    if now_varying:
        reasons.append(f"sensors constant in the reference now vary: {', '.join(now_varying)}")
    drift = raw["drifted_share"] >= drift_share or bool(settings_drifted)
    if not drift:
        reasons.append(f"raw drifted share {raw['drifted_share']:.2f} < {drift_share:.2f}, no operating setting drifted "
                       f"(features share {features['drifted_share']:.2f})")
    return drift, settings_drifted, "; ".join(reasons)


def load_reference(reference_set: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Per regime: (raw cycles of the training engines, feature table of the training engines)."""
    reference = {}
    for regime in REFERENCE_SETS[reference_set]:
        raw_path, features_path = REFERENCE_FILES[regime]
        features = pd.read_parquet(features_path)
        raw = pd.read_parquet(raw_path)
        raw = raw[raw[UNIT].isin(features[UNIT].unique())]  # training engines only, never held-out ones
        reference[regime] = (raw, features)
    return reference


def analyse_part(regime: str, compared_to: str, reference: tuple[pd.DataFrame, pd.DataFrame],
                 current_raw: pd.DataFrame, current_features: pd.DataFrame, args) -> dict:
    ref_raw, ref_features = reference
    feature_cols = feature_columns(ref_features)
    raw_result = run_report(ref_raw[list(RAW_MONITORED)], current_raw[list(RAW_MONITORED)],
                            args.drift_share, args.column_threshold, args.out / f"drift_raw_{regime}.html")
    feature_result = run_report(ref_features[feature_cols], current_features[feature_cols],
                                args.drift_share, args.column_threshold, args.out / f"drift_features_{regime}.html")
    now_varying = constant_columns_now_varying(ref_raw, current_raw)
    drift, settings_drifted, reason = decide(raw_result, feature_result, now_varying, args.drift_share)
    return {
        "regime": regime, "compared_to": compared_to, "records": int(len(current_raw)),
        "engines": int(current_raw[UNIT].nunique()), "drift": drift, "reason": reason,
        "reference": {"raw_rows": int(len(ref_raw)), "feature_rows": int(len(ref_features)), "engines": int(ref_features[UNIT].nunique())},
        "raw": {k: v for k, v in raw_result.items() if k != "per_column"},
        "features": {k: v for k, v in feature_result.items() if k != "per_column"},
        "settings_drifted": settings_drifted, "constant_in_reference_now_varying": now_varying,
        "per_column": {"raw": raw_result["per_column"], "features": feature_result["per_column"]},
    }


def performance(outputs: pd.DataFrame, labels_from: list[Path] | None, reference_auc: float | None) -> dict | None:
    """Join predictions with derivable labels (run-to-failure engines) and score the model on them."""
    if not labels_from or outputs.empty:
        return None
    frames = []
    for path in labels_from:
        table = pd.read_parquet(path, columns=[UNIT, CYCLE, RUL])
        table["label_true"] = (table[RUL] <= LABEL_HORIZON).astype(int)
        frames.append(table[[UNIT, CYCLE, "label_true"]])
    labels = pd.concat(frames, ignore_index=True).drop_duplicates([UNIT, CYCLE])
    joined = outputs.merge(labels, on=[UNIT, CYCLE], how="inner")
    if joined.empty or joined["label_true"].nunique() < 2:
        return {"labeled_records": int(len(joined)), "note": "not enough labeled records of both classes"}

    def score(part: pd.DataFrame) -> dict:
        p, r, f, _ = precision_recall_fscore_support(part["label_true"], part["label"], average="binary", zero_division=0)
        return {"labeled_records": int(len(part)), "engines": int(part[UNIT].nunique()),
                "positive_rate": round(float(part["label_true"].mean()), 4),
                "roc_auc_current": round(float(roc_auc_score(part["label_true"], part["probability"])), 4) if part["label_true"].nunique() == 2 else None,
                "precision_current": round(float(p), 4), "recall_current": round(float(r), 4), "f1_current": round(float(f), 4)}

    result = {**score(joined), "roc_auc_reference": reference_auc, "by_regime": {}}
    for regime, part in joined.groupby(joined[UNIT].map(regime_of)):
        result["by_regime"][regime] = score(part)
    return result


def to_markdown(verdict: dict) -> str:
    cur, parts, perf = verdict["current"], verdict["parts"], verdict.get("performance")
    heading = "DRIFT" if verdict["drift"] else ("no verdict (not enough data)" if verdict.get("insufficient_data") else "no drift")
    lines = [
        f"## Drift verdict: {heading}",
        "",
        f"Window: last {cur['window_hours']} h, {cur['records']} predictions from {cur['engines']} engines"
        f"{', filtered to ' + cur['only_regime'] if cur.get('only_regime') else ''}. "
        f"Reference set: {verdict['reference_set']} (the champion's training engines, compared per regime).",
        "",
        "| Traffic | compared to | records | raw columns drifted | features drifted | settings drifted | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in parts:
        if p.get("skipped"):
            lines.append(f"| {p['regime']} | | {p['records']} | | | | skipped: {p['skipped']} |")
            continue
        lines.append(f"| {p['regime']} | {p['compared_to']} | {p['records']} | {p['raw']['drifted_count']} of {p['raw']['columns_monitored']} "
                     f"({p['raw']['drifted_share']:.2f}) | {p['features']['drifted_count']} of {p['features']['columns_monitored']} "
                     f"({p['features']['drifted_share']:.2f}) | {', '.join(p['settings_drifted']) or 'none'} | {'DRIFT' if p['drift'] else 'no drift'} |")
    lines += ["", f"Reason: {verdict['reason']}"]
    for p in parts:
        if p.get("constant_in_reference_now_varying"):
            lines += ["", f"{p['regime']}: constant in the reference, varying now: `{'`, `'.join(p['constant_in_reference_now_varying'])}`"]
    if perf and perf.get("roc_auc_current") is not None:
        lines += ["", "| Model on the current window | ROC-AUC | precision | recall | labeled records |", "|---|---|---|---|---|",
                  f"| champion, all labeled traffic | {perf['roc_auc_current']:.4f} (reference {perf['roc_auc_reference']}) | "
                  f"{perf['precision_current']:.3f} | {perf['recall_current']:.3f} | {perf['labeled_records']} |"]
        for regime, s in perf.get("by_regime", {}).items():
            if s.get("roc_auc_current") is not None:
                lines.append(f"| champion, {regime} traffic | {s['roc_auc_current']:.4f} | {s['precision_current']:.3f} | {s['recall_current']:.3f} | {s['labeled_records']} |")
    return "\n".join(lines) + "\n"


def write_verdict(out: Path, verdict: dict, markdown: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    (out / "verdict.md").write_text(markdown)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-set", choices=sorted(REFERENCE_SETS), default="fd001",
                        help="what the champion trained on (its registry `data` tag)")
    parser.add_argument("--logs-from", choices=("blob", "postgres", "local"), default="blob")
    parser.add_argument("--account", default=os.environ.get("AZURE_STORAGE_ACCOUNT"))
    parser.add_argument("--container", default=os.environ.get("PREDICTION_CONTAINER", "predictions"))
    parser.add_argument("--prefix", default=os.environ.get("PREDICTION_PREFIX", "predictions"))
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--path", type=Path, help="local JSONL file or directory")
    parser.add_argument("--since-hours", type=float, default=24.0, help="0 = all records")
    parser.add_argument("--only-regime", choices=("fd001", "fd002"), help="keep only FD001 (units <= 1000) or replayed FD002 (units > 1000) traffic")
    parser.add_argument("--labels-from", type=Path, nargs="+", help="ingested cycles with RUL (any subset), for performance where labels are derivable")
    parser.add_argument("--reference-auc", type=float, help="the champion's held-out ROC-AUC, for the performance line")
    parser.add_argument("--drift-share", type=float, default=0.3, help="share of drifted raw columns that declares drift")
    parser.add_argument("--column-threshold", type=float, default=0.2,
                        help="per-column normed Wasserstein cut, in reference standard deviations (Evidently default 0.1)")
    parser.add_argument("--min-records", type=int, default=200, help="a part below this many predictions cannot drive the verdict")
    parser.add_argument("--min-engines", type=int, default=5, help="a part below this many distinct engines cannot drive the verdict")
    parser.add_argument("--out", type=Path, default=Path("monitoring/out"))
    args = parser.parse_args(argv)

    since = logs.since_hours(args.since_hours) if args.since_hours > 0 else None
    if args.logs_from == "blob":
        if not args.account:
            raise SystemExit("--account or AZURE_STORAGE_ACCOUNT is required for --logs-from blob")
        records = logs.read_blob(args.account, args.container, args.prefix, since)
    elif args.logs_from == "postgres":
        if not args.dsn:
            raise SystemExit("--dsn or DATABASE_URL is required for --logs-from postgres")
        records = logs.read_postgres(args.dsn, since)
    else:
        if not args.path:
            raise SystemExit("--path is required for --logs-from local")
        records = logs.read_local(args.path, since)
    if args.only_regime:
        records = [r for r in records if regime_of(r["unit"]) == args.only_regime]

    now = datetime.now(timezone.utc).isoformat()
    base = {"generated_at": now, "reference_set": args.reference_set, "drift_share_threshold": args.drift_share,
            "column_threshold": args.column_threshold, "min_records": args.min_records, "min_engines": args.min_engines}
    if not records:
        verdict = {"drift": False, "reason": "no predictions in the window", **base,
                   "current": {"records": 0, "engines": 0, "window_hours": args.since_hours, "only_regime": args.only_regime}, "parts": []}
        write_verdict(args.out, verdict, f"## Drift verdict: no data\n\nNo predictions in the last {args.since_hours} h.\n")
        log.info("no predictions in the window; wrote a no-data verdict")
        return

    current_raw, current_features, outputs = logs.to_frames(records)
    reference = load_reference(args.reference_set)
    everything = (pd.concat([r for r, _ in reference.values()], ignore_index=True),
                  pd.concat([f for _, f in reference.values()], ignore_index=True))

    parts = []
    regimes = current_raw[UNIT].map(regime_of)
    for regime in sorted(regimes.unique()):
        mask = (regimes == regime).to_numpy()
        part_raw, part_features = current_raw[mask], current_features[mask]
        n_records, n_engines = int(len(part_raw)), int(part_raw[UNIT].nunique())
        if n_records < args.min_records or n_engines < args.min_engines:
            parts.append({"regime": regime, "records": n_records, "engines": n_engines, "drift": False,
                          "skipped": f"{n_records} predictions from {n_engines} engines (need {args.min_records} and {args.min_engines})"})
            log.info("%s: %s, not enough to judge", regime, parts[-1]["skipped"])
            continue
        compared_to = regime if regime in reference else args.reference_set
        parts.append(analyse_part(regime, compared_to, reference.get(regime, everything), part_raw, part_features, args))
        log.info("%s traffic vs %s reference: %s (%s)", regime, compared_to, "DRIFT" if parts[-1]["drift"] else "no drift", parts[-1]["reason"])

    judged = [p for p in parts if not p.get("skipped")]
    drift = any(p["drift"] for p in judged)
    if judged:
        reason = "; ".join(f"{p['regime']} vs {p['compared_to']}: {p['reason']}" for p in judged)
    else:
        reason = "not enough data for a verdict: " + "; ".join(f"{p['regime']} {p['skipped']}" for p in parts)
    aggregate_raw = {  # what retrain_trigger puts in the dispatch payload
        "drifted_share": max((p["raw"]["drifted_share"] for p in judged), default=0.0),
        "drifted_columns": sorted({c for p in judged for c in p["raw"]["drifted_columns"]}),
    }
    verdict = {
        "drift": drift, "reason": reason, **base,
        "insufficient_data": not judged,
        "current": {"records": int(len(outputs)), "engines": int(outputs[UNIT].nunique()),
                    "regime_records": int((outputs[UNIT] > REPLAY_UNIT_OFFSET).sum()), "window_hours": args.since_hours,
                    "only_regime": args.only_regime, "from": outputs["timestamp"].min().isoformat(),
                    "to": outputs["timestamp"].max().isoformat(),
                    "model_versions": sorted(outputs["model_version"].astype(str).unique().tolist())},
        "raw": aggregate_raw,
        "settings_drifted": sorted({c for p in judged for c in p["settings_drifted"]}),
        "constant_in_reference_now_varying": sorted({c for p in judged for c in p["constant_in_reference_now_varying"]}),
        "performance": performance(outputs, args.labels_from, args.reference_auc),
        "parts": parts,
    }
    markdown = to_markdown(verdict)
    write_verdict(args.out, verdict, markdown)
    log.info("verdict: %s", "DRIFT" if drift else ("no verdict" if not judged else "no drift"))
    print(markdown)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("azure", "urllib3", "evidently"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
