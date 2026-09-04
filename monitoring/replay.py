"""Replay the quarantined regime through the live endpoint as production traffic.

    python -m monitoring.replay --endpoint https://<container-app>/ --engines 24 --every 5

FD002 engines have six operating conditions where FD001 has one, so their sensor readings are a
genuine distribution shift, not synthetic jitter (decision 6). Engines are drawn from the FD002
held-out split (the champion-vs-challenger bench, decision 14) so nothing replayed here is ever
trained on. Every engine runs to failure, so the true label at each replayed cycle is known and
the model's performance on the new regime can be measured right here.

The endpoint sees ordinary requests: the last WINDOW raw cycles of one engine. Unit numbers carry
the 1000 offset from ingestion, so the prediction log identifies this traffic without any special
field in the API.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from data.features import LABEL_HORIZON, WINDOW
from data.schema import CYCLE, RAW_COLUMNS, RUL, UNIT

log = logging.getLogger("driftwatch.monitoring.replay")

DEFAULT_SOURCE = Path("data/interim/fd002_train.parquet")
DEFAULT_UNITS_FROM = Path("data/processed/fd002_test.parquet")


def choose_engines(units_from: Path, n: int, seed: int) -> list[int]:
    units = np.sort(pd.read_parquet(units_from)[UNIT].unique())
    chosen = np.random.default_rng(seed).choice(units, size=min(n, len(units)), replace=False)
    return sorted(int(u) for u in chosen)


def windows(engine: pd.DataFrame, every: int):
    """Yield (last_cycle, rows) for a WINDOW-cycle window ending every `every` cycles."""
    engine = engine.sort_values(CYCLE).reset_index(drop=True)
    for end in range(WINDOW, len(engine) + 1, every):
        chunk = engine.iloc[end - WINDOW:end]
        yield int(chunk[CYCLE].iloc[-1]), chunk[list(RAW_COLUMNS)].to_dict(orient="records")


def post_predict(endpoint: str, rows: list[dict], timeout: float) -> tuple[dict, float]:
    body = json.dumps({"cycles": rows}).encode("utf-8")
    request = urllib.request.Request(f"{endpoint.rstrip('/')}/predict", data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload, (time.perf_counter() - started) * 1000


def replay(endpoint: str, source: pd.DataFrame, engines: list[int], every: int, timeout: float, max_requests: int) -> pd.DataFrame:
    results = []
    sent = failed = 0
    for unit in engines:
        engine = source[source[UNIT] == unit]
        life = int(engine[CYCLE].max())
        for last_cycle, rows in windows(engine, every):
            if sent >= max_requests:
                break
            sent += 1
            rul = life - last_cycle
            try:
                payload, round_trip = post_predict(endpoint, rows, timeout)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                failed += 1
                log.warning("unit %d cycle %d failed: %s", unit, last_cycle, exc)
                continue
            results.append({
                UNIT: unit, CYCLE: last_cycle, RUL: rul, "label_true": int(rul <= LABEL_HORIZON),
                "probability": payload["probability"], "label_pred": payload["label"],
                "prediction_id": payload["prediction_id"], "server_latency_ms": payload["latency_ms"],
                "round_trip_ms": round(round_trip, 1), "model_version": payload["model"]["version"],
            })
            if sent == 1:
                log.info("first response (includes any cold start): %.0f ms", round_trip)
        log.info("unit %d: life %d cycles, replayed through cycle %d", unit, life, last_cycle)
    log.info("sent %d requests, %d failed", sent, failed)
    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> dict:
    y, p, yhat = results["label_true"], results["probability"], results["label_pred"]
    precision, recall, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    rt = sorted(results["round_trip_ms"])
    return {
        "requests": int(len(results)),
        "engines": int(results[UNIT].nunique()),
        "positive_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if y.nunique() == 2 else None,
        "precision": round(float(precision), 4), "recall": round(float(recall), 4), "f1": round(float(f1), 4),
        "round_trip_ms_p50": round(statistics.median(rt), 1),
        "round_trip_ms_p95": round(rt[int(len(rt) * 0.95) - 1], 1),
        "server_latency_ms_p50": round(float(results["server_latency_ms"].median()), 1),
        "model_version": str(results["model_version"].iloc[0]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True, help="base URL of the live serving endpoint")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="ingested regime cycles (units already offset)")
    parser.add_argument("--units-from", type=Path, default=DEFAULT_UNITS_FROM, help="table whose engines are eligible (held-out split)")
    parser.add_argument("--engines", type=int, default=24)
    parser.add_argument("--every", type=int, default=5, help="send a window every N cycles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds; the first request may wake a replica from zero")
    parser.add_argument("--max-requests", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("monitoring/out/replay.parquet"))
    args = parser.parse_args(argv)

    source = pd.read_parquet(args.source)
    engines = choose_engines(args.units_from, args.engines, args.seed)
    log.info("replaying %d engines (%s...) every %d cycles to %s", len(engines), engines[:5], args.every, args.endpoint)
    results = replay(args.endpoint, source, engines, args.every, args.timeout, args.max_requests)
    if results.empty:
        raise SystemExit("no successful requests")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.out, index=False)
    summary = summarize(results)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wrote %s and %s", args.out, args.out.with_suffix(".json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("azure", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
