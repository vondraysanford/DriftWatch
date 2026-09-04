"""Champion vs challenger on the mixed held-out bench (decision 14).

    python -m training.challenge --margin 0.005            # registers the challenger if it wins
    python -m training.challenge --dry-run                 # evaluate and report only

Champion: the registry version tagged stage=champion, whatever it was trained on. Challenger: the
best finished run in the experiment that trained on FD001 plus the replayed regime (tag
data=fd001+fd002), by its held-out ROC-AUC. Both are scored here on the same table: the FD001
held-out engines plus the FD002 held-out engines, split by engine unit, so neither model has seen
any of it. Per-regime numbers are reported too, because a challenger that wins on FD002 by giving
up FD001 is not an improvement.

The challenger is registered only if it beats the champion's overall ROC-AUC by --margin. It is
registered as stage=challenger; promotion is a separate, human-approved step (training.promote).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from data.schema import UNIT, regime_of
from training.common import configure_mlflow, load_tables, require_env, setup_logging, split_xy
from training.promote import champion_version
from training.register import register_run

log = logging.getLogger("driftwatch.challenge")

MIXED_TAG = "fd001+fd002"


def score(model, test: pd.DataFrame) -> dict:
    X, y, units = split_xy(test)
    proba = model.predict_proba(X)[:, 1]
    result = {"roc_auc": round(float(roc_auc_score(y, proba)), 4), "rows": int(len(test)), "engines": int(units.nunique())}
    regimes = units.map(regime_of)
    for regime in sorted(regimes.unique()):
        mask = (regimes == regime).to_numpy()
        if y[mask].nunique() == 2:
            result[f"roc_auc_{regime}"] = round(float(roc_auc_score(y[mask], proba[mask])), 4)
    return result


def best_mixed_run(experiment: str):
    runs = mlflow.search_runs(experiment_names=[experiment], output_format="pandas")
    if runs.empty or "tags.data" not in runs.columns:
        raise SystemExit(f"no runs in {experiment!r} trained on {MIXED_TAG}; run training.train/tune with --with-regime first")
    candidates = runs[(runs["status"] == "FINISHED") & (runs["tags.data"] == MIXED_TAG) & runs["metrics.test_roc_auc"].notna()]
    if candidates.empty:
        raise SystemExit(f"no finished runs in {experiment!r} tagged data={MIXED_TAG} with a test_roc_auc")
    return candidates.sort_values("metrics.test_roc_auc", ascending=False).iloc[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--margin", type=float, default=0.005, help="challenger must beat the champion's ROC-AUC by this much")
    parser.add_argument("--dry-run", action="store_true", help="evaluate and report, never register")
    parser.add_argument("--out", type=Path, default=Path("monitoring/out/challenge.json"))
    args = parser.parse_args(argv)

    experiment = configure_mlflow()
    model_name = require_env("DRIFTWATCH_MODEL_NAME")
    client = MlflowClient()
    test = load_tables(with_regime=True)["test"]

    champ_version = champion_version(client, model_name)
    champion = mlflow.sklearn.load_model(f"models:/{model_name}/{champ_version}")
    champion_result = {"version": champ_version, **score(champion, test)}
    log.info("champion v%s on the mixed bench: %s", champ_version, champion_result)

    run = best_mixed_run(experiment)
    challenger = mlflow.sklearn.load_model(f"{client.get_run(run['run_id']).info.artifact_uri}/model")
    challenger_result = {"run_id": run["run_id"], "run_name": run.get("tags.mlflow.runName", "?"),
                         "model_kind": run.get("tags.model_kind", "?"), **score(challenger, test)}
    log.info("challenger %s on the mixed bench: %s", challenger_result["run_name"], challenger_result)

    gain = round(challenger_result["roc_auc"] - champion_result["roc_auc"], 4)
    wins = gain >= args.margin
    registered = None
    if wins and not args.dry_run:
        registered = register_run(client, model_name, run["run_id"], {
            "bench": MIXED_TAG, "bench_roc_auc": f"{challenger_result['roc_auc']:.4f}",
            "champion_version": champ_version, "champion_bench_roc_auc": f"{champion_result['roc_auc']:.4f}",
        })

    summary = {"bench": MIXED_TAG, "margin": args.margin, "champion": champion_result, "challenger": challenger_result,
               "gain": gain, "challenger_wins": bool(wins), "registered_version": registered, "dry_run": args.dry_run}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["## Champion vs challenger", "", f"Bench: FD001 held-out plus FD002 held-out engines ({champion_result['engines']} engines, {champion_result['rows']} rows), margin {args.margin}.", "",
             "| | ROC-AUC (mixed) | FD001 part | FD002 part |", "|---|---|---|---|",
             f"| champion v{champ_version} | {champion_result['roc_auc']:.4f} | {champion_result.get('roc_auc_fd001', 'n/a')} | {champion_result.get('roc_auc_fd002', 'n/a')} |",
             f"| challenger {challenger_result['run_name']} | {challenger_result['roc_auc']:.4f} | {challenger_result.get('roc_auc_fd001', 'n/a')} | {challenger_result.get('roc_auc_fd002', 'n/a')} |",
             "", f"Gain {gain:+.4f}: " + (f"challenger registered as version {registered}" if registered else
                                        ("challenger wins (dry run, not registered)" if wins else "champion holds; nothing registered"))]
    print("\n".join(lines))
    log.info("gain %+.4f (margin %.4f): %s", gain, args.margin, "challenger wins" if wins else "champion holds")


if __name__ == "__main__":
    setup_logging()
    main()
