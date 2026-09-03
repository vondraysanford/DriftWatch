"""Promote the best run's model into the workspace model registry.

    python -m training.register --metric roc_auc          # best test_roc_auc in the experiment
    python -m training.register --metric pr_auc --split holdout

The registered version is tagged with the source run, its metrics, and the DVC data hash, and is
what Phase 4's CI build pulls into the serving image. Nothing loads a model from anywhere else.
"""

from __future__ import annotations

import argparse
import logging

import mlflow
from mlflow.tracking import MlflowClient

from training.common import configure_mlflow, require_env, setup_logging

log = logging.getLogger("driftwatch.register")


def best_run(experiment: str, metric_column: str):
    """Newest-first search of finished runs that logged the metric, best value on top."""
    runs = mlflow.search_runs(experiment_names=[experiment], output_format="pandas")
    if runs.empty or metric_column not in runs.columns:
        raise SystemExit(f"no finished runs in {experiment!r} logged {metric_column}")
    candidates = runs[(runs["status"] == "FINISHED") & runs[metric_column].notna()]
    if candidates.empty:
        raise SystemExit(f"no finished runs in {experiment!r} logged {metric_column}")
    return candidates.sort_values(metric_column, ascending=False).iloc[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="roc_auc", help="metric name without split prefix")
    parser.add_argument("--split", default="test", choices=("test", "holdout"), help="which evaluation set to rank by")
    args = parser.parse_args(argv)

    experiment = configure_mlflow()
    model_name = require_env("DRIFTWATCH_MODEL_NAME")
    column = f"metrics.{args.split}_{args.metric}"

    run = best_run(experiment, column)
    run_name = run.get("tags.mlflow.runName", "?")
    log.info("best run: %s (%s) with %s = %.4f", run["run_id"], run_name, column, run[column])

    # Address the model by the run's artifact URI rather than runs:/<id>/model. On Azure ML,
    # MLflow 3 resolves runs:/ URIs through a logged-models endpoint the workspace does not
    # implement (404); the artifact URI goes straight to storage. run_id keeps the lineage link.
    client = MlflowClient()
    source = f"{client.get_run(run['run_id']).info.artifact_uri}/model"
    try:
        client.create_registered_model(model_name)
        log.info("created registered model %s", model_name)
    except Exception:  # already exists, which is the normal case after the first registration
        pass
    version = client.create_model_version(model_name, source, run_id=run["run_id"])
    tags = {
        "run_id": run["run_id"],
        "run_name": run_name,
        "model_kind": run.get("tags.model_kind", "?"),
        f"{args.split}_{args.metric}": f"{run[column]:.4f}",
        "data_version": run.get("tags.data_version", "?"),
        "git_commit": run.get("tags.git_commit", "?"),
    }
    for key, value in tags.items():
        try:
            client.set_model_version_tag(model_name, version.version, key, str(value))
        except Exception as exc:  # registry backends differ in tag support; the version itself is what matters
            log.warning("could not tag version %s with %s: %s", version.version, key, exc)
    log.info("registered %s version %s from run %s", model_name, version.version, run["run_id"])


if __name__ == "__main__":
    setup_logging()
    main()
