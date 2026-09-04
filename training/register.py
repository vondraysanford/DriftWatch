"""Register a run's model in the workspace model registry.

    python -m training.register --metric roc_auc            # best test_roc_auc in the experiment
    python -m training.register --metric pr_auc --split holdout

A registered version is not a deployed one. New versions are tagged ``stage=challenger``; what
serving pulls is the single version tagged ``stage=champion`` (see training.promote and
serving.fetch_model). The first registration (Phase 2) was promoted by hand; the retrain loop
registers challengers and a human approves promotion.

Each version is tagged with the source run, its metrics, its operating threshold, and the DVC
data hash, so any deployment can read what it needs from the registry alone.
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


def register_run(client: MlflowClient, model_name: str, run_id: str, extra_tags: dict[str, str] | None = None) -> str:
    """Create a model version from a run's ``model`` artifact and tag it. Returns the version."""
    run = client.get_run(run_id)
    # Address the model by the run's artifact URI rather than runs:/<id>/model. On Azure ML,
    # MLflow 3 resolves runs:/ URIs through a logged-models endpoint the workspace does not
    # implement (404); the artifact URI goes straight to storage. run_id keeps the lineage link.
    source = f"{run.info.artifact_uri}/model"
    try:
        client.create_registered_model(model_name)
        log.info("created registered model %s", model_name)
    except Exception:  # already exists, which is the normal case after the first registration
        pass
    version = client.create_model_version(model_name, source, run_id=run_id)

    tags = {
        "stage": "challenger",
        "run_id": run_id,
        "run_name": run.data.tags.get("mlflow.runName", "?"),
        "model_kind": run.data.tags.get("model_kind", "?"),
        "data": run.data.tags.get("data", "fd001"),  # what it trained on; the drift monitor's reference follows this
        "operating_threshold": run.data.params.get("operating_threshold", "?"),
        "data_version": run.data.tags.get("data_version", "?"),
        "git_commit": run.data.tags.get("git_commit", "?"),
        **(extra_tags or {}),
    }
    for key, value in tags.items():
        try:
            client.set_model_version_tag(model_name, version.version, key, str(value))
        except Exception as exc:  # registry backends differ in tag support; the version itself is what matters
            log.warning("could not tag version %s with %s: %s", version.version, key, exc)
    log.info("registered %s version %s from run %s (stage=%s)", model_name, version.version, run_id, tags["stage"])
    return str(version.version)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="roc_auc", help="metric name without split prefix")
    parser.add_argument("--split", default="test", choices=("test", "holdout"), help="which evaluation set to rank by")
    args = parser.parse_args(argv)

    experiment = configure_mlflow()
    model_name = require_env("DRIFTWATCH_MODEL_NAME")
    column = f"metrics.{args.split}_{args.metric}"

    run = best_run(experiment, column)
    log.info("best run: %s (%s) with %s = %.4f", run["run_id"], run.get("tags.mlflow.runName", "?"), column, run[column])
    register_run(MlflowClient(), model_name, run["run_id"], {f"{args.split}_{args.metric}": f"{run[column]:.4f}"})


if __name__ == "__main__":
    setup_logging()
    main()
