"""Train one model on the training engines, evaluate on the held-out engines and on the official
holdout, and log params, metrics, plots, and the model to MLflow in the Azure ML workspace.

    python -m training.train --model logreg                       # the baseline to beat
    python -m training.train --model xgboost                      # untuned defaults
    python -m training.train --model xgboost --params best.json   # settings from training.tune

``fit_evaluate_log`` is also what training.tune calls for its final model, so tuned and untuned
runs are logged identically and compare cleanly in the studio.
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import mlflow
from mlflow.models import infer_signature

from training.common import (
    MODEL_KINDS,
    choose_threshold,
    configure_mlflow,
    evaluate,
    lineage_tags,
    load_tables,
    log_model,
    log_plots,
    make_model,
    out_of_fold_proba,
    setup_logging,
    split_xy,
)

log = logging.getLogger("driftwatch.train")

DEFAULT_PARAMS: dict[str, dict] = {
    "logreg": {"C": 1.0},
    "xgboost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
}
EVAL_SETS = ("test", "holdout")


def fit_evaluate_log(kind: str, params: dict, tables: dict, run_name: str, tags: dict | None = None) -> tuple[str, dict]:
    """Fit on the training engines and log into the active MLflow run, or start one named run_name.

    Returns (run_id, metrics). Metrics are prefixed cv_ (out-of-fold on training engines),
    test_ (the 20 held-out engines) and holdout_ (NASA's official split).
    """
    X, y, groups = split_xy(tables["train"])
    oof = out_of_fold_proba(kind, params, X, y, groups)
    threshold = choose_threshold(y, oof)
    model = make_model(kind, params).fit(X, y)

    context = nullcontext(mlflow.active_run()) if mlflow.active_run() else mlflow.start_run(run_name=run_name)
    with context as run:
        mlflow.set_tags({"model_kind": kind, **lineage_tags(), **(tags or {})})
        mlflow.log_params({
            **params,
            "operating_threshold": round(threshold, 4),
            "n_train_engines": int(groups.nunique()),
            "n_train_rows": len(X),
        })
        metrics = {f"cv_{k}": v for k, v in evaluate(y, oof, threshold).items()}
        for name in EVAL_SETS:
            X_eval, y_eval, _ = split_xy(tables[name])
            proba = model.predict_proba(X_eval)[:, 1]
            metrics.update({f"{name}_{k}": v for k, v in evaluate(y_eval, proba, threshold).items()})
            log_plots(y_eval, proba, threshold, name)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X.head(), model.predict(X.head()))
        log_model(model, signature, X.head(3))

        log.info(
            "%s run %s: threshold %.3f | test roc_auc %.4f pr_auc %.4f recall %.3f precision %.3f | holdout roc_auc %.4f",
            kind, run.info.run_id, threshold, metrics["test_roc_auc"], metrics["test_pr_auc"],
            metrics["test_recall"], metrics["test_precision"], metrics["holdout_roc_auc"],
        )
        return run.info.run_id, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODEL_KINDS, required=True)
    parser.add_argument("--params", type=Path, help="JSON file of model settings (default: built-in defaults)")
    parser.add_argument("--run-name", help="MLflow run name (default: <model>-baseline or <model>-default)")
    args = parser.parse_args(argv)

    params = json.loads(args.params.read_text()) if args.params else DEFAULT_PARAMS[args.model]
    run_name = args.run_name or (f"{args.model}-baseline" if args.model == "logreg" else f"{args.model}-default")
    stage = "baseline" if args.model == "logreg" else ("from-params" if args.params else "default")

    experiment = configure_mlflow()
    log.info("experiment %r, model %s, params %s", experiment, args.model, params)
    fit_evaluate_log(args.model, params, load_tables(), run_name, tags={"stage": stage})


if __name__ == "__main__":
    setup_logging()
    main()
