"""Optuna search over XGBoost settings, scored by grouped cross-validation inside the training
engines. The held-out engines never influence a choice.

Every trial is a nested MLflow run. The parent run ends by fitting the best settings through the
same function training.train uses, so it carries the final model and its held-out metrics and
compares directly with the baseline in the studio.

    python -m training.tune --trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from training.common import N_FOLDS, SEED, configure_mlflow, load_tables, make_model, setup_logging, split_xy
from training.train import fit_evaluate_log

log = logging.getLogger("driftwatch.tune")


def suggest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 6.0),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("training/best_params.json"),
                        help="where to write the best settings (also logged as a run artifact)")
    parser.add_argument("--with-regime", action="store_true",
                        help="tune and train on FD001 plus the replayed FD002 regime (the retrain loop)")
    args = parser.parse_args(argv)

    experiment = configure_mlflow()
    tables = load_tables(with_regime=args.with_regime)
    data_tag = "fd001+fd002" if args.with_regime else "fd001"
    parent_name = "xgboost-tuned-mixed" if args.with_regime else "xgboost-tuned"
    if args.with_regime and args.out == Path("training/best_params.json"):
        args.out = Path("training/best_params_mixed.json")
    X, y, groups = split_xy(tables["train"])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            aucs = []
            for fit_idx, val_idx in folds:
                model = make_model("xgboost", params).fit(X.iloc[fit_idx], y.iloc[fit_idx])
                aucs.append(roc_auc_score(y.iloc[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1]))
            mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
            mlflow.log_params(params)
            mlflow.log_metrics({"cv_roc_auc": mean_auc, "cv_roc_auc_std": std_auc})
        log.info("trial %3d: cv roc_auc %.4f (+/- %.4f)", trial.number, mean_auc, std_auc)
        return mean_auc

    log.info("experiment %r: %d trials, %d-fold grouped CV over %d engines", experiment, args.trials, N_FOLDS, groups.nunique())
    with mlflow.start_run(run_name=parent_name):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=args.trials)
        best = study.best_params
        log.info("best trial %d: cv roc_auc %.4f, params %s", study.best_trial.number, study.best_value, best)

        mlflow.log_params({"n_trials": args.trials, "best_trial": study.best_trial.number})
        mlflow.log_metric("best_cv_roc_auc", study.best_value)
        mlflow.log_dict(best, "best_params.json")
        args.out.write_text(json.dumps(best, indent=2) + "\n")

        fit_evaluate_log("xgboost", best, tables, run_name=parent_name, tags={"stage": "tuned", "data": data_tag})


if __name__ == "__main__":
    setup_logging()
    main()
