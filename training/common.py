"""Shared pieces for the training scripts: configuration, data, models, metrics, plots, lineage.

Configuration is environment-only (see .env.example); nothing Azure-related is hardcoded:

    MLFLOW_TRACKING_URI      the workspace's MLflow URI (az ml workspace show --query mlflow_tracking_uri)
    MLFLOW_EXPERIMENT_NAME   experiment that receives every run
    DRIFTWATCH_MODEL_NAME    registry name used by training.register

Evaluation discipline: models fit on the training engines only. The operating threshold is chosen
on out-of-fold predictions inside those engines, so the held-out engines never influence a choice.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from data.features import LABEL_HORIZON, WINDOW, feature_columns  # noqa: E402
from data.schema import LABEL, UNIT  # noqa: E402

log = logging.getLogger("driftwatch.training")

SEED = 42
N_FOLDS = 5
PROCESSED_DIR = Path("data/processed")
TABLES = {"train": "train.parquet", "test": "test.parquet", "holdout": "holdout_official.parquet"}
# The replayed regime (FD002, units offset by 1000), split by engine unit like FD001. Retraining
# concatenates these onto the FD001 tables; the mixed test set is the champion-vs-challenger bench.
REGIME_TABLES = {"train": "fd002_train.parquet", "test": "fd002_test.parquet"}
MODEL_KINDS = ("logreg", "xgboost")


def setup_logging() -> None:
    """INFO for our loggers; the Azure SDK's per-request INFO lines are noise here."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("azure", "azureml", "urllib3", "msal"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


def configure_tracking() -> None:
    """Point MLflow at the workspace. Enough for registry reads (serving.fetch_model)."""
    mlflow.set_tracking_uri(require_env("MLFLOW_TRACKING_URI"))


def configure_mlflow() -> str:
    """Point MLflow at the workspace and select the experiment. Returns the experiment name."""
    configure_tracking()
    experiment = require_env("MLFLOW_EXPERIMENT_NAME")
    mlflow.set_experiment(experiment)
    return experiment


def load_tables(with_regime: bool = False) -> dict[str, pd.DataFrame]:
    """FD001 tables, optionally with the replayed regime concatenated onto train and test."""
    tables = {name: pd.read_parquet(PROCESSED_DIR / file) for name, file in TABLES.items()}
    if with_regime:
        for name, file in REGIME_TABLES.items():
            tables[name] = pd.concat([tables[name], pd.read_parquet(PROCESSED_DIR / file)], ignore_index=True)
    return tables


def split_xy(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features, label, and the engine unit used as the grouping key for any cross-validation."""
    return table[feature_columns(table)], table[LABEL], table[UNIT]


def make_model(kind: str, params: dict | None = None):
    params = dict(params or {})
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=SEED, **params)
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if kind == "xgboost":
        return XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss", **params)
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


def out_of_fold_proba(kind: str, params: dict | None, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Probabilities for every training row from a model that never saw that row's engine."""
    cv = GroupKFold(n_splits=N_FOLDS)
    return cross_val_predict(make_model(kind, params), X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]


def choose_threshold(y: pd.Series, proba: np.ndarray) -> float:
    """Probability cut-off that maximizes F1 on out-of-fold training predictions."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Threshold-free scores plus precision/recall/F1 at the operating threshold and at 0.5."""
    metrics = {"roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba)}
    for suffix, cut in (("", threshold), ("_at_0_5", 0.5)):
        p, r, f, _ = precision_recall_fscore_support(y, proba >= cut, average="binary", zero_division=0)
        metrics[f"precision{suffix}"], metrics[f"recall{suffix}"], metrics[f"f1{suffix}"] = p, r, f
    return {k: float(v) for k, v in metrics.items()}


def log_plots(y: pd.Series, proba: np.ndarray, threshold: float, prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y, proba, ax=axes[0])
    axes[0].set_title(f"{prefix}: ROC")
    PrecisionRecallDisplay.from_predictions(y, proba, ax=axes[1])
    axes[1].set_title(f"{prefix}: precision-recall")
    fig.tight_layout()
    mlflow.log_figure(fig, f"plots/{prefix}_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, proba >= threshold, ax=ax, colorbar=False)
    ax.set_title(f"{prefix}: confusion at {threshold:.3f}")
    fig.tight_layout()
    mlflow.log_figure(fig, f"plots/{prefix}_confusion.png")
    plt.close(fig)


def log_model(model, signature, input_example: pd.DataFrame) -> str:
    """Save the model in MLflow's sklearn flavor and upload it as run artifacts under ``model/``.

    MLflow 3's ``log_model`` first creates a "logged model" entity through an endpoint the Azure ML
    tracking server does not implement (404 on /api/2.0/mlflow/logged-models). Uploading the saved
    flavor directory gives the identical layout, and ``runs:/<run_id>/model`` resolves as usual for
    the registry and for loading. Returns that model URI.
    """
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_model needs an active MLflow run")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model"
        mlflow.sklearn.save_model(model, path, signature=signature, input_example=input_example)
        mlflow.log_artifacts(str(path), artifact_path="model")
    return f"runs:/{run.info.run_id}/model"


def data_version() -> str:
    """DVC's hash of train.parquet, so every run records exactly which data it saw."""
    try:
        lock = yaml.safe_load(Path("dvc.lock").read_text())
        for out in lock["stages"]["split"]["outs"]:
            if out["path"] == "data/processed/train.parquet":
                return out["md5"]
    except (OSError, KeyError, TypeError):
        pass
    return "unknown"


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lineage_tags() -> dict[str, str]:
    return {
        "data_version": data_version(),
        "git_commit": git_commit(),
        "label_horizon": str(LABEL_HORIZON),
        "window": str(WINDOW),
    }
