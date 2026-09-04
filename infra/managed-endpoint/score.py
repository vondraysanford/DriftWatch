"""Scoring script for the Azure ML managed online endpoint demonstration.

Same contract as the Container Apps endpoint: the last WINDOW+ raw cycles of one engine in, a
failure probability out. Features come from the same data/features.py used in training (the
package is copied beside this file when the code snapshot is assembled), and the model is the
registered version Azure mounts at AZUREML_MODEL_DIR.

Why this exists: Azure's no-code MLflow scoring script failed to start because the environment
Azure built for it lacked azureml-ai-monitoring, one of Azure's own packages. Owning the script
and the environment removes that dependency on Azure's internals entirely.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from data.features import WINDOW, build_features, feature_columns
from data.schema import CYCLE, RAW_COLUMNS, UNIT

log = logging.getLogger("driftwatch.managed-endpoint")

model = None
threshold = 0.5
model_version = "unknown"


def _find_mlmodel_dir(root: Path) -> Path:
    """The registered artifact is the MLflow flavor directory; locate it wherever Azure mounted it."""
    for candidate in [root, *root.rglob("*")]:
        if candidate.is_dir() and (candidate / "MLmodel").exists():
            return candidate
    raise FileNotFoundError(f"no MLmodel found under {root}")


def init() -> None:
    """Called once when the container starts. A failure here fails the liveness probe, loudly."""
    global model, threshold, model_version
    root = Path(os.environ["AZUREML_MODEL_DIR"])
    model_dir = _find_mlmodel_dir(root)
    model = mlflow.sklearn.load_model(str(model_dir))
    threshold = float(os.environ["OPERATING_THRESHOLD"])  # deliberately required, no silent default
    model_version = root.name  # Azure mounts .../azureml-models/<name>/<version>
    log.info("loaded %s from %s, threshold %.4f", type(model).__name__, model_dir, threshold)


def run(raw_data: str) -> dict:
    """Body: {"cycles": [<raw cycle rows>]}, the same shape serving/examples/*.json use."""
    body = json.loads(raw_data)
    rows = body.get("cycles")
    if not isinstance(rows, list) or len(rows) < WINDOW:
        raise ValueError(f"'cycles' must hold at least {WINDOW} raw cycles of one engine")
    cycles = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    if cycles.isna().any().any():
        raise ValueError(f"every cycle needs all columns: {', '.join(RAW_COLUMNS)}")
    features = build_features(cycles)
    if features.empty:
        raise ValueError("window too short to compute features")
    last = features.tail(1)
    X = last[feature_columns(last)]
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "unit": int(last[UNIT].iloc[0]),
        "cycle": int(last[CYCLE].iloc[0]),
        "probability": probability,
        "label": int(probability >= threshold),
        "threshold": threshold,
        "model_version": model_version,
    }
