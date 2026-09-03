"""Load the baked model and score one engine's window with the training feature code."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from data.features import build_features, feature_columns
from data.schema import CYCLE, RAW_COLUMNS
from serving.schemas import ModelInfo


@dataclass(frozen=True)
class LoadedModel:
    estimator: object
    info: ModelInfo


@dataclass(frozen=True)
class Scored:
    cycle: int
    probability: float
    label: int
    features: dict[str, float]
    latency_ms: float


def load_model(model_dir: Path) -> LoadedModel:
    """Read model_info.json and the MLflow sklearn flavor from the directory fetch_model wrote."""
    info = ModelInfo.model_validate_json((model_dir / "model_info.json").read_text())
    estimator = mlflow.sklearn.load_model(str(model_dir / "model"))
    return LoadedModel(estimator=estimator, info=info)


def score_window(loaded: LoadedModel, rows: list[dict]) -> Scored:
    """Features for the last cycle of the window, then one probability. Same code path as training."""
    started = time.perf_counter()
    cycles = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    features = build_features(cycles)
    if features.empty:
        raise ValueError("window too short to compute features")
    last = features.tail(1)
    X = last[feature_columns(last)]
    probability = float(loaded.estimator.predict_proba(X)[0, 1])
    latency_ms = (time.perf_counter() - started) * 1000
    return Scored(
        cycle=int(last[CYCLE].iloc[0]),
        probability=probability,
        label=int(probability >= loaded.info.threshold),
        features={k: float(v) for k, v in X.iloc[0].items()},
        latency_ms=round(latency_ms, 3),
    )


def read_model_info(model_dir: Path) -> dict:
    return json.loads((model_dir / "model_info.json").read_text())
