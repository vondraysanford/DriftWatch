"""Rolling-window features over raw sensor cycles, plus the binary failure label.

This module is shared by the training pipeline and the serving endpoint. ``/predict`` receives the
last WINDOW raw cycles of one engine and calls ``build_features`` on them, so whatever is computed
here is exactly what the model sees in both places.

Decisions behind the constants come from notebooks/01_fd001_exploration.ipynb (guide, Phase 1 step 3):

- WINDOW = 20 cycles: the shortest engine in the official test split is observed for 31 cycles,
  and no training engine loses labelled rows.
- LABEL_HORIZON = 30 cycles: the strongest sensors sit 2 to 4 standard deviations from healthy
  baseline inside the last 30 cycles and fade below 1 past 60.
- Sensors with two or fewer distinct values in FD001 carry nothing and are dropped (7 of 21).
  ``select_sensors`` re-applies that rule so the list can be verified against the data.
- The three operating settings are constant in FD001 and are not features. They stay in the raw
  prediction log, where they become the drift fingerprint once FD002 replays.

Usage as a DVC stage:

    python -m data.features --input data/interim/fd001_train.parquet \\
        --output data/processed/fd001_features.parquet --verify-sensor-rule
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from data.schema import CYCLE, LABEL, RUL, SENSOR_COLUMNS, UNIT

log = logging.getLogger("driftwatch.features")

WINDOW = 20
LABEL_HORIZON = 30
ROLLING_STATS: tuple[str, ...] = ("mean", "std", "min", "max")
DELTA_LAGS: tuple[int, ...] = (5, 10)
MAX_DISTINCT_VALUES_TO_DROP = 2

DROPPED_SENSORS: tuple[str, ...] = ("s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19")
KEPT_SENSORS: tuple[str, ...] = tuple(s for s in SENSOR_COLUMNS if s not in DROPPED_SENSORS)

# Columns carried through the feature table that a model must never train on.
NON_FEATURE_COLUMNS: tuple[str, ...] = (UNIT, RUL, LABEL)


def select_sensors(train_cycles: pd.DataFrame) -> tuple[str, ...]:
    """Apply the drop rule to training data: keep sensors with more than two distinct values."""
    distinct = train_cycles[list(SENSOR_COLUMNS)].nunique()
    return tuple(s for s in SENSOR_COLUMNS if distinct[s] > MAX_DISTINCT_VALUES_TO_DROP)


def verify_sensor_rule(train_cycles: pd.DataFrame) -> None:
    """Fail if the hardcoded sensor list no longer matches what the rule selects from the data."""
    selected = select_sensors(train_cycles)
    if selected != KEPT_SENSORS:
        raise ValueError(f"KEPT_SENSORS is stale: rule selects {selected}, module has {KEPT_SENSORS}")


def build_features(cycles: pd.DataFrame) -> pd.DataFrame:
    """Compute one feature row per cycle that has a full WINDOW of history for its engine.

    ``cycles`` holds raw rows (schema.RAW_COLUMNS, optionally RUL) for one or many engines.
    The result keeps unit and cycle, the current value of each kept sensor, rolling statistics
    and lagged deltas over the window, and RUL when present. An engine with fewer than WINDOW
    cycles yields no rows, which is how serving detects a request that is too short.
    """
    df = cycles.sort_values([UNIT, CYCLE]).reset_index(drop=True)
    sensors = list(KEPT_SENSORS)
    by_unit = df.groupby(UNIT, sort=False)[sensors]

    parts = [df[[UNIT, CYCLE]], df[sensors]]
    rolling = by_unit.rolling(WINDOW, min_periods=WINDOW)
    for stat in ROLLING_STATS:
        block = getattr(rolling, stat)().reset_index(level=UNIT, drop=True)
        parts.append(block.add_suffix(f"_{stat}{WINDOW}"))
    for lag in DELTA_LAGS:
        parts.append((df[sensors] - by_unit.shift(lag)).add_suffix(f"_delta{lag}"))
    if RUL in df.columns:
        parts.append(df[[RUL]])

    features = pd.concat(parts, axis=1)
    has_full_window = features.filter(like=f"_mean{WINDOW}").notna().all(axis=1)
    return features[has_full_window].reset_index(drop=True)


def add_label(features: pd.DataFrame) -> pd.DataFrame:
    """Binary target: the engine fails within LABEL_HORIZON cycles."""
    features[LABEL] = (features[RUL] <= LABEL_HORIZON).astype("int8")
    return features


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers and targets. Engine age (cycle) is a feature."""
    return [c for c in table.columns if c not in NON_FEATURE_COLUMNS]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="interim parquet from data.ingest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-sensor-rule", action="store_true",
                        help="assert KEPT_SENSORS matches the drop rule on this (training) data")
    args = parser.parse_args(argv)

    cycles = pd.read_parquet(args.input)
    if args.verify_sensor_rule:
        verify_sensor_rule(cycles)
    features = add_label(build_features(cycles))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    log.info(
        "wrote %s: %d rows, %d engines, %d feature columns, positive rate %.3f (dropped %d warm-up rows)",
        args.output, len(features), features[UNIT].nunique(), len(feature_columns(features)),
        features[LABEL].mean(), len(cycles) - len(features),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
