"""Fast pre-deploy checks that need no data pull, no Azure, and no model.

Guards the two invariants that would break serving silently:
  1. build_features produces the exact feature contract the model was trained against.
  2. The request schema rejects windows the feature code cannot score.

Run:  python -m scripts.sanity_check
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from pydantic import ValidationError

from data.features import DROPPED_SENSORS, KEPT_SENSORS, LABEL_HORIZON, WINDOW, build_features, feature_columns
from data.schema import RAW_COLUMNS, SENSOR_COLUMNS
from serving.schemas import PredictRequest

EXPECTED_FEATURE_COUNT = 99
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f': {detail}' if detail else ''}")
    if not condition:
        failures.append(name)


def synthetic_window(n_cycles: int = WINDOW, unit: int = 1) -> pd.DataFrame:
    """A deterministic engine: sensors drift linearly, which is enough to exercise the feature code."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_cycles):
        row = {"unit": unit, "cycle": i + 1}
        row.update({s: 0.0 for s in ("setting_1", "setting_2", "setting_3")})
        row.update({s: float(100 + i * 0.5 + rng.normal(0, 0.01)) for s in SENSOR_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


print("== feature contract")
window = synthetic_window()
features = build_features(window)
check("a full window yields exactly one scorable row", len(features) == 1, f"{len(features)} rows")
cols = feature_columns(features)
check(f"feature count is {EXPECTED_FEATURE_COUNT}", len(cols) == EXPECTED_FEATURE_COUNT, f"got {len(cols)}")
check("no dropped sensor leaks into features", not any(c.startswith(f"{s}_") or c == s for s in DROPPED_SENSORS for c in cols))
check("every kept sensor contributes features", all(any(c == s or c.startswith(f"{s}_") for c in cols) for s in KEPT_SENSORS))
check("operating settings are not features", not any(c.startswith("setting_") for c in cols))
check("engine identity is not a feature", "unit" not in cols)
check("no nulls in a full window", int(features[cols].isna().sum().sum()) == 0)

print("== window length")
check("a short window yields no rows", build_features(synthetic_window(WINDOW - 1)).empty)
check("a longer window still yields the last cycle", int(build_features(synthetic_window(WINDOW + 5)).cycle.max()) == WINDOW + 5)

print("== request validation")
valid = {"cycles": synthetic_window().to_dict(orient="records")}
try:
    PredictRequest.model_validate(valid)
    check("a valid window is accepted", True)
except ValidationError as exc:
    check("a valid window is accepted", False, str(exc)[:120])


def rejects(name: str, payload: dict) -> None:
    try:
        PredictRequest.model_validate(payload)
        check(name, False, "accepted but should not be")
    except ValidationError:
        check(name, True)


rejects("too few cycles is rejected", {"cycles": valid["cycles"][: WINDOW - 1]})
rejects("two engines in one request is rejected", {"cycles": valid["cycles"][:-1] + [{**valid["cycles"][-1], "unit": 99}]})
rejects("a gap in cycle numbers is rejected", {"cycles": valid["cycles"][:5] + valid["cycles"][6:]})
rejects("an unknown column is rejected", {"cycles": [{**valid["cycles"][0], "surprise": 1.0}] + valid["cycles"][1:]})
rejects("a missing sensor is rejected", {"cycles": [{k: v for k, v in valid["cycles"][0].items() if k != "s_2"}] + valid["cycles"][1:]})

print("== constants match the recorded decisions")
check("label horizon is 30 cycles", LABEL_HORIZON == 30, str(LABEL_HORIZON))
check("rolling window is 20 cycles", WINDOW == 20, str(WINDOW))
check("7 of 21 sensors dropped", len(DROPPED_SENSORS) == 7 and len(KEPT_SENSORS) == 14)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("All sanity checks passed.")
