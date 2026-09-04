"""Column layout of the raw C-MAPSS cycle files, shared by ingestion, features, and serving.

Each row is one operational cycle of one engine: unit number, cycle number, three operating
settings, and 21 sensor measurements. Column order matches the NASA readme.
"""

UNIT = "unit"
CYCLE = "cycle"
RUL = "rul"  # remaining useful life in cycles, derived at ingestion
LABEL = "label"  # 1 when the engine fails within features.LABEL_HORIZON cycles

# Engines from the quarantined replay regime (FD002) carry unit numbers offset by this amount,
# so 1..100 is always FD001 and 1001..1260 is always FD002: in every table, in the prediction log,
# and in the label join the drift monitor does after the fact.
REPLAY_UNIT_OFFSET = 1000


def regime_of(unit: int) -> str:
    return "fd002" if unit > REPLAY_UNIT_OFFSET else "fd001"


ID_COLUMNS: tuple[str, ...] = (UNIT, CYCLE)
SETTING_COLUMNS: tuple[str, ...] = ("setting_1", "setting_2", "setting_3")
SENSOR_COLUMNS: tuple[str, ...] = tuple(f"s_{i}" for i in range(1, 22))
RAW_COLUMNS: tuple[str, ...] = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
