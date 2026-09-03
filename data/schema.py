"""Column layout of the raw C-MAPSS cycle files, shared by ingestion, features, and serving.

Each row is one operational cycle of one engine: unit number, cycle number, three operating
settings, and 21 sensor measurements. Column order matches the NASA readme.
"""

UNIT = "unit"
CYCLE = "cycle"
RUL = "rul"  # remaining useful life in cycles, derived at ingestion
LABEL = "label"  # 1 when the engine fails within features.LABEL_HORIZON cycles

ID_COLUMNS: tuple[str, ...] = (UNIT, CYCLE)
SETTING_COLUMNS: tuple[str, ...] = ("setting_1", "setting_2", "setting_3")
SENSOR_COLUMNS: tuple[str, ...] = tuple(f"s_{i}" for i in range(1, 22))
RAW_COLUMNS: tuple[str, ...] = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
