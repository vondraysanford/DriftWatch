"""Request and response shapes for ``/predict``.

A request is the last cycles of one engine as raw rows: exactly the 26 columns of the C-MAPSS
files (unit, cycle, three settings, 21 sensors), at least ``WINDOW`` of them, contiguous. The
row shape is generated from ``data.schema`` so serving can never drift from ingestion.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from data.features import WINDOW
from data.schema import CYCLE, RAW_COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS, UNIT

_row_fields = {UNIT: (int, Field(ge=1)), CYCLE: (int, Field(ge=1))}
_row_fields.update({column: (float, ...) for column in SETTING_COLUMNS + SENSOR_COLUMNS})
CycleRow = create_model("CycleRow", __config__=ConfigDict(extra="forbid"), **_row_fields)
CycleRow.__doc__ = f"One raw operational cycle: {', '.join(RAW_COLUMNS)}."


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycles: list[CycleRow] = Field(min_length=WINDOW, description=f"the last {WINDOW}+ raw cycles of one engine, oldest first")

    @model_validator(mode="after")
    def one_engine_contiguous(self) -> "PredictRequest":
        units = {getattr(row, UNIT) for row in self.cycles}
        if len(units) != 1:
            raise ValueError(f"cycles must belong to exactly one engine, got units {sorted(units)}")
        numbers = [getattr(row, CYCLE) for row in self.cycles]
        if any(b - a != 1 for a, b in zip(numbers, numbers[1:])):
            raise ValueError("cycles must be contiguous and ascending")
        return self


class ModelRef(BaseModel):
    name: str
    version: str


class PredictResponse(BaseModel):
    prediction_id: str
    timestamp: datetime
    unit: int
    cycle: int = Field(description="the last cycle in the window; the prediction is as of this cycle")
    probability: float = Field(description=f"probability of failure within the label horizon")
    label: int = Field(description="1 when probability >= threshold")
    threshold: float
    model: ModelRef
    latency_ms: float = Field(description="feature computation plus model inference")


class ModelInfo(BaseModel):
    """Written by serving.fetch_model at build time from registry metadata; served at /model."""

    name: str
    version: str
    run_id: str
    run_name: str | None = None
    model_kind: str | None = None
    threshold: float
    metrics: dict[str, float] = {}
    data_version: str | None = None
    git_commit: str | None = None
    fetched_at: datetime
