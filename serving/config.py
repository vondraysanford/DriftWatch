"""Serving configuration, environment-only. Nothing network- or Azure-related is hardcoded.

    SERVING_MODEL_DIR        directory written by serving.fetch_model (default: serving/model)
    PREDICTION_SINK          "postgres" or "blob"
    DATABASE_URL             postgres sink: libpq connection string
    AZURE_STORAGE_ACCOUNT    blob sink: storage account name (auth via managed identity or az login)
    PREDICTION_CONTAINER     blob sink: container name (default: predictions)
    PREDICTION_PREFIX        blob sink: path prefix inside the container (default: predictions)
    LOG_LEVEL                default INFO
    CORS_ALLOW_ORIGINS       comma-separated browser origins allowed to call /api, /model, /health
                             (the Cloudflare Pages dashboard); empty means same-origin only
    DASHBOARD_DIR            built dashboard to serve at /dashboard (default: dashboard/dist)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SINK_KINDS = ("postgres", "blob")


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    sink: str
    database_url: str | None
    storage_account: str | None
    container: str
    prefix: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        sink = os.environ.get("PREDICTION_SINK", "").strip().lower()
        if sink not in SINK_KINDS:
            raise SystemExit(f"PREDICTION_SINK must be one of {SINK_KINDS}, got {sink!r}")
        settings = cls(
            model_dir=Path(os.environ.get("SERVING_MODEL_DIR", "serving/model")),
            sink=sink,
            database_url=os.environ.get("DATABASE_URL") or None,
            storage_account=os.environ.get("AZURE_STORAGE_ACCOUNT") or None,
            container=os.environ.get("PREDICTION_CONTAINER", "predictions"),
            prefix=os.environ.get("PREDICTION_PREFIX", "predictions").strip("/"),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        )
        if sink == "postgres" and not settings.database_url:
            raise SystemExit("PREDICTION_SINK=postgres needs DATABASE_URL")
        if sink == "blob" and not settings.storage_account:
            raise SystemExit("PREDICTION_SINK=blob needs AZURE_STORAGE_ACCOUNT")
        return settings
