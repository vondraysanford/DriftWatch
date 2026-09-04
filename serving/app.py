"""FastAPI scoring service.

    POST /predict   the last WINDOW+ raw cycles of one engine -> failure probability (and a log record)
    GET  /health    model loaded and the prediction sink reachable
    GET  /model     registry metadata of the baked model

Run locally:  uvicorn serving.app:app --port 8000   (never 5000; AirPlay owns it on macOS)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from serving.config import Settings
from serving.metrics import MetricsStore
from serving.metrics import router as metrics_router
from serving.model import LoadedModel, load_model, score_window
from serving.schemas import ModelInfo, ModelRef, PredictRequest, PredictResponse
from serving.sinks import PredictionRecord, PredictionSink, make_sink, now_utc

log = logging.getLogger("driftwatch.serving")

# The built React dashboard (dashboard/dist), produced before the image build like the model.
DASHBOARD_DIR = Path(os.environ.get("DASHBOARD_DIR", "dashboard/dist"))


class JsonFormatter(logging.Formatter):
    """One JSON object per line (UTC), so Container Apps log queries can filter on fields."""

    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for noisy in ("azure", "urllib3", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    loaded = load_model(settings.model_dir)
    sink = make_sink(settings)
    sink.ping()
    app.state.model = loaded
    app.state.sink = sink
    app.state.metrics = MetricsStore(settings)
    log.info("ready", extra={"fields": {
        "model": loaded.info.name, "version": loaded.info.version, "threshold": loaded.info.threshold,
        "sink": sink.kind,
    }})
    try:
        yield
    finally:
        sink.close()


app = FastAPI(
    title="DriftWatch",
    description="Turbofan failure-within-30-cycles prediction on raw C-MAPSS cycle windows. Every prediction is logged.",
    lifespan=lifespan,
)
app.include_router(metrics_router)

# The dashboard also lives on Cloudflare Pages at its own subdomain and calls this API directly,
# so those origins are allowed for the read-only routes. Same-origin (/dashboard) needs nothing.
_cors_origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["GET"], allow_headers=["Accept"], max_age=3600)

if DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


@app.get("/", include_in_schema=False)
def root():
    if DASHBOARD_DIR.is_dir():
        return RedirectResponse("/dashboard/")
    return {"service": "DriftWatch", "predict": "/predict", "health": "/health", "model": "/model", "metrics": "/api/summary"}


@app.get("/health")
def health(request: Request) -> dict:
    loaded: LoadedModel = request.app.state.model
    sink: PredictionSink = request.app.state.sink
    try:
        sink.ping()
    except Exception as exc:  # any sink failure means predictions could not be logged
        log.error("sink unreachable", extra={"fields": {"sink": sink.kind, "error": str(exc)}})
        raise HTTPException(status_code=503, detail=f"prediction sink ({sink.kind}) unreachable") from exc
    return {"status": "ok", "model": loaded.info.name, "version": loaded.info.version, "sink": sink.kind}


@app.get("/model", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    return request.app.state.model.info


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    loaded: LoadedModel = request.app.state.model
    sink: PredictionSink = request.app.state.sink
    rows = [row.model_dump() for row in body.cycles]

    try:
        scored = score_window(loaded, rows)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record = PredictionRecord(
        prediction_id=str(uuid.uuid4()),
        timestamp=now_utc().isoformat(),
        model_name=loaded.info.name,
        model_version=loaded.info.version,
        unit=rows[-1]["unit"],
        cycle=scored.cycle,
        probability=scored.probability,
        label=scored.label,
        threshold=loaded.info.threshold,
        latency_ms=scored.latency_ms,
        raw=rows,
        features=scored.features,
    )
    try:
        sink.write(record)
    except Exception as exc:
        log.error("prediction not logged", extra={"fields": {"prediction_id": record.prediction_id, "error": str(exc)}})
        raise HTTPException(status_code=500, detail="prediction could not be logged; not returning an unlogged result") from exc

    log.info("prediction", extra={"fields": {
        "prediction_id": record.prediction_id, "unit": record.unit, "cycle": record.cycle,
        "probability": round(record.probability, 4), "label": record.label, "latency_ms": record.latency_ms,
    }})
    return PredictResponse(
        prediction_id=record.prediction_id,
        timestamp=record.timestamp,
        unit=record.unit,
        cycle=record.cycle,
        probability=record.probability,
        label=record.label,
        threshold=record.threshold,
        model=ModelRef(name=record.model_name, version=record.model_version),
        latency_ms=record.latency_ms,
    )
