"""Prediction log sinks. Every prediction is written, synchronously, before the response goes out.

Same record either way: raw inputs (the cycles as received, operating settings included), the
computed features, the output, the threshold, the model version, and a timestamp. Drift
monitoring reads these; a prediction that is not logged does not exist.

- PostgresSink: local development (Postgres in Docker). One row per prediction, JSONB payloads.
- BlobSink: Azure Container Apps. JSONL appended to an append blob per day, hour, and replica,
  authenticated with DefaultAzureCredential (managed identity on Azure, az login locally).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Protocol

from serving.config import Settings

log = logging.getLogger("driftwatch.serving.sink")


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    timestamp: str  # ISO 8601, UTC
    model_name: str
    model_version: str
    unit: int
    cycle: int
    probability: float
    label: int
    threshold: float
    latency_ms: float
    raw: list[dict]
    features: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class PredictionSink(Protocol):
    kind: str

    def write(self, record: PredictionRecord) -> None: ...

    def ping(self) -> None: ...

    def close(self) -> None: ...


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL,
    model_name    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    unit          INTEGER NOT NULL,
    cycle         INTEGER NOT NULL,
    probability   DOUBLE PRECISION NOT NULL,
    label         SMALLINT NOT NULL,
    threshold     DOUBLE PRECISION NOT NULL,
    latency_ms    DOUBLE PRECISION NOT NULL,
    raw           JSONB NOT NULL,
    features      JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_ts_idx ON predictions (ts);
"""

INSERT = """
INSERT INTO predictions
    (prediction_id, ts, model_name, model_version, unit, cycle, probability, label, threshold, latency_ms, raw, features)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class PostgresSink:
    kind = "postgres"

    def __init__(self, dsn: str) -> None:
        import psycopg  # imported here so the blob-only image does not need the driver

        self._psycopg = psycopg
        self._dsn = dsn
        self._conn = psycopg.connect(dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(CREATE_TABLE)

    def _execute(self, query: str, params: tuple) -> None:
        try:
            with self._conn.cursor() as cur:
                cur.execute(query, params)
        except self._psycopg.OperationalError:
            log.warning("postgres connection lost; reconnecting once")
            self._conn = self._psycopg.connect(self._dsn, autocommit=True)
            with self._conn.cursor() as cur:
                cur.execute(query, params)

    def write(self, record: PredictionRecord) -> None:
        Json = self._psycopg.types.json.Jsonb
        self._execute(INSERT, (
            record.prediction_id, record.timestamp, record.model_name, record.model_version,
            record.unit, record.cycle, record.probability, record.label, record.threshold,
            record.latency_ms, Json(record.raw), Json(record.features),
        ))

    def ping(self) -> None:
        self._execute("SELECT 1", ())

    def close(self) -> None:
        self._conn.close()


class BlobSink:
    kind = "blob"

    def __init__(self, account: str, container: str, prefix: str) -> None:
        from azure.core.exceptions import ResourceExistsError
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        self._exists_error = ResourceExistsError
        self._service = BlobServiceClient(f"https://{account}.blob.core.windows.net", credential=DefaultAzureCredential())
        self._container = self._service.get_container_client(container)
        self._prefix = prefix
        # One append blob per replica per hour: a Container Apps replica name plus a start-up id.
        self._instance = f"{os.environ.get('CONTAINER_APP_REPLICA_NAME') or socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._created: set[str] = set()

    def _blob_name(self, ts: datetime) -> str:
        return f"{self._prefix}/dt={ts:%Y-%m-%d}/hour={ts:%H}/{self._instance}.jsonl"

    def write(self, record: PredictionRecord) -> None:
        name = self._blob_name(datetime.fromisoformat(record.timestamp))
        blob = self._container.get_blob_client(name)
        if name not in self._created:
            try:
                blob.create_append_blob()
            except self._exists_error:
                pass
            self._created.add(name)
        blob.append_block((record.to_json() + "\n").encode("utf-8"))

    def ping(self) -> None:
        self._container.get_container_properties()

    def close(self) -> None:
        self._service.close()


def make_sink(settings: Settings) -> PredictionSink:
    if settings.sink == "postgres":
        return PostgresSink(settings.database_url)  # type: ignore[arg-type]
    return BlobSink(settings.storage_account, settings.container, settings.prefix)  # type: ignore[arg-type]
