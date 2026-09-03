# Build from the repo root so the image gets the shared data/ package:
#   python -m serving.fetch_model                       # pull the registered model first
#   docker build -f serving/Dockerfile -t drift-watch .
# Base image pinned by digest: python:3.11-slim as pulled and verified 2026-09-03.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVING_MODEL_DIR=/app/serving/model

WORKDIR /app

COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

# Same feature code as training, plus the app and the model fetched at build time.
COPY data/ data/
COPY serving/ serving/
RUN test -f serving/model/model_info.json || (echo "serving/model is missing: run python -m serving.fetch_model first" && exit 1)

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
