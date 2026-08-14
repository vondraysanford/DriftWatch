# DriftWatch — End-to-End ML Pipeline with Drift Monitoring

A production-grade machine learning pipeline that predicts equipment failure from sensor time-series data, deployed to Azure with CI/CD, and monitored for model drift in real time. Covers the full MLOps lifecycle: ingestion, feature engineering, training with hyperparameter tuning, deployment, and a live dashboard tracking prediction quality and data drift.

This project bridges enterprise software engineering with machine learning operations — proving you can not only train a model but ship and maintain it as a reliable production service.

## Why This Project

Plenty of candidates can train a model in a notebook. Far fewer can operationalize one: version the data and model, automate retraining, deploy behind an API, and detect when the model silently degrades in production. That operational maturity is exactly what separates an ML engineer from a notebook hobbyist, and it directly reuses your enterprise deployment background (CI/CD, containers, cloud, incident response).

## What It Does

- Ingests streaming/batch sensor data and engineers time-series features (rolling stats, lag features, frequency-domain signals).
- Trains a failure-prediction model (XGBoost baseline, optional LSTM) with automated hyperparameter search.
- Tracks every experiment — params, metrics, artifacts — and registers the best model.
- Deploys the registered model to Azure as a containerized REST endpoint via a CI/CD pipeline.
- Monitors live predictions for data drift and performance decay, surfacing alerts on a dashboard.
- Triggers retraining when drift crosses a threshold.

## Tech Stack

**Data & Modeling**
- Python 3.11
- pandas / NumPy — data wrangling and feature engineering
- scikit-learn — preprocessing, baselines, metrics
- XGBoost — primary failure-prediction model
- PyTorch — optional LSTM for sequence modeling
- Optuna — hyperparameter optimization

**MLOps & Tracking**
- MLflow — experiment tracking + model registry
- Evidently AI — data drift and model performance monitoring
- DVC (Data Version Control) — dataset and pipeline versioning

**Deployment & Infrastructure**
- Azure Machine Learning — managed training + model endpoints
- Azure Container Registry — image hosting
- FastAPI — model-serving API
- Docker — containerization
- GitHub Actions — CI/CD (test → build → deploy)

**Monitoring Dashboard**
- React + Vite — frontend
- Recharts — metric and drift visualizations
- FastAPI — dashboard backend / metrics API

**Dataset**
- NASA C-MAPSS Turbofan Engine Degradation (public) — run-to-failure sensor trajectories
- Alternative: Microsoft Azure Predictive Maintenance dataset

## Architecture

```
Sensor data (batch/stream)
        │
        ▼
   DVC-versioned ingestion ──► feature engineering
        │
        ▼
   Training + Optuna search ──► MLflow (tracking + registry)
        │
        ▼
   Best model ──► GitHub Actions CI/CD ──► Azure ML endpoint (Docker)
        │                                        │
        │                                        ▼
        │                                 Live predictions
        │                                        │
        ▼                                        ▼
   Evidently drift checks ◄──────────────── prediction logs
        │
        ▼
   Dashboard (drift + performance)  ──►  retrain trigger on threshold
```

## Results To Report

- Failure-prediction metrics: ROC-AUC, precision/recall at the operating threshold, and remaining-useful-life error (RMSE).
- Detected vs injected drift on a held-out shifted distribution.
- CI/CD deploy time from merge to live endpoint.
- Endpoint latency (p50/p95) and uptime over the monitoring window.

## Repository Layout

```
drift-watch/
├── data/
│   ├── ingest.py           # load + version raw sensor data (DVC)
│   └── features.py         # rolling/lag/frequency features
├── training/
│   ├── train.py            # model training + MLflow logging
│   ├── tune.py             # Optuna hyperparameter search
│   └── register.py         # promote best run to registry
├── serving/
│   ├── app.py              # FastAPI scoring endpoint
│   └── Dockerfile
├── monitoring/
│   ├── drift.py            # Evidently drift + performance checks
│   └── retrain_trigger.py
├── dashboard/              # React drift/performance UI
├── .github/workflows/
│   └── deploy.yml          # CI/CD: test, build, push, deploy to Azure
├── dvc.yaml                # pipeline stages
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pull versioned data + reproduce the pipeline
dvc pull
dvc repro

# 3. Tune + train (logs to MLflow)
python training/tune.py --trials 50
python training/register.py --metric roc_auc

# 4. Serve locally
docker build -t drift-watch serving/
docker run -p 8000:8000 drift-watch

# 5. Run drift monitoring
python monitoring/drift.py --reference data/ref.parquet --current data/live.parquet
```

## License

MIT
