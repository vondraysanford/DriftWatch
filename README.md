# DriftWatch — End-to-End ML Pipeline with Drift Monitoring

A production-grade machine learning pipeline that predicts equipment failure from sensor time-series data, deployed to Azure through a secretless (OIDC) GitHub Actions pipeline, and monitored for real distribution shift in production. Covers the full MLOps lifecycle: ingestion, feature engineering, training with hyperparameter tuning, deployment, drift detection, and automated retraining.

This project bridges enterprise software engineering with machine learning operations — proving you can not only train a model but ship and maintain it as a reliable production service.

## Why This Project

Plenty of candidates can train a model in a notebook. Far fewer can operationalize one: version the data and model, automate retraining, deploy behind an API, and detect when the model silently degrades in production. That operational maturity is exactly what separates an ML engineer from a notebook hobbyist, and it directly reuses enterprise deployment experience (CI/CD, containers, cloud, incident response).

## What It Does

- Ingests batch sensor data and engineers time-series features (rolling stats, lag features, optional frequency-domain signals).
- Trains a failure-prediction model (XGBoost vs a logistic-regression baseline, optional LSTM) with automated hyperparameter search.
- Tracks every experiment — params, metrics, artifacts — in the **Azure ML workspace via MLflow** and registers the best model in the workspace registry.
- Deploys through **GitHub Actions with OIDC federated credentials** (no stored cloud secrets): an Azure ML managed online endpoint is demonstrated and torn down, while a **scale-to-zero Azure Container Apps** endpoint stays live as the persistent demo.
- Replays a **held-out C-MAPSS operating regime (FD002/FD004) as production traffic** and detects the resulting real data drift with Evidently.
- Closes the loop automatically: drift threshold crossed → `repository_dispatch` → retraining workflow → challenger evaluated against champion → new version registered.
- Surfaces predictions, drift, and performance trends on a React dashboard.

## Tech Stack

**Data & Modeling**
- Python 3.11
- pandas / NumPy — data wrangling and feature engineering
- scikit-learn — preprocessing, baselines, metrics
- XGBoost — primary failure-prediction model
- PyTorch — optional LSTM for sequence modeling
- Optuna — hyperparameter optimization

**MLOps & Tracking**
- MLflow — experiment tracking + model registry, hosted in the Azure ML workspace
- Evidently AI — data drift and model performance monitoring
- DVC (Data Version Control) — dataset and pipeline versioning

**Deployment & Infrastructure**
- Azure Machine Learning — training, registry, and a demonstrated managed online endpoint
- Azure Container Apps — persistent scale-to-zero demo endpoint
- Azure Container Registry — image hosting
- FastAPI — model-serving API
- Docker — containerization
- GitHub Actions — CI/CD with OIDC federated credentials (no stored secrets)

**Monitoring Dashboard**
- React + Vite — frontend
- Recharts — metric and drift visualizations
- FastAPI — dashboard backend / metrics API

**Dataset**
- NASA C-MAPSS Turbofan Engine Degradation (public) — four subsets with distinct operating conditions and fault modes. FD001 trains the model; FD002/FD004 are held out untouched as "production" replay traffic for drift detection.

## Architecture

```
C-MAPSS FD001 (train)                      C-MAPSS FD002/FD004 (held out)
        │                                              │
        ▼                                              │  replayed as
  DVC-versioned ingestion ──► feature engineering      │  "production" traffic
        │                                              │
        ▼                                              │
  Training + Optuna search ──► MLflow tracking + registry (Azure ML workspace)
        │                                              │
        ▼                                              ▼
  GitHub Actions CI/CD (OIDC, no stored secrets)
        ├──► Azure ML managed endpoint (demonstrated, then torn down)
        └──► Azure Container Apps (persistent demo, scales to zero)
                                │
                                ▼
                         prediction logs
                                │
                                ▼
                  Evidently drift + performance checks
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
  Dashboard (drift + performance)    repository_dispatch ──► retrain
                                     workflow ──► evaluate ──► register v2
```

## Results To Report

- Failure-prediction metrics: ROC-AUC and precision/recall at the operating threshold, baseline vs tuned. (Add RUL RMSE only if the regression stretch goal ships.)
- Drift caught on the FD002/FD004 regime replay: which Evidently metrics fired, and at what values.
- Retrain loop: time from drift dispatch to a newly registered model version.
- CI/CD deploy time from merge to live endpoint.
- Endpoint latency (p50/p95) and idle cost of the persistent demo (target: $0 at zero replicas).

## Build Log

Built in public — posts land at [vondraysanford.com](https://vondraysanford.com) as each phase ships:

1. Data + baseline
2. Training + registry on Azure ML
3. Secretless CI/CD to a live endpoint
4. Catching real drift + closing the retrain loop

## Repository Layout

```
drift-watch/
├── data/
│   ├── ingest.py           # load + version raw sensor data (DVC)
│   └── features.py         # rolling/lag/frequency features
├── training/
│   ├── train.py            # model training + MLflow logging (Azure ML workspace)
│   ├── tune.py             # Optuna hyperparameter search
│   └── register.py         # promote best run to the workspace registry
├── serving/
│   ├── app.py              # FastAPI scoring endpoint
│   └── Dockerfile
├── monitoring/
│   ├── drift.py            # Evidently drift + performance checks
│   └── retrain_trigger.py  # threshold → repository_dispatch
├── dashboard/              # React drift/performance UI
├── .github/workflows/
│   ├── deploy.yml          # CI/CD: test, build, push, deploy (OIDC)
│   └── retrain.yml         # dispatch-triggered retrain → evaluate → register
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

# 3. Tune + train (logs to the Azure ML workspace via MLflow)
python training/tune.py --trials 50
python training/register.py --metric roc_auc

# 4. Serve locally
docker build -t drift-watch serving/
docker run -p 8000:8000 drift-watch

# 5. Replay the held-out regime and run drift monitoring
python monitoring/drift.py --reference data/ref.parquet --current data/live_fd002.parquet
```

## License

MIT
