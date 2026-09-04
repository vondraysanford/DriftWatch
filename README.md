# DriftWatch — End-to-End ML Pipeline with Drift Monitoring

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/github/license/vondraysanford/DriftWatch)](LICENSE)
[![Build plan: phase 3 of 6 complete](https://img.shields.io/badge/build_plan-phase_3_of_6_complete-orange)](DriftWatch-Guide.md)
[![Last commit](https://img.shields.io/github/last-commit/vondraysanford/DriftWatch)](https://github.com/vondraysanford/DriftWatch/commits/main)

[![Azure ML](https://img.shields.io/badge/Azure_ML-0078D4)](https://azure.microsoft.com/products/machine-learning)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white)](https://dvc.org/)
[![Evidently](https://img.shields.io/badge/Evidently-ed0400)](https://www.evidentlyai.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)

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
- Bicep — Azure infrastructure as code (workspace, registry, budget alerts)
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

## Results So Far

Measured 2026-09-02 on FD001. Models train on 80 engines and are scored on 20 held-out engines (split by engine unit, never by row) and on NASA's official `test_FD001` holdout (100 unseen engines, cut off before failure). Label: fails within 30 cycles. The operating threshold is chosen by max F1 on out-of-fold predictions inside the training engines; the held-out engines never influence it.

| Model | Held-out ROC-AUC | Held-out PR-AUC | Recall / precision at threshold | Official holdout ROC-AUC |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.9923 | 0.9673 | 0.897 / 0.914 (t = 0.407) | 0.9929 |
| XGBoost, defaults | 0.9899 | 0.9589 | 0.873 / 0.897 (t = 0.461) | 0.9908 |
| XGBoost, Optuna, 50 trials | 0.9899 | 0.9593 | 0.845 / 0.903 (t = 0.537) | 0.9912 |

The baseline won. On 20-cycle rolling features, FD001's degradation is close to linear, and fifty tuned trials could not beat a scaled logistic regression on engines it had never seen. Version 1 in the workspace registry is therefore the baseline, chosen by held-out ROC-AUC. Every run records the DVC hash of the data it trained on and the git commit. Phase 5's challenger-vs-champion check re-runs this comparison once the replayed regime exists.

**Local container** (measured 2026-09-03, Apple Silicon, `docker compose up`): image builds in 22 s, healthy 2 s after start, and 50 sequential `/predict` calls run at p50 6.7 ms and p95 8.8 ms end to end. Every prediction is written to the log sink before the response returns; with the sink stopped, `/predict` returns 500 rather than an unlogged result.

**Deployed** (Azure Container Apps, eastus2, min replicas 0): the same image returns identical probabilities to local and writes prediction logs to Blob Storage as its own managed identity, with no connection string or account key anywhere. Authentication to Azure from CI is an OIDC federated credential; the app registration holds zero password credentials.

| | |
|---|---|
| Warm request | 12-27 ms |
| Cold start from zero replicas | 32.7 s |
| Scale-down after last request | 5 min |
| Idle cost | $0 (no replica runs) |

The 32.7-second cold start is the price of the zero-idle-cost design, not a number to hide. A minimum of one replica would remove it and cost roughly $30/month.

## Still To Report

- Drift caught on the FD002/FD004 regime replay: which Evidently metrics fired, and at what values.
- Retrain loop: time from drift dispatch to a newly registered model version.
- CI/CD deploy time from merge to live endpoint.
- Deployed endpoint latency (p50/p95) and idle cost of the persistent demo (target: $0 at zero replicas).

## Build Log

Built in public — one post per phase lands at [vondraysanford.com](https://vondraysanford.com) as each phase ships:

1. Data + features: quarantine, secretless DVC on Azure, leakage-safe splits
2. Training + registry on Azure ML
3. Serving: raw cycle windows in, every prediction logged
4. Secretless CI/CD to a live endpoint
5. Catching real drift + closing the retrain loop
6. Dashboard + measured results

## Repository Layout

```
drift-watch/
├── data/
│   ├── schema.py           # raw C-MAPSS column layout shared by every stage
│   ├── ingest.py           # load + validate raw cycles, derive RUL (DVC-versioned)
│   ├── features.py         # rolling/lag features + label, shared with serving
│   └── split.py            # hold out whole engines for evaluation
├── notebooks/              # exploration only, never a pipeline stage
├── training/
│   ├── common.py           # config from env, data, models, metrics, plots, lineage tags
│   ├── train.py            # baseline or given params → MLflow run in the Azure ML workspace
│   ├── tune.py             # Optuna over XGBoost, grouped CV inside the training engines
│   └── register.py         # promote the best run to the workspace registry
├── serving/
│   ├── app.py              # FastAPI: /predict, /health, /model
│   ├── schemas.py          # request/response shapes generated from data/schema.py
│   ├── model.py            # load the baked model, score one window
│   ├── sinks.py            # prediction log: Postgres locally, Blob JSONL on Azure
│   ├── fetch_model.py      # build-time pull of the registered model
│   └── Dockerfile
├── monitoring/
│   ├── drift.py            # Evidently drift + performance checks
│   └── retrain_trigger.py  # threshold → repository_dispatch
├── dashboard/              # React drift/performance UI
├── infra/                  # Bicep: resource group, ML workspace, ACR, budget alert
├── scripts/
│   └── sanity_check.py     # feature contract + request validation, no data or cloud needed
├── .github/workflows/
│   ├── deploy.yml          # CI/CD: test, build, push, deploy (OIDC)
│   ├── managed-endpoint-demo.yml  # manual only: stand up, prove, tear down
│   └── retrain.yml         # dispatch-triggered retrain → evaluate → register
├── dvc.yaml                # pipeline stages
├── .env.example            # MLflow URI, experiment, and registry name (no secrets, never hardcoded)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill MLFLOW_TRACKING_URI from `az ml workspace show`
set -a; source .env; set +a

# 2. Pull versioned data + reproduce the pipeline (auth: `az login`, no keys)
dvc pull
dvc repro

# 3. Baseline, tuned XGBoost, then register the best run (all logged to the Azure ML workspace)
python -m training.train --model logreg
python -m training.tune --trials 50
python -m training.register --metric roc_auc

# 4. Serve locally (API + Postgres prediction log)
python -m serving.fetch_model          # registry -> serving/model/, baked in at build time
docker compose up --build
curl -f localhost:8000/health
curl -fs -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d @serving/examples/near_failure.json

# 5. Replay the held-out regime and run drift monitoring
python monitoring/drift.py --reference data/ref.parquet --current data/live_fd002.parquet
```

## License

MIT
