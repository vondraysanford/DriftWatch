# DriftWatch — End-to-End ML Pipeline with Drift Monitoring

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/github/license/vondraysanford/DriftWatch)](LICENSE)
[![Build plan: phase 5 of 6 complete](https://img.shields.io/badge/build_plan-phase_5_of_6_complete-orange)](DriftWatch-Guide.md)
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
| Server-side per request (features + inference + log write) | 12 ms |
| Round trip from a laptop to eastus2 | p50 93 ms, p95 101 ms |
| Cold start from zero replicas | 32.7 s |
| Scale-down after last request | 5 min |
| Idle cost | $0 (no replica runs) |

The 32.7-second cold start is the price of the zero-idle-cost design, not a number to hide. A minimum of one replica would remove it and cost roughly $30/month. The round-trip figure is mostly network distance, which is why the server-side number is reported separately.

**Pipeline** (`deploy.yml`, OIDC, no stored cloud secrets): a push to `main` runs the feature-contract check, pulls the registered model from the workspace registry, builds and pushes the image, deploys the Bicep template, and smoke-tests the live endpoint, failing the run if the near-failure example is not flagged. A docs-only commit is correctly skipped by `paths-ignore`.

**Managed online endpoint, demonstrated and torn down** (`managed-endpoint-demo.yml`, manual only): the registered model deployed to an Azure ML managed online endpoint with our own scoring script, so it takes the same raw cycles as the Container App. Run #4 went green in 18m 29s: endpoint up in 1m 6s, environment built and deployment live in 9m 25s, five invocations answered correctly (1.0000 / label 1 for the near-failure window, 0.0352 / label 0 for the healthy one, identical to the Container App; the run fails on a wrong label), logs captured, then teardown confirmed with nothing left billing. Round trips were 2.9 to 3.3 s, timed around the `az ml online-endpoint invoke` CLI call, which is dominated by CLI start-up and token acquisition: the same scoring script answers in about 10 ms locally, and the server-side figure was not isolated in this run. It never runs on merge, because it bills per instance-hour with no scale-to-zero. Evidence: [docs/evidence](docs/evidence/README.md).

**Drift, caught on real data** (measured 2026-09-04). The quarantined FD002 regime, six operating conditions where FD001 has one, was replayed through the live endpoint as production traffic: 24 held-out engines, a 20-cycle window every fifth cycle, 889 requests, zero failures. A control replay of 20 held-out FD001 engines (781 requests) went through the same endpoint. The detector compares the prediction log, per regime, against the champion's training engines for that regime, using Evidently (per-column normed Wasserstein distance, cut at 0.2 reference standard deviations; drift declared at 30% of raw input columns or any operating setting). A regime the champion never trained on is compared against everything it did train on, which is exactly when drift should be declared.

| Traffic through the live endpoint | Raw input columns drifted | Features drifted | Operating settings | Champion ROC-AUC on it |
|---|---|---|---|---|
| FD001, 20 held-out engines | 0 of 17 | 0 of 99 | unchanged | 0.9921 |
| FD002, 24 held-out engines | 17 of 17 | 98 of 99 | all three drifted; `setting_1` 10,911 SD from baseline | **0.5007** |

On the new regime the champion is a coin flip that flags every window as failing (recall 1.00, precision 0.16). Six sensors that are constant in FD001 started varying. The model's own inputs say the world changed, and the labels, derivable because every replayed engine runs to failure, confirm the damage. The detector refuses to issue a verdict on fewer than 200 predictions from 5 engines; its first negative control, 25 repeats of two windows, showed why.

**Retraining answers it.** On the mixed held-out bench (20 FD001 plus 52 FD002 engines, never trained on), champion v1 scores 0.5463 overall: 0.9923 on the FD001 part, 0.5003 on the FD002 part. A logistic regression retrained on FD001 plus the FD002 training split scores 0.9875 overall, 0.9846 on FD001 and 0.9887 on FD002. The regime is recovered at a cost of 0.008 on the original one.

**The loop, closed end to end** (2026-09-04, with a human in exactly one place). The scheduled drift workflow returned DRIFT and fired `repository_dispatch`. The retrain workflow trained a baseline and a 20-trial XGBoost search on FD001 plus the FD002 training split, judged them against the champion on the mixed bench, registered the winner as version 2 tagged `challenger`, and dispatched a promotion request. The deploy workflow paused in the `production` environment until a reviewer approved it ("Promoting to champion"), then tagged version 2 as champion and shipped it. Registering never changes what serves; only that approved promotion does. From drift verdict to registered challenger took about 9 minutes; to a green promotion run, about 16, the approval wait being the largest piece.

| Same 24 held-out FD002 engines through the live endpoint | Before the loop (v1) | After (v2) |
|---|---|---|
| ROC-AUC | 0.5007 | **0.9933** |
| Precision / recall at the operating threshold | 0.164 / 1.000 | 0.918 / 0.849 |
| Drift verdict, reference following the champion | DRIFT, 17 of 17 raw columns | no drift, 0 of 17 |

Two bugs the first run exposed, both fixed and kept in the record: the image tag was the commit SHA, which a promotion does not change, so Container Apps kept the old revision serving while the deploy reported success (tags are now commit plus model version, and the smoke test asserts the served version); and a mutable tag means a cold start after scale-to-zero can change production with no deploy at all. Merge to live endpoint on an ordinary push: 4m 46s.

## Still To Report

- The managed endpoint's server-side latency (the demonstration timed only the CLI round trip).

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
│   ├── train.py            # baseline or given params → MLflow run (--with-regime for the retrain)
│   ├── tune.py             # Optuna over XGBoost, grouped CV inside the training engines
│   ├── register.py         # register a run as a challenger version, tagged with its lineage
│   ├── challenge.py        # champion vs challenger on the mixed held-out bench; register on a win
│   └── promote.py          # move the champion tag; the one human-approved step
├── serving/
│   ├── app.py              # FastAPI: /predict, /health, /model
│   ├── schemas.py          # request/response shapes generated from data/schema.py
│   ├── model.py            # load the baked model, score one window
│   ├── sinks.py            # prediction log: Postgres locally, Blob JSONL on Azure
│   ├── fetch_model.py      # build-time pull of the registered model
│   └── Dockerfile
├── monitoring/
│   ├── logs.py             # read the prediction log back (Blob, Postgres, or local JSONL)
│   ├── replay.py           # send the quarantined regime through the live endpoint as traffic
│   ├── drift.py            # Evidently reference-vs-current, verdict, performance on labeled traffic
│   └── retrain_trigger.py  # verdict → repository_dispatch (fine-grained PAT, a GitHub credential)
├── dashboard/              # React drift/performance UI
├── infra/                  # Bicep: resource group, ML workspace, ACR, budget alert
├── scripts/
│   └── sanity_check.py     # feature contract + request validation, no data or cloud needed
├── .github/workflows/
│   ├── deploy.yml          # CI/CD: test, build, push, deploy (OIDC); promotions wait for a reviewer
│   ├── managed-endpoint-demo.yml  # manual only: stand up, prove, tear down
│   ├── drift.yml           # every 6 h: read the log, run Evidently, dispatch on drift
│   └── retrain.yml         # on dispatch: train on FD001 + regime, challenge the champion, register
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
