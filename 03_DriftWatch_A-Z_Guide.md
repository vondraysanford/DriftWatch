# DriftWatch — A-to-Z Build Guide

A step-by-step path from empty folder to a deployed, monitored ML pipeline on Azure. Plan for roughly 4–5 focused weekends — this is the most infrastructure-heavy of the three projects, which is exactly why it's valuable. Build the model first, then layer operations on top.

The core lesson this project teaches: **the model is maybe 20% of an ML system.** The other 80% — versioning, deployment, monitoring, retraining — is what "ML engineering" actually means, and what your enterprise background uniquely equips you for.

---

## Phase 0 — Prerequisites and Setup (½ weekend)

1. Install Python 3.11 and create a virtual environment.
2. Set up an Azure account. The free tier plus student credits cover this project. Create a resource group, an Azure ML workspace, and an Azure Container Registry.
3. Install the Azure CLI and authenticate (`az login`).
4. Initialize the git repo using the README layout.
5. Install core libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `mlflow`, `evidently`, `dvc`, `fastapi`, `uvicorn`, plus `azure-ai-ml`.

**Checkpoint:** `az ml workspace show` returns your workspace, and you can run a trivial MLflow logging script locally.

---

## Phase 1 — Data and Feature Engineering (1 weekend)

1. Download the NASA C-MAPSS Turbofan dataset. It contains run-to-failure sensor trajectories for many engines — ideal for predicting remaining useful life or imminent failure.
2. Initialize DVC in the repo and add the raw data under DVC tracking. This is data versioning — a core MLOps practice that signals maturity.
3. Explore the data in a notebook: understand the sensors, the failure cycles, and how to frame the target (binary "fails within N cycles" is the simplest start; remaining-useful-life regression is a stretch goal).
4. Write `data/ingest.py` (load + version) and `data/features.py`:
   - Rolling-window statistics (mean, std, min, max over the last k cycles).
   - Lag features.
   - Optionally frequency-domain features.
5. Frame the supervised problem and produce a clean feature table with train/test separation **by engine unit**, not by row (avoid leakage).

**Checkpoint:** A versioned, reproducible feature table exists, and `dvc repro` regenerates it.

---

## Phase 2 — Model Training and Tracking (1 weekend)

1. Write `training/train.py`: train an XGBoost classifier on the features, log everything to MLflow (params, ROC-AUC, precision/recall, confusion matrix, the trained model artifact).
2. Establish a simple baseline first (e.g., logistic regression) so XGBoost's value is demonstrable.
3. Write `training/tune.py` using Optuna to search XGBoost hyperparameters, with each trial logged to MLflow.
4. Write `training/register.py` to promote the best run's model into the MLflow model registry.
5. (Optional stretch) Add an LSTM in PyTorch for sequence modeling and compare it to XGBoost — a nice depth signal, but don't block the pipeline on it.

**Checkpoint:** MLflow shows tuned experiments, and the best model is registered with versioning.

---

## Phase 3 — Serve the Model (½ weekend)

1. Write `serving/app.py` with FastAPI: a `/predict` endpoint that accepts sensor feature input and returns a failure probability.
2. Load the registered model once at startup.
3. Log every prediction (inputs + output + timestamp) to a store — these logs feed drift monitoring later, so this step is not optional.
4. Write the `Dockerfile` and test the container locally.

**Checkpoint:** A local container serves predictions and writes prediction logs.

---

## Phase 4 — CI/CD to Azure (1 weekend)

This phase is where your existing DevOps experience shines and most ML candidates fall short.

1. Write `.github/workflows/deploy.yml` (GitHub Actions) with stages:
   - **Test:** run unit tests and a data/feature sanity check.
   - **Build:** build the Docker image, push to Azure Container Registry.
   - **Deploy:** deploy the image to an Azure ML managed online endpoint.
2. Store Azure credentials as GitHub repository secrets.
3. Trigger the pipeline on merge to `main`. Confirm a code change flows automatically to a live Azure endpoint.
4. Smoke-test the deployed endpoint with a real request.

**Checkpoint:** Merging to `main` automatically deploys the model to a live Azure endpoint; you can curl it.

---

## Phase 5 — Drift Monitoring (1 weekend)

This is the project's differentiator. Detecting silent model decay is exactly what production ML teams need.

1. Write `monitoring/drift.py` using Evidently:
   - Compare a **reference** distribution (training data) against **current** production inputs from your prediction logs.
   - Compute data-drift metrics and, where labels become available, performance decay.
2. Create a deliberately shifted test distribution and confirm your monitoring actually detects it. (Proving the detector works is as important as building it.)
3. Write `monitoring/retrain_trigger.py`: when drift crosses a threshold, kick off retraining (re-run the training pipeline; in a full system this would be an automated job).

**Checkpoint:** You can show drift being detected on a shifted distribution and a retrain being triggered.

---

## Phase 6 — Dashboard, Documentation, and Ship (1 weekend)

1. Build the React dashboard (`dashboard/`) with Recharts:
   - Live prediction volume and failure-probability distribution.
   - Drift metrics over time.
   - Model performance trend.
2. Back it with a small FastAPI metrics endpoint reading from your logs/monitoring outputs.
3. Finalize the README: fill in every "Results To Report" number (ROC-AUC, drift detection results, deploy time, endpoint latency/uptime).
4. Add an architecture diagram and 2–3 dashboard screenshots.
5. Write a short reflection on the operational lessons — this ties the project back to real ML engineering and reads well for both jobs and grad school.

**Final checkpoint:** A live (or recently-live, screenshotted) Azure endpoint, a monitoring dashboard, and a README that walks a reader from data to deployment to drift detection with real numbers.

---

## Common Pitfalls

- **Treating it as a modeling project.** The infrastructure is the point; a mediocre model in a great pipeline beats a great model in a notebook here.
- **Row-level train/test splits.** Split by engine unit or you'll leak and overstate performance.
- **Not logging predictions.** Without prediction logs you cannot do drift monitoring — wire this in at serving time.
- **A drift monitor you never validate.** Always prove it fires on a known shifted distribution.
- **Letting Azure costs run.** Use the free tier, scale endpoints to zero when idle, and tear down resources you're not actively demoing.
- **Vague results.** Report concrete numbers: AUC, deploy time, p95 latency, drift caught.
