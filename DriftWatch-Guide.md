# DriftWatch — A-to-Z Build Guide

A step-by-step path from empty folder to a deployed, monitored ML pipeline on Azure. Plan for roughly 5–7 focused weekends — this is the most infrastructure-heavy of the three projects, which is exactly why it's valuable. Build the model first, then layer operations on top.

The core lesson this project teaches: **the model is maybe 20% of an ML system.** The other 80% — versioning, deployment, monitoring, retraining — is what "ML engineering" actually means, and what your enterprise background uniquely equips you for.

---

## Build-Log Plan (decide this before Phase 0)

DriftWatch is a build-in-public project. One post per phase, written the moment the phase's checkpoint is verified, so screenshots and numbers are captured *as they happen* instead of reconstructed in Phase 6. Posts are generated from a factual brief (measured numbers, decisions and why, gotchas, what is explicitly not done yet) through the portfolio site's blog post generator.

- **Post 1 — Data + features (Phase 1):** the C-MAPSS framing and quarantine, secretless DVC on Azure, the evidence behind N = 30, leakage-safe splits, training/serving feature parity. *(Brief written 2026-09-02.)*
- **Post 2 — Training + registry (Phase 2):** the baseline number to beat, tuned vs baseline results, Azure ML experiment and registry views.
- **Post 3 — Serving (Phase 3):** `/predict` on raw cycle windows, the model baked into the image, every prediction logged.
- **Post 4 — Secretless CI/CD to a live endpoint (Phase 4):** merge-to-live pipeline, the managed-endpoint demonstration, the scale-to-zero demo.
- **Post 5 — Catching real drift + closing the loop (Phase 5):** the regime-replay detection and the automated retrain.
- **Post 6 — Dashboard + measured results (Phase 6):** the numbers the README reports and what they cost.

At every checkpoint below, screenshot the evidence (Azure ML runs, Actions pipeline, drift report) the moment it exists.

---

## Phase 0 — Prerequisites and Setup (½ weekend)

1. Install Python 3.11 and create a virtual environment.
2. Set up an Azure account. The free tier plus new-account credits cover this project **if** you follow the cost rules in Common Pitfalls — read them now, especially the managed-endpoint one. Stand up the durable Azure footprint with Bicep templates in `infra/`, deployed with `az deployment`: resource group (eastus2), Azure ML workspace and its dependencies, and an Azure Container Registry. The same deployment puts a budget alert on the resource group (percentage notifications at 50/80/100%) before anything else exists. Identity setup (Entra app registration, OIDC federated credential) and the ephemeral managed-endpoint demo stay imperative CLI, on purpose.
3. Install the Azure CLI and authenticate (`az login`).
4. Initialize the git repo using the README layout.
5. Install core libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `mlflow`, `evidently`, `dvc[azure]`, `fastapi`, `uvicorn`, plus `azure-ai-ml` and `azureml-mlflow` (the plugin MLflow needs to reach the workspace). All exact-pinned in `requirements.txt`.
6. **Point MLflow at the workspace.** Retrieve the workspace's MLflow tracking URI (`az ml workspace show --query mlflow_tracking_uri`) and set `MLFLOW_TRACKING_URI`. The Azure ML workspace *is* an MLflow tracking server and model registry — no local server to run, and every run you log shows up in Azure ML studio. Same code, dramatically more AI-300-aligned evidence.

**Checkpoint:** `az ml workspace show` returns your workspace, and a trivial MLflow test run appears in Azure ML studio's experiment view.

✅ **Done 2026-08-23.** Deployment `driftwatch-infra` created the footprint (resource group `DriftWatch`, eastus2, six resources plus the $30 budget); the `phase0-smoke-test` run with `setup_complete=1` confirmed in studio.

---

## Phase 1 — Data and Feature Engineering (1 weekend)

1. Download the NASA C-MAPSS Turbofan dataset. It ships as **four subsets (FD001–FD004) with different operating conditions and fault modes.** Train on FD001 only. FD003 goes unused (it shares FD001's single operating condition, so it makes a weak drift case). **Set FD002 and FD004 aside untouched — they are your "production" traffic for drift detection in Phase 5.** Don't explore them, don't peek; contamination here weakens the entire drift story.

   ✅ *Done 2026-08-23.* All four subsets, NASA's readme, and the PHM08 paper sit in `data/raw/` (14 files). Nothing has opened FD002/FD004.

2. Initialize DVC in the repo and add the raw data under DVC tracking. Use an Azure Blob container in the project resource group as the DVC remote. This is data versioning — a core MLOps practice that signals maturity.

   ✅ *Done 2026-09-02.* `dvc init`; `data/raw` tracked as one artifact (`data/raw.dvc`); default remote `azure://dvc` on the Phase 0 storage account, authenticated through `az login` via a Storage Blob Data Contributor role added to `infra/` (no keys). Verified: `dvc status -c` reports cache and remote in sync, and a `dvc pull` into a fresh repo matched the originals byte for byte.

3. Explore FD001 in a notebook: understand the sensors, the failure cycles, and how to frame the target (binary "fails within N cycles" is the simplest start, default N = 30, finalized during this exploration; remaining-useful-life regression is a stretch goal — treat it as optional and don't promise its numbers anywhere).

   ✅ *Done 2026-09-02.* `notebooks/01_fd001_exploration.ipynb`, committed with outputs, reads FD001 only (single `SUBSET` constant). Findings: 100 engines, lifetimes 128 to 362 cycles; operating settings constant, single regime confirmed; 7 of 21 sensors have two or fewer distinct values and are dropped by that rule; the strongest sensors sit 2 to 4 SD from healthy baseline inside the last 30 cycles and under 1 SD past 60. Decisions: N = 30 (15% positive rate), rolling window k = 20 (shortest official test engine is 31 cycles), settings excluded from features but kept in the raw prediction log.

4. Write `data/ingest.py` (load + version) and `data/features.py`:
   - Rolling-window statistics (mean, std, min, max over the last k cycles).
   - Lag features.
   - Optionally frequency-domain features.

   ✅ *Done 2026-09-02.* `data/schema.py` (shared column layout), `data/ingest.py` (parse, validate, derive RUL; refuses FD002/FD004 by name), `data/features.py` (per kept sensor: current value, rolling mean/std/min/max over 20 cycles, deltas at lags 5 and 10, plus engine age; label = RUL ≤ 30). Frequency-domain features skipped: the trends are plain in the time domain. Serving parity verified: `build_features` on the last 20 raw cycles of one engine reproduces the training row to within 1e-10.

5. Frame the supervised problem and produce a clean feature table with train/test separation **by engine unit**, not by row (avoid leakage).

   ✅ *Done 2026-09-02.* `data/split.py` holds out 20 of 100 engines by seed (42): train 80 engines / 14,870 rows, test 20 engines / 3,861 rows, positive rate 0.167 / 0.161, with an assertion that no engine sits on both sides. `dvc.yaml` runs five stages (ingest_train, ingest_holdout, features_train, features_holdout, split). The official `test_FD001` becomes `holdout_official.parquet` (11,196 rows, positive rate 0.030) as a secondary evaluation set with unseen, not-yet-failed engines.

**Checkpoint:** A versioned, reproducible feature table exists, `dvc repro` regenerates it, and FD002/FD004 sit untouched in raw storage.

✅ **Done 2026-09-02.** `dvc repro` builds every table from `data/raw.dvc` in a few seconds; a second `dvc repro` reports everything up to date; `dvc push` sent the six outputs to `azure://dvc` and `dvc status -c` is in sync. The ingest stages depend on `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` by name, so the DAG itself shows FD002/FD004 are never read.

---

## Phase 2 — Model Training and Tracking (1 weekend)

1. Write `training/train.py`: train an XGBoost classifier on the features, log everything to MLflow — which now lands in the Azure ML workspace (params, ROC-AUC, precision/recall, confusion matrix, the trained model artifact).
2. Establish a simple baseline first (e.g., logistic regression) so XGBoost's value is demonstrable.
3. Write `training/tune.py` using Optuna to search XGBoost hyperparameters, with each trial logged as a run.
4. Write `training/register.py` to promote the best run's model into the **workspace model registry** via MLflow's registry API. This registered, versioned model is exactly what CI/CD deploys in Phase 4.

   ✅ *Steps 1 to 4 done 2026-09-02.* `training/common.py` (config from env only, data, models, metrics, plots, lineage tags), `train.py`, `tune.py`, `register.py`, all run as modules with `.env.example` variables. Experiment `driftwatch-training` in the workspace. Baseline (scaler + logistic regression): held-out ROC-AUC 0.9923, PR-AUC 0.9673, recall 0.897 / precision 0.914 at threshold 0.407 (chosen by max F1 on out-of-fold predictions inside the training engines), official holdout ROC-AUC 0.9929. XGBoost defaults: 0.9899. XGBoost after 50 Optuna trials (5-fold GroupKFold by engine, about 8 s per trial): best CV 0.9886, held-out 0.9899, holdout 0.9912. The baseline won; `register.py` picked it by held-out ROC-AUC and registered `driftwatch-failure-classifier` version 1, tagged with run id, metric, DVC data hash, and git commit. Gotcha: MLflow 3's `log_model` and `runs:/` URIs go through a logged-models endpoint the Azure ML tracking server does not implement (404). Models are uploaded as run artifacts with `mlflow.sklearn.save_model` + `log_artifacts`, and addressed by the run's artifact URI or `models:/<name>/<version>`.

5. (Optional stretch) Add an LSTM in PyTorch for sequence modeling and compare it to XGBoost — a nice depth signal, but don't block the pipeline on it.

   *Skipped on purpose (2026-09-02): momentum over extras until the six phases are done.*

**Checkpoint:** Azure ML studio shows tuned experiments and a registered, versioned model. *(Build-log post 2 material: baseline-vs-tuned table, registry screenshot.)*

✅ **Done 2026-09-03.** Three top-level runs plus 50 nested trials in the workspace; registry version 1 loads fresh via `models:/driftwatch-failure-classifier/1` and reproduces the held-out ROC-AUC of 0.9923; `az ml model list` shows the model. All training ran on the Mac; the workspace was tracking and registry only, so no Azure compute was billed.

---

## Phase 3 — Serve the Model (½ weekend)

1. Write `serving/app.py` with FastAPI: a `/predict` endpoint that accepts a window of raw cycles for one engine, computes features server-side with the same `data/features.py` used in training, and returns a failure probability.
2. Bake the model into the image at build time: the Docker build pulls the registered version from the workspace registry (same pattern as DocQuery and AgentReview), and the app loads it from a local path at startup. No runtime registry pull, no runtime credential for model loading.
3. Log every prediction (raw inputs + computed features + output + timestamp). The sink is configuration-driven: a Postgres instance in Docker locally, JSONL in Azure Blob Storage via managed identity on Container Apps (the Container Apps filesystem is ephemeral at min replicas 0). These logs feed drift monitoring later, so this step is not optional.
4. Write the `Dockerfile` and test the container locally.

   ✅ *Done 2026-09-03.* `serving/` holds `app.py` (`/predict`, `/health`, `/model`, JSON logs to stdout), `schemas.py` (request rows generated from `data.schema`, so serving cannot drift from ingestion), `model.py`, `sinks.py`, `config.py`, `fetch_model.py`, `requirements.txt`, `Dockerfile`, and two example bodies under `serving/examples/`. Build from the repo root (`docker build -f serving/Dockerfile .`) so the image gets the shared `data/` package. Base images pinned by digest: `python:3.11-slim` (sha256:9534e5a8…) and `postgres:17-alpine` (sha256:18cfe3ef…). Image builds in 22 s; the stack is healthy 2 s after `docker compose up`.

**Checkpoint:** A local container serves predictions and writes prediction logs.

✅ **Done 2026-09-03.** Held-out engine 8 scores 1.0000 (label 1) at its final cycle and 0.0352 (label 0) at cycles 41-60; the served probability matches the DVC feature table for the same engine and cycle to 6e-15. Short windows (19 cycles), mixed engine units, and cycle gaps all return 422. Latency over 50 sequential requests through the container: p50 6.7 ms, p95 8.8 ms. 53 predictions logged to Postgres, each with 20 raw cycles (all 26 columns, operating settings included), 99 features, output, threshold, and model version. With the database stopped, `/health` returns 503 and `/predict` returns 500 rather than an unlogged prediction, then recovers 2 s after the database restarts. The blob sink was exercised separately against the real `predictions` container (test blobs since deleted).

---

## Phase 4 — CI/CD to Azure (1 weekend)

This phase is where your existing DevOps experience shines and most ML candidates fall short.

1. **Set up OIDC federated credentials — no stored cloud secrets.** Create a Microsoft Entra app registration (or user-assigned managed identity) with a federated credential scoped to your repo and `main` branch, grant it RBAC on the resource group, and authenticate in Actions with `azure/login` using client/tenant/subscription IDs — identifiers, not secrets. There is nothing to leak and nothing to rotate. Given your KodeKloud contribution is literally a hardcoded-credentials fix, your own pipeline being secretless is on-brand and interviews well.

   ✅ *Done 2026-09-03.* App registration `driftwatch-github-oidc` with a federated credential on subject `repo:vondraysanford/DriftWatch:ref:refs/heads/main`; `az ad app credential list` returns 0 password credentials. Four scoped roles rather than blanket subscription access: AcrPush on the registry, Contributor on the resource group, AzureML Data Scientist on the workspace, Storage Blob Data Contributor on the storage account (Contributor is control-plane only, so each data-plane action needs its own role). Every command is recorded in `infra/oidc-setup.md`, including teardown. **Gotcha: the first pipeline run failed with `AADSTS700213`** because GitHub presents an ID-qualified subject (`repo:<owner>@<owner-id>/<repo>@<repo-id>:ref:refs/heads/main`), not the name-based form in most documentation. Matching the presented subject fixed it, and that form is the stronger one: a rename cannot transfer trust. **Second variant (Phase 5):** a job that declares a GitHub `environment:` presents `...:environment:<name>` instead of the branch, so the `ci` and `production` environments introduced for gated promotion each needed their own federated credential. Four credentials now cover every subject the workflows can present; `infra/oidc-setup.md` lists them.
2. Write `.github/workflows/deploy.yml` with stages:
   - **Test:** run unit tests and a data/feature sanity check.
   - **Build:** build the Docker image, push to Azure Container Registry.
   - **Deploy:** Container Apps only; the managed-endpoint demonstration is a separate manually triggered workflow (see step 3).
3. **Deploy twice, on purpose:**
   - **Azure ML managed online endpoint — the demonstration.** A separate `workflow_dispatch`-triggered workflow, never part of the merge pipeline. Deploy the registered model, smoke-test it, and capture invocation logs, latency numbers, and screenshots. Then **tear it down.** Managed online endpoints bill for at least one instance around the clock — there is no scale-to-zero — which quietly runs $70+/month if left up.

   ✅ *Done 2026-09-03, run #4 of `managed-endpoint-demo`, green in 18m 29s* (`docs/evidence/phase-4/`, run page and summary table): endpoint created in 1m 6s, environment built and deployment created in 9m 25s, five invocations with the same raw-cycle example bodies the Container App is smoke-tested with, all correct (1.0000 / label 1 and 0.0352 / label 0, identical to the Container App; the run fails on a wrong label), round trips 2.9 to 3.3 s around the `az ml online-endpoint invoke` CLI call, which is mostly CLI start-up and token acquisition since the same script scores in about 10 ms locally, logs captured as a run artifact, teardown, and "Confirm nothing is left billing" passing after 6m 47s. Three failed runs came first, each a lesson and each torn down cleanly by the `always()` steps: (1) `Microsoft.PolicyInsights` was not registered, and Azure's error named neither the provider nor the subscription; (2) `az ml online-deployment create` requires a YAML file although `online-endpoint create` takes `--name`; (3) Azure's no-code MLflow scoring script crashed on `ModuleNotFoundError: No module named 'azureml'`, its own `azureml-ai-monitoring` missing from the environment Azure built. The fix for (3) was to own the scoring script and environment (`infra/managed-endpoint/`), which also gave the managed endpoint the same raw-cycles contract as the Container App. A fourth attempt with `Standard_DS3_v2` was rejected at validation: managed endpoints check quota at double the SKU's vCPUs, so 4 cores asked for 8 against a family limit of 6; `DS2_v2` needs 4 and the model peaks at 254 MB. The registry version now carries its operating threshold as a tag so any deployment can read it.
   - **Azure Container Apps — the persistent demo.** Deploy the same image with min replicas set to 0. This is the endpoint that stays live for your README and dashboard, and it costs ~$0 when idle. It's the same pattern AgentReview already runs on.

   ✅ *Container Apps live 2026-09-03,* proven by hand before wiring CI so a pipeline failure could not be confused with an application failure. `infra/modules/containerapp-env.bicep` adds the managed environment plus a user-assigned identity holding AcrPull and Storage Blob Data Contributor; `infra/containerapp.bicep` deploys the app itself and is the only template CI touches, so a later `az deployment sub create` cannot revert production to a placeholder image. Ingress external on 8000, min replicas 0, max 1. Live results: identical probabilities to local (1.0000 / 0.0352 for the two examples), 12-27 ms per request, and prediction logs written to `predictions/dt=<date>/hour=<hour>/<replica>.jsonl` by the app's own identity with no connection string. **Gotcha: the local image was arm64 and Container Apps runs amd64,** so the build needs `--platform linux/amd64`; the workflow states it explicitly. Scale-to-zero measured: the app drops to 0 replicas 5 minutes after the last request, and the next request takes 32.7 s to answer while a replica starts. That is the honest cost of $0 idle.
4. Trigger the pipeline on merge to `main`. Confirm a code change flows automatically to the live endpoint, then smoke-test it with a real request.

   ✅ *Done 2026-09-04.* First green run deployed image tag `ee495b2…` to revision `ca-driftwatch--0000001` with no human touching Azure. Verified afterwards: `/model` reports registry version 1 with the training run's data hash and commit, both examples score identically to local (1.0000 / 0.0352), server-side latency 12 ms, and 25 predictions are logged in Blob Storage across two replica files. The follow-up docs-only commit was correctly skipped by `paths-ignore`, so a markdown edit does not redeploy.

**Checkpoint:** Merging to `main` deploys automatically; managed-endpoint evidence is captured and the endpoint torn down; a scale-to-zero Container Apps endpoint answers `curl`. *(Build-log post 4 material.)*

✅ **Done 2026-09-03.** All three parts hold: `deploy.yml` deploys on push to `main` with no stored cloud secret (federated credential, zero password credentials on the app registration, four resource-scoped roles); the managed-endpoint demonstration ran green, captured its evidence, and confirmed its own teardown; the Container App answers `curl` at min replicas 0 (server-side 12 ms, cold start 32.7 s, idle cost $0). Month-to-date spend at the end of the phase was under $1 against the $30 budget.

---

## Phase 5 — Drift Monitoring (1 weekend)

This is the project's differentiator. Detecting silent model decay is exactly what production ML teams need.

1. Write `monitoring/drift.py` using Evidently:
   - **Reference** distribution: the FD001 training features.
   - **Current** distribution: production inputs from your prediction logs.
   - Compute data-drift metrics and, where labels become available, performance decay.
   - Run it as a scheduled GitHub Actions workflow, not by hand.

   ✅ *Done 2026-09-04.* `monitoring/drift.py` (Evidently `DataDriftPreset`) compares the prediction log, per regime, against the champion's training engines for that regime; reference set follows the champion's registry `data` tag; performance is scored wherever labels are derivable. `drift.yml` runs every 6 hours (and on demand), pulls the reference via DVC with the federated identity, reads the champion's tags from the registry, publishes the HTML reports and verdict as an artifact, and dispatches `drift-detected` only on a drift verdict. First run: 2m 17s, DRIFT (`docs/evidence/phase-5/drift-run-1-green.png`).

2. **Create the drift with real data, not synthetic jitter: replay FD002 (or FD004) through the live endpoint as "production" traffic.** Replay the run-to-failure trajectories so labels are derivable after the fact (the retrain and the champion-vs-challenger evaluation need them). These subsets have genuinely different operating conditions and fault modes, so your detector fires on an *actual regime change*. "My monitor caught a real distribution shift" is a far better interview line than "I perturbed my features." (Optionally keep a small synthetic-shift case as a unit test of the detector itself.)

   ✅ *Done 2026-09-04.* FD002 opened deliberately: `data/ingest.py --replay` lifts the quarantine with a warning and offsets unit numbers by 1000, and `dvc.yaml` gained `ingest_regime`, `features_regime`, `split_regime` (260 engines, 53,759 cycles, split 208/52 by engine; FD001 hashes unchanged). `monitoring/replay.py` sent 24 held-out FD002 engines through the live Container App, a 20-cycle window every fifth cycle: 889 requests, 0 failures, first response 33.4 s (cold start), then p50 96.9 ms round trip / 11.3 ms server-side. Labels are derivable because every engine runs to failure: the champion scored ROC-AUC **0.5007** on the new regime with recall 1.00 and precision 0.16 (it flags everything). A control replay of 20 held-out FD001 engines (781 requests) scored 0.9918. `monitoring/drift.py` (Evidently `DataDriftPreset`) on the same two-hour window: FD001 traffic 0 of 17 raw columns and 0 of 99 features drifted, no verdict of drift; FD002 traffic 17 of 17 and 98 of 99, all three operating settings drifted (`setting_1` at 10,911 reference SD), six FD001-constant sensors now varying, verdict DRIFT. Two detector lessons: a minimum-sample gate (200 predictions, 5 engines) after 25 repeats of two windows produced a false alarm, and the per-column cut raised from Evidently's 0.1 to 0.2 SD after ordinary engine-to-engine variation reached 0.15. The comparison is per regime against the champion's training engines for that regime, and the reference set follows the champion's `data` tag. Verified on the same window: with a champion trained on FD001 only, FD001 traffic is clean and FD002 traffic is DRIFT; with a champion trained on FD001+FD002, FD002 traffic compared with FD002 training engines drifts on 0 of 17 raw columns and 1 of 99 features, so the loop stops re-firing once it has answered. (A first version compared pure FD002 traffic with the mixed reference as a whole and flagged 15 of 17 columns: that was the composition difference, not a regime change. Caught by checking the claim before writing it down.)
3. **Make the retrain trigger a real mechanism.** In `monitoring/retrain_trigger.py`: when drift crosses the threshold, fire a `repository_dispatch` event to GitHub using a fine-grained PAT scoped to this repo, stored as an Actions secret (the built-in `GITHUB_TOKEN` cannot start workflows; this is a GitHub credential, not a cloud secret, so the "no stored cloud secrets" claim holds). A `retrain.yml` Actions workflow retrains on FD001 plus the replayed regime, evaluates the challenger against the current champion on a mixed held-out test set (split by engine unit), and registers the new version. (Optional: require manual approval via a protected environment before the new version deploys.)

   ✅ *Done 2026-09-04.* `drift.yml` (manual dispatch, 24 h window) returned DRIFT and dispatched `retrain.yml`, which trained a logistic-regression baseline and a 20-trial XGBoost search on FD001 plus the FD002 training split, judged them against the champion on the mixed bench (champion v1 0.5463; best challenger 0.9875), registered the winner as version 2 (`stage=challenger`), and dispatched `model-registered` to `deploy.yml`. That run paused in the `production` environment ("waiting for review"), was approved with the comment "Promoting to champion", promoted version 2 (version 1 to `stage=previous`), and finished green in 6m 51s. Gotchas: (1) jobs in a GitHub environment present `...:environment:<name>` as the OIDC subject, so `ci` and `production` each needed a federated credential; (2) the promotion's image tag was the commit SHA, identical to the previous push, so Container Apps kept the old revision (model version 1) serving while the deploy reported success. Fixed by tagging images with commit plus model version and asserting the served version in the smoke test. The same mutable tag then showed its other face: after the app scaled to zero, the next cold start re-pulled the tag, which by then resolved to the version 2 image, and production changed with no deploy at all. Unique tags close both doors. Verified after the fix: the next push created revision `ca-driftwatch--0000008` from image `<sha>-v2` and the smoke test asserted the served version.

**Checkpoint:** Drift detected on the FD002/FD004 replay, and the dispatch → retrain → evaluate → register loop demonstrably runs end to end.

✅ **Done 2026-09-04.** Drift detected on the FD002 replay (17 of 17 raw columns, all operating settings, champion ROC-AUC 0.5007 on that traffic vs 0.9921 on FD001 traffic); the loop ran dispatch → retrain → challenge → register → approved promotion with no human touching Azure; version 2 (logistic regression on FD001+FD002, bench 0.9875 vs 0.5463) is champion. Verified live afterwards: the same 24 FD002 engines score ROC-AUC 0.9933 through the endpoint (from 0.5007), precision 0.918 / recall 0.849, and the monitor with the reference following the champion reports no drift on FD002 traffic (0 of 17 raw, 1 of 99 features). About 9 minutes from drift verdict to registered challenger, about 16 to a green promotion run. Five screenshots in `docs/evidence/phase-5/`, including the green run that had not actually deployed, kept on purpose.

---

## Phase 6 — Dashboard, Documentation, and Ship (1 weekend)

1. Build the React dashboard (`dashboard/`) with Recharts:
   - Live prediction volume and failure-probability distribution.
   - Drift metrics over time.
   - Model performance trend.
   ✅ *Done 2026-09-04.* `dashboard/` is Vite + React + TypeScript + Recharts (react 19.2.8, recharts 3.10.1, vite 8.2.2, typescript 7.0.2, exact pins with a lockfile), built before the image and served by the same container at `/dashboard`. Views: KPI tiles (champion, latest verdict with status dot and label, ROC-AUC on the replayed regime, prediction volume, latest deployment); predictions per hour stacked by regime; failure-probability distribution with the operating threshold; drift share per regime across detector runs against the 30% line; champion ROC-AUC on labeled traffic with promotions as reference lines; champion-vs-challenger, deployments, and recent-prediction tables. Design followed the data-viz method: form before color, two categorical slots (blue FD001, orange FD002) validated for CVD separation in light and dark (worst ΔE 24.7 / 26.8), status colors only for verdicts and always with a label, 2px lines and ≥8px markers with surface rings, hairline solid grids, a legend on every two-series chart, a table twin behind every chart, one filter row, tooltips on every mark, straight segments between detector runs (they are discrete), and a headless render checked by eye before shipping (`docs/evidence/phase-6/`).

2. Back it with a small FastAPI metrics endpoint reading from your logs/monitoring outputs.

   ✅ *Done 2026-09-04.* `serving/metrics.py` mounts `/api/summary`, `/api/predictions`, `/api/drift`, `/api/performance` in the serving app, reading the prediction log through `monitoring/logs.py` (Blob on Azure, Postgres under docker compose) and the monitoring feeds the workflows now publish to blob storage under `monitoring/` (drift.yml verdicts, retrain.yml challenge results, deploy.yml deployment records), cached for 60 s. One container, one URL, no second deploy pipeline, and the app's existing managed identity is the only credential.
3. Finalize the README: fill in every "Results To Report" number — ROC-AUC and precision/recall at the operating threshold, drift caught on the regime replay, merge-to-live deploy time, p50/p95 latency, and the idle cost of the persistent demo. Include RUL RMSE **only** if the regression stretch actually shipped.
   ✅ *Done 2026-09-04.* Every number in the README's results is measured: held-out and holdout ROC-AUC and precision/recall for all three Phase 2 models, drift metrics that fired on the replay (17 of 17 raw columns, all settings, `setting_1` at 10,911 SD), merge-to-live 4m 46s, server-side 12 ms and laptop round trip p50 93 / p95 101 ms, cold start 32.7 s, idle cost $0, before/after ROC-AUC on the replayed regime 0.5007 → 0.9933. RUL RMSE is absent because the regression stretch did not ship. The one unmeasured number, the managed endpoint's server-side latency, is listed as such rather than implied.

4. Add an architecture diagram and 2–3 dashboard screenshots.

   ✅ *Done 2026-09-04.* Mermaid architecture diagram in the README (rendered by GitHub, nothing to install) replacing the ASCII sketch; dashboard renders in light and dark under `docs/evidence/phase-6/` and embedded in the README.
5. Write a short reflection on the operational lessons — this ties the project back to real ML engineering. *(Build-log post 6.)*

   ✅ *Done 2026-09-04.* The README's "Lessons" section (eight of them, each tied to a measured event in this build) and the post 6 brief.

**Final checkpoint:** The Container Apps endpoint live, the managed-endpoint demonstration documented, a monitoring dashboard up, and a README that walks a reader from data to deployment to drift detection with real numbers.

*Finishing touches, 2026-09-04:* the dashboard restyled to the vondraysanford.com design system (tokens, fonts, title block, drafting grid; chart colors re-validated in both modes), prepared for Cloudflare Pages at its own subdomain like the sibling projects (`VITE_API_BASE_URL` to the Container App, `VITE_BASE=/`, and `CORS_ALLOW_ORIGINS` as an environment variable on the Container App, set in `infra/containerapp.bicep` so it survives redeploys, the same place the sibling projects keep their allowed origin), and a monitoring bug fixed that the dashboard exposed: the first scheduled verdict after the promotion pooled predictions from v1 and v2 into one "champion" ROC-AUC of 0.62. Performance is now attributed to the champion's own predictions (`--champion-version`, read from the registry in `drift.yml`); the corrected run scores v2 at 0.9934 with 1,702 v1 predictions excluded, and the pooled verdict was replaced in the feed with a note.

✅ **Done 2026-09-04.** Container App live on revision 9 (image `<sha>-v2`, champion version 2, $0 idle); managed-endpoint demonstration documented with its four attempts and teardown proof; dashboard served from the same container at `/dashboard`, verified from the public URL with every `/api/*` route returning the production log and feeds, including a deployment record the deploy workflow wrote from CI; README walks data → training → serving → CI/CD → drift → retrain → dashboard with measured numbers throughout and one honestly unmeasured item. Six phases, six build-log posts.

---

## Common Pitfalls

- **Treating it as a modeling project.** The infrastructure is the point; a mediocre model in a great pipeline beats a great model in a notebook here.
- **Row-level train/test splits.** Split by engine unit or you'll leak and overstate performance.
- **Peeking at FD002/FD004.** They're your drift story. Touching them during training or evaluation contaminates the regime-replay demonstration.
- **Not logging predictions.** Without prediction logs you cannot do drift monitoring — wire this in at serving time.
- **A drift monitor you never validate.** Prove it fires on the regime replay before you claim it works.
- **The managed-endpoint bill.** Azure ML managed online endpoints do not scale to zero — they bill per instance-hour continuously. Demonstrate, capture evidence, tear down. Keep the persistent demo on Container Apps at min replicas 0, and set a budget alert on the resource group.
- **Long-lived cloud secrets in CI.** Use OIDC federated credentials; there is nothing to store, leak, or rotate.
- **Vague results.** Report concrete numbers: AUC, drift metrics that fired, deploy time, p95 latency, idle cost.
