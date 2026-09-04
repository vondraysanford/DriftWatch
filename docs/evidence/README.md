# Evidence

Screenshots and recordings captured the moment each checkpoint was verified, one folder per
phase. Every entry says what it proves and when it was taken, so the build-log posts and the
README's numbers can point at something a reader can look at rather than take on trust.

Conventions: `phase-N/<what-it-shows>.png` (or `.gif` for recordings), kebab-case, no spaces.
Recordings are screen captures converted to GIF with ffmpeg. Nothing here is required by the
pipeline; the folder is excluded from the Docker build context and from deploy triggers.

## Phase 4 — CI/CD

| File | What it proves | Taken |
|---|---|---|
| `phase-4/managed-endpoint-demo-run-4-green.png` | The manual `managed-endpoint-demo` workflow, run #4, green end to end in 18m 29s: endpoint created (1m 6s), custom environment built and deployment created (9m 25s), five invocations (15s), logs captured, teardown, and "Confirm nothing is left billing" passing after 6m 47s. | 2026-09-03 |
| `phase-4/managed-endpoint-demo-run-4-summary-table.png` | The same run's summary: registered model version 1 on one `Standard_DS2_v2` at threshold 0.4068; five invocations all correct (near-failure window 1.0000 / label 1, healthy window 0.0352 / label 0, identical to the Container App); round trips 3335, 2890, 2923, 2978, 2951 ms measured around the `az ml online-endpoint invoke` CLI call (mostly CLI start-up and token acquisition, not inference); "Endpoint torn down and confirmed deleted." | 2026-09-03 |

## Phase 6 — Dashboard

| File | What it proves | Taken |
|---|---|---|
| `phase-6/dashboard-light.png` | The dashboard in the vondraysanford.com "engineering paper" theme (drafting grid, DM Serif Display / DM Sans / JetBrains Mono, the site's ink and signal-blue tokens), rendered headless from a local instance reading the production prediction log and monitoring feeds: champion v2 title block, 2,593 predictions from 44 engines, predictions per hour by regime, the probability distribution against the threshold, drift share per regime across the detector's runs, the champion's ROC-AUC on labeled traffic, and the champion-vs-challenger, deployments, and recent-prediction tables. The signal blue was validated for color-vision-deficiency separation against the orange on the panel white. | 2026-09-04 |
| `phase-6/dashboard-dark.png` | The same page in dark mode: same tokens stepped for the dark surface, with `#5B7BE8` as the signal blue because the site's `#2244CC` fails the lightness and 3:1 contrast checks on a dark panel (validated). | 2026-09-04 |
| `phase-6/dashboard-live-pages.png` | The dashboard at its own subdomain, `driftwatch.vondraysanford.com`, hosted on Cloudflare Pages like the sibling projects and rendered headless from the public URL. The static bundle carries the Container App's URL as `VITE_API_BASE_URL`; the browser's cross-origin calls succeed because the Container App's `CORS_ALLOW_ORIGINS` variable (set in the Bicep template, revision 10) names that origin and nothing else. | 2026-09-04 |
| `phase-6/dashboard-live-azure.png` | The dashboard served by the Azure Container App itself (revision 9, image `<sha>-v2`), rendered headless from the public URL after the first `deploy.yml` run that built the dashboard on the runner. The third row of its deployments table is the record that run wrote to blob storage from CI. | 2026-09-04 |

## Phase 5 — Drift + retrain loop

| File | What it proves | Taken |
|---|---|---|
| `phase-5/deploy-run-ci-environment-green.png` | `deploy.yml` after the promotion-gating change, run in the `ci` environment on an ordinary push: green in 4m 46s (test 32s, build/push/deploy 4m 6s, Docker layer cache 20%), model version 1 deployed, "Promotion: none (deployed the current champion)". Ordinary pushes are not gated; only promotions are. | 2026-09-04 |
| `phase-5/drift-run-1-green.png` | The first `drift.yml` run (manual dispatch, 24 h window): verdict DRIFT on 1,701 predictions from 44 engines. FD001 traffic (812 records) vs the FD001 reference: 0 of 17 raw columns, 0 of 99 features, no drift. FD002 traffic (889) vs the same reference: 17 of 17 raw, 98 of 99 features, all three operating settings, six FD001-constant sensors now varying, DRIFT. Champion ROC-AUC 0.9921 on FD001 traffic and 0.5007 on FD002 traffic (0.6140 over all 1,701 labeled records). This run's dispatch started `retrain.yml`. | 2026-09-04 |
| `phase-5/deploy-run-9-production-waiting-for-review.png` | The promotion gate. `deploy` run #9, triggered by `repository_dispatch` (`model-registered`) from the retrain, with `test` green and `build, push, deploy` paused: "production waiting for review", a "Review deployments" button, and the deployment protection rule listing the requested reviewer. Nothing has changed in production at this moment; the challenger is registered but not serving. | 2026-09-04 |
| `phase-5/deploy-run-9-approved-promoting-to-champion.png` | The same run after approval: "The deployments have been approved", the job "Deploying to production", and the protection-rule row showing who approved, when, and the comment "Promoting to champion". The human decision is recorded in the run itself. | 2026-09-04 |
| `phase-5/deploy-run-9-green-while-version-1-kept-serving.png` | A green run that was not a successful deploy. Run #9 finished in 6m 51s reporting "Model version: 2, Promotion: 2", but the live endpoint kept answering version 1: the image tag was the commit SHA, identical to the previous push, so Container Apps saw no template change and kept revision 7 running. The smoke test passed because it checked only the label. Kept as evidence of the bug; the fix tags images with commit plus model version and makes the smoke test assert the served version, and the corrected deploy is the next entry. | 2026-09-04 |
