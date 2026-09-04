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
