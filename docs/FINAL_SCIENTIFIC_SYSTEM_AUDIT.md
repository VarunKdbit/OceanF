# OceanEmbed Final Scientific and System Audit

Audit date: 2026-09-26
Scope: read-only review of the repository, checked-in data artifacts, and
non-training verification commands. No production code, model, data, or
configuration was changed for this audit.

## 1. AUDIT STATUS

**STATUS: WARNING**

**EVIDENCE:** The scientific contract and core V1 inference path are supported
by the active configuration, source code, actual harmonized files, and a
successful real-data raw inference check. Python compilation, the frontend
lint/build, backend unit tests, and ARGO unit tests passed. However, the
containerized backend has no live-status file mount/configuration, actual
service-to-service/PostgreSQL runtime integration was not exercised, and two
legacy smoke-test scripts fail after initialization/inference due to stale
attribute references. Deployment readiness is therefore not established.

## 2. SCIENTIFIC CONTRACT CHECK

**STATUS: PASS**

**EVIDENCE:** [`data/processed/ML/ml_config.json`](../data/processed/ML/ml_config.json)
declares the North Indian Ocean domain (5–30°N, 45–105°E), 0.25° target grid,
daily dates from 2025-07-01 through 2025-12-31, seven input features, `thetao`,
15 target depths, a seven-day history, and chronological train/validation/test
periods. [`scripts/oceanembed_model.py`](../scripts/oceanembed_model.py)
documents and implements 49 input channels, 128 latent channels, and 15
outputs. The held-out period is December 2025.

The supported statements are **seven-day retrospective surface-input window**
and **subsurface temperature reconstruction**, not a seven-day future
forecast. The ensemble spread is model spread across three seeds, not absolute
physical uncertainty.

**STATUS: WARNING**

**EVIDENCE:** [`scripts/ml_dataset.py`](../scripts/ml_dataset.py) builds each
sample's seven-day window ending on the target date. Consequently, the
validation window beginning 2025-11-01 includes 2025-10-26–2025-10-31, and the
test window beginning 2025-12-01 includes 2025-11-25–2025-11-30. Target-date
splits are chronological and disjoint, but input windows are not purged at
split boundaries. This is retrospective context rather than evidence of
held-out target labels entering model inputs; document this distinction when
describing split independence.

## 3. DATA PIPELINE CHECK

**STATUS: PASS**

**EVIDENCE:** All six files in
[`data/processed/ML/harmonized/`](../data/processed/ML/harmonized/) were
opened and checked during the audit:

- `SST_harmonized.nc`: `sst`, units `degree_Celsius`.
- `SSS_harmonized.nc`: `sss`, units `.001`.
- `SLA_harmonized.nc`: `sla` (plus source/harmonization auxiliaries).
- `Currents_harmonized.nc`: `uo`, `vo`, units `m/s`.
- `Winds_harmonized.nc`: `u_wind`, `v_wind` (plus `wind_speed`), units `m s-1`.
- `SubsurfaceTemp_harmonized.nc`: target `thetao`, units `degrees_C`.

The six datasets have aligned coordinates and daily time coordinates from
2025-07-01 through 2025-12-31 (184 days). Their latitude coordinates are
monotonic ascending from 5°N to 30°N at 0.25°; longitude runs from 45°E to
105°E at 0.25°. The `thetao` depth coordinate exactly matches
0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, and 1000 m.
The required surface feature names match the dataset loader and inference
feature mapping.

Missing values are present in surface and target datasets. The loader records
input validity masks, normalizes inputs and fills non-finite normalized input
values with zero; invalid target cells remain masked and are excluded from the
loss. Wind arrays were finite in the inspected harmonized file.

**STATUS: WARNING**

**EVIDENCE:** The active [`ml_config.json`](../data/processed/ML/ml_config.json)
declares train-only normalization (`training_period_mean_std` and
`training_period_mean_std_by_depth`), and V1 inference reads that configuration.
However, [`data/processed/ML/final_training/final_training_config.json`](../data/processed/ML/final_training/final_training_config.json)
describes a different July–November training configuration. It is not the
configuration used by the inspected V1 inference implementation; do not
describe it as provenance for the current V1 checkpoints.

The input validity mask is retained by the dataset but is not a separate model
input channel in the frozen 49-channel model. Missing input values therefore
enter as the normalized zero fill. No independent recomputation of every
stored normalization statistic from the raw training-only source period or
source-product provenance audit was performed.

**STATUS: PASS — ARGO separation**

**EVIDENCE:** The training dataset and training script consume the seven
harmonized surface inputs and `thetao` targets; the ARGO runner is a separate
validation script and is not referenced by the training or V1 inference
paths. No ARGO-derived data was found in the training data contract inspected
for this audit.

## 4. ML CHECK

**STATUS: PASS**

**EVIDENCE:** [`scripts/oceanembed_model.py`](../scripts/oceanembed_model.py)
defines `OceanEmbedCNN` with 49 input channels, 128 latent channels, and 15
output channels, mapping 64×64 inputs to 32×32 output tiles.
[`scripts/masked_huber_loss.py`](../scripts/masked_huber_loss.py) calculates
Huber loss only on valid target-mask cells. The data loader retains target
NaNs and masks invalid labels.

[`scripts/inference/oceanembed_inference.py`](../scripts/inference/oceanembed_inference.py)
loads seeds 42, 123, and 2024 from
[`data/processed/ML/checkpoints/`](../data/processed/ML/checkpoints/), builds
the matching architecture, loads each checkpoint, and uses the original
`ml_config.json` for V1 normalization. It computes the ensemble mean and the
population standard deviation across the three seed predictions, then
converts spread to temperature units using target standard deviations. Model
outputs and spread have the expected 15×32×32 shape; the inference API reports
temperature and spread in °C.

The raw inference smoke check loaded all three existing checkpoints and
completed real 2025-12-01 input inference with finite 15×32×32 predictions.
No training or retraining was run.

**STATUS: WARNING**

**EVIDENCE:** [`scripts/inference/test_raw_inference.py`](../scripts/inference/test_raw_inference.py)
prints that the V1 check used “final Jul-Nov statistics,” while the inspected
production inference source uses the original `ml_config.json` statistics.
Treat that test's normalization sentence as stale; its successful forward pass
does not establish use of final-training statistics.

Checkpoint files were present and loaded, but binary checkpoint provenance,
source-data snapshots used for training, and embedded training-run metadata
were not independently reproducible from the repository. This audit verifies
the configured paths and runtime load, not historical provenance.

## 5. LIVE INGESTION CHECK

**STATUS: PASS**

**EVIDENCE:** [`scripts/ingestion/run_daily_ingestion.py`](../scripts/ingestion/run_daily_ingestion.py)
reuses the downloader and harmonizer, validates candidate data and the
retrospective window, stages the processed NetCDF, and promotes the validated
candidate with `os.replace`. Its failure path retains the previous usable
date. Status JSON is also written via a temporary file and atomically replaced.

[`data/processed/live/live_status.json`](../data/processed/live/live_status.json)
reports `READY`, latest usable date **2026-09-20**, input window
**2026-09-14 through 2026-09-20**, and **7/7** required variables. Its
candidate date is 2026-09-21; no claim that this date is usable is made. The
published [`oceanembed_live_2026-09-20.nc`](../data/processed/live/2026-09-20/oceanembed_live_2026-09-20.nc)
was opened: it contains seven daily time steps, the 101×241 domain grid, and
all seven expected surface variables with finite observations.

**STATUS: WARNING**

**EVIDENCE:** Publication of the processed dataset and status file is atomic,
but downloader `--force` can delete a pre-existing raw source before fetching
its replacement, and the harmonizer writes directly to the requested output
path. These details do not invalidate the current published status, but the
whole ingestion operation is not transactional.

No ingestion run was started during this audit. The reported current status was
verified by reading the status JSON and opening the existing NetCDF, not by
re-fetching Copernicus sources or proving their upstream availability.

## 6. BACKEND CHECK

**STATUS: PASS**

**EVIDENCE:** The source request path is wired through FastAPI inference,
Spring orchestration/persistence, and response mapping:
[`ml-services/app/model.py`](../ml-services/app/model.py),
[`PredictionController.java`](../backend-api/src/main/java/com/oceanembed/backend/controller/PredictionController.java),
[`PredictionService.java`](../backend-api/src/main/java/com/oceanembed/backend/service/PredictionService.java),
and [`schema.sql`](../backend-api/src/main/resources/schema.sql). FastAPI
prefers a date-matched live file and falls back to historical harmonized
data; both loaders validate the domain, grid, required variables, and seven
retrospective days. The prediction response maps all requested depth
temperatures, seven final-day surface observations, ensemble spread, source,
and model version. Spring tests for response mapping and live availability
passed.

**STATUS: FAIL — containerized live-status path**

**EVIDENCE:** [`LiveAvailabilityService.java`](../backend-api/src/main/java/com/oceanembed/backend/service/LiveAvailabilityService.java)
reads `../data/processed/live/live_status.json` by default. The backend
container in [`docker-compose.yml`](../docker-compose.yml) has no volume mount
for the live data directory and does not set `LIVE_STATUS_FILE`; its
[`Dockerfile`](../backend-api/Dockerfile) runs from `/app`. Therefore the
container's default status-file path does not point to the repository's live
status file, and `/api/v1/live/status` will report the file missing in that
Compose deployment. The Spring unit test verifies service mapping with a test
file, not the container filesystem wiring.

**STATUS: WARNING — runtime and deployment integration**

**EVIDENCE:** Services were not started and no HTTP request/database
round-trip was run. `docker-compose.yml` and
[`application.yml`](../backend-api/src/main/resources/application.yml) contain
local default PostgreSQL credentials; Compose exposes PostgreSQL on the host.
Set strong deployment secrets and restrict database network exposure before
cloud deployment. The checked Spring request DTO allows geographic coordinate
ranges; the North Indian Ocean bounds are enforced downstream by the ML data
loader rather than by the DTO itself.

**STATUS: WARNING — technical API error details**

**EVIDENCE:** [`FastApiClient.java`](../backend-api/src/main/java/com/oceanembed/backend/service/FastApiClient.java)
copies FastAPI's raw error response body into `ModelServiceException`, and
[`GlobalExceptionHandler.java`](../backend-api/src/main/java/com/oceanembed/backend/exception/GlobalExceptionHandler.java)
returns that message in the REST error response. The current React client
converts non-success responses to generic user-facing messages and maps the
known unsupported-grid error to its friendly state, but other API consumers
can receive raw backend/validation details.

## 7. FRONTEND CHECK

**STATUS: PASS**

**EVIDENCE:** [`frontend/src/App.tsx`](../frontend/src/App.tsx) derives a
single validation state for incomplete/out-of-domain/unavailable-date cases,
uses the 5–30°N / 45–105°E domain, and disables reconstruction for coordinates
outside it or dates after the reported latest usable date. It displays live
availability and the corresponding input window, prediction profile, all 15
depth rows, and ensemble spread. [`frontend/src/api.ts`](../frontend/src/api.ts)
maps values from the API response; `SurfaceInputs.tsx` renders returned
surface observations, and the UI identifies the seven-day window as
retrospective. The frontend lint and production build passed.

**STATUS: WARNING**

**EVIDENCE:** The frontend's date picker is gated by live availability and
does not expose the historical fallback accepted by the backend. The rendered
UI includes prediction and spread views, but the repository search found no
frontend rendering of the ARGO RMSE/MAE/bias/Pearson validation metrics.
[`DifferenceMap.tsx`](../frontend/src/components/DifferenceMap.tsx) contains
sin/cos-generated synthetic difference values, but it is not imported or
referenced by the active `App.tsx` path during this audit. Do not present or
wire that unused component as real scientific diagnostics.

The build emits a large-chunk warning: the minified JavaScript bundle is
approximately 4.93 MB (1.50 MB gzip). This is not a build failure, but should
be considered for cloud delivery performance.

## 8. ARGO VALIDATION CHECK

**STATUS: PASS**

**EVIDENCE:** The generated
[`argo_validation_metrics.json`](../data/processed/ML/argo_validation/argo_validation_metrics.json),
[`argo_profile_catalog.csv`](../data/processed/ML/argo_validation/argo_profile_catalog.csv),
and [`argo_matchups.csv`](../data/processed/ML/argo_validation/argo_matchups.csv)
were cross-checked. They contain **352 candidates**, **76 used profiles**,
and **997 valid matchups** across all 15 requested depths. The exclusion counts
are 68 outside existing model output coverage, 207 without adjusted
measurements passing QC=1, and 1 with insufficient depth coverage.

Overall metrics are RMSE **1.532850 °C**, MAE **1.166499 °C**, bias
**−0.182070 °C**, and pooled Pearson correlation **0.978705**. The 0 m
metric has **one observation**. Depth-wise counts are present for all 15
depths; full statistics are in the metrics JSON.

[`scripts/validation/run_argo_validation.py`](../scripts/validation/run_argo_validation.py)
requires adjusted pressure/temperature and both corresponding QC flags equal
to `1`, matches the same calendar day, converts pressure with TEOS-10, and
interpolates only within observed depth bounds. Its four unit tests passed.
There is no “97.9% accuracy” claim in the checked README/project documentation;
the validation documentation explicitly says Pearson correlation is not an
accuracy percentage and cautions that pooled correlation combines depths.

## 9. PPT CLAIM CHECK

**STATUS: PASS**

**EVIDENCE:** A repository documentation search found no implemented-capability
claims for LangChain, RAG, MCP, n8n, a 7-day forecast head, future seven-day
forecasting, absolute uncertainty, “97.9% accuracy,” ARGO training, or
real-time global ocean prediction in the README/project documentation.
[`docs/ML_BACKEND_HANDOFF.md`](../docs/ML_BACKEND_HANDOFF.md) describes the
seven-day history as retrospective. The defensible project description is
Python-based daily ingestion, a seven-day retrospective surface-input window,
subsurface temperature reconstruction with OceanEmbed-CNN, three-seed
ensemble spread, and independent ARGO validation.

**STATUS: WARNING**

**EVIDENCE:** [`README-backend.md`](../README-backend.md) is stale legacy
documentation: it calls the ML model a placeholder, describes replacing a
synthetic profile, and documents an older input/output shape and partial depth
list. Its statements conflict with the current source implementation.
Do not use it as presentation or deployment documentation without correction.
The unused synthetic `DifferenceMap` component is also not evidence of real
prediction-minus-reference diagnostics.

## 10. TEST RESULTS

**STATUS: PASS**

**EVIDENCE:**

- Python bytecode compilation: **39 scripts passed**.
- ARGO validation unit tests: **4 passed**.
- `npm run lint` in `frontend/`: **passed**.
- `npm run build` in `frontend/`: **passed** (`tsc -b` and Vite production
  build); Vite reported the large JavaScript chunk warning noted above.
- `mvn test` in `backend-api/`: **4 tests passed, 0 failures, 0 errors**.
- `scripts/inference/test_raw_inference.py`: **passed** real-data
  retrospective-window extraction, normalization, three-checkpoint loading,
  finite 15×32×32 inference.
- `git diff --check`: **passed** for tracked changes at audit time.
- Live NetCDF inspection: **passed** for expected 2026-09-14–2026-09-20
  window, grid, seven variables, and finite data.

**STATUS: WARNING — legacy smoke checks**

**EVIDENCE:** `scripts/test_ml_dataset.py` initializes the dataset and reports
the expected [49,64,64] inputs and [15,32,32] targets, then fails because it
accesses nonexistent `dataset.num_lat`. `scripts/inference/test_real_inference.py`
loads the three checkpoints and completes finite [15,32,32] inference, then
fails when it accesses nonexistent `engine.depths_m`. These are stale smoke
test attribute references; they prevented those scripts from completing
their remaining assertions/printing. The separate raw inference script passed.

No model training was run. No live service, cloud deployment, PostgreSQL
round-trip, or live-status HTTP request was exercised.

## 11. ISSUES FOUND

**STATUS: FAIL**

**EVIDENCE:** Containerized Spring Boot cannot read the live status JSON using
the configured default path because Compose does not mount that file and does
not configure `LIVE_STATUS_FILE`. This breaks the deployed live-status
endpoint in the inspected Compose topology.

**STATUS: WARNING**

**EVIDENCE:**

1. Cloud Compose uses default database credentials and exposes PostgreSQL.
2. Two legacy smoke-test scripts fail on missing attributes despite successful
   setup/forward inference.
3. `test_raw_inference.py` prints a final-training normalization statement
   inconsistent with the V1 implementation's `ml_config.json` source.
4. `README-backend.md` describes the older placeholder model/contract.
5. An unused frontend component generates synthetic difference values.
6. No full runtime HTTP/database integration or cloud deployment check was
   performed.
7. Train/validation/test target dates are disjoint, while retrospective input
   windows include preceding dates from the prior split.
8. The frontend production bundle triggers Vite's >500 kB chunk warning.
9. REST error responses can expose raw FastAPI error-body text to non-React
   clients even though the current frontend sanitizes it.

## 12. REQUIRED FIXES

**STATUS: FAIL — required before cloud deployment**

**EVIDENCE:** Configure the backend container with a valid read-only mount and
an explicit `LIVE_STATUS_FILE` path for the same live status artifact, or
otherwise provide an equivalent status source. Verify `/api/v1/live/status`
inside the container returns the actual latest usable date and window.

**STATUS: WARNING — required before cloud deployment**

**EVIDENCE:** Replace local default PostgreSQL credentials with managed
secrets, restrict database network access, and run a deployment-level
health/prediction/persistence test against the intended database, ML service,
model files, and live-data mounts. This audit did not verify the cloud
environment or secrets configuration.

**STATUS: WARNING — recommended API hardening**

**EVIDENCE:** Normalize/sanitize FastAPI error responses at the Spring
boundary before returning them from the public REST API; preserve diagnostic
details in server logs rather than returning raw service bodies. The current
frontend sanitization does not protect other API clients.

**STATUS: WARNING — required for reliable audit/reporting**

**EVIDENCE:** Correct the stale dataset/inference smoke-test attributes and
the inaccurate normalization message; update or retire `README-backend.md`;
ensure the unused synthetic difference component cannot be mistaken for
scientific output. None of these files were changed as part of this audit.

## 13. ITEMS VERIFIED AND READY

**STATUS: PASS**

**EVIDENCE:** The 0.25° domain grid, six harmonized datasets, daily alignment,
target depths, target masks, masked Huber loss, three-seed V1 checkpoint load,
V1 configuration-based inference normalization, real 15-depth inference,
current seven-day live status, ARGO QC/date/distance/depth-selection rules,
reported ARGO counts/metrics, Spring response mapping tests, frontend
validation/display source, and requested Python/backend/frontend checks were
verified as described above.

PPT-safe language: **OceanEmbed reconstructs subsurface temperature from
surface observations using a seven-day retrospective input window and an
OceanEmbed-CNN three-seed ensemble; the ensemble spread is not absolute
physical uncertainty; independent ARGO validation used adjusted QC=1,
same-day profiles within 0.5° spatial separation.** State the number and
depth-wise sparsity of ARGO matchups and do not call Pearson correlation
“accuracy.”

## 14. FINAL READINESS SUMMARY

**STATUS: WARNING**

**EVIDENCE:** The scientific/model/data path and reported ARGO validation
support accurate final PPT preparation if claims remain within the stated
scope and caveats. The project is **not ready to certify for cloud deployment**
from this audit: the Compose live-status path is not wired, default database
credentials/network exposure require deployment hardening, and live
service-to-service/database integration was not tested. The current live
dataset/status is dated through 2026-09-20; 2026-09-21 is only a candidate,
not a usable date.

This is an audit report, not a code or data remediation. No commit or push was
made.
