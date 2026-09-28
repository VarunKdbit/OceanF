# OceanEmbed Target-Date End-to-End Validation

Validation date: 2026-09-26
Scope: exact requested-date retrospective-window resolution, real inference,
Spring Boot/PostgreSQL persistence, historical fallback, and regression checks.
No model architecture, checkpoints, normalization statistics, training,
inference mathematics, or ARGO validation results were changed. No commit or
push was made.

## 1. Services tested

**STATUS: PASS — local FastAPI and Spring Boot**

**EVIDENCE:** FastAPI started from this workspace on port 8001. `GET /health`
returned `status=ok`, `model_loaded=true`, and
`model_version=oceanembed-v1.0`. Spring Boot started from this workspace on
port 8081; `GET /api/v1/health` returned HTTP 200 and
`ml_service_reachable=true`. The Spring service connected to the local
PostgreSQL service.

**STATUS: WARNING — Docker**

**EVIDENCE:** The Docker CLI is unavailable. Docker Compose parsing, image
builds, and container-level validation were not performed.

## 2. Live status and target-date windows

**STATUS: PASS**

**EVIDENCE:** Spring `GET /api/v1/live/status` reported:

- `latestUsableDate`: **2026-09-21**
- input window: **2026-09-15 through 2026-09-21**
- `variablesReady`: **7**
- `requiredVariables`: **7**
- `ready`: **true**

Validated daily cache files cover **2026-09-13 through 2026-09-21**. The
selected target date resolves to exactly `D-6…D`; no later date is substituted.

Coverage results from the live ML service and Spring proxy:

- **15.0°N, 70.0°E, 2026-09-19:** ready, **7/7** variables, window
  **2026-09-13 through 2026-09-19**.
- **15.0°N, 70.0°E, 2026-09-20:** ready, **7/7**, window
  **2026-09-14 through 2026-09-20**.
- **20.0°N, 50.0°E, 2026-09-20:** not ready, **2/7**; ready variables
  `u_wind`, `v_wind`; missing `sst`, `sss`, `sla`, `uo`, `vo`.
- **15.0°N, 70.0°E, 2026-09-21:** ready, **7/7**, window
  **2026-09-15 through 2026-09-21**.

The request for 2026-09-21 found its one missing day, fetched and harmonized it
through the existing Copernicus ingestion path, validated it, and cached it.
Consequently, the older “DATA NOT READY” expectation for that date no longer
describes the current data state.

## 3. FastAPI inference

**STATUS: PASS**

**EVIDENCE:** A real `POST /predict` for **15.0°N, 70.0°E,
2026-09-19** returned model version `oceanembed-v1.0`, the exact 15 depths
`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000`,
seven surface observations, finite temperatures, and finite non-negative
ensemble spreads. Its input window was **2026-09-13 through 2026-09-19**.

A real `POST /predict` for **2026-09-21** also succeeded with 15 predictions,
seven surface observations, and the exact **2026-09-15 through 2026-09-21**
input window.

**STATUS: PASS — incomplete-data guard**

**EVIDENCE:** A direct FastAPI prediction for **20.0°N, 50.0°E,
2026-09-20** returned HTTP **422**. The unit test
`test_prediction_rejects_incomplete_coverage_before_model_inference` verifies
that the inference engine is not invoked when a required final-day observation
is missing.

## 4. Spring Boot end-to-end

**STATUS: PASS**

**EVIDENCE:** Spring coverage matched the ML service for the valid 2026-09-19
point and the incomplete 2026-09-20 point. A Spring prediction request for
**15.0°N, 70.0°E, 2026-09-19** created job **50**, returned `SUCCESS`,
`oceanembed-v1.0`, 15 predictions at all configured depths, seven surface
observations, ensemble spread for each prediction, and the correct
**2026-09-13 through 2026-09-19** window. `GET /api/v1/predictions/50`
retrieved the persisted successful result.

**STATUS: PASS — incomplete request is rejected before job creation**

**EVIDENCE:** A direct Spring prediction request for **20.0°N, 50.0°E,
2026-09-20** returned HTTP **400**. Looking up the next job ID (**51**)
returned HTTP **404**; no job was created for the rejected request.

## 5. PostgreSQL

**STATUS: PASS**

**EVIDENCE:** Spring health reported the ML service reachable and the Spring
application started with its PostgreSQL connection. The successful job 50 and
its predictions/observations were retrieved through the Spring API. The
incomplete request created no job 51.

## 6. Historical fallback

**STATUS: PASS**

**EVIDENCE:** FastAPI coverage for **15.0°N, 70.0°E, 2025-12-31** resolved
the existing historical window **2025-12-25 through 2025-12-31** and reported
7/7 variables. This confirms the existing historical source-selection path
continues to serve a date in the configured historical period.

## 7. Frontend

**STATUS: PASS — lint, build, and dev server response**

**EVIDENCE:** `npm run lint` and `npm run build` passed. The Vite development
server started from this workspace and returned HTTP 200 for its entry page.
The current UI source gates reconstruction on a successful selected-date
coverage result, renders an incomplete-data state, and disables the
reconstruction button unless validation state is `VALID`.

**STATUS: WARNING — interactive browser verification**

**EVIDENCE:** The integrated browser could not attach to its browser process
(CDP connection timed out), so field interaction and rendered UI states for
the newly resolved dates were not independently exercised in a browser during
this run. The frontend build and served entry page were verified.

## 8. Negative and unavailable-window cases

**STATUS: PASS — incomplete live observations**

**EVIDENCE:** The valid point reports 7/7; the known incomplete point reports
2/7. FastAPI returns 422, Spring returns 400, and the Spring request creates
no job. Unit coverage tests verify missing requested dates are reported
instead of silently substituting a different window.

**STATUS: WARNING — outside-domain UI and currently unavailable future date**

**EVIDENCE:** Interactive browser checks for outside-domain fields could not
be repeated because browser attachment failed. The frontend source retains
the supported-domain guard before coverage checks. The former 2026-09-21
not-ready case is now a valid ready date after the actual daily data was
successfully prepared; no additional future-date download was attempted merely
to manufacture an unavailable-data case.

## 9. Regression test results

**STATUS: PASS**

**EVIDENCE:**

- `ml-services/tests/test_live_data_resolution.py`: **4 tests passed**.
- `ml-services/tests/test_surface_coverage.py`: **4 tests passed**.
- `python -m compileall -q ml-services\app scripts\ingestion`: passed.
- `mvn test` in `backend-api/`, with
  `JAVA_TOOL_OPTIONS=-Dnet.bytebuddy.experimental=true`: **5 tests passed**,
  0 failures, 0 errors.
- `npm run lint` in `frontend/`: passed.
- `npm run build` in `frontend/`: passed.
- `git diff --check`: passed.
- No training or retraining was run.

## 10. Warnings and failures

**STATUS: WARNING**

**EVIDENCE:**

1. Docker/container validation was unavailable.
2. Interactive browser verification was unavailable because the integrated
   browser connection timed out.
3. Vite reports the main JavaScript bundle at approximately **4.93 MB**,
   above its advisory threshold; the build succeeds.
4. The live-window unit run emitted NumPy/netCDF compatibility deprecation
   warnings, but all four tests passed.

**STATUS: PASS — no functional failures in the exercised inference path**

**EVIDENCE:** Target-date windows for September 19, 20, and 21 resolved to
their exact requested dates; real inference succeeded for valid September 19
and 21 requests; the valid Spring request persisted successfully; and
incomplete coverage was rejected before Spring created a job.

## 11. Final E2E readiness

**FINAL E2E STATUS: READY**

**EVIDENCE:** The valid **15.0°N / 70.0°E / 2026-09-19** request completed
through FastAPI, Spring Boot, and PostgreSQL with all 15 depths, seven surface
observations, and ensemble spread. The **20.0°N / 50.0°E / 2026-09-20**
request is rejected by FastAPI and Spring before successful prediction
persistence. The exact target-date window for **2026-09-21** was prepared and
used for a successful real inference.

Local API and database readiness is **READY** for the tested cases. Browser
interaction and Docker/cloud-container readiness are not certified by this
run.

## Relevant implementation and validation files

- `scripts/ingestion/live_window.py`
- `scripts/ingestion/fetch_daily_inputs.py`
- `scripts/ingestion/harmonize_live_data.py`
- `scripts/ingestion/run_daily_ingestion.py`
- `ml-services/app/data.py`
- `ml-services/app/model.py`
- `ml-services/app/schemas.py`
- `ml-services/tests/test_live_data_resolution.py`
- `ml-services/tests/test_surface_coverage.py`
- `ml-services/requirements-ingestion.txt`
- `ml-services/Dockerfile`
- `backend-api/src/main/java/com/oceanembed/backend/Service/FastApiClient.java`
- `backend-api/src/main/java/com/oceanembed/backend/Service/PredictionService.java`
- `backend-api/src/main/java/com/oceanembed/backend/controller/PredictionController.java`
- `backend-api/src/main/java/com/oceanembed/backend/dto/SurfaceCoverageDTO.java`
- `backend-api/src/main/resources/application.yml`
- `backend-api/src/test/java/com/oceanembed/backend/service/PredictionCoverageGuardTest.java`
- `frontend/src/App.tsx`
- `frontend/src/api.ts`
- `docker-compose.yml`
- `docs/LIVE_DATE_RESOLUTION.md`
- `docs/FINAL_E2E_VALIDATION.md`

No commit or push was made.
