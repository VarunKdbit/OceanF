# Target-Date Live Input Resolution

OceanEmbed resolves the user-selected target date by constructing the
required seven-day retrospective input window ending on that date.
Available daily observations are reused from cache or fetched through
the automated ingestion pipeline. Reconstruction is enabled only after
the complete required window passes validation.

## Resolution and cache

For target date `D`, the resolver checks exactly `D-6` through `D`. It first
uses validated files under `data/processed/live/daily/`. Existing published
seven-day live files are migrated to daily cache entries only after the whole
source window validates. Existing combined files remain as backward-compatible
inputs; new target requests do not publish additional seven-day NetCDF files.

Missing contiguous date ranges are fetched using the existing Copernicus
request builder, quality/harmonization functions, and source products. Raw
downloads are cached under `data/raw/live/ranges/`. Harmonized data is staged,
validated against the expected dates, variables, grid and units, then its
requested days are atomically published to the daily cache. A repeated
request reuses validated daily files and does not download them again.

No later date is substituted for a missing observation. If Copernicus does
not provide a required day, preparation returns not-ready with the missing
dates and inference is not run.

## API behavior

`GET /coverage` and Spring Boot's
`GET /api/v1/predictions/coverage` resolve the requested date, then check
finite values for all seven surface variables at the model's snapped location
on the final day. Their response includes the target date, requested window,
available/missing dates and variables, readiness, and snapped coordinates
when the window can be loaded.

Prediction uses the same resolver and rejects an incomplete window before
inference or job creation. Historical dates continue to use the existing
historical dataset fallback when the requested date is present there.

The frontend requests coverage for the selected location and date. It shows
the requested input window and enables reconstruction only after the API
confirms readiness. An unavailable window is shown as `DATA NOT READY` with
missing date information where known.

## Container configuration

The ML service needs writable mounts for `data/processed/live` and
`data/raw/live` so validated daily data and source downloads persist across
container restarts. The service image installs the existing
`copernicusmarine` CLI client dependency used by the ingestion scripts.
For Compose, the CLI configuration is persisted in the untracked
`copernicus-config` named volume; authenticate the CLI in the ML service
environment before requesting dates that are not already cached.
