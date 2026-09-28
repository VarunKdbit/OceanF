# Independent ARGO Observational Validation

## Scientific independence

ARGO is an independent observational validation source. It is not used for
OceanEmbed training, normalization, model selection, hyperparameter or
threshold tuning, or generation of GLORYS targets. The primary comparison is
the existing OceanEmbed prediction against adjusted ARGO temperature; GLORYS
is not part of these ARGO metrics.

The validation runner uses the existing OceanEmbed V1 three-seed inference
engine (`scripts/inference/oceanembed_inference.py`), existing seeds
42/123/2024, its existing checkpoints, and the original train-only statistics
in `data/processed/ML/ml_config.json`. It does not implement a second model.

## Fixed validation contract

- Held-out period: **2025-12-01 through 2025-12-31**, inclusive.
- Domain: **5–30°N, 45–105°E**, 0.25° grid.
- Temporal matching: same calendar day only; no ±1-day tolerance.
- Spatial matching: nearest existing OceanEmbed grid coordinate, with a
  maximum great-circle separation of **0.5 degrees** (about 55.6 km).
- Input: the existing seven-day retrospective window of SST, SSS, SLA, U/V
  current, and U/V wind.
- Output depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500,
  700, and 1000 m.

The model only emits its existing centered 32×32 prediction region from each
64×64 tile. Candidate points outside the output coverage provided by the
existing tiling are explicitly listed as excluded; the runner does not alter
tile geometry or infer a point from an adjacent location.

## Official data and profile selection

The runner reads the official Argo GDAC global profile index
(`ar_index_global_prof.txt.gz`) and downloads only profile NetCDF files
selected for the requested period and domain. The default source is the
Euro-Argo GDAC:

- GDAC index and profile data: <https://data-argo.ifremer.fr/>
- Official GDAC documentation: <https://argo.ucsd.edu/data/data-from-gdacs/>

The GDAC base and index URL can be changed for another official HTTP/HTTPS
GDAC mirror. The complete global profile archive is never downloaded.
Temporary index and selected profile files are stored beneath
`data/raw/argo_validation/`, which is already covered by the repository's
`data/raw/` ignore rule.

Delayed-mode profiles are processed first when available. The strict primary
result requires `TEMP_ADJUSTED`, `PRES_ADJUSTED`, `TEMP_ADJUSTED_QC`, and
`PRES_ADJUSTED_QC`; both adjusted QC flags must equal `'1'`. Unadjusted
temperature or pressure is never substituted. Missing adjusted fields, poor
QC, insufficient vertical coverage, download failures, and spatial/output
coverage exclusions are recorded in `argo_profile_catalog.csv`.

Pressure is converted from dbar to positive-down geometric depth using
TEOS-10 `gsw.z_from_p()` at the profile latitude. Valid samples are sorted by
depth, duplicate depths are averaged, and linear interpolation is performed
only at target depths bracketed by the observed profile. The runner does not
extrapolate.

## Outputs

Running the command writes:

- `data/processed/ML/argo_validation/argo_matchups.csv`
- `data/processed/ML/argo_validation/argo_profile_catalog.csv`
- `data/processed/ML/argo_validation/argo_validation_metrics.json`
- `data/processed/ML/argo_validation/figures/argo_vs_oceanembed_profiles.png`
- `data/processed/ML/argo_validation/figures/argo_oceanembed_error_by_depth.png`
- `data/processed/ML/argo_validation/figures/argo_matchup_counts_by_depth.png`

The JSON contains the actual considered/used profile and matchup counts,
exclusion counts, per-depth and pooled RMSE, MAE, bias, and Pearson
correlation. Pearson correlation is not an accuracy percentage; the pooled
value also combines different depths and can reflect the vertical temperature
gradient. It is null when mathematically undefined. Per-depth counts include
only QC-valid ARGO observations with an available same-day OceanEmbed estimate
at that depth.

The profile catalog records each candidate profile, its float and date,
location, data mode, adjusted-variable availability, valid and matched depth
counts, and exclusion reason. The matchups CSV records the observed and
predicted temperature, ensemble spread, coordinate match, and signed
prediction-minus-observation error for each valid profile/depth pair.

## Reproduction

Use the existing Python environment that contains the repository's PyTorch,
NumPy, xarray, pandas, and netCDF4 dependencies, then install the two
validation-only packages:

```powershell
python -m pip install -r scripts/validation/requirements-argo-validation.txt
python scripts/validation/run_argo_validation.py
```

To check index discovery without downloading individual profiles or producing
metrics:

```powershell
python scripts/validation/run_argo_validation.py --discover-only
```

The defaults are intentionally fixed to the December held-out test period,
test split, a 0.5-degree maximum spatial separation, and automatic CPU/CUDA
selection. A smaller date subrange can be selected within December for
diagnostics, but the script rejects dates outside the held-out period and
does not permit increasing the spatial threshold.

## Results and limitations

The output metrics and profile counts are generated only from downloaded
official profiles and actual OceanEmbed predictions. The 2026-09-26 run found
352 same-day profile candidates in the fixed 2025-12-01 through 2025-12-31
period; 76 profiles produced 997 valid profile/depth matchups. Exclusions
comprised 68 candidates outside existing model-output tile coverage, 207
without adjusted measurements passing QC=1, and one without sufficient valid
depth coverage. These are exclusions, not substituted or fabricated
observations.

Across all valid matchups, RMSE was **1.533 °C**, MAE **1.166 °C**, and bias
**−0.182 °C** (prediction minus observation). The pooled Pearson correlation
was 0.979, with the depth-gradient caveat above. Per-depth RMSE ranged from
0.465 °C at 0 m (one observation) to 2.457 °C at 75 m (75 observations);
most other target depths had 67–75 observations. The metrics JSON contains
all depth-level scores and counts. These results describe only the selected
same-day, QC-passing ARGO/model matchups and do not claim full-domain coverage.

ARGO sampling is spatially and temporally irregular, adjusted-mode and QC=1
availability varies, and the existing centered model output tiles do not
cover every point in the full declared domain. These restrictions can leave
some depths or dates with few/no matchups. The pipeline reports that outcome
rather than relaxing the selection criteria.
