OceanEmbed - ML to Backend Handoff
==================================

1. PURPOSE
----------

This document defines the interface between the OceanEmbed ML pipeline
and the backend/frontend integration layer.

The ML pipeline is responsible for:

1. Preparing the required 7-day surface-observation input.
2. Applying V1 preprocessing and normalization.
3. Running the trained OceanEmbed-CNN models.
4. Combining the three model predictions into a 3-seed ensemble.
5. Returning subsurface ocean temperature at 15 target depths.

The backend is responsible for exposing this inference capability through
an application/API and passing the resulting prediction to the frontend.


2. PROJECT CONTRACT
-------------------

Project:
OceanEmbed

Experiment:
OceanEmbed-CNN

Primary experiment:
E2 - 7-day retrospective input

Domain:
5N to 30N latitude
45E to 105E longitude

Spatial resolution:
0.25 degree x 0.25 degree

Temporal resolution:
Daily


3. ML INPUT CONTRACT
--------------------

The model requires seven surface-observation variables:

1. SST     - Sea Surface Temperature
2. SSS     - Sea Surface Salinity
3. SLA     - Sea Level Anomaly
4. UO      - Surface Ocean Current U component
5. VO      - Surface Ocean Current V component
6. U Wind  - Surface Wind U component
7. V Wind  - Surface Wind V component

The model uses a 7-day retrospective window.

Therefore:

7 features x 7 days = 49 input channels.


4. RAW INPUT INTERFACE
----------------------

The production inference engine accepts a dictionary containing
seven feature arrays.

Required keys:

    sst
    sss
    sla
    uo
    vo
    u_wind
    v_wind

Each feature must have shape:

    [7, 64, 64]

Expected logical structure:

    feature_arrays = {
        "sst":    [7,64,64],
        "sss":    [7,64,64],
        "sla":    [7,64,64],
        "uo":     [7,64,64],
        "vo":     [7,64,64],
        "u_wind": [7,64,64],
        "v_wind": [7,64,64]
    }

The seven days must be ordered chronologically from oldest
to newest, ending on the requested prediction date.


5. NORMALIZED MODEL INPUT
-------------------------

The production inference engine converts the raw feature dictionary
into the normalized model tensor:

    [49, 64, 64]

Feature-major channel ordering is fixed:

    SST day1 ... day7
    SSS day1 ... day7
    SLA day1 ... day7
    UO day1 ... day7
    VO day1 ... day7
    U Wind day1 ... day7
    V Wind day1 ... day7

The backend must preserve this ordering if it performs any preprocessing
outside the production inference engine.


6. V1 NORMALIZATION
-------------------

IMPORTANT:

The currently trained and validated checkpoints are V1.

V1 normalization statistics are loaded from:

    data/processed/ML/ml_config.json

V1 input normalization uses training-period statistics from:

    2025-07-01 to 2025-10-31

Validation period:

    2025-11-01 to 2025-11-30

Test period:

    2025-12-01 to 2025-12-31

The production inference engine already loads these V1 statistics.

The backend should NOT replace them with another statistics file.

IMPORTANT:

    data/processed/ML/final_training/final_training_statistics.json

is NOT used by the current V1 checkpoints.

That file belongs to the prepared future final-training/V2 workflow.


7. MODEL
--------

Primary model:

    OceanEmbed-CNN

Input:

    49 channels

Input spatial size:

    64 x 64

Latent representation:

    128 channels

Output:

    15 channels

Output spatial size:

    32 x 32

Trainable parameters:

    989,967


8. MODEL CHECKPOINTS
--------------------

The current V1 ensemble contains exactly three trained checkpoints:

    oceanembed_e2_seed42.pt
    oceanembed_e2_seed123.pt
    oceanembed_e2_seed2024.pt

Seeds:

    42
    123
    2024

The production inference engine loads all three checkpoints.

The ensemble prediction is the arithmetic mean of the three
model predictions.

The three-seed ensemble spread is a diagnostic only and must NOT
be represented as calibrated physical uncertainty.


9. PRODUCTION INFERENCE ENGINE
------------------------------

Production inference implementation:

    scripts/inference/oceanembed_inference.py

Main class:

    OceanEmbedEnsemble

Constructor:

    OceanEmbedEnsemble(device="cuda")

Primary method:

    predict_from_raw_window(feature_arrays)

Input:

    Dictionary containing seven features.
    Each feature has shape [7,64,64].

Output:

    [15,32,32]

Units:

    degrees Celsius

The production method performs:

    raw feature arrays
            |
            v
    V1 normalization
            |
            v
    three model predictions
            |
            v
    three-seed ensemble
            |
            v
    temperature prediction


10. OUTPUT CONTRACT
-------------------

The model predicts subsurface potential temperature at:

    0 m
    5 m
    10 m
    20 m
    30 m
    50 m
    75 m
    100 m
    125 m
    150 m
    200 m
    300 m
    500 m
    700 m
    1000 m

Output tensor:

    [15,32,32]

Dimension order:

    [depth, latitude, longitude]

Units:

    degrees Celsius


11. SPATIAL TILE CONTRACT
-------------------------

Input tile:

    64 x 64

Output tile:

    32 x 32

Tile stride:

    32 pixels

The output region is centered inside the input tile.

Therefore:

    64 x 64 input
          |
          +-- 16 pixels context
          |
       32 x 32 target
          |
          +-- 16 pixels context

The backend must preserve this geometry when assembling tiles
or displaying prediction maps.


12. HARMONIZED GRID
-------------------

All ML datasets are harmonized to:

    Latitude: 5N to 30N
    Longitude: 45E to 105E
    Resolution: 0.25 degree

Spatial harmonization method:

    Linear interpolation

No extrapolation is used.


13. HARMONIZED DATASETS
-----------------------

The ML pipeline uses:

    SST_harmonized.nc
    SSS_harmonized.nc
    SLA_harmonized.nc
    Currents_harmonized.nc
    Winds_harmonized.nc
    SubsurfaceTemp_harmonized.nc

Location:

    data/processed/ML/harmonized/

For production prediction, only the seven surface variables
are required.

The SubsurfaceTemp dataset is required for training/evaluation,
not for normal production prediction.


14. MISSING INPUT DATA
----------------------

The ML pipeline retains input validity information.

Missing input values are handled during preprocessing and are
filled with zero after normalization.

The corresponding validity information is retained.

The backend must NOT invent physical observations for missing data.


15. EXAMPLE PRODUCTION PREDICTION
---------------------------------

Validated example:

    Prediction date:
    2025-12-01

    Input window:
    2025-11-25 to 2025-12-01

    Tile:
    row 0
    column 0

Input:

    7 features
    7 days
    64 x 64 spatial tile

Normalized model input:

    [49,64,64]

Output:

    [15,32,32]

Temperature units:

    degrees Celsius

The validated prediction was fully finite.

Example prediction range:

    6.855 C to 28.691 C

This range belongs only to this validation example and should
not be treated as a universal expected temperature range.


16. BACKEND INTEGRATION FLOW
----------------------------

Recommended backend flow:

    User selects date/location
            |
            v
    Backend determines required tile
            |
            v
    Backend obtains previous 7 days
            |
            v
    Extract seven required variables
            |
            v
    Create feature dictionary
            |
            v
    OceanEmbedEnsemble.predict_from_raw_window()
            |
            v
    [15,32,32] temperature prediction
            |
            v
    Convert prediction into API response
            |
            v
    Frontend visualization


17. RECOMMENDED API RESPONSE
----------------------------

The backend can expose a response conceptually similar to:

    {
      "project": "OceanEmbed",
      "prediction_date": "2025-12-01",
      "tile": {
        "row": 0,
        "column": 0
      },
      "depths_m": [
        0,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        200,
        300,
        500,
        700,
        1000
      ],
      "temperature_units": "degC",
      "shape": [
        15,
        32,
        32
      ],
      "ensemble": {
        "type": "three_seed_mean",
        "seeds": [
          42,
          123,
          2024
        ]
      }
    }

The exact API response can be adapted to the backend architecture.


18. BACKEND-RELEVANT FILES
--------------------------

Production inference engine:

    scripts/inference/oceanembed_inference.py

Prediction export/example:

    scripts/inference/export_prediction.py

Real raw-data inference test:

    scripts/inference/test_raw_inference.py

Additional real-data test:

    scripts/inference/test_real_inference.py


19. VALIDATION STATUS
---------------------

The following checks have passed:

    Model architecture validation
    CUDA inference
    Training step validation
    Seed 42 training
    Seed 123 training
    Seed 2024 training
    Three-seed ensemble evaluation
    Diagnostic evaluation
    Real 7-day raw input extraction
    Production raw-data inference
    Prediction NetCDF export
    Prediction metadata validation
    V1 normalization validation

The exported prediction was successfully reopened from NetCDF
and validated.

The exported metadata was also validated against the V1 contract.


20. V1 AND FUTURE V2 DISTINCTION
--------------------------------

CURRENT PRODUCTION VERSION:

    V1

Current checkpoints:

    oceanembed_e2_seed42.pt
    oceanembed_e2_seed123.pt
    oceanembed_e2_seed2024.pt

Current normalization:

    data/processed/ML/ml_config.json

Current normalization training period:

    2025-07-01 to 2025-10-31

Current validation period:

    2025-11-01 to 2025-11-30

Current test period:

    2025-12-01 to 2025-12-31


FUTURE OPTIONAL FINAL TRAINING:

    data/processed/ML/final_training/

Files:

    final_training_config.json
    final_training_statistics.json

The future final-training statistics extend the training
statistics through:

    2025-11-30

These files are preparation artifacts for a future optional
retraining workflow.

They must NOT be mixed with the current V1 checkpoints.

The current backend integration should use V1.


21. CURRENT ML HANDOFF STATUS
-----------------------------

ML development and validation are complete for the current
V1 prototype.

The ML pipeline is ready for backend integration.

The backend team can now begin:

1. Loading the production inference engine.
2. Connecting the seven input variables.
3. Implementing the prediction API.
4. Returning the 15-depth temperature prediction.
5. Building frontend visualization.
6. Integrating the prediction output format.

Heavy final retraining is intentionally postponed.

The backend should integrate the existing validated V1 checkpoints first.


22. QUICK REFERENCE
-------------------

INPUT:

    7 features
    7 days
    64 x 64 tile

    feature shape:
    [7,64,64]


NORMALIZED MODEL INPUT:

    [49,64,64]


MODEL:

    OceanEmbed-CNN
    128 latent channels
    989,967 trainable parameters


ENSEMBLE:

    Seed 42
    Seed 123
    Seed 2024

    Arithmetic mean


OUTPUT:

    [15,32,32]

    15 depths
    degrees Celsius


V1 NORMALIZATION:

    ml_config.json

    Training statistics:
    2025-07-01 to 2025-10-31


PRODUCTION ENTRY POINT:

    scripts/inference/oceanembed_inference.py

    class:
    OceanEmbedEnsemble

    method:
    predict_from_raw_window(feature_arrays)


STATUS:

    ML -> Backend handoff ready.