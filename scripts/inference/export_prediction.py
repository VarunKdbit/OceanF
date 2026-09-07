from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from scripts.inference.oceanembed_inference import OceanEmbedEnsemble


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HARMONIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "harmonized"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "inference_outputs"
)

TARGET_DATE = "2025-12-01"

TILE_ROW = 0
TILE_COL = 0

HISTORY_DAYS = 7

INPUT_FEATURES = [
    "sst",
    "sss",
    "sla",
    "uo",
    "vo",
    "u_wind",
    "v_wind",
]

DEPTHS_M = [
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
    1000,
]

DATASET_FILES = {
    "sst": HARMONIZED_DIR / "SST_harmonized.nc",
    "sss": HARMONIZED_DIR / "SSS_harmonized.nc",
    "sla": HARMONIZED_DIR / "SLA_harmonized.nc",
    "uo": HARMONIZED_DIR / "Currents_harmonized.nc",
    "vo": HARMONIZED_DIR / "Currents_harmonized.nc",
    "u_wind": HARMONIZED_DIR / "Winds_harmonized.nc",
    "v_wind": HARMONIZED_DIR / "Winds_harmonized.nc",
}

VARIABLE_NAMES = {
    "sst": "sst",
    "sss": "sss",
    "sla": "sla",
    "uo": "uo",
    "vo": "vo",
    "u_wind": "u_wind",
    "v_wind": "v_wind",
}


def get_spatial_tile_slices() -> tuple[slice, slice]:
    """
    Return the 64x64 input tile slices.

    Tile stride is 32 pixels.

    Each 64x64 input tile contains a centered
    32x32 target region with 16 pixels of context
    on every side.
    """

    row_start = TILE_ROW * 32
    col_start = TILE_COL * 32

    row_end = row_start + 64
    col_end = col_start + 64

    return (
        slice(row_start, row_end),
        slice(col_start, col_end),
    )


def load_raw_window() -> dict[str, np.ndarray]:
    """
    Load the real seven-day raw/harmonized input window.

    Returns:

        {
            "sst": [7,64,64],
            "sss": [7,64,64],
            "sla": [7,64,64],
            "uo": [7,64,64],
            "vo": [7,64,64],
            "u_wind": [7,64,64],
            "v_wind": [7,64,64]
        }

    The inference engine performs V1 train-only
    normalization using ml_config.json.
    """

    target_date = np.datetime64(TARGET_DATE)

    window_start = (
        target_date
        - np.timedelta64(HISTORY_DAYS - 1, "D")
    )

    row_slice, col_slice = (
        get_spatial_tile_slices()
    )

    feature_arrays: dict[str, np.ndarray] = {}

    opened_datasets: dict[Path, xr.Dataset] = {}

    try:
        for feature in INPUT_FEATURES:

            path = DATASET_FILES[feature]

            if path not in opened_datasets:
                opened_datasets[path] = (
                    xr.open_dataset(path)
                )

            ds = opened_datasets[path]

            variable = VARIABLE_NAMES[feature]

            if variable not in ds:
                raise KeyError(
                    f"Variable '{variable}' not found in "
                    f"{path}"
                )

            data = ds[variable]

            selected = data.sel(
                time=slice(
                    window_start,
                    target_date,
                )
            )

            if "time" not in selected.dims:
                raise ValueError(
                    f"Feature '{feature}' has no time "
                    "dimension."
                )

            values = selected.values.astype(
                np.float32
            )

            if values.shape[0] != HISTORY_DAYS:
                raise ValueError(
                    f"Feature '{feature}' has "
                    f"{values.shape[0]} days; "
                    f"expected {HISTORY_DAYS}."
                )

            if values.shape[-2:] != (
                101,
                241,
            ):
                raise ValueError(
                    f"Feature '{feature}' has spatial "
                    f"shape {values.shape[-2:]}; "
                    "expected (101, 241)."
                )

            tile = values[
                :,
                row_slice,
                col_slice,
            ]

            expected_shape = (
                HISTORY_DAYS,
                64,
                64,
            )

            if tile.shape != expected_shape:
                raise ValueError(
                    f"Feature '{feature}' produced "
                    f"shape {tile.shape}; "
                    f"expected {expected_shape}."
                )

            feature_arrays[feature] = tile

    finally:

        for ds in opened_datasets.values():
            ds.close()

    if set(feature_arrays.keys()) != set(
        INPUT_FEATURES
    ):
        raise RuntimeError(
            "Not all required input features "
            "were loaded."
        )

    return feature_arrays


def get_output_coordinates() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Read the actual harmonized grid coordinates.

    The model output is the centered 32x32 region
    inside the 64x64 input tile.
    """

    path = DATASET_FILES["sst"]

    with xr.open_dataset(path) as ds:

        if "latitude" not in ds.coords:
            raise ValueError(
                "Harmonized SST dataset does not "
                "contain a latitude coordinate."
            )

        if "longitude" not in ds.coords:
            raise ValueError(
                "Harmonized SST dataset does not "
                "contain a longitude coordinate."
            )

        latitude = ds["latitude"].values
        longitude = ds["longitude"].values

    row_start = TILE_ROW * 32 + 16
    col_start = TILE_COL * 32 + 16

    row_end = row_start + 32
    col_end = col_start + 32

    output_latitude = latitude[
        row_start:row_end
    ]

    output_longitude = longitude[
        col_start:col_end
    ]

    if output_latitude.shape != (32,):
        raise ValueError(
            "Unexpected output latitude shape: "
            f"{output_latitude.shape}"
        )

    if output_longitude.shape != (32,):
        raise ValueError(
            "Unexpected output longitude shape: "
            f"{output_longitude.shape}"
        )

    return (
        output_latitude.astype(np.float32),
        output_longitude.astype(np.float32),
    )


def tensor_to_numpy(
    prediction: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """
    Safely convert a PyTorch CPU/CUDA tensor or NumPy
    array to a NumPy float32 array.
    """

    if isinstance(prediction, torch.Tensor):

        return (
            prediction
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return np.asarray(
        prediction,
        dtype=np.float32,
    )


def save_prediction(
    prediction: np.ndarray,
    window_start: str,
    window_end: str,
    output_netcdf: Path,
    output_metadata: Path,
) -> None:

    output_netcdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction = tensor_to_numpy(
        prediction
    )

    expected_shape = (
        15,
        32,
        32,
    )

    if prediction.shape != expected_shape:
        raise ValueError(
            f"Unexpected prediction shape: "
            f"{prediction.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(prediction).all():
        raise ValueError(
            "Prediction contains non-finite values."
        )

    latitude, longitude = (
        get_output_coordinates()
    )

    ds = xr.Dataset(
        data_vars={
            "thetao_prediction": (
                (
                    "depth",
                    "latitude",
                    "longitude",
                ),
                prediction,
            )
        },
        coords={
            "depth": np.asarray(
                DEPTHS_M,
                dtype=np.float32,
            ),
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "project": "OceanEmbed",
            "experiment": "OceanEmbed-CNN",
            "experiment_variant": (
                "E2_7day_retrospective"
            ),
            "checkpoint_version": "V1",
            "prediction_date": TARGET_DATE,
            "temperature_units": "degC",
            "history_days": HISTORY_DAYS,
            "ensemble_type": "three_seed_mean",
            "ensemble_seeds": "42,123,2024",
            "spatial_resolution_degrees": 0.25,
            "tile_row": TILE_ROW,
            "tile_col": TILE_COL,
            "input_size": "64x64",
            "output_size": "32x32",
            "target_context_pixels": 16,
        },
    )

    ds[
        "thetao_prediction"
    ].attrs.update(
        {
            "long_name": (
                "OceanEmbed ensemble subsurface "
                "ocean potential temperature "
                "prediction"
            ),
            "units": "degC",
        }
    )

    ds.to_netcdf(
        output_netcdf
    )

    ds.close()

    metadata = {
        "project": "OceanEmbed",
        "experiment": "OceanEmbed-CNN",
        "experiment_variant": (
            "E2_7day_retrospective"
        ),
        "checkpoint_version": "V1",
        "prediction_date": TARGET_DATE,
        "input_window": {
            "start": window_start,
            "end": window_end,
            "days": HISTORY_DAYS,
        },
        "input_features": INPUT_FEATURES,
        "input_shape": [
            49,
            64,
            64,
        ],
        "output_shape": [
            15,
            32,
            32,
        ],
        "depths_m": DEPTHS_M,
        "temperature_units": "degC",
        "ensemble": {
            "type": "three_seed_mean",
            "seeds": [
                42,
                123,
                2024,
            ],
            "size": 3,
        },
        "spatial": {
            "resolution_degrees": 0.25,
            "tile_row": TILE_ROW,
            "tile_col": TILE_COL,
            "input_size": [
                64,
                64,
            ],
            "output_size": [
                32,
                32,
            ],
            "target_context_pixels": 16,
            "output_latitude_min": float(
                latitude.min()
            ),
            "output_latitude_max": float(
                latitude.max()
            ),
            "output_longitude_min": float(
                longitude.min()
            ),
            "output_longitude_max": float(
                longitude.max()
            ),
        },
        "normalization": {
            "statistics_file": (
                "data/processed/ML/ml_config.json"
            ),
            "source": "ml_config.json",
            "training_period": {
                "start": "2025-07-01",
                "end": "2025-10-31",
            },
            "validation_period": {
                "start": "2025-11-01",
                "end": "2025-11-30",
            },
            "test_period": {
                "start": "2025-12-01",
                "end": "2025-12-31",
            },
            "note": (
                "V1 checkpoints use the original "
                "train-only normalization stored "
                "in ml_config.json. "
                "final_training_statistics.json "
                "is not used by the current V1 "
                "checkpoints."
            ),
        },
        "output_files": {
            "netcdf": output_netcdf.name,
        },
        "prediction_range_degC": {
            "min": float(
                np.min(prediction)
            ),
            "max": float(
                np.max(prediction)
            ),
        },
    }

    output_metadata.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def validate_exported_files(
    output_netcdf: Path,
    output_metadata: Path,
) -> None:

    with xr.open_dataset(
        output_netcdf
    ) as ds:

        if (
            "thetao_prediction"
            not in ds
        ):
            raise ValueError(
                "Saved NetCDF is missing "
                "'thetao_prediction'."
            )

        prediction = (
            ds[
                "thetao_prediction"
            ].values
        )

        if prediction.shape != (
            15,
            32,
            32,
        ):
            raise ValueError(
                "Saved NetCDF has incorrect "
                f"prediction shape: "
                f"{prediction.shape}"
            )

        if not np.isfinite(
            prediction
        ).all():
            raise ValueError(
                "Saved NetCDF contains "
                "non-finite prediction values."
            )

        if len(ds["depth"]) != 15:
            raise ValueError(
                "Saved NetCDF has incorrect "
                "depth count."
            )

        if len(ds["latitude"]) != 32:
            raise ValueError(
                "Saved NetCDF has incorrect "
                "latitude count."
            )

        if len(ds["longitude"]) != 32:
            raise ValueError(
                "Saved NetCDF has incorrect "
                "longitude count."
            )

    with output_metadata.open(
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    if metadata[
        "checkpoint_version"
    ] != "V1":
        raise ValueError(
            "Metadata checkpoint version "
            "is not V1."
        )

    normalization = metadata[
        "normalization"
    ]

    if normalization[
        "source"
    ] != "ml_config.json":
        raise ValueError(
            "V1 normalization source is "
            "not ml_config.json."
        )

    if normalization[
        "statistics_file"
    ] != (
        "data/processed/ML/ml_config.json"
    ):
        raise ValueError(
            "V1 normalization statistics "
            "file is incorrect."
        )

    if normalization[
        "training_period"
    ] != {
        "start": "2025-07-01",
        "end": "2025-10-31",
    }:
        raise ValueError(
            "Incorrect V1 training "
            "normalization period."
        )

    if normalization[
        "validation_period"
    ] != {
        "start": "2025-11-01",
        "end": "2025-11-30",
    }:
        raise ValueError(
            "Incorrect V1 validation period."
        )

    if normalization[
        "test_period"
    ] != {
        "start": "2025-12-01",
        "end": "2025-12-31",
    }:
        raise ValueError(
            "Incorrect V1 test period."
        )

    if metadata[
        "ensemble"
    ]["size"] != 3:
        raise ValueError(
            "Incorrect ensemble size."
        )

    if metadata[
        "ensemble"
    ]["seeds"] != [
        42,
        123,
        2024,
    ]:
        raise ValueError(
            "Incorrect ensemble seeds."
        )


def main() -> None:

    print("=" * 72)
    print("OceanEmbed PREDICTION EXPORT")
    print("=" * 72)
    print()

    engine = OceanEmbedEnsemble(
        device="cuda"
    )

    print(
        "PASS: ensemble loaded"
    )

    print()

    feature_arrays = (
        load_raw_window()
    )

    target_date = np.datetime64(
        TARGET_DATE
    )

    window_start = (
        target_date
        - np.timedelta64(
            HISTORY_DAYS - 1,
            "D",
        )
    )

    window_start_string = str(
        window_start
    )

    window_end_string = str(
        target_date
    )

    print(
        f"Target date: {TARGET_DATE}"
    )

    print(
        f"Input window: "
        f"{window_start_string} to "
        f"{window_end_string}"
    )

    for feature in INPUT_FEATURES:

        values = feature_arrays[
            feature
        ]

        print(
            f"{feature}: "
            f"{values.shape}, "
            f"finite="
            f"{np.isfinite(values).sum()}/"
            f"{values.size}"
        )

    print()

    print(
        "PASS: seven-day input window extracted"
    )

    print()

    prediction = (
        engine.predict_from_raw_window(
            feature_arrays
        )
    )

    prediction = tensor_to_numpy(
        prediction
    )

    if prediction.shape != (
        15,
        32,
        32,
    ):
        raise ValueError(
            f"Unexpected prediction shape: "
            f"{prediction.shape}"
        )

    if not np.isfinite(
        prediction
    ).all():
        raise ValueError(
            "Prediction contains "
            "non-finite values."
        )

    print(
        "PASS: prediction generated"
    )

    print(
        f"Prediction range: "
        f"{prediction.min():.3f} °C to "
        f"{prediction.max():.3f} °C"
    )

    print()

    output_netcdf = (
        OUTPUT_DIR
        / (
            "oceanembed_prediction_"
            "2025-12-01_tile_r0_c0.nc"
        )
    )

    output_metadata = (
        OUTPUT_DIR
        / (
            "oceanembed_prediction_"
            "2025-12-01_tile_r0_c0_metadata.json"
        )
    )

    save_prediction(
        prediction=prediction,
        window_start=window_start_string,
        window_end=window_end_string,
        output_netcdf=output_netcdf,
        output_metadata=output_metadata,
    )

    print(
        "PASS: NetCDF prediction saved:"
    )

    print(output_netcdf)

    print()

    print(
        "PASS: metadata JSON saved:"
    )

    print(output_metadata)

    print()

    validate_exported_files(
        output_netcdf=output_netcdf,
        output_metadata=output_metadata,
    )

    print(
        "PASS: saved NetCDF "
        "re-opened and validated"
    )

    print(
        "PASS: V1 normalization "
        "metadata validated"
    )

    print(
        "PASS: ensemble metadata validated"
    )

    print()

    print("=" * 72)
    print("PREDICTION EXPORT PASSED")
    print("=" * 72)
    print()

    print(
        "Backend-ready prediction files:"
    )

    print(output_netcdf)

    print(output_metadata)


if __name__ == "__main__":
    main()