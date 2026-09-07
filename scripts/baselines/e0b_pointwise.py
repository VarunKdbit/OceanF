"""
OceanEmbed E0b - Pointwise Linear Regression Baseline

Definition
----------
E0b uses same-day surface observations at the same spatial location to
predict subsurface temperature using a global multivariate linear
regression.

Input features:
    SST, SSS, SLA, U, V, U_wind, V_wind

Output:
    15 thetao depths

Scientific constraints:
    - Regression is fitted ONLY on the training period.
    - No spatial neighborhood is used.
    - No retrospective temporal window is used.
    - Validation/test data are never used for fitting.
    - Each target depth has its own regression coefficients.
    - Features and targets use training-period normalization statistics.
    - Evaluation is performed in physical degrees Celsius.

Regression form:

    y_depth =
        beta_0
        + beta_1 * SST
        + beta_2 * SSS
        + beta_3 * SLA
        + beta_4 * U
        + beta_5 * V
        + beta_6 * U_wind
        + beta_7 * V_wind

The regression is fitted in normalized space.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
from netCDF4 import Dataset, num2date


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "ml_config.json"
)

HARMONIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "harmonized"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "baselines"
    / "E0b"
)


INPUT_FEATURES = [
    "sst",
    "sss",
    "sla",
    "uo",
    "vo",
    "u_wind",
    "v_wind",
]

TARGET_VARIABLE = "thetao"

TARGET_DEPTHS = [
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


FEATURE_FILES = {
    "sst": "SST_harmonized.nc",
    "sss": "SSS_harmonized.nc",
    "sla": "SLA_harmonized.nc",
    "uo": "Currents_harmonized.nc",
    "vo": "Currents_harmonized.nc",
    "u_wind": "Winds_harmonized.nc",
    "v_wind": "Winds_harmonized.nc",
}


FEATURE_VARIABLES = {
    "sst": "sst",
    "sss": "sss",
    "sla": "sla",
    "uo": "uo",
    "vo": "vo",
    "u_wind": "u_wind",
    "v_wind": "v_wind",
}


def load_config() -> dict:
    """Load the project ML configuration."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"ML configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def open_harmonized_dataset(
    filename: str,
) -> Dataset:
    """Open one harmonized NetCDF dataset."""
    path = HARMONIZED_ROOT / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Harmonized dataset not found: {path}"
        )

    return Dataset(path, "r")


def load_time_values(
    dataset: Dataset,
) -> np.ndarray:
    """Return dataset dates as numpy datetime64[D]."""
    time_variable = dataset.variables["time"]

    units = getattr(
        time_variable,
        "units",
        None,
    )

    calendar = getattr(
        time_variable,
        "calendar",
        "standard",
    )

    if units is None:
        raise ValueError(
            "Time variable has no units attribute."
        )

    converted = num2date(
        time_variable[:],
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )

    return np.asarray(
        [
            np.datetime64(
                value.strftime("%Y-%m-%d"),
                "D",
            )
            for value in converted
        ]
    )


def find_date_indices(
    dates: np.ndarray,
    start_date: str,
    end_date: str,
) -> np.ndarray:
    """Find inclusive date indices."""
    start = np.datetime64(
        start_date,
        "D",
    )

    end = np.datetime64(
        end_date,
        "D",
    )

    mask = (
        (dates >= start)
        & (dates <= end)
    )

    return np.flatnonzero(mask)


def validate_dataset(
    dataset: Dataset,
    variable_name: str,
    expected_time_count: int,
    expected_lat_count: int,
    expected_lon_count: int,
) -> None:
    """Validate a harmonized surface dataset."""
    if variable_name not in dataset.variables:
        raise ValueError(
            f"Variable '{variable_name}' not found."
        )

    for dimension in [
        "time",
        "latitude",
        "longitude",
    ]:
        if dimension not in dataset.dimensions:
            raise ValueError(
                f"Missing dimension '{dimension}'."
            )

    variable = dataset.variables[
        variable_name
    ]

    if variable.ndim != 3:
        raise ValueError(
            f"{variable_name} must be 3-dimensional."
        )

    if (
        len(dataset.dimensions["time"])
        != expected_time_count
    ):
        raise ValueError(
            f"{variable_name} time dimension mismatch."
        )

    if (
        len(dataset.dimensions["latitude"])
        != expected_lat_count
    ):
        raise ValueError(
            f"{variable_name} latitude dimension mismatch."
        )

    if (
        len(dataset.dimensions["longitude"])
        != expected_lon_count
    ):
        raise ValueError(
            f"{variable_name} longitude dimension mismatch."
        )


def validate_target_dataset(
    dataset: Dataset,
) -> None:
    """Validate the harmonized thetao dataset."""
    if TARGET_VARIABLE not in dataset.variables:
        raise ValueError(
            "thetao variable not found in target dataset."
        )

    variable = dataset.variables[
        TARGET_VARIABLE
    ]

    if variable.ndim != 4:
        raise ValueError(
            "thetao must have dimensions "
            "[time, depth, latitude, longitude]."
        )

    depth = np.asarray(
        dataset.variables["depth"][:],
        dtype=np.float64,
    )

    expected = np.asarray(
        TARGET_DEPTHS,
        dtype=np.float64,
    )

    if len(depth) != len(expected):
        raise ValueError(
            "Target depth count mismatch."
        )

    if not np.allclose(
        depth,
        expected,
        atol=1e-6,
    ):
        raise ValueError(
            "Target depth labels do not match "
            "the OceanEmbed contract."
        )

    latitude_count = len(
        dataset.dimensions["latitude"]
    )

    longitude_count = len(
        dataset.dimensions["longitude"]
    )

    if latitude_count != 101:
        raise ValueError(
            f"Expected 101 latitude cells, "
            f"got {latitude_count}."
        )

    if longitude_count != 241:
        raise ValueError(
            f"Expected 241 longitude cells, "
            f"got {longitude_count}."
        )


def load_feature_statistics(
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training-period input normalization statistics.

    Returns:
        means: [7]
        stds:  [7]
    """
    statistics = config[
        "input_statistics"
    ]

    means = np.asarray(
        [
            statistics[name]["mean"]
            for name in INPUT_FEATURES
        ],
        dtype=np.float64,
    )

    stds = np.asarray(
        [
            statistics[name]["std"]
            for name in INPUT_FEATURES
        ],
        dtype=np.float64,
    )

    if np.any(stds <= 0):
        raise ValueError(
            "All feature standard deviations "
            "must be positive."
        )

    return means, stds


def load_target_statistics(
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training-period target normalization statistics.

    Returns:
        means: [15]
        stds:  [15]
    """
    statistics = config[
        "target_statistics"
    ]

    means = np.asarray(
        [
            statistics[str(float(depth))]["mean"]
            for depth in TARGET_DEPTHS
        ],
        dtype=np.float64,
    )

    stds = np.asarray(
        [
            statistics[str(float(depth))]["std"]
            for depth in TARGET_DEPTHS
        ],
        dtype=np.float64,
    )

    if np.any(stds <= 0):
        raise ValueError(
            "All target standard deviations "
            "must be positive."
        )

    return means, stds


def build_training_normal_equations(
    feature_datasets: Dict[str, Dataset],
    target_dataset: Dataset,
    dates: np.ndarray,
    train_start: str,
    train_end: str,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    target_means: np.ndarray,
    target_stds: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build depth-specific normal equations.

    IMPORTANT:
        Every depth has its own X^T X because target validity
        differs by depth.

    Returns:
        xtx:
            [depth, 8, 8]

        xty:
            [depth, 8]

        valid_training_counts:
            [depth]
    """
    indices = find_date_indices(
        dates,
        train_start,
        train_end,
    )

    if len(indices) == 0:
        raise ValueError(
            "No training dates found."
        )

    feature_count = len(
        INPUT_FEATURES
    )

    parameter_count = (
        feature_count + 1
    )

    depth_count = len(
        TARGET_DEPTHS
    )

    # One X^T X matrix PER depth.
    xtx = np.zeros(
        (
            depth_count,
            parameter_count,
            parameter_count,
        ),
        dtype=np.float64,
    )

    # One X^T y vector PER depth.
    xty = np.zeros(
        (
            depth_count,
            parameter_count,
        ),
        dtype=np.float64,
    )

    valid_training_counts = np.zeros(
        depth_count,
        dtype=np.int64,
    )

    thetao = target_dataset.variables[
        TARGET_VARIABLE
    ]

    for date_index in indices:

        feature_arrays = []

        for feature_index, feature_name in enumerate(
            INPUT_FEATURES
        ):
            variable = feature_datasets[
                feature_name
            ].variables[
                FEATURE_VARIABLES[
                    feature_name
                ]
            ]

            values = np.asarray(
                variable[
                    date_index,
                    :,
                    :,
                ],
                dtype=np.float64,
            )

            normalized = (
                values
                - feature_means[
                    feature_index
                ]
            ) / feature_stds[
                feature_index
            ]

            feature_arrays.append(
                normalized
            )

        target_values = np.asarray(
            thetao[
                date_index,
                :,
                :,
                :,
            ],
            dtype=np.float64,
        )

        height, width = (
            feature_arrays[0].shape
        )

        x_features = np.stack(
            [
                array.reshape(-1)
                for array in feature_arrays
            ],
            axis=1,
        )

        intercept = np.ones(
            (
                height * width,
                1,
            ),
            dtype=np.float64,
        )

        x_matrix = np.concatenate(
            [
                intercept,
                x_features,
            ],
            axis=1,
        )

        all_input_valid = np.all(
            np.isfinite(x_matrix),
            axis=1,
        )

        for depth_index in range(
            depth_count
        ):
            target = target_values[
                depth_index
            ].reshape(-1)

            normalized_target = (
                target
                - target_means[
                    depth_index
                ]
            ) / target_stds[
                depth_index
            ]

            valid = (
                all_input_valid
                & np.isfinite(
                    normalized_target
                )
            )

            if not np.any(valid):
                continue

            x_valid = x_matrix[
                valid
            ]

            y_valid = normalized_target[
                valid
            ]

            # CRITICAL:
            # Only this depth's X^T X is updated.
            xtx[
                depth_index
            ] += x_valid.T @ x_valid

            xty[
                depth_index
            ] += x_valid.T @ y_valid

            valid_training_counts[
                depth_index
            ] += int(
                np.count_nonzero(valid)
            )

    return (
        xtx,
        xty,
        valid_training_counts,
    )


def fit_regression(
    xtx: np.ndarray,
    xty: np.ndarray,
) -> np.ndarray:
    """
    Solve depth-specific linear regressions.

    Input:
        xtx: [15, 8, 8]
        xty: [15, 8]

    Output:
        coefficients: [15, 8]

    Each row contains:

        [intercept,
         SST,
         SSS,
         SLA,
         U,
         V,
         U_wind,
         V_wind]
    """
    if xtx.ndim != 3:
        raise ValueError(
            "xtx must have shape [depth, 8, 8]."
        )

    if xty.ndim != 2:
        raise ValueError(
            "xty must have shape [depth, 8]."
        )

    depth_count = xtx.shape[0]
    parameter_count = xtx.shape[1]

    if xtx.shape != (
        depth_count,
        parameter_count,
        parameter_count,
    ):
        raise ValueError(
            "Invalid xtx shape."
        )

    if xty.shape != (
        depth_count,
        parameter_count,
    ):
        raise ValueError(
            "xtx/xty shape mismatch."
        )

    coefficients = np.zeros(
        (
            depth_count,
            parameter_count,
        ),
        dtype=np.float64,
    )

    for depth_index in range(
        depth_count
    ):
        try:
            coefficients[
                depth_index
            ] = np.linalg.solve(
                xtx[depth_index],
                xty[depth_index],
            )
        except np.linalg.LinAlgError:
            coefficients[
                depth_index
            ] = np.linalg.lstsq(
                xtx[depth_index],
                xty[depth_index],
                rcond=None,
            )[0]

    if not np.all(
        np.isfinite(coefficients)
    ):
        raise ValueError(
            "Regression coefficients contain "
            "non-finite values."
        )

    return coefficients


def predict_depths(
    feature_arrays: list[np.ndarray],
    coefficients: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    target_means: np.ndarray,
    target_stds: np.ndarray,
) -> np.ndarray:
    """
    Predict all 15 depths for one day.

    Returns:
        prediction:
            [15, latitude, longitude]
            in degC
    """
    if coefficients.shape != (
        len(TARGET_DEPTHS),
        len(INPUT_FEATURES) + 1,
    ):
        raise ValueError(
            "Unexpected coefficient shape."
        )

    height, width = (
        feature_arrays[0].shape
    )

    normalized_features = []

    for index, values in enumerate(
        feature_arrays
    ):
        normalized_features.append(
            (
                values
                - feature_means[index]
            ) / feature_stds[index]
        )

    x_features = np.stack(
        [
            values.reshape(-1)
            for values in normalized_features
        ],
        axis=1,
    )

    intercept = np.ones(
        (
            height * width,
            1,
        ),
        dtype=np.float64,
    )

    x_matrix = np.concatenate(
        [
            intercept,
            x_features,
        ],
        axis=1,
    )

    normalized_prediction = (
        x_matrix @ coefficients.T
    )

    prediction = (
        normalized_prediction
        * target_stds.reshape(
            1,
            -1,
        )
        + target_means.reshape(
            1,
            -1,
        )
    )

    prediction = prediction.T.reshape(
        len(TARGET_DEPTHS),
        height,
        width,
    )

    raw_feature_stack = np.stack(
        feature_arrays,
        axis=0,
    )

    invalid_input = ~np.all(
        np.isfinite(
            raw_feature_stack
        ),
        axis=0,
    )

    prediction[
        :,
        invalid_input,
    ] = np.nan

    return prediction


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Calculate RMSE, MAE, bias and Pearson."""
    valid = (
        mask.astype(bool)
        & np.isfinite(prediction)
        & np.isfinite(target)
    )

    count = int(
        np.count_nonzero(valid)
    )

    if count == 0:
        return {
            "rmse_c": math.nan,
            "mae_c": math.nan,
            "bias_c": math.nan,
            "pearson": math.nan,
            "valid_count": 0,
        }

    pred = prediction[
        valid
    ].astype(np.float64)

    obs = target[
        valid
    ].astype(np.float64)

    error = pred - obs

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    bias = float(
        np.mean(error)
    )

    pred_centered = (
        pred - np.mean(pred)
    )

    obs_centered = (
        obs - np.mean(obs)
    )

    denominator = float(
        np.sqrt(
            np.sum(
                pred_centered ** 2
            )
            * np.sum(
                obs_centered ** 2
            )
        )
    )

    if denominator == 0.0:
        pearson = math.nan
    else:
        pearson = float(
            np.sum(
                pred_centered
                * obs_centered
            )
            / denominator
        )

    return {
        "rmse_c": rmse,
        "mae_c": mae,
        "bias_c": bias,
        "pearson": pearson,
        "valid_count": count,
    }


def evaluate_split(
    feature_datasets: Dict[str, Dataset],
    target_dataset: Dataset,
    dates: np.ndarray,
    coefficients: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    target_means: np.ndarray,
    target_stds: np.ndarray,
    start_date: str,
    end_date: str,
) -> dict:
    """Evaluate E0b on validation or test."""
    indices = find_date_indices(
        dates,
        start_date,
        end_date,
    )

    if len(indices) == 0:
        raise ValueError(
            "No dates found for evaluation period."
        )

    thetao = target_dataset.variables[
        TARGET_VARIABLE
    ]

    depth_predictions = [
        []
        for _ in TARGET_DEPTHS
    ]

    depth_targets = [
        []
        for _ in TARGET_DEPTHS
    ]

    depth_masks = [
        []
        for _ in TARGET_DEPTHS
    ]

    for date_index in indices:

        feature_arrays = []

        for feature_name in INPUT_FEATURES:
            variable = feature_datasets[
                feature_name
            ].variables[
                FEATURE_VARIABLES[
                    feature_name
                ]
            ]

            values = np.asarray(
                variable[
                    date_index,
                    :,
                    :,
                ],
                dtype=np.float64,
            )

            feature_arrays.append(
                values
            )

        target_values = np.asarray(
            thetao[
                date_index,
                :,
                :,
                :,
            ],
            dtype=np.float64,
        )

        prediction = predict_depths(
            feature_arrays,
            coefficients,
            feature_means,
            feature_stds,
            target_means,
            target_stds,
        )

        input_valid = np.all(
            np.isfinite(
                np.stack(
                    feature_arrays,
                    axis=0,
                )
            ),
            axis=0,
        )

        for depth_index in range(
            len(TARGET_DEPTHS)
        ):
            target = target_values[
                depth_index
            ]

            valid = (
                input_valid
                & np.isfinite(target)
            )

            depth_predictions[
                depth_index
            ].append(
                prediction[
                    depth_index
                ]
            )

            depth_targets[
                depth_index
            ].append(
                target
            )

            depth_masks[
                depth_index
            ].append(
                valid
            )

    results = {}

    for depth_index, depth in enumerate(
        TARGET_DEPTHS
    ):
        prediction_array = np.stack(
            depth_predictions[
                depth_index
            ],
            axis=0,
        )

        target_array = np.stack(
            depth_targets[
                depth_index
            ],
            axis=0,
        )

        mask_array = np.stack(
            depth_masks[
                depth_index
            ],
            axis=0,
        )

        results[
            str(depth)
        ] = calculate_metrics(
            prediction_array,
            target_array,
            mask_array,
        )

    return results


def print_results(
    split: str,
    results: dict,
) -> None:
    """Print depth-wise E0b results."""
    print()
    print("=" * 80)
    print(
        "OceanEmbed E0b POINTWISE "
        "BASELINE RESULTS"
    )
    print("=" * 80)
    print(
        f"Split: {split}"
    )
    print()

    print(
        f"{'Depth':>8} "
        f"{'RMSE °C':>14} "
        f"{'MAE °C':>14} "
        f"{'Bias °C':>14} "
        f"{'Pearson':>12} "
        f"{'Valid':>12}"
    )

    for depth in TARGET_DEPTHS:
        metrics = results[
            str(depth)
        ]

        print(
            f"{depth:8.1f} "
            f"{metrics['rmse_c']:14.6f} "
            f"{metrics['mae_c']:14.6f} "
            f"{metrics['bias_c']:14.6f} "
            f"{metrics['pearson']:12.6f} "
            f"{metrics['valid_count']:12d}"
        )

    print("=" * 80)


def save_json(
    path: Path,
    payload: dict,
) -> None:
    """Save results as JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=True,
        )


def self_test() -> None:
    """Run E0b implementation self-tests."""
    print("=" * 80)
    print(
        "OceanEmbed E0b POINTWISE "
        "BASELINE SELF-TEST"
    )
    print("=" * 80)

    config = load_config()

    print(
        "PASS: ML configuration loaded."
    )

    assert (
        config["input_features"]
        == INPUT_FEATURES
    )

    assert (
        config["target_variable"]
        == TARGET_VARIABLE
    )

    assert (
        config["target_depths_m"]
        == TARGET_DEPTHS
    )

    print(
        "PASS: feature and target "
        "contract validated."
    )

    feature_means, feature_stds = (
        load_feature_statistics(
            config
        )
    )

    target_means, target_stds = (
        load_target_statistics(
            config
        )
    )

    assert feature_means.shape == (
        7,
    )

    assert feature_stds.shape == (
        7,
    )

    assert target_means.shape == (
        15,
    )

    assert target_stds.shape == (
        15,
    )

    assert np.all(
        np.isfinite(
            feature_means
        )
    )

    assert np.all(
        np.isfinite(
            feature_stds
        )
    )

    assert np.all(
        np.isfinite(
            target_means
        )
    )

    assert np.all(
        np.isfinite(
            target_stds
        )
    )

    print(
        "PASS: normalization "
        "statistics loaded."
    )

    # ---------------------------------------------------------
    # Synthetic regression test.
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    sample_count = 500
    depth_count = 15
    parameter_count = 8

    x = rng.normal(
        size=(
            sample_count,
            7,
        )
    )

    true_coefficients = rng.normal(
        size=(
            depth_count,
            parameter_count,
        )
    )

    x_with_intercept = np.concatenate(
        [
            np.ones(
                (
                    sample_count,
                    1,
                )
            ),
            x,
        ],
        axis=1,
    )

    y = (
        x_with_intercept
        @ true_coefficients.T
    )

    xtx = np.zeros(
        (
            depth_count,
            parameter_count,
            parameter_count,
        ),
        dtype=np.float64,
    )

    xty = np.zeros(
        (
            depth_count,
            parameter_count,
        ),
        dtype=np.float64,
    )

    for depth_index in range(
        depth_count
    ):
        xtx[
            depth_index
        ] = (
            x_with_intercept.T
            @ x_with_intercept
        )

        xty[
            depth_index
        ] = (
            x_with_intercept.T
            @ y[:, depth_index]
        )

    recovered = fit_regression(
        xtx,
        xty,
    )

    assert recovered.shape == (
        15,
        8,
    )

    assert np.allclose(
        recovered,
        true_coefficients,
        atol=1e-8,
    )

    print(
        "PASS: depth-specific "
        "multivariate regression fitting."
    )

    # ---------------------------------------------------------
    # Verify depth-specific X^T X behavior.
    # ---------------------------------------------------------

    synthetic_xtx = np.zeros(
        (
            2,
            2,
            2,
        ),
        dtype=np.float64,
    )

    synthetic_xty = np.zeros(
        (
            2,
            2,
        ),
        dtype=np.float64,
    )

    synthetic_xtx[0] = np.eye(
        2
    ) * 2.0

    synthetic_xtx[1] = np.eye(
        2
    ) * 4.0

    synthetic_xty[0] = np.array(
        [2.0, 4.0]
    )

    synthetic_xty[1] = np.array(
        [4.0, 8.0]
    )

    small_coefficients = fit_regression(
        synthetic_xtx,
        synthetic_xty,
    )

    assert np.allclose(
        small_coefficients[0],
        np.array(
            [1.0, 2.0]
        ),
        atol=1e-10,
    )

    assert np.allclose(
        small_coefficients[1],
        np.array(
            [1.0, 2.0]
        ),
        atol=1e-10,
    )

    print(
        "PASS: depth-specific "
        "normal-equation isolation."
    )

    # ---------------------------------------------------------
    # Prediction test.
    # ---------------------------------------------------------

    feature_arrays = [
        x[
            :,
            index,
        ].reshape(
            20,
            25,
        )
        for index in range(7)
    ]

    predictions = predict_depths(
        feature_arrays,
        true_coefficients,
        np.zeros(7),
        np.ones(7),
        np.zeros(15),
        np.ones(15),
    )

    assert predictions.shape == (
        15,
        20,
        25,
    )

    assert np.all(
        np.isfinite(
            predictions
        )
    )

    print(
        "PASS: pointwise prediction "
        "shape and finiteness."
    )

    # ---------------------------------------------------------
    # Missing-input behavior.
    # ---------------------------------------------------------

    feature_arrays[0][
        0,
        0,
    ] = np.nan

    missing_prediction = predict_depths(
        feature_arrays,
        true_coefficients,
        np.zeros(7),
        np.ones(7),
        np.zeros(15),
        np.ones(15),
    )

    assert np.all(
        np.isnan(
            missing_prediction[
                :,
                0,
                0,
            ]
        )
    )

    print(
        "PASS: missing-input masking."
    )

    # ---------------------------------------------------------
    # Metric test.
    # ---------------------------------------------------------

    prediction = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    target = np.array(
        [
            [1.0, 3.0],
            [2.0, 5.0],
        ]
    )

    mask = np.ones(
        (
            2,
            2,
        ),
        dtype=bool,
    )

    metrics = calculate_metrics(
        prediction,
        target,
        mask,
    )

    assert (
        metrics["valid_count"]
        == 4
    )

    assert (
        metrics["rmse_c"]
        >= 0.0
    )

    assert (
        metrics["mae_c"]
        >= 0.0
    )

    print(
        "PASS: regression "
        "metric calculation."
    )

    print()
    print("=" * 80)
    print(
        "E0b POINTWISE BASELINE "
        "SELF-TEST PASSED"
    )
    print("=" * 80)


def run_baseline(
    split: str,
) -> None:
    """Fit E0b on training data and evaluate one split."""
    config = load_config()

    feature_means, feature_stds = (
        load_feature_statistics(
            config
        )
    )

    target_means, target_stds = (
        load_target_statistics(
            config
        )
    )

    feature_datasets = {}

    target_dataset = open_harmonized_dataset(
        "SubsurfaceTemp_harmonized.nc"
    )

    try:
        validate_target_dataset(
            target_dataset
        )

        dates = load_time_values(
            target_dataset
        )

        expected_time_count = len(
            dates
        )

        expected_lat_count = len(
            target_dataset.dimensions[
                "latitude"
            ]
        )

        expected_lon_count = len(
            target_dataset.dimensions[
                "longitude"
            ]
        )

        print("=" * 80)
        print(
            "OceanEmbed E0b POINTWISE "
            "BASELINE"
        )
        print("=" * 80)

        print(
            "Training period: "
            f"{config['time']['train_start']} "
            "→ "
            f"{config['time']['train_end']}"
        )

        if split == "validation":
            eval_start = config[
                "time"
            ][
                "validation_start"
            ]

            eval_end = config[
                "time"
            ][
                "validation_end"
            ]

        elif split == "test":
            eval_start = config[
                "time"
            ][
                "test_start"
            ]

            eval_end = config[
                "time"
            ][
                "test_end"
            ]

        else:
            raise ValueError(
                f"Unsupported split: {split}"
            )

        print(
            "Evaluation period: "
            f"{eval_start} → {eval_end}"
        )

        opened_datasets = set()

        for feature_name in INPUT_FEATURES:

            filename = FEATURE_FILES[
                feature_name
            ]

            if filename not in opened_datasets:
                feature_datasets[
                    filename
                ] = open_harmonized_dataset(
                    filename
                )

                opened_datasets.add(
                    filename
                )

            feature_datasets[
                feature_name
            ] = feature_datasets[
                filename
            ]

            validate_dataset(
                feature_datasets[
                    feature_name
                ],
                FEATURE_VARIABLES[
                    feature_name
                ],
                expected_time_count,
                expected_lat_count,
                expected_lon_count,
            )

        print()
        print(
            "All harmonized input "
            "datasets validated."
        )

        print()
        print(
            "Fitting training-only "
            "pointwise linear regression..."
        )

        (
            xtx,
            xty,
            training_valid_counts,
        ) = build_training_normal_equations(
            feature_datasets,
            target_dataset,
            dates,
            config["time"][
                "train_start"
            ],
            config["time"][
                "train_end"
            ],
            feature_means,
            feature_stds,
            target_means,
            target_stds,
        )

        coefficients = fit_regression(
            xtx,
            xty,
        )

        print(
            "Regression coefficients fitted."
        )

        print()
        print(
            "Training valid observations "
            "by depth:"
        )

        for depth_index, depth in enumerate(
            TARGET_DEPTHS
        ):
            print(
                f"{depth:7.1f} m : "
                f"{training_valid_counts[depth_index]}"
            )

        results = evaluate_split(
            feature_datasets,
            target_dataset,
            dates,
            coefficients,
            feature_means,
            feature_stds,
            target_means,
            target_stds,
            eval_start,
            eval_end,
        )

        print_results(
            split,
            results,
        )

        coefficient_output = (
            OUTPUT_ROOT
            / "e0b_pointwise_coefficients.json"
        )

        coefficient_payload = {
            "experiment": "E0b",
            "name": (
                "pointwise_linear_regression"
            ),
            "target_variable": (
                TARGET_VARIABLE
            ),
            "input_features": (
                INPUT_FEATURES
            ),
            "target_depths_m": (
                TARGET_DEPTHS
            ),
            "training_period": {
                "start": config["time"][
                    "train_start"
                ],
                "end": config["time"][
                    "train_end"
                ],
            },
            "normalization": {
                "inputs": (
                    "training_period_mean_std"
                ),
                "target": (
                    "training_period_mean_std_by_depth"
                ),
            },
            "intercept_included": True,
            "coefficients_layout": [
                "intercept",
                "sst",
                "sss",
                "sla",
                "uo",
                "vo",
                "u_wind",
                "v_wind",
            ],
            "coefficients_normalized_space": (
                coefficients.tolist()
            ),
            "training_valid_observations_by_depth": {
                str(depth): int(
                    training_valid_counts[
                        depth_index
                    ]
                )
                for depth_index, depth in enumerate(
                    TARGET_DEPTHS
                )
            },
        }

        save_json(
            coefficient_output,
            coefficient_payload,
        )

        result_output = (
            OUTPUT_ROOT
            / f"e0b_pointwise_{split}.json"
        )

        result_payload = {
            "experiment": "E0b",
            "name": (
                "pointwise_linear_regression"
            ),
            "description": (
                "Same-day surface observations "
                "mapped to subsurface temperature "
                "using depth-specific global "
                "multivariate linear regressions."
            ),
            "split": split,
            "target_variable": (
                TARGET_VARIABLE
            ),
            "units": "degC",
            "input_features": (
                INPUT_FEATURES
            ),
            "target_depths_m": (
                TARGET_DEPTHS
            ),
            "training_period": {
                "start": config["time"][
                    "train_start"
                ],
                "end": config["time"][
                    "train_end"
                ],
            },
            "evaluation_period": {
                "start": eval_start,
                "end": eval_end,
            },
            "grid": {
                "latitude_count": 101,
                "longitude_count": 241,
                "resolution_deg": 0.25,
            },
            "training_valid_observations_by_depth": {
                str(depth): int(
                    training_valid_counts[
                        depth_index
                    ]
                )
                for depth_index, depth in enumerate(
                    TARGET_DEPTHS
                )
            },
            "metrics": results,
        }

        save_json(
            result_output,
            result_payload,
        )

        print()
        print(
            f"Coefficients saved: "
            f"{coefficient_output}"
        )

        print(
            f"Results saved: "
            f"{result_output}"
        )

    finally:
        target_dataset.close()

        unique_datasets = set(
            feature_datasets.values()
        )

        for dataset in unique_datasets:
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OceanEmbed E0b pointwise "
            "linear regression baseline."
        )
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run E0b implementation "
            "self-test."
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="validation",
        help="Evaluation split.",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_baseline(
        split=args.split
    )


if __name__ == "__main__":
    main()