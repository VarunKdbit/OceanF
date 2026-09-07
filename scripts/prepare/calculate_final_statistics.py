"""
OceanEmbed Final Training Statistics Calculator
================================================

Purpose
-------
Calculate normalization statistics for the FINAL OceanEmbed training run.

Final training period:
2025-07-01 through 2025-11-30

The statistics are calculated directly from the persistent
harmonized NetCDF files.

These statistics are kept separate from the benchmark
ml_config.json statistics.

Scientific contract
-------------------
Input features:
    SST
    SSS
    SLA
    U current
    V current
    U wind
    V wind

Target:
    GLORYS thetao potential temperature

Target depths:
    0, 5, 10, 20, 30, 50, 75, 100, 125,
    150, 200, 300, 500, 700, 1000 m

Statistics:
    mean
    population standard deviation
    valid count
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ML"
HARMONIZED_DIR = ML_DIR / "harmonized"

OUTPUT_DIR = ML_DIR / "final_training"
OUTPUT_PATH = OUTPUT_DIR / "final_training_statistics.json"


# ---------------------------------------------------------------------
# FILES
# ---------------------------------------------------------------------

FILES = {
    "sst": HARMONIZED_DIR / "SST_harmonized.nc",
    "sss": HARMONIZED_DIR / "SSS_harmonized.nc",
    "sla": HARMONIZED_DIR / "SLA_harmonized.nc",
    "uo": HARMONIZED_DIR / "Currents_harmonized.nc",
    "vo": HARMONIZED_DIR / "Currents_harmonized.nc",
    "u_wind": HARMONIZED_DIR / "Winds_harmonized.nc",
    "v_wind": HARMONIZED_DIR / "Winds_harmonized.nc",
    "thetao": HARMONIZED_DIR / "SubsurfaceTemp_harmonized.nc",
}


VARIABLES = {
    "sst": "sst",
    "sss": "sss",
    "sla": "sla",
    "uo": "uo",
    "vo": "vo",
    "u_wind": "u_wind",
    "v_wind": "v_wind",
    "thetao": "thetao",
}


INPUT_FEATURES = [
    "sst",
    "sss",
    "sla",
    "uo",
    "vo",
    "u_wind",
    "v_wind",
]


TARGET_DEPTHS = [
    0.0,
    5.0,
    10.0,
    20.0,
    30.0,
    50.0,
    75.0,
    100.0,
    125.0,
    150.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0,
]


FINAL_TRAIN_START = "2025-07-01"
FINAL_TRAIN_END = "2025-11-30"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def get_dates(ds):
    """
    Return dataset dates as YYYY-MM-DD strings.
    """

    if "time" in ds.variables:
        time_var = ds.variables["time"]

    elif "valid_time" in ds.variables:
        time_var = ds.variables["valid_time"]

    else:
        raise KeyError(
            "No time or valid_time coordinate found."
        )

    units = getattr(time_var, "units", None)

    if units is None:
        raise ValueError(
            "Time variable has no units attribute."
        )

    calendar = getattr(
        time_var,
        "calendar",
        "standard",
    )

    converted = num2date(
        time_var[:],
        units=units,
        calendar=calendar,
    )

    dates = []

    for value in converted:
        dates.append(
            f"{value.year:04d}-"
            f"{value.month:02d}-"
            f"{value.day:02d}"
        )

    return dates


def get_training_indices(ds):
    """
    Return indices belonging to the final training period.
    """

    dates = get_dates(ds)

    indices = [
        i
        for i, date in enumerate(dates)
        if FINAL_TRAIN_START <= date <= FINAL_TRAIN_END
    ]

    if len(indices) == 0:
        raise ValueError(
            "No dates found inside the final training period."
        )

    return indices, dates


def calculate_statistics(array):
    """
    Calculate mean, population standard deviation and valid count.
    """

    array = np.asarray(
        array,
        dtype=np.float64,
    )

    valid = np.isfinite(array)

    valid_values = array[valid]

    if valid_values.size == 0:
        raise ValueError(
            "No valid values found."
        )

    mean = float(
        np.mean(valid_values)
    )

    std = float(
        np.std(
            valid_values,
            ddof=0,
        )
    )

    count = int(
        valid_values.size
    )

    return mean, std, count


def validate_file(path):
    """
    Confirm that a required harmonized file exists.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required harmonized file not found:\n{path}"
        )


# ---------------------------------------------------------------------
# INPUT STATISTICS
# ---------------------------------------------------------------------


def calculate_input_statistics():
    """
    Calculate statistics for all seven input features.
    """

    statistics = {}

    print()
    print("=" * 72)
    print("FINAL TRAINING INPUT STATISTICS")
    print("=" * 72)

    for feature in INPUT_FEATURES:

        path = FILES[feature]

        validate_file(path)

        print()
        print(
            f"Processing: {feature}"
        )

        print(
            f"File: {path}"
        )

        with Dataset(
            path,
            "r",
        ) as ds:

            train_indices, dates = get_training_indices(ds)

            variable_name = VARIABLES[feature]

            if variable_name not in ds.variables:
                raise KeyError(
                    f"Variable '{variable_name}' "
                    f"not found in {path.name}"
                )

            variable = ds.variables[
                variable_name
            ]

            print(
                f"Training dates found: "
                f"{len(train_indices)}"
            )

            print(
                f"First date: "
                f"{dates[train_indices[0]]}"
            )

            print(
                f"Last date: "
                f"{dates[train_indices[-1]]}"
            )

            array = np.asarray(
                variable[
                    train_indices,
                    :,
                    :
                ],
                dtype=np.float64,
            )

        mean, std, count = calculate_statistics(
            array
        )

        statistics[feature] = {
            "mean": mean,
            "std": std,
            "valid_count": count,
        }

        print(
            f"Mean: {mean:.12f}"
        )

        print(
            f"Population std: {std:.12f}"
        )

        print(
            f"Valid count: {count}"
        )

    return statistics


# ---------------------------------------------------------------------
# TARGET STATISTICS
# ---------------------------------------------------------------------


def calculate_target_statistics():
    """
    Calculate thetao statistics independently for every target depth.
    """

    statistics = {}

    print()
    print("=" * 72)
    print("FINAL TRAINING TARGET STATISTICS")
    print("=" * 72)

    path = FILES["thetao"]

    validate_file(path)

    with Dataset(
        path,
        "r",
    ) as ds:

        train_indices, dates = get_training_indices(ds)

        variable_name = VARIABLES["thetao"]

        if variable_name not in ds.variables:
            raise KeyError(
                f"Variable '{variable_name}' "
                f"not found in {path.name}"
            )

        variable = ds.variables[
            variable_name
        ]

        if "depth" not in ds.variables:
            raise KeyError(
                "Target dataset does not contain a depth coordinate."
            )

        depth = np.asarray(
            ds.variables["depth"][:],
            dtype=np.float64,
        )

        expected_depths = np.asarray(
            TARGET_DEPTHS,
            dtype=np.float64,
        )

        if not np.allclose(
            depth,
            expected_depths,
            atol=1e-5,
        ):
            raise ValueError(
                "Target depths do not match "
                "the OceanEmbed scientific contract."
            )

        print()
        print(
            f"Training dates found: "
            f"{len(train_indices)}"
        )

        print(
            f"First date: "
            f"{dates[train_indices[0]]}"
        )

        print(
            f"Last date: "
            f"{dates[train_indices[-1]]}"
        )

        print(
            f"Target depths: "
            f"{len(TARGET_DEPTHS)}"
        )

        for depth_index, depth_value in enumerate(
            TARGET_DEPTHS
        ):

            print()
            print(
                f"Processing depth: "
                f"{depth_value:.1f} m"
            )

            array = np.asarray(
                variable[
                    train_indices,
                    depth_index,
                    :,
                    :
                ],
                dtype=np.float64,
            )

            mean, std, count = calculate_statistics(
                array
            )

            statistics[str(float(depth_value))] = {
                "mean": mean,
                "std": std,
                "valid_count": count,
            }

            print(
                f"Mean: {mean:.12f}"
            )

            print(
                f"Population std: {std:.12f}"
            )

            print(
                f"Valid count: {count}"
            )

    return statistics


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():

    print()
    print("=" * 72)
    print("OceanEmbed FINAL TRAINING STATISTICS")
    print("=" * 72)

    print()
    print(
        f"Final training period:"
        f" {FINAL_TRAIN_START} -> {FINAL_TRAIN_END}"
    )

    print(
        "Statistics source:"
        " persistent harmonized NetCDF files"
    )

    print(
        "Existing benchmark ml_config.json:"
        " NOT modified"
    )

    print()

    # -------------------------------------------------------------
    # Verify all required files first.
    # -------------------------------------------------------------

    print("=" * 72)
    print("VERIFYING INPUT FILES")
    print("=" * 72)

    for name, path in FILES.items():

        validate_file(path)

        print(
            f"PASS: {name} -> {path.name}"
        )

    # -------------------------------------------------------------
    # Calculate statistics.
    # -------------------------------------------------------------

    input_statistics = (
        calculate_input_statistics()
    )

    target_statistics = (
        calculate_target_statistics()
    )

    # -------------------------------------------------------------
    # Build output object.
    # -------------------------------------------------------------

    output = {
        "project": "OceanF",
        "experiment": "OceanEmbed-CNN",
        "statistics_type": "final_training_normalization",
        "training_period": {
            "start": FINAL_TRAIN_START,
            "end": FINAL_TRAIN_END,
        },
        "input_features": INPUT_FEATURES,
        "target_variable": "thetao",
        "target_depths_m": TARGET_DEPTHS,
        "statistics_definition": {
            "mean": "arithmetic_mean_of_finite_values",
            "std": "population_standard_deviation_ddof_0",
            "valid_count": "number_of_finite_values",
        },
        "source": {
            "type": "persistent_harmonized_netcdf",
            "directory": str(HARMONIZED_DIR),
        },
        "input_statistics": input_statistics,
        "target_statistics": target_statistics,
    }

    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL TRAINING STATISTICS COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Training period:"
        f" {FINAL_TRAIN_START} -> {FINAL_TRAIN_END}"
    )

    print(
        f"Input features: {len(INPUT_FEATURES)}"
    )

    print(
        f"Target depths: {len(TARGET_DEPTHS)}"
    )

    print()
    print(
        f"Saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Existing ml_config.json was not modified."
    )

    print("=" * 72)
    print()


if __name__ == "__main__":
    main()