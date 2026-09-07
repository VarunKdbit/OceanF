"""
OceanEmbed ML Normalization Verification
=========================================

Purpose
-------
Independently calculate normalization statistics directly from the
persistent harmonized NetCDF files using ONLY the training period.

The calculated statistics are compared against:

    data/processed/ML/ml_config.json

This verifies that the configured normalization statistics are
consistent with the actual ML training data.

Scientific contract
-------------------
Training period:
    2025-07-01 through 2025-10-31

Input features:
    SST
    SSS
    SLA
    U current
    V current
    U wind
    V wind

Target:
    GLORYS thetao

Target depths:
    0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ML"

HARMONIZED_DIR = ML_DIR / "harmonized"

CONFIG_PATH = ML_DIR / "ml_config.json"


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


TRAIN_START = "2025-07-01"
TRAIN_END = "2025-10-31"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def load_config():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dates(ds):

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


def get_train_indices(ds):

    dates = get_dates(ds)

    indices = [
        i
        for i, date in enumerate(dates)
        if TRAIN_START <= date <= TRAIN_END
    ]

    if len(indices) == 0:
        raise ValueError(
            "No training dates found."
        )

    return indices, dates


def calculate_statistics(array):

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


def compare_value(
    name,
    calculated,
    configured,
    tolerance=1e-5,
):

    difference = abs(
        calculated - configured
    )

    passed = difference <= tolerance

    status = "PASS" if passed else "FAIL"

    print(
        f"{status}: {name}"
    )

    print(
        f"       calculated = {calculated:.12f}"
    )

    print(
        f"       configured = {configured:.12f}"
    )

    print(
        f"       difference = {difference:.12e}"
    )

    return passed


# ---------------------------------------------------------------------
# MAIN VERIFICATION
# ---------------------------------------------------------------------


def main():

    print()
    print("=" * 72)
    print("OceanEmbed ML NORMALIZATION VERIFICATION")
    print("=" * 72)

    print()
    print(
        f"Training period: {TRAIN_START} -> {TRAIN_END}"
    )

    print(
        "Statistics source: persistent harmonized NetCDF files"
    )

    print(
        "Requirement: training-period data ONLY"
    )

    print()

    config = load_config()

    all_passed = True

    # ---------------------------------------------------------------
    # INPUT STATISTICS
    # ---------------------------------------------------------------

    print("=" * 72)
    print("INPUT STATISTICS")
    print("=" * 72)

    for feature in INPUT_FEATURES:

        path = FILES[feature]

        print()
        print(
            f"Processing {feature}: {path.name}"
        )

        with Dataset(path, "r") as ds:

            train_indices, dates = get_train_indices(ds)

            variable_name = VARIABLES[feature]

            variable = ds.variables[
                variable_name
            ]

            # Read ONLY training-period data.
            array = np.asarray(
                variable[
                    train_indices,
                    :,
                    :
                ],
                dtype=np.float64,
            )

        calculated_mean, calculated_std, calculated_count = (
            calculate_statistics(array)
        )

        configured = config[
            "input_statistics"
        ][feature]

        configured_mean = float(
            configured["mean"]
        )

        configured_std = float(
            configured["std"]
        )

        configured_count = int(
            configured["valid_count"]
        )

        print(
            f"Training days used: {len(train_indices)}"
        )

        print(
            f"Calculated valid count: {calculated_count}"
        )

        print(
            f"Configured valid count: {configured_count}"
        )

        if calculated_count != configured_count:

            print(
                f"FAIL: {feature} valid count mismatch."
            )

            all_passed = False

        else:

            print(
                f"PASS: {feature} valid count"
            )

        if not compare_value(
            f"{feature} mean",
            calculated_mean,
            configured_mean,
        ):

            all_passed = False

        if not compare_value(
            f"{feature} std",
            calculated_std,
            configured_std,
        ):

            all_passed = False

    # ---------------------------------------------------------------
    # TARGET STATISTICS
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("TARGET STATISTICS")
    print("=" * 72)

    target_path = FILES["thetao"]

    with Dataset(
        target_path,
        "r",
    ) as ds:

        train_indices, dates = get_train_indices(ds)

        variable = ds.variables[
            VARIABLES["thetao"]
        ]

        depth = np.asarray(
            ds.variables["depth"][:],
            dtype=np.float64,
        )

        print(
            f"Training days used: {len(train_indices)}"
        )

        print(
            f"Depth channels: {len(depth)}"
        )

        if not np.allclose(
            depth,
            np.asarray(
                TARGET_DEPTHS,
                dtype=np.float64,
            ),
            atol=1e-5,
        ):

            raise ValueError(
                "Target depths do not match OceanEmbed contract."
            )

        for depth_index, depth_value in enumerate(
            TARGET_DEPTHS
        ):

            print()
            print(
                f"Depth: {depth_value:.1f} m"
            )

            # -------------------------------------------------------
            # Read ONLY this depth and ONLY training dates.
            # -------------------------------------------------------

            array = np.asarray(
                variable[
                    train_indices,
                    depth_index,
                    :,
                    :
                ],
                dtype=np.float64,
            )

            (
                calculated_mean,
                calculated_std,
                calculated_count,
            ) = calculate_statistics(array)

            configured = config[
                "target_statistics"
            ][str(float(depth_value))]

            configured_mean = float(
                configured["mean"]
            )

            configured_std = float(
                configured["std"]
            )

            configured_count = int(
                configured["valid_count"]
            )

            print(
                f"Calculated valid count: "
                f"{calculated_count}"
            )

            print(
                f"Configured valid count: "
                f"{configured_count}"
            )

            if (
                calculated_count
                != configured_count
            ):

                print(
                    "FAIL: valid count mismatch."
                )

                all_passed = False

            else:

                print(
                    "PASS: valid count"
                )

            if not compare_value(
                f"{depth_value:.1f} m mean",
                calculated_mean,
                configured_mean,
            ):

                all_passed = False

            if not compare_value(
                f"{depth_value:.1f} m std",
                calculated_std,
                configured_std,
            ):

                all_passed = False

    # ---------------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------------

    print()
    print("=" * 72)

    if all_passed:

        print(
            "NORMALIZATION VERIFICATION PASSED"
        )

        print(
            "All configured statistics match the "
            "training-period harmonized data."
        )

    else:

        print(
            "NORMALIZATION VERIFICATION FAILED"
        )

        print(
            "At least one configured statistic "
            "does not match the training-period data."
        )

    print("=" * 72)
    print()


if __name__ == "__main__":
    main()