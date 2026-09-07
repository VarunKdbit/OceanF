from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr


# =====================================================================
# PROJECT PATH
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from inference.oceanembed_inference import (
    OceanEmbedEnsemble,
    INPUT_FEATURES,
    HISTORY_DAYS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_CHANNELS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    DEPTHS_M,
)


# =====================================================================
# DATA PATHS
# =====================================================================

HARMONIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "harmonized"
)


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


# =====================================================================
# TARGET TEST DATE / TILE
# =====================================================================

# December is the held-out test period.
TARGET_DATE = np.datetime64("2025-12-01")

# Same first tile used previously for real-data validation.
TILE_ROW = 0
TILE_COL = 0

# 64x64 input tile.
INPUT_START_ROW = TILE_ROW * 32
INPUT_START_COL = TILE_COL * 32


# =====================================================================
# HELPERS
# =====================================================================

def find_coordinate(
    dataset: xr.Dataset,
    candidates: list[str],
) -> str:

    for candidate in candidates:

        if candidate in dataset.coords:
            return candidate

        if candidate in dataset.dims:
            return candidate

    raise ValueError(
        f"Could not find coordinate from candidates: {candidates}. "
        f"Available coordinates: {list(dataset.coords)}"
    )


def find_time_coordinate(
    dataset: xr.Dataset,
) -> str:

    return find_coordinate(
        dataset,
        ["time", "valid_time", "date"],
    )


def find_latitude_coordinate(
    dataset: xr.Dataset,
) -> str:

    return find_coordinate(
        dataset,
        ["latitude", "lat"],
    )


def find_longitude_coordinate(
    dataset: xr.Dataset,
) -> str:

    return find_coordinate(
        dataset,
        ["longitude", "lon"],
    )


def extract_feature_window(
    dataset: xr.Dataset,
    variable: str,
    time_indices: list[int],
) -> torch.Tensor:

    time_name = find_time_coordinate(dataset)
    lat_name = find_latitude_coordinate(dataset)
    lon_name = find_longitude_coordinate(dataset)

    data = dataset[variable]

    # Select the seven required dates.
    selected = data.isel(
        {
            time_name: time_indices,
        }
    )

    # Select the 64x64 spatial input tile.
    selected = selected.isel(
        {
            lat_name: slice(
                INPUT_START_ROW,
                INPUT_START_ROW + INPUT_HEIGHT,
            ),
            lon_name: slice(
                INPUT_START_COL,
                INPUT_START_COL + INPUT_WIDTH,
            ),
        }
    )

    values = selected.values.astype(
        np.float32
    )

    # Remove singleton dimensions if present.
    values = np.squeeze(values)

    expected_shape = (
        HISTORY_DAYS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
    )

    if values.shape != expected_shape:

        raise RuntimeError(
            f"Variable {variable} produced shape "
            f"{values.shape}; expected {expected_shape}."
        )

    return torch.from_numpy(values)


# =====================================================================
# MAIN TEST
# =====================================================================

def main() -> None:

    print()
    print("=" * 72)
    print("OceanEmbed REAL RAW-DATA PRODUCTION INFERENCE TEST")
    print("=" * 72)

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    # -------------------------------------------------------------
    # Load production inference engine
    # -------------------------------------------------------------

    engine = OceanEmbedEnsemble(
        device=device
    )

    print()
    print(
        "PASS: production inference engine loaded"
    )

    # -------------------------------------------------------------
    # Open all required datasets
    # -------------------------------------------------------------

    datasets = {}

    try:

        for feature in INPUT_FEATURES:

            filename = FEATURE_FILES[feature]

            path = (
                HARMONIZED_DIR
                / filename
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Harmonized file not found: {path}"
                )

            if feature in datasets:
                continue

            datasets[feature] = xr.open_dataset(
                path
            )

            print(
                f"Loaded {feature}: {filename}"
            )

        print()
        print(
            "PASS: required harmonized datasets opened"
        )

        # ---------------------------------------------------------
        # Determine the seven-day retrospective window.
        # ---------------------------------------------------------

        reference_dataset = datasets["sst"]

        time_name = find_time_coordinate(
            reference_dataset
        )

        times = reference_dataset[
            time_name
        ].values

        target_matches = np.where(
            times == TARGET_DATE
        )[0]

        if len(target_matches) != 1:

            raise RuntimeError(
                f"Could not find unique target date "
                f"{TARGET_DATE}. Matches: {target_matches}"
            )

        target_index = int(
            target_matches[0]
        )

        history_start = (
            target_index
            - HISTORY_DAYS
            + 1
        )

        if history_start < 0:

            raise RuntimeError(
                "Not enough historical days available "
                "for the requested 7-day window."
            )

        time_indices = list(
            range(
                history_start,
                target_index + 1,
            )
        )

        selected_dates = [
            times[index]
            for index in time_indices
        ]

        print()
        print(
            "Target date:",
            str(TARGET_DATE),
        )

        print(
            "7-day retrospective window:"
        )

        for date in selected_dates:
            print(
                "  ",
                str(date),
            )

        # ---------------------------------------------------------
        # Extract seven days for every feature.
        # ---------------------------------------------------------

        raw_window = {}

        for feature in INPUT_FEATURES:

            variable = FEATURE_VARIABLES[feature]

            # Currents contains both uo and vo.
            if feature in ("uo", "vo"):

                dataset = datasets[feature]

            else:

                dataset = datasets[feature]

            values = extract_feature_window(
                dataset,
                variable,
                time_indices,
            )

            raw_window[feature] = values

            print(
                f"PASS: {feature} raw window = "
                f"{list(values.shape)}"
            )

        # ---------------------------------------------------------
        # Check raw data.
        # ---------------------------------------------------------

        for feature, values in raw_window.items():

            finite_count = torch.isfinite(
                values
            ).sum().item()

            total_count = values.numel()

            if finite_count == 0:

                raise RuntimeError(
                    f"{feature} contains no finite values."
                )

            print(
                f"  {feature}: "
                f"{finite_count}/{total_count} finite"
            )

        print()
        print(
            "PASS: real 7-day raw input window extracted"
        )

        # ---------------------------------------------------------
        # Normalize using FINAL Jul-Nov statistics.
        # ---------------------------------------------------------

        normalized_input = (
            engine.normalize_input_window(
                raw_window
            )
        )

        print()
        print(
            "PASS: real raw data normalized using "
            "final Jul-Nov statistics"
        )

        print(
            "Normalized input shape:",
            list(normalized_input.shape),
        )

        # ---------------------------------------------------------
        # Run production ensemble.
        # ---------------------------------------------------------

        prediction = engine.predict_single(
            normalized_input
        )

        expected_shape = (
            OUTPUT_CHANNELS,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
        )

        if tuple(prediction.shape) != expected_shape:

            raise RuntimeError(
                "Unexpected prediction shape: "
                f"{tuple(prediction.shape)}"
            )

        print()
        print(
            "PASS: production ensemble inference completed"
        )

        print(
            "Prediction shape:",
            list(prediction.shape),
        )

        # ---------------------------------------------------------
        # Validate prediction.
        # ---------------------------------------------------------

        if not torch.isfinite(
            prediction
        ).all().item():

            raise RuntimeError(
                "Prediction contains non-finite values."
            )

        print(
            "PASS: prediction is fully finite"
        )

        # ---------------------------------------------------------
        # Print depth-wise summary.
        # ---------------------------------------------------------

        prediction_cpu = (
            prediction.detach()
            .cpu()
            .numpy()
        )

        print()
        print("=" * 72)
        print("PREDICTION SUMMARY")
        print("=" * 72)

        for depth_index, depth in enumerate(
            DEPTHS_M
        ):

            depth_field = (
                prediction_cpu[
                    depth_index
                ]
            )

            print(
                f"{depth:>4} m | "
                f"mean = {depth_field.mean():8.3f} °C | "
                f"min = {depth_field.min():8.3f} °C | "
                f"max = {depth_field.max():8.3f} °C"
            )

        print()
        print(
            "Overall prediction range: "
            f"{prediction_cpu.min():.3f} °C "
            f"to "
            f"{prediction_cpu.max():.3f} °C"
        )

        print()
        print("=" * 72)
        print(
            "REAL RAW-DATA PRODUCTION INFERENCE TEST PASSED"
        )
        print("=" * 72)

        print()
        print(
            "OceanEmbed successfully converted real "
            "7-day satellite-observation data into "
            "15-depth subsurface temperature predictions."
        )

    finally:

        for dataset in datasets.values():

            dataset.close()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()