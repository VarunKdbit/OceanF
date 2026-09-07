"""
OceanEmbed Final Training Configuration Generator
==================================================

Purpose
-------
Create the final-training configuration from the independently
calculated Jul-Nov normalization statistics.

This script does NOT modify:
    data/processed/ML/ml_config.json

Final training period:
    2025-07-01 through 2025-11-30

Final held-out test period:
    2025-12-01 through 2025-12-31

There is intentionally no validation split in the final-training
configuration because the E2 architecture and hyperparameters have
already been selected using the benchmark train/validation split.

The final-training configuration is saved separately under:
    data/processed/ML/final_training/
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ML"

STATISTICS_PATH = (
    ML_DIR
    / "final_training"
    / "final_training_statistics.json"
)

OUTPUT_DIR = (
    ML_DIR
    / "final_training"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_training_config.json"
)


# ---------------------------------------------------------------------
# SCIENTIFIC CONTRACT
# ---------------------------------------------------------------------

DOMAIN = {
    "latitude_min": 5.0,
    "latitude_max": 30.0,
    "longitude_min": 45.0,
    "longitude_max": 105.0,
    "resolution": 0.25,
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


# ---------------------------------------------------------------------
# FINAL TIME SPLIT
# ---------------------------------------------------------------------

FINAL_TRAIN_START = "2025-07-01"
FINAL_TRAIN_END = "2025-11-30"

FINAL_TEST_START = "2025-12-01"
FINAL_TEST_END = "2025-12-31"


# ---------------------------------------------------------------------
# MODEL CONTRACT
# ---------------------------------------------------------------------

HISTORY_DAYS = 7

NUM_INPUT_FEATURES = 7

INPUT_CHANNELS = (
    HISTORY_DAYS * NUM_INPUT_FEATURES
)

LATENT_CHANNELS = 128

OUTPUT_CHANNELS = 15

INPUT_TILE_SIZE = 64

OUTPUT_TILE_SIZE = 32

CONTEXT_PIXELS = 16

TILE_STRIDE = 32


# ---------------------------------------------------------------------
# TRAINING CONTRACT
# ---------------------------------------------------------------------

LOSS = {
    "name": "masked_huber",
    "delta": 1.0,
}

OPTIMIZER = {
    "name": "Adam",
    "learning_rate": 0.001,
}

TRAINING = {
    "batch_size": 4,
    "epochs": 20,
    "gradient_clip": 1.0,
    "num_workers": 0,
    "early_stopping": False,
}


# ---------------------------------------------------------------------
# ENSEMBLE CONTRACT
# ---------------------------------------------------------------------

ENSEMBLE = {
    "enabled": True,
    "type": "three_seed_arithmetic_mean",
    "seeds": [
        42,
        123,
        2024,
    ],
    "size": 3,
}


# ---------------------------------------------------------------------
# NORMALIZATION CONTRACT
# ---------------------------------------------------------------------

NORMALIZATION = {
    "inputs": "final_training_period_mean_std",
    "target": "final_training_period_mean_std_by_depth",
    "statistics_file": str(
        STATISTICS_PATH.relative_to(PROJECT_ROOT)
    ),
}


# ---------------------------------------------------------------------
# MISSING-DATA CONTRACT
# ---------------------------------------------------------------------

MISSING_DATA = {
    "input_missing_values": (
        "filled_with_zero_after_normalization"
    ),
    "input_validity_mask": "retained",
    "target_missing_values": "retained",
    "target_validity_mask": "retained",
    "target_loss": "masked",
}


# ---------------------------------------------------------------------
# SPATIAL CONTRACT
# ---------------------------------------------------------------------

SPATIAL_HARMONIZATION = {
    "method": "linear_interpolation",
    "target_resolution": 0.25,
}


# ---------------------------------------------------------------------
# SPLIT CONTRACT
# ---------------------------------------------------------------------

SPLIT_STRATEGY = {
    "type": "chronological",
    "validation": "none",
    "final_test_is_held_out": True,
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def load_statistics():
    """
    Load the independently calculated final-training statistics.
    """

    if not STATISTICS_PATH.exists():
        raise FileNotFoundError(
            "Final training statistics file not found:\n"
            f"{STATISTICS_PATH}"
        )

    with open(
        STATISTICS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_statistics(statistics):
    """
    Validate the statistics file before creating the config.
    """

    if statistics.get("project") != "OceanF":
        raise ValueError(
            "Statistics project is not OceanF."
        )

    if statistics.get("experiment") != "OceanEmbed-CNN":
        raise ValueError(
            "Statistics experiment is not OceanEmbed-CNN."
        )

    training_period = statistics.get(
        "training_period",
        {},
    )

    if training_period.get("start") != FINAL_TRAIN_START:
        raise ValueError(
            "Statistics training start does not match "
            "the final training contract."
        )

    if training_period.get("end") != FINAL_TRAIN_END:
        raise ValueError(
            "Statistics training end does not match "
            "the final training contract."
        )

    input_features = statistics.get(
        "input_features",
        [],
    )

    if input_features != INPUT_FEATURES:
        raise ValueError(
            "Statistics input features do not match "
            "the OceanEmbed scientific contract."
        )

    target_depths = statistics.get(
        "target_depths_m",
        [],
    )

    if target_depths != [
        float(depth)
        for depth in TARGET_DEPTHS
    ]:
        raise ValueError(
            "Statistics target depths do not match "
            "the OceanEmbed scientific contract."
        )

    input_statistics = statistics.get(
        "input_statistics",
        {},
    )

    if len(input_statistics) != 7:
        raise ValueError(
            "Expected statistics for exactly "
            "7 input features."
        )

    target_statistics = statistics.get(
        "target_statistics",
        {},
    )

    if len(target_statistics) != 15:
        raise ValueError(
            "Expected statistics for exactly "
            "15 target depths."
        )

    for feature in INPUT_FEATURES:

        if feature not in input_statistics:
            raise ValueError(
                f"Missing input statistics for {feature}."
            )

        entry = input_statistics[feature]

        for key in [
            "mean",
            "std",
            "valid_count",
        ]:

            if key not in entry:
                raise ValueError(
                    f"Missing '{key}' for input "
                    f"feature {feature}."
                )

        if float(entry["std"]) <= 0:
            raise ValueError(
                f"Non-positive standard deviation "
                f"for input feature {feature}."
            )

    for depth in TARGET_DEPTHS:

        key = str(float(depth))

        if key not in target_statistics:
            raise ValueError(
                f"Missing target statistics for "
                f"depth {depth} m."
            )

        entry = target_statistics[key]

        for field in [
            "mean",
            "std",
            "valid_count",
        ]:

            if field not in entry:
                raise ValueError(
                    f"Missing '{field}' for "
                    f"target depth {depth} m."
                )

        if float(entry["std"]) <= 0:
            raise ValueError(
                f"Non-positive standard deviation "
                f"for target depth {depth} m."
            )


# ---------------------------------------------------------------------
# CONFIGURATION CREATION
# ---------------------------------------------------------------------


def create_config(statistics):
    """
    Build the complete final-training configuration.
    """

    config = {
        "project": "OceanF",

        "experiment": "OceanEmbed-CNN",

        "experiment_variant": (
            "E2_7day_retrospective_final_training"
        ),

        "purpose": (
            "Final training configuration using "
            "2025-07-01 through 2025-11-30"
        ),

        "domain": DOMAIN,

        "time": {
            "final_training_start": FINAL_TRAIN_START,
            "final_training_end": FINAL_TRAIN_END,
            "final_test_start": FINAL_TEST_START,
            "final_test_end": FINAL_TEST_END,
            "validation": None,
        },

        "input_features": INPUT_FEATURES,

        "target_variable": TARGET_VARIABLE,

        "target_depths_m": TARGET_DEPTHS,

        "model": {
            "architecture": "OceanEmbed-CNN",
            "history_days": HISTORY_DAYS,
            "input_channels": INPUT_CHANNELS,
            "latent_channels": LATENT_CHANNELS,
            "output_channels": OUTPUT_CHANNELS,
            "input_tile_size": INPUT_TILE_SIZE,
            "output_tile_size": OUTPUT_TILE_SIZE,
            "context_pixels": CONTEXT_PIXELS,
            "tile_stride": TILE_STRIDE,
        },

        "loss": LOSS,

        "optimizer": OPTIMIZER,

        "training": TRAINING,

        "normalization": NORMALIZATION,

        "missing_data": MISSING_DATA,

        "spatial_harmonization": SPATIAL_HARMONIZATION,

        "split_strategy": SPLIT_STRATEGY,

        "ensemble": ENSEMBLE,

        "input_statistics": (
            statistics["input_statistics"]
        ),

        "target_statistics": (
            statistics["target_statistics"]
        ),

        "source_statistics": {
            "file": str(
                STATISTICS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "training_start": FINAL_TRAIN_START,
            "training_end": FINAL_TRAIN_END,
            "statistics_are_training_only": True,
        },

        "data_policy": {
            "final_test_data_is_not_used_for_training": True,
            "final_test_data_is_not_used_for_normalization": True,
            "final_test_data_is_not_used_for_model_selection": True,
        },
    }

    return config


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():

    print()
    print("=" * 72)
    print("OceanEmbed FINAL TRAINING CONFIGURATION")
    print("=" * 72)

    print()
    print(
        f"Final training:"
        f" {FINAL_TRAIN_START} -> {FINAL_TRAIN_END}"
    )

    print(
        f"Final test:"
        f" {FINAL_TEST_START} -> {FINAL_TEST_END}"
    )

    print(
        "Validation: none"
    )

    print()
    print(
        f"Statistics file:"
        f" {STATISTICS_PATH}"
    )

    # -------------------------------------------------------------
    # Load statistics.
    # -------------------------------------------------------------

    statistics = load_statistics()

    print(
        "PASS: statistics file found"
    )

    # -------------------------------------------------------------
    # Validate statistics.
    # -------------------------------------------------------------

    validate_statistics(
        statistics
    )

    print(
        "PASS: statistics validated"
    )

    # -------------------------------------------------------------
    # Create configuration.
    # -------------------------------------------------------------

    config = create_config(
        statistics
    )

    # -------------------------------------------------------------
    # Save configuration.
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
            config,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final output.
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL TRAINING CONFIGURATION CREATED")
    print("=" * 72)

    print()
    print(
        f"Input channels: {INPUT_CHANNELS}"
    )

    print(
        f"Target channels: {OUTPUT_CHANNELS}"
    )

    print(
        f"History days: {HISTORY_DAYS}"
    )

    print(
        f"Latent channels: {LATENT_CHANNELS}"
    )

    print(
        f"Training period:"
        f" {FINAL_TRAIN_START} -> {FINAL_TRAIN_END}"
    )

    print(
        f"Final test:"
        f" {FINAL_TEST_START} -> {FINAL_TEST_END}"
    )

    print()
    print(
        "December test data is explicitly held out."
    )

    print(
        "Existing ml_config.json was NOT modified."
    )

    print()
    print(
        f"Saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 72)
    print()


if __name__ == "__main__":
    main()