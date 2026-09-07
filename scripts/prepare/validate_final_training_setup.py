from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ML"

FINAL_TRAINING_DIR = ML_DIR / "final_training"

CONFIG_PATH = FINAL_TRAINING_DIR / "final_training_config.json"
STATISTICS_PATH = FINAL_TRAINING_DIR / "final_training_statistics.json"

HARMONIZED_DIR = ML_DIR / "harmonized"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import ml_dataset
from ml_dataset import OceanEmbedDataset
from oceanembed_model import OceanEmbedCNN


# =====================================================================
# FROZEN SCIENTIFIC CONTRACT
# =====================================================================

EXPECTED_PROJECT = "OceanF"
EXPECTED_EXPERIMENT = "OceanEmbed-CNN"

EXPECTED_TRAIN_START = "2025-07-01"
EXPECTED_TRAIN_END = "2025-11-30"

EXPECTED_TEST_START = "2025-12-01"
EXPECTED_TEST_END = "2025-12-31"

EXPECTED_FEATURES = [
    "sst",
    "sss",
    "sla",
    "uo",
    "vo",
    "u_wind",
    "v_wind",
]

EXPECTED_DEPTHS = [
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

EXPECTED_HISTORY_DAYS = 7

EXPECTED_INPUT_CHANNELS = 49
EXPECTED_OUTPUT_CHANNELS = 15

EXPECTED_INPUT_SHAPE = (49, 64, 64)
EXPECTED_TARGET_SHAPE = (15, 32, 32)

EXPECTED_TRAIN_SAMPLES = 2940

EXPECTED_HARMONIZED_FILES = [
    "SST_harmonized.nc",
    "SSS_harmonized.nc",
    "SLA_harmonized.nc",
    "Currents_harmonized.nc",
    "Winds_harmonized.nc",
    "SubsurfaceTemp_harmonized.nc",
]


# =====================================================================
# HELPERS
# =====================================================================

def require(condition: bool, message: str) -> None:

    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:

    require(
        path.exists(),
        f"Required file not found: {path}",
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def section(title: str) -> None:

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def close_dataset(dataset) -> None:

    if dataset is None:
        return

    try:
        dataset.close()
    except Exception:
        pass


# =====================================================================
# CONFIGURATION
# =====================================================================

def validate_configuration(config: dict[str, Any]) -> None:

    section("FINAL TRAINING CONFIGURATION")

    require(
        config.get("project") == EXPECTED_PROJECT,
        f"Unexpected project: {config.get('project')}",
    )

    print(f"PASS: project = {config['project']}")

    require(
        config.get("experiment") == EXPECTED_EXPERIMENT,
        f"Unexpected experiment: {config.get('experiment')}",
    )

    print(f"PASS: experiment = {config['experiment']}")

    time_config = config["time"]

    require(
        time_config["train_start"] == EXPECTED_TRAIN_START,
        "Incorrect training start date.",
    )

    require(
        time_config["train_end"] == EXPECTED_TRAIN_END,
        "Incorrect training end date.",
    )

    require(
        time_config["test_start"] == EXPECTED_TEST_START,
        "Incorrect test start date.",
    )

    require(
        time_config["test_end"] == EXPECTED_TEST_END,
        "Incorrect test end date.",
    )

    require(
        time_config["validation_start"] is None,
        "Final training must not contain validation.",
    )

    require(
        time_config["validation_end"] is None,
        "Final training must not contain validation.",
    )

    print("PASS: training/test dates and no-validation policy")

    require(
        config["input_features"] == EXPECTED_FEATURES,
        "Input feature list does not match frozen contract.",
    )

    print("PASS: 7 input features")

    require(
        config["target_depths_m"] == EXPECTED_DEPTHS,
        "Target depth list does not match frozen contract.",
    )

    print("PASS: 15 target depths")


# =====================================================================
# NORMALIZATION STATISTICS
# =====================================================================

def validate_statistics(
    config: dict[str, Any],
    statistics: dict[str, Any],
) -> None:

    section("NORMALIZATION STATISTICS")

    period = statistics.get("training_period", {})

    if period:

        require(
            period.get("start") == EXPECTED_TRAIN_START,
            "Statistics do not start on July 1, 2025.",
        )

        require(
            period.get("end") == EXPECTED_TRAIN_END,
            "Statistics include data after November 30, 2025.",
        )

    statistics_type = statistics.get("statistics_type")

    if statistics_type is not None:

        require(
            statistics_type == "final_training_normalization",
            f"Unexpected statistics type: {statistics_type}",
        )

    print("PASS: statistics use Jul-Nov final training period")

    input_statistics = statistics["input_statistics"]

    require(
        len(input_statistics) == 7,
        "Expected 7 input normalization statistics.",
    )

    for feature in EXPECTED_FEATURES:

        require(
            feature in input_statistics,
            f"Missing statistics for input feature: {feature}",
        )

    print("PASS: 7 input normalization statistics")

    target_statistics = statistics["target_statistics"]

    require(
        len(target_statistics) == 15,
        "Expected 15 target normalization statistics.",
    )

    for depth in EXPECTED_DEPTHS:

        key = str(float(depth))

        if key not in target_statistics:
            key = str(depth)

        require(
            key in target_statistics,
            f"Missing statistics for target depth: {depth}m",
        )

    print("PASS: 15 target normalization statistics")

    require(
        config["input_statistics"] == input_statistics,
        "Config input statistics differ from statistics file.",
    )

    print("PASS: input statistics exactly match statistics file")

    require(
        config["target_statistics"] == target_statistics,
        "Config target statistics differ from statistics file.",
    )

    print("PASS: target statistics exactly match statistics file")


# =====================================================================
# HARMONIZED DATA
# =====================================================================

def validate_harmonized_files() -> None:

    section("HARMONIZED DATA FILES")

    for filename in EXPECTED_HARMONIZED_FILES:

        path = HARMONIZED_DIR / filename

        require(
            path.exists(),
            f"Missing harmonized file: {path}",
        )

        print(f"PASS: {filename}")


# =====================================================================
# FINAL TEST ISOLATION
# =====================================================================

def validate_test_isolation(
    config: dict[str, Any],
    statistics: dict[str, Any],
) -> None:

    section("FINAL TEST ISOLATION")

    time_config = config["time"]

    require(
        time_config["train_end"] < time_config["test_start"],
        "Training and test periods overlap.",
    )

    require(
        time_config["train_end"] == "2025-11-30",
        "Training does not end before December.",
    )

    require(
        time_config["test_start"] == "2025-12-01",
        "Final test does not begin in December.",
    )

    print("PASS: December is excluded from training")

    period = statistics.get("training_period", {})

    if period:

        require(
            period.get("end") == EXPECTED_TRAIN_END,
            "December appears in normalization period.",
        )

    print("PASS: December is excluded from normalization")

    require(
        time_config["validation_start"] is None,
        "Validation period exists.",
    )

    require(
        time_config["validation_end"] is None,
        "Validation period exists.",
    )

    print("PASS: December is excluded from model selection")


# =====================================================================
# DATASET VALIDATION
# =====================================================================

def validate_dataset() -> OceanEmbedDataset:

    section("ML DATASET VALIDATION")

    original_config_path = ml_dataset.CONFIG_PATH

    dataset = None

    try:

        # ml_dataset.py is frozen.
        # Temporarily point it to final_training_config.json.
        ml_dataset.CONFIG_PATH = CONFIG_PATH

        dataset = OceanEmbedDataset(
            split="train",
            tile_stride=32,
            return_metadata=True,
            normalize=True,
            history_days=EXPECTED_HISTORY_DAYS,
        )

        print("PASS: final configuration loaded by ml_dataset.py")

        print(f"Final training samples: {len(dataset)}")

        require(
            len(dataset) == EXPECTED_TRAIN_SAMPLES,
            (
                f"Unexpected final training sample count: "
                f"{len(dataset)}. Expected {EXPECTED_TRAIN_SAMPLES}."
            ),
        )

        print("PASS: final training dataset contains expected samples")

        # -------------------------------------------------------------
        # Get one real sample.
        # -------------------------------------------------------------

        sample = dataset[0]

        require(
            len(sample) == 5,
            f"Unexpected dataset sample tuple length: {len(sample)}",
        )

        x, x_mask, y, y_mask, _ = sample

        # -------------------------------------------------------------
        # Shapes
        # -------------------------------------------------------------

        require(
            tuple(x.shape) == EXPECTED_INPUT_SHAPE,
            f"Unexpected input shape: {tuple(x.shape)}",
        )

        print(f"PASS: input shape = {list(x.shape)}")

        require(
            tuple(y.shape) == EXPECTED_TARGET_SHAPE,
            f"Unexpected target shape: {tuple(y.shape)}",
        )

        print(f"PASS: target shape = {list(y.shape)}")

        require(
            tuple(x_mask.shape) == EXPECTED_INPUT_SHAPE,
            "Input mask shape mismatch.",
        )

        require(
            tuple(y_mask.shape) == EXPECTED_TARGET_SHAPE,
            "Target mask shape mismatch.",
        )

        print("PASS: mask shapes")

        # -------------------------------------------------------------
        # Input
        # -------------------------------------------------------------

        require(
            torch.isfinite(x).all().item(),
            "Input contains non-finite values.",
        )

        print("PASS: input contains only finite values")

        # -------------------------------------------------------------
        # Target
        # -------------------------------------------------------------
        #
        # y is ALLOWED to contain NaNs.
        #
        # y_mask identifies valid cells.
        #
        # Convert mask explicitly to bool because the frozen dataset
        # does not guarantee the returned mask dtype is bool.
        # -------------------------------------------------------------

        target_mask = y_mask.to(torch.bool)

        require(
            target_mask.any().item(),
            "Target mask contains no valid cells.",
        )

        valid_target = y[target_mask]

        require(
            valid_target.numel() > 0,
            "No valid target cells found.",
        )

        require(
            torch.isfinite(valid_target).all().item(),
            "Valid target cells contain non-finite values.",
        )

        print("PASS: valid target cells are finite")

        invalid_target = y[~target_mask]

        if invalid_target.numel() > 0:

            invalid_count = int(invalid_target.numel())

            print(
                "INFO: invalid target cells are retained for masked loss "
                f"({invalid_count} cells)"
            )

        else:

            print("INFO: sample contains no invalid target cells")

        # -------------------------------------------------------------
        # Mask sanity
        # -------------------------------------------------------------

        require(
            x_mask.numel() == x.numel(),
            "Input mask size mismatch.",
        )

        require(
            y_mask.numel() == y.numel(),
            "Target mask size mismatch.",
        )

        print("PASS: mask element counts")

        return dataset

    finally:

        ml_dataset.CONFIG_PATH = original_config_path


# =====================================================================
# MODEL VALIDATION
# =====================================================================

def validate_model() -> None:

    section("OCEANEMBED MODEL VALIDATION")

    model = OceanEmbedCNN(
        input_channels=EXPECTED_INPUT_CHANNELS,
        latent_channels=128,
        output_channels=EXPECTED_OUTPUT_CHANNELS,
    )

    model.eval()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    require(
        trainable_parameters > 0,
        "Model has no trainable parameters.",
    )

    print("PASS: model contains trainable parameters")

    test_input = torch.randn(
        1,
        EXPECTED_INPUT_CHANNELS,
        64,
        64,
        dtype=torch.float32,
    )

    with torch.no_grad():

        output = model(test_input)

    expected_output = (
        1,
        EXPECTED_OUTPUT_CHANNELS,
        32,
        32,
    )

    require(
        tuple(output.shape) == expected_output,
        (
            f"Unexpected model output shape: "
            f"{tuple(output.shape)}"
        ),
    )

    print(
        f"PASS: model output shape = {list(output.shape)}"
    )

    require(
        torch.isfinite(output).all().item(),
        "Model output contains non-finite values.",
    )

    print("PASS: model output is finite")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 72)
    print("OceanEmbed FINAL TRAINING SETUP VALIDATION")
    print("=" * 72)

    print()
    print("This validation does not modify frozen ML files.")

    print()
    print("Configuration:")
    print(CONFIG_PATH)

    print()
    print("Statistics:")
    print(STATISTICS_PATH)

    # -------------------------------------------------------------
    # Load files
    # -------------------------------------------------------------

    config = load_json(CONFIG_PATH)
    statistics = load_json(STATISTICS_PATH)

    print()
    print("PASS: final configuration loaded")
    print("PASS: final statistics loaded")

    # -------------------------------------------------------------
    # Validate everything
    # -------------------------------------------------------------

    validate_configuration(config)

    validate_statistics(
        config,
        statistics,
    )

    validate_harmonized_files()

    validate_test_isolation(
        config,
        statistics,
    )

    dataset = validate_dataset()

    try:

        validate_model()

    finally:

        close_dataset(dataset)

    # -------------------------------------------------------------
    # Final success
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL TRAINING SETUP VALIDATION PASSED")
    print("=" * 72)

    print()
    print("Confirmed:")

    print("  - Jul-Nov 2025 final training period")
    print("  - December 2025 held out")
    print("  - Jul-Nov-only normalization statistics")
    print("  - 7 input features")
    print("  - 7-day retrospective window")
    print("  - 49 input channels")
    print("  - 15 target depths")
    print("  - 64x64 input tiles")
    print("  - 32x32 target tiles")
    print("  - 2,940 final training samples")
    print("  - Masked target handling")
    print("  - OceanEmbed-CNN forward pass")
    print()
    print("No frozen ML files were modified.")


if __name__ == "__main__":
    main()