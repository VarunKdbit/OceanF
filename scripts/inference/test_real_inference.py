from __future__ import annotations

import sys
from pathlib import Path

import torch


# =====================================================================
# PROJECT PATH
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# =====================================================================
# IMPORTS
# =====================================================================

from ml_dataset import OceanEmbedDataset
from inference.oceanembed_inference import OceanEmbedEnsemble


# =====================================================================
# CONFIGURATION
# =====================================================================

HISTORY_DAYS = 7

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 72)
    print("OceanEmbed REAL-DATA INFERENCE TEST")
    print("=" * 72)

    print()
    print(f"Device: {DEVICE}")

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Use the ORIGINAL ml_config.json here because we want to test
    # inference against the existing December 2025 held-out test
    # dataset.
    #
    # The production inference engine itself uses the final-training
    # normalization statistics.
    # -------------------------------------------------------------

    original_config_path = __import__(
        "ml_dataset"
    ).CONFIG_PATH

    try:

        # ---------------------------------------------------------
        # Create December test dataset
        # ---------------------------------------------------------

        dataset = OceanEmbedDataset(
            split="test",
            tile_stride=32,
            return_metadata=True,
            normalize=True,
            history_days=HISTORY_DAYS,
        )

        print()
        print("TEST DATASET")
        print("-" * 72)

        print(f"Samples: {len(dataset)}")

        if len(dataset) == 0:
            raise RuntimeError(
                "December test dataset contains zero samples."
            )

        # ---------------------------------------------------------
        # Select first real sample
        # ---------------------------------------------------------

        x, x_mask, y, y_mask, metadata = dataset[0]

        print(f"Input shape:  {list(x.shape)}")
        print(f"Target shape: {list(y.shape)}")

        print()
        print("PASS: real satellite input sample loaded")

        # ---------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------

        if not torch.isfinite(x).all().item():
            raise RuntimeError(
                "Real input contains non-finite values."
            )

        print("PASS: real input is finite")

        # ---------------------------------------------------------
        # Display metadata without assuming a fixed metadata format
        # ---------------------------------------------------------

        print()
        print("Sample metadata:")

        if isinstance(metadata, dict):

            for key, value in metadata.items():
                print(f"  {key}: {value}")

        else:

            print(f"  {metadata}")

        # ---------------------------------------------------------
        # Load production ensemble
        # ---------------------------------------------------------

        print()
        print("LOADING OCEANEMBED ENSEMBLE")
        print("-" * 72)

        engine = OceanEmbedEnsemble(
            device=DEVICE
        )

        print()
        print("PASS: three-seed ensemble loaded")

        # ---------------------------------------------------------
        # Run inference
        # ---------------------------------------------------------

        print()
        print("RUNNING REAL-DATA INFERENCE")
        print("-" * 72)

        prediction = engine.predict_single(x)

        # ---------------------------------------------------------
        # Validate output
        # ---------------------------------------------------------

        expected_shape = (
            15,
            32,
            32,
        )

        if tuple(prediction.shape) != expected_shape:

            raise RuntimeError(
                f"Unexpected prediction shape: "
                f"{tuple(prediction.shape)}"
            )

        print(
            "PASS: prediction shape = "
            f"{list(prediction.shape)}"
        )

        if not torch.isfinite(prediction).all().item():

            raise RuntimeError(
                "Prediction contains non-finite values."
            )

        print(
            "PASS: prediction contains only finite values"
        )

        # ---------------------------------------------------------
        # Temperature summary
        # ---------------------------------------------------------

        print()
        print("PREDICTED SUBSURFACE TEMPERATURE")
        print("-" * 72)

        for depth_index, depth in enumerate(
            engine.depths_m
        ):

            depth_prediction = prediction[
                depth_index
            ]

            mean_temperature = (
                depth_prediction.mean().item()
            )

            minimum_temperature = (
                depth_prediction.min().item()
            )

            maximum_temperature = (
                depth_prediction.max().item()
            )

            print(
                f"{depth:>4} m : "
                f"mean={mean_temperature:7.3f} °C  "
                f"min={minimum_temperature:7.3f} °C  "
                f"max={maximum_temperature:7.3f} °C"
            )

        # ---------------------------------------------------------
        # Overall output range
        # ---------------------------------------------------------

        print()
        print(
            "Overall prediction range: "
            f"{prediction.min().item():.3f} °C "
            f"to "
            f"{prediction.max().item():.3f} °C"
        )

        # ---------------------------------------------------------
        # Final result
        # ---------------------------------------------------------

        print()
        print("=" * 72)
        print("REAL-DATA INFERENCE TEST PASSED")
        print("=" * 72)

        print()
        print(
            "OceanEmbed successfully converted a real "
            "7-day satellite-observation input into a "
            "15-depth subsurface temperature prediction."
        )

        print()
        print(
            "Output:"
        )

        print(
            "  15 depths × 32 × 32 spatial field"
        )

        print(
            "  Units: °C"
        )

    finally:

        # Restore original configuration reference.
        __import__(
            "ml_dataset"
        ).CONFIG_PATH = original_config_path

        try:
            dataset.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()