"""
OceanEmbed-CNN Real Dataset Integration Test
=============================================

Tests the OceanEmbed-CNN using an actual batch from the
OceanEmbed ML Dataset.

This verifies:

    NetCDF data
        ↓
    OceanEmbedDataset
        ↓
    PyTorch DataLoader
        ↓
    [B, 49, 64, 64]
        ↓
    OceanEmbed-CNN
        ↓
    [B, 15, 32, 32]

This is an integration test only.
No model training is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# MAKE scripts IMPORTABLE
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------
# IMPORT OCEANEMBED COMPONENTS
# ---------------------------------------------------------------------

from ml_dataset import OceanEmbedDataset
from oceanembed_model import (
    OceanEmbedCNN,
    count_parameters,
)


# ---------------------------------------------------------------------
# TEST SETTINGS
# ---------------------------------------------------------------------

BATCH_SIZE = 2

TILE_STRIDE = 32

NUM_WORKERS = 0


# ---------------------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------------------


def main():

    print()
    print("=" * 72)
    print("OceanEmbed-CNN REAL DATASET INTEGRATION TEST")
    print("=" * 72)

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = torch.device("cpu")

    print()
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # Create real OceanEmbed training dataset.
    # ---------------------------------------------------------------

    print()
    print("Creating OceanEmbed training dataset...")

    dataset = OceanEmbedDataset(
        split="train",
        tile_stride=TILE_STRIDE,
        return_metadata=True,
        normalize=True,
    )

    print(
        f"Dataset samples: {len(dataset):,}"
    )

    # ---------------------------------------------------------------
    # Create DataLoader.
    # ---------------------------------------------------------------

    print()
    print("Creating DataLoader...")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # ---------------------------------------------------------------
    # Get one real batch.
    # ---------------------------------------------------------------

    print()
    print("Loading one real batch...")

    batch = next(iter(loader))

    (
        x,
        x_mask,
        y,
        y_mask,
        metadata,
    ) = batch

    print()
    print(
        f"x shape       : {tuple(x.shape)}"
    )

    print(
        f"x mask shape  : {tuple(x_mask.shape)}"
    )

    print(
        f"y shape       : {tuple(y.shape)}"
    )

    print(
        f"y mask shape  : {tuple(y_mask.shape)}"
    )

    # ---------------------------------------------------------------
    # Verify real dataset shapes.
    # ---------------------------------------------------------------

    expected_x_shape = (
        BATCH_SIZE,
        49,
        64,
        64,
    )

    expected_x_mask_shape = (
        BATCH_SIZE,
        49,
        64,
        64,
    )

    expected_y_shape = (
        BATCH_SIZE,
        15,
        32,
        32,
    )

    expected_y_mask_shape = (
        BATCH_SIZE,
        15,
        32,
        32,
    )

    assert tuple(x.shape) == expected_x_shape, (
        f"Input shape mismatch: "
        f"expected {expected_x_shape}, "
        f"got {tuple(x.shape)}"
    )

    assert tuple(x_mask.shape) == expected_x_mask_shape, (
        f"Input mask shape mismatch: "
        f"expected {expected_x_mask_shape}, "
        f"got {tuple(x_mask.shape)}"
    )

    assert tuple(y.shape) == expected_y_shape, (
        f"Target shape mismatch: "
        f"expected {expected_y_shape}, "
        f"got {tuple(y.shape)}"
    )

    assert tuple(y_mask.shape) == expected_y_mask_shape, (
        f"Target mask shape mismatch: "
        f"expected {expected_y_mask_shape}, "
        f"got {tuple(y_mask.shape)}"
    )

    print()
    print(
        "PASS: real dataset tensor shapes."
    )

    # ---------------------------------------------------------------
    # Verify input tensors.
    # ---------------------------------------------------------------

    assert x.dtype == torch.float32

    assert x_mask.dtype == torch.float32

    assert y.dtype == torch.float32

    assert y_mask.dtype == torch.float32

    print(
        "PASS: tensor dtypes are float32."
    )

    # ---------------------------------------------------------------
    # Inputs must contain no NaN/Inf.
    # ---------------------------------------------------------------

    assert torch.isfinite(x).all()

    assert torch.isfinite(x_mask).all()

    print(
        "PASS: input and input-mask tensors contain "
        "no NaN/Inf."
    )

    # ---------------------------------------------------------------
    # Masks must contain only 0 and 1.
    # ---------------------------------------------------------------

    unique_input_mask = torch.unique(
        x_mask
    )

    unique_target_mask = torch.unique(
        y_mask
    )

    print()
    print(
        f"Input mask values  : "
        f"{unique_input_mask.tolist()}"
    )

    print(
        f"Target mask values : "
        f"{unique_target_mask.tolist()}"
    )

    assert torch.all(
        (unique_input_mask == 0)
        | (unique_input_mask == 1)
    )

    assert torch.all(
        (unique_target_mask == 0)
        | (unique_target_mask == 1)
    )

    print(
        "PASS: masks contain only 0 and 1."
    )

    # ---------------------------------------------------------------
    # Verify target validity behavior.
    #
    # Valid target cells must be finite.
    # Missing target cells are represented by NaN.
    # ---------------------------------------------------------------

    valid_target = y_mask > 0

    if valid_target.any():

        assert torch.isfinite(
            y[valid_target]
        ).all()

        print(
            "PASS: all valid target values are finite."
        )

    missing_target = y_mask == 0

    if missing_target.any():

        assert torch.isnan(
            y[missing_target]
        ).all()

        print(
            "PASS: missing target values remain NaN."
        )

    # ---------------------------------------------------------------
    # Create model.
    # ---------------------------------------------------------------

    print()
    print("Creating OceanEmbed-CNN...")

    model = OceanEmbedCNN()

    model = model.to(device)

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ---------------------------------------------------------------
    # Move real input to device.
    # ---------------------------------------------------------------

    x_device = x.to(device)

    # ---------------------------------------------------------------
    # Forward pass.
    #
    # IMPORTANT:
    # This is NOT training.
    # ---------------------------------------------------------------

    model.eval()

    print()
    print("Running real-data forward pass...")

    with torch.no_grad():

        prediction = model(
            x_device
        )

    # ---------------------------------------------------------------
    # Verify prediction shape.
    # ---------------------------------------------------------------

    expected_prediction_shape = (
        BATCH_SIZE,
        15,
        32,
        32,
    )

    print()
    print(
        f"Prediction shape: "
        f"{tuple(prediction.shape)}"
    )

    assert (
        tuple(prediction.shape)
        == expected_prediction_shape
    ), (
        f"Prediction shape mismatch: "
        f"expected {expected_prediction_shape}, "
        f"got {tuple(prediction.shape)}"
    )

    print(
        "PASS: model output matches target shape."
    )

    # ---------------------------------------------------------------
    # Verify prediction numerical stability.
    # ---------------------------------------------------------------

    assert torch.isfinite(
        prediction
    ).all()

    print(
        "PASS: model predictions contain "
        "no NaN/Inf."
    )

    # ---------------------------------------------------------------
    # Print basic ranges.
    # ---------------------------------------------------------------

    print()
    print(
        f"Input range      : "
        f"{float(x.min()):.6f} -> "
        f"{float(x.max()):.6f}"
    )

    print(
        f"Prediction range : "
        f"{float(prediction.min()):.6f} -> "
        f"{float(prediction.max()):.6f}"
    )

    if valid_target.any():

        print(
            f"Target valid range: "
            f"{float(y[valid_target].min()):.6f} -> "
            f"{float(y[valid_target].max()):.6f}"
        )

    # ---------------------------------------------------------------
    # Metadata check.
    # ---------------------------------------------------------------

    print()
    print(
        "First batch target dates:"
    )

    for i in range(BATCH_SIZE):

        print(
            f"  sample {i}: "
            f"{metadata['target_date'][i]}"
        )

    # ---------------------------------------------------------------
    # Cleanup.
    # ---------------------------------------------------------------

    dataset.close()

    print()
    print("=" * 72)
    print("REAL DATASET INTEGRATION TEST PASSED")
    print("=" * 72)
    print()
    print(
        "Real OceanEmbed data successfully flows "
        "through OceanEmbed-CNN."
    )
    print()


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------


if __name__ == "__main__":
    main()