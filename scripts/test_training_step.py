"""
OceanEmbed End-to-End Training Step Test
=========================================

Purpose
-------
Verify that one complete training step works using REAL OceanEmbed
data.

Pipeline:

    OceanEmbedDataset
            ↓
        DataLoader
            ↓
        OceanEmbed-CNN
            ↓
        Masked Huber Loss
            ↓
        Backpropagation
            ↓
        Adam Optimizer
            ↓
        Parameter Update

IMPORTANT
---------
This is NOT model training.

Only ONE optimizer step is performed.

No checkpoint is saved.

No model is trained to convergence.

This test exists only to verify that the complete ML pipeline
is technically functional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


# ============================================================================
# PATH SETUP
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ============================================================================
# IMPORT OCEANEMBED COMPONENTS
# ============================================================================

from ml_dataset import OceanEmbedDataset
from oceanembed_model import (
    OceanEmbedCNN,
    count_parameters,
)
from masked_huber_loss import MaskedHuberLoss


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

BATCH_SIZE = 2

LEARNING_RATE = 1e-3

TILE_STRIDE = 32

NUM_WORKERS = 0


# ============================================================================
# MAIN
# ============================================================================


def main():

    print()
    print("=" * 72)
    print("OceanEmbed END-TO-END TRAINING STEP TEST")
    print("=" * 72)

    print()
    print("IMPORTANT:")
    print("Only ONE optimizer step will be performed.")
    print("This is NOT full model training.")

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = torch.device("cpu")

    print()
    print(f"Device: {device}")

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    print()
    print("Creating real OceanEmbed training dataset...")

    dataset = OceanEmbedDataset(
        split="train",
        tile_stride=TILE_STRIDE,
        return_metadata=True,
        normalize=True,
    )

    print(
        f"Dataset size: {len(dataset):,}"
    )

    # ------------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # ------------------------------------------------------------------------
    # Load one batch
    # ------------------------------------------------------------------------

    print()
    print("Loading one real training batch...")

    (
        x,
        x_mask,
        y,
        y_mask,
        metadata,
    ) = next(iter(loader))

    print(
        f"x shape      : {tuple(x.shape)}"
    )

    print(
        f"y shape      : {tuple(y.shape)}"
    )

    print(
        f"y mask shape : {tuple(y_mask.shape)}"
    )

    # ------------------------------------------------------------------------
    # Verify input
    # ------------------------------------------------------------------------

    assert tuple(x.shape) == (
        BATCH_SIZE,
        49,
        64,
        64,
    )

    assert tuple(y.shape) == (
        BATCH_SIZE,
        15,
        32,
        32,
    )

    assert tuple(y_mask.shape) == (
        BATCH_SIZE,
        15,
        32,
        32,
    )

    assert torch.isfinite(x).all()

    print()
    print(
        "PASS: real training batch is valid."
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    print()
    print("Creating OceanEmbed-CNN...")

    model = OceanEmbedCNN()

    model = model.to(device)

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ------------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------------

    criterion = MaskedHuberLoss(
        delta=1.0
    )

    print(
        "Loss: Masked Huber"
    )

    # ------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(
        f"Optimizer: Adam"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    # ------------------------------------------------------------------------
    # Move tensors to device
    # ------------------------------------------------------------------------

    x = x.to(device)

    y = y.to(device)

    y_mask = y_mask.to(device)

    # ------------------------------------------------------------------------
    # Save one parameter BEFORE update.
    #
    # This allows us to verify that the optimizer actually changed
    # the model.
    # ------------------------------------------------------------------------

    parameter_before = next(
        model.parameters()
    ).detach().clone()

    # ------------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------------

    print()
    print("Running forward pass...")

    model.train()

    prediction = model(x)

    print(
        f"Prediction shape: "
        f"{tuple(prediction.shape)}"
    )

    assert tuple(prediction.shape) == (
        BATCH_SIZE,
        15,
        32,
        32,
    )

    assert torch.isfinite(
        prediction
    ).all()

    print(
        "PASS: forward pass."
    )

    # ------------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------------

    print()
    print("Calculating masked Huber loss...")

    loss = criterion(
        prediction,
        y,
        y_mask,
    )

    print(
        f"Initial loss: "
        f"{loss.detach().item():.6f}"
    )

    assert torch.isfinite(loss)

    assert loss.item() > 0

    print(
        "PASS: loss is finite and positive."
    )

    # ------------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------------

    print()
    print("Running backward pass...")

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    print(
        "Backward pass completed."
    )

    # ------------------------------------------------------------------------
    # Gradient validation
    # ------------------------------------------------------------------------

    total_gradient_norm = 0.0

    gradient_parameter_count = 0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all():

            raise AssertionError(
                "NaN/Inf gradient detected."
            )

        gradient_parameter_count += 1

        gradient_norm = (
            parameter.grad.detach()
            .norm()
            .item()
        )

        total_gradient_norm += (
            gradient_norm ** 2
        )

    total_gradient_norm = (
        total_gradient_norm ** 0.5
    )

    print()
    print(
        f"Parameters with gradients: "
        f"{gradient_parameter_count}"
    )

    print(
        f"Total gradient norm: "
        f"{total_gradient_norm:.6f}"
    )

    assert (
        gradient_parameter_count > 0
    )

    assert (
        total_gradient_norm > 0
    )

    print(
        "PASS: gradients are finite and non-zero."
    )

    # ------------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------------

    print()
    print("Running ONE Adam optimizer step...")

    optimizer.step()

    print(
        "Optimizer step completed."
    )

    # ------------------------------------------------------------------------
    # Verify parameter update
    # ------------------------------------------------------------------------

    parameter_after = next(
        model.parameters()
    ).detach().clone()

    parameter_change = (
        parameter_after
        - parameter_before
    ).abs().sum().item()

    print()
    print(
        f"Parameter absolute change: "
        f"{parameter_change:.12f}"
    )

    assert parameter_change > 0

    print(
        "PASS: model parameters changed after optimizer step."
    )

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------

    dataset.close()

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("END-TO-END TRAINING STEP TEST PASSED")
    print("=" * 72)
    print()

    print(
        "Verified:"
    )

    print(
        "  Real OceanEmbed data"
    )

    print(
        "  → OceanEmbed-CNN"
    )

    print(
        "  → Masked Huber loss"
    )

    print(
        "  → Backpropagation"
    )

    print(
        "  → Finite gradients"
    )

    print(
        "  → Adam optimizer"
    )

    print(
        "  → Actual parameter update"
    )

    print()
    print(
        "The complete ML training pipeline is "
        "technically functional."
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()