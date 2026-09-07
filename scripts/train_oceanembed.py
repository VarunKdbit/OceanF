"""
OceanEmbed Training Engine
==========================

Main training and validation loop for OceanEmbed-CNN.

This script supports:

    - Train / validation datasets
    - DataLoaders
    - OceanEmbed-CNN
    - Masked Huber loss
    - Adam optimizer
    - Reproducible seeds
    - Gradient clipping
    - RMSE / MAE / Bias / Pearson metrics
    - Best-checkpoint saving
    - Training history
    - Development smoke runs
    - Training and validation timing

IMPORTANT
---------
The default configuration is intentionally a SMALL DEVELOPMENT RUN.

It is NOT the final scientific training configuration.

Before final training, hyperparameters and experiment settings must
be explicitly selected and documented.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

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
# OCEANEMBED IMPORTS
# ============================================================================

from ml_dataset import OceanEmbedDataset
from oceanembed_model import (
    OceanEmbedCNN,
    count_parameters,
)
from masked_huber_loss import MaskedHuberLoss
from training_utils import (
    calculate_masked_metrics,
    check_model_finite,
    count_trainable_parameters,
    get_device,
    gradient_norm,
    save_checkpoint,
    set_random_seed,
)


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_SEED = 42

DEFAULT_BATCH_SIZE = 2

DEFAULT_LEARNING_RATE = 1e-3

DEFAULT_EPOCHS = 1

DEFAULT_TILE_STRIDE = 32

DEFAULT_NUM_WORKERS = 0

DEFAULT_GRADIENT_CLIP_NORM = 1.0

DEFAULT_PATIENCE = 5

DEFAULT_MAX_TRAIN_BATCHES = 2

DEFAULT_MAX_VALIDATION_BATCHES = 2


# ============================================================================
# CHECKPOINT DIRECTORY
# ============================================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "checkpoints"
)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================


def parse_args():
    """
    Parse command-line training configuration.
    """

    parser = argparse.ArgumentParser(
        description="Train OceanEmbed-CNN."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs.",
    )

    parser.add_argument(
        "--tile-stride",
        type=int,
        default=DEFAULT_TILE_STRIDE,
        help="Spatial tile stride.",
    )

    parser.add_argument(
        "--history-days",
        type=int,
        default=7,
        choices=range(1, 8),
        metavar="{1..7}",
        help="Number of retrospective input days.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker count.",
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
        help="Maximum global gradient norm.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stopping patience.",
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=DEFAULT_MAX_TRAIN_BATCHES,
        help=(
            "Maximum training batches per epoch. "
            "Use -1 for all batches."
        ),
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=DEFAULT_MAX_VALIDATION_BATCHES,
        help=(
            "Maximum validation batches per epoch. "
            "Use -1 for all batches."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu, cuda, or automatic selection.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="oceanembed_dev_best.pt",
        help="Checkpoint filename.",
    )

    return parser.parse_args()


# ============================================================================
# BATCH UNPACKING
# ============================================================================


def unpack_batch(batch):
    """
    Extract tensors from the OceanEmbedDataset batch.

    Dataset format:

        x
        x_mask
        y
        y_mask
        metadata
    """

    if len(batch) != 5:

        raise ValueError(
            "Expected dataset batch containing "
            "x, x_mask, y, y_mask and metadata."
        )

    x, x_mask, y, y_mask, metadata = batch

    return (
        x,
        x_mask,
        y,
        y_mask,
        metadata,
    )


# ============================================================================
# TRAINING EPOCH
# ============================================================================


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: Optional[float] = None,
    max_batches: int = -1,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Returns:
        Dictionary containing average loss and metrics.
    """

    model.train()

    total_loss = 0.0

    total_rmse = 0.0

    total_mae = 0.0

    total_bias = 0.0

    total_pearson = 0.0

    batch_count = 0

    for batch_index, batch in enumerate(loader):

        if (
            max_batches >= 0
            and batch_index >= max_batches
        ):
            break

        (
            x,
            x_mask,
            y,
            y_mask,
            metadata,
        ) = unpack_batch(batch)

        # --------------------------------------------------------------
        # Move required tensors to device.
        # --------------------------------------------------------------

        x = x.to(device)

        y = y.to(device)

        y_mask = y_mask.to(device)

        # --------------------------------------------------------------
        # Forward pass.
        # --------------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(x)

        # --------------------------------------------------------------
        # Validate prediction.
        # --------------------------------------------------------------

        if not torch.isfinite(
            prediction
        ).all():

            raise FloatingPointError(
                "Model produced NaN or Inf predictions."
            )

        # --------------------------------------------------------------
        # Calculate masked Huber loss.
        # --------------------------------------------------------------

        loss = criterion(
            prediction,
            y,
            y_mask,
        )

        if not torch.isfinite(loss):

            raise FloatingPointError(
                "Training loss became NaN or Inf."
            )

        # --------------------------------------------------------------
        # Backpropagation.
        # --------------------------------------------------------------

        loss.backward()

        # --------------------------------------------------------------
        # Gradient clipping.
        # --------------------------------------------------------------

        if gradient_clip_norm is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        # --------------------------------------------------------------
        # Gradient validation.
        # --------------------------------------------------------------

        current_gradient_norm = gradient_norm(
            model
        )

        if not torch.isfinite(
            torch.tensor(current_gradient_norm)
        ):

            raise FloatingPointError(
                "Gradient norm became NaN or Inf."
            )

        # --------------------------------------------------------------
        # Optimizer update.
        # --------------------------------------------------------------

        optimizer.step()

        # --------------------------------------------------------------
        # Validate model parameters.
        # --------------------------------------------------------------

        check_model_finite(
            model
        )

        # --------------------------------------------------------------
        # Calculate metrics.
        # --------------------------------------------------------------

        metrics = calculate_masked_metrics(
            prediction.detach(),
            y.detach(),
            y_mask.detach(),
        )

        # --------------------------------------------------------------
        # Accumulate batch results.
        # --------------------------------------------------------------

        total_loss += (
            loss.detach()
            .cpu()
            .item()
        )

        total_rmse += metrics["rmse"]

        total_mae += metrics["mae"]

        total_bias += metrics["bias"]

        total_pearson += metrics["pearson"]

        batch_count += 1

    # ------------------------------------------------------------------------
    # Ensure at least one batch was processed.
    # ------------------------------------------------------------------------

    if batch_count == 0:

        raise RuntimeError(
            "No training batches were processed."
        )

    return {
        "loss": total_loss / batch_count,
        "rmse": total_rmse / batch_count,
        "mae": total_mae / batch_count,
        "bias": total_bias / batch_count,
        "pearson": total_pearson / batch_count,
        "batches": batch_count,
    }


# ============================================================================
# VALIDATION EPOCH
# ============================================================================


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int = -1,
) -> Dict[str, float]:
    """
    Run one validation epoch.

    No gradients or optimizer updates are performed.
    """

    model.eval()

    total_loss = 0.0

    total_rmse = 0.0

    total_mae = 0.0

    total_bias = 0.0

    total_pearson = 0.0

    batch_count = 0

    for batch_index, batch in enumerate(loader):

        if (
            max_batches >= 0
            and batch_index >= max_batches
        ):
            break

        (
            x,
            x_mask,
            y,
            y_mask,
            metadata,
        ) = unpack_batch(batch)

        x = x.to(device)

        y = y.to(device)

        y_mask = y_mask.to(device)

        # --------------------------------------------------------------
        # Forward pass.
        # --------------------------------------------------------------

        prediction = model(x)

        if not torch.isfinite(
            prediction
        ).all():

            raise FloatingPointError(
                "Model produced NaN or Inf predictions during validation."
            )

        # --------------------------------------------------------------
        # Validation loss.
        # --------------------------------------------------------------

        loss = criterion(
            prediction,
            y,
            y_mask,
        )

        if not torch.isfinite(loss):

            raise FloatingPointError(
                "Validation loss became NaN or Inf."
            )

        # --------------------------------------------------------------
        # Metrics.
        # --------------------------------------------------------------

        metrics = calculate_masked_metrics(
            prediction,
            y,
            y_mask,
        )

        total_loss += (
            loss.cpu().item()
        )

        total_rmse += metrics["rmse"]

        total_mae += metrics["mae"]

        total_bias += metrics["bias"]

        total_pearson += metrics["pearson"]

        batch_count += 1

    if batch_count == 0:

        raise RuntimeError(
            "No validation batches were processed."
        )

    return {
        "loss": total_loss / batch_count,
        "rmse": total_rmse / batch_count,
        "mae": total_mae / batch_count,
        "bias": total_bias / batch_count,
        "pearson": total_pearson / batch_count,
        "batches": batch_count,
    }


# ============================================================================
# HISTORY SAVE
# ============================================================================


def save_training_history(
    history,
    path: Path,
):
    """
    Save training history as JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def run_training(
    args,
):
    """
    Execute the configured training run.
    """

    # ------------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------------

    set_random_seed(
        args.seed
    )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = get_device(
        args.device
    )

    print()
    print("=" * 72)
    print("OceanEmbed TRAINING ENGINE")
    print("=" * 72)

    print()
    print(
        f"Device                : {device}"
    )

    print(
        f"Seed                  : {args.seed}"
    )

    print(
        f"Batch size            : {args.batch_size}"
    )

    print(
        f"Learning rate         : {args.learning_rate}"
    )

    print(
        f"Epochs                : {args.epochs}"
    )

    print(
        f"Tile stride           : {args.tile_stride}"
    )

    print(
        f"Max train batches     : {args.max_train_batches}"
    )

    print(
        f"Max validation batches: "
        f"{args.max_validation_batches}"
    )

    print(
        f"History days          : {args.history_days}"
    )

    print(
        f"Input channels        : {args.history_days * 7}"
    )

    # ------------------------------------------------------------------------
    # Create datasets.
    # ------------------------------------------------------------------------

    print()
    print("Creating training dataset...")

    train_dataset = OceanEmbedDataset(
        split="train",
        tile_stride=args.tile_stride,
        return_metadata=True,
        normalize=True,
        history_days=args.history_days,
    )

    print()
    print("Creating validation dataset...")

    validation_dataset = OceanEmbedDataset(
        split="validation",
        tile_stride=args.tile_stride,
        return_metadata=True,
        normalize=True,
        history_days=args.history_days,
    )

    # ------------------------------------------------------------------------
    # DataLoaders.
    # ------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # ------------------------------------------------------------------------
    # Model.
    # ------------------------------------------------------------------------

    print()
    print("Creating OceanEmbed-CNN...")

    input_channels = args.history_days * 7

    latent_channels = 128

    output_channels = 15

    model = OceanEmbedCNN(
        input_channels=input_channels,
        latent_channels=latent_channels,
        output_channels=output_channels,
    )

    model = model.to(device)

    parameter_count = count_trainable_parameters(
        model
    )

    print(
        f"Trainable parameters: {parameter_count:,}"
    )

    # ------------------------------------------------------------------------
    # Loss.
    # ------------------------------------------------------------------------

    criterion = MaskedHuberLoss(
        delta=1.0
    )

    print(
        "Loss function: Masked Huber"
    )

    # ------------------------------------------------------------------------
    # Optimizer.
    # ------------------------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    print(
        "Optimizer: Adam"
    )

    # ------------------------------------------------------------------------
    # Training state.
    # ------------------------------------------------------------------------

    history = []

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    global_step = 0

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / args.checkpoint_name
    )

    history_path = (
        CHECKPOINT_DIR
        / "oceanembed_training_history.json"
    )

    # ------------------------------------------------------------------------
    # Complete experiment configuration.
    #
    # This configuration is stored inside every best checkpoint so that
    # the exact experiment can be reconstructed later.
    # ------------------------------------------------------------------------

    checkpoint_config = {
        "project": "OceanF",
        "experiment": "OceanEmbed-CNN",
        "experiment_variant": (
            "E2_7day_retrospective"
            if args.history_days == 7
            else f"E{args.history_days}_history_{args.history_days}day"
        ),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "tile_stride": args.tile_stride,
        "history_days": args.history_days,
        "input_channels": input_channels,
        "latent_channels": latent_channels,
        "output_channels": output_channels,
        "gradient_clip": args.gradient_clip,
        "num_workers": args.num_workers,
        "patience": args.patience,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "loss": "masked_huber",
        "huber_delta": 1.0,
        "optimizer": "Adam",
        "normalization": "training_period_mean_std",
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "model_parameters": parameter_count,
    }

    # ------------------------------------------------------------------------
    # Epoch loop.
    # ------------------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start_time = time.perf_counter()

        print()
        print("=" * 72)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("=" * 72)

        # --------------------------------------------------------------------
        # Training.
        # --------------------------------------------------------------------

        train_start_time = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=args.gradient_clip,
            max_batches=args.max_train_batches,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        train_elapsed_seconds = (
            time.perf_counter()
            - train_start_time
        )

        global_step += int(
            train_metrics["batches"]
        )

        # --------------------------------------------------------------------
        # Validation.
        # --------------------------------------------------------------------

        validation_start_time = time.perf_counter()

        validation_metrics = validate_one_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_validation_batches,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        validation_elapsed_seconds = (
            time.perf_counter()
            - validation_start_time
        )

        epoch_elapsed_seconds = (
            time.perf_counter()
            - epoch_start_time
        )

        # --------------------------------------------------------------------
        # Calculate throughput.
        # --------------------------------------------------------------------

        train_batches = train_metrics["batches"]

        validation_batches = validation_metrics["batches"]

        train_samples_processed = (
            train_batches * args.batch_size
        )

        validation_samples_processed = (
            validation_batches * args.batch_size
        )

        train_batches_per_second = (
            train_batches / train_elapsed_seconds
            if train_elapsed_seconds > 0
            else 0.0
        )

        train_samples_per_second = (
            train_samples_processed / train_elapsed_seconds
            if train_elapsed_seconds > 0
            else 0.0
        )

        validation_batches_per_second = (
            validation_batches / validation_elapsed_seconds
            if validation_elapsed_seconds > 0
            else 0.0
        )

        validation_samples_per_second = (
            validation_samples_processed
            / validation_elapsed_seconds
            if validation_elapsed_seconds > 0
            else 0.0
        )

        # --------------------------------------------------------------------
        # Print timing.
        # --------------------------------------------------------------------

        print()
        print("TIMING")
        print(
            f"  Training time       : "
            f"{train_elapsed_seconds:.3f} seconds"
        )
        print(
            f"  Validation time     : "
            f"{validation_elapsed_seconds:.3f} seconds"
        )
        print(
            f"  Total epoch time    : "
            f"{epoch_elapsed_seconds:.3f} seconds"
        )
        print(
            f"  Train batches/sec   : "
            f"{train_batches_per_second:.3f}"
        )
        print(
            f"  Train samples/sec   : "
            f"{train_samples_per_second:.3f}"
        )
        print(
            f"  Validation batches/sec: "
            f"{validation_batches_per_second:.3f}"
        )
        print(
            f"  Validation samples/sec: "
            f"{validation_samples_per_second:.3f}"
        )

        # --------------------------------------------------------------------
        # Print results.
        # --------------------------------------------------------------------

        print()
        print("TRAIN")
        print(
            f"  Loss    : {train_metrics['loss']:.6f}"
        )
        print(
            f"  RMSE    : {train_metrics['rmse']:.6f}"
        )
        print(
            f"  MAE     : {train_metrics['mae']:.6f}"
        )
        print(
            f"  Bias    : {train_metrics['bias']:.6f}"
        )
        print(
            f"  Pearson : {train_metrics['pearson']:.6f}"
        )
        print(
            f"  Batches : {train_metrics['batches']}"
        )

        print()
        print("VALIDATION")
        print(
            f"  Loss    : {validation_metrics['loss']:.6f}"
        )
        print(
            f"  RMSE    : {validation_metrics['rmse']:.6f}"
        )
        print(
            f"  MAE     : {validation_metrics['mae']:.6f}"
        )
        print(
            f"  Bias    : {validation_metrics['bias']:.6f}"
        )
        print(
            f"  Pearson : {validation_metrics['pearson']:.6f}"
        )
        print(
            f"  Batches : {validation_metrics['batches']}"
        )

        # --------------------------------------------------------------------
        # Save history.
        # --------------------------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "timing": {
                "training_seconds": train_elapsed_seconds,
                "validation_seconds": validation_elapsed_seconds,
                "epoch_seconds": epoch_elapsed_seconds,
                "train_batches_per_second": train_batches_per_second,
                "train_samples_per_second": train_samples_per_second,
                "validation_batches_per_second": validation_batches_per_second,
                "validation_samples_per_second": validation_samples_per_second,
            },
        }

        history.append(
            epoch_record
        )

        save_training_history(
            history,
            history_path,
        )

        # --------------------------------------------------------------------
        # Best model checkpoint.
        # --------------------------------------------------------------------

        current_validation_loss = (
            validation_metrics["loss"]
        )

        if (
            current_validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                current_validation_loss
            )

            epochs_without_improvement = 0

            checkpoint = save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                train_loss=train_metrics["loss"],
                validation_loss=validation_metrics["loss"],
                metrics={
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "timing": {
                        "training_seconds": train_elapsed_seconds,
                        "validation_seconds": validation_elapsed_seconds,
                        "epoch_seconds": epoch_elapsed_seconds,
                        "train_batches_per_second": train_batches_per_second,
                        "train_samples_per_second": train_samples_per_second,
                        "validation_batches_per_second": validation_batches_per_second,
                        "validation_samples_per_second": validation_samples_per_second,
                    },
                },
                config=checkpoint_config,
            )

            print()
            print(
                "BEST CHECKPOINT SAVED:"
            )
            print(
                f"  {checkpoint}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "No validation improvement."
            )

            print(
                f"Early-stopping counter: "
                f"{epochs_without_improvement}/"
                f"{args.patience}"
            )

        # --------------------------------------------------------------------
        # Early stopping.
        # --------------------------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ------------------------------------------------------------------------
    # Cleanup.
    # ------------------------------------------------------------------------

    train_dataset.close()

    validation_dataset.close()

    # ------------------------------------------------------------------------
    # Final model validation.
    # ------------------------------------------------------------------------

    check_model_finite(
        model
    )

    # ------------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("OCEANEMBED TRAINING RUN COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"History: "
        f"{history_path}"
    )

    print()
    print(
        "Model parameters remain finite."
    )

    print(
        "Training engine completed successfully."
    )

    print()


# ============================================================================
# MAIN
# ============================================================================


def main():

    args = parse_args()

    # ------------------------------------------------------------------------
    # Validate arguments.
    # ------------------------------------------------------------------------

    if args.batch_size <= 0:

        raise ValueError(
            "batch-size must be greater than zero."
        )

    if args.learning_rate <= 0:

        raise ValueError(
            "learning-rate must be greater than zero."
        )

    if args.epochs <= 0:

        raise ValueError(
            "epochs must be greater than zero."
        )

    if args.tile_stride <= 0:

        raise ValueError(
            "tile-stride must be greater than zero."
        )

    if args.num_workers < 0:

        raise ValueError(
            "num-workers cannot be negative."
        )

    if args.gradient_clip <= 0:

        raise ValueError(
            "gradient-clip must be greater than zero."
        )

    if args.patience <= 0:

        raise ValueError(
            "patience must be greater than zero."
        )

    run_training(
        args
    )


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()