"""
OceanEmbed Training Utilities
==============================

Reusable utilities for the OceanEmbed ML training pipeline.

This file provides:

    1. Reproducibility / random seed setup
    2. Device selection
    3. Masked regression metrics
    4. Parameter and gradient utilities
    5. Checkpoint save/load helpers
    6. Basic training configuration helpers

This file does NOT contain the main training loop.

The main training loop will be implemented separately.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ML"

CHECKPOINT_DIR = ML_DIR / "checkpoints"


# ============================================================================
# REPRODUCIBILITY
# ============================================================================


def set_random_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Integer random seed.

    Notes
    -----
    This configures Python, NumPy and PyTorch random generators.

    CUDA settings are also configured when CUDA is available.
    """

    if not isinstance(seed, int):
        raise TypeError(
            f"seed must be an integer, received {type(seed)}"
        )

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        # --------------------------------------------------------------
        # Request deterministic CUDA behavior where possible.
        # --------------------------------------------------------------

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================================
# DEVICE
# ============================================================================


def get_device(
    preferred: Optional[str] = None,
) -> torch.device:
    """
    Select the computation device.

    Parameters
    ----------
    preferred:
        Optional requested device.

        Examples:
            None
            "cpu"
            "cuda"

    Returns
    -------
    torch.device
    """

    if preferred is not None:

        preferred = preferred.lower().strip()

    # ------------------------------------------------------------------------
    # Explicit CPU request.
    # ------------------------------------------------------------------------

    if preferred == "cpu":

        return torch.device("cpu")

    # ------------------------------------------------------------------------
    # Explicit CUDA request.
    # ------------------------------------------------------------------------

    if preferred == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was explicitly requested, but CUDA is not available."
            )

        return torch.device("cuda")

    # ------------------------------------------------------------------------
    # Automatic selection.
    # ------------------------------------------------------------------------

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================================
# PARAMETER UTILITIES
# ============================================================================


def count_trainable_parameters(
    model: torch.nn.Module,
) -> int:
    """
    Count trainable model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def gradient_norm(
    model: torch.nn.Module,
) -> float:
    """
    Calculate the global L2 norm of all available model gradients.

    Returns
    -------
    float
        Global gradient norm.

    Returns 0.0 if no gradients are present.
    """

    total_squared_norm = 0.0

    found_gradient = False

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        found_gradient = True

        gradient = parameter.grad.detach()

        if not torch.isfinite(gradient).all():

            raise FloatingPointError(
                "NaN or Inf detected in model gradients."
            )

        norm = gradient.norm(2).item()

        total_squared_norm += norm ** 2

    if not found_gradient:

        return 0.0

    return total_squared_norm ** 0.5


def check_model_finite(
    model: torch.nn.Module,
) -> None:
    """
    Verify that all model parameters contain finite values.
    """

    for name, parameter in model.named_parameters():

        if not torch.isfinite(parameter).all():

            raise FloatingPointError(
                f"Model parameter '{name}' contains NaN or Inf."
            )


# ============================================================================
# MASK UTILITIES
# ============================================================================


def validate_mask(
    mask: torch.Tensor,
) -> None:
    """
    Validate a binary validity mask.

    Valid values:
        0 = invalid
        1 = valid
    """

    if not torch.isfinite(mask).all():

        raise ValueError(
            "Mask contains NaN or Inf values."
        )

    unique_values = torch.unique(mask)

    if not torch.all(
        (unique_values == 0)
        | (unique_values == 1)
    ):

        raise ValueError(
            "Mask must contain only 0 and 1."
        )


def valid_count(
    mask: torch.Tensor,
) -> int:
    """
    Return the number of valid cells.
    """

    validate_mask(mask)

    return int(
        (mask > 0).sum().item()
    )


# ============================================================================
# MASKED METRICS
# ============================================================================


def _select_valid_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Select only valid prediction/target pairs.

    Invalid target values are never used in arithmetic.
    """

    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target shapes must match.\n"
            f"Prediction: {tuple(prediction.shape)}\n"
            f"Target:     {tuple(target.shape)}"
        )

    if prediction.shape != mask.shape:

        raise ValueError(
            "Prediction and mask shapes must match.\n"
            f"Prediction: {tuple(prediction.shape)}\n"
            f"Mask:       {tuple(mask.shape)}"
        )

    validate_mask(mask)

    valid = mask > 0

    if valid.sum().item() == 0:

        raise ValueError(
            "No valid cells available for metric calculation."
        )

    prediction_valid = prediction[valid]

    target_valid = target[valid]

    if not torch.isfinite(
        prediction_valid
    ).all():

        raise ValueError(
            "Valid prediction values contain NaN or Inf."
        )

    if not torch.isfinite(
        target_valid
    ).all():

        raise ValueError(
            "Valid target values contain NaN or Inf."
        )

    return (
        prediction_valid,
        target_valid,
    )


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean Absolute Error over valid target cells only.
    """

    prediction_valid, target_valid = (
        _select_valid_values(
            prediction,
            target,
            mask,
        )
    )

    return torch.mean(
        torch.abs(
            prediction_valid
            - target_valid
        )
    )


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean Squared Error over valid target cells only.
    """

    prediction_valid, target_valid = (
        _select_valid_values(
            prediction,
            target,
            mask,
        )
    )

    error = (
        prediction_valid
        - target_valid
    )

    return torch.mean(
        error ** 2
    )


def masked_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Root Mean Squared Error over valid target cells only.
    """

    mse = masked_mse(
        prediction,
        target,
        mask,
    )

    return torch.sqrt(mse)


def masked_bias(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean prediction bias over valid target cells.

    Bias:

        mean(prediction - target)
    """

    prediction_valid, target_valid = (
        _select_valid_values(
            prediction,
            target,
            mask,
        )
    )

    return torch.mean(
        prediction_valid
        - target_valid
    )


def masked_pearson_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Pearson correlation over valid target cells.

    Returns:
        Scalar correlation coefficient.
    """

    prediction_valid, target_valid = (
        _select_valid_values(
            prediction,
            target,
            mask,
        )
    )

    prediction_centered = (
        prediction_valid
        - prediction_valid.mean()
    )

    target_centered = (
        target_valid
        - target_valid.mean()
    )

    numerator = torch.sum(
        prediction_centered
        * target_centered
    )

    denominator = torch.sqrt(
        torch.sum(
            prediction_centered ** 2
        )
        *
        torch.sum(
            target_centered ** 2
        )
    )

    if denominator.item() == 0:

        raise ValueError(
            "Pearson correlation is undefined because "
            "one variable has zero variance."
        )

    return numerator / denominator


# ============================================================================
# METRIC COLLECTION
# ============================================================================


def calculate_masked_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate the standard OceanEmbed regression metrics.

    Metrics:
        RMSE
        MAE
        Bias
        Pearson correlation

    All metrics use valid target cells only.
    """

    rmse = masked_rmse(
        prediction,
        target,
        mask,
    )

    mae = masked_mae(
        prediction,
        target,
        mask,
    )

    bias = masked_bias(
        prediction,
        target,
        mask,
    )

    correlation = masked_pearson_correlation(
        prediction,
        target,
        mask,
    )

    return {
        "rmse": float(
            rmse.detach().cpu().item()
        ),
        "mae": float(
            mae.detach().cpu().item()
        ),
        "bias": float(
            bias.detach().cpu().item()
        ),
        "pearson": float(
            correlation.detach().cpu().item()
        ),
        "valid_count": valid_count(mask),
    }


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    train_loss: Optional[float] = None,
    validation_loss: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a training checkpoint.

    The checkpoint stores:

        model state
        optimizer state
        epoch
        global step
        training loss
        validation loss
        metrics
        configuration

    Returns
    -------
    Path
        Actual checkpoint path.
    """

    checkpoint_path = Path(path)

    if not checkpoint_path.is_absolute():

        checkpoint_path = (
            PROJECT_ROOT
            / checkpoint_path
        )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "metrics": metrics,
        "config": config,
    }

    if optimizer is not None:

        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Parameters
    ----------
    path:
        Checkpoint file.

    model:
        Model receiving the saved parameters.

    optimizer:
        Optional optimizer receiving its saved state.

    device:
        Device on which checkpoint tensors should be loaded.
    """

    checkpoint_path = Path(path)

    if not checkpoint_path.is_absolute():

        checkpoint_path = (
            PROJECT_ROOT
            / checkpoint_path
        )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    if device is None:

        device = torch.device("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    return checkpoint


# ============================================================================
# JSON CONFIGURATION
# ============================================================================


def save_json(
    path: str | Path,
    data: Dict[str, Any],
) -> Path:
    """
    Save a JSON dictionary.
    """

    output_path = Path(path)

    if not output_path.is_absolute():

        output_path = (
            PROJECT_ROOT
            / output_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            default=str,
        )

    return output_path


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================


def create_training_config(
    seed: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
) -> Dict[str, Any]:
    """
    Create a compact training configuration dictionary.
    """

    if learning_rate <= 0:

        raise ValueError(
            "learning_rate must be > 0."
        )

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be > 0."
        )

    if epochs <= 0:

        raise ValueError(
            "epochs must be > 0."
        )

    return {
        "seed": int(seed),
        "learning_rate": float(
            learning_rate
        ),
        "batch_size": int(
            batch_size
        ),
        "epochs": int(
            epochs
        ),
    }


# ============================================================================
# UTILITY SELF-TEST
# ============================================================================


def utility_self_test():
    """
    Run basic tests for all major utilities.
    """

    print()
    print("=" * 72)
    print("OceanEmbed TRAINING UTILITIES SELF-TEST")
    print("=" * 72)

    # ------------------------------------------------------------------------
    # Seed test
    # ------------------------------------------------------------------------

    set_random_seed(42)

    random_a = torch.rand(5)

    set_random_seed(42)

    random_b = torch.rand(5)

    assert torch.equal(
        random_a,
        random_b,
    )

    print(
        "PASS: random seed reproducibility."
    )

    # ------------------------------------------------------------------------
    # Device test
    # ------------------------------------------------------------------------

    device = get_device()

    print(
        f"Device detected: {device}"
    )

    assert isinstance(
        device,
        torch.device,
    )

    print(
        "PASS: device selection."
    )

    # ------------------------------------------------------------------------
    # Metric test data
    # ------------------------------------------------------------------------

    prediction = torch.tensor(
        [
            [
                [
                    [1.0, 3.0],
                    [5.0, 7.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [5.0, float("nan")],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    # Valid errors:
    #
    #   0
    #   2
    #   0
    #
    # MAE = 2/3
    # RMSE = sqrt(4/3)
    # Bias = 2/3

    metrics = calculate_masked_metrics(
        prediction,
        target,
        mask,
    )

    print()
    print(
        "Metric test results:"
    )

    for name, value in metrics.items():

        print(
            f"  {name}: {value}"
        )

    expected_mae = 2.0 / 3.0

    expected_rmse = (
        (4.0 / 3.0) ** 0.5
    )

    expected_bias = 2.0 / 3.0

    assert abs(
        metrics["mae"]
        - expected_mae
    ) < 1e-6

    assert abs(
        metrics["rmse"]
        - expected_rmse
    ) < 1e-6

    assert abs(
        metrics["bias"]
        - expected_bias
    ) < 1e-6

    assert metrics[
        "valid_count"
    ] == 3

    print(
        "PASS: masked regression metrics."
    )

    # ------------------------------------------------------------------------
    # Mask validation
    # ------------------------------------------------------------------------

    validate_mask(mask)

    print(
        "PASS: binary mask validation."
    )

    # ------------------------------------------------------------------------
    # Model parameter utility
    # ------------------------------------------------------------------------

    model = torch.nn.Linear(
        4,
        2,
    )

    parameter_count = count_trainable_parameters(
        model
    )

    assert parameter_count == 10

    print(
        "PASS: trainable parameter counting."
    )

    # ------------------------------------------------------------------------
    # Gradient norm
    # ------------------------------------------------------------------------

    x = torch.randn(
        3,
        4,
    )

    output = model(x)

    loss = output.mean()

    loss.backward()

    norm = gradient_norm(
        model
    )

    assert norm > 0

    print(
        "PASS: gradient norm calculation."
    )

    # ------------------------------------------------------------------------
    # Model finite check
    # ------------------------------------------------------------------------

    check_model_finite(
        model
    )

    print(
        "PASS: model finite-value check."
    )

    # ------------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------------

    config = create_training_config(
        seed=42,
        learning_rate=1e-3,
        batch_size=2,
        epochs=5,
    )

    assert config["seed"] == 42

    assert config[
        "learning_rate"
    ] == 1e-3

    print(
        "PASS: training configuration creation."
    )

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("TRAINING UTILITIES SELF-TEST PASSED")
    print("=" * 72)
    print()


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    utility_self_test()