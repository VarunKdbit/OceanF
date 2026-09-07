"""
OceanEmbed Evaluation Engine

Evaluates a trained OceanEmbed-CNN checkpoint on validation or test data.

Responsibilities:
- Load a saved OceanEmbed-CNN checkpoint.
- Run inference on real OceanEmbed data.
- Convert normalized predictions back to degrees Celsius.
- Apply the target validity mask.
- Calculate dataset-level RMSE, MAE, Bias, and Pearson correlation.
- Calculate metrics independently for all 15 target depths.
- Save evaluation results as JSON.
- Provide a self-test without requiring a trained checkpoint.

This file performs evaluation only.
It does NOT perform final scientific model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_dataset import OceanEmbedDataset
from oceanembed_model import OceanEmbedCNN
from training_utils import get_device, set_random_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "ml_config.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "evaluation"
)

DEFAULT_CHECKPOINT = (
    CHECKPOINT_DIR
    / "oceanembed_dev_best.pt"
)


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


def load_ml_config() -> dict:
    """Load the persistent OceanEmbed ML configuration."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"ML configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_target_statistics(
    config: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-depth target mean/std.

    Returns:
        means: shape [15]
        stds: shape [15]
    """

    if "target_statistics" not in config:
        raise KeyError(
            "ml_config.json does not contain target_statistics."
        )

    statistics = config["target_statistics"]

    means = np.array(
        [
            float(statistics[str(depth)]["mean"])
            for depth in TARGET_DEPTHS
        ],
        dtype=np.float32,
    )

    stds = np.array(
        [
            float(statistics[str(depth)]["std"])
            for depth in TARGET_DEPTHS
        ],
        dtype=np.float32,
    )

    if means.shape != (15,):
        raise ValueError(
            f"Expected 15 target means, received {means.shape}."
        )

    if stds.shape != (15,):
        raise ValueError(
            f"Expected 15 target stds, received {stds.shape}."
        )

    if not np.all(np.isfinite(means)):
        raise ValueError(
            "Target means contain NaN or Inf."
        )

    if not np.all(np.isfinite(stds)):
        raise ValueError(
            "Target standard deviations contain NaN or Inf."
        )

    if np.any(stds <= 0):
        raise ValueError(
            "All target standard deviations must be > 0."
        )

    return means, stds


def denormalize_targets(
    values: torch.Tensor,
    means: np.ndarray,
    stds: np.ndarray,
) -> torch.Tensor:
    """
    Convert normalized temperature values back to degrees Celsius.

    Expected input shape:
        [B, 15, H, W]
    """

    if values.ndim != 4:
        raise ValueError(
            "Expected tensor with shape [B, 15, H, W]. "
            f"Received {tuple(values.shape)}."
        )

    if values.shape[1] != 15:
        raise ValueError(
            "Expected 15 depth channels. "
            f"Received {values.shape[1]}."
        )

    mean_tensor = torch.as_tensor(
        means,
        dtype=values.dtype,
        device=values.device,
    ).view(
        1,
        15,
        1,
        1,
    )

    std_tensor = torch.as_tensor(
        stds,
        dtype=values.dtype,
        device=values.device,
    ).view(
        1,
        15,
        1,
        1,
    )

    return (
        values * std_tensor
        + mean_tensor
    )


def calculate_regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate masked regression metrics.

    Metrics:
        RMSE
        MAE
        Bias
        Pearson correlation
    """

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    valid = (
        mask
        & np.isfinite(prediction)
        & np.isfinite(target)
    )

    prediction = prediction[valid]
    target = target[valid]

    count = int(prediction.size)

    if count == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "pearson": float("nan"),
            "valid_count": 0,
        }

    error = prediction - target

    rmse = float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    bias = float(
        np.mean(error)
    )

    if count < 2:
        pearson = float("nan")
    else:
        prediction_std = float(
            np.std(prediction)
        )

        target_std = float(
            np.std(target)
        )

        if (
            prediction_std == 0.0
            or target_std == 0.0
        ):
            pearson = float("nan")
        else:
            pearson = float(
                np.corrcoef(
                    prediction,
                    target,
                )[0, 1]
            )

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "pearson": pearson,
        "valid_count": count,
    }


class DepthMetricAccumulator:
    """
    Accumulates sufficient statistics so final metrics are calculated
    across the complete evaluation dataset rather than averaging
    batch-level metrics.
    """

    def __init__(self) -> None:

        self.count = 0

        self.sum_prediction = 0.0
        self.sum_target = 0.0

        self.sum_prediction_squared = 0.0
        self.sum_target_squared = 0.0

        self.sum_prediction_target = 0.0

        self.sum_error = 0.0
        self.sum_absolute_error = 0.0
        self.sum_squared_error = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Add one batch of valid values."""

        prediction = np.asarray(
            prediction,
            dtype=np.float64,
        )

        target = np.asarray(
            target,
            dtype=np.float64,
        )

        mask = np.asarray(
            mask,
            dtype=bool,
        )

        valid = (
            mask
            & np.isfinite(prediction)
            & np.isfinite(target)
        )

        prediction = prediction[valid]
        target = target[valid]

        if prediction.size == 0:
            return

        error = prediction - target

        self.count += int(
            prediction.size
        )

        self.sum_prediction += float(
            np.sum(prediction)
        )

        self.sum_target += float(
            np.sum(target)
        )

        self.sum_prediction_squared += float(
            np.sum(prediction ** 2)
        )

        self.sum_target_squared += float(
            np.sum(target ** 2)
        )

        self.sum_prediction_target += float(
            np.sum(
                prediction * target
            )
        )

        self.sum_error += float(
            np.sum(error)
        )

        self.sum_absolute_error += float(
            np.sum(
                np.abs(error)
            )
        )

        self.sum_squared_error += float(
            np.sum(error ** 2)
        )

    def compute(self) -> Dict[str, float]:
        """Calculate final metrics from all accumulated values."""

        if self.count == 0:
            return {
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "pearson": float("nan"),
                "valid_count": 0,
            }

        count = float(
            self.count
        )

        rmse = float(
            np.sqrt(
                self.sum_squared_error
                / count
            )
        )

        mae = float(
            self.sum_absolute_error
            / count
        )

        bias = float(
            self.sum_error
            / count
        )

        if self.count < 2:
            pearson = float("nan")

        else:
            covariance_numerator = (
                self.sum_prediction_target
                - (
                    self.sum_prediction
                    * self.sum_target
                    / count
                )
            )

            prediction_variance = (
                self.sum_prediction_squared
                - (
                    self.sum_prediction ** 2
                    / count
                )
            )

            target_variance = (
                self.sum_target_squared
                - (
                    self.sum_target ** 2
                    / count
                )
            )

            denominator = np.sqrt(
                max(
                    prediction_variance,
                    0.0,
                )
                * max(
                    target_variance,
                    0.0,
                )
            )

            if denominator == 0.0:
                pearson = float("nan")
            else:
                pearson = float(
                    covariance_numerator
                    / denominator
                )

        return {
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "pearson": pearson,
            "valid_count": int(
                self.count
            ),
        }


def validate_model_output(
    prediction: torch.Tensor,
    expected_shape: Tuple[int, int, int, int],
) -> None:
    """Validate model output shape and numerical stability."""

    if tuple(prediction.shape) != tuple(
        expected_shape
    ):
        raise ValueError(
            "Unexpected model output shape: "
            f"{tuple(prediction.shape)} != "
            f"{expected_shape}"
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise ValueError(
            "Model prediction contains NaN or Inf."
        )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """
    Load a PyTorch checkpoint.

    Supports:
    - checkpoint containing model_state_dict
    - direct state dictionary
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif isinstance(checkpoint, dict):
        model.load_state_dict(
            checkpoint
        )

    else:
        raise ValueError(
            "Unsupported checkpoint format."
        )

    return (
        checkpoint
        if isinstance(checkpoint, dict)
        else {}
    )


def run_evaluation(
    split: str,
    checkpoint_path: Path,
    batch_size: int,
    max_batches: Optional[int],
    seed: int,
    history_days: int,
) -> dict:
    """Run evaluation on the selected split."""

    if split not in {
        "validation",
        "test",
    }:
        raise ValueError(
            "split must be validation or test."
        )

    set_random_seed(
        seed
    )

    device = get_device()

    config = load_ml_config()

    target_means, target_stds = (
        load_target_statistics(
            config
        )
    )

    print()
    print("=" * 72)
    print("OceanEmbed EVALUATION ENGINE")
    print("=" * 72)

    print(
        f"Split                 : {split}"
    )

    print(
        f"Device                : {device}"
    )

    print(
        f"Checkpoint            : {checkpoint_path}"
    )

    print(
        f"Batch size            : {batch_size}"
    )

    print(
        f"Max batches           : {max_batches}"
    )

    print()
    print(
        "Creating evaluation dataset..."
    )

    dataset = OceanEmbedDataset(
        split=split,
        tile_stride=32,
        history_days=history_days,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(
        f"Dataset samples       : {len(dataset)}"
    )

    print()
    print(
        "Creating OceanEmbed-CNN..."
    )

    input_channels = history_days * 7

    model = OceanEmbedCNN(
        input_channels=input_channels,
        output_channels=15,
        latent_channels=128,
    )

    model.to(
        device
    )

    checkpoint_info = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    accumulators = [
        DepthMetricAccumulator()
        for _ in TARGET_DEPTHS
    ]

    processed_batches = 0

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader
        ):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            x, _, y, y_mask = batch

            x = x.to(
                device
            )

            y = y.to(
                device
            )

            y_mask = y_mask.to(
                device
            )

            prediction = model(
                x
            )

            expected_shape = (
                x.shape[0],
                15,
                32,
                32,
            )

            validate_model_output(
                prediction,
                expected_shape,
            )

            prediction_c = (
                denormalize_targets(
                    prediction,
                    target_means,
                    target_stds,
                )
            )

            target_c = (
                denormalize_targets(
                    y,
                    target_means,
                    target_stds,
                )
            )

            prediction_np = (
                prediction_c
                .detach()
                .cpu()
                .numpy()
            )

            target_np = (
                target_c
                .detach()
                .cpu()
                .numpy()
            )

            mask_np = (
                y_mask
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

            for depth_index in range(
                len(TARGET_DEPTHS)
            ):

                accumulators[
                    depth_index
                ].update(
                    prediction_np[
                        :,
                        depth_index,
                    ],
                    target_np[
                        :,
                        depth_index,
                    ],
                    mask_np[
                        :,
                        depth_index,
                    ],
                )

            processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No evaluation batches were processed."
        )

    depth_metrics = []

    for depth_index, depth in enumerate(
        TARGET_DEPTHS
    ):

        metrics = (
            accumulators[
                depth_index
            ].compute()
        )

        metrics["depth_m"] = float(
            depth
        )

        depth_metrics.append(
            metrics
        )

    overall_valid_count = sum(
        int(
            result["valid_count"]
        )
        for result in depth_metrics
    )

    print()
    print("=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)

    print(
        f"{'Depth':>8} "
        f"{'RMSE °C':>12} "
        f"{'MAE °C':>12} "
        f"{'Bias °C':>12} "
        f"{'Pearson':>12} "
        f"{'Valid':>12}"
    )

    print("-" * 72)

    for result in depth_metrics:

        print(
            f"{result['depth_m']:8.1f} "
            f"{result['rmse']:12.6f} "
            f"{result['mae']:12.6f} "
            f"{result['bias']:12.6f} "
            f"{result['pearson']:12.6f} "
            f"{result['valid_count']:12d}"
        )

    output = {
        "project": "OceanEmbed",
        "split": split,
        "device": str(device),
        "checkpoint": str(
            checkpoint_path
        ),
        "batch_size": batch_size,
        "processed_batches": processed_batches,
        "dataset_size": len(dataset),
        "target_depths_m": TARGET_DEPTHS,
        "metrics_units": {
            "rmse": "degC",
            "mae": "degC",
            "bias": "degC",
            "pearson": "dimensionless",
        },
        "depth_metrics": depth_metrics,
        "overall_valid_count": (
            overall_valid_count
        ),
        "checkpoint_metadata": {
            key: value
            for key, value in (
                checkpoint_info.items()
            )
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            )
        },
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_seed = checkpoint_path.stem.split("_seed")[-1] if "_seed" in checkpoint_path.stem else "unknown"
    output_path = RESULTS_DIR / split / f"seed{checkpoint_seed}_{split}_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print(
        f"Results saved: {output_path}"
    )

    return output


def evaluation_self_test() -> None:
    """Run evaluator self-tests without a checkpoint."""

    print()
    print("=" * 72)
    print("OceanEmbed EVALUATION ENGINE SELF-TEST")
    print("=" * 72)

    set_random_seed(
        42
    )

    config = load_ml_config()

    means, stds = (
        load_target_statistics(
            config
        )
    )

    assert means.shape == (
        15,
    )

    assert stds.shape == (
        15,
    )

    assert np.all(
        np.isfinite(means)
    )

    assert np.all(
        np.isfinite(stds)
    )

    assert np.all(
        stds > 0
    )

    print(
        "PASS: target statistics loaded."
    )

    normalized = torch.zeros(
        2,
        15,
        4,
        4,
        dtype=torch.float32,
    )

    denormalized = (
        denormalize_targets(
            normalized,
            means,
            stds,
        )
    )

    expected = torch.as_tensor(
        means,
        dtype=torch.float32,
    ).view(
        1,
        15,
        1,
        1,
    )

    assert torch.allclose(
        denormalized,
        expected,
    )

    print(
        "PASS: target denormalization."
    )

    prediction = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    target = np.array(
        [1.0, 2.0, 2.0],
        dtype=np.float64,
    )

    mask = np.array(
        [1, 1, 1],
        dtype=bool,
    )

    metrics = (
        calculate_regression_metrics(
            prediction,
            target,
            mask,
        )
    )

    assert metrics[
        "valid_count"
    ] == 3

    assert np.isclose(
        metrics["mae"],
        1.0 / 3.0,
    )

    assert np.isclose(
        metrics["bias"],
        1.0 / 3.0,
    )

    assert np.isfinite(
        metrics["rmse"]
    )

    assert np.isfinite(
        metrics["pearson"]
    )

    print(
        "PASS: regression metric calculation."
    )

    masked_metrics = (
        calculate_regression_metrics(
            prediction,
            target,
            np.array(
                [1, 0, 1],
                dtype=bool,
            ),
        )
    )

    assert masked_metrics[
        "valid_count"
    ] == 2

    assert np.isclose(
        masked_metrics["mae"],
        0.5,
    )

    print(
        "PASS: validity-mask handling."
    )

    accumulator = (
        DepthMetricAccumulator()
    )

    accumulator.update(
        np.array(
            [1.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [1.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [1, 1],
            dtype=bool,
        ),
    )

    accumulator.update(
        np.array(
            [3.0],
            dtype=np.float64,
        ),
        np.array(
            [2.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=bool,
        ),
    )

    accumulated_metrics = (
        accumulator.compute()
    )

    expected_metrics = (
        calculate_regression_metrics(
            np.array(
                [1.0, 2.0, 3.0]
            ),
            np.array(
                [1.0, 1.0, 2.0]
            ),
            np.array(
                [1, 1, 1]
            ),
        )
    )

    assert np.isclose(
        accumulated_metrics["rmse"],
        expected_metrics["rmse"],
    )

    assert np.isclose(
        accumulated_metrics["mae"],
        expected_metrics["mae"],
    )

    assert np.isclose(
        accumulated_metrics["bias"],
        expected_metrics["bias"],
    )

    assert np.isclose(
        accumulated_metrics["pearson"],
        expected_metrics["pearson"],
    )

    assert (
        accumulated_metrics[
            "valid_count"
        ]
        == expected_metrics[
            "valid_count"
        ]
    )

    print(
        "PASS: dataset-level metric accumulation."
    )

    model = OceanEmbedCNN(
        input_channels=49,
        output_channels=15,
        latent_channels=128,
    )

    model.eval()

    sample_input = torch.randn(
        1,
        49,
        64,
        64,
    )

    with torch.no_grad():

        sample_output = model(
            sample_input
        )

    validate_model_output(
        sample_output,
        (
            1,
            15,
            32,
            32,
        ),
    )

    print(
        "PASS: model output validation."
    )

    print()
    print("=" * 72)
    print(
        "EVALUATION ENGINE SELF-TEST PASSED"
    )
    print("=" * 72)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OceanEmbed-CNN "
            "in physical units."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="test",
        help=(
            "Dataset split to evaluate."
        ),
    )

    parser.add_argument(
        "--history-days",
        type=int,
        choices=range(1, 8),
        default=7,
        help="Number of retrospective input days.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to trained checkpoint."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "Evaluation batch size."
        ),
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Maximum number of batches. "
            "Omit for complete evaluation."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run evaluation-engine self-test."
        ),
    )

    return parser


def main() -> None:
    """Program entry point."""

    parser = (
        build_argument_parser()
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be greater than zero."
        )

    if (
        args.max_batches is not None
        and args.max_batches <= 0
    ):
        raise ValueError(
            "max-batches must be greater than zero."
        )

    if args.self_test:
        evaluation_self_test()
        return

    run_evaluation(
        split=args.split,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        seed=args.seed,
        history_days=args.history_days,
    )


if __name__ == "__main__":
    main()



