"""
OceanEmbed 3-Seed Ensemble Evaluation

Evaluates the production OceanEmbed-CNN ensemble using exactly three
independently trained seeds:

- Seed 42
- Seed 123
- Seed 2024

The ensemble prediction is the arithmetic mean of the three model
predictions in normalized target space.

The ensemble is evaluated on the same held-out test dataset and uses
the same target denormalization and target validity masking as the
standard evaluation engine.

This script does NOT train models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from oceanembed_model import OceanEmbedCNN
from training_utils import (
    get_device,
    load_checkpoint,
    set_random_seed,
)
from ml_dataset import OceanEmbedDataset


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
)

CHECKPOINT_DIR = ML_DIR / "checkpoints"

RESULTS_DIR = (
    ML_DIR
    / "evaluation"
    / "ensemble"
)

DEFAULT_CHECKPOINTS = [
    CHECKPOINT_DIR / "oceanembed_e2_seed42.pt",
    CHECKPOINT_DIR / "oceanembed_e2_seed123.pt",
    CHECKPOINT_DIR / "oceanembed_e2_seed2024.pt",
]

ML_CONFIG_PATH = ML_DIR / "ml_config.json"

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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def load_ml_config() -> dict:
    """Load the ML configuration JSON."""

    with ML_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_target_statistics(
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load target mean/std for all 15 depths.

    Returns:
        target_means: [15]
        target_stds:  [15]
    """

    statistics = config["target_statistics"]

    means = []
    stds = []

    for depth in TARGET_DEPTHS:

        key = str(depth)

        if key not in statistics:
            raise KeyError(
                f"Missing target statistics for depth {depth} m."
            )

        means.append(
            float(
                statistics[key]["mean"]
            )
        )

        stds.append(
            float(
                statistics[key]["std"]
            )
        )

    return (
        torch.tensor(
            means,
            dtype=torch.float32,
        ),
        torch.tensor(
            stds,
            dtype=torch.float32,
        ),
    )


def denormalize_targets(
    values: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    """
    Convert normalized target values back to degC.
    """

    means = means.to(
        device=values.device,
        dtype=values.dtype,
    ).view(
        1,
        -1,
        1,
        1,
    )

    stds = stds.to(
        device=values.device,
        dtype=values.dtype,
    ).view(
        1,
        -1,
        1,
        1,
    )

    return (
        values * stds
        + means
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

class DepthMetricAccumulator:
    """
    Accumulate sufficient statistics for exact dataset-level metrics.

    Metrics:
        RMSE
        MAE
        Bias
        Pearson correlation
    """

    def __init__(self) -> None:

        self.count = 0

        self.sum_error = 0.0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0

        self.sum_prediction = 0.0
        self.sum_target = 0.0

        self.sum_prediction_squared = 0.0
        self.sum_target_squared = 0.0

        self.sum_prediction_target = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> None:

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

        valid_prediction = prediction[mask]
        valid_target = target[mask]

        if valid_prediction.size == 0:
            return

        error = (
            valid_prediction
            - valid_target
        )

        self.count += int(
            valid_prediction.size
        )

        self.sum_error += float(
            np.sum(error)
        )

        self.sum_abs_error += float(
            np.sum(np.abs(error))
        )

        self.sum_squared_error += float(
            np.sum(error * error)
        )

        self.sum_prediction += float(
            np.sum(valid_prediction)
        )

        self.sum_target += float(
            np.sum(valid_target)
        )

        self.sum_prediction_squared += float(
            np.sum(
                valid_prediction
                * valid_prediction
            )
        )

        self.sum_target_squared += float(
            np.sum(
                valid_target
                * valid_target
            )
        )

        self.sum_prediction_target += float(
            np.sum(
                valid_prediction
                * valid_target
            )
        )

    def compute(self) -> dict:

        if self.count == 0:

            return {
                "rmse": None,
                "mae": None,
                "bias": None,
                "pearson": None,
                "valid_count": 0,
            }

        count = float(
            self.count
        )

        mse = (
            self.sum_squared_error
            / count
        )

        mae = (
            self.sum_abs_error
            / count
        )

        bias = (
            self.sum_error
            / count
        )

        covariance = (
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
                self.sum_prediction
                * self.sum_prediction
                / count
            )
        )

        target_variance = (
            self.sum_target_squared
            - (
                self.sum_target
                * self.sum_target
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

        if denominator <= 0.0:

            pearson = None

        else:

            pearson = (
                covariance
                / denominator
            )

        return {
            "rmse": float(
                np.sqrt(
                    max(mse, 0.0)
                )
            ),
            "mae": float(mae),
            "bias": float(bias),
            "pearson": (
                float(pearson)
                if pearson is not None
                else None
            ),
            "valid_count": int(
                self.count
            ),
        }


# ---------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------

def validate_checkpoint_paths(
    checkpoint_paths: list[Path],
) -> None:
    """Ensure exactly three production checkpoints exist."""

    if len(checkpoint_paths) != 3:

        raise ValueError(
            "OceanEmbed V1 ensemble requires exactly "
            "three checkpoints."
        )

    for checkpoint_path in checkpoint_paths:

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )


def run_ensemble_evaluation(
    split: str,
    checkpoint_paths: list[Path],
    batch_size: int,
    max_batches: Optional[int],
    seed: int,
    history_days: int,
) -> dict:
    """Run the three-seed ensemble evaluation."""

    if split not in {
        "validation",
        "test",
    }:

        raise ValueError(
            "split must be validation or test."
        )

    validate_checkpoint_paths(
        checkpoint_paths
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
    print("OceanEmbed 3-SEED ENSEMBLE EVALUATION")
    print("=" * 72)

    print(
        f"Split                 : {split}"
    )

    print(
        f"Device                : {device}"
    )

    print(
        f"Batch size            : {batch_size}"
    )

    print(
        f"History days          : {history_days}"
    )

    print(
        f"Number of seeds       : {len(checkpoint_paths)}"
    )

    print()

    for checkpoint_path in checkpoint_paths:

        print(
            f"Checkpoint            : {checkpoint_path}"
        )

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    print()
    print(
        "Creating ensemble evaluation dataset..."
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

    # -------------------------------------------------------------
    # Models
    # -------------------------------------------------------------

    print()
    print(
        "Creating three OceanEmbed-CNN models..."
    )

    input_channels = (
        history_days * 7
    )

    models = []

    checkpoint_metadata = []

    for checkpoint_path in checkpoint_paths:

        model = OceanEmbedCNN(
            input_channels=input_channels,
            output_channels=15,
            latent_channels=128,
        )

        model.to(
            device
        )

        metadata = load_checkpoint(
            checkpoint_path,
            model,
            device=device,
        )

        model.eval()

        models.append(
            model
        )

        checkpoint_metadata.append(
            {
                key: value
                for key, value in metadata.items()
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
            }
        )

        print(
            f"Loaded: {checkpoint_path.name}"
        )

    print()
    print(
        "All three checkpoints loaded successfully."
    )

    # -------------------------------------------------------------
    # Metric accumulators
    # -------------------------------------------------------------

    accumulators = [
        DepthMetricAccumulator()
        for _ in TARGET_DEPTHS
    ]

    processed_batches = 0

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------

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

            predictions = []

            for model in models:

                prediction = model(
                    x
                )

                expected_shape = (
                    x.shape[0],
                    15,
                    32,
                    32,
                )

                if tuple(
                    prediction.shape
                ) != expected_shape:

                    raise RuntimeError(
                        "Unexpected model output "
                        f"shape: {prediction.shape}. "
                        f"Expected {expected_shape}."
                    )

                if not torch.isfinite(
                    prediction
                ).all():

                    raise RuntimeError(
                        "Non-finite prediction "
                        "encountered."
                    )

                predictions.append(
                    prediction
                )

            # -----------------------------------------------------
            # Ensemble average in normalized space
            # -----------------------------------------------------

            ensemble_prediction = (
                torch.stack(
                    predictions,
                    dim=0,
                ).mean(
                    dim=0
                )
            )

            if not torch.isfinite(
                ensemble_prediction
            ).all():

                raise RuntimeError(
                    "Non-finite ensemble prediction."
                )

            # -----------------------------------------------------
            # Convert to degC
            # -----------------------------------------------------

            prediction_c = (
                denormalize_targets(
                    ensemble_prediction,
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
            "No ensemble evaluation batches "
            "were processed."
        )

    # -------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("ENSEMBLE EVALUATION RESULTS")
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

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    output = {
        "project": "OceanEmbed",
        "experiment": "E2_7day_retrospective",
        "evaluation_type": "three_seed_ensemble",
        "split": split,
        "device": str(device),
        "history_days": history_days,
        "batch_size": batch_size,
        "processed_batches": processed_batches,
        "dataset_size": len(dataset),
        "ensemble_size": len(models),
        "ensemble_method": (
            "arithmetic_mean_of_three_model_predictions"
        ),
        "ensemble_seeds": [
            42,
            123,
            2024,
        ],
        "checkpoints": [
            str(path)
            for path in checkpoint_paths
        ],
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
        "checkpoint_metadata": checkpoint_metadata,
    }

    output_dir = (
        RESULTS_DIR
        / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "seed42_seed123_seed2024_ensemble_evaluation.json"
    )

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


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def ensemble_self_test() -> None:
    """Run lightweight mathematical self-tests."""

    print()
    print("=" * 72)
    print("OceanEmbed ENSEMBLE SELF-TEST")
    print("=" * 72)

    prediction_1 = torch.tensor(
        [
            [
                [[1.0, 2.0]],
            ]
        ]
    )

    prediction_2 = torch.tensor(
        [
            [
                [[3.0, 4.0]],
            ]
        ]
    )

    prediction_3 = torch.tensor(
        [
            [
                [[5.0, 6.0]],
            ]
        ]
    )

    stacked = torch.stack(
        [
            prediction_1,
            prediction_2,
            prediction_3,
        ],
        dim=0,
    )

    ensemble = stacked.mean(
        dim=0
    )

    expected = torch.tensor(
        [
            [
                [[3.0, 4.0]],
            ]
        ]
    )

    if not torch.allclose(
        ensemble,
        expected,
    ):

        raise AssertionError(
            "Ensemble averaging self-test failed."
        )

    accumulator = (
        DepthMetricAccumulator()
    )

    prediction = np.array(
        [[1.0, 2.0]],
        dtype=np.float64,
    )

    target = np.array(
        [[1.0, 1.0]],
        dtype=np.float64,
    )

    mask = np.array(
        [[True, True]],
        dtype=bool,
    )

    accumulator.update(
        prediction,
        target,
        mask,
    )

    metrics = accumulator.compute()

    if metrics["valid_count"] != 2:

        raise AssertionError(
            "Metric accumulator count self-test failed."
        )

    print(
        "PASS: ensemble averaging"
    )

    print(
        "PASS: metric accumulation"
    )

    print()
    print("=" * 72)
    print(
        "ENSEMBLE SELF-TEST PASSED"
    )
    print("=" * 72)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the OceanEmbed three-seed ensemble."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--history-days",
        type=int,
        choices=range(1, 8),
        default=7,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
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
    )

    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        type=Path,
        help=(
            "Checkpoint path. Supply exactly three "
            "times to override defaults."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run ensemble self-test."
        ),
    )

    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_argument_parser()

    args = parser.parse_args()

    if args.self_test:

        ensemble_self_test()

        return

    if args.checkpoints is None:

        checkpoint_paths = [
            path
            for path in DEFAULT_CHECKPOINTS
        ]

    else:

        checkpoint_paths = [
            Path(path)
            for path in args.checkpoints
        ]

    run_ensemble_evaluation(
        split=args.split,
        checkpoint_paths=checkpoint_paths,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        seed=args.seed,
        history_days=args.history_days,
    )


if __name__ == "__main__":

    main()
