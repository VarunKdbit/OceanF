"""
OceanEmbed Scientific Diagnostics

Runs one complete diagnostic inference pass using the official
three-seed OceanEmbed-CNN ensemble.

Diagnostics:
    1. Depth-wise RMSE / MAE / Bias / Pearson
    2. Spatial tile error statistics
    3. Temporal error statistics
    4. Training-mean anomaly diagnostics
    5. Three-seed ensemble spread
    6. Prediction and dataset consistency checks

This script does NOT train models.
This script does NOT modify the official ensemble evaluation JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

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
    / "diagnostics"
)

ML_CONFIG_PATH = ML_DIR / "ml_config.json"


DEFAULT_CHECKPOINTS = [
    CHECKPOINT_DIR / "oceanembed_e2_seed42.pt",
    CHECKPOINT_DIR / "oceanembed_e2_seed123.pt",
    CHECKPOINT_DIR / "oceanembed_e2_seed2024.pt",
]


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
    """Load ML configuration."""

    with ML_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_target_statistics(
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load target mean/std for all 15 depths."""

    statistics = config["target_statistics"]

    means = []
    stds = []

    for depth in TARGET_DEPTHS:

        key = str(depth)

        if key not in statistics:
            raise KeyError(
                f"Missing target statistics for depth {depth} m."
            )

        mean = float(
            statistics[key]["mean"]
        )

        std = float(
            statistics[key]["std"]
        )

        if not np.isfinite(mean):
            raise ValueError(
                f"Invalid target mean at {depth} m."
            )

        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(
                f"Invalid target std at {depth} m."
            )

        means.append(mean)
        stds.append(std)

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
    """Convert normalized target values to degrees Celsius."""

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
# Metric accumulator
# ---------------------------------------------------------------------

class MetricAccumulator:
    """Accumulate exact dataset-level regression statistics."""

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
            np.sum(
                np.abs(error)
            )
        )

        self.sum_squared_error += float(
            np.sum(
                error * error
            )
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
                    max(
                        mse,
                        0.0,
                    )
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
# Helper functions
# ---------------------------------------------------------------------

def validate_checkpoint_paths(
    checkpoint_paths: list[Path],
) -> None:
    """Require exactly the three production checkpoints."""

    if len(checkpoint_paths) != 3:
        raise ValueError(
            "OceanEmbed diagnostics require exactly "
            "three checkpoints."
        )

    for checkpoint_path in checkpoint_paths:

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )


def to_python_value(
    value: Any,
) -> Any:
    """Convert common tensor/numpy metadata values to Python values."""

    if isinstance(
        value,
        torch.Tensor,
    ):

        if value.numel() == 1:
            return value.item()

        return value.detach().cpu().tolist()

    if isinstance(
        value,
        np.ndarray,
    ):

        if value.size == 1:
            return value.item()

        return value.tolist()

    return value


def normalize_metadata_value(
    value: Any,
) -> Any:
    """Convert metadata into a clean Python scalar/string/list."""

    value = to_python_value(value)

    if isinstance(
        value,
        (list, tuple),
    ):

        return [
            normalize_metadata_value(
                item
            )
            for item in value
        ]

    return value


def get_batch_metadata_value(
    metadata: Any,
    key: str,
    index: int,
) -> Any:
    """
    Safely extract one sample's metadata value.

    Supports dictionaries containing lists/tensors/arrays and
    scalar metadata values.
    """

    if not isinstance(
        metadata,
        dict,
    ):
        return None

    if key not in metadata:
        return None

    value = metadata[key]

    value = to_python_value(value)

    if isinstance(
        value,
        (list, tuple),
    ):

        if index < len(value):
            return normalize_metadata_value(
                value[index]
            )

        return None

    return normalize_metadata_value(
        value
    )


def metadata_date(
    metadata: Any,
    index: int,
) -> Optional[str]:
    """Extract a date from dataset metadata."""

    for key in [
        "date",
        "target_date",
        "sample_date",
    ]:

        value = get_batch_metadata_value(
            metadata,
            key,
            index,
        )

        if value is not None:

            return str(value)

    return None


def metadata_tile(
    metadata: Any,
    index: int,
) -> Optional[str]:
    """
    Extract tile identity when available.

    The dataset may expose different names for tile coordinates.
    """

    row = None
    col = None

    for key in [
        "tile_row",
        "target_tile_row",
        "row",
    ]:

        value = get_batch_metadata_value(
            metadata,
            key,
            index,
        )

        if value is not None:
            row = value
            break

    for key in [
        "tile_col",
        "target_tile_col",
        "col",
    ]:

        value = get_batch_metadata_value(
            metadata,
            key,
            index,
        )

        if value is not None:
            col = value
            break

    if row is not None and col is not None:

        return (
            f"row_{row}_col_{col}"
        )

    for key in [
        "tile_index",
        "target_tile_index",
    ]:

        value = get_batch_metadata_value(
            metadata,
            key,
            index,
        )

        if value is not None:

            return (
                f"tile_{value}"
            )

    return None


def safe_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except (
        TypeError,
        ValueError,
    ):
        pass

    return None


# ---------------------------------------------------------------------
# Main diagnostic evaluation
# ---------------------------------------------------------------------

def run_diagnostics(
    split: str,
    checkpoint_paths: list[Path],
    batch_size: int,
    max_batches: Optional[int],
    seed: int,
    history_days: int,
) -> dict:
    """Run one complete three-seed diagnostic inference pass."""

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

    set_random_seed(seed)

    device = get_device()

    config = load_ml_config()

    target_means, target_stds = (
        load_target_statistics(
            config
        )
    )

    print()
    print("=" * 72)
    print("OceanEmbed SCIENTIFIC DIAGNOSTICS")
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
        "Ensemble size         : 3"
    )

    print()

    for checkpoint_path in checkpoint_paths:

        print(
            f"Checkpoint            : "
            f"{checkpoint_path}"
        )

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    print()
    print(
        "Creating diagnostic dataset..."
    )

    dataset = OceanEmbedDataset(
        split=split,
        tile_stride=32,
        history_days=history_days,
        return_metadata=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(
        f"Dataset samples       : "
        f"{len(dataset)}"
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

        model.to(device)

        metadata = load_checkpoint(
            checkpoint_path,
            model,
            device=device,
        )

        model.eval()

        models.append(model)

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
            f"Loaded: "
            f"{checkpoint_path.name}"
        )

    print()
    print(
        "All three checkpoints loaded successfully."
    )

    # -------------------------------------------------------------
    # Accumulators
    # -------------------------------------------------------------

    depth_accumulators = [
        MetricAccumulator()
        for _ in TARGET_DEPTHS
    ]

    spread_sums = np.zeros(
        len(TARGET_DEPTHS),
        dtype=np.float64,
    )

    spread_counts = np.zeros(
        len(TARGET_DEPTHS),
        dtype=np.int64,
    )

    # Training-period-mean anomaly diagnostics.
    anomaly_accumulators = [
        MetricAccumulator()
        for _ in TARGET_DEPTHS
    ]

    temporal_accumulators: dict[
        str,
        list[MetricAccumulator],
    ] = {}

    tile_accumulators: dict[
        str,
        list[MetricAccumulator],
    ] = {}

    processed_batches = 0

    consistency_checks = {
        "nonfinite_predictions": 0,
        "shape_errors": 0,
        "empty_target_masks": 0,
        "metadata_dates_found": 0,
        "metadata_dates_missing": 0,
        "metadata_tiles_found": 0,
        "metadata_tiles_missing": 0,
    }

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    with torch.no_grad():

        for batch_index, batch in enumerate(loader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            if len(batch) != 5:
                raise RuntimeError(
                    "Expected dataset batch with five "
                    "items: x, x_mask, y, y_mask, metadata."
                )

            x, _, y, y_mask, metadata = batch

            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)

            predictions = []

            for model in models:

                prediction = model(x)

                expected_shape = (
                    x.shape[0],
                    15,
                    32,
                    32,
                )

                if tuple(
                    prediction.shape
                ) != expected_shape:

                    consistency_checks[
                        "shape_errors"
                    ] += 1

                    raise RuntimeError(
                        "Unexpected model output "
                        f"shape: {prediction.shape}. "
                        f"Expected {expected_shape}."
                    )

                if not torch.isfinite(
                    prediction
                ).all():

                    consistency_checks[
                        "nonfinite_predictions"
                    ] += 1

                    raise RuntimeError(
                        "Non-finite prediction "
                        "encountered."
                    )

                predictions.append(
                    prediction
                )

            stacked_predictions = torch.stack(
                predictions,
                dim=0,
            )

            ensemble_prediction = (
                stacked_predictions.mean(
                    dim=0
                )
            )

            ensemble_spread = (
                stacked_predictions.std(
                    dim=0,
                    unbiased=False,
                )
            )

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

            spread_c = (
                ensemble_spread
                * target_stds.to(
                    device=device,
                    dtype=ensemble_spread.dtype,
                ).view(
                    1,
                    -1,
                    1,
                    1,
                )
            )

            prediction_np = (
                prediction_c
               .cpu()
                .numpy()
            )

            target_np = (
                target_c
                .cpu()
                .numpy()
            )

            spread_np = (
                spread_c
                .cpu()
                .numpy()
            )

            mask_np = (
                y_mask
                .cpu()
                .numpy()
                .astype(bool)
            )

            # -----------------------------------------------------
            # Per-sample metadata diagnostics
            # -----------------------------------------------------

            batch_size_actual = x.shape[0]

            for sample_index in range(
                batch_size_actual
            ):

                sample_has_valid_target = bool(
                    np.any(
                        mask_np[
                            sample_index
                        ]
                    )
                )

                if not sample_has_valid_target:

                    consistency_checks[
                        "empty_target_masks"
                    ] += 1

            # -----------------------------------------------------
            # Per-depth diagnostics
            # -----------------------------------------------------

            for depth_index, depth in enumerate(
                TARGET_DEPTHS
            ):

                prediction_depth = (
                    prediction_np[
                        :,
                        depth_index,
                    ]
                )

                target_depth = (
                    target_np[
                        :,
                        depth_index,
                    ]
                )

                mask_depth = (
                    mask_np[
                        :,
                        depth_index,
                    ]
                )

                depth_accumulators[
                    depth_index
                ].update(
                    prediction_depth,
                    target_depth,
                    mask_depth,
                )

                # -------------------------------------------------
                # Training-mean anomaly diagnostics
                #
                # Anomaly reference = training-period target mean
                # stored in ml_config.json.
                # -------------------------------------------------

                training_mean = float(
                    target_means[
                        depth_index
                    ].item()
                )

                prediction_anomaly = (
                    prediction_depth
                    - training_mean
                )

                target_anomaly = (
                    target_depth
                    - training_mean
                )

                anomaly_accumulators[
                    depth_index
                ].update(
                    prediction_anomaly,
                    target_anomaly,
                    mask_depth,
                )

                # -------------------------------------------------
                # Ensemble spread
                # -------------------------------------------------

                valid_spread = spread_np[
                    :,
                    depth_index,
                ][mask_depth]

                if valid_spread.size > 0:

                    spread_sums[
                        depth_index
                    ] += float(
                        np.sum(
                            valid_spread
                        )
                    )

                    spread_counts[
                        depth_index
                    ] += int(
                        valid_spread.size
                    )

                # -------------------------------------------------
                # Temporal diagnostics
                # -------------------------------------------------

                for sample_index in range(
                    batch_size_actual
                ):

                    date = metadata_date(
                        metadata,
                        sample_index,
                    )

                    if date is None:

                        consistency_checks[
                            "metadata_dates_missing"
                        ] += 1

                        continue

                    consistency_checks[
                        "metadata_dates_found"
                    ] += 1

                    if date not in temporal_accumulators:

                        temporal_accumulators[
                            date
                        ] = [
                            MetricAccumulator()
                            for _ in TARGET_DEPTHS
                        ]

                    temporal_accumulators[
                        date
                    ][depth_index].update(
                        prediction_depth[
                            sample_index
                        ],
                        target_depth[
                            sample_index
                        ],
                        mask_depth[
                            sample_index
                        ],
                    )

                    # -------------------------------------------------
                    # Spatial tile diagnostics
                    # -------------------------------------------------

                    tile = metadata_tile(
                        metadata,
                        sample_index,
                    )

                    if tile is None:

                        consistency_checks[
                            "metadata_tiles_missing"
                        ] += 1

                        continue

                    consistency_checks[
                        "metadata_tiles_found"
                    ] += 1

                    if tile not in tile_accumulators:

                        tile_accumulators[
                            tile
                        ] = [
                            MetricAccumulator()
                            for _ in TARGET_DEPTHS
                        ]

                    tile_accumulators[
                        tile
                    ][depth_index].update(
                        prediction_depth[
                            sample_index
                        ],
                        target_depth[
                            sample_index
                        ],
                        mask_depth[
                            sample_index
                        ],
                    )

            processed_batches += 1

            if (
                processed_batches == 1
                or processed_batches % 25 == 0
            ):

                print(
                    f"Processed batches: "
                    f"{processed_batches}"
                )

    if processed_batches == 0:

        raise RuntimeError(
            "No diagnostic batches were processed."
        )

    # -------------------------------------------------------------
    # Depth metrics
    # -------------------------------------------------------------

    depth_metrics = []

    for depth_index, depth in enumerate(
        TARGET_DEPTHS
    ):

        metrics = (
            depth_accumulators[
                depth_index
            ].compute()
        )

        metrics["depth_m"] = float(
            depth
        )

        depth_metrics.append(
            metrics
        )

    # -------------------------------------------------------------
    # Anomaly metrics
    # -------------------------------------------------------------

    anomaly_metrics = []

    for depth_index, depth in enumerate(
        TARGET_DEPTHS
    ):

        metrics = (
            anomaly_accumulators[
                depth_index
            ].compute()
        )

        anomaly_metrics.append(
            {
                "depth_m": float(
                    depth
                ),
                "anomaly_rmse_degC": metrics[
                    "rmse"
                ],
                "anomaly_mae_degC": metrics[
                    "mae"
                ],
                "anomaly_bias_degC": metrics[
                    "bias"
                ],
                "anomaly_pearson": metrics[
                    "pearson"
                ],
                "valid_count": metrics[
                    "valid_count"
                ],
                "reference_mean": float(
                    target_means[
                        depth_index
                    ].item()
                ),
            }
        )

    # -------------------------------------------------------------
    # Ensemble spread
    # -------------------------------------------------------------

    ensemble_spread_metrics = []

    for depth_index, depth in enumerate(
        TARGET_DEPTHS
    ):

        count = int(
            spread_counts[
                depth_index
            ]
        )

        if count == 0:
            mean_spread = None
        else:
            mean_spread = float(
                spread_sums[
                    depth_index
                ]
                / count
            )

        ensemble_spread_metrics.append(
            {
                "depth_m": float(
                    depth
                ),
                "mean_prediction_spread_degC":
                    mean_spread,
                "valid_count": count,
            }
        )

    # -------------------------------------------------------------
    # Temporal metrics
    # -------------------------------------------------------------

    temporal_metrics = []

    for date in sorted(
        temporal_accumulators.keys()
    ):

        per_depth = []

        for depth_index, depth in enumerate(
            TARGET_DEPTHS
        ):

            metrics = (
                temporal_accumulators[
                    date
                ][depth_index].compute()
            )

            per_depth.append(
                {
                    "depth_m": float(
                        depth
                    ),
                    **metrics,
                }
            )

        temporal_metrics.append(
            {
                "date": date,
                "depth_metrics": per_depth,
            }
        )

    # -------------------------------------------------------------
    # Spatial tile metrics
    # -------------------------------------------------------------

    spatial_tile_metrics = []

    for tile in sorted(
        tile_accumulators.keys()
    ):

        per_depth = []

        for depth_index, depth in enumerate(
            TARGET_DEPTHS
        ):

            metrics = (
                tile_accumulators[
                    tile
                ][depth_index].compute()
            )

            per_depth.append(
                {
                    "depth_m": float(
                        depth
                    ),
                    **metrics,
                }
            )

        spatial_tile_metrics.append(
            {
                "tile": tile,
                "depth_metrics": per_depth,
            }
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    rmse_values = [
        result["rmse"]
        for result in depth_metrics
        if result["rmse"] is not None
    ]

    mae_values = [
        result["mae"]
        for result in depth_metrics
        if result["mae"] is not None
    ]

    pearson_values = [
        result["pearson"]
        for result in depth_metrics
        if result["pearson"] is not None
    ]

    if rmse_values:

        best_rmse_index = int(
            np.argmin(
                rmse_values
            )
        )

        worst_rmse_index = int(
            np.argmax(
                rmse_values
            )
        )

        best_rmse_depth = float(
            depth_metrics[
                best_rmse_index
            ]["depth_m"]
        )

        worst_rmse_depth = float(
            depth_metrics[
                worst_rmse_index
            ]["depth_m"]
        )

    else:

        best_rmse_depth = None
        worst_rmse_depth = None

    summary = {
        "mean_depth_rmse_degC": (
            float(
                np.mean(
                    rmse_values
                )
            )
            if rmse_values
            else None
        ),
        "mean_depth_mae_degC": (
            float(
                np.mean(
                    mae_values
                )
            )
            if mae_values
            else None
        ),
        "mean_depth_pearson": (
            float(
                np.mean(
                    pearson_values
                )
            )
            if pearson_values
            else None
        ),
        "best_rmse_depth_m": (
            best_rmse_depth
        ),
        "worst_rmse_depth_m": (
            worst_rmse_depth
        ),
        "temporal_dates_available": int(
            len(temporal_metrics)
        ),
        "spatial_tiles_available": int(
            len(spatial_tile_metrics)
        ),
    }

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    output = {
        "project": "OceanEmbed",
        "experiment": "E2_7day_retrospective",
        "evaluation_type": (
            "scientific_diagnostics"
        ),
        "split": split,
        "device": str(device),
        "history_days": history_days,
        "batch_size": batch_size,
        "processed_batches": (
            processed_batches
        ),
        "dataset_size": len(dataset),
        "ensemble_size": 3,
        "ensemble_seeds": [
            42,
            123,
            2024,
        ],
        "ensemble_method": (
            "arithmetic_mean_of_three_model_predictions"
        ),
        "target_depths_m": TARGET_DEPTHS,
        "metrics_units": {
            "rmse": "degC",
            "mae": "degC",
            "bias": "degC",
            "pearson": "dimensionless",
            "anomaly_rmse": "degC",
            "ensemble_spread": "degC",
        },
        "depth_metrics": depth_metrics,
        "anomaly_diagnostics": anomaly_metrics,
        "ensemble_spread": (
            ensemble_spread_metrics
        ),
        "temporal_diagnostics": (
            temporal_metrics
        ),
        "spatial_tile_diagnostics": (
            spatial_tile_metrics
        ),
        "summary": summary,
        "consistency_checks": (
            consistency_checks
        ),
        "checkpoint_metadata": (
            checkpoint_metadata
        ),
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
        / "seed42_seed123_seed2024_diagnostics.json"
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

    # -------------------------------------------------------------
    # Console results
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("DEPTH-WISE DIAGNOSTICS")
    print("=" * 72)

    print(
        f"{'Depth':>8} "
        f"{'RMSE':>10} "
        f"{'MAE':>10} "
        f"{'Bias':>10} "
        f"{'Pearson':>10} "
        f"{'Spread':>10}"
    )

    print("-" * 72)

    for depth_index, result in enumerate(
        depth_metrics
    ):

        spread = (
            ensemble_spread_metrics[
                depth_index
            ]["mean_prediction_spread_degC"]
        )

        spread_text = (
            f"{spread:.4f}"
            if spread is not None
            else "N/A"
        )

        print(
            f"{result['depth_m']:8.1f} "
            f"{result['rmse']:10.4f} "
            f"{result['mae']:10.4f} "
            f"{result['bias']:10.4f} "
            f"{result['pearson']:10.4f} "
            f"{spread_text:>10}"
        )

    print()
    print("=" * 72)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 72)

    print(
        f"Mean depth RMSE      : "
        f"{summary['mean_depth_rmse_degC']:.4f} °C"
    )

    print(
        f"Mean depth MAE       : "
        f"{summary['mean_depth_mae_degC']:.4f} °C"
    )

    print(
        f"Mean depth Pearson   : "
        f"{summary['mean_depth_pearson']:.4f}"
    )

    print(
        f"Best RMSE depth      : "
        f"{summary['best_rmse_depth_m']} m"
    )

    print(
        f"Worst RMSE depth     : "
        f"{summary['worst_rmse_depth_m']} m"
    )

    print(
        f"Temporal dates       : "
        f"{summary['temporal_dates_available']}"
    )

    print(
        f"Spatial tiles        : "
        f"{summary['spatial_tiles_available']}"
    )

    print()
    print(
        f"Results saved: "
        f"{output_path}"
    )

    return output


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def diagnostic_self_test() -> None:
    """Run lightweight mathematical self-tests."""

    print()
    print("=" * 72)
    print("OceanEmbed DIAGNOSTIC SELF-TEST")
    print("=" * 72)

    prediction = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    target = np.array(
        [1.0, 1.0, 2.0],
        dtype=np.float64,
    )

    mask = np.array(
        [True, True, True],
        dtype=bool,
    )

    accumulator = MetricAccumulator()

    accumulator.update(
        prediction,
        target,
        mask,
    )

    metrics = accumulator.compute()

    if metrics["valid_count"] != 3:
        raise AssertionError(
            "Metric accumulator count "
            "self-test failed."
        )

    if not np.isclose(
        metrics["bias"],
        2.0 / 3.0,
    ):
        raise AssertionError(
            "Bias self-test failed."
        )

    if not np.isclose(
        metrics["mae"],
        2.0 / 3.0,
    ):
        raise AssertionError(
            "MAE self-test failed."
        )

    if not np.isclose(
        metrics["rmse"],
        np.sqrt(
            2.0 / 3.0
        ),
    ):
        raise AssertionError(
            "RMSE self-test failed."
        )

    # Metadata helper test.
    metadata = {
        "date": [
            "2025-12-01",
            "2025-12-02",
        ],
        "tile_row": [0, 1],
        "tile_col": [2, 3],
    }

    if metadata_date(
        metadata,
        0,
    ) != "2025-12-01":
        raise AssertionError(
            "Date metadata self-test failed."
        )

    if metadata_tile(
        metadata,
        1,
    ) != "row_1_col_3":
        raise AssertionError(
            "Tile metadata self-test failed."
        )

    print(
        "PASS: metric accumulator"
    )

    print(
        "PASS: metadata extraction"
    )

    print()
    print("=" * 72)
    print(
        "DIAGNOSTIC SELF-TEST PASSED"
    )
    print("=" * 72)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run OceanEmbed scientific diagnostics."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="test",
        help="Dataset split to diagnose.",
    )

    parser.add_argument(
        "--history-days",
        type=int,
        choices=range(1, 8),
        default=7,
        help="Retrospective input history.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size.",
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
        help="Run diagnostic self-tests.",
    )

    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_argument_parser()

    args = parser.parse_args()

    if args.self_test:

        diagnostic_self_test()

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

    run_diagnostics(
        split=args.split,
        checkpoint_paths=checkpoint_paths,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        seed=args.seed,
        history_days=args.history_days,
    )


if __name__ == "__main__":

    main()