"""
OceanEmbed E0 - Spatial Climatology Baseline

Baseline definition:
    For every depth and spatial grid cell, predict the mean subsurface
    temperature calculated exclusively from the training period.

Training period:
    Defined by data/processed/ML/ml_config.json

Evaluation:
    Validation and/or test periods using the same target validity masking
    and regression metrics used by the OceanEmbed evaluation pipeline.

This baseline does not use satellite inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "ML" / "ml_config.json"

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "harmonized"
    / "SubsurfaceTemp_harmonized.nc"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
    / "baselines"
    / "E0"
)


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


def load_config() -> dict:
    """Load the project ML configuration."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"ML configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_date(value: str) -> np.datetime64:
    """Convert ISO date string to numpy datetime64."""
    return np.datetime64(value, "D")


def date_to_string(value: np.datetime64) -> str:
    """Convert numpy datetime64 to YYYY-MM-DD."""
    return str(value.astype("datetime64[D]"))


def load_target_dataset() -> Dataset:
    """Open the harmonized target dataset."""
    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            f"Harmonized target dataset not found: {TARGET_PATH}"
        )

    dataset = Dataset(TARGET_PATH, "r")

    if "thetao" not in dataset.variables:
        dataset.close()
        raise ValueError(
            f"Expected variable 'thetao' not found in {TARGET_PATH}"
        )

    return dataset


def validate_target_dataset(dataset: Dataset) -> None:
    """Validate the target dataset structure."""
    required_dimensions = ["time", "depth", "latitude", "longitude"]

    for dimension in required_dimensions:
        if dimension not in dataset.dimensions:
            raise ValueError(
                f"Missing required dimension '{dimension}' "
                f"in {TARGET_PATH}"
            )

    thetao = dataset.variables["thetao"]

    if thetao.ndim != 4:
        raise ValueError(
            f"Expected thetao to have 4 dimensions, got {thetao.ndim}"
        )

    depth = np.asarray(dataset.variables["depth"][:], dtype=np.float64)

    if len(depth) != len(TARGET_DEPTHS):
        raise ValueError(
            "Target depth count mismatch: "
            f"dataset={len(depth)}, expected={len(TARGET_DEPTHS)}"
        )

    expected_depths = np.asarray(TARGET_DEPTHS, dtype=np.float64)

    if not np.allclose(depth, expected_depths, atol=1e-6):
        raise ValueError(
            "Target depth labels do not match the OceanEmbed contract.\n"
            f"Dataset depths: {depth.tolist()}\n"
            f"Expected depths: {expected_depths.tolist()}"
        )

    latitude = np.asarray(
        dataset.variables["latitude"][:],
        dtype=np.float64,
    )

    longitude = np.asarray(
        dataset.variables["longitude"][:],
        dtype=np.float64,
    )

    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("Latitude and longitude must be one-dimensional.")

    if len(latitude) != 101 or len(longitude) != 241:
        raise ValueError(
            "Unexpected target grid size. "
            f"Got {len(latitude)} x {len(longitude)}, expected 101 x 241."
        )


def get_time_values(dataset: Dataset) -> np.ndarray:
    """Return target dataset dates as YYYY-MM-DD datetime64 values."""
    time_variable = dataset.variables["time"]

    dates = []

    units = getattr(time_variable, "units", None)
    calendar = getattr(time_variable, "calendar", "standard")

    if units is None:
        raise ValueError("Target time variable has no units attribute.")

    from netCDF4 import num2date

    converted = num2date(
        time_variable[:],
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )

    for value in converted:
        dates.append(
            np.datetime64(
                value.strftime("%Y-%m-%d"),
                "D",
            )
        )

    return np.asarray(dates)


def find_date_indices(
    dates: np.ndarray,
    start_date: str,
    end_date: str,
) -> np.ndarray:
    """Return indices within an inclusive date range."""
    start = parse_date(start_date)
    end = parse_date(end_date)

    mask = (dates >= start) & (dates <= end)

    return np.flatnonzero(mask)


def calculate_spatial_climatology(
    dataset: Dataset,
    dates: np.ndarray,
    train_start: str,
    train_end: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial climatology from training data only.

    Returns:
        climatology:
            [depth, latitude, longitude]

        valid_count:
            Number of valid training observations per grid cell.
    """
    indices = find_date_indices(
        dates,
        train_start,
        train_end,
    )

    if len(indices) == 0:
        raise ValueError(
            "No training target dates found between "
            f"{train_start} and {train_end}."
        )

    thetao = dataset.variables["thetao"]

    total = np.zeros(
        (
            len(TARGET_DEPTHS),
            len(dataset.dimensions["latitude"]),
            len(dataset.dimensions["longitude"]),
        ),
        dtype=np.float64,
    )

    valid_count = np.zeros_like(total, dtype=np.int64)

    for index in indices:
        values = np.asarray(
            thetao[index, :, :, :],
            dtype=np.float64,
        )

        valid = np.isfinite(values)

        total[valid] += values[valid]
        valid_count[valid] += 1

    climatology = np.full_like(
        total,
        np.nan,
        dtype=np.float64,
    )

    valid_cells = valid_count > 0

    climatology[valid_cells] = (
        total[valid_cells]
        / valid_count[valid_cells]
    )

    return climatology, valid_count


def calculate_depth_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
) -> Dict[str, float]:
    """Calculate regression metrics for one depth."""
    valid = (
        target_mask.astype(bool)
        & np.isfinite(target)
        & np.isfinite(prediction)
    )

    count = int(np.count_nonzero(valid))

    if count == 0:
        return {
            "rmse_c": math.nan,
            "mae_c": math.nan,
            "bias_c": math.nan,
            "pearson": math.nan,
            "valid_count": 0,
        }

    pred = prediction[valid].astype(np.float64)
    obs = target[valid].astype(np.float64)

    error = pred - obs

    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))

    pred_centered = pred - np.mean(pred)
    obs_centered = obs - np.mean(obs)

    denominator = float(
        np.sqrt(
            np.sum(pred_centered ** 2)
            * np.sum(obs_centered ** 2)
        )
    )

    if denominator == 0.0:
        pearson = math.nan
    else:
        pearson = float(
            np.sum(pred_centered * obs_centered)
            / denominator
        )

    return {
        "rmse_c": float(np.sqrt(mse)),
        "mae_c": mae,
        "bias_c": bias,
        "pearson": pearson,
        "valid_count": count,
    }


def evaluate_period(
    dataset: Dataset,
    dates: np.ndarray,
    climatology: np.ndarray,
    start_date: str,
    end_date: str,
) -> dict:
    """Evaluate climatology against one target period."""
    indices = find_date_indices(
        dates,
        start_date,
        end_date,
    )

    if len(indices) == 0:
        raise ValueError(
            "No target dates found for evaluation period "
            f"{start_date} to {end_date}."
        )

    thetao = dataset.variables["thetao"]

    accumulated = []

    for depth_index in range(len(TARGET_DEPTHS)):
        predictions = []
        targets = []
        masks = []

        for index in indices:
            target = np.asarray(
                thetao[
                    index,
                    depth_index,
                    :,
                    :,
                ],
                dtype=np.float64,
            )

            prediction = climatology[depth_index]

            mask = (
                np.isfinite(target)
                & np.isfinite(prediction)
            )

            predictions.append(prediction)
            targets.append(target)
            masks.append(mask)

        prediction_array = np.stack(predictions, axis=0)
        target_array = np.stack(targets, axis=0)
        mask_array = np.stack(masks, axis=0)

        metrics = calculate_depth_metrics(
            prediction_array,
            target_array,
            mask_array,
        )

        accumulated.append(metrics)

    results = {}

    for depth, metrics in zip(TARGET_DEPTHS, accumulated):
        results[str(depth)] = metrics

    return results


def save_json(path: Path, payload: dict) -> None:
    """Save a JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=True,
        )


def print_results(
    split: str,
    results: dict,
) -> None:
    """Print baseline metrics."""
    print()
    print("=" * 80)
    print("OceanEmbed E0 CLIMATOLOGY BASELINE RESULTS")
    print("=" * 80)
    print(f"Split: {split}")
    print()
    print(
        f"{'Depth':>8} "
        f"{'RMSE °C':>14} "
        f"{'MAE °C':>14} "
        f"{'Bias °C':>14} "
        f"{'Pearson':>12} "
        f"{'Valid':>12}"
    )

    for depth in TARGET_DEPTHS:
        metrics = results[str(depth)]

        print(
            f"{depth:8.1f} "
            f"{metrics['rmse_c']:14.6f} "
            f"{metrics['mae_c']:14.6f} "
            f"{metrics['bias_c']:14.6f} "
            f"{metrics['pearson']:12.6f} "
            f"{metrics['valid_count']:12d}"
        )

    print("=" * 80)


def self_test() -> None:
    """Run deterministic E0 implementation checks."""
    print("=" * 80)
    print("OceanEmbed E0 CLIMATOLOGY BASELINE SELF-TEST")
    print("=" * 80)

    config = load_config()

    assert config["target_variable"] == "thetao"

    configured_depths = config["target_depths_m"]

    assert configured_depths == TARGET_DEPTHS

    print("PASS: ML configuration loaded.")
    print("PASS: target variable validated.")
    print("PASS: target depths validated.")

    dataset = load_target_dataset()

    try:
        validate_target_dataset(dataset)
        print("PASS: harmonized target dataset structure.")

        dates = get_time_values(dataset)

        assert len(dates) == 184
        assert date_to_string(dates[0]) == "2025-07-01"
        assert date_to_string(dates[-1]) == "2025-12-31"

        print("PASS: target time axis validated.")

        climatology, valid_count = calculate_spatial_climatology(
            dataset,
            dates,
            config["time"]["train_start"],
            config["time"]["train_end"],
        )

        assert climatology.shape == (15, 101, 241)
        assert valid_count.shape == (15, 101, 241)

        finite_count = int(
            np.count_nonzero(np.isfinite(climatology))
        )

        assert finite_count > 0
        assert np.all(valid_count >= 0)

        print("PASS: spatial training climatology calculated.")

        test_prediction = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float64,
        )

        test_target = np.array(
            [[1.0, 3.0], [2.0, 5.0]],
            dtype=np.float64,
        )

        test_mask = np.ones(
            (2, 2),
            dtype=bool,
        )

        metrics = calculate_depth_metrics(
            test_prediction,
            test_target,
            test_mask,
        )

        assert metrics["valid_count"] == 4
        assert metrics["mae_c"] >= 0.0
        assert metrics["rmse_c"] >= metrics["mae_c"]

        print("PASS: regression metric calculation.")

        invalid_mask = np.array(
            [[True, False], [True, False]],
            dtype=bool,
        )

        masked_metrics = calculate_depth_metrics(
            test_prediction,
            test_target,
            invalid_mask,
        )

        assert masked_metrics["valid_count"] == 2

        print("PASS: validity-mask handling.")

    finally:
        dataset.close()

    print()
    print("=" * 80)
    print("E0 CLIMATOLOGY BASELINE SELF-TEST PASSED")
    print("=" * 80)


def run_baseline(
    split: str,
) -> None:
    """Run E0 climatology baseline."""
    config = load_config()

    dataset = load_target_dataset()

    try:
        print("=" * 80)
        print("OceanEmbed E0 CLIMATOLOGY BASELINE")
        print("=" * 80)

        print(f"Target dataset : {TARGET_PATH}")
        print(
            "Training period: "
            f"{config['time']['train_start']} → "
            f"{config['time']['train_end']}"
        )

        if split == "validation":
            start_date = config["time"]["validation_start"]
            end_date = config["time"]["validation_end"]
        elif split == "test":
            start_date = config["time"]["test_start"]
            end_date = config["time"]["test_end"]
        else:
            raise ValueError(
                f"Unsupported split: {split}"
            )

        print(
            f"Evaluation period: {start_date} → {end_date}"
        )

        validate_target_dataset(dataset)

        dates = get_time_values(dataset)

        print()
        print("Calculating spatial training climatology...")

        climatology, valid_count = (
            calculate_spatial_climatology(
                dataset,
                dates,
                config["time"]["train_start"],
                config["time"]["train_end"],
            )
        )

        print("Training climatology calculated.")

        results = evaluate_period(
            dataset,
            dates,
            climatology,
            start_date,
            end_date,
        )

        print_results(
            split,
            results,
        )

        output_path = (
            OUTPUT_ROOT
            / f"e0_climatology_{split}.json"
        )

        payload = {
            "experiment": "E0",
            "name": "spatial_climatology",
            "description": (
                "Training-period spatial mean thetao baseline."
            ),
            "split": split,
            "target_variable": "thetao",
            "units": "degC",
            "target_depths_m": TARGET_DEPTHS,
            "training_period": {
                "start": config["time"]["train_start"],
                "end": config["time"]["train_end"],
            },
            "evaluation_period": {
                "start": start_date,
                "end": end_date,
            },
            "grid": {
                "latitude_count": 101,
                "longitude_count": 241,
                "resolution_deg": 0.25,
            },
            "training_valid_cells_by_depth": {
                str(depth): int(
                    np.count_nonzero(
                        valid_count[
                            depth_index
                        ] > 0
                    )
                )
                for depth_index, depth in enumerate(
                    TARGET_DEPTHS
                )
            },
            "metrics": results,
        }

        save_json(
            output_path,
            payload,
        )

        print()
        print(f"Results saved: {output_path}")

    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OceanEmbed E0 spatial climatology baseline."
        )
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run E0 implementation self-test.",
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Evaluation split.",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_baseline(
        split=args.split,
    )


if __name__ == "__main__":
    main()