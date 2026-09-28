from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import gsw
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
HARMONIZED_ROOT = PROJECT_ROOT / "data" / "processed" / "ML" / "harmonized"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "argo_validation"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "ML" / "argo_validation"
V1_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "ML" / "ml_config.json"
INDEX_URL = "https://data-argo.ifremer.fr/ar_index_global_prof.txt.gz"
GDAC_ROOT = "https://data-argo.ifremer.fr/dac"

VALIDATION_START = date(2025, 12, 1)
VALIDATION_END = date(2025, 12, 31)
DOMAIN = {
    "latitude_min": 5.0,
    "latitude_max": 30.0,
    "longitude_min": 45.0,
    "longitude_max": 105.0,
}
GRID_RESOLUTION_DEG = 0.25
SPATIAL_THRESHOLD_DEG = 0.5
METERS_PER_DEGREE = 111.195
TARGET_DEPTHS_M = [
    0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000,
]
INPUT_FEATURES = ["sst", "sss", "sla", "uo", "vo", "u_wind", "v_wind"]
FEATURE_FILES = {
    "sst": ("SST_harmonized.nc", "sst"),
    "sss": ("SSS_harmonized.nc", "sss"),
    "sla": ("SLA_harmonized.nc", "sla"),
    "uo": ("Currents_harmonized.nc", "uo"),
    "vo": ("Currents_harmonized.nc", "vo"),
    "u_wind": ("Winds_harmonized.nc", "u_wind"),
    "v_wind": ("Winds_harmonized.nc", "v_wind"),
}
CSV_MATCHUP_FIELDS = [
    "profile_id",
    "argo_date",
    "argo_lat",
    "argo_lon",
    "model_lat",
    "model_lon",
    "spatial_distance_km",
    "depth_m",
    "argo_temperature_c",
    "oceanembed_temperature_c",
    "ensemble_spread_c",
    "temperature_error_c",
]
CSV_CATALOG_FIELDS = [
    "profile_id",
    "float_id",
    "date",
    "latitude",
    "longitude",
    "data_mode",
    "adjusted_available",
    "valid_depth_count",
    "matched_depth_count",
    "exclusion_reason",
]

LOGGER = logging.getLogger("argo_validation")


@dataclass
class ProfileCandidate:
    file: str
    profile_id: str
    argo_date: date
    latitude: float
    longitude: float
    float_id: str = ""
    data_mode: str = "unknown"
    adjusted_available: bool = False
    valid_depth_count: int = 0
    matched_depth_count: int = 0
    exclusion_reason: str = ""
    model_latitude: float | None = None
    model_longitude: float | None = None
    spatial_distance_km: float | None = None
    grid_row: int | None = None
    grid_column: int | None = None
    tile_row: int | None = None
    tile_column: int | None = None
    output_row: int | None = None
    output_column: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently validate OceanEmbed V1 predictions against "
            "adjusted, QC=1 official Argo profiles."
        )
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=VALIDATION_START)
    parser.add_argument("--end-date", type=date.fromisoformat, default=VALIDATION_END)
    parser.add_argument(
        "--max-distance-deg",
        type=float,
        default=SPATIAL_THRESHOLD_DEG,
        help="Maximum great-circle profile-to-grid separation in degrees.",
    )
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gdac-url", default=GDAC_ROOT)
    parser.add_argument("--index-url", default=INDEX_URL)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-profiles", action="store_true")
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Fetch/read the official index and report candidates without profiles or metrics.",
    )
    return parser.parse_args()


def validate_heldout_contract(start_date: date, end_date: date) -> None:
    with V1_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    time_config = config.get("time", {})
    if (
        time_config.get("test_start") != VALIDATION_START.isoformat()
        or time_config.get("test_end") != VALIDATION_END.isoformat()
    ):
        raise ValueError("ml_config.json test split differs from the fixed ARGO period")
    if time_config.get("train_end", "9999-12-31") >= start_date.isoformat():
        raise ValueError("The ARGO validation period overlaps OceanEmbed training data")
    if time_config.get("validation_end", "9999-12-31") >= start_date.isoformat():
        raise ValueError("The ARGO validation period overlaps OceanEmbed model selection data")
    if start_date < VALIDATION_START or end_date > VALIDATION_END:
        raise ValueError("ARGO dates must remain inside the fixed December 2025 test split")
    if config.get("normalization", {}).get("inputs") != "training_period_mean_std":
        raise ValueError("V1 input normalization is not declared train-only")
    if config.get("normalization", {}).get("target") != "training_period_mean_std_by_depth":
        raise ValueError("V1 target normalization is not declared train-only")


def iso_date(value: str) -> date:
    value = value.strip()
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        for date_format in ("%Y%m%d%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(value, date_format).date()
            except ValueError:
                continue
    raise ValueError(f"Unsupported GDAC profile date: {value}")


def normalized_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").strip().replace("\x00", "")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("ascii", errors="replace").strip().replace("\x00", "")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return normalized_text(value.item())
    return str(value).strip().replace("\x00", "")


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OceanEmbed-Argo-Validation/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if temporary.stat().st_size == 0:
                raise OSError("GDAC returned an empty file")
            temporary.replace(destination)
            LOGGER.info("Downloaded %s (%d bytes)", url, destination.stat().st_size)
            return destination
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            LOGGER.warning("Download attempt %d/3 failed for %s: %s", attempt, url, exc)
            if attempt < 3:
                time.sleep(attempt * 2)
    raise OSError(f"Unable to download {url}: {last_error}") from last_error


def index_rows(index_path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if index_path.suffix == ".gz" else open
    with opener(index_path, "rt", encoding="utf-8", errors="replace", newline="") as stream:
        header: str | None = None
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            header = line
            break
        if header is None:
            raise ValueError(f"GDAC index has no CSV header: {index_path}")
        reader = csv.DictReader(chain([header], stream), skipinitialspace=True)
        for raw in reader:
            row = {
                str(key).strip().lower(): (value or "").strip()
                for key, value in raw.items()
                if key is not None
            }
            yield row


def get_column(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        if row.get(key, ""):
            return row[key]
    return default


def float_column(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
        return value if np.isfinite(value) else None
    except (KeyError, ValueError, TypeError):
        return None


def preferred_mode_key(row: dict[str, str]) -> tuple[int, str, str]:
    mode = get_column(row, "parameter_data_mode", "data_mode").upper()
    return (0 if "D" in mode else 1, mode, get_column(row, "file"))


def nearest_grid_coordinate(value: float, minimum: float, maximum: float) -> tuple[int, float]:
    count = round((maximum - minimum) / GRID_RESOLUTION_DEG) + 1
    index = int(np.clip(round((value - minimum) / GRID_RESOLUTION_DEG), 0, count - 1))
    return index, minimum + index * GRID_RESOLUTION_DEG


def great_circle_distance_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    radius_km = 6371.0088
    lat_a, lat_b = np.radians([latitude_a, latitude_b])
    delta_lat = lat_b - lat_a
    delta_lon = np.radians(longitude_b - longitude_a)
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2) ** 2
    )
    return float(2 * radius_km * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0))))


def output_tile_for_grid_point(
    row: int,
    column: int,
    latitude_count: int,
    longitude_count: int,
) -> tuple[int, int, int, int] | None:
    """Find an existing 64x64 input tile whose centered 32x32 output contains the point."""
    row_starts = range(0, max(0, latitude_count - 63), 32)
    column_starts = range(0, max(0, longitude_count - 63), 32)
    for tile_row in row_starts:
        output_row = row - tile_row - 16
        if not 0 <= output_row < 32:
            continue
        for tile_column in column_starts:
            output_column = column - tile_column - 16
            if 0 <= output_column < 32:
                return tile_row, tile_column, output_row, output_column
    return None


def discover_candidates(
    index_path: Path,
    start_date: date,
    end_date: date,
    max_distance_deg: float,
) -> list[ProfileCandidate]:
    maximum_distance_km = max_distance_deg * METERS_PER_DEGREE
    padded_lat_min = DOMAIN["latitude_min"] - max_distance_deg
    padded_lat_max = DOMAIN["latitude_max"] + max_distance_deg
    padded_lon_min = DOMAIN["longitude_min"] - max_distance_deg
    padded_lon_max = DOMAIN["longitude_max"] + max_distance_deg
    candidates: dict[tuple[str, str], ProfileCandidate] = {}

    for row in index_rows(index_path):
        profile_path = get_column(row, "file")
        date_text = get_column(row, "date")
        latitude = float_column(row, "latitude")
        longitude = float_column(row, "longitude")
        if not profile_path or not date_text or latitude is None or longitude is None:
            continue
        try:
            profile_date = iso_date(date_text)
        except ValueError:
            continue
        if not start_date <= profile_date <= end_date:
            continue
        if not (
            padded_lat_min <= latitude <= padded_lat_max
            and padded_lon_min <= longitude <= padded_lon_max
        ):
            continue

        row_index, model_latitude = nearest_grid_coordinate(
            latitude, DOMAIN["latitude_min"], DOMAIN["latitude_max"]
        )
        column_index, model_longitude = nearest_grid_coordinate(
            longitude, DOMAIN["longitude_min"], DOMAIN["longitude_max"]
        )
        distance = great_circle_distance_km(
            latitude, longitude, model_latitude, model_longitude
        )
        profile_id = f"{profile_path}#{date_text}"
        candidate = ProfileCandidate(
            file=profile_path,
            profile_id=profile_id,
            argo_date=profile_date,
            latitude=latitude,
            longitude=longitude,
            data_mode=(
                "D" if PurePosixPath(profile_path).name.upper().startswith("D")
                else "R" if PurePosixPath(profile_path).name.upper().startswith("R")
                else "unknown"
            ),
            model_latitude=model_latitude,
            model_longitude=model_longitude,
            spatial_distance_km=distance,
            grid_row=row_index,
            grid_column=column_index,
        )

        if distance > maximum_distance_km:
            candidate.exclusion_reason = "spatial_distance_exceeds_threshold"
        else:
            tile = output_tile_for_grid_point(row_index, column_index, 101, 241)
            if tile is None:
                candidate.exclusion_reason = "outside_model_output_coverage"
            else:
                (
                    candidate.tile_row,
                    candidate.tile_column,
                    candidate.output_row,
                    candidate.output_column,
                ) = tile

        candidates[(profile_path, date_text)] = candidate

    return sorted(candidates.values(), key=lambda item: (preferred_mode_key({"file": item.file, "parameter_data_mode": item.data_mode}), item.profile_id))


def safe_profile_path(relative_path: str) -> Path:
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"Unsafe GDAC profile path: {relative_path}")
    return RAW_ROOT / "profiles" / Path(*pure_path.parts)


def profile_url(gdac_url: str, relative_path: str) -> str:
    normalized_root = gdac_url.rstrip("/")
    path = urllib.parse.quote(relative_path.lstrip("/"), safe="/")
    return f"{normalized_root}/{path}"


def profile_mode(dataset: xr.Dataset, profile_index: int, index_mode: str) -> str:
    if "DATA_MODE" not in dataset:
        return index_mode or "unknown"
    variable = dataset["DATA_MODE"]
    value = variable.values
    if value.ndim == 0:
        return normalized_text(value)
    if value.ndim > 1:
        value = value[profile_index]
    else:
        value = value[min(profile_index, value.shape[0] - 1)]
    return normalized_text(value)


def select_profile_position(dataset: xr.Dataset, profile_path: str) -> int:
    count = int(dataset.sizes.get("N_PROF", 1))
    cycle_match = re.search(r"_(\d+)\.nc$", profile_path, flags=re.IGNORECASE)
    if count <= 1 or cycle_match is None or "CYCLE_NUMBER" not in dataset:
        return 0
    cycle = int(cycle_match.group(1))
    values = np.asarray(dataset["CYCLE_NUMBER"].values).reshape(-1)
    matches = np.flatnonzero(values == cycle)
    if len(matches) == 1:
        return int(matches[0])
    return 0


def profile_vector(dataset: xr.Dataset, variable_name: str, profile_position: int) -> np.ndarray:
    values = np.asarray(dataset[variable_name].values)
    if values.ndim >= 2:
        values = values[profile_position]
    return np.asarray(values).reshape(-1)


def profile_float_id(dataset: xr.Dataset, profile_position: int, profile_path: str) -> str:
    if "PLATFORM_NUMBER" not in dataset.variables:
        return PurePosixPath(profile_path).stem.split("_")[0]
    values = np.asarray(dataset["PLATFORM_NUMBER"].values)
    if values.ndim == 0:
        return normalized_text(values)
    values = values[min(profile_position, values.shape[0] - 1)]
    if np.asarray(values).ndim > 0 and np.asarray(values).dtype.kind in {"S", "U"}:
        return "".join(normalized_text(value) for value in np.asarray(values).reshape(-1)).strip()
    return normalized_text(values)


def strict_qc_mask(qc_values: np.ndarray, expected_length: int) -> np.ndarray:
    flattened = np.asarray(qc_values).reshape(-1)
    if len(flattened) != expected_length:
        raise ValueError("adjusted QC array length does not match adjusted data")
    return np.fromiter(
        (normalized_text(value) == "1" for value in flattened),
        dtype=bool,
        count=expected_length,
    )


def interpolate_adjusted_profile(
    dataset: xr.Dataset,
    profile_position: int,
    latitude: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    required = (
        "TEMP_ADJUSTED",
        "PRES_ADJUSTED",
        "TEMP_ADJUSTED_QC",
        "PRES_ADJUSTED_QC",
    )
    if any(name not in dataset.variables for name in required):
        raise LookupError("adjusted_variables_unavailable")

    temperature = profile_vector(dataset, "TEMP_ADJUSTED", profile_position).astype(float)
    pressure = profile_vector(dataset, "PRES_ADJUSTED", profile_position).astype(float)
    temperature_qc = strict_qc_mask(
        profile_vector(dataset, "TEMP_ADJUSTED_QC", profile_position), len(temperature)
    )
    pressure_qc = strict_qc_mask(
        profile_vector(dataset, "PRES_ADJUSTED_QC", profile_position), len(pressure)
    )
    if len(temperature) != len(pressure):
        raise ValueError("adjusted temperature and pressure lengths differ")

    valid = (
        temperature_qc
        & pressure_qc
        & np.isfinite(temperature)
        & np.isfinite(pressure)
        & (pressure >= 0)
    )
    if not valid.any():
        raise LookupError("no_adjusted_measurements_with_qc_1")

    depth = -np.asarray(gsw.z_from_p(pressure[valid], latitude), dtype=float)
    temperature = temperature[valid]
    valid_depth = np.isfinite(depth) & (depth >= 0)
    depth = depth[valid_depth]
    temperature = temperature[valid_depth]
    if len(depth) < 3:
        raise LookupError("insufficient_valid_adjusted_measurements")

    order = np.argsort(depth)
    depth = depth[order]
    temperature = temperature[order]
    unique_depths, inverse, counts = np.unique(
        depth,
        return_inverse=True,
        return_counts=True,
    )
    if len(unique_depths) != len(depth):
        temperature_sums = np.zeros(len(unique_depths), dtype=float)
        np.add.at(temperature_sums, inverse, temperature)
        temperature = temperature_sums / counts
        depth = unique_depths
    if len(depth) < 3:
        raise LookupError("insufficient_unique_adjusted_depths")

    interpolated = np.full(len(TARGET_DEPTHS_M), np.nan, dtype=float)
    targets = np.asarray(TARGET_DEPTHS_M, dtype=float)
    within_observed_range = (targets >= depth[0]) & (targets <= depth[-1])
    interpolated[within_observed_range] = np.interp(
        targets[within_observed_range],
        depth,
        temperature,
    )
    if np.isfinite(interpolated).sum() < 2:
        raise LookupError("insufficient_depth_coverage_for_interpolation")
    return targets, interpolated, len(depth)


class OceanEmbedPredictor:
    def __init__(self, device: str) -> None:
        if str(SCRIPTS_ROOT) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_ROOT))
        import torch
        from inference.oceanembed_inference import OceanEmbedEnsemble

        selected_device = device
        if device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        if selected_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        self.torch = torch
        self.engine = OceanEmbedEnsemble(device=selected_device)
        self.datasets: dict[Path, xr.Dataset] = {}
        self.prediction_cache: dict[
            tuple[date, int, int], tuple[np.ndarray, np.ndarray]
        ] = {}
        LOGGER.info("Loaded the existing OceanEmbed V1 ensemble on %s", selected_device)

    def close(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()

    def _dataset(self, feature: str) -> tuple[xr.Dataset, str]:
        filename, variable = FEATURE_FILES[feature]
        path = HARMONIZED_ROOT / filename
        if path not in self.datasets:
            if not path.is_file():
                raise FileNotFoundError(f"Required historical input dataset is missing: {path}")
            self.datasets[path] = xr.open_dataset(path)
        dataset = self.datasets[path]
        if variable not in dataset:
            raise KeyError(f"Input feature {feature} not found as {variable} in {path}")
        return dataset, variable

    def predict_tile(
        self,
        target_date: date,
        tile_row: int,
        tile_column: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (target_date, tile_row, tile_column)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        window_start = np.datetime64(target_date - timedelta(days=6))
        window_end = np.datetime64(target_date)
        feature_arrays: dict[str, Any] = {}
        for feature in INPUT_FEATURES:
            dataset, variable = self._dataset(feature)
            selected = dataset[variable].sel(time=slice(window_start, window_end))
            times = selected.time.values.astype("datetime64[D]")
            expected_times = np.arange(
                window_start,
                window_end + np.timedelta64(1, "D"),
                dtype="datetime64[D]",
            )
            if not np.array_equal(times, expected_times):
                raise ValueError(
                    f"{feature} does not have the exact retrospective window "
                    f"{window_start}..{window_end}"
                )
            values = np.asarray(selected.values, dtype=np.float32)
            if values.shape != (7, 101, 241):
                raise ValueError(
                    f"{feature} has shape {values.shape}; expected (7, 101, 241)"
                )
            feature_arrays[feature] = self.torch.as_tensor(
                values[
                    :,
                    tile_row:tile_row + 64,
                    tile_column:tile_column + 64,
                ],
                dtype=self.torch.float32,
            )

        mean, spread = self.engine.predict_from_raw_window_with_spread(feature_arrays)
        prediction = mean.detach().cpu().numpy()
        ensemble_spread = spread.detach().cpu().numpy()
        if prediction.shape != (15, 32, 32) or ensemble_spread.shape != (15, 32, 32):
            raise ValueError("Existing OceanEmbed inference returned unexpected output shape")
        if not np.isfinite(prediction).all() or not np.isfinite(ensemble_spread).all():
            raise ValueError("Existing OceanEmbed inference returned non-finite values")
        self.prediction_cache[cache_key] = (prediction, ensemble_spread)
        return prediction, ensemble_spread


def extract_profile(
    path: Path,
    candidate: ProfileCandidate,
) -> tuple[np.ndarray, np.ndarray, int]:
    with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as dataset:
        profile_position = select_profile_position(dataset, candidate.file)
        candidate.float_id = profile_float_id(
            dataset, profile_position, candidate.file
        )
        candidate.data_mode = profile_mode(
            dataset, profile_position, candidate.data_mode
        )
        candidate.adjusted_available = all(
            name in dataset.variables
            for name in (
                "TEMP_ADJUSTED",
                "PRES_ADJUSTED",
                "TEMP_ADJUSTED_QC",
                "PRES_ADJUSTED_QC",
            )
        )
        if not candidate.adjusted_available:
            raise LookupError("adjusted_variables_unavailable")
        return interpolate_adjusted_profile(
            dataset,
            profile_position,
            candidate.latitude,
        )


def build_metrics(matchups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def metric_record(rows: list[dict[str, Any]], depth: int | None) -> dict[str, Any]:
        observed = np.asarray(
            [row["argo_temperature_c"] for row in rows],
            dtype=float,
        )
        predicted = np.asarray(
            [row["oceanembed_temperature_c"] for row in rows],
            dtype=float,
        )
        error = predicted - observed
        count = len(rows)
        correlation: float | None = None
        if count >= 2 and np.std(observed) > 0 and np.std(predicted) > 0:
            value = float(np.corrcoef(observed, predicted)[0, 1])
            correlation = value if np.isfinite(value) else None
        return {
            "depth_m": depth,
            "valid_observations": count,
            "rmse_c": float(np.sqrt(np.mean(error**2))) if count else None,
            "mae_c": float(np.mean(np.abs(error))) if count else None,
            "bias_c": float(np.mean(error)) if count else None,
            "pearson_correlation": correlation,
        }

    per_depth = [
        metric_record(
            [row for row in matchups if row["depth_m"] == depth],
            depth,
        )
        for depth in TARGET_DEPTHS_M
    ]
    overall = metric_record(matchups, None)
    return per_depth, overall


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(matchups: list[dict[str, Any]]) -> None:
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    depths = np.asarray(TARGET_DEPTHS_M, dtype=float)
    per_depth_observations = [
        [row for row in matchups if row["depth_m"] == depth]
        for depth in TARGET_DEPTHS_M
    ]
    counts = np.asarray([len(rows) for rows in per_depth_observations])
    observed_means = np.asarray(
        [
            np.mean([row["argo_temperature_c"] for row in rows]) if rows else np.nan
            for rows in per_depth_observations
        ]
    )
    prediction_means = np.asarray(
        [
            np.mean([row["oceanembed_temperature_c"] for row in rows]) if rows else np.nan
            for rows in per_depth_observations
        ]
    )

    figure, axis = plt.subplots(figsize=(7, 8))
    axis.plot(observed_means, depths, marker="o", label="ARGO adjusted observations")
    axis.plot(prediction_means, depths, marker="o", label="OceanEmbed 3-seed mean")
    axis.set_ylim(max(TARGET_DEPTHS_M), min(TARGET_DEPTHS_M))
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel("Depth (m)")
    axis.set_title("ARGO vs OceanEmbed mean temperature profiles")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_root / "argo_vs_oceanembed_profiles.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 7))
    for depth, rows in zip(TARGET_DEPTHS_M, per_depth_observations):
        if rows:
            errors = [row["temperature_error_c"] for row in rows]
            axis.scatter(
                errors,
                np.full(len(errors), depth),
                color="tab:blue",
                alpha=0.38,
                s=16,
            )
    axis.axvline(0, linewidth=1)
    axis.set_ylim(max(TARGET_DEPTHS_M) + 20, -20)
    axis.set_xlabel("OceanEmbed − ARGO temperature (°C)")
    axis.set_ylabel("Depth (m)")
    axis.set_title("OceanEmbed error by depth")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_root / "argo_oceanembed_error_by_depth.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(depths, counts, width=np.maximum(depths * 0.05, 2))
    axis.set_xlabel("Depth (m)")
    axis.set_ylabel("Valid profile matchups")
    axis.set_title("Valid ARGO–OceanEmbed matchups by depth")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_root / "argo_matchup_counts_by_depth.png", dpi=180)
    plt.close(figure)


def process_candidates(
    candidates: list[ProfileCandidate],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[ProfileCandidate], dict[str, int]]:
    index_counts: dict[str, int] = {}
    matchups: list[dict[str, Any]] = []
    maximum_distance_km = args.max_distance_deg * METERS_PER_DEGREE
    predictor: OceanEmbedPredictor | None = None
    eligible = [candidate for candidate in candidates if not candidate.exclusion_reason]
    try:
        for candidate in eligible:
            relative_path = candidate.file
            local_path = safe_profile_path(relative_path)
            try:
                download_file(
                    profile_url(args.gdac_url, relative_path),
                    local_path,
                    force=args.force_profiles,
                )
            except (OSError, ValueError) as exc:
                candidate.exclusion_reason = "profile_download_failed"
                LOGGER.warning("Excluded %s: %s", candidate.profile_id, exc)
                index_counts[candidate.exclusion_reason] = (
                    index_counts.get(candidate.exclusion_reason, 0) + 1
                )
                continue

            try:
                _, argo_temperature, valid_depth_count = extract_profile(
                    local_path,
                    candidate,
                )
                candidate.valid_depth_count = valid_depth_count
            except LookupError as exc:
                candidate.exclusion_reason = str(exc)
                index_counts[candidate.exclusion_reason] = (
                    index_counts.get(candidate.exclusion_reason, 0) + 1
                )
                continue
            except (OSError, ValueError, KeyError, IndexError) as exc:
                candidate.exclusion_reason = "profile_read_failed"
                LOGGER.warning("Excluded %s: %s", candidate.profile_id, exc)
                index_counts[candidate.exclusion_reason] = (
                    index_counts.get(candidate.exclusion_reason, 0) + 1
                )
                continue

            if candidate.spatial_distance_km is None or (
                candidate.spatial_distance_km > maximum_distance_km
            ):
                candidate.exclusion_reason = "spatial_distance_exceeds_threshold"
                index_counts[candidate.exclusion_reason] = (
                    index_counts.get(candidate.exclusion_reason, 0) + 1
                )
                continue
            if None in (
                candidate.tile_row,
                candidate.tile_column,
                candidate.output_row,
                candidate.output_column,
            ):
                candidate.exclusion_reason = "outside_model_output_coverage"
                index_counts[candidate.exclusion_reason] = (
                    index_counts.get(candidate.exclusion_reason, 0) + 1
                )
                continue

            if predictor is None:
                predictor = OceanEmbedPredictor(args.device)
            try:
                mean, spread = predictor.predict_tile(
                    candidate.argo_date,
                    candidate.tile_row,
                    candidate.tile_column,
                )
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                candidate.exclusion_reason = "oceanembed_inference_failed"
                LOGGER.error(
                    "Inference failed for %s on %s: %s",
                    candidate.profile_id,
                    candidate.argo_date,
                    exc,
                )
                index_counts[candidate.exclusion_reason] = (
                    index_counts.get(candidate.exclusion_reason, 0) + 1
                )
                continue

            assert candidate.output_row is not None
            assert candidate.output_column is not None
            prediction = mean[:, candidate.output_row, candidate.output_column]
            uncertainty = spread[:, candidate.output_row, candidate.output_column]
            valid_targets = np.isfinite(argo_temperature) & np.isfinite(prediction)
            profile_matchups = 0
            for depth_index, is_valid in enumerate(valid_targets):
                if not is_valid:
                    continue
                observed = float(argo_temperature[depth_index])
                predicted = float(prediction[depth_index])
                profile_matchups += 1
                matchups.append(
                    {
                        "profile_id": candidate.profile_id,
                        "argo_date": candidate.argo_date.isoformat(),
                        "argo_lat": candidate.latitude,
                        "argo_lon": candidate.longitude,
                        "model_lat": candidate.model_latitude,
                        "model_lon": candidate.model_longitude,
                        "spatial_distance_km": candidate.spatial_distance_km,
                        "depth_m": TARGET_DEPTHS_M[depth_index],
                        "argo_temperature_c": observed,
                        "oceanembed_temperature_c": predicted,
                        "ensemble_spread_c": float(uncertainty[depth_index]),
                        "temperature_error_c": predicted - observed,
                    }
                )
            candidate.matched_depth_count = profile_matchups
            if profile_matchups:
                candidate.exclusion_reason = ""
            else:
                candidate.exclusion_reason = "no_target_depths_within_observed_range"
                index_counts[candidate.exclusion_reason] = (
                    index_counts.get(candidate.exclusion_reason, 0) + 1
                )
    finally:
        if predictor is not None:
            predictor.close()

    return matchups, candidates, index_counts


def write_outputs(
    args: argparse.Namespace,
    candidates: list[ProfileCandidate],
    matchups: list[dict[str, Any]],
    index_url: str,
) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_csv(
        OUTPUT_ROOT / "argo_matchups.csv",
        matchups,
        CSV_MATCHUP_FIELDS,
    )
    catalog_rows = [
        {
            "profile_id": item.profile_id,
            "float_id": item.float_id,
            "date": item.argo_date.isoformat(),
            "latitude": item.latitude,
            "longitude": item.longitude,
            "data_mode": item.data_mode,
            "adjusted_available": item.adjusted_available,
            "valid_depth_count": item.valid_depth_count,
            "matched_depth_count": item.matched_depth_count,
            "exclusion_reason": item.exclusion_reason,
        }
        for item in candidates
    ]
    save_csv(
        OUTPUT_ROOT / "argo_profile_catalog.csv",
        catalog_rows,
        CSV_CATALOG_FIELDS,
    )

    per_depth_metrics, overall_metrics = build_metrics(matchups)
    actual_exclusion_counts = dict(
        Counter(
            candidate.exclusion_reason
            for candidate in candidates
            if candidate.exclusion_reason
        )
    )
    metrics = {
        "validation_period": {
            "start": args.start_date.isoformat(),
            "end": args.end_date.isoformat(),
        },
        "split": args.split,
        "domain": DOMAIN,
        "grid_resolution_deg": GRID_RESOLUTION_DEG,
        "spatial_threshold": {
            "maximum_degrees": args.max_distance_deg,
            "maximum_approx_km": args.max_distance_deg * METERS_PER_DEGREE,
            "distance_method": "great-circle haversine distance to nearest OceanEmbed grid point",
        },
        "temporal_matching_rule": "same calendar day; no tolerance window",
        "depths_m": TARGET_DEPTHS_M,
        "profile_count_considered": len(candidates),
        "profile_count_used": len(
            {row["profile_id"] for row in matchups}
        ),
        "total_valid_matchups": len(matchups),
        "per_depth_metrics": per_depth_metrics,
        "overall_metrics": overall_metrics,
        "data_source": {
            "name": "Official Argo GDAC profile NetCDF",
            "profile_index_url": index_url,
            "profile_base_url": args.gdac_url,
        },
        "data_mode": {
            "rule": "Delayed-mode profiles are processed first when available; only adjusted variables enter metrics.",
            "counts": {
                mode: sum(1 for item in candidates if item.data_mode.upper() == mode)
                for mode in ("D", "A", "R", "UNKNOWN")
            },
        },
        "qc_rule": (
            "TEMP_ADJUSTED and PRES_ADJUSTED only; corresponding "
            "TEMP_ADJUSTED_QC and PRES_ADJUSTED_QC must both equal '1'."
        ),
        "normalization_and_model": (
            "Existing OceanEmbed V1 three-seed ensemble mean and spread; "
            "original train-only statistics from ml_config.json."
        ),
        "exclusion_counts": actual_exclusion_counts,
        "notes": [
            "ARGO observations are independent validation data and were not used for training, normalization, model selection, hyperparameter tuning, threshold tuning, or GLORYS targets.",
            "Only same-calendar-day profile/model matchups are included.",
            "Adjusted pressure was converted to positive-down geometric depth with TEOS-10 gsw.z_from_p at the profile latitude.",
            "Vertical interpolation is linear and only performed within the observed valid-depth range; no extrapolation is used.",
            "Index candidates outside the existing model's centered 32x32 output coverage are explicitly excluded.",
            "Per-depth metrics use only profiles with a QC-valid adjusted observation that brackets that target depth.",
            "The primary comparison is OceanEmbed prediction versus ARGO observation; GLORYS is not part of these metrics.",
        ],
    }
    with (OUTPUT_ROOT / "argo_validation_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")
    make_figures(matchups)
    return metrics


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.start_date > args.end_date:
        raise SystemExit("--start-date must be on or before --end-date")
    if args.max_distance_deg <= 0 or args.max_distance_deg > SPATIAL_THRESHOLD_DEG:
        raise SystemExit("--max-distance-deg must be greater than 0 and no more than 0.5")
    validate_heldout_contract(args.start_date, args.end_date)

    index_path = RAW_ROOT / "index" / "ar_index_global_prof.txt.gz"
    download_file(args.index_url, index_path, force=args.force_index)
    candidates = discover_candidates(
        index_path,
        args.start_date,
        args.end_date,
        args.max_distance_deg,
    )
    LOGGER.info(
        "Official GDAC index yielded %d same-day December/domain candidates.",
        len(candidates),
    )
    if args.discover_only:
        reason_counts: dict[str, int] = {}
        for candidate in candidates:
            reason = candidate.exclusion_reason or "eligible_for_profile_download"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        LOGGER.info("Candidate selection counts: %s", reason_counts)
        for candidate in candidates[:10]:
            LOGGER.info(
                "%s %s %.3fN %.3fE mode=%s exclusion=%s",
                candidate.profile_id,
                candidate.argo_date,
                candidate.latitude,
                candidate.longitude,
                candidate.data_mode,
                candidate.exclusion_reason or "eligible_for_profile_download",
            )
        if len(candidates) > 10:
            LOGGER.info("Showing 10 of %d candidates.", len(candidates))
        LOGGER.info("Discovery only: no profile NetCDF files, predictions, or metrics were produced.")
        return 0
    if not candidates:
        raise SystemExit(
            "No official Argo profile-index candidates were found for the fixed validation period."
        )

    matchups, candidates, _ = process_candidates(candidates, args)
    metrics = write_outputs(
        args,
        candidates,
        matchups,
        args.index_url,
    )
    LOGGER.info(
        "ARGO validation complete: %d candidate profiles, %d used profiles, %d valid matchups.",
        metrics["profile_count_considered"],
        metrics["profile_count_used"],
        metrics["total_valid_matchups"],
    )
    LOGGER.info("Metrics written to %s", OUTPUT_ROOT / "argo_validation_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
