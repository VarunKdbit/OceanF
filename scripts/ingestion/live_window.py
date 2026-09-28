from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from .copernicus_sources import (
        DOMAIN,
        GRID_RESOLUTION_DEG,
        HISTORY_DAYS,
        OCEANEMBED_FEATURE_ORDER,
    )
except ImportError:
    from copernicus_sources import (
        DOMAIN,
        GRID_RESOLUTION_DEG,
        HISTORY_DAYS,
        OCEANEMBED_FEATURE_ORDER,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "live"
DAILY_CACHE_ROOT = LIVE_PROCESSED_ROOT / "daily"
EXPECTED_LATITUDE = np.arange(
    DOMAIN["lat_min"],
    DOMAIN["lat_max"] + GRID_RESOLUTION_DEG / 2,
    GRID_RESOLUTION_DEG,
)
EXPECTED_LONGITUDE = np.arange(
    DOMAIN["lon_min"],
    DOMAIN["lon_max"] + GRID_RESOLUTION_DEG / 2,
    GRID_RESOLUTION_DEG,
)
REQUIRED_VARIABLES = OCEANEMBED_FEATURE_ORDER


def requested_dates(target_date: date) -> list[date]:
    return [
        target_date - timedelta(days=HISTORY_DAYS - 1 - index)
        for index in range(HISTORY_DAYS)
    ]


def daily_cache_path(day: date) -> Path:
    return DAILY_CACHE_ROOT / f"oceanembed_live_{day.isoformat()}.nc"


def validate_window_file(
    path: Path,
    target_date: date,
    window_start: date | None = None,
) -> list[str]:
    start = window_start or target_date - timedelta(days=HISTORY_DAYS - 1)
    expected_dates = [
        np.datetime64(start + timedelta(days=index))
        for index in range((target_date - start).days + 1)
    ]
    errors: list[str] = []
    if start > target_date or len(expected_dates) > HISTORY_DAYS:
        return ["requested harmonized range must contain 1 to 7 days"]

    try:
        with xr.open_dataset(path) as dataset:
            missing = [
                name for name in REQUIRED_VARIABLES
                if name not in dataset.data_vars
            ]
            if missing:
                return [f"missing variables: {', '.join(missing)}"]

            if "time" not in dataset.coords:
                errors.append("missing time coordinate")
            else:
                actual_dates = dataset.time.values.astype("datetime64[D]")
                if not np.array_equal(actual_dates, np.asarray(expected_dates)):
                    errors.append(
                        "time axis does not exactly cover "
                        f"{start.isoformat()} through {target_date.isoformat()}"
                    )

            for coordinate, expected in (
                ("latitude", EXPECTED_LATITUDE),
                ("longitude", EXPECTED_LONGITUDE),
            ):
                if coordinate not in dataset.coords:
                    errors.append(f"missing {coordinate} coordinate")
                elif not np.allclose(dataset[coordinate].values, expected, atol=1e-6):
                    errors.append(
                        f"{coordinate} grid does not match the "
                        "5-30N / 45-105E 0.25-degree domain"
                    )

            expected_shape = (
                len(expected_dates),
                len(EXPECTED_LATITUDE),
                len(EXPECTED_LONGITUDE),
            )
            for name in REQUIRED_VARIABLES:
                if name not in dataset.data_vars:
                    continue
                values = dataset[name].values
                if values.shape != expected_shape:
                    errors.append(
                        f"{name} dimensions {values.shape} do not match "
                        f"the {len(expected_dates)}-day model grid"
                    )
                    continue
                finite_per_day = np.isfinite(values).reshape(len(expected_dates), -1).any(axis=1)
                for index, has_data in enumerate(finite_per_day):
                    if not has_data:
                        errors.append(
                            f"{name} has no finite data for "
                            f"{(start + timedelta(days=index)).isoformat()}"
                        )
                if not dataset[name].attrs.get("units"):
                    errors.append(f"{name} is missing its units metadata")

            if str(dataset.attrs.get("target_date", "")) != target_date.isoformat():
                errors.append("target_date metadata does not match the requested date")
            if int(dataset.attrs.get("history_days", -1)) != len(expected_dates):
                errors.append(
                    f"history_days metadata must be {len(expected_dates)}"
                )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"cannot open or validate NetCDF: {exc}")

    return errors


def validate_daily_file(path: Path, day: date) -> list[str]:
    return validate_window_file(path, day, day)


def missing_cached_dates(target_date: date) -> list[date]:
    missing: list[date] = []
    for day in requested_dates(target_date):
        path = daily_cache_path(day)
        if not path.is_file() or validate_daily_file(path, day):
            missing.append(day)
    return missing


def _write_daily_slice(
    dataset: xr.Dataset,
    day: date,
    index: int,
    overwrite: bool,
) -> bool:
    destination = daily_cache_path(day)
    if not overwrite and destination.is_file() and not validate_daily_file(destination, day):
        return False

    DAILY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.stem}.staging.nc")
    daily = dataset.isel(time=[index]).load()
    daily.attrs = dict(dataset.attrs)
    daily.attrs.update(
        target_date=day.isoformat(),
        history_days=1,
        window_start=day.isoformat(),
        window_end=day.isoformat(),
    )
    try:
        daily.to_netcdf(staged, mode="w")
    finally:
        daily.close()

    errors = validate_daily_file(staged, day)
    if errors:
        staged.unlink(missing_ok=True)
        raise ValueError(
            f"Daily cache entry {day.isoformat()} failed validation: "
            + "; ".join(errors)
        )
    os.replace(staged, destination)
    return True


def cache_window_days(
    path: Path,
    target_date: date,
    window_start: date | None = None,
    only_dates: set[date] | None = None,
) -> list[date]:
    errors = validate_window_file(path, target_date, window_start)
    if errors:
        raise ValueError(
            f"Cannot cache invalid harmonized window {path}: "
            + "; ".join(errors)
        )

    start = window_start or target_date - timedelta(days=HISTORY_DAYS - 1)
    published: list[date] = []
    with xr.open_dataset(path) as dataset:
        for index in range(dataset.sizes["time"]):
            day = start + timedelta(days=index)
            if only_dates is not None and day not in only_dates:
                continue
            if _write_daily_slice(dataset, day, index, overwrite=False):
                published.append(day)
    return published


def migrate_legacy_windows() -> None:
    for path in sorted(
        LIVE_PROCESSED_ROOT.glob("????-??-??/oceanembed_live_*.nc")
    ):
        try:
            target_date = date.fromisoformat(path.parent.name)
        except ValueError:
            continue
        if validate_window_file(path, target_date):
            continue
        cache_window_days(path, target_date)


def latest_complete_target_date() -> date | None:
    available_dates: set[date] = set()
    if DAILY_CACHE_ROOT.is_dir():
        for path in DAILY_CACHE_ROOT.glob("oceanembed_live_????-??-??.nc"):
            name = path.stem.removeprefix("oceanembed_live_")
            try:
                day = date.fromisoformat(name)
            except ValueError:
                continue
            if not validate_daily_file(path, day):
                available_dates.add(day)

    complete_targets = [
        day for day in available_dates
        if all(required in available_dates for required in requested_dates(day))
    ]
    return max(complete_targets, default=None)
