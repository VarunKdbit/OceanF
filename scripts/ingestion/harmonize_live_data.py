from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from copernicus_sources import (
    DOMAIN,
    GRID_RESOLUTION_DEG,
    HISTORY_DAYS,
    SOURCES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIVE_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "live"
LIVE_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "live"


TARGET_LAT = np.arange(
    DOMAIN["lat_min"],
    DOMAIN["lat_max"] + GRID_RESOLUTION_DEG / 2,
    GRID_RESOLUTION_DEG,
    dtype=np.float64,
)

TARGET_LON = np.arange(
    DOMAIN["lon_min"],
    DOMAIN["lon_max"] + GRID_RESOLUTION_DEG / 2,
    GRID_RESOLUTION_DEG,
    dtype=np.float64,
)


SSS_SURFACE_DEPTH_M = 0.49402499198913574
SSS_DEPTH_TOLERANCE_M = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonize live Copernicus surface inputs "
            "for OceanEmbed."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Target date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--window-start",
        help=(
            "First day of a requested date range. Defaults to the "
            "seven-day retrospective window."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output path. Use a staging path when "
            "the caller will validate before publishing."
        ),
    )

    return parser.parse_args()


def load_dataset(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    return xr.open_dataset(path)


def standardize_latitude(
    ds: xr.Dataset,
    latitude_name: str = "latitude",
) -> xr.Dataset:
    if latitude_name not in ds.coords:
        raise ValueError(
            f"Dataset does not contain coordinate "
            f"'{latitude_name}'."
        )

    latitude = ds[latitude_name]

    if latitude.values[0] > latitude.values[-1]:
        ds = ds.sortby(latitude_name)

    return ds


def normalize_longitude(
    ds: xr.Dataset,
    longitude_name: str = "longitude",
) -> xr.Dataset:
    if longitude_name not in ds.coords:
        raise ValueError(
            f"Dataset does not contain coordinate "
            f"'{longitude_name}'."
        )

    longitude = ds[longitude_name]

    if float(longitude.max()) > 180.0:
        converted = ((longitude + 180.0) % 360.0) - 180.0

        ds = ds.assign_coords(
            {longitude_name: converted}
        )

        ds = ds.sortby(longitude_name)

    return ds


def interpolate_to_target_grid(
    da: xr.DataArray,
) -> xr.DataArray:

    ds = da.to_dataset(name="_tmp")

    ds = standardize_latitude(ds)
    ds = normalize_longitude(ds)

    da = ds["_tmp"]

    return da.interp(
        latitude=xr.DataArray(
            TARGET_LAT,
            dims="latitude",
        ),
        longitude=xr.DataArray(
            TARGET_LON,
            dims="longitude",
        ),
        method="linear",
    )


def ensure_time_dimension(
    da: xr.DataArray,
) -> xr.DataArray:

    if "time" not in da.dims:
        raise ValueError(
            f"Expected a time dimension, "
            f"got dimensions: {da.dims}"
        )

    return da


def load_sst(
    path: Path,
) -> xr.DataArray:

    ds = load_dataset(path)

    variable = "analysed_sst"

    if variable not in ds:
        raise ValueError(
            f"SST variable '{variable}' "
            f"not found in {path}"
        )

    da = ds[variable]

    da = ensure_time_dimension(da)

    units = str(
        da.attrs.get("units", "")
    ).lower()

    # OceanEmbed training SST is in Celsius.
    # Copernicus analysed_sst is supplied in Kelvin.
    if units in {"kelvin", "k"}:

        da = da - 273.15

    elif (
        "degree_celsius" in units
        or "degrees_celsius" in units
        or "celsius" in units
    ):

        pass

    else:

        raise ValueError(
            f"Unsupported SST units "
            f"'{da.attrs.get('units')}'. "
            "Expected Kelvin or Celsius."
        )

    da.attrs = {
        **da.attrs,
        "units": "degree_Celsius",
        "description": (
            "Live Copernicus SST converted "
            "from Kelvin to Celsius for "
            "OceanEmbed model compatibility."
        ),
    }

    da = interpolate_to_target_grid(da)

    return da


def load_sss(
    path: Path,
) -> tuple[xr.DataArray, dict]:

    ds = load_dataset(path)

    variable = "so"

    if variable not in ds:
        raise ValueError(
            f"SSS variable '{variable}' "
            f"not found in {path}"
        )

    da = ds[variable]

    da = ensure_time_dimension(da)

    if "depth" not in da.dims:
        raise ValueError(
            "Expected SSS source to contain "
            "a depth dimension."
        )

    if da.sizes["depth"] != 1:
        raise ValueError(
            "Live SSS input must contain exactly "
            "one selected depth level."
        )

    source_depth = float(
        da["depth"].values[0]
    )

    if (
        abs(
            source_depth
            - SSS_SURFACE_DEPTH_M
        )
        > SSS_DEPTH_TOLERANCE_M
    ):
        raise ValueError(
            "Unexpected SSS source depth: "
            f"{source_depth} m. Expected approximately "
            f"{SSS_SURFACE_DEPTH_M} m."
        )

    da = da.isel(
        depth=0,
        drop=True,
    )

    da.attrs = {
        **da.attrs,
        "units": "1e-3",
        "description": (
            "Copernicus ocean analysis/forecast "
            "salinity selected from the uppermost "
            "model level at approximately 0.494 m "
            "depth for OceanEmbed surface SSS."
        ),
    }

    da = interpolate_to_target_grid(da)

    metadata = {
        "sss_processing": (
            "Copernicus ocean analysis/forecast "
            "salinity selected at approximately "
            "0.494 m surface level"
        ),
        "sss_source_dataset": (
            SOURCES["sss"].dataset_id
        ),
        "sss_source_variable": (
            SOURCES["sss"].variable_names[0]
        ),
        "sss_source_depth_m": str(
            source_depth
        ),
    }

    return da, metadata


def load_sla(
    path: Path,
) -> xr.DataArray:

    ds = load_dataset(path)

    variable = "sla"

    if variable not in ds:
        raise ValueError(
            f"SLA variable '{variable}' "
            f"not found in {path}"
        )

    da = ds[variable]

    da = ensure_time_dimension(da)

    da.attrs = {
        **da.attrs,
        "units": "m",
        "description": (
            "Sea level anomaly used as "
            "the OceanEmbed SLA input."
        ),
    }

    da = interpolate_to_target_grid(da)

    return da


def load_currents(
    path: Path,
) -> tuple[
    xr.DataArray,
    xr.DataArray,
]:

    ds = load_dataset(path)

    if "uo" not in ds:
        raise ValueError(
            "Currents dataset does not "
            "contain 'uo'."
        )

    if "vo" not in ds:
        raise ValueError(
            "Currents dataset does not "
            "contain 'vo'."
        )

    uo = ds["uo"]
    vo = ds["vo"]

    uo = ensure_time_dimension(uo)
    vo = ensure_time_dimension(vo)

    if "depth" in uo.dims:

        if uo.sizes["depth"] < 1:
            raise ValueError(
                "Current U dataset has "
                "no depth levels."
            )

        uo = uo.isel(
            depth=0,
            drop=True,
        )

    if "depth" in vo.dims:

        if vo.sizes["depth"] < 1:
            raise ValueError(
                "Current V dataset has "
                "no depth levels."
            )

        vo = vo.isel(
            depth=0,
            drop=True,
        )

    uo.attrs = {
        **uo.attrs,
        "units": "m/s",
        "description": (
            "Eastward surface ocean current "
            "velocity at 0 m depth."
        ),
    }

    vo.attrs = {
        **vo.attrs,
        "units": "m/s",
        "description": (
            "Northward surface ocean current "
            "velocity at 0 m depth."
        ),
    }

    uo = interpolate_to_target_grid(uo)
    vo = interpolate_to_target_grid(vo)

    return uo, vo


def load_winds(
    path: Path,
) -> tuple[
    xr.DataArray,
    xr.DataArray,
]:

    ds = load_dataset(path)

    if "eastward_wind" not in ds:
        raise ValueError(
            "Wind dataset does not contain "
            "'eastward_wind'."
        )

    if "northward_wind" not in ds:
        raise ValueError(
            "Wind dataset does not contain "
            "'northward_wind'."
        )

    u_wind = ds["eastward_wind"]
    v_wind = ds["northward_wind"]

    u_wind = ensure_time_dimension(
        u_wind
    )

    v_wind = ensure_time_dimension(
        v_wind
    )

    # Source is hourly.
    # OceanEmbed requires daily inputs.
    u_wind = (
        u_wind
        .resample(time="1D")
        .mean(skipna=True)
    )

    v_wind = (
        v_wind
        .resample(time="1D")
        .mean(skipna=True)
    )

    u_wind.attrs = {
        **u_wind.attrs,
        "units": "m s-1",
        "description": (
            "10 m eastward wind component "
            "aggregated from hourly Copernicus "
            "observations to daily mean."
        ),
    }

    v_wind.attrs = {
        **v_wind.attrs,
        "units": "m s-1",
        "description": (
            "10 m northward wind component "
            "aggregated from hourly Copernicus "
            "observations to daily mean."
        ),
    }

    u_wind = interpolate_to_target_grid(
        u_wind
    )

    v_wind = interpolate_to_target_grid(
        v_wind
    )

    return u_wind, v_wind


def validate_time_axis(
    arrays: dict[str, xr.DataArray],
    expected_days: int = HISTORY_DAYS,
) -> None:

    reference = arrays["sst"].time.values

    if len(reference) != expected_days:
        raise ValueError(
            f"Expected {expected_days} daily "
            f"timesteps, got {len(reference)}."
        )

    for name, da in arrays.items():

        if "time" not in da.dims:
            raise ValueError(
                f"{name} does not contain "
                "a time dimension."
            )

        if len(da.time) != expected_days:
            raise ValueError(
                f"{name} contains {len(da.time)} "
                f"timesteps; expected {expected_days}."
            )

        if not np.array_equal(
            reference,
            da.time.values,
        ):
            raise ValueError(
                f"Time axis mismatch for {name}."
            )


def validate_grid(
    arrays: dict[str, xr.DataArray],
) -> None:

    for name, da in arrays.items():

        if (
            da.sizes.get("latitude")
            != len(TARGET_LAT)
        ):
            raise ValueError(
                f"{name} latitude size mismatch: "
                f"{da.sizes.get('latitude')} != "
                f"{len(TARGET_LAT)}"
            )

        if (
            da.sizes.get("longitude")
            != len(TARGET_LON)
        ):
            raise ValueError(
                f"{name} longitude size mismatch: "
                f"{da.sizes.get('longitude')} != "
                f"{len(TARGET_LON)}"
            )

        if not np.allclose(
            da.latitude.values,
            TARGET_LAT,
            atol=1e-6,
        ):
            raise ValueError(
                f"{name} latitude grid mismatch."
            )

        if not np.allclose(
            da.longitude.values,
            TARGET_LON,
            atol=1e-6,
        ):
            raise ValueError(
                f"{name} longitude grid mismatch."
            )


def finite_fraction(
    da: xr.DataArray,
) -> float:

    values = da.values

    return float(
        np.isfinite(values).mean()
    )


def build_output(
    target_date: str,
    arrays: dict[str, xr.DataArray],
    sss_metadata: dict,
) -> xr.Dataset:
    history_days = arrays["sst"].sizes["time"]

    output = xr.Dataset(
        {
            "sst": arrays["sst"],
            "sss": arrays["sss"],
            "sla": arrays["sla"],
            "uo": arrays["uo"],
            "vo": arrays["vo"],
            "u_wind": arrays["u_wind"],
            "v_wind": arrays["v_wind"],
        }
    )

    output.attrs = {
        "title": (
            "OceanEmbed Live Surface Inputs"
        ),
        "target_date": target_date,
        "history_days": history_days,
        "grid_resolution": (
            GRID_RESOLUTION_DEG
        ),
        "latitude_min": DOMAIN["lat_min"],
        "latitude_max": DOMAIN["lat_max"],
        "longitude_min": DOMAIN["lon_min"],
        "longitude_max": DOMAIN["lon_max"],
        "wind_processing": (
            "Hourly winds aggregated "
            "to daily mean"
        ),
        "current_processing": (
            "Surface depth selected at depth=0"
        ),
        "spatial_interpolation": (
            "NumPy bilinear interpolation"
        ),
        "source": (
            "Copernicus Marine live products"
        ),
        **sss_metadata,
    }

    return output


def main() -> None:

    args = parse_args()

    target_date = args.date
    target_day = date.fromisoformat(target_date)
    window_start_day = (
        date.fromisoformat(args.window_start)
        if args.window_start
        else target_day - timedelta(days=HISTORY_DAYS - 1)
    )
    if window_start_day > target_day:
        raise ValueError("--window-start must not be later than --date")
    range_start = window_start_day.isoformat()
    range_suffix = (
        f"{target_date}_7day"
        if window_start_day == target_day - timedelta(days=HISTORY_DAYS - 1)
        else f"{range_start}_{target_date}"
    )

    raw_dir = (
        LIVE_RAW_ROOT
        / (
            target_date
            if not args.window_start
            else "ranges"
        )
    )

    output_dir = (
        LIVE_PROCESSED_ROOT
        / target_date
    )

    output_path = args.output or (
        output_dir
        / f"oceanembed_live_{target_date}.nc"
    )

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw input directory does not exist: "
            f"{raw_dir}"
        )

    sst_path = (
        raw_dir
        / f"sst_{range_suffix}.nc"
    )

    sss_path = (
        raw_dir
        / f"sss_{range_suffix}.nc"
    )

    sla_path = (
        raw_dir
        / f"sla_{range_suffix}.nc"
    )

    currents_path = (
        raw_dir
        / f"currents_{range_suffix}.nc"
    )

    winds_path = (
        raw_dir
        / f"winds_{range_suffix}_hourly.nc"
    )

    print("=" * 72)
    print("OceanEmbed Live Data Harmonization")
    print("=" * 72)

    print(f"Target date : {target_date}")
    print(f"Raw input   : {raw_dir}")
    print(f"Output      : {output_path}")
    print()

    print("Loading live Copernicus inputs...")

    sst = load_sst(sst_path)

    sss, sss_metadata = load_sss(
        sss_path
    )

    sla = load_sla(sla_path)

    uo, vo = load_currents(
        currents_path
    )

    u_wind, v_wind = load_winds(
        winds_path
    )

    arrays = {
        "sst": sst,
        "sss": sss,
        "sla": sla,
        "uo": uo,
        "vo": vo,
        "u_wind": u_wind,
        "v_wind": v_wind,
    }

    print("Validating time axis...")

    expected_days = (target_day - window_start_day).days + 1
    validate_time_axis(arrays, expected_days)

    print("Validating target grid...")

    validate_grid(arrays)

    print()
    print("Finite-data fractions:")

    for name, da in arrays.items():

        print(
            f"  {name:8s}: "
            f"{finite_fraction(da):.4f}"
        )

    output = build_output(
        target_date=target_date,
        arrays=arrays,
        sss_metadata=sss_metadata,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Writing harmonized live file:")
    print(f"  {output_path}")

    output.to_netcdf(
        output_path,
        mode="w",
    )

    output.close()

    print()
    print("=" * 72)
    print("HARMONIZATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()