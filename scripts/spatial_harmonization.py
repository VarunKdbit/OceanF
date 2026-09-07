"""
OceanEmbed Spatial Harmonization
================================

Purpose:
    Convert all OceanEmbed surface input datasets to the common
    0.25° × 0.25° target grid and save persistent harmonized NetCDF files.

Target domain:
    Latitude  : 5°N to 30°N
    Longitude : 45°E to 105°E
    Resolution: 0.25°

Method:
    Linear interpolation
    No extrapolation outside source coverage
    Missing values remain NaN

Common period:
    2025-07-01 to 2025-12-31

This script implements the spatial harmonization policy established
in notebooks/spatial_harmonization.ipynb.
"""

from pathlib import Path
import shutil

import numpy as np
import xarray as xr


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "ML" / "harmonized"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OCEANEMBED TARGET GRID
# ============================================================

TARGET_LAT = np.arange(
    5.0,
    30.0 + 0.25 / 2,
    0.25,
    dtype=np.float32,
)

TARGET_LON = np.arange(
    45.0,
    105.0 + 0.25 / 2,
    0.25,
    dtype=np.float32,
)


# ============================================================
# COMMON TIME PERIOD
# ============================================================

COMMON_START = np.datetime64("2025-07-01")
COMMON_END = np.datetime64("2025-12-31")


# ============================================================
# DATASET DEFINITIONS
# ============================================================

DATASETS = {
    "SST": {
        "input": DATA_DIR / "SST" / "SST_processed.nc",
        "output": OUTPUT_DIR / "SST_harmonized.nc",
        "time_dim": "time",
    },

    "SSS": {
        "input": DATA_DIR / "SSS" / "SSS_processed.nc",
        "output": OUTPUT_DIR / "SSS_harmonized.nc",
        "time_dim": "time",
    },

    "SLA": {
        "input": DATA_DIR / "SSA" / "SSA_processed.nc",
        "output": OUTPUT_DIR / "SLA_harmonized.nc",
        "time_dim": "time",
    },

    "Currents": {
        "input": DATA_DIR / "Currents" / "Currents_processed.nc",
        "output": OUTPUT_DIR / "Currents_harmonized.nc",
        "time_dim": "time",
    },

    "Winds": {
        "input": DATA_DIR / "Winds" / "Winds_processed.nc",
        "output": OUTPUT_DIR / "Winds_harmonized.nc",
        "time_dim": "valid_time",
    },

    "SubsurfaceTemp": {
        "input": DATA_DIR / "SubsurfaceTemp" / "SubsurfaceTemp_processed.nc",
        "output": OUTPUT_DIR / "SubsurfaceTemp_harmonized.nc",
        "time_dim": "time",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found:\n{path}"
        )


def standardize_dataset(ds, time_dim):
    """
    Standardize coordinate names/orientation.

    Winds uses valid_time instead of time.
    Latitude is forced to ascending order.
    """

    if time_dim == "valid_time":
        ds = ds.rename({"valid_time": "time"})

    if "latitude" not in ds.coords:
        raise ValueError("Dataset does not contain latitude coordinate.")

    if "longitude" not in ds.coords:
        raise ValueError("Dataset does not contain longitude coordinate.")

    # Ensure latitude is ascending.
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.sortby("latitude")

    # Ensure longitude is ascending.
    if ds.longitude.values[0] > ds.longitude.values[-1]:
        ds = ds.sortby("longitude")

    return ds


def select_common_period(ds):
    """
    Restrict dataset to OceanEmbed common period.
    """

    ds = ds.sel(
        time=slice(
            COMMON_START,
            COMMON_END,
        )
    )

    if ds.sizes.get("time", 0) == 0:
        raise ValueError(
            "No data found in common period "
            "2025-07-01 to 2025-12-31."
        )

    return ds


def interpolate_to_target_grid(ds):
    """
    Apply the exact spatial harmonization method used in the
    OceanEmbed spatial_harmonization notebook.

    Linear interpolation.
    No extrapolation.
    """

    return ds.interp(
        latitude=TARGET_LAT,
        longitude=TARGET_LON,
        method="linear",
        kwargs={
            "fill_value": np.nan,
        },
    )


def validate_target_grid(ds, name):
    """
    Verify exact 101 × 241 OceanEmbed grid.
    """

    lat = ds.latitude.values
    lon = ds.longitude.values

    if len(lat) != 101:
        raise ValueError(
            f"{name}: expected 101 latitude points, got {len(lat)}"
        )

    if len(lon) != 241:
        raise ValueError(
            f"{name}: expected 241 longitude points, got {len(lon)}"
        )

    if not np.array_equal(lat, TARGET_LAT):
        raise ValueError(
            f"{name}: latitude grid does not exactly match target grid."
        )

    if not np.array_equal(lon, TARGET_LON):
        raise ValueError(
            f"{name}: longitude grid does not exactly match target grid."
        )

    if not np.all(np.diff(lat) > 0):
        raise ValueError(
            f"{name}: latitude is not ascending."
        )

    if not np.all(np.diff(lon) > 0):
        raise ValueError(
            f"{name}: longitude is not ascending."
        )


def validate_time(ds, name):
    """
    Verify that the harmonized dataset contains exactly
    the expected 184-day common period.
    """

    times = ds.time.values

    if len(times) != 184:
        raise ValueError(
            f"{name}: expected 184 days, got {len(times)}"
        )

    if times[0] != COMMON_START:
        raise ValueError(
            f"{name}: unexpected first date {times[0]}"
        )

    if times[-1] != COMMON_END:
        raise ValueError(
            f"{name}: unexpected last date {times[-1]}"
        )


def save_dataset(ds, output_path):
    """
    Save harmonized dataset as compressed float32 NetCDF.

    Data are written to a temporary file first so an interrupted
    write does not destroy an existing valid output.
    """

    temp_path = output_path.with_suffix(".tmp.nc")

    if temp_path.exists():
        temp_path.unlink()

    if output_path.exists():
        print(f"[INFO] Existing output will be replaced:")
        print(f"       {output_path}")

    encoding = {}

    for variable_name, variable in ds.data_vars.items():

        if np.issubdtype(variable.dtype, np.floating):

            encoding[variable_name] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
            }

    print("[SAVE] Writing NetCDF...")
    print(f"       {temp_path}")

    ds.to_netcdf(
        temp_path,
        mode="w",
        format="NETCDF4",
        encoding=encoding,
    )

    if output_path.exists():
        output_path.unlink()

    shutil.move(
        str(temp_path),
        str(output_path),
    )

    print("[PASS] Saved:")
    print(f"       {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "OCEANEMBED SPATIAL HARMONIZATION"
    )

    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Target grid:")
    print("  Latitude  : 5.0°N to 30.0°N")
    print("  Longitude : 45.0°E to 105.0°E")
    print("  Resolution: 0.25°")
    print("  Grid size  : 101 × 241")

    print()
    print("Common period:")
    print("  Start:", COMMON_START)
    print("  End  :", COMMON_END)
    print("  Days :", 184)

    print()
    print("Interpolation:")
    print("  Method      : linear")
    print("  Extrapolate : NO")
    print("  Outside     : NaN")

    print_header(
        "CHECKING INPUT DATASETS"
    )

    for name, config in DATASETS.items():

        print(f"[CHECK] {name}")

        check_file(config["input"])

        print(f"        {config['input']}")

    print()
    print("[PASS] All required input files found.")

    # --------------------------------------------------------
    # PROCESS EACH DATASET
    # --------------------------------------------------------

    for name, config in DATASETS.items():

        print_header(
            f"HARMONIZING {name}"
        )

        input_path = config["input"]
        output_path = config["output"]
        time_dim = config["time_dim"]

        print("[OPEN]")
        print(input_path)

        # Use chunking so large datasets do not need to be loaded
        # completely into RAM.
        ds = xr.open_dataset(
            input_path,
            chunks="auto",
        )

        try:

            print()
            print("Original dimensions:")
            print(ds.sizes)

            # ------------------------------------------------
            # STANDARDIZE COORDINATES
            # ------------------------------------------------

            ds = standardize_dataset(
                ds,
                time_dim,
            )

            print()
            print("[PASS] Coordinate orientation standardized.")

            # ------------------------------------------------
            # SELECT COMMON TIME PERIOD
            # ------------------------------------------------

            ds = select_common_period(ds)

            print()
            print("[PASS] Common period selected.")
            print(
                f"       {ds.time.values[0]} -> "
                f"{ds.time.values[-1]}"
            )
            print(
                f"       Days: {ds.sizes['time']}"
            )

            # ------------------------------------------------
            # SPATIAL HARMONIZATION
            # ------------------------------------------------

            print()
            print("[INTERPOLATE]")
            print("  Source grid:")
            print(
                f"    {ds.sizes['latitude']} × "
                f"{ds.sizes['longitude']}"
            )

            ds_h = interpolate_to_target_grid(ds)

            print("  Target grid:")
            print(
                f"    {ds_h.sizes['latitude']} × "
                f"{ds_h.sizes['longitude']}"
            )

            # ------------------------------------------------
            # VALIDATE
            # ------------------------------------------------

            validate_target_grid(
                ds_h,
                name,
            )

            validate_time(
                ds_h,
                name,
            )

            print()
            print("[PASS] Spatial grid validation.")
            print("[PASS] Time validation.")

            # ------------------------------------------------
            # SUBSURFACE DEPTH VALIDATION
            # ------------------------------------------------

            if name == "SubsurfaceTemp":

                expected_depths = np.array(
                    [
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
                    ],
                    dtype=np.float64,
                )

                actual_depths = ds_h.depth.values

                if not np.array_equal(
                    actual_depths,
                    expected_depths,
                ):
                    raise ValueError(
                        "SubsurfaceTemp depth coordinates "
                        "do not match OceanEmbed target depths."
                    )

                print()
                print(
                    "[PASS] Subsurface target depths verified."
                )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            save_dataset(
                ds_h,
                output_path,
            )

        finally:

            ds.close()

        print()
        print(f"[PASS] {name} complete.")

    # ========================================================
    # FINAL QC
    # ========================================================

    print_header(
        "FINAL HARMONIZATION QC"
    )

    for name, config in DATASETS.items():

        output_path = config["output"]

        print(f"[CHECK] {name}")

        if not output_path.exists():
            raise FileNotFoundError(
                f"Expected output was not created:\n"
                f"{output_path}"
            )

        with xr.open_dataset(
            output_path,
        ) as ds:

            validate_target_grid(
                ds,
                name,
            )

            validate_time(
                ds,
                name,
            )

            print(
                f"  Grid : "
                f"{ds.sizes['latitude']} × "
                f"{ds.sizes['longitude']}"
            )

            print(
                f"  Time : "
                f"{ds.time.values[0]} -> "
                f"{ds.time.values[-1]}"
            )

            if "depth" in ds.sizes:
                print(
                    f"  Depth channels: "
                    f"{ds.sizes['depth']}"
                )

            print("  PASS")

    print_header(
        "SPATIAL HARMONIZATION COMPLETE"
    )

    print(
        "All six datasets are now available on the "
        "exact OceanEmbed 0.25° grid."
    )

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("Created files:")

    for name, config in DATASETS.items():
        print(f"  {config['output'].name}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()