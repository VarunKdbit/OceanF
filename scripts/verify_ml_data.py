from pathlib import Path
import json
import sys
import xarray as xr


# ============================================================
# OceanEmbed ML DATA VERIFICATION
# Role 2 -> Role 3 handoff verification
# ============================================================

# Make Windows console safe for non-ASCII characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


# ============================================================
# PROJECT PATHS
# ============================================================

# This script is executed from the OceanF project root.
PROJECT_ROOT = Path.cwd()

PROCESSED = PROJECT_ROOT / "data" / "processed"

CONFIG_PATH = (
    PROCESSED
    / "ML"
    / "ml_config.json"
)


# ============================================================
# DATA FILE PATHS
# ============================================================

# IMPORTANT:
# Use relative paths intentionally.
# Do NOT convert these paths to absolute paths.

FILES = {
    "sst": Path("data/processed/SST/SST_processed.nc"),
    "sss": Path("data/processed/SSS/SSS_processed.nc"),
    "sla": Path("data/processed/SSA/SSA_processed.nc"),
    "currents": Path("data/processed/Currents/Currents_processed.nc"),
    "winds": Path("data/processed/Winds/Winds_processed.nc"),
    "thetao": Path("data/processed/SubsurfaceTemp/SubsurfaceTemp_processed.nc"),
}


# ============================================================
# HELPER
# ============================================================

def check(condition, message):

    if condition:
        print(f"[PASS] {message}")
        return True

    print(f"[FAIL] {message}")
    return False


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("OceanF ML DATA VERIFICATION")
print("=" * 70)


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURATION")
print("=" * 70)


check(
    CONFIG_PATH.exists(),
    "ml_config.json exists"
)


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)


print(
    f"Project: {config['project']}"
)

print(
    f"Resolution: "
    f"{config['domain']['resolution']}"
)

print(
    f"Time: "
    f"{config['time']['common_start']} "
    f"to "
    f"{config['time']['common_end']}"
)


print("\nInput features:")

for feature in config["input_features"]:

    print(
        f"  - {feature}"
    )


print("\nTarget:")

print(
    f"  {config['target_variable']}"
)


print("\nTarget depths:")

print(
    config["target_depths_m"]
)


# ============================================================
# 2. FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FILE CHECK")
print("=" * 70)


for name, path in FILES.items():

    check(
        path.exists(),
        f"{name}: {path}"
    )


# ============================================================
# 3. DATASET STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("DATASET STRUCTURE")
print("=" * 70)


datasets = {}


for name, path in FILES.items():

    print()
    print(f"[{name}]")
    print(f"  File: {path}")

    if not path.exists():

        print(
            "  [FAIL] File does not exist"
        )

        continue


    try:

        # Use the relative path directly.
        # This avoids the Windows/OneDrive absolute-path issue.
        ds = xr.open_dataset(
            path.as_posix()
        )

        datasets[name] = ds


        print(
            f"  Dimensions: "
            f"{dict(ds.sizes)}"
        )


        print(
            f"  Variables: "
            f"{list(ds.data_vars)}"
        )


        print(
            f"  Coordinates: "
            f"{list(ds.coords)}"
        )


        print(
            "  [PASS] Dataset opened successfully"
        )


    except Exception as e:

        print(
            "  [FAIL] Could not open dataset"
        )

        print(
            f"  Error: {e}"
        )


# ============================================================
# 4. VARIABLE CHECK
# ============================================================

print("\n" + "=" * 70)
print("VARIABLE CHECK")
print("=" * 70)


expected_variables = {

    "sst": [
        "sst"
    ],

    "sss": [
        "sss"
    ],

    "sla": [
        "sla"
    ],

    "currents": [
        "uo",
        "vo"
    ],

    "winds": [
        "u_wind",
        "v_wind"
    ],

    "thetao": [
        "thetao"
    ],
}


for name, expected in expected_variables.items():

    if name not in datasets:

        print(
            f"[FAIL] {name}: "
            f"dataset not available"
        )

        continue


    available = list(
        datasets[name].data_vars
    )


    for variable in expected:

        check(
            variable in available,
            f"{name}: "
            f"variable '{variable}' present"
        )


# ============================================================
# 5. UNITS
# ============================================================

print("\n" + "=" * 70)
print("UNITS")
print("=" * 70)


for name, ds in datasets.items():

    for variable in ds.data_vars:

        units = ds[
            variable
        ].attrs.get(
            "units",
            "NOT PROVIDED"
        )


        print(
            f"[INFO] "
            f"{name}/{variable}: "
            f"units = {units}"
        )


# ============================================================
# 6. TIME CHECK
# ============================================================

print("\n" + "=" * 70)
print("TIME CHECK")
print("=" * 70)


for name, ds in datasets.items():

    if "time" not in ds.coords:

        print(
            f"[FAIL] {name}: "
            f"time coordinate missing"
        )

        continue


    times = ds["time"].values


    if len(times) == 0:

        print(
            f"[FAIL] {name}: "
            f"time coordinate is empty"
        )

        continue


    print(
        f"[PASS] {name}: "
        f"{len(times)} time steps | "
        f"{times[0]} -> {times[-1]}"
    )


# ============================================================
# 7. DEPTH CHECK
# ============================================================

print("\n" + "=" * 70)
print("DEPTH CHECK")
print("=" * 70)


thetao = datasets.get(
    "thetao"
)


if thetao is not None and "depth" in thetao.coords:

    actual_depths = (
        thetao["depth"].values
    )


    requested_depths = (
        config["target_depths_m"]
    )


    print(
        f"Actual target depth count: "
        f"{len(actual_depths)}"
    )


    print(
        f"Requested depth count: "
        f"{len(requested_depths)}"
    )


    print(
        f"Actual depths: "
        f"{actual_depths}"
    )


    check(
        len(actual_depths)
        == len(requested_depths),

        "thetao depth count "
        "matches configuration"
    )


else:

    print(
        "[FAIL] thetao depth "
        "coordinate unavailable"
    )


# ============================================================
# 8. SPATIAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL CHECK")
print("=" * 70)


for name, ds in datasets.items():

    if (
        "latitude" not in ds.coords
        or
        "longitude" not in ds.coords
    ):

        print(
            f"[FAIL] {name}: "
            f"spatial coordinates missing"
        )

        continue


    lat = ds[
        "latitude"
    ].values


    lon = ds[
        "longitude"
    ].values


    print(
        f"[PASS] {name}: "
        f"latitude={len(lat)}, "
        f"longitude={len(lon)}"
    )


    print(
        f"       lat range: "
        f"{float(lat.min())} "
        f"to "
        f"{float(lat.max())}"
    )


    print(
        f"       lon range: "
        f"{float(lon.min())} "
        f"to "
        f"{float(lon.max())}"
    )


# ============================================================
# 9. MISSING DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("MISSING DATA CHECK")
print("=" * 70)


for name, ds in datasets.items():

    for variable in ds.data_vars:

        try:

            nan_count = int(
                ds[variable]
                .isnull()
                .sum()
                .values
            )


            total_count = (
                ds[variable].size
            )


            print(
                f"[INFO] "
                f"{name}/{variable}: "
                f"NaN count = "
                f"{nan_count} / "
                f"{total_count}"
            )


        except Exception as e:

            print(
                f"[WARN] "
                f"Could not calculate "
                f"NaNs for "
                f"{name}/{variable}: "
                f"{e}"
            )


# ============================================================
# 10. CLOSE DATASETS
# ============================================================

for ds in datasets.values():

    ds.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)