from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from copernicus_sources import (
    DOMAIN,
    SOURCES,
    GRID_RESOLUTION_DEG,
    HISTORY_DAYS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "live"


# ============================================================
# OceanEmbed live-source constants
# ============================================================

# Copernicus global ocean analysis/forecast salinity dataset
# uses approximately 0.494 m as its uppermost model level.
SSS_SURFACE_DEPTH_M = 0.49402499198913574


# ============================================================
# Download request definition
# ============================================================

@dataclass(frozen=True)
class DownloadRequest:
    name: str
    dataset_id: str
    variable_names: tuple[str, ...]
    start_datetime: str
    end_datetime: str
    output_filename: str
    minimum_depth: float | None = None
    maximum_depth: float | None = None


# ============================================================
# Date helpers
# ============================================================

def parse_date(value: str) -> date:
    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def build_datetime_range(
    target_date: date,
) -> tuple[str, str]:

    first_day = target_date - timedelta(
        days=HISTORY_DAYS - 1
    )

    start_datetime = (
        f"{first_day.isoformat()} 00:00:00"
    )

    end_datetime = (
        f"{target_date.isoformat()} 23:59:59"
    )

    return start_datetime, end_datetime


# ============================================================
# Source helpers
# ============================================================

def get_source_variables(source) -> tuple[str, ...]:
    """
    Return the variables configured for a Copernicus source.
    """

    return tuple(source.variable_names)


# ============================================================
# Build download requests
# ============================================================

def build_requests(
    target_date: date,
    window_start: date | None = None,
) -> list[DownloadRequest]:

    default_start = target_date - timedelta(days=HISTORY_DAYS - 1)
    start_date = window_start or default_start
    if start_date > target_date:
        raise ValueError("window_start must not be later than target_date")

    start_datetime = f"{start_date.isoformat()} 00:00:00"
    end_datetime = f"{target_date.isoformat()} 23:59:59"
    suffix = (
        f"{target_date.isoformat()}_7day"
        if start_date == default_start
        else f"{start_date.isoformat()}_{target_date.isoformat()}"
    )

    requests: list[DownloadRequest] = []

    # --------------------------------------------------------
    # SST
    # --------------------------------------------------------

    sst = SOURCES["sst"]

    requests.append(
        DownloadRequest(
            name="sst",
            dataset_id=sst.dataset_id,
            variable_names=get_source_variables(sst),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_filename=(
                f"sst_{suffix}.nc"
            ),
        )
    )

    # --------------------------------------------------------
    # SSS
    #
    # IMPORTANT:
    # The Copernicus `so` dataset contains many depth levels.
    # OceanEmbed only needs the uppermost model level at
    # approximately 0.494 m.
    #
    # Therefore the downloader explicitly requests only this
    # depth level.
    # --------------------------------------------------------

    sss = SOURCES["sss"]

    requests.append(
        DownloadRequest(
            name="sss",
            dataset_id=sss.dataset_id,
            variable_names=get_source_variables(sss),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_filename=(
                f"sss_{suffix}.nc"
            ),
            minimum_depth=SSS_SURFACE_DEPTH_M,
            maximum_depth=SSS_SURFACE_DEPTH_M,
        )
    )

    # --------------------------------------------------------
    # SLA
    # --------------------------------------------------------

    sla = SOURCES["sla"]

    requests.append(
        DownloadRequest(
            name="sla",
            dataset_id=sla.dataset_id,
            variable_names=get_source_variables(sla),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_filename=(
                f"sla_{suffix}.nc"
            ),
        )
    )

    # --------------------------------------------------------
    # Surface currents
    # --------------------------------------------------------

    currents = SOURCES["currents"]

    requests.append(
        DownloadRequest(
            name="currents",
            dataset_id=currents.dataset_id,
            variable_names=get_source_variables(currents),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_filename=(
                f"currents_{suffix}.nc"
            ),
        )
    )

    # --------------------------------------------------------
    # Surface winds
    #
    # Wind is hourly. We intentionally download the hourly
    # fields here. Daily averaging happens during
    # harmonization.
    # --------------------------------------------------------

    winds = SOURCES["winds"]

    requests.append(
        DownloadRequest(
            name="winds",
            dataset_id=winds.dataset_id,
            variable_names=get_source_variables(winds),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_filename=(
                f"winds_{suffix}_hourly.nc"
            ),
        )
    )

    return requests


# ============================================================
# Build Copernicus subset command
# ============================================================

def build_subset_command(
    request: DownloadRequest,
    output_directory: Path,
) -> list[str]:

    executable_name = "copernicusmarine.exe" if os.name == "nt" else "copernicusmarine"
    local_executable = Path(sys.executable).parent / executable_name
    executable = (
        str(local_executable)
        if local_executable.is_file()
        else shutil.which("copernicusmarine") or "copernicusmarine"
    )
    command = [
        executable,
        "subset",
        "--dataset-id",
        request.dataset_id,
    ]

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    for variable_name in request.variable_names:

        command.extend(
            [
                "--variable",
                variable_name,
            ]
        )

    # --------------------------------------------------------
    # Spatial + temporal subset
    # --------------------------------------------------------

    command.extend(
        [
            "--start-datetime",
            request.start_datetime,

            "--end-datetime",
            request.end_datetime,

            "--minimum-longitude",
            str(DOMAIN["lon_min"]),

            "--maximum-longitude",
            str(DOMAIN["lon_max"]),

            "--minimum-latitude",
            str(DOMAIN["lat_min"]),

            "--maximum-latitude",
            str(DOMAIN["lat_max"]),

            "--coordinates-selection-method",
            "inside",
        ]
    )

    # --------------------------------------------------------
    # Optional depth subset
    #
    # Used only for SSS.
    # --------------------------------------------------------

    if request.minimum_depth is not None:

        command.extend(
            [
                "--minimum-depth",
                str(request.minimum_depth),
            ]
        )

    if request.maximum_depth is not None:

        command.extend(
            [
                "--maximum-depth",
                str(request.maximum_depth),
            ]
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    command.extend(
        [
            "--output-directory",
            str(output_directory),

            "--output-filename",
            request.output_filename,

            "--file-format",
            "netcdf",

            "--skip-existing",

            "--netcdf-compression-level",
            "4",

            "--disable-progress-bar",
        ]
    )

    return command


# ============================================================
# Output path
# ============================================================

def expected_output_path(
    output_directory: Path,
    request: DownloadRequest,
) -> Path:

    return output_directory / request.output_filename


# ============================================================
# Execute download
# ============================================================

def run_download(
    request: DownloadRequest,
    output_directory: Path,
    force: bool = False,
) -> Path:

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = expected_output_path(
        output_directory,
        request,
    )

    if output_path.exists() and not force:

        print(
            f"[SKIP] {request.name}: "
            f"{output_path.name} already exists"
        )

        return output_path

    if output_path.exists() and force:
        print(
            f"[RETRY] {request.name}: replacing existing "
            f"{output_path.name}"
        )
        output_path.unlink()

    command = build_subset_command(
        request,
        output_directory,
    )

    print()
    print("=" * 72)
    print(f"Downloading: {request.name}")
    print(f"Dataset:     {request.dataset_id}")
    print(
        f"Variables:   "
        f"{', '.join(request.variable_names)}"
    )

    if request.minimum_depth is not None:

        print(
            f"Depth:       "
            f"{request.minimum_depth:.15f} m"
        )

    if request.maximum_depth is not None:

        print(
            f"Depth max:   "
            f"{request.maximum_depth:.15f} m"
        )

    print(
        f"Time:        "
        f"{request.start_datetime} -> "
        f"{request.end_datetime}"
    )

    print(
        f"Domain:      "
        f"{DOMAIN['lat_min']}-{DOMAIN['lat_max']} N, "
        f"{DOMAIN['lon_min']}-{DOMAIN['lon_max']} E"
    )

    print(f"Output:      {output_path}")
    print("=" * 72)

    try:

        subprocess.run(
            command,
            check=True,
        )

    except FileNotFoundError as exc:

        raise RuntimeError(
            "The 'copernicusmarine' command was not found. "
            "Make sure the .copernicus-venv environment is activated."
        ) from exc

    except subprocess.CalledProcessError as exc:

        raise RuntimeError(
            f"Copernicus download failed for "
            f"'{request.name}' "
            f"with exit code {exc.returncode}."
        ) from exc

    if not output_path.exists():

        raise RuntimeError(
            "Copernicus command completed, "
            "but the expected output was not found:\n"
            f"{output_path}"
        )

    print(
        f"[OK] {request.name}: "
        f"{output_path}"
    )

    return output_path


# ============================================================
# Summary
# ============================================================

def print_summary(
    target_date: date,
    output_directory: Path,
    downloaded_files: list[Path],
) -> None:

    start_date = target_date - timedelta(
        days=HISTORY_DAYS - 1
    )

    print()
    print("=" * 72)
    print("LIVE INPUT DOWNLOAD COMPLETE")
    print("=" * 72)

    print(
        f"Target date : "
        f"{target_date.isoformat()}"
    )

    print(
        f"Window      : "
        f"{start_date.isoformat()} -> "
        f"{target_date.isoformat()}"
    )

    print(
        f"History     : "
        f"{HISTORY_DAYS} days"
    )

    print(
        f"Grid target : "
        f"{GRID_RESOLUTION_DEG} x "
        f"{GRID_RESOLUTION_DEG}"
    )

    print(
        f"Domain      : "
        f"{DOMAIN['lat_min']}-"
        f"{DOMAIN['lat_max']} N, "
        f"{DOMAIN['lon_min']}-"
        f"{DOMAIN['lon_max']} E"
    )

    print(
        f"Output dir  : "
        f"{output_directory}"
    )

    print()
    print("Files:")

    for path in downloaded_files:

        print(
            f"  - {path.name}"
        )

    print()
    print(
        "Next step: inspect these raw datasets, "
        "then perform temporal/spatial harmonization."
    )

    print("=" * 72)


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Download the 7-day Copernicus Marine "
            "input window required by OceanEmbed."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        type=parse_date,
        help=(
            "Target date in YYYY-MM-DD format."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace existing source files so incomplete or delayed "
            "downloads can be retried."
        ),
    )

    args = parser.parse_args()

    target_date: date = args.date

    start_date = target_date - timedelta(
        days=HISTORY_DAYS - 1
    )

    output_directory = (
        LIVE_RAW_ROOT
        / target_date.isoformat()
    )

    print("=" * 72)
    print("OceanEmbed Live Copernicus Ingestion")
    print("=" * 72)

    print(
        f"Target date : "
        f"{target_date.isoformat()}"
    )

    print(
        f"7-day window: "
        f"{start_date.isoformat()} -> "
        f"{target_date.isoformat()}"
    )

    print(
        f"Output      : "
        f"{output_directory}"
    )

    print()

    requests = build_requests(
        target_date
    )

    downloaded_files: list[Path] = []

    for request in requests:

        path = run_download(
            request,
            output_directory,
            force=args.force,
        )

        downloaded_files.append(
            path
        )

    print_summary(
        target_date,
        output_directory,
        downloaded_files,
    )


if __name__ == "__main__":
    main()