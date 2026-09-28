from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from copernicus_sources import (
    HISTORY_DAYS,
    OCEANEMBED_FEATURE_ORDER,
)
from fetch_daily_inputs import build_requests, run_download
from live_window import (
    cache_window_days,
    latest_complete_target_date,
    migrate_legacy_windows,
    missing_cached_dates,
    validate_window_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "live"
LIVE_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "live"
STATUS_PATH = LIVE_PROCESSED_ROOT / "live_status.json"
HARMONIZE_SCRIPT = Path(__file__).with_name("harmonize_live_data.py")

REQUIRED_VARIABLES = OCEANEMBED_FEATURE_ORDER

LOGGER = logging.getLogger("oceanembed.daily_ingestion")


@contextmanager
def daily_cache_lock():
    LIVE_PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = LIVE_PROCESSED_ROOT / ".daily-cache.lock"
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt
            import time

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            while True:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch, harmonize, validate, and publish the next "
            "complete OceanEmbed retrospective input window."
        )
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="Explicit Copernicus candidate date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--prepare-window",
        action="store_true",
        help="Resolve and cache daily inputs for the requested target date.",
    )
    return parser.parse_args()


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def validate_harmonized_file(
    path: Path,
    target_date: date,
    window_start: date | None = None,
) -> list[str]:
    return validate_window_file(path, target_date, window_start)


def discover_latest_usable() -> date | None:
    with daily_cache_lock():
        migrate_legacy_windows()
        latest = latest_complete_target_date()
    if latest is not None:
        LOGGER.info("Latest complete local Copernicus window ends at %s", latest)
    return latest


def discover_candidate(latest_usable: date | None) -> date:
    if latest_usable is not None:
        return latest_usable + timedelta(days=1)

    raw_dates = sorted(
        (
            parsed
            for path in LIVE_RAW_ROOT.iterdir()
            if path.is_dir() and (parsed := parse_iso_date(path.name)) is not None
        )
    ) if LIVE_RAW_ROOT.is_dir() else []
    if raw_dates:
        LOGGER.info("Using latest actual raw-ingestion date as candidate: %s", raw_dates[-1])
        return raw_dates[-1]

    # With no local ingestion history, yesterday is only a fetch attempt;
    # readiness is determined exclusively by successful source and file validation.
    candidate = datetime.now(timezone.utc).date() - timedelta(days=1)
    LOGGER.info("No local live data found; attempting latest candidate date %s", candidate)
    return candidate


def write_status(
    latest_usable: date | None,
    candidate: date,
    message: str,
) -> None:
    window_start = (
        latest_usable - timedelta(days=HISTORY_DAYS - 1)
        if latest_usable is not None
        else None
    )
    payload = {
        "latestUsableDate": latest_usable.isoformat() if latest_usable else None,
        "inputWindowStart": window_start.isoformat() if window_start else None,
        "inputWindowEnd": latest_usable.isoformat() if latest_usable else None,
        "status": "READY" if latest_usable else "NOT_READY",
        "variablesReady": len(REQUIRED_VARIABLES) if latest_usable else 0,
        "requiredVariables": len(REQUIRED_VARIABLES),
        "readyVariables": list(REQUIRED_VARIABLES) if latest_usable else [],
        "ready": latest_usable is not None,
        "candidateDate": candidate.isoformat(),
        "lastChecked": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = STATUS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, STATUS_PATH)
    LOGGER.info("Availability status updated: %s", STATUS_PATH)


def run_script(script: Path, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=PROJECT_ROOT,
        check=True,
    )


def process_candidate(candidate: date) -> tuple[bool, str]:
    return prepare_requested_window(candidate)


def prepare_requested_window(target_date: date) -> tuple[bool, str]:
    with daily_cache_lock():
        return _prepare_requested_window_locked(target_date)


def _prepare_requested_window_locked(target_date: date) -> tuple[bool, str]:
    migrate_legacy_windows()
    missing = missing_cached_dates(target_date)
    if not missing:
        return True, (
            f"Complete 7-day input window available for {target_date}; "
            "all daily observations were reused from cache"
        )

    raw_directory = LIVE_RAW_ROOT / "ranges"
    staging_directory = LIVE_PROCESSED_ROOT / ".staging"
    staging_directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Resolving target %s input window; fetching uncached dates %s",
        target_date,
        ", ".join(day.isoformat() for day in missing),
    )
    missing_groups: list[list[date]] = []
    for day in missing:
        if (
            not missing_groups
            or day != missing_groups[-1][-1] + timedelta(days=1)
        ):
            missing_groups.append([day])
        else:
            missing_groups[-1].append(day)

    for group in missing_groups:
        window_start = group[0]
        fetch_end = group[-1]
        range_name = f"{window_start.isoformat()}_{fetch_end.isoformat()}"
        staging_path = staging_directory / f"oceanembed_live_{range_name}.nc"
        try:
            requests = build_requests(fetch_end, window_start=window_start)
            for attempt in range(2):
                staging_path.unlink(missing_ok=True)
                for request in requests:
                    run_download(
                        request,
                        raw_directory,
                        force=attempt == 1,
                    )
                run_script(
                    HARMONIZE_SCRIPT,
                    "--date",
                    fetch_end.isoformat(),
                    "--window-start",
                    window_start.isoformat(),
                    "--output",
                    str(staging_path),
                )
                errors = validate_harmonized_file(
                    staging_path,
                    fetch_end,
                    window_start,
                )
                if not errors:
                    break
                if attempt == 1:
                    raise ValueError("; ".join(errors))
                LOGGER.warning(
                    "Downloaded range %s failed validation; refreshing cached "
                    "source files once: %s",
                    range_name,
                    "; ".join(errors),
                )
            published = cache_window_days(
                staging_path,
                fetch_end,
                window_start,
                only_dates=set(group),
            )
            LOGGER.info(
                "Cached validated daily observations: %s",
                ", ".join(day.isoformat() for day in published),
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            remaining = missing_cached_dates(target_date)
            return False, (
                f"Could not prepare requested input window for {target_date}: {exc}. "
                "Missing dates: "
                + (", ".join(day.isoformat() for day in remaining) or "none")
            )
        finally:
            staging_path.unlink(missing_ok=True)

    remaining = missing_cached_dates(target_date)
    if remaining:
        return False, (
            f"Requested input window for {target_date} is incomplete. "
            "Missing dates: "
            + ", ".join(day.isoformat() for day in remaining)
        )
    return True, (
        f"Complete 7-day input window available for {target_date}; "
        "daily observations were fetched, harmonized, validated, and cached"
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.prepare_window:
        if args.date is None:
            LOGGER.error("--prepare-window requires --date")
            return 2
        success, message = prepare_requested_window(args.date)
        latest_usable = latest_complete_target_date()
        write_status(latest_usable, args.date, message)
        if success:
            LOGGER.info("%s", message)
            return 0
        LOGGER.warning("%s", message)
        return 1

    latest_usable = discover_latest_usable()
    candidate = args.date or discover_candidate(latest_usable)

    if latest_usable is not None and candidate <= latest_usable:
        LOGGER.info(
            "Candidate %s is already covered by validated live data through %s.",
            candidate,
            latest_usable,
        )
        write_status(
            latest_usable,
            candidate,
            f"Complete 7-day input window available through {latest_usable}",
        )
        return 0

    success, message = process_candidate(candidate)
    if success:
        latest_usable = candidate
    write_status(latest_usable, candidate, message)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
