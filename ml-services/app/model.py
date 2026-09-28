from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib.util import find_spec
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import settings
from app.data import (
    LIVE_DIR,
    LiveOceanEmbedDataLoader,
    LiveWindowUnavailableError,
    OceanEmbedDataLoader,
)
from app.schemas import (
    DepthPrediction,
    PredictionRequest,
    PredictionResponse,
    SurfaceCoverageResponse,
    SurfaceObservation,
)

from scripts.inference.oceanembed_inference import (
    DEPTHS_M,
    HISTORY_DAYS,
    INPUT_CHANNELS,
    INPUT_FEATURES,
    OUTPUT_CHANNELS,
    OceanEmbedEnsemble,
)


logger = logging.getLogger("oceanembed.model")


# ============================================================
# Frozen OceanEmbed V1 runtime contract
# ============================================================

INPUT_SIZE = 64
OUTPUT_SIZE = 32
GRID_RESOLUTION = 0.25
SURFACE_OBSERVATIONS = (
    ("sst", "SST", "°C"),
    ("sss", "SSS", "PSU"),
    ("sla", "SLA", "m"),
    ("uo", "U Current", "m/s"),
    ("vo", "V Current", "m/s"),
    ("u_wind", "U Wind", "m/s"),
    ("v_wind", "V Wind", "m/s"),
)
SURFACE_FEATURES = tuple(feature for feature, _, _ in SURFACE_OBSERVATIONS)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTION_SCRIPT = PROJECT_ROOT / "scripts" / "ingestion" / "run_daily_ingestion.py"
# Historical coverage begins here; live dates may extend through today.
SUPPORTED_APPLICATION_START = date(2025, 7, 1)


@dataclass
class _LivePreparation:
    completed: threading.Event = field(default_factory=threading.Event)
    waiters: int = 0
    error: Exception | None = None


_LIVE_PREPARATIONS: dict[date, _LivePreparation] = {}
_LIVE_PREPARATIONS_GUARD = threading.Lock()


def _live_preparation(target_date: date) -> tuple[_LivePreparation, bool]:
    with _LIVE_PREPARATIONS_GUARD:
        preparation = _LIVE_PREPARATIONS.get(target_date)
        if preparation is not None:
            preparation.waiters += 1
            return preparation, False
        preparation = _LivePreparation()
        _LIVE_PREPARATIONS[target_date] = preparation
        return preparation, True


class OceanEmbedModel:
    """
    Backend wrapper around the real OceanEmbed V1 inference engine.

    Responsibilities:
    - Load the three-seed OceanEmbed ensemble.
    - Load and validate historical harmonized NetCDF datasets.
    - Detect and use live Copernicus-harmonized datasets when available.
    - Extract the real 7-day retrospective input window.
    - Run real OceanEmbed inference.
    - Return requested depth predictions at the requested location.
    """

    def __init__(self) -> None:
        self.engine: OceanEmbedEnsemble | None = None

        # Historical loader remains available as the fallback.
        self.data_loader: OceanEmbedDataLoader | None = None

        self.loaded: bool = False

        # Required by FastAPI /health.
        self.model_version: str = settings.model_version

        self.device: str = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def load(self) -> None:
        """
        Load the real OceanEmbed V1 ensemble and
        historical harmonized datasets.
        """

        try:
            logger.info(
                "Loading OceanEmbed V1 model on %s...",
                self.device,
            )

            # ------------------------------------------------
            # Load the real three-seed ensemble.
            # ------------------------------------------------

            self.engine = OceanEmbedEnsemble(
                device=self.device
            )

            logger.info(
                "OceanEmbed V1 ensemble loaded successfully."
            )

            # ------------------------------------------------
            # Load historical datasets.
            #
            # These remain the fallback for historical dates.
            # ------------------------------------------------

            logger.info(
                "Loading historical OceanEmbed datasets..."
            )

            self.data_loader = OceanEmbedDataLoader()

            logger.info(
                "Historical OceanEmbed datasets loaded successfully."
            )

            # ------------------------------------------------
            # Validate frozen runtime contract.
            # ------------------------------------------------

            self._validate_runtime_contract()

            self.loaded = True

            logger.info(
                "OceanEmbed V1 backend model is ready."
            )

        except Exception:
            self.loaded = False

            logger.exception(
                "Failed to load OceanEmbed V1 backend model."
            )

            raise

    # ========================================================
    # RUNTIME CONTRACT VALIDATION
    # ========================================================

    def _validate_runtime_contract(self) -> None:
        """
        Validate the frozen OceanEmbed V1 inference contract.
        """

        if self.engine is None:
            raise RuntimeError(
                "OceanEmbed inference engine is not loaded."
            )

        if self.data_loader is None:
            raise RuntimeError(
                "OceanEmbed historical data loader is not loaded."
            )

        if HISTORY_DAYS != 7:
            raise RuntimeError(
                f"Expected 7-day history, got {HISTORY_DAYS}."
            )

        if len(INPUT_FEATURES) != 7:
            raise RuntimeError(
                "Expected exactly 7 input features."
            )

        if INPUT_CHANNELS != 49:
            raise RuntimeError(
                f"Expected 49 input channels, got {INPUT_CHANNELS}."
            )

        if OUTPUT_CHANNELS != 15:
            raise RuntimeError(
                f"Expected 15 output channels, got {OUTPUT_CHANNELS}."
            )

        if INPUT_SIZE != 64:
            raise RuntimeError(
                f"Expected 64x64 input tile, got {INPUT_SIZE}."
            )

        if OUTPUT_SIZE != 32:
            raise RuntimeError(
                f"Expected 32x32 output tile, got {OUTPUT_SIZE}."
            )

        if list(DEPTHS_M) != list(settings.depth_levels):
            raise RuntimeError(
                "Backend depth configuration does not match "
                "OceanEmbed V1 depth configuration."
            )

        logger.info(
            "Runtime contract validation passed."
        )

    # ========================================================
    # DATE HANDLING
    # ========================================================

    def _parse_request_date(
        self,
        request_date: date | str,
    ) -> date:
        """
        Normalize the request date.

        Pydantic converts PredictionRequest.date into a
        Python datetime.date object, but this method also
        accepts a string for direct/internal callers.
        """

        if isinstance(request_date, date):
            return request_date

        return date.fromisoformat(
            str(request_date)
        )

    # ========================================================
    # LIVE DATA DETECTION
    # ========================================================

    def _live_file_for_date(
        self,
        target_date: date,
    ):
        """
        Return the expected live harmonized NetCDF path.
        """

        return (
            LIVE_DIR
            / target_date.isoformat()
            / (
                f"oceanembed_live_"
                f"{target_date.isoformat()}.nc"
            )
        )

    # ========================================================
    # INPUT WINDOW SELECTION
    # ========================================================

    def _get_input_window(
        self,
        request: PredictionRequest,
    ):
        """
        Select the correct seven-day input source.

        Priority:
            1. Live Copernicus-harmonized dataset
            2. Historical OceanEmbed dataset

        Live file example:

        data/processed/live/2026-09-20/
            oceanembed_live_2026-09-20.nc
        """

        target_date = self._parse_request_date(
            request.date
        )

        live_file = self._live_file_for_date(target_date)
        live_error: Exception | None = None

        def load_live_window():
            live_loader = LiveOceanEmbedDataLoader(target_date)
            try:
                feature_arrays, metadata = live_loader.get_window(
                    latitude=request.latitude,
                    longitude=request.longitude,
                )
                metadata["source"] = "live"
                metadata["live_file"] = str(live_file)
                return feature_arrays, metadata
            finally:
                live_loader.close()

        try:
            return load_live_window()
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            live_error = exc

        target_text = target_date.isoformat()
        historical_has_target = (
            self.data_loader is not None
            and target_text in self.data_loader.dates
        )
        if not historical_has_target:
            # Final defensive guard: never launch live ingestion for dates
            # outside the supported application range.
            if (
                target_date < SUPPORTED_APPLICATION_START
                or target_date > date.today()
            ):
                logger.info(
                    "Skipping live ingestion for out-of-range date %s "
                    "(supported %s through %s)",
                    target_text,
                    SUPPORTED_APPLICATION_START.isoformat(),
                    date.today().isoformat(),
                )
            else:
                self._prepare_live_window(target_date)
                try:
                    return load_live_window()
                except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
                    live_error = exc

        if self.data_loader is None:
            raise RuntimeError(
                "Historical OceanEmbed data loader is unavailable."
            )

        if historical_has_target:
            logger.info(
                "Live daily window is unavailable for %s; using the existing "
                "historical fallback for the same target date.",
                target_text,
            )
            feature_arrays, metadata = self.data_loader.get_window(
                target_date=target_date,
                latitude=request.latitude,
                longitude=request.longitude,
            )
            metadata["source"] = "historical"
            return feature_arrays, metadata

        missing_dates = [
            (target_date - timedelta(days=HISTORY_DAYS - 1 - index)).isoformat()
            for index in range(HISTORY_DAYS)
            if not (
                LIVE_DIR
                / "daily"
                / f"oceanembed_live_{(target_date - timedelta(days=HISTORY_DAYS - 1 - index)).isoformat()}.nc"
            ).is_file()
        ]
        raise LiveWindowUnavailableError(missing_dates) from live_error

    @staticmethod
    def _ingestion_python() -> str:
        configured = os.environ.get("OCEANEMBED_INGESTION_PYTHON")
        if configured:
            return configured
        if find_spec("copernicusmarine") is not None:
            return sys.executable
        executable_name = "python.exe" if os.name == "nt" else "python"
        environment = (
            ".copernicus-venv/Scripts"
            if os.name == "nt"
            else ".copernicus-venv/bin"
        )
        candidate = PROJECT_ROOT / environment / executable_name
        if candidate.is_file():
            return str(candidate)
        return sys.executable

    def _prepare_live_window(self, target_date: date) -> None:
        preparation, is_owner = _live_preparation(target_date)
        if not is_owner:
            preparation.completed.wait()
            with _LIVE_PREPARATIONS_GUARD:
                preparation.waiters -= 1
            if preparation.error is not None:
                raise preparation.error
            return

        try:
            try:
                live_loader = LiveOceanEmbedDataLoader(target_date)
            except (FileNotFoundError, OSError, KeyError, ValueError):
                pass
            else:
                live_loader.close()
                return
            self._run_live_window_preparation(target_date)
        except Exception as exc:
            preparation.error = exc
            raise
        finally:
            with _LIVE_PREPARATIONS_GUARD:
                if _LIVE_PREPARATIONS.get(target_date) is preparation:
                    del _LIVE_PREPARATIONS[target_date]
                preparation.completed.set()

    def _run_live_window_preparation(self, target_date: date) -> None:
        timeout_seconds = int(
            os.environ.get("OCEANEMBED_LIVE_PREPARATION_TIMEOUT_SECONDS", "1800")
        )
        command = [
            self._ingestion_python(),
            str(INGESTION_SCRIPT),
            "--prepare-window",
            "--date",
            target_date.isoformat(),
        ]
        logger.info(
            "Resolving live observations for target %s using the ingestion pipeline",
            target_date.isoformat(),
        )
        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "Live-data preparation failed for %s: %s",
                target_date.isoformat(),
                exc,
            )
            return

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            logger.warning(
                "Live-data preparation was incomplete for %s (exit %d): %s",
                target_date.isoformat(),
                result.returncode,
                detail[-2000:],
            )
        elif result.stdout.strip():
            logger.info("Live-data preparation: %s", result.stdout.strip()[-1000:])

    # ========================================================
    # PREDICTION
    # ========================================================

    @staticmethod
    def _coverage_for_window(
        feature_arrays: dict[str, np.ndarray],
        metadata: dict[str, int | float | str],
        latitude: float,
        longitude: float,
        target_date: date,
    ) -> SurfaceCoverageResponse:
        output_row = int(metadata["output_row"])
        output_col = int(metadata["output_col"])
        input_row = output_row + 16
        input_col = output_col + 16
        ready_variables = [
            feature
            for feature in SURFACE_FEATURES
            if feature in feature_arrays
            and np.isfinite(feature_arrays[feature][-1, input_row, input_col])
        ]
        missing_variables = [
            feature
            for feature in SURFACE_FEATURES
            if feature not in ready_variables
        ]
        ready = not missing_variables
        window_start = date.fromisoformat(
            str(metadata.get("window_start", target_date - timedelta(days=HISTORY_DAYS - 1)))
        )
        window_end = date.fromisoformat(
            str(metadata.get("window_end", target_date))
        )
        return SurfaceCoverageResponse(
            ready=ready,
            latitude=latitude,
            longitude=longitude,
            snappedLatitude=float(metadata["snapped_latitude"]),
            snappedLongitude=float(metadata["snapped_longitude"]),
            date=target_date,
            targetDate=target_date,
            windowStart=window_start,
            windowEnd=window_end,
            availableDates=[
                (window_start + timedelta(days=index)).isoformat()
                for index in range((window_end - window_start).days + 1)
            ],
            missingDates=[],
            requiredVariables=len(SURFACE_FEATURES),
            variablesReady=len(ready_variables),
            readyVariables=ready_variables,
            missingVariables=missing_variables,
            message=(
                "All required surface observations are available"
                if ready
                else (
                    f"Incomplete surface observations: "
                    f"{len(ready_variables)}/{len(SURFACE_FEATURES)} variables available"
                )
            ),
        )

    def check_coverage(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
    ) -> SurfaceCoverageResponse:
        if not self.loaded:
            raise RuntimeError("Real OceanEmbed model/data are not loaded")
        request = PredictionRequest(
            latitude=latitude,
            longitude=longitude,
            date=target_date,
        )
        try:
            feature_arrays, metadata = self._get_input_window(request)
        except LiveWindowUnavailableError as exc:
            start = target_date - timedelta(days=HISTORY_DAYS - 1)
            missing_dates = exc.missing_dates
            available_dates = [
                day.isoformat()
                for day in (
                    start + timedelta(days=index)
                    for index in range(HISTORY_DAYS)
                )
                if day.isoformat() not in missing_dates
            ]
            return SurfaceCoverageResponse(
                ready=False,
                latitude=latitude,
                longitude=longitude,
                snappedLatitude=None,
                snappedLongitude=None,
                date=target_date,
                targetDate=target_date,
                windowStart=start,
                windowEnd=target_date,
                availableDates=available_dates,
                missingDates=missing_dates,
                requiredVariables=len(SURFACE_FEATURES),
                variablesReady=0,
                readyVariables=[],
                missingVariables=list(SURFACE_FEATURES),
                message=(
                    "Requested target-date input window is not available. "
                    "Missing dates: " + ", ".join(missing_dates)
                ),
            )
        return self._coverage_for_window(
            feature_arrays,
            metadata,
            latitude,
            longitude,
            target_date,
        )

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """
        Run real OceanEmbed inference for one
        location/date request.
        """

        if not self.loaded:
            raise RuntimeError(
                "OceanEmbed V1 model is not loaded."
            )

        if self.engine is None:
            raise RuntimeError(
                "OceanEmbed inference engine is unavailable."
            )

        # ----------------------------------------------------
        # 1. Extract the real 7-day input window.
        #
        # This automatically chooses:
        #
        # LIVE:
        #   Copernicus-harmonized data
        #
        # OR
        #
        # HISTORICAL:
        #   Original training/test harmonized data
        # ----------------------------------------------------

        feature_arrays, metadata = (
            self._get_input_window(
                request
            )
        )
        coverage = self._coverage_for_window(
            feature_arrays,
            metadata,
            request.latitude,
            request.longitude,
            request.date,
        )
        if not coverage.ready:
            raise ValueError(coverage.message + ": " + ", ".join(coverage.missingVariables))

        # ----------------------------------------------------
        # 2. Validate feature ordering.
        # ----------------------------------------------------

        expected_features = tuple(
            INPUT_FEATURES
        )

        actual_features = tuple(
            feature_arrays.keys()
        )

        if actual_features != expected_features:
            raise RuntimeError(
                "Feature ordering mismatch. "
                f"Expected {expected_features}, "
                f"got {actual_features}."
            )

        # ----------------------------------------------------
        # 3. Validate every input shape.
        # ----------------------------------------------------

        expected_feature_shape = (
            HISTORY_DAYS,
            INPUT_SIZE,
            INPUT_SIZE,
        )

        for feature_name in expected_features:

            array = feature_arrays[
                feature_name
            ]

            if array.shape != expected_feature_shape:
                raise RuntimeError(
                    f"Unexpected shape for "
                    f"{feature_name}: "
                    f"{array.shape}. "
                    f"Expected "
                    f"{expected_feature_shape}."
                )

        # ----------------------------------------------------
        # 4. Run the real three-seed ensemble.
        # ----------------------------------------------------

        logger.info(
            "Running OceanEmbed inference for "
            "date=%s lat=%s lon=%s source=%s",
            request.date,
            request.latitude,
            request.longitude,
            metadata.get(
                "source",
                "unknown",
            ),
        )

        prediction, spread = (
            self.engine.predict_from_raw_window_with_spread(
                feature_arrays
            )
        )

        if not isinstance(prediction, torch.Tensor) or not isinstance(spread, torch.Tensor):
            raise RuntimeError(
                "OceanEmbed inference returned "
                "an unexpected type."
            )

        # ----------------------------------------------------
        # 5. Validate prediction shape.
        # ----------------------------------------------------

        expected_prediction_shape = (
            OUTPUT_CHANNELS,
            OUTPUT_SIZE,
            OUTPUT_SIZE,
        )

        if tuple(
            prediction.shape
        ) != expected_prediction_shape:

            raise RuntimeError(
                "Unexpected OceanEmbed prediction shape: "
                f"{tuple(prediction.shape)}. "
                f"Expected "
                f"{expected_prediction_shape}."
            )

        if tuple(spread.shape) != expected_prediction_shape:
            raise RuntimeError(
                "Unexpected OceanEmbed ensemble spread shape: "
                f"{tuple(spread.shape)}. "
                f"Expected {expected_prediction_shape}."
            )

        # ----------------------------------------------------
        # 6. Validate prediction values.
        # ----------------------------------------------------

        if not torch.isfinite(
            prediction
        ).all():

            raise RuntimeError(
                "OceanEmbed prediction contains "
                "non-finite values."
            )

        if not torch.isfinite(spread).all() or torch.any(spread < 0):
            raise RuntimeError(
                "OceanEmbed ensemble spread contains invalid values."
            )

        # ----------------------------------------------------
        # 7. Get requested output pixel.
        # ----------------------------------------------------

        output_row = int(
            metadata["output_row"]
        )

        output_col = int(
            metadata["output_col"]
        )

        if not (
            0 <= output_row < OUTPUT_SIZE
            and 0 <= output_col < OUTPUT_SIZE
        ):
            raise RuntimeError(
                "Invalid output location: "
                f"row={output_row}, "
                f"col={output_col}."
            )

        point_prediction = prediction[
            :,
            output_row,
            output_col,
        ]
        point_spread = spread[
            :,
            output_row,
            output_col,
        ]

        # ----------------------------------------------------
        # 8. Build requested depth response.
        # ----------------------------------------------------

        predictions: list[
            DepthPrediction
        ] = []

        for depth in request.depths:

            try:
                depth_index = list(
                    DEPTHS_M
                ).index(
                    depth
                )

            except ValueError as exc:

                raise ValueError(
                    f"Unsupported depth: {depth} m"
                ) from exc

            temperature = float(
                point_prediction[
                    depth_index
                ].item()
            )
            ensemble_spread = float(
                point_spread[depth_index].item()
            )

            predictions.append(
                DepthPrediction(
                    depth_m=depth,
                    temperature_c=temperature,
                    uncertainty_c=ensemble_spread,
                )
            )

        # Extract the final observation day at the exact input-grid
        # point corresponding to the requested output pixel.
        input_row = output_row + 16
        input_col = output_col + 16
        surface_observations: list[SurfaceObservation] = []
        for feature, label, unit in SURFACE_OBSERVATIONS:
            raw_value = float(
                feature_arrays[feature][-1, input_row, input_col]
            )
            surface_observations.append(
                SurfaceObservation(
                    variable=label,
                    value=raw_value if isfinite(raw_value) else None,
                    unit=unit,
                )
            )

        snapped_latitude = float(
            metadata[
                "snapped_latitude"
            ]
        )

        snapped_longitude = float(
            metadata[
                "snapped_longitude"
            ]
        )

        return PredictionResponse(
            latitude=snapped_latitude,
            longitude=snapped_longitude,
            date=request.date,
            input_window_start=date.fromisoformat(
                str(metadata["window_start"])
            ),
            input_window_end=date.fromisoformat(
                str(metadata["window_end"])
            ),
            model_version=self.model_version,
            grid_resolution=GRID_RESOLUTION,
            predictions=predictions,
            surface_observations=surface_observations,
        )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_model_info(
        self,
    ) -> dict[str, Any]:
        """
        Return OceanEmbed V1 model metadata.
        """

        if self.engine is None:
            raise RuntimeError(
                "OceanEmbed V1 model is not loaded."
            )

        info = self.engine.get_model_info()

        info[
            "model_version"
        ] = self.model_version

        info[
            "model_loaded"
        ] = self.loaded

        info[
            "runtime_device"
        ] = self.device

        info[
            "grid_resolution"
        ] = GRID_RESOLUTION

        info[
            "live_inference_enabled"
        ] = True

        return info

    # ========================================================
    # CLOSE / CLEANUP
    # ========================================================

    def close(self) -> None:
        """
        Release model and dataset resources.
        """

        self.loaded = False

        self.engine = None

        if self.data_loader is not None:

            try:
                self.data_loader.close()

            except Exception:
                logger.exception(
                    "Failed to close OceanEmbed "
                    "data loader cleanly."
                )

        self.data_loader = None


# ============================================================
# GLOBAL MODEL INSTANCE
# ============================================================

model = OceanEmbedModel()