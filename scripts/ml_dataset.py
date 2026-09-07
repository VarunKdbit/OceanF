"""
OceanEmbed ML Dataset Loader
============================

Role 3 - ML Model Development

Purpose
-------
Build training-ready samples from the persistent spatially harmonized
OceanEmbed datasets.

Scientific contract
-------------------
Domain:
    5°N–30°N, 45°E–105°E

Grid:
    0.25° × 0.25°

Inputs:
    SST
    SSS
    SLA
    U current
    V current
    U wind
    V wind

Target:
    GLORYS thetao potential temperature

Target depths:
    0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m

Temporal input:
    Configurable retrospective window from 1 to 7 days.

Spatial sample:
    64 × 64 input tile
    32 × 32 target tile

Input channels:
    7 variables × history_days

Target channels:
    15 depth levels

Important
---------
The loader uses the persistent harmonized NetCDF files created by
scripts/spatial_harmonization.py.

No interpolation is performed inside this loader.

Samples whose complete 32×32 target region contains zero valid
thetao cells are excluded from the dataset index. This prevents
land-only or otherwise completely invalid target tiles from reaching
the masked loss function.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from netCDF4 import Dataset as NetCDFDataset
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ML"
HARMONIZED_DIR = ML_DIR / "harmonized"
CONFIG_PATH = ML_DIR / "ml_config.json"


# ---------------------------------------------------------------------
# SCIENTIFIC CONSTANTS
# ---------------------------------------------------------------------

EXPECTED_LAT_MIN = 5.0
EXPECTED_LAT_MAX = 30.0
EXPECTED_LON_MIN = 45.0
EXPECTED_LON_MAX = 105.0
EXPECTED_RESOLUTION = 0.25

EXPECTED_LAT_SIZE = 101
EXPECTED_LON_SIZE = 241

EXPECTED_DAYS = 184

TARGET_DEPTHS = np.array(
    [
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
    ],
    dtype=np.float32,
)

NUM_INPUT_FEATURES = 7
WINDOW_SIZE = 7

INPUT_TILE_SIZE = 64
OUTPUT_TILE_SIZE = 32

# The 32×32 target sits in the center of the 64×64 input.
CONTEXT = (INPUT_TILE_SIZE - OUTPUT_TILE_SIZE) // 2

INPUT_FEATURE_NAMES = [
    "sst",
    "sss",
    "sla",
    "uo",
    "vo",
    "u_wind",
    "v_wind",
]

TARGET_VARIABLE = "thetao"


# ---------------------------------------------------------------------
# HARMONIZED FILES
# ---------------------------------------------------------------------

DATA_FILES = {
    "sst": HARMONIZED_DIR / "SST_harmonized.nc",
    "sss": HARMONIZED_DIR / "SSS_harmonized.nc",
    "sla": HARMONIZED_DIR / "SLA_harmonized.nc",
    "uo": HARMONIZED_DIR / "Currents_harmonized.nc",
    "vo": HARMONIZED_DIR / "Currents_harmonized.nc",
    "u_wind": HARMONIZED_DIR / "Winds_harmonized.nc",
    "v_wind": HARMONIZED_DIR / "Winds_harmonized.nc",
    "thetao": HARMONIZED_DIR / "SubsurfaceTemp_harmonized.nc",
}


# ---------------------------------------------------------------------
# VARIABLE NAME CANDIDATES
# ---------------------------------------------------------------------

VARIABLE_CANDIDATES = {
    "sst": [
        "sst",
        "analysed_sst",
        "sea_surface_temperature",
        "temperature",
    ],
    "sss": [
        "sss",
        "so",
        "salinity",
        "sea_surface_salinity",
    ],
    "sla": [
        "sla",
        "adt",
        "ssh",
        "sea_level_anomaly",
        "sea_surface_height_anomaly",
    ],
    "uo": [
        "uo",
        "u",
        "u_current",
        "eastward_current",
    ],
    "vo": [
        "vo",
        "v",
        "v_current",
        "northward_current",
    ],
    "u_wind": [
        "u_wind",
        "u10",
        "u10m",
        "eastward_wind",
        "10m_u_component_of_wind",
    ],
    "v_wind": [
        "v_wind",
        "v10",
        "v10m",
        "northward_wind",
        "10m_v_component_of_wind",
    ],
    "thetao": [
        "thetao",
        "temperature",
        "potential_temperature",
    ],
}


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------


class OceanEmbedDataset(Dataset):
    """
    PyTorch Dataset for OceanEmbed.

    Each sample contains:

        x:
            shape = [history_days * 7, 64, 64]

        x_mask:
            shape = [history_days * 7, 64, 64]

            1 where original input data are valid,
            0 where input data were missing.

        y:
            shape = [15, 32, 32]

            15 subsurface temperature depths.

            Invalid target cells are represented by NaN.

        y_mask:
            shape = [15, 32, 32]

            1 where target temperature is valid,
            0 where target temperature is missing.

        metadata:
            target date and tile information.

    Samples with zero valid target cells in the complete 32×32
    prediction region are excluded during dataset initialization.
    """

    def __init__(
        self,
        split: str = "train",
        tile_stride: int = 32,
        return_metadata: bool = False,
        normalize: bool = True,
        history_days: int = 7,
    ):
        super().__init__()

        self.split = split.lower()
        self.tile_stride = int(tile_stride)
        self.return_metadata = return_metadata
        self.normalize = normalize
        self.history_days = int(history_days)

        if self.history_days < 1 or self.history_days > 7:
            raise ValueError(
                "history_days must be between 1 and 7."
            )

        if self.split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be one of: train, validation, test"
            )

        if self.tile_stride <= 0:
            raise ValueError("tile_stride must be > 0")

        if INPUT_TILE_SIZE != 64 or OUTPUT_TILE_SIZE != 32:
            raise ValueError("OceanEmbed tile contract was changed.")

        self.config = self._load_config()

        # -------------------------------------------------------------
        # Open all unique NetCDF files.
        # -------------------------------------------------------------

        self._datasets: Dict[Path, NetCDFDataset] = {}

        for path in set(DATA_FILES.values()):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required harmonized file not found:\n{path}"
                )

            self._datasets[path] = NetCDFDataset(path, "r")

        # -------------------------------------------------------------
        # Discover actual variable names.
        # -------------------------------------------------------------

        self.variable_names = {}

        for feature in INPUT_FEATURE_NAMES + [TARGET_VARIABLE]:
            path = DATA_FILES[feature]
            ds = self._datasets[path]

            self.variable_names[feature] = self._find_variable(
                ds,
                feature,
            )

        # -------------------------------------------------------------
        # Validate all files and grids.
        # -------------------------------------------------------------

        self._validate_datasets()

        # -------------------------------------------------------------
        # Common time axis.
        # -------------------------------------------------------------

        self.dates = self._get_common_dates()

        if len(self.dates) != EXPECTED_DAYS:
            raise ValueError(
                f"Expected {EXPECTED_DAYS} common dates, "
                f"found {len(self.dates)}."
            )

        # -------------------------------------------------------------
        # Split dates.
        # -------------------------------------------------------------

        self.target_date_indices = self._build_split_indices()

        # -------------------------------------------------------------
        # Retrospective window.
        #
        # A target at index i requires:
        #
        #     i-history_days+1, ..., i
        #
        # -------------------------------------------------------------

        self.sample_date_indices = []

        for target_index in self.target_date_indices:

            window_start = target_index - self.history_days + 1

            if window_start < 0:
                continue

            self.sample_date_indices.append(target_index)

        if len(self.sample_date_indices) == 0:
            raise ValueError(
                f"No usable {self.split} samples after applying "
                f"{self.history_days}-day retrospective window."
            )

        # -------------------------------------------------------------
        # Tile positions.
        # -------------------------------------------------------------

        self.tile_positions = self._build_tile_positions()

        if len(self.tile_positions) == 0:
            raise ValueError(
                "No valid 64×64 input tiles were generated."
            )

        # -------------------------------------------------------------
        # Normalization statistics.
        # -------------------------------------------------------------

        self.input_stats = self._load_input_statistics()
        self.target_stats = self._load_target_statistics()

        # -------------------------------------------------------------
        # Build the actual valid sample index.
        #
        # A sample is represented by:
        #
        #     (target-date-position, tile-position-index)
        #
        # Any sample whose complete 32×32 target region has zero
        # finite thetao cells is excluded.
        # -------------------------------------------------------------

        self.sample_index = self._build_valid_sample_index()

        if len(self.sample_index) == 0:
            raise ValueError(
                f"No valid {self.split} samples remain after "
                "target-validity filtering."
            )

        self.invalid_target_samples = (
            len(self.sample_date_indices) * len(self.tile_positions)
            - len(self.sample_index)
        )

        self.valid_target_samples = len(self.sample_index)

        print()
        print("=" * 72)
        print("OceanEmbedDataset initialized")
        print("=" * 72)
        print(f"Split                 : {self.split}")
        print(f"Common dates          : {len(self.dates)}")
        print(
            f"Target dates          : "
            f"{len(self.target_date_indices)}"
        )
        print(
            f"History days          : {self.history_days}"
        )
        print(
            f"Input channels        : "
            f"{self.history_days * len(INPUT_FEATURE_NAMES)}"
        )
        print(
            f"Usable {self.history_days}-day target dates : "
            f"{len(self.sample_date_indices)}"
        )
        print(
            f"Tile positions        : "
            f"{len(self.tile_positions)}"
        )
        print(
            f"Valid target samples  : "
            f"{self.valid_target_samples}"
        )
        print(
            f"Excluded invalid     : "
            f"{self.invalid_target_samples}"
        )
        print(f"Tile stride           : {self.tile_stride}")
        print(
            f"Input shape/sample    : "
            f"[{self.history_days * len(INPUT_FEATURE_NAMES)}, "
            f"{INPUT_TILE_SIZE}, {INPUT_TILE_SIZE}]"
        )
        print(
            f"Target shape/sample   : "
            f"[15, {OUTPUT_TILE_SIZE}, {OUTPUT_TILE_SIZE}]"
        )
        print("=" * 72)
        print()

    # -----------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------

    def _load_config(self) -> dict:

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"ML configuration file not found:\n{CONFIG_PATH}"
            )

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------------------------------
    # VARIABLE DISCOVERY
    # -----------------------------------------------------------------

    @staticmethod
    def _find_variable(
        ds: NetCDFDataset,
        feature: str,
    ) -> str:

        candidates = VARIABLE_CANDIDATES[feature]

        for candidate in candidates:
            if candidate in ds.variables:
                return candidate

        # Exclude coordinate variables.
        coordinate_names = {
            "time",
            "valid_time",
            "latitude",
            "longitude",
            "depth",
            "lat",
            "lon",
        }

        data_variables = [
            name
            for name in ds.variables
            if name not in coordinate_names
        ]

        if len(data_variables) == 1:
            return data_variables[0]

        raise KeyError(
            f"Could not identify variable for '{feature}'.\n"
            f"Available variables: {list(ds.variables.keys())}"
        )

    # -----------------------------------------------------------------
    # COORDINATE HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _get_coordinate(
        ds: NetCDFDataset,
        names: List[str],
    ) -> np.ndarray:

        for name in names:
            if name in ds.variables:
                return np.asarray(ds.variables[name][:])

        raise KeyError(
            f"Could not find coordinate. Tried: {names}"
        )

    @staticmethod
    def _get_time_coordinate(
        ds: NetCDFDataset,
    ) -> np.ndarray:

        if "time" in ds.variables:
            return np.asarray(ds.variables["time"][:])

        if "valid_time" in ds.variables:
            return np.asarray(ds.variables["valid_time"][:])

        raise KeyError(
            "Could not find 'time' or 'valid_time' coordinate."
        )

    # -----------------------------------------------------------------
    # DATASET VALIDATION
    # -----------------------------------------------------------------

    def _validate_datasets(self):

        print("Validating harmonized datasets...")

        reference_lat = None
        reference_lon = None
        reference_time = None

        for feature in INPUT_FEATURE_NAMES + [TARGET_VARIABLE]:

            path = DATA_FILES[feature]
            ds = self._datasets[path]
            variable_name = self.variable_names[feature]

            lat = self._get_coordinate(
                ds,
                ["latitude", "lat"],
            ).astype(np.float64)

            lon = self._get_coordinate(
                ds,
                ["longitude", "lon"],
            ).astype(np.float64)

            time = self._get_time_coordinate(ds)

            # ---------------------------------------------------------
            # Latitude
            # ---------------------------------------------------------

            if len(lat) != EXPECTED_LAT_SIZE:
                raise ValueError(
                    f"{feature}: expected {EXPECTED_LAT_SIZE} "
                    f"latitude points, got {len(lat)}"
                )

            if not np.all(np.diff(lat) > 0):
                raise ValueError(
                    f"{feature}: latitude is not strictly ascending."
                )

            # ---------------------------------------------------------
            # Longitude
            # ---------------------------------------------------------

            if len(lon) != EXPECTED_LON_SIZE:
                raise ValueError(
                    f"{feature}: expected {EXPECTED_LON_SIZE} "
                    f"longitude points, got {len(lon)}"
                )

            if not np.all(np.diff(lon) > 0):
                raise ValueError(
                    f"{feature}: longitude is not strictly ascending."
                )

            # ---------------------------------------------------------
            # Grid values
            # ---------------------------------------------------------

            if not np.isclose(
                lat[0],
                EXPECTED_LAT_MIN,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{feature}: latitude minimum mismatch."
                )

            if not np.isclose(
                lat[-1],
                EXPECTED_LAT_MAX,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{feature}: latitude maximum mismatch."
                )

            if not np.isclose(
                lon[0],
                EXPECTED_LON_MIN,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{feature}: longitude minimum mismatch."
                )

            if not np.isclose(
                lon[-1],
                EXPECTED_LON_MAX,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{feature}: longitude maximum mismatch."
                )

            if not np.allclose(
                np.diff(lat),
                EXPECTED_RESOLUTION,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{feature}: latitude resolution mismatch."
                )

            if not np.allclose(
                np.diff(lon),
                EXPECTED_RESOLUTION,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{feature}: longitude resolution mismatch."
                )

            # ---------------------------------------------------------
            # Reference grid consistency
            # ---------------------------------------------------------

            if reference_lat is None:
                reference_lat = lat
                reference_lon = lon
                reference_time = time

            else:

                if not np.allclose(
                    lat,
                    reference_lat,
                    atol=1e-5,
                ):
                    raise ValueError(
                        f"{feature}: latitude grid differs."
                    )

                if not np.allclose(
                    lon,
                    reference_lon,
                    atol=1e-5,
                ):
                    raise ValueError(
                        f"{feature}: longitude grid differs."
                    )

                if len(time) != len(reference_time):
                    raise ValueError(
                        f"{feature}: time dimension differs."
                    )

            # ---------------------------------------------------------
            # Variable dimensionality
            # ---------------------------------------------------------

            dimensions = ds.variables[variable_name].dimensions

            if feature == TARGET_VARIABLE:

                if "depth" not in ds.variables:
                    raise ValueError(
                        "thetao dataset has no depth coordinate."
                    )

                depth = np.asarray(
                    ds.variables["depth"][:]
                ).astype(np.float32)

                if len(depth) != len(TARGET_DEPTHS):
                    raise ValueError(
                        f"thetao: expected {len(TARGET_DEPTHS)} "
                        f"depth levels, got {len(depth)}"
                    )

                if not np.allclose(
                    depth,
                    TARGET_DEPTHS,
                    atol=1e-4,
                ):
                    raise ValueError(
                        "thetao depth labels do not match the "
                        "OceanEmbed target-depth contract."
                    )

                if len(dimensions) != 4:
                    raise ValueError(
                        f"thetao expected 4 dimensions, "
                        f"got {dimensions}"
                    )

            else:

                if len(dimensions) != 3:
                    raise ValueError(
                        f"{feature} expected 3 dimensions, "
                        f"got {dimensions}"
                    )

            print(
                f"  PASS: {feature:8s} -> "
                f"{path.name} -> {variable_name}"
            )

        print("Dataset validation PASSED.")

    # -----------------------------------------------------------------
    # COMMON DATES
    # -----------------------------------------------------------------

    def _get_common_dates(self) -> List[str]:

        """
        Use the harmonized file dates.

        The six harmonized datasets share the common
        2025-07-01 through 2025-12-31 period.
        """

        reference_path = DATA_FILES["sst"]
        ds = self._datasets[reference_path]

        time_var = (
            ds.variables["time"]
            if "time" in ds.variables
            else ds.variables["valid_time"]
        )

        dates = []

        units = getattr(time_var, "units", None)
        calendar = getattr(time_var, "calendar", "standard")

        if units is not None:

            from netCDF4 import num2date

            converted = num2date(
                time_var[:],
                units=units,
                calendar=calendar,
            )

            for value in converted:
                dates.append(
                    f"{value.year:04d}-"
                    f"{value.month:02d}-"
                    f"{value.day:02d}"
                )

        else:

            raw = np.asarray(time_var[:])

            for value in raw:
                dates.append(str(value)[:10])

        if len(dates) == 0:
            raise ValueError("No dates found.")

        return dates

    # -----------------------------------------------------------------
    # SPLIT INDICES
    # -----------------------------------------------------------------

    def _build_split_indices(self) -> List[int]:

        time_config = self.config["time"]

        if self.split == "train":
            start = time_config["train_start"]
            end = time_config["train_end"]

        elif self.split == "validation":
            start = time_config["validation_start"]
            end = time_config["validation_end"]

        else:
            start = time_config["test_start"]
            end = time_config["test_end"]

        indices = [
            i
            for i, date in enumerate(self.dates)
            if start <= date <= end
        ]

        if len(indices) == 0:
            raise ValueError(
                f"No dates found for split '{self.split}' "
                f"between {start} and {end}."
            )

        return indices

    # -----------------------------------------------------------------
    # TILE POSITIONS
    # -----------------------------------------------------------------

    def _build_tile_positions(
        self,
    ) -> List[Tuple[int, int]]:

        """
        Generate full 64×64 input tiles.

        The 32×32 prediction region is the center of the input tile.

        Therefore:

            input:
                [row:row+64, col:col+64]

            target:
                [row+16:row+48, col+16:col+48]

        Only full 64×64 tiles are used. No artificial spatial padding
        is introduced.
        """

        max_row = EXPECTED_LAT_SIZE - INPUT_TILE_SIZE
        max_col = EXPECTED_LON_SIZE - INPUT_TILE_SIZE

        positions = []

        row_starts = list(
            range(
                0,
                max_row + 1,
                self.tile_stride,
            )
        )

        col_starts = list(
            range(
                0,
                max_col + 1,
                self.tile_stride,
            )
        )

        # Ensure the final boundary is included if the stride does
        # not land exactly on it.
        if row_starts[-1] != max_row:
            row_starts.append(max_row)

        if col_starts[-1] != max_col:
            col_starts.append(max_col)

        for row in row_starts:
            for col in col_starts:
                positions.append((row, col))

        return positions

    # -----------------------------------------------------------------
    # VALID TARGET SAMPLE INDEX
    # -----------------------------------------------------------------

    def _build_valid_sample_index(
        self,
    ) -> List[Tuple[int, int]]:
        """
        Build the flattened sample index using target validity.

        Each returned tuple is:

            (date_sample_position, tile_position_index)

        A sample is included only if its complete 32×32 target region
        contains at least one finite thetao value.

        This filtering is performed on the target data itself, before
        any normalization, so land-only or completely missing target
        regions cannot enter the training/evaluation loop.
        """

        target_path = DATA_FILES[TARGET_VARIABLE]
        target_ds = self._datasets[target_path]
        target_variable = target_ds.variables[
            self.variable_names[TARGET_VARIABLE]
        ]

        valid_samples: List[Tuple[int, int]] = []

        total_candidates = (
            len(self.sample_date_indices)
            * len(self.tile_positions)
        )

        checked = 0

        for date_sample_position, target_time_index in enumerate(
            self.sample_date_indices
        ):

            # Read one complete target day once, then inspect all
            # spatial tiles from that in-memory array.
            target_day = np.asarray(
                target_variable[
                    target_time_index,
                    :,
                    :,
                    :,
                ],
                dtype=np.float32,
            )

            finite_target = np.isfinite(target_day)

            for tile_index, (tile_row, tile_col) in enumerate(
                self.tile_positions
            ):

                target_row = tile_row + CONTEXT
                target_col = tile_col + CONTEXT

                target_mask_patch = finite_target[
                    :,
                    target_row : target_row + OUTPUT_TILE_SIZE,
                    target_col : target_col + OUTPUT_TILE_SIZE,
                ]

                if bool(target_mask_patch.any()):
                    valid_samples.append(
                        (
                            date_sample_position,
                            tile_index,
                        )
                    )

                checked += 1

        if checked != total_candidates:
            raise RuntimeError(
                "Internal error while building target-validity "
                "sample index."
            )

        return valid_samples

    # -----------------------------------------------------------------
    # NORMALIZATION STATISTICS
    # -----------------------------------------------------------------

    def _load_input_statistics(
        self,
    ) -> Dict[str, Tuple[float, float]]:

        stats = self.config.get("input_statistics")

        if stats is None:
            raise KeyError(
                "input_statistics missing from ml_config.json"
            )

        result = {}

        for feature in INPUT_FEATURE_NAMES:

            if feature not in stats:
                raise KeyError(
                    f"Missing normalization statistics for {feature}"
                )

            mean = float(stats[feature]["mean"])
            std = float(stats[feature]["std"])

            if std <= 0:
                raise ValueError(
                    f"Invalid standard deviation for {feature}: {std}"
                )

            result[feature] = (mean, std)

        return result

    def _load_target_statistics(
        self,
    ) -> Dict[float, Tuple[float, float]]:

        stats = self.config.get("target_statistics")

        if stats is None:
            raise KeyError(
                "target_statistics missing from ml_config.json"
            )

        result = {}

        for depth in TARGET_DEPTHS:

            key = str(float(depth))

            if key not in stats:
                raise KeyError(
                    f"Missing target statistics for depth {depth}"
                )

            mean = float(stats[key]["mean"])
            std = float(stats[key]["std"])

            if std <= 0:
                raise ValueError(
                    f"Invalid target std at depth {depth}: {std}"
                )

            result[float(depth)] = (mean, std)

        return result

    # -----------------------------------------------------------------
    # READ SINGLE INPUT VARIABLE
    # -----------------------------------------------------------------

    def _read_input_day(
        self,
        feature: str,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        path = DATA_FILES[feature]
        ds = self._datasets[path]
        variable_name = self.variable_names[feature]

        variable = ds.variables[variable_name]

        array = np.asarray(
            variable[time_index, :, :],
            dtype=np.float32,
        )

        valid_mask = np.isfinite(array)

        if self.normalize:

            mean, std = self.input_stats[feature]

            normalized = (
                array.astype(np.float32) - mean
            ) / std

            # Missing values become zero AFTER normalization.
            normalized = np.where(
                valid_mask,
                normalized,
                0.0,
            ).astype(np.float32)

            return (
                normalized,
                valid_mask.astype(np.float32),
            )

        else:

            array = np.where(
                valid_mask,
                array,
                0.0,
            ).astype(np.float32)

            return (
                array,
                valid_mask.astype(np.float32),
            )

    # -----------------------------------------------------------------
    # READ TARGET
    # -----------------------------------------------------------------

    def _read_target_day(
        self,
        time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        path = DATA_FILES[TARGET_VARIABLE]
        ds = self._datasets[path]
        variable_name = self.variable_names[TARGET_VARIABLE]

        variable = ds.variables[variable_name]

        array = np.asarray(
            variable[time_index, :, :, :],
            dtype=np.float32,
        )

        valid_mask = np.isfinite(array)

        if self.normalize:

            normalized = np.empty_like(array)

            for depth_index, depth in enumerate(TARGET_DEPTHS):

                mean, std = self.target_stats[
                    float(depth)
                ]

                normalized[depth_index] = (
                    array[depth_index] - mean
                ) / std

            # Keep invalid target values as NaN.
            normalized = np.where(
                valid_mask,
                normalized,
                np.nan,
            ).astype(np.float32)

            return (
                normalized,
                valid_mask.astype(np.float32),
            )

        else:

            return (
                array,
                valid_mask.astype(np.float32),
            )

    # -----------------------------------------------------------------
    # DATASET LENGTH
    # -----------------------------------------------------------------

    def __len__(self) -> int:

        return len(self.sample_index)

    # -----------------------------------------------------------------
    # DATASET GETITEM
    # -----------------------------------------------------------------

    def __getitem__(self, index: int):

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset "
                f"of length {len(self)}."
            )

        # -------------------------------------------------------------
        # Decode filtered sample index.
        # -------------------------------------------------------------

        date_sample_index, tile_index = self.sample_index[index]

        target_time_index = self.sample_date_indices[
            date_sample_index
        ]

        tile_row, tile_col = self.tile_positions[tile_index]

        # -------------------------------------------------------------
        # Build retrospective input.
        #
        # Channel order:
        #
        # For each chronological day:
        #
        #     sst, sss, sla, uo, vo, u_wind, v_wind
        #
        # Total:
        #
        #     history_days × 7 channels
        # -------------------------------------------------------------

        input_channels = []
        input_masks = []

        window_start = (
            target_time_index - self.history_days + 1
        )

        for time_index in range(
            window_start,
            target_time_index + 1,
        ):

            for feature in INPUT_FEATURE_NAMES:

                array, mask = self._read_input_day(
                    feature,
                    time_index,
                )

                input_patch = array[
                    tile_row : tile_row + INPUT_TILE_SIZE,
                    tile_col : tile_col + INPUT_TILE_SIZE,
                ]

                mask_patch = mask[
                    tile_row : tile_row + INPUT_TILE_SIZE,
                    tile_col : tile_col + INPUT_TILE_SIZE,
                ]

                input_channels.append(input_patch)
                input_masks.append(mask_patch)

        x = np.stack(
            input_channels,
            axis=0,
        ).astype(np.float32)

        x_mask = np.stack(
            input_masks,
            axis=0,
        ).astype(np.float32)

        # -------------------------------------------------------------
        # Target.
        #
        # The target is the center 32×32 region of the 64×64 input.
        # -------------------------------------------------------------

        target, target_mask = self._read_target_day(
            target_time_index
        )

        target_row = tile_row + CONTEXT
        target_col = tile_col + CONTEXT

        y = target[
            :,
            target_row : target_row + OUTPUT_TILE_SIZE,
            target_col : target_col + OUTPUT_TILE_SIZE,
        ]

        y_mask = target_mask[
            :,
            target_row : target_row + OUTPUT_TILE_SIZE,
            target_col : target_col + OUTPUT_TILE_SIZE,
        ]

        # Safety assertion:
        # the sample index was constructed to guarantee at least one
        # valid target cell.
        if not bool(y_mask.any()):
            raise RuntimeError(
                "Dataset sample index inconsistency: a sample with "
                "zero valid target cells was selected."
            )

        # -------------------------------------------------------------
        # Convert to PyTorch tensors.
        # -------------------------------------------------------------

        x_tensor = torch.from_numpy(x)
        x_mask_tensor = torch.from_numpy(x_mask)

        y_tensor = torch.from_numpy(y)
        y_mask_tensor = torch.from_numpy(y_mask)

        if self.return_metadata:

            metadata = {
                "target_date": self.dates[target_time_index],
                "target_time_index": int(target_time_index),
                "tile_row": int(tile_row),
                "tile_col": int(tile_col),
                "input_tile_size": INPUT_TILE_SIZE,
                "output_tile_size": OUTPUT_TILE_SIZE,
                "window_size": self.history_days,
                "history_days": self.history_days,
                "input_features": list(
                    INPUT_FEATURE_NAMES
                ),
                "target_depths_m": TARGET_DEPTHS.tolist(),
                "valid_target_cells": int(
                    y_mask_tensor.sum().item()
                ),
            }

            return (
                x_tensor,
                x_mask_tensor,
                y_tensor,
                y_mask_tensor,
                metadata,
            )

        return (
            x_tensor,
            x_mask_tensor,
            y_tensor,
            y_mask_tensor,
        )

    # -----------------------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------------------

    def close(self):

        for ds in self._datasets.values():

            try:
                ds.close()
            except Exception:
                pass

        self._datasets.clear()

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------


def smoke_test(history_days: int = 7):

    print()
    print("=" * 72)
    print("OceanEmbed ML DATASET SMOKE TEST")
    print("=" * 72)

    for split in ["train", "validation", "test"]:

        print()
        print(f"Testing split: {split}")

        dataset = OceanEmbedDataset(
            split=split,
            tile_stride=32,
            return_metadata=True,
            normalize=True,
            history_days=history_days,
        )

        print(f"Dataset length: {len(dataset)}")

        (
            x,
            x_mask,
            y,
            y_mask,
            metadata,
        ) = dataset[0]

        print(f"x shape      : {tuple(x.shape)}")
        print(f"x mask shape : {tuple(x_mask.shape)}")
        print(f"y shape      : {tuple(y.shape)}")
        print(f"y mask shape : {tuple(y_mask.shape)}")

        print(
            f"x dtype      : {x.dtype}"
        )

        print(
            f"y dtype      : {y.dtype}"
        )

        print(
            f"x finite     : "
            f"{bool(torch.isfinite(x).all())}"
        )

        valid_y = y_mask > 0

        print(
            f"y has valid cells: "
            f"{bool(valid_y.any())}"
        )

        if valid_y.any():
            print(
                f"y valid finite: "
                f"{bool(torch.isfinite(y[valid_y]).all())}"
            )

        print(
            f"x range     : "
            f"{float(x.min()):.5f} to {float(x.max()):.5f}"
        )

        if valid_y.any():
            print(
                f"y valid range: "
                f"{float(y[valid_y].min()):.5f} to "
                f"{float(y[valid_y].max()):.5f}"
            )

        print(
            f"target date : "
            f"{metadata['target_date']}"
        )

        print(
            f"tile        : "
            f"row={metadata['tile_row']}, "
            f"col={metadata['tile_col']}"
        )

        print(
            f"valid target cells: "
            f"{metadata['valid_target_cells']}"
        )

        dataset.close()

    print()
    print("=" * 72)
    print("SMOKE TEST PASSED")
    print("=" * 72)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OceanEmbed ML dataset smoke test"
    )

    parser.add_argument(
        "--history-days",
        type=int,
        default=7,
        choices=range(1, 8),
        metavar="{1..7}",
        help="Number of retrospective input days (default: 7).",
    )

    args = parser.parse_args()

    smoke_test(history_days=args.history_days)