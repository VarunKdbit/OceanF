from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Dict

import numpy as np
from netCDF4 import Dataset, num2date


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "data" / "processed" / "ML"
HARMONIZED_DIR = ML_DIR / "harmonized"
LIVE_DIR = PROJECT_ROOT / "data" / "processed" / "live"
LIVE_DAILY_DIR = LIVE_DIR / "daily"
CONFIG_PATH = ML_DIR / "ml_config.json"


FEATURE_FILES = {
    "sst": "SST_harmonized.nc",
    "sss": "SSS_harmonized.nc",
    "sla": "SLA_harmonized.nc",
    "uo": "Currents_harmonized.nc",
    "vo": "Currents_harmonized.nc",
    "u_wind": "Winds_harmonized.nc",
    "v_wind": "Winds_harmonized.nc",
}


FEATURE_VARIABLES = {
    "sst": ["sst", "analysed_sst", "sea_surface_temperature", "temperature"],
    "sss": ["sss", "so", "salinity", "sea_surface_salinity"],
    "sla": [
        "sla",
        "adt",
        "ssh",
        "sea_level_anomaly",
        "sea_surface_height_anomaly",
    ],
    "uo": ["uo", "u", "u_current", "eastward_current"],
    "vo": ["vo", "v", "v_current", "northward_current"],
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
}


FEATURES = tuple(FEATURE_FILES)

HISTORY_DAYS = 7
INPUT_SIZE = 64
OUTPUT_SIZE = 32
RESOLUTION = 0.25

LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0

LAT_SIZE = 101
LON_SIZE = 241


class OceanEmbedDataLoader:
    """
    Loads the real historical harmonized OceanEmbed seven-day
    spatial input window.

    This loader is intentionally kept compatible with the original
    V1 historical ML pipeline.
    """

    def __init__(self) -> None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Required ML config not found: {CONFIG_PATH}"
            )

        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            self.config = json.load(fh)

        self._datasets: Dict[Path, Dataset] = {}
        self.variable_names: dict[str, str] = {}
        self.dates: list[str] = []
        self.latitudes: np.ndarray | None = None
        self.longitudes: np.ndarray | None = None

        try:
            self._open_and_validate()
        except Exception:
            self.close()
            raise

    def _open_and_validate(self) -> None:
        for feature, filename in FEATURE_FILES.items():
            path = HARMONIZED_DIR / filename

            if not path.exists():
                raise FileNotFoundError(
                    f"Required harmonized dataset not found: {path}"
                )

            if path not in self._datasets:
                self._datasets[path] = Dataset(path, "r")

        reference_lat = None
        reference_lon = None
        reference_dates = None

        for feature in FEATURES:
            ds = self._datasets[
                HARMONIZED_DIR / FEATURE_FILES[feature]
            ]

            self.variable_names[feature] = self._find_variable(
                ds,
                feature,
            )

            lat = self._coordinate(
                ds,
                ("latitude", "lat"),
            ).astype(np.float64)

            lon = self._coordinate(
                ds,
                ("longitude", "lon"),
            ).astype(np.float64)

            dates = self._dates_from_dataset(ds)

            expected_lat = np.arange(
                LAT_MIN,
                LAT_MAX + RESOLUTION / 2,
                RESOLUTION,
            )

            expected_lon = np.arange(
                LON_MIN,
                LON_MAX + RESOLUTION / 2,
                RESOLUTION,
            )

            if (
                len(lat) != LAT_SIZE
                or not np.allclose(
                    lat,
                    expected_lat,
                    atol=1e-5,
                )
            ):
                raise ValueError(
                    f"{feature}: latitude grid does not match "
                    "5..30N at 0.25 degrees"
                )

            if (
                len(lon) != LON_SIZE
                or not np.allclose(
                    lon,
                    expected_lon,
                    atol=1e-5,
                )
            ):
                raise ValueError(
                    f"{feature}: longitude grid does not match "
                    "45..105E at 0.25 degrees"
                )

            if reference_lat is None:
                reference_lat = lat
                reference_lon = lon
                reference_dates = dates
            else:
                if not np.allclose(
                    lat,
                    reference_lat,
                    atol=1e-5,
                ):
                    raise ValueError(
                        f"{feature}: latitude grid differs "
                        "from reference"
                    )

                if not np.allclose(
                    lon,
                    reference_lon,
                    atol=1e-5,
                ):
                    raise ValueError(
                        f"{feature}: longitude grid differs "
                        "from reference"
                    )

                if dates != reference_dates:
                    raise ValueError(
                        f"{feature}: time axis differs "
                        "from reference"
                    )

            variable = ds.variables[
                self.variable_names[feature]
            ]

            if len(variable.dimensions) != 3:
                raise ValueError(
                    f"{feature}: expected [time, lat, lon], "
                    f"got {variable.dimensions}"
                )

        self.latitudes = reference_lat
        self.longitudes = reference_lon
        self.dates = reference_dates

        expected_start = self.config["time"]["common_start"]
        expected_end = self.config["time"]["common_end"]

        if (
            not self.dates
            or self.dates[0] != expected_start
            or self.dates[-1] != expected_end
        ):
            raise ValueError(
                f"Unexpected harmonized time range: "
                f"{self.dates[0]}..{self.dates[-1]}"
            )

    @staticmethod
    def _find_variable(
        ds: Dataset,
        feature: str,
    ) -> str:
        for candidate in FEATURE_VARIABLES[feature]:
            if candidate in ds.variables:
                return candidate

        coordinate_names = {
            "time",
            "valid_time",
            "date",
            "latitude",
            "longitude",
            "lat",
            "lon",
            "depth",
        }

        data_variables = [
            name
            for name in ds.variables
            if name not in coordinate_names
        ]

        if len(data_variables) == 1:
            return data_variables[0]

        raise KeyError(
            f"Could not identify variable for {feature}; "
            f"variables={list(ds.variables)}"
        )

    @staticmethod
    def _coordinate(
        ds: Dataset,
        names: tuple[str, ...],
    ) -> np.ndarray:
        for name in names:
            if name in ds.variables:
                return np.asarray(
                    ds.variables[name][:]
                )

        raise KeyError(
            f"Missing coordinate; tried {names}"
        )

    @staticmethod
    def _dates_from_dataset(
        ds: Dataset,
    ) -> list[str]:
        name = (
            "time"
            if "time" in ds.variables
            else "valid_time"
        )

        if name not in ds.variables:
            raise KeyError(
                "Missing time/valid_time coordinate"
            )

        variable = ds.variables[name]
        values = variable[:]

        units = getattr(
            variable,
            "units",
            None,
        )

        calendar = getattr(
            variable,
            "calendar",
            "standard",
        )

        if units:
            converted = num2date(
                values,
                units=units,
                calendar=calendar,
            )

            return [
                f"{v.year:04d}-{v.month:02d}-{v.day:02d}"
                for v in converted
            ]

        return [
            str(v)[:10]
            for v in values
        ]

    @staticmethod
    def _masked_to_float32(
        values,
    ) -> np.ndarray:
        if np.ma.isMaskedArray(values):
            values = values.filled(np.nan)

        return np.asarray(
            values,
            dtype=np.float32,
        )

    @staticmethod
    def _tile_start(
        index: int,
        max_start: int,
    ) -> int:
        """
        Same 64->32 centered target convention as the
        existing OceanEmbed dataset.

        Target region:
            [tile+16 : tile+48]
        """

        return max(
            0,
            min(
                index - 16,
                max_start,
            ),
        )

    def get_window(
        self,
        target_date: date,
        latitude: float,
        longitude: float,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, int | float | str],
    ]:

        if not (
            LAT_MIN
            <= latitude
            <= LAT_MAX
        ):
            raise ValueError(
                f"Latitude must be within "
                f"{LAT_MIN}..{LAT_MAX}"
            )

        if not (
            LON_MIN
            <= longitude
            <= LON_MAX
        ):
            raise ValueError(
                f"Longitude must be within "
                f"{LON_MIN}..{LON_MAX}"
            )

        target = target_date.isoformat()

        if target not in self.dates:
            raise ValueError(
                f"Requested date {target} is unavailable. "
                f"Available range is "
                f"{self.dates[0]}..{self.dates[-1]}."
            )

        target_index = self.dates.index(target)

        window_start = (
            target_index
            - HISTORY_DAYS
            + 1
        )

        if window_start < 0:
            raise ValueError(
                f"Requested date {target} does not have "
                f"{HISTORY_DAYS} retrospective days."
            )

        assert self.latitudes is not None
        assert self.longitudes is not None

        lat_index = int(
            np.abs(
                self.latitudes - latitude
            ).argmin()
        )

        lon_index = int(
            np.abs(
                self.longitudes - longitude
            ).argmin()
        )

        snapped_lat = float(
            self.latitudes[lat_index]
        )

        snapped_lon = float(
            self.longitudes[lon_index]
        )

        tile_row = self._tile_start(
            lat_index,
            LAT_SIZE - INPUT_SIZE,
        )

        tile_col = self._tile_start(
            lon_index,
            LON_SIZE - INPUT_SIZE,
        )

        output_row = (
            lat_index
            - (tile_row + 16)
        )

        output_col = (
            lon_index
            - (tile_col + 16)
        )

        if not (
            0 <= output_row < OUTPUT_SIZE
            and
            0 <= output_col < OUTPUT_SIZE
        ):
            raise RuntimeError(
                "Requested grid point could not be represented "
                "inside the 32x32 prediction region of the "
                "selected 64x64 tile."
            )

        feature_arrays: dict[
            str,
            np.ndarray,
        ] = {}

        for feature in FEATURES:
            ds = self._datasets[
                HARMONIZED_DIR
                / FEATURE_FILES[feature]
            ]

            variable = ds.variables[
                self.variable_names[feature]
            ]

            values = self._masked_to_float32(
                variable[
                    window_start:target_index + 1,
                    tile_row:tile_row + INPUT_SIZE,
                    tile_col:tile_col + INPUT_SIZE,
                ]
            )

            expected = (
                HISTORY_DAYS,
                INPUT_SIZE,
                INPUT_SIZE,
            )

            if values.shape != expected:
                raise RuntimeError(
                    f"{feature} window has shape "
                    f"{values.shape}; expected "
                    f"{expected}"
                )

            feature_arrays[feature] = values

        metadata = {
            "target_date": target,
            "window_start": self.dates[window_start],
            "window_end": target,
            "lat_index": lat_index,
            "lon_index": lon_index,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "output_row": output_row,
            "output_col": output_col,
            "snapped_latitude": snapped_lat,
            "snapped_longitude": snapped_lon,
        }

        return feature_arrays, metadata

    def close(self) -> None:
        for ds in self._datasets.values():
            try:
                ds.close()
            except Exception:
                pass

        self._datasets.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class LiveWindowUnavailableError(FileNotFoundError):
    def __init__(self, missing_dates: list[str]) -> None:
        self.missing_dates = missing_dates
        super().__init__(
            "Requested live input window is missing daily data for: "
            + ", ".join(missing_dates)
        )


class LiveOceanEmbedDataLoader:
    """
    Loads a live OceanEmbed input window from either a legacy
    seven-day file or the validated daily cache.

    Legacy file:

        data/processed/live/<date>/
            oceanembed_live_<date>.nc

    Daily cache:

        data/processed/live/daily/oceanembed_live_<date>.nc

    Expected variables:

        sst
        sss
        sla
        uo
        vo
        u_wind
        v_wind

    Expected dimensions:

        time    = 7
        latitude  = 101
        longitude = 241
    """

    def __init__(
        self,
        target_date: date,
    ) -> None:

        self.target_date = target_date.isoformat()

        self.path = (
            LIVE_DIR
            / self.target_date
            / (
                f"oceanembed_live_"
                f"{self.target_date}.nc"
            )
        )

        self.ds: Dataset | None = None
        self.daily_datasets: list[Dataset] = []

        self.variable_names: dict[
            str,
            str,
        ] = {}

        self.dates: list[str] = []

        self.latitudes: np.ndarray | None = None
        self.longitudes: np.ndarray | None = None

        try:
            target = date.fromisoformat(self.target_date)
            from datetime import timedelta

            daily_paths = [
                LIVE_DAILY_DIR
                / f"oceanembed_live_{(target - timedelta(days=HISTORY_DAYS - 1 - index)).isoformat()}.nc"
                for index in range(HISTORY_DAYS)
            ]
            if all(path.is_file() for path in daily_paths):
                try:
                    self._open_daily_cache()
                except (KeyError, OSError, ValueError):
                    self.close()
                    self.latitudes = None
                    self.longitudes = None
                    self.variable_names = {}
                    self.dates = []
                    if not self.path.is_file():
                        raise
                    self.ds = Dataset(self.path, "r")
                    self._validate_legacy()
            elif self.path.is_file():
                self.ds = Dataset(self.path, "r")
                try:
                    self._validate_legacy()
                except (KeyError, OSError, ValueError):
                    self.ds.close()
                    self.ds = None
                    self.dates = []
                    self.latitudes = None
                    self.longitudes = None
                    self.variable_names = {}
                    self._open_daily_cache()
            else:
                self._open_daily_cache()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _expected_coordinates() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.arange(LAT_MIN, LAT_MAX + RESOLUTION / 2, RESOLUTION),
            np.arange(LON_MIN, LON_MAX + RESOLUTION / 2, RESOLUTION),
        )

    def _validate_coordinates(self, dataset: Dataset) -> None:
        if "latitude" not in dataset.variables or "longitude" not in dataset.variables:
            raise KeyError("Live dataset is missing latitude/longitude coordinates.")

        latitudes = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        longitudes = np.asarray(dataset.variables["longitude"][:], dtype=np.float64)
        expected_lat, expected_lon = self._expected_coordinates()
        if len(latitudes) != LAT_SIZE or not np.allclose(
            latitudes, expected_lat, atol=1e-5
        ):
            raise ValueError("Live dataset latitude grid does not match 5..30N at 0.25 degrees.")
        if len(longitudes) != LON_SIZE or not np.allclose(
            longitudes, expected_lon, atol=1e-5
        ):
            raise ValueError("Live dataset longitude grid does not match 45..105E at 0.25 degrees.")
        if self.latitudes is None:
            self.latitudes = latitudes
            self.longitudes = longitudes
        elif not np.array_equal(self.latitudes, latitudes) or not np.array_equal(
            self.longitudes, longitudes
        ):
            raise ValueError("Daily live files do not share the same model grid.")

    def _validate_variables(
        self,
        dataset: Dataset,
        expected_days: int,
    ) -> list[str]:
        required_variables = [
            "sst",
            "sss",
            "sla",
            "uo",
            "vo",
            "u_wind",
            "v_wind",
        ]

        for feature in required_variables:
            if feature not in dataset.variables:
                raise KeyError(
                    f"Live dataset is missing required "
                    f"variable: {feature}"
                )

            self.variable_names[
                feature
            ] = feature

            if tuple(dataset.variables[feature].shape) != (
                expected_days,
                LAT_SIZE,
                LON_SIZE,
            ):
                raise ValueError(
                    f"{feature}: unexpected live dataset shape "
                    f"{tuple(dataset.variables[feature].shape)}"
                )

        if "time" not in dataset.variables:
            raise KeyError(
                "Live dataset is missing time coordinate."
            )

        dates = OceanEmbedDataLoader._dates_from_dataset(dataset)
        if len(dates) != expected_days:
            raise ValueError(
                f"Live dataset must contain exactly {expected_days} days; "
                f"got {len(dates)}."
            )
        self._validate_coordinates(dataset)
        return dates

    def _validate_legacy(self) -> None:
        assert self.ds is not None
        self.dates = self._validate_variables(self.ds, HISTORY_DAYS)
        if self.dates[-1] != self.target_date:
            raise ValueError(
                f"Live dataset final date is "
                f"{self.dates[-1]}, expected "
                f"{self.target_date}."
            )

    def _open_daily_cache(self) -> None:
        from datetime import timedelta

        target = date.fromisoformat(self.target_date)
        expected_days = [
            target - timedelta(days=HISTORY_DAYS - 1 - index)
            for index in range(HISTORY_DAYS)
        ]
        missing_paths = [
            LIVE_DAILY_DIR / f"oceanembed_live_{day.isoformat()}.nc"
            for day in expected_days
            if not (LIVE_DAILY_DIR / f"oceanembed_live_{day.isoformat()}.nc").is_file()
        ]
        if missing_paths:
            missing_dates = [
                path.stem.removeprefix("oceanembed_live_")
                for path in missing_paths
            ]
            raise LiveWindowUnavailableError(missing_dates)

        dates: list[str] = []
        for day in expected_days:
            path = LIVE_DAILY_DIR / f"oceanembed_live_{day.isoformat()}.nc"
            try:
                dataset = Dataset(path, "r")
                self.daily_datasets.append(dataset)
                dataset_dates = self._validate_variables(dataset, 1)
                if dataset_dates != [day.isoformat()]:
                    raise ValueError(
                        f"Daily live dataset {path} contains {dataset_dates}, "
                        f"expected {day.isoformat()}."
                    )
                self._validate_coordinates(dataset)
                dates.extend(dataset_dates)
            except (KeyError, OSError, ValueError) as exc:
                raise LiveWindowUnavailableError([day.isoformat()]) from exc
        self.dates = dates
        if self.dates != [day.isoformat() for day in expected_days]:
            raise ValueError(
                "Daily live cache does not exactly match the requested input window."
            )
        self.variable_names = {feature: feature for feature in FEATURES}

    def get_window(
        self,
        latitude: float,
        longitude: float,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, int | float | str],
    ]:

        if not (
            LAT_MIN
            <= latitude
            <= LAT_MAX
        ):
            raise ValueError(
                f"Latitude must be within "
                f"{LAT_MIN}..{LAT_MAX}"
            )

        if not (
            LON_MIN
            <= longitude
            <= LON_MAX
        ):
            raise ValueError(
                f"Longitude must be within "
                f"{LON_MIN}..{LON_MAX}"
            )

        assert self.latitudes is not None
        assert self.longitudes is not None

        lat_index = int(
            np.abs(
                self.latitudes - latitude
            ).argmin()
        )

        lon_index = int(
            np.abs(
                self.longitudes - longitude
            ).argmin()
        )

        snapped_lat = float(
            self.latitudes[lat_index]
        )

        snapped_lon = float(
            self.longitudes[lon_index]
        )

        tile_row = (
            OceanEmbedDataLoader
            ._tile_start(
                lat_index,
                LAT_SIZE - INPUT_SIZE,
            )
        )

        tile_col = (
            OceanEmbedDataLoader
            ._tile_start(
                lon_index,
                LON_SIZE - INPUT_SIZE,
            )
        )

        output_row = (
            lat_index
            - (tile_row + 16)
        )

        output_col = (
            lon_index
            - (tile_col + 16)
        )

        if not (
            0 <= output_row < OUTPUT_SIZE
            and
            0 <= output_col < OUTPUT_SIZE
        ):
            raise RuntimeError(
                "Requested grid point could not be represented "
                "inside the 32x32 prediction region of the "
                "selected 64x64 tile."
            )

        feature_arrays: dict[
            str,
            np.ndarray,
        ] = {}

        for feature in FEATURES:
            if self.ds is not None:
                variable = self.ds.variables[self.variable_names[feature]]
                values = variable[
                    :,
                    tile_row:tile_row + INPUT_SIZE,
                    tile_col:tile_col + INPUT_SIZE,
                ]
                values = OceanEmbedDataLoader._masked_to_float32(values)
            else:
                daily_values = []
                for dataset in self.daily_datasets:
                    values_for_day = dataset.variables[feature][
                        :,
                        tile_row:tile_row + INPUT_SIZE,
                        tile_col:tile_col + INPUT_SIZE,
                    ]
                    daily_values.append(
                        OceanEmbedDataLoader._masked_to_float32(values_for_day)[0]
                    )
                values = np.stack(daily_values, axis=0)

            expected = (
                HISTORY_DAYS,
                INPUT_SIZE,
                INPUT_SIZE,
            )

            if values.shape != expected:
                raise RuntimeError(
                    f"{feature} live window has shape "
                    f"{values.shape}; expected "
                    f"{expected}"
                )

            feature_arrays[feature] = values

        metadata = {
            "target_date": self.target_date,
            "window_start": self.dates[0],
            "window_end": self.dates[-1],
            "lat_index": lat_index,
            "lon_index": lon_index,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "output_row": output_row,
            "output_col": output_col,
            "snapped_latitude": snapped_lat,
            "snapped_longitude": snapped_lon,
            "source": "live",
            "live_file": str(self.path) if self.ds is not None else str(LIVE_DAILY_DIR),
        }

        return feature_arrays, metadata

    def close(self) -> None:
        try:
            if self.ds is not None:
                self.ds.close()
        except Exception:
            pass
        for dataset in self.daily_datasets:
            try:
                dataset.close()
            except Exception:
                pass
        self.daily_datasets.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass