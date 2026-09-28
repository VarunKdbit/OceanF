from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from scripts.ingestion import live_window


class LiveDataResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.target_date = date(2026, 9, 20)
        cls.days = live_window.requested_dates(cls.target_date)
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.cache_root = Path(cls.tempdir.name) / "daily"
        cls.cache_patch = patch.object(
            live_window,
            "DAILY_CACHE_ROOT",
            cls.cache_root,
        )
        cls.cache_patch.start()
        cls.coordinate_patch = patch.multiple(
            live_window,
            EXPECTED_LATITUDE=np.array([5.0, 5.25]),
            EXPECTED_LONGITUDE=np.array([45.0, 45.25, 45.5]),
        )
        cls.coordinate_patch.start()
        cls.window_path = Path(cls.tempdir.name) / "window.nc"

        latitudes = live_window.EXPECTED_LATITUDE
        longitudes = live_window.EXPECTED_LONGITUDE
        values = np.ones(
            (len(cls.days), len(latitudes), len(longitudes)),
            dtype=np.float32,
        )
        dataset = xr.Dataset(
            {
                name: (
                    ("time", "latitude", "longitude"),
                    values.copy(),
                    {"units": "test-unit"},
                )
                for name in live_window.REQUIRED_VARIABLES
            },
            coords={
                "time": np.asarray(cls.days, dtype="datetime64[ns]"),
                "latitude": latitudes,
                "longitude": longitudes,
            },
            attrs={
                "target_date": cls.target_date.isoformat(),
                "history_days": len(cls.days),
            },
        )
        dataset.to_netcdf(cls.window_path)
        dataset.close()
        live_window.cache_window_days(cls.window_path, cls.target_date)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.coordinate_patch.stop()
        cls.cache_patch.stop()
        cls.tempdir.cleanup()

    def test_window_dates_end_on_the_requested_target_date(self) -> None:
        self.assertEqual(
            self.days,
            [date(2026, 9, 14) + timedelta(days=index) for index in range(7)],
        )
        self.assertEqual(self.days[-1], self.target_date)
        self.assertNotIn(self.target_date + timedelta(days=1), self.days)

    def test_missing_first_day_is_reported_until_that_day_is_cached(self) -> None:
        missing_path = live_window.daily_cache_path(self.days[0])
        missing_path.unlink()
        self.assertEqual(
            live_window.missing_cached_dates(self.target_date),
            [self.days[0]],
        )

        live_window.cache_window_days(
            self.window_path,
            self.target_date,
            only_dates={self.days[0]},
        )
        self.assertEqual(live_window.missing_cached_dates(self.target_date), [])

    def test_missing_target_day_is_reported(self) -> None:
        target_path = live_window.daily_cache_path(self.target_date)
        target_path.unlink()

        self.assertEqual(
            live_window.missing_cached_dates(self.target_date),
            [self.target_date],
        )

        live_window.cache_window_days(
            self.window_path,
            self.target_date,
            only_dates={self.target_date},
        )
        self.assertEqual(live_window.missing_cached_dates(self.target_date), [])

    def test_complete_cached_window_is_reused_without_rewriting_daily_files(self) -> None:
        first_path = live_window.daily_cache_path(self.days[0])
        first_write_time = first_path.stat().st_mtime_ns

        republished = live_window.cache_window_days(
            self.window_path,
            self.target_date,
        )

        self.assertEqual(republished, [])
        self.assertEqual(first_path.stat().st_mtime_ns, first_write_time)
        self.assertEqual(live_window.missing_cached_dates(self.target_date), [])


if __name__ == "__main__":
    unittest.main()
