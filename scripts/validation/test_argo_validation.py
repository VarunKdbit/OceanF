from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from run_argo_validation import (
    DOMAIN,
    ProfileCandidate,
    discover_candidates,
    interpolate_adjusted_profile,
    output_tile_for_grid_point,
)


class ArgoValidationTests(unittest.TestCase):
    def test_index_dates_are_filtered_without_temporal_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.txt"
            index_path.write_text(
                "file,date,latitude,longitude,parameters,parameter_data_mode\n"
                "dac/a/one.nc,20251201120000,15.0,70.0,\"PRES,TEMP\",\"DD\"\n"
                "dac/a/two.nc,20251202120000,15.0,70.0,\"PRES,TEMP\",\"DD\"\n"
                "dac/a/outside.nc,20251201120000,15.0,106.0,\"PRES,TEMP\",\"DD\"\n",
                encoding="utf-8",
            )
            candidates = discover_candidates(
                index_path,
                date(2025, 12, 1),
                date(2025, 12, 1),
                0.5,
            )
        self.assertEqual([candidate.file for candidate in candidates], ["dac/a/one.nc"])

    def test_existing_output_tile_geometry_is_not_extended(self) -> None:
        self.assertIsNotNone(output_tile_for_grid_point(40, 100, 101, 241))
        self.assertIsNone(output_tile_for_grid_point(5, 100, 101, 241))
        self.assertIsNone(output_tile_for_grid_point(40, 230, 101, 241))

    def test_interpolation_uses_adjusted_qc_and_does_not_extrapolate(self) -> None:
        pressure = np.asarray([[0, 10, 30, 100, 1000]], dtype=np.float32)
        dataset = xr.Dataset(
            {
                "TEMP_ADJUSTED": (("N_PROF", "N_LEVELS"), [[29, 28, 27, 20, 5]]),
                "PRES_ADJUSTED": (("N_PROF", "N_LEVELS"), pressure),
                "TEMP_ADJUSTED_QC": (("N_PROF", "N_LEVELS"), [list("11111")]),
                "PRES_ADJUSTED_QC": (("N_PROF", "N_LEVELS"), [list("11111")]),
            }
        )
        _, temperatures, valid_depth_count = interpolate_adjusted_profile(
            dataset, 0, 15.0
        )
        self.assertEqual(valid_depth_count, 5)
        self.assertTrue(np.isnan(temperatures[-1]))
        self.assertTrue(np.isfinite(temperatures[:5]).any())

    def test_non_one_qc_measurements_are_excluded(self) -> None:
        dataset = xr.Dataset(
            {
                "TEMP_ADJUSTED": (("N_PROF", "N_LEVELS"), [[29, 28, 27, 20, 5]]),
                "PRES_ADJUSTED": (("N_PROF", "N_LEVELS"), [[0, 10, 30, 100, 1000]]),
                "TEMP_ADJUSTED_QC": (("N_PROF", "N_LEVELS"), [list("11111")]),
                "PRES_ADJUSTED_QC": (("N_PROF", "N_LEVELS"), [list("11224")]),
            }
        )
        with self.assertRaisesRegex(LookupError, "insufficient_valid"):
            interpolate_adjusted_profile(dataset, 0, 15.0)


if __name__ == "__main__":
    unittest.main()
