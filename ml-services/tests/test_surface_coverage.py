from datetime import date
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import app.model as model_module
from app.model import OceanEmbedModel
from app.data import LiveWindowUnavailableError
from app.schemas import PredictionRequest


class SurfaceCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = {
            name: np.ones((7, 64, 64), dtype=np.float32)
            for name in ("sst", "sss", "sla", "uo", "vo", "u_wind", "v_wind")
        }
        self.metadata = {
            "output_row": 2,
            "output_col": 3,
            "snapped_latitude": 15.0,
            "snapped_longitude": 70.0,
        }

    def test_reports_complete_finite_observations_at_model_output_point(self) -> None:
        result = OceanEmbedModel._coverage_for_window(
            self.features,
            self.metadata,
            15.0,
            70.0,
            date(2026, 9, 20),
        )

        self.assertTrue(result.ready)
        self.assertEqual(result.variablesReady, 7)
        self.assertEqual(result.missingVariables, [])

    def test_counts_only_finite_values_at_final_day_and_snapped_point(self) -> None:
        self.features["sst"][-1, 18, 19] = np.nan
        self.features["sss"][-2, 18, 19] = np.nan
        self.features["sla"][-1, 17, 19] = np.nan

        result = OceanEmbedModel._coverage_for_window(
            self.features,
            self.metadata,
            20.0,
            50.0,
            date(2026, 9, 20),
        )

        self.assertFalse(result.ready)
        self.assertEqual(result.variablesReady, 6)
        self.assertEqual(result.missingVariables, ["sst"])
        self.assertEqual(result.snappedLatitude, 15.0)
        self.assertEqual(result.snappedLongitude, 70.0)

    def test_prediction_rejects_incomplete_coverage_before_model_inference(self) -> None:
        self.features["sst"][-1, 18, 19] = np.nan
        model = OceanEmbedModel()
        model.loaded = True
        model.engine = Mock()
        request = PredictionRequest(
            latitude=20.0,
            longitude=50.0,
            date=date(2026, 9, 20),
        )

        with patch.object(
            model,
            "_get_input_window",
            return_value=(self.features, self.metadata),
        ):
            with self.assertRaisesRegex(ValueError, "Incomplete surface observations"):
                model.predict(request)

        model.engine.predict.assert_not_called()

    def test_unavailable_requested_window_reports_missing_dates(self) -> None:
        model = OceanEmbedModel()
        model.loaded = True
        target = date(2026, 9, 21)

        with patch.object(
            model,
            "_get_input_window",
            side_effect=LiveWindowUnavailableError(["2026-09-15", "2026-09-21"]),
        ):
            result = model.check_coverage(15.0, 70.0, target)

        self.assertFalse(result.ready)
        self.assertEqual(result.windowStart, date(2026, 9, 15))
        self.assertEqual(result.windowEnd, target)
        self.assertEqual(result.missingDates, ["2026-09-15", "2026-09-21"])
        self.assertEqual(result.variablesReady, 0)

    def test_simultaneous_same_date_preparation_runs_one_ingestion_job(self) -> None:
        model = OceanEmbedModel()
        target = date(2026, 9, 20)
        ingestion_started = Event()
        allow_ingestion_to_finish = Event()
        call_count = 0
        call_count_lock = Lock()

        def check_cache(*args, **kwargs):
            raise FileNotFoundError("Cache is not prepared yet")

        def run_ingestion(*args, **kwargs):
            nonlocal call_count
            with call_count_lock:
                call_count += 1
            ingestion_started.set()
            if not allow_ingestion_to_finish.wait(timeout=5):
                raise TimeoutError("Test did not release the ingestion job")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch("app.model.LiveOceanEmbedDataLoader", side_effect=check_cache),
            patch("app.model.subprocess.run", side_effect=run_ingestion),
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(model._prepare_live_window, target)
                self.assertTrue(ingestion_started.wait(timeout=5))
                second = executor.submit(model._prepare_live_window, target)
                deadline = time.monotonic() + 5
                while True:
                    with model_module._LIVE_PREPARATIONS_GUARD:
                        preparation = model_module._LIVE_PREPARATIONS.get(target)
                        waiter_count = preparation.waiters if preparation else 0
                    if waiter_count == 1:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("Second request did not join the in-flight preparation")
                    time.sleep(0.01)
                try:
                    self.assertEqual(call_count, 1)
                finally:
                    allow_ingestion_to_finish.set()

                first.result(timeout=5)
                second.result(timeout=5)
        self.assertEqual(call_count, 1)
        self.assertEqual(call_count, 1)


if __name__ == "__main__":
    unittest.main()
