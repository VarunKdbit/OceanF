import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from app.config import settings
from app.schemas import DepthPrediction, PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference.oceanembed_inference import DEPTHS_M, OceanEmbedEnsemble

logger = logging.getLogger("oceanembed.model")

FEATURE_NAMES = (
    "sst",
    "sss",
    "sla",
    "uo",
    "vo",
    "u_wind",
    "v_wind",
)

HISTORY_DAYS = 7
INPUT_SIZE = 64
INPUT_CHANNELS = HISTORY_DAYS * len(FEATURE_NAMES)
OUTPUT_CHANNELS = 15
OUTPUT_SIZE = 32


class OceanEmbedModel:
    """
    Wrapper around the OceanEmbed subsurface temperature reconstruction model.

    Reference approach (from the research papers this service is built against):
      - Subsurface Temperature Reconstruction for the Global Ocean from 1993-2020
        Using Satellite Observations and Deep Learning (MDPI Remote Sensing 14(13):3198)
      - A Deep Learning Method for Inversing 3D Temperature Fields Using Sea Surface
        Data in Offshore China and the Northwest Pacific Ocean (MDPI JMSE 12(12):2337)

        This wrapper owns service integration only. OceanEmbedEnsemble owns the
        model architecture, normalization, checkpoint loading, and inference.
    """

    def __init__(self):
        self.model_version = settings.model_version
        self.loaded = False
        self._ensemble: OceanEmbedEnsemble | None = None

    def load(self) -> None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._ensemble = OceanEmbedEnsemble(device=device)
            self.loaded = True
            logger.info("OceanEmbed ensemble '%s' loaded on %s", self.model_version, device)
        except Exception as exc:
            logger.exception("Failed to load OceanEmbed checkpoints: %s", exc)
            self._ensemble = None
            self.loaded = False

    def predict(self, request: PredictionRequest) -> List[DepthPrediction]:
        if not self.loaded or self._ensemble is None:
            raise RuntimeError("Model is not loaded")

        feature_values = {
            "sst": request.surface.sst,
            "sss": request.surface.sss,
            "sla": request.surface.ssh,
            "uo": 0.0,
            "vo": 0.0,
            "u_wind": request.surface.wind_u or 0.0,
            "v_wind": request.surface.wind_v or 0.0,
        }

        feature_arrays = {
            feature: np.full(
                (HISTORY_DAYS, INPUT_SIZE, INPUT_SIZE),
                value,
                dtype=np.float32,
            )
            for feature, value in feature_values.items()
        }

        prediction = self._ensemble.predict_from_raw_window(feature_arrays)
        prediction_array = prediction.detach().cpu().numpy()
        return [
            DepthPrediction(
                depth_m=depth,
                temperature_c=round(
                    float(prediction_array[depth_index, 16, 16]),
                    3,
                ),
            )
            for depth in request.depths
            for depth_index, trained_depth in enumerate(DEPTHS_M)
            if trained_depth == depth
        ]

    @torch.no_grad()
    def predict_input(self, model_input: torch.Tensor | np.ndarray) -> np.ndarray:
        """Run real inference for a normalized [49, 64, 64] input window."""
        if not self.loaded or self._ensemble is None:
            raise RuntimeError("Model is not loaded")

        tensor = torch.as_tensor(model_input, dtype=torch.float32)
        if tuple(tensor.shape) != (INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE):
            raise ValueError(
                f"Expected model input shape {(INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)}, "
                f"got {tuple(tensor.shape)}"
            )

        prediction = self._ensemble.predict_single(tensor)
        result = prediction.detach().cpu().numpy()
        if result.shape != (OUTPUT_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE):
            raise RuntimeError(f"Unexpected prediction shape: {result.shape}")
        return result


model = OceanEmbedModel()
