import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from app.config import settings
from app.data import OceanEmbedDataLoader
from app.schemas import DepthPrediction, PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.inference.oceanembed_inference import (
    DEPTHS_M,
    HISTORY_DAYS,
    INPUT_CHANNELS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_CHANNELS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    INPUT_FEATURES,
    OceanEmbedEnsemble,
)

logger = logging.getLogger("oceanembed.model")