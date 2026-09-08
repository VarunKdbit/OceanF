from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for the OceanEmbed ML service.
    Values can be overridden with OCEANEMBED_* environment variables.
    """
    model_version: str = "oceanembed-v1.0"
    model_weights_path: str = "weights/oceanembed.pt"

    # Standard depth levels (meters) the model was trained to predict,
    # matching the subsurface reconstruction papers (surface to ~1000m).
    depth_levels: List[int] = [0, 10, 20, 30, 50, 75, 100, 125, 150,
                                200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]

    grid_resolution_deg: float = 0.25
    max_batch_size: int = 64

    class Config:
        env_prefix = "OCEANEMBED_"


settings = Settings()
