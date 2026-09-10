from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_version: str = "oceanembed-v1.0"
    grid_resolution_deg: float = 0.25
    depth_levels: List[int] = [
        0, 5, 10, 20, 30, 50, 75, 100,
        125, 150, 200, 300, 500, 700, 1000,
    ]

    class Config:
        env_prefix = "OCEANEMBED_"


settings = Settings()
