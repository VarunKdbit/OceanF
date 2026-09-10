from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

SUPPORTED_DEPTHS = (
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000,
)


class SurfaceVariables(BaseModel):
    # Optional metadata only. Real model fields are always loaded from
    # harmonized spatial datasets inside the ML service.
    sst: Optional[float] = None
    sss: Optional[float] = None
    sla: Optional[float] = None
    uo: Optional[float] = None
    vo: Optional[float] = None
    u_wind: Optional[float] = None
    v_wind: Optional[float] = None


class PredictionRequest(BaseModel):
    """
    Final Spring -> FastAPI contract.

    The frontend sends only point/date/depth metadata. The ML service obtains
    the complete seven-day, seven-variable spatial window from the harmonized
    OceanEmbed datasets.
    """
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    date: date
    surface: Optional[SurfaceVariables] = None
    depths: List[int] = Field(
        default_factory=lambda: list(SUPPORTED_DEPTHS),
        description="OceanEmbed V1 trained output depths in meters",
    )
    model_version: Optional[str] = None

    @field_validator("depths")
    @classmethod
    def validate_depths(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("At least one depth must be requested")
        unsupported = sorted(set(value) - set(SUPPORTED_DEPTHS))
        if unsupported:
            raise ValueError(
                f"Unsupported depth(s): {unsupported}. "
                f"Supported depths: {list(SUPPORTED_DEPTHS)}"
            )
        return sorted(set(value))


class DepthPrediction(BaseModel):
    depth_m: int
    temperature_c: float
    uncertainty_c: Optional[float] = None


class PredictionResponse(BaseModel):
    latitude: float
    longitude: float
    date: date
    model_version: str
    grid_resolution_deg: float = 0.25
    predictions: List[DepthPrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
