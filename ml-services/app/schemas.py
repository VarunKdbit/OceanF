from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SurfaceVariables(BaseModel):
    """
    Satellite-observed surface inputs used by the OceanEmbed model,
    per the reconstruction papers (SST, SSS, SSH/SLA, surface wind).
    """
    sst: float = Field(..., description="Sea Surface Temperature (degC)")
    sss: float = Field(..., description="Sea Surface Salinity (PSU)")
    ssh: float = Field(..., description="Sea Surface Height / Sea Level Anomaly (m)")
    wind_u: Optional[float] = Field(None, description="Zonal (east-west) wind component (m/s)")
    wind_v: Optional[float] = Field(None, description="Meridional (north-south) wind component (m/s)")


class PredictionRequest(BaseModel):
    """
    Single-point inference request. lat/lon are snapped conceptually to the
    model's 0.25 deg x 0.25 deg grid; date selects the daily surface fields.
    """
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    date: date
    surface: SurfaceVariables
    depths: List[int] = Field(
        default_factory=lambda: [0, 10, 20, 50, 100, 200, 500],
        description="Requested depth levels in meters"
    )
    model_version: Optional[str] = Field(None, description="Pin a specific model version; defaults to latest")

    @field_validator("depths")
    @classmethod
    def validate_depths(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("At least one depth must be requested")
        if any(d < 0 or d > 2000 for d in v):
            raise ValueError("Depth must be between 0 and 2000 meters")
        return sorted(set(v))


class DepthPrediction(BaseModel):
    depth_m: int
    temperature_c: float
    uncertainty_c: Optional[float] = Field(None, description="Estimated 1-sigma uncertainty, if available")


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
