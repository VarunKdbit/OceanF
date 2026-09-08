import logging

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.model import model
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oceanembed.api")

app = FastAPI(
    title="OceanEmbed ML Service",
    description="Serves subsurface ocean temperature predictions from the OceanEmbed model. "
                 "Called internally by the Spring Boot backend-api service.",
    version=settings.model_version,
)


@app.on_event("startup")
def startup_event() -> None:
    model.load()


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model.loaded else "degraded",
        model_loaded=model.loaded,
        model_version=model.model_version,
    )


@app.get("/model/info", tags=["ops"])
def model_info() -> dict:
    return {
        "model_version": model.model_version,
        "grid_resolution_deg": settings.grid_resolution_deg,
        "available_depth_levels_m": settings.depth_levels,
        "input_variables": ["sst", "sss", "ssh_sla", "wind_u", "wind_v"],
        "training_reference_data": "GLORYS reanalysis",
        "validation_reference_data": "ARGO float profiles (held out from training)",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    if not model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = model.predict(request)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictionResponse(
        latitude=request.latitude,
        longitude=request.longitude,
        date=request.date,
        model_version=request.model_version or model.model_version,
        grid_resolution_deg=settings.grid_resolution_deg,
        predictions=predictions,
    )
