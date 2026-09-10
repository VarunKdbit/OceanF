import logging

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.model import model
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oceanembed.api")

app = FastAPI(
    title="OceanEmbed ML Service",
    description="Real OceanEmbed V1 E2 inference service.",
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
    return model.get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    if not model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Real OceanEmbed model/data are not loaded",
        )

    try:
        predictions = model.predict(request)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Invalid/unavailable OceanEmbed request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OceanEmbed inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"OceanEmbed inference error: {exc}",
        ) from exc

    return PredictionResponse(
        latitude=request.latitude,
        longitude=request.longitude,
        date=request.date,
        model_version=request.model_version or model.model_version,
        grid_resolution_deg=settings.grid_resolution_deg,
        predictions=predictions,
    )
