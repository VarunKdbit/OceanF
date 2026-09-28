import logging
from datetime import date as date_type

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.model import SURFACE_FEATURES, model
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    SurfaceCoverageResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oceanembed.api")

# Historical coverage begins here; live dates may extend through today.
SUPPORTED_APPLICATION_START = date_type(2025, 7, 1)

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


@app.get(
    "/coverage",
    response_model=SurfaceCoverageResponse,
    tags=["inference"],
)
def surface_coverage(
    latitude: float,
    longitude: float,
    date: date_type,
) -> SurfaceCoverageResponse:
    logger.info(
        "Coverage request received: latitude=%s longitude=%s date=%s",
        latitude,
        longitude,
        date.isoformat(),
    )

    today = date_type.today()
    if date < SUPPORTED_APPLICATION_START or date > today:
        logger.info(
            "Coverage request rejected by date validation: latitude=%s "
            "longitude=%s date=%s (supported %s through %s)",
            latitude,
            longitude,
            date.isoformat(),
            SUPPORTED_APPLICATION_START.isoformat(),
            today.isoformat(),
        )
        return SurfaceCoverageResponse(
            ready=False,
            latitude=latitude,
            longitude=longitude,
            snappedLatitude=None,
            snappedLongitude=None,
            date=date,
            targetDate=date,
            windowStart=date,
            windowEnd=date,
            availableDates=[],
            missingDates=[date.isoformat()],
            requiredVariables=len(SURFACE_FEATURES),
            variablesReady=0,
            readyVariables=[],
            missingVariables=list(SURFACE_FEATURES),
            message=(
                f"Requested date {date.isoformat()} is outside the currently "
                f"supported/available date range "
                f"({SUPPORTED_APPLICATION_START.isoformat()} through "
                f"{today.isoformat()})."
            ),
        )

    if not model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Real OceanEmbed model/data are not loaded",
        )
    try:
        logger.info(
            "Coverage request passed validation; calling "
            "model.check_coverage() for date=%s",
            date.isoformat(),
        )
        return model.check_coverage(latitude, longitude, date)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        status_code = 422 if "Requested grid point could not be represented" in message else 500
        raise HTTPException(status_code=status_code, detail=message) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Real OceanEmbed model/data are not loaded",
        )

    try:
        # model.predict() already returns the complete
        # PredictionResponse object.
        return model.predict(request)

    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "Invalid/unavailable OceanEmbed request: %s",
            exc,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        message = str(exc)

        # A complete 64x64 input tile is required to produce
        # the centered 32x32 prediction region. Coordinates
        # without sufficient spatial context are therefore
        # valid geographic coordinates but unsupported for
        # OceanEmbed inference.
        if (
            "Requested grid point could not be represented"
            " inside the 32x32 prediction region"
            in message
        ):
            logger.warning(
                "Unsupported OceanEmbed inference location: %s",
                message,
            )

            raise HTTPException(
                status_code=422,
                detail=message,
            ) from exc

        # Other RuntimeErrors are genuine inference/service
        # failures and must remain HTTP 500.
        logger.exception(
            "OceanEmbed inference failed with RuntimeError"
        )

        raise HTTPException(
            status_code=500,
            detail=f"OceanEmbed inference error: {message}",
        ) from exc

    except Exception as exc:
        logger.exception(
            "OceanEmbed inference failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"OceanEmbed inference error: {exc}",
        ) from exc