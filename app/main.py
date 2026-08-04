"""FastAPI entry point for ECG classification inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.model_service import ModelService

DISCLAIMER = "Research demonstration only; not a medical device or clinical diagnosis."


class PredictionRequest(BaseModel):
    ecg_signal: Annotated[
        list[float],
        Field(
            min_length=1000,
            max_length=1000,
            description="Exactly 1,000 Lead-I ECG samples recorded at 100 Hz.",
        ),
    ]


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_version: str
    disclaimer: str = DISCLAIMER


def create_app(model_service: ModelService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = model_service or ModelService()
        service.load()
        app.state.model_service = service
        yield

    app = FastAPI(
        title="PTB-XL Lead-I ECG Classifier",
        version="1.0.0",
        description=(
            "Portfolio inference API for a three-class Lead-I ECG research model. " + DISCLAIMER
        ),
        lifespan=lifespan,
    )

    @app.get("/", tags=["service"])
    def root() -> dict[str, str]:
        return {
            "service": "PTB-XL Lead-I ECG Classifier",
            "docs": "/docs",
            "health": "/health",
            "readiness": "/ready",
        }

    @app.get("/health", tags=["service"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["service"])
    def ready(request: Request) -> dict[str, str]:
        if not request.app.state.model_service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready",
            )
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        try:
            result = request.app.state.model_service.predict(payload.ecg_signal)
            return PredictionResponse(**result, disclaimer=DISCLAIMER)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    return app


app = create_app()
