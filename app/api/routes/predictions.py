"""Kick prediction routes."""

import os
import time

import psutil
from fastapi import APIRouter, Depends, HTTPException

from app.db.crud import (
    create_prediction_input,
    delete_prediction_input,
    get_prediction_input,
    list_prediction_inputs,
)
from app.db.database import SessionDep
from app.ml.model_manager import model_manager
from app.models.schemas import (
    KickPredictionRequest,
    KickPredictionResponse,
    PredictionInputResponse,
)
from app.security.auth import verify_api_key

router = APIRouter(tags=["predictions"])

process = psutil.Process(os.getpid())


@router.post("/predict", response_model=KickPredictionResponse)
async def predict_kick(
    request: KickPredictionRequest,
    session: SessionDep,
    _: str = Depends(verify_api_key),
):
    """Predict rugby kick success probability.

    Args:
        request: Kick features for prediction
        session: Database session

    Returns:
        Prediction probability and confidence score
    """

    start_time = time.time()
    process.cpu_percent(interval=None)
    prediction = None
    confidence = None
    status_code = 200
    error_message = None

    try:
        if not model_manager.initialized:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable.",
            )

        # Convert request to dictionary
        features = request.model_dump()

        prediction, confidence = model_manager.predict(features)

        return KickPredictionResponse(
            prediction=prediction,
            confidence=confidence,
        )
    except HTTPException as he:
        status_code = he.status_code
        error_message = he.detail
        raise he
    except Exception as e:
        status_code = 500
        error_message = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
    finally:
        latency_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        cpu_usage = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

        # Save to database
        create_prediction_input(
            session=session,
            request=request,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_mb,
            status_code=status_code,
            error_message=error_message,
        )


@router.get("/predictions/{prediction_id}", response_model=PredictionInputResponse)
async def get_prediction(
    prediction_id: int,
    session: SessionDep,
):
    """Get a specific prediction record.

    Args:
        prediction_id: ID of the prediction
        session: Database session

    Returns:
        Prediction record or 404 if not found
    """
    db_prediction = get_prediction_input(session, prediction_id)
    if not db_prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return db_prediction


@router.get("/predictions", response_model=list[PredictionInputResponse])
async def list_predictions(
    session: SessionDep,
    skip: int = 0,
    limit: int = 1000,
):
    """List all prediction records.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        session: Database session

    Returns:
        List of prediction records
    """
    predictions = list_prediction_inputs(session, skip=skip, limit=limit)
    return predictions


@router.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: int,
    session: SessionDep,
):
    """Delete a prediction record.

    Args:
        prediction_id: ID of the prediction to delete
        session: Database session

    Returns:
        Success message or 404 if not found
    """
    if not delete_prediction_input(session, prediction_id):
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"message": "Prediction deleted successfully"}
