"""Kick prediction routes."""

from fastapi import APIRouter, Depends, HTTPException

from app.db.crud import (
    create_prediction_input,
    delete_prediction_input,
    get_prediction_input,
    list_prediction_inputs,
)
from app.db.database import SessionDep
from app.ml.model_manager import model_manager
from app.models.schemas import KickPredictionRequest, KickPredictionResponse
from app.security.auth import verify_api_key

router = APIRouter(tags=["predictions"])


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
    try:
        if not model_manager.initialized:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable.",
            )

        # Convert request to dictionary
        features = request.model_dump()

        # Make prediction
        prediction, confidence = model_manager.predict(features)

        # Save to database
        create_prediction_input(
            session=session,
            request=request,
            prediction=prediction,
            confidence=confidence,
        )

        return KickPredictionResponse(
            prediction=prediction,
            confidence=confidence,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.get("/predictions/{prediction_id}")
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
    return {
        "id": db_prediction.id,
        "time_norm": db_prediction.time_norm,
        "distance": db_prediction.distance,
        "angle": db_prediction.angle,
        "wind_speed": db_prediction.wind_speed,
        "precipitation_probability": db_prediction.precipitation_probability,
        "is_left_footed": db_prediction.is_left_footed,
        "game_away": db_prediction.game_away,
        "is_endgame": db_prediction.is_endgame,
        "is_start": db_prediction.is_start,
        "is_left_side": db_prediction.is_left_side,
        "has_previous_attempts": db_prediction.has_previous_attempts,
        "prediction": db_prediction.prediction,
        "confidence": db_prediction.confidence,
        "created_at": db_prediction.created_at,
    }


@router.get("/predictions")
async def list_predictions(
    session: SessionDep,
    skip: int = 0,
    limit: int = 100,
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
    return [
        {
            "id": p.id,
            "time_norm": p.time_norm,
            "distance": p.distance,
            "angle": p.angle,
            "wind_speed": p.wind_speed,
            "precipitation_probability": p.precipitation_probability,
            "is_left_footed": p.is_left_footed,
            "game_away": p.game_away,
            "is_endgame": p.is_endgame,
            "is_start": p.is_start,
            "is_left_side": p.is_left_side,
            "has_previous_attempts": p.has_previous_attempts,
            "prediction": p.prediction,
            "confidence": p.confidence,
            "created_at": p.created_at,
        }
        for p in predictions
    ]


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
