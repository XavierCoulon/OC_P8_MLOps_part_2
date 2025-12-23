"""CRUD operations for prediction inputs."""

from sqlalchemy.orm import Session

from app.db.models import PredictionInput
from app.models.schemas import KickPredictionRequest


def create_prediction_input(
    session: Session,
    request: KickPredictionRequest,
    prediction: float | None,
    confidence: float | None,
    latency_ms: float,
    cpu_usage_percent: float,
    memory_usage_mb: float,
    status_code: int,
    error_message: str | None,
) -> PredictionInput:
    """Create a new prediction input record in the database.

    Args:
        session: Database session
        request: Prediction request data
        prediction: Model prediction value
        confidence: Model confidence value
        latency_ms: Prediction latency in milliseconds
        cpu_usage_percent: CPU usage percentage at prediction time
        memory_usage_mb: Memory usage in MB at prediction time
        status_code: HTTP status code of the prediction request
        error_message: Error message if any

    Returns:
        Created PredictionInput record
    """

    db_prediction = PredictionInput(
        time_norm=request.time_norm,
        distance=request.distance,
        angle=request.angle,
        wind_speed=request.wind_speed,
        precipitation_probability=request.precipitation_probability,
        is_left_footed=request.is_left_footed,
        game_away=request.game_away,
        is_endgame=request.is_endgame,
        is_start=request.is_start,
        is_left_side=request.is_left_side,
        has_previous_attempts=request.has_previous_attempts,
        prediction=prediction,
        confidence=confidence,
        latency_ms=latency_ms,
        cpu_usage_percent=cpu_usage_percent,
        memory_usage_mb=memory_usage_mb,
        status_code=status_code,
        error_message=error_message,
    )
    session.add(db_prediction)
    session.commit()
    session.refresh(db_prediction)
    return db_prediction


def get_prediction_input(
    session: Session, prediction_id: int
) -> PredictionInput | None:
    """Get a prediction input by ID.

    Args:
        session: Database session
        prediction_id: ID of the prediction input

    Returns:
        PredictionInput record or None if not found
    """
    return (
        session.query(PredictionInput)
        .filter(PredictionInput.id == prediction_id)
        .first()
    )


def list_prediction_inputs(
    session: Session, skip: int = 0, limit: int = 100
) -> list[PredictionInput]:
    """List all prediction inputs with pagination.

    Args:
        session: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of PredictionInput records
    """
    return session.query(PredictionInput).offset(skip).limit(limit).all()


def delete_prediction_input(session: Session, prediction_id: int) -> bool:
    """Delete a prediction input by ID.

    Args:
        session: Database session
        prediction_id: ID of the prediction input

    Returns:
        True if deleted, False if not found
    """
    db_prediction = (
        session.query(PredictionInput)
        .filter(PredictionInput.id == prediction_id)
        .first()
    )

    if db_prediction is not None:
        session.delete(db_prediction)
        session.commit()
        return True
    return False
