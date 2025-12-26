"""Tests for CRUD operations."""

import pytest
from sqlalchemy.orm import Session

from app.db.crud import (
    create_prediction_input,
    delete_prediction_input,
    get_prediction_input,
    list_prediction_inputs,
)
from app.models.schemas import KickPredictionRequest


@pytest.fixture
def sample_kick_request():
    """Sample kick prediction request."""
    return KickPredictionRequest(
        time_norm=0.5,
        distance=35,
        angle=0,
        wind_speed=5.0,
        precipitation_probability=0.0,
        is_left_footed=0,
        game_away=0,
        is_endgame=0,
        is_start=0,
        is_left_side=0,
        has_previous_attempts=0,
    )


class TestCreatePredictionInput:
    """Test suite for create_prediction_input function."""

    def test_create_prediction_input_success(
        self, test_db: Session, sample_kick_request
    ):
        """Test creating a prediction input record."""
        result = create_prediction_input(
            session=test_db,
            request=sample_kick_request,
            prediction=0.75,
            confidence=0.85,
            latency_ms=50.5,
            cpu_usage_percent=25.3,
            memory_usage_mb=150.2,
            status_code=200,
            error_message=None,
        )
        test_db.flush()
        test_db.refresh(result)

        assert result.id is not None
        assert result.distance == 35.0
        assert result.prediction == 0.75
        assert result.confidence == 0.85
        assert result.latency_ms == 50.5
        assert result.status_code == 200
        assert result.error_message is None

    def test_create_prediction_input_with_error(
        self, test_db: Session, sample_kick_request
    ):
        """Test creating a prediction input with error."""
        result = create_prediction_input(
            session=test_db,
            request=sample_kick_request,
            prediction=None,
            confidence=None,
            latency_ms=10.0,
            cpu_usage_percent=15.0,
            memory_usage_mb=100.0,
            status_code=500,
            error_message="Model prediction failed",
        )

        assert result.id is not None
        assert result.prediction is None
        assert result.confidence is None
        assert result.status_code == 500
        assert result.error_message == "Model prediction failed"


class TestGetPredictionInput:
    """Test suite for get_prediction_input function."""

    def test_get_existing_prediction(self, test_db: Session, sample_kick_request):
        """Test getting an existing prediction input."""
        # Create a prediction first
        created = create_prediction_input(
            session=test_db,
            request=sample_kick_request,
            prediction=0.65,
            confidence=0.75,
            latency_ms=30.0,
            cpu_usage_percent=20.0,
            memory_usage_mb=120.0,
            status_code=200,
            error_message=None,
        )

        # Get it back
        result = get_prediction_input(test_db, created.id)

        assert result is not None
        assert result.id == created.id
        assert result.prediction == 0.65
        assert result.confidence == 0.75

    def test_get_nonexistent_prediction(self, test_db: Session):
        """Test getting a prediction that doesn't exist."""
        result = get_prediction_input(test_db, 99999)

        assert result is None


class TestListPredictionInputs:
    """Test suite for list_prediction_inputs function."""

    def test_list_predictions_empty(self, test_db: Session):
        """Test listing predictions when database is empty."""
        result = list_prediction_inputs(test_db)

        assert result == []

    def test_list_predictions_with_data(self, test_db: Session, sample_kick_request):
        """Test listing predictions with data."""
        # Create multiple predictions
        for i in range(5):
            create_prediction_input(
                session=test_db,
                request=sample_kick_request,
                prediction=0.5 + i * 0.1,
                confidence=0.6 + i * 0.05,
                latency_ms=20.0 + i,
                cpu_usage_percent=15.0,
                memory_usage_mb=100.0,
                status_code=200,
                error_message=None,
            )

        result = list_prediction_inputs(test_db)

        assert len(result) == 5


class TestDeletePredictionInput:
    """Test suite for delete_prediction_input function."""

    def test_delete_existing_prediction(self, test_db: Session, sample_kick_request):
        """Test deleting an existing prediction."""
        # Create a prediction
        created = create_prediction_input(
            session=test_db,
            request=sample_kick_request,
            prediction=0.7,
            confidence=0.8,
            latency_ms=25.0,
            cpu_usage_percent=18.0,
            memory_usage_mb=110.0,
            status_code=200,
            error_message=None,
        )

        # Delete it
        result = delete_prediction_input(test_db, created.id)

        assert result is True

        # Verify it's gone
        get_result = get_prediction_input(test_db, created.id)
        assert get_result is None

    def test_delete_nonexistent_prediction(self, test_db: Session):
        """Test deleting a prediction that doesn't exist."""
        result = delete_prediction_input(test_db, 99999)

        assert result is False


class TestPredictionInputModel:
    """Test suite for PredictionInput model."""

    def test_model_repr(self, test_db: Session, sample_kick_request):
        """Test the string representation of PredictionInput model."""
        # Create a prediction
        db_prediction = create_prediction_input(
            session=test_db,
            request=sample_kick_request,
            prediction=1.0,
            confidence=0.92,
            latency_ms=45.0,
            cpu_usage_percent=20.0,
            memory_usage_mb=120.0,
            status_code=200,
            error_message=None,
        )

        # Test __repr__
        repr_str = repr(db_prediction)
        assert "PredictionInput" in repr_str
        assert f"id={db_prediction.id}" in repr_str
        assert f"distance={db_prediction.distance}" in repr_str
        assert f"angle={db_prediction.angle}" in repr_str
        assert f"prediction={db_prediction.prediction}" in repr_str
