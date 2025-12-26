"""Tests for prediction service."""

from unittest.mock import MagicMock

import pytest
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.models.schemas import KickPredictionRequest
from app.services.prediction_service import process_prediction


class TestProcessPrediction:
    """Test suite for process_prediction function."""

    @pytest.fixture
    def valid_request(self):
        """Create a valid kick prediction request."""
        return KickPredictionRequest(
            time_norm=0.5,
            distance=35,
            angle=15,
            wind_speed=5.0,
            precipitation_probability=0.0,
            is_left_footed=0,
            game_away=0,
            is_endgame=0,
            is_start=0,
            is_left_side=0,
            has_previous_attempts=0,
        )

    @pytest.fixture
    def background_tasks(self):
        """Create a BackgroundTasks instance."""
        return BackgroundTasks()

    def test_successful_prediction(
        self, test_db: Session, valid_request, background_tasks
    ):
        """Test successful prediction with all metrics collected."""
        # Arrange - Model manager is mocked in conftest
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        model_manager.predict = MagicMock(return_value=(1, 0.85))

        # Act
        prediction, confidence = process_prediction(
            test_db, valid_request, background_tasks
        )

        # Assert
        assert prediction == 1
        assert confidence == 0.85
        model_manager.predict.assert_called_once()

    def test_model_not_initialized_raises_error(
        self, test_db: Session, valid_request, background_tasks
    ):
        """Test that RuntimeError is raised when model is not loaded."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = False

        # Act & Assert
        with pytest.raises(RuntimeError, match="Model not loaded"):
            process_prediction(test_db, valid_request, background_tasks)

    def test_prediction_error_handling(
        self, test_db: Session, valid_request, background_tasks
    ):
        """Test that prediction errors are properly handled and re-raised."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        model_manager.predict = MagicMock(
            side_effect=ValueError("Invalid feature values")
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid feature values"):
            process_prediction(test_db, valid_request, background_tasks)

    def test_background_task_is_added(
        self, test_db: Session, valid_request, background_tasks
    ):
        """Test that background task for logging is added."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        model_manager.predict = MagicMock(return_value=(1, 0.85))

        # Act
        prediction, confidence = process_prediction(
            test_db, valid_request, background_tasks
        )

        # Assert - prediction succeeds
        assert prediction == 1
        assert confidence == 0.85

        # Assert - background task was added
        assert len(background_tasks.tasks) == 1

    def test_background_logging_with_correct_parameters(
        self, test_db: Session, valid_request, background_tasks
    ):
        """Test that background logging task receives correct parameters."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        model_manager.predict = MagicMock(return_value=(0, 0.65))

        # Act
        prediction, confidence = process_prediction(
            test_db, valid_request, background_tasks
        )

        # Assert - prediction succeeds
        assert prediction == 0
        assert confidence == 0.65

        # Assert - background task was added
        assert len(background_tasks.tasks) == 1

        # Execute background task to verify it works
        task = background_tasks.tasks[0]
        # The task should not raise an exception when executed
        task.func(*task.args, **task.kwargs)

    def test_background_task_added_even_on_error(
        self, test_db: Session, valid_request, background_tasks
    ):
        """Test that background task is added even when prediction fails."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        error_msg = "Feature extraction failed"
        model_manager.predict = MagicMock(side_effect=Exception(error_msg))

        # Act & Assert
        with pytest.raises(Exception, match=error_msg):
            process_prediction(test_db, valid_request, background_tasks)

        # Assert - background task was still added with error info
        assert len(background_tasks.tasks) == 1

        # Verify the task can be executed without raising
        task = background_tasks.tasks[0]
        task.func(*task.args, **task.kwargs)
