"""Tests for prediction service."""

from unittest.mock import MagicMock, patch

import pytest
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

    def test_successful_prediction(self, test_db: Session, valid_request):
        """Test successful prediction with all metrics collected."""
        # Arrange - Model manager is mocked in conftest
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        model_manager.predict = MagicMock(return_value=(1, 0.85))

        # Act
        prediction, confidence = process_prediction(test_db, valid_request)

        # Assert
        assert prediction == 1
        assert confidence == 0.85
        model_manager.predict.assert_called_once()

    def test_model_not_initialized_raises_error(self, test_db: Session, valid_request):
        """Test that RuntimeError is raised when model is not loaded."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = False

        # Act & Assert
        with pytest.raises(RuntimeError, match="Model not loaded"):
            process_prediction(test_db, valid_request)

    def test_prediction_error_handling(self, test_db: Session, valid_request):
        """Test that prediction errors are properly handled and re-raised."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        model_manager.predict = MagicMock(
            side_effect=ValueError("Invalid feature values")
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid feature values"):
            process_prediction(test_db, valid_request)

    def test_database_error_is_caught_and_logged(self, test_db: Session, valid_request):
        """Test that database errors are caught and logged without affecting the response."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        model_manager.predict = MagicMock(return_value=(1, 0.85))

        # Mock create_prediction_input to raise an exception
        with patch(
            "app.services.prediction_service.create_prediction_input"
        ) as mock_create:
            mock_create.side_effect = Exception("Database connection failed")

            # Capture print output
            with patch("builtins.print") as mock_print:
                # Act
                prediction, confidence = process_prediction(test_db, valid_request)

                # Assert - prediction still succeeds
                assert prediction == 1
                assert confidence == 0.85

                # Assert - error was logged
                mock_print.assert_called_once()
                call_args = mock_print.call_args[0][0]
                assert "⚠️ Erreur logging DB:" in call_args
                assert "Database connection failed" in call_args

    def test_metrics_are_collected(self, test_db: Session, valid_request):
        """Test that CPU and memory metrics are collected during prediction."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        model_manager.predict = MagicMock(return_value=(0, 0.65))

        # Mock create_prediction_input to verify metrics
        with patch(
            "app.services.prediction_service.create_prediction_input"
        ) as mock_create:
            # Act
            process_prediction(test_db, valid_request)

            # Assert
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs

            # Verify all expected parameters are passed
            assert call_kwargs["session"] == test_db
            assert call_kwargs["request"] == valid_request
            assert call_kwargs["prediction"] == 0
            assert call_kwargs["confidence"] == 0.65
            assert call_kwargs["status_code"] == 200
            assert call_kwargs["error_message"] is None

            # Verify metrics are present and reasonable
            assert "latency_ms" in call_kwargs
            assert call_kwargs["latency_ms"] >= 0
            assert "cpu_usage_percent" in call_kwargs
            assert "memory_usage_mb" in call_kwargs

    def test_error_metrics_on_prediction_failure(self, test_db: Session, valid_request):
        """Test that error status and message are logged when prediction fails."""
        # Arrange
        from app.ml.model_manager import model_manager

        model_manager.initialized = True
        error_msg = "Feature extraction failed"
        model_manager.predict = MagicMock(side_effect=Exception(error_msg))

        # Mock create_prediction_input to verify error metrics
        with patch(
            "app.services.prediction_service.create_prediction_input"
        ) as mock_create:
            # Act & Assert
            with pytest.raises(Exception, match=error_msg):
                process_prediction(test_db, valid_request)

            # Assert - DB logging was still attempted with error info
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs

            assert call_kwargs["prediction"] is None
            assert call_kwargs["confidence"] is None
            assert call_kwargs["status_code"] == 500
            assert call_kwargs["error_message"] == error_msg
