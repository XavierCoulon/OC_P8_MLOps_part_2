"""Tests for predictions endpoint."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TestPredictEndpoint:
    """Test suite for /predict endpoint."""

    @pytest.fixture
    def valid_prediction_data(self):
        """Sample valid prediction data."""
        return {
            "distance": 35,
            "angle": 0,
            "time_norm": 0.5,  # Normalized value between 0 and 1
            "wind_speed": 5.0,
            "precipitation_probability": 0.0,
            "is_left_footed": 0,
            "game_away": 0,
            "is_endgame": 0,
            "is_start": 0,
            "is_left_side": 0,
            "has_previous_attempts": 0,
        }

    @pytest.fixture
    def headers(self):
        """Valid API key headers."""
        return {"X-API-Key": os.getenv("API_KEY", "test-api-key-12345")}

    def test_predict_endpoint_requires_authentication(
        self, client: TestClient, valid_prediction_data
    ):
        """Test that predict endpoint requires API key."""
        response = client.post("/api/v1/predict", json=valid_prediction_data)
        assert response.status_code == 403
        assert "detail" in response.json()

    def test_predict_endpoint_with_valid_api_key(
        self, client: TestClient, valid_prediction_data, headers
    ):
        """Test that predict endpoint accepts valid API key."""
        response = client.post(
            "/api/v1/predict", json=valid_prediction_data, headers=headers
        )
        assert response.status_code == 200

    def test_predict_endpoint_returns_prediction(
        self, client: TestClient, valid_prediction_data, headers
    ):
        """Test that predict endpoint returns a prediction value."""
        response = client.post(
            "/api/v1/predict", json=valid_prediction_data, headers=headers
        )
        data = response.json()

        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
        assert 0 <= data["prediction"] <= 1

    def test_predict_endpoint_returns_confidence(
        self, client: TestClient, valid_prediction_data, headers
    ):
        """Test that predict endpoint returns confidence score."""
        response = client.post(
            "/api/v1/predict", json=valid_prediction_data, headers=headers
        )
        data = response.json()

        assert "confidence" in data
        assert isinstance(data["confidence"], (int, float))
        assert 0 <= data["confidence"] <= 1

    def test_predict_endpoint_with_invalid_data_type(self, client: TestClient, headers):
        """Test that predict endpoint rejects invalid data types."""
        invalid_data = {
            "distance": "not_a_number",  # Invalid type
            "angle": 0,
        }
        response = client.post("/api/v1/predict", json=invalid_data, headers=headers)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_with_missing_fields(self, client: TestClient, headers):
        """Test that predict endpoint rejects incomplete data."""
        incomplete_data = {
            "distance": 35,
            # Missing other required fields
        }
        response = client.post("/api/v1/predict", json=incomplete_data, headers=headers)
        assert response.status_code == 422

    def test_predict_endpoint_handles_prediction_error(
        self, client: TestClient, valid_prediction_data, headers
    ):
        """Test that predict endpoint handles prediction errors gracefully."""
        # Mock process_prediction to raise an exception
        with patch("app.api.routes.predictions.process_prediction") as mock_predict:
            mock_predict.side_effect = RuntimeError("Model not loaded")

            response = client.post(
                "/api/v1/predict", json=valid_prediction_data, headers=headers
            )

            assert response.status_code == 500
            assert "detail" in response.json()
            assert "Model not loaded" in response.json()["detail"]


class TestGetPredictionsEndpoint:
    """Test suite for GET /predictions endpoint."""

    @pytest.fixture
    def headers(self):
        """Valid API key headers."""
        return {"X-API-Key": os.getenv("API_KEY", "test-api-key-12345")}

    def test_get_predictions_requires_authentication(self, client: TestClient):
        """Test that get predictions endpoint requires API key."""
        response = client.get("/api/v1/predictions")
        assert response.status_code == 403
        assert "detail" in response.json()

    def test_get_predictions_returns_list(self, client: TestClient, headers):
        """Test that get predictions endpoint returns a list with valid API key."""
        response = client.get("/api/v1/predictions", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_predictions_with_invalid_api_key(self, client: TestClient):
        """Test that endpoint rejects invalid API key."""
        invalid_headers = {"X-API-Key": "wrong-api-key-12345"}
        response = client.get("/api/v1/predictions", headers=invalid_headers)
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]


class TestGetPredictionByIdEndpoint:
    """Test suite for GET /predictions/{id} endpoint."""

    @pytest.fixture
    def headers(self):
        """Valid API key headers."""
        return {"X-API-Key": os.getenv("API_KEY", "test-api-key-12345")}

    def test_get_prediction_by_id_requires_authentication(self, client: TestClient):
        """Test that get prediction by id endpoint requires API key."""
        response = client.get("/api/v1/predictions/1")
        assert response.status_code == 403
        assert "detail" in response.json()

    def test_get_prediction_by_id_not_found(self, client: TestClient, headers):
        """Test that get prediction by id returns 404 for non-existent prediction."""
        response = client.get("/api/v1/predictions/99999", headers=headers)
        assert response.status_code == 404
        assert "detail" in response.json()
        assert "not found" in response.json()["detail"].lower()

    def test_get_prediction_by_id_success(self, client: TestClient, headers):
        """Test successful retrieval of a prediction by id."""
        # First, create a prediction
        valid_data = {
            "distance": 30,
            "angle": 15,
            "time_norm": 0.5,
            "wind_speed": 5.0,
            "precipitation_probability": 0.0,
            "is_left_footed": 0,
            "game_away": 0,
            "is_endgame": 0,
            "is_start": 0,
            "is_left_side": 0,
            "has_previous_attempts": 0,
        }

        # Create prediction (this will save to DB)
        create_response = client.post(
            "/api/v1/predict", json=valid_data, headers=headers
        )
        assert create_response.status_code == 200

        # Now try to get it (should be id 1 in fresh test DB)
        response = client.get("/api/v1/predictions/1", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "distance" in data


class TestDeletePredictionEndpoint:
    """Test suite for DELETE /predictions/{id} endpoint."""

    @pytest.fixture
    def headers(self):
        """Valid API key headers."""
        return {"X-API-Key": os.getenv("API_KEY", "test-api-key-12345")}

    def test_delete_prediction_requires_authentication(self, client: TestClient):
        """Test that delete prediction endpoint requires API key."""
        response = client.delete("/api/v1/predictions/1")
        assert response.status_code == 403
        assert "detail" in response.json()

    def test_delete_prediction_not_found(self, client: TestClient, headers):
        """Test that delete prediction returns 404 for non-existent prediction."""
        response = client.delete("/api/v1/predictions/99999", headers=headers)
        assert response.status_code == 404
        assert "detail" in response.json()
        assert "not found" in response.json()["detail"].lower()

    def test_delete_prediction_success(self, client: TestClient, headers):
        """Test successful deletion of a prediction."""
        # First, create a prediction
        valid_data = {
            "distance": 25,
            "angle": 10,
            "time_norm": 0.3,
            "wind_speed": 3.0,
            "precipitation_probability": 0.1,
            "is_left_footed": 1,
            "game_away": 1,
            "is_endgame": 0,
            "is_start": 1,
            "is_left_side": 0,
            "has_previous_attempts": 1,
        }

        create_response = client.post(
            "/api/v1/predict", json=valid_data, headers=headers
        )
        assert create_response.status_code == 200

        # Delete it
        response = client.delete("/api/v1/predictions/1", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "success" in data["message"].lower()

        # Verify it's deleted
        get_response = client.get("/api/v1/predictions/1", headers=headers)
        assert get_response.status_code == 404
