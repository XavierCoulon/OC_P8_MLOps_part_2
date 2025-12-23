"""Tests for health check endpoint."""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_endpoint_returns_200(self, client: TestClient):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_healthy_status(self, client: TestClient):
        """Test that health endpoint returns healthy status."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_has_required_fields(self, client: TestClient):
        """Test that health endpoint response has required fields."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "message" in data

    def test_health_endpoint_message_format(self, client: TestClient):
        """Test that health endpoint returns valid message."""
        response = client.get("/api/v1/health")
        data = response.json()
        # Message should be a non-empty string
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0
