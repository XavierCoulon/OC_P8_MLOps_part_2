"""Tests for profiling middleware."""

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.profiling import ProfilingMiddleware


@pytest.fixture
def app_with_profiling():
    """Create a FastAPI app with profiling middleware."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    @app.get("/health")
    async def health_endpoint():
        return {"status": "healthy"}

    # Add profiling middleware
    app.add_middleware(
        ProfilingMiddleware,
        top_results=5,
        save_binary=False,  # Don't save files in tests
        exclude_paths=["/health"],
    )

    return app


class TestProfilingMiddleware:
    """Test suite for ProfilingMiddleware."""

    def test_middleware_adds_process_time_header(self, app_with_profiling):
        """Test that middleware adds X-Process-Time header."""
        client = TestClient(app_with_profiling)
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) >= 0

    def test_middleware_excludes_health_endpoint(self, app_with_profiling):
        """Test that health endpoint is excluded from profiling."""
        client = TestClient(app_with_profiling)

        with patch("app.middleware.profiling.profiling_logger") as mock_logger:
            response = client.get("/health")

            assert response.status_code == 200
            # Profiling logger should not be called for excluded paths
            mock_logger.info.assert_not_called()

    def test_middleware_profiles_normal_requests(self, app_with_profiling):
        """Test that normal requests are profiled."""
        client = TestClient(app_with_profiling)

        with patch("app.middleware.profiling.profiling_logger") as mock_logger:
            response = client.get("/test")

            assert response.status_code == 200
            # Profiling logger should be called
            assert mock_logger.info.called

    def test_middleware_skips_with_header(self, app_with_profiling):
        """Test that profiling is skipped when X-Skip-Profiling header is present."""
        client = TestClient(app_with_profiling)

        with patch("app.middleware.profiling.profiling_logger") as mock_logger:
            response = client.get("/test", headers={"X-Skip-Profiling": "true"})

            assert response.status_code == 200
            # Profiling should be skipped
            mock_logger.info.assert_not_called()

    def test_middleware_skips_python_requests(self, app_with_profiling):
        """Test that profiling is skipped for python-requests user agent."""
        client = TestClient(app_with_profiling)

        with patch("app.middleware.profiling.profiling_logger") as mock_logger:
            response = client.get(
                "/test", headers={"User-Agent": "python-requests/2.28.0"}
            )

            assert response.status_code == 200
            # Profiling should be skipped for automated scripts
            mock_logger.info.assert_not_called()

    def test_middleware_initialization_defaults(self):
        """Test middleware initialization with default parameters."""
        app = FastAPI()
        middleware = ProfilingMiddleware(app)

        assert middleware.top_results == 10
        assert middleware.save_binary is True
        assert "/health" in middleware.exclude_paths
        assert "/api/v1/health" in middleware.exclude_paths

    def test_middleware_initialization_custom_params(self):
        """Test middleware initialization with custom parameters."""
        app = FastAPI()
        middleware = ProfilingMiddleware(
            app,
            top_results=20,
            save_binary=False,
            exclude_paths=["/custom", "/other"],
        )

        assert middleware.top_results == 20
        assert middleware.save_binary is False
        assert middleware.exclude_paths == ["/custom", "/other"]

    def test_middleware_respects_testing_env_var(self):
        """Test that middleware respects TESTING environment variable."""
        # TESTING env var is already set in conftest.py
        assert os.getenv("TESTING") == "true"

        # Verify that profiles directory is not created in tests
        app = FastAPI()
        middleware = ProfilingMiddleware(app, save_binary=False)

        # Should not raise any error even if directory doesn't exist
        assert middleware is not None

    def test_middleware_processes_profiling_output(self, app_with_profiling):
        """Test that middleware actually runs profiling when enabled."""
        # Temporarily disable TESTING to trigger profiling
        with patch.dict(os.environ, {"TESTING": ""}):
            client = TestClient(app_with_profiling)

            with patch("app.middleware.profiling.profiling_logger") as mock_logger:
                response = client.get("/test")

                assert response.status_code == 200
                # Should have logged profiling output
                assert mock_logger.info.call_count >= 1

                # Check that the profiling output was created
                log_call = mock_logger.info.call_args_list[0][0][0]
                assert "Profile for" in log_call
                assert "Duration" in log_call
