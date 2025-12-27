"""Tests for model manager."""

from unittest.mock import MagicMock, patch

import pytest

from app.ml.model_manager import ModelManager


class TestModelManager:
    """Test suite for ModelManager class."""

    def test_model_manager_singleton(self):
        """Test that ModelManager follows singleton pattern."""
        manager1 = ModelManager()
        manager2 = ModelManager()
        assert manager1 is manager2

    def test_load_model_failure(self):
        """Test that load_model raises RuntimeError on failure."""
        manager = ModelManager()

        # Mock hf_hub_download to raise an exception
        with patch("app.ml.model_manager.hf_hub_download") as mock_download:
            mock_download.side_effect = Exception("Download failed")

            with pytest.raises(RuntimeError, match="Unable to load ML model"):
                manager.load_model("fake-repo/fake-model")

    def test_predict_without_loading_model(self):
        """Test that predict raises ValueError when model not loaded."""
        manager = ModelManager()
        manager.initialized = False

        with pytest.raises(ValueError, match="Model not loaded"):
            manager.predict({"distance": 30, "angle": 15})

    def test_predict_error_handling(self):
        """Test that predict handles errors during prediction."""
        manager = ModelManager()
        manager.initialized = True

        # Mock ONNX session that raises exception during run
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Prediction error")
        manager._session = mock_session

        with pytest.raises(Exception, match="Prediction error"):
            manager.predict({"distance": 30, "angle": 15})
