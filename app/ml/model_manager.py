"""Model manager for loading and using Hugging Face models."""

import logging
from typing import Optional

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for ML model loading and inference using joblib."""

    _instance: Optional["ModelManager"] = None
    _model = None

    def __new__(cls):
        """Singleton pattern to ensure only one model is loaded."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model manager."""
        if not hasattr(self, "initialized"):
            self.initialized = False
            self.model_name = None

    def load_model(self, hf_repo_id: str) -> None:
        """Load model from Hugging Face Hub.

        Args:
            hf_repo_id: Hugging Face repository ID (e.g., "XavierCoulon/rugby-model")
        """
        if self.initialized and self.model_name == hf_repo_id:
            logger.info(f"Model {hf_repo_id} already loaded")
            return

        logger.info(f"🌐 Downloading model from Hugging Face Hub ({hf_repo_id})...")
        try:
            model_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename="model.pkl",
            )

            self._model = joblib.load(model_path)
            self.model_name = hf_repo_id
            self.initialized = True

            logger.info("✅ Model downloaded and loaded from Hugging Face Hub.")

        except Exception as e:
            logger.error(f"❌ Failed to download model from Hugging Face: {e}")
            raise RuntimeError("Unable to load ML model.") from e

    def predict(self, features: dict) -> tuple[float, float]:
        """Make prediction on input features.

        Args:
            features: Dictionary with model features

        Returns:
            Tuple of (prediction, confidence)

        Raises:
            ValueError: If model is not initialized
        """
        if not self.initialized or self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Prepare feature DataFrame
            feature_df = self._prepare_input(features)

            # Make prediction
            prediction = self._model.predict(feature_df)[0]
            confidence = self._model.predict_proba(feature_df).max()

            return float(prediction), float(confidence)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _prepare_input(self, features: dict) -> pd.DataFrame:
        """Prepare input features for model inference.

        Args:
            features: Dictionary with model features

        Returns:
            DataFrame with features in correct order
        """
        feature_order = [
            "time_norm",
            "distance",
            "angle",
            "wind_speed",
            "precipitation_probability",
            "is_left_footed",
            "game_away",
            "is_endgame",
            "is_start",
            "is_left_side",
            "has_previous_attempts",
        ]

        # Create DataFrame from features
        return pd.DataFrame([{f: features.get(f, 0) for f in feature_order}])


# Global instance
model_manager = ModelManager()
