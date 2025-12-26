"""Model manager optimized for Low Latency inference (Fast Pandas)."""

import logging
import warnings
from typing import Optional

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# Suppress scikit-learn feature names warning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for ML model loading and inference using joblib."""

    _instance: Optional["ModelManager"] = None
    _model = None

    # Défini une seule fois pour éviter la reconstruction à chaque appel
    # C'est l'ordre EXACT attendu par ton modèle (ColumnTransformer)
    FEATURE_ORDER = [
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

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model manager."""
        if not hasattr(self, "initialized"):
            self.initialized = False
            self.model_name = None

    def load_model(self, hf_repo_id: str) -> None:
        """Load model from Hugging Face Hub."""
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
            logger.info("✅ Model downloaded and loaded (Optimized Pandas Mode).")

        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise RuntimeError("Unable to load ML model.") from e

    def predict(self, features: dict) -> tuple[float, float]:
        """Make prediction using Optimized DataFrame construction.

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
            # --- OPTIMISATION 1 : Fast Pandas Construction ---
            # Au lieu de pd.DataFrame([features]), qui est lent car il analyse les clés,
            # on extrait les valeurs dans une liste et on spécifie les colonnes.
            # C'est la méthode la plus rapide pour créer un DataFrame compatible Sklearn.

            data_values = [features.get(f, 0) for f in self.FEATURE_ORDER]
            input_df = pd.DataFrame([data_values], columns=self.FEATURE_ORDER)

            # --- OPTIMISATION 2 : Single Inference Call ---
            # On appelle UNIQUEMENT predict_proba.
            # Cela évite de parcourir les arbres de décision deux fois (predict + predict_proba).

            probas = self._model.predict_proba(input_df)[0]

            # La classe prédite est l'index de la proba max (0 ou 1)
            prediction = int(probas.argmax())
            # La confiance est la valeur max
            confidence = float(probas.max())

            return prediction, confidence

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


# Global instance
model_manager = ModelManager()
