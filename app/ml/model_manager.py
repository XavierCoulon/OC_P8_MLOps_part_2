"""Model manager optimized for Low Latency inference with ONNX Runtime."""

import logging
from typing import Optional

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for ML model loading and inference using ONNX Runtime."""

    _instance: Optional["ModelManager"] = None
    _session = None

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
        """Load ONNX model from Hugging Face Hub."""
        if self.initialized and self.model_name == hf_repo_id:
            logger.info(f"Model {hf_repo_id} already loaded")
            return

        logger.info(f"🌐 Downloading ONNX model from Hugging Face Hub ({hf_repo_id})...")
        try:
            model_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename="model.onnx",
            )

            # Create ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            self.model_name = hf_repo_id
            self.initialized = True
            logger.info("✅ ONNX model downloaded and loaded (Optimized ONNX Runtime).")

        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise RuntimeError("Unable to load ML model.") from e

    def predict(self, features: dict) -> tuple[float, float]:
        """Make prediction using ONNX Runtime.

        Args:
            features: Dictionary with model features

        Returns:
            Tuple of (prediction, confidence)

        Raises:
            ValueError: If model is not initialized
        """
        if not self.initialized or self._session is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # --- OPTIMISATION : Préparer les inputs pour ONNX ---
            # Le modèle ONNX attend chaque feature comme input séparé
            # Conversion en float32 pour compatibilité ONNX
            input_feed = {
                feature: np.array([[features.get(feature, 0)]], dtype=np.float32)
                for feature in self.FEATURE_ORDER
            }

            # --- ONNX Runtime Inference ---
            # Exécution de l'inférence avec le dictionnaire d'inputs
            outputs = self._session.run(None, input_feed)

            # Format ONNX: [label, [{0: prob_class_0, 1: prob_class_1}]]
            # outputs[1] est une liste contenant un dict de probabilités
            probas_dict = outputs[1][0]  # {0: 0.54, 1: 0.46}

            # Extraire les probabilités dans un array
            probas = np.array([probas_dict[0], probas_dict[1]])

            # La classe prédite est l'index de la proba max (0 ou 1)
            prediction = int(np.argmax(probas))
            # La confiance est la valeur max
            confidence = float(np.max(probas))

            return prediction, confidence

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


# Global instance
model_manager = ModelManager()
