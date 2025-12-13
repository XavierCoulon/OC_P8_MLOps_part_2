"""Prediction service - orchestrates ML prediction, metrics collection and database logging."""

import os
import time

import psutil
from sqlalchemy.orm import Session

from app.db.crud import create_prediction_input
from app.ml.model_manager import model_manager
from app.models.schemas import KickPredictionRequest

process = psutil.Process(os.getpid())


def process_prediction(db: Session, request: KickPredictionRequest):
    """
    Logique centrale : Prédit, Mesure, et Loggue.
    Utilisé à la fois par l'API et Gradio.

    Args:
        db: Database session
        request: Kick prediction request with features

    Returns:
        Tuple of (prediction, confidence)

    Raises:
        RuntimeError: If model is not loaded
        Exception: Any prediction-related errors
    """
    # 1. INIT MESURES
    process.cpu_percent(interval=None)  # Reset
    start_time = time.time()

    prediction = None
    confidence = None
    status_code = 200
    error_msg = None

    try:
        if not model_manager.initialized:
            raise RuntimeError("Model not loaded")

        # 2. PRÉDICTION
        features = request.model_dump()
        prediction, confidence = model_manager.predict(features)

        return prediction, confidence

    except Exception as e:
        status_code = 500
        error_msg = str(e)
        raise e  # On relance l'erreur pour que l'appelant sache qu'il y a eu un souci

    finally:
        # 3. MESURES FINALES & LOGGING
        latency_ms = (time.time() - start_time) * 1000
        cpu_usage = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # 4. SAUVEGARDE DB
        # On attrape les erreurs DB ici pour ne pas faire planter la réponse utilisateur
        try:
            create_prediction_input(
                session=db,
                request=request,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_mb,
                status_code=status_code,
                error_message=error_msg,
            )
        except Exception as db_e:
            print(f"⚠️ Erreur logging DB: {db_e}")
