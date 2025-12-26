"""Prediction service - orchestrates ML prediction, metrics collection and database logging."""

import logging
import os
import time

import psutil
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.db.crud import create_prediction_input
from app.ml.model_manager import model_manager
from app.models.schemas import KickPredictionRequest

logger = logging.getLogger(__name__)
process = psutil.Process(os.getpid())


def log_prediction_background(
    db: Session,
    request_data: dict,
    prediction: int | float | None,  # <--- CORRECTION 1 : On accepte float
    confidence: float | None,
    latency_ms: float,
    status_code: int,
    error_msg: str | None,
):
    """
    Tâche d'arrière-plan pour logger les métriques et la prédiction.
    S'exécute APRES que la réponse soit envoyée à l'utilisateur.
    """
    try:
        cpu_usage = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # CORRECTION 2 : Conversion explicite en int pour la DB
        # Si le modèle renvoie 1.0 (float), on le transforme en 1 (int)
        final_prediction = int(prediction) if prediction is not None else None

        # CORRECTION 3 : Dict -> Pydantic
        # On reconstruit l'objet pour satisfaire le typage du CRUD
        request_object = KickPredictionRequest(**request_data)

        create_prediction_input(
            session=db,
            request=request_object,
            prediction=final_prediction,  # On passe le int propre
            confidence=confidence,
            latency_ms=latency_ms,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_mb,
            status_code=status_code,
            error_message=error_msg,
        )
        db.commit()

    except Exception as e:
        logger.error(f"⚠️ Background Logging Error: {e}")


def process_prediction(
    db: Session, request: KickPredictionRequest, background_tasks: BackgroundTasks
):
    """
    Logique centrale optimisée : Prédit (Vite) et délègue le Log (Background).
    """
    start_time = time.time()

    prediction = None
    confidence = None
    status_code = 200
    error_msg = None

    # Capture des données brutes (rapide)
    request_dict = request.model_dump()

    try:
        if not model_manager.initialized:
            raise RuntimeError("Model not loaded")

        # 1. PRÉDICTION
        prediction, confidence = model_manager.predict(request_dict)

        return prediction, confidence

    except Exception as e:
        status_code = 500
        error_msg = str(e)
        prediction = None
        confidence = 0.0
        raise e

    finally:
        # 2. CALCUL LATENCE
        latency_ms = (time.time() - start_time) * 1000

        # 3. DÉLÉGATION
        background_tasks.add_task(
            log_prediction_background,
            db,
            request_dict,
            prediction,  # Pylance est content car log_... accepte int | float
            confidence,
            latency_ms,
            status_code,
            error_msg,
        )
