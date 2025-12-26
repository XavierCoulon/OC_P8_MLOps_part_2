"""Gradio app launcher for rugby kick prediction."""

# flake8: noqa=E231

import argparse
import logging

import gradio as gr

from app.config.settings import settings
from app.db.database import _get_session_local
from app.ml.model_manager import model_manager
from app.models.schemas import KickPredictionRequest
from app.services import process_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Configuration des labels ===
CLEAN_LABELS = {
    "time_norm": "Temps Normalisé (0-1)",
    "distance": "Distance (mètres)",
    "angle": "Angle (degrés)",
    "wind_speed": "Vitesse du Vent (km/h)",
    "precipitation_probability": "Probabilité de Précipitations (0-1)",
    "is_left_footed": "Gaucher",
    "game_away": "Match à l'Extérieur",
    "is_endgame": "Fin du Match",
    "is_start": "Début du Match",
    "is_left_side": "Côté Gauche",
    "has_previous_attempts": "Tentatives Précédentes",
}

# === Organisation des champs par section ===
PLAYER_CHARACTERISTICS = [
    "is_left_footed",
    "has_previous_attempts",
]

MATCH_CONDITIONS = [
    "time_norm",
    "is_start",
    "is_endgame",
    "game_away",
]

KICK_PARAMETERS = [
    "distance",
    "angle",
    "is_left_side",
]

WEATHER_CONDITIONS = [
    "wind_speed",
    "precipitation_probability",
]

# === Valeurs par défaut réalistes ===
DEFAULT_VALUES = {
    "time_norm": 0.5,
    "distance": 30.0,
    "angle": 40.0,
    "wind_speed": 5.0,
    "precipitation_probability": 0.2,
    "is_left_footed": False,
    "game_away": False,
    "is_endgame": False,
    "is_start": False,
    "is_left_side": False,
    "has_previous_attempts": False,
}

# === Plages de champs ===
FIELD_RANGES = {
    "time_norm": (0.01, 1.0, 0.01),
    "distance": (2.0, 100.0, 1.0),
    "angle": (0.0, 90.0, 1.0),
    "wind_speed": (0.0, 50.0, 1.0),
    "precipitation_probability": (0.0, 1.0, 0.01),
}


def create_input_component(feature: str):
    """Crée un composant d'entrée approprié pour une feature.

    Args:
        feature: Nom du champ

    Returns:
        Composant Gradio approprié
    """
    clean_label = CLEAN_LABELS.get(feature, feature.replace("_", " ").title())
    default_value = DEFAULT_VALUES.get(feature, False)

    if feature in [
        "is_left_footed",
        "game_away",
        "is_endgame",
        "is_start",
        "is_left_side",
        "has_previous_attempts",
    ]:
        # Checkbox pour les champs booléens
        return gr.Checkbox(label=clean_label, value=default_value)
    elif feature in FIELD_RANGES:
        # Slider pour les champs numériques
        min_val, max_val, step = FIELD_RANGES[feature]
        return gr.Slider(
            minimum=min_val,
            maximum=max_val,
            value=default_value,
            step=step,
            label=clean_label,
        )
    else:
        # Default: Number
        return gr.Number(label=clean_label, value=default_value)


def predict_from_ui(**kwargs) -> tuple[str, str]:
    """Effectue une prédiction via le modèle directement et sauvegarde en BD.

    Args:
        **kwargs: Paramètres du tir

    Returns:
        Tuple (prédiction, confiance)
    """
    # Création session manuelle (spécifique à Gradio)
    SessionLocal = _get_session_local()
    session = SessionLocal()

    try:
        # Conversion en Pydantic (validation gratuite !)
        request = KickPredictionRequest(**kwargs)

        # Appel du service partagé
        prediction, confidence = process_prediction(session, request)

        return f"{prediction:.4f}", f"{confidence:.4f}"

    except Exception as e:
        logger.error(f"Erreur UI: {e}")
        return "Erreur", str(e)

    finally:
        session.close()  # Toujours fermer la session manuelle


def predict_wrapper(*args) -> tuple[str, dict]:
    """Wrapper pour la fonction de prédiction avec formatage.

    Args:
        *args: Arguments depuis l'interface

    Returns:
        Tuple (résultat formaté, détails)
    """
    try:
        all_features = (
            PLAYER_CHARACTERISTICS
            + MATCH_CONDITIONS
            + KICK_PARAMETERS
            + WEATHER_CONDITIONS
        )
        data = dict(zip(all_features, args))

        prediction_str, confidence_str = predict_from_ui(**data)

        if prediction_str == "Erreur":
            error_msg = f"❌ **Erreur**: {confidence_str}"
            return (error_msg, {"error": confidence_str})

        prediction = float(prediction_str)
        confidence = float(confidence_str)

        # Formatage du résultat
        prob_pct = f"{prediction:.2%}"
        conf_score = f"{confidence:.4f}"
        main_output = f"🎯 **Probabilité de Réussite**: {prob_pct}\n"
        main_output += f"📊 **Score de Confiance**: {conf_score}\n"

        if prediction >= 0.7:
            main_output += "✅ Tir avec **forte probabilité de réussite**"
        elif prediction >= 0.4:
            main_output += "⚠️ Tir avec **probabilité modérée**"
        else:
            main_output += "❌ Tir avec **faible probabilité de réussite**"

        details = {
            "probability": prob_pct,
            "confidence": confidence,
            "input_data": data,
        }

        return main_output, details

    except Exception as e:
        error_msg = f"❌ **Erreur lors de la prédiction**: {str(e)}"
        return error_msg, {"error": str(e)}


def build_interface() -> gr.Blocks:
    """Construit l'interface Gradio organisée.

    Returns:
        Interface Gradio
    """
    with gr.Blocks(title="Prédiction de Tir Rugby") as demo:
        gr.Markdown("# 🏉 Prédiction de Tir Rugby")
        gr.Markdown(
            "Prédisez la probabilité de réussite d'un tir au but en fonction "
            "des conditions du match"
        )

        # Section Joueur et Match
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👤 Joueur")
                player_inputs = [
                    create_input_component(f) for f in PLAYER_CHARACTERISTICS
                ]

            with gr.Column():
                gr.Markdown("### 🏟️ Match")
                match_inputs = [create_input_component(f) for f in MATCH_CONDITIONS]

        # Section Tir et Météo
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 Tir")
                kick_inputs = [create_input_component(f) for f in KICK_PARAMETERS]

            with gr.Column():
                gr.Markdown("### 🌦️ Météo")
                weather_inputs = [create_input_component(f) for f in WEATHER_CONDITIONS]

        # Bouton et résultats
        gr.Markdown("---")
        predict_btn = gr.Button("🎯 Prédire", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                prediction_output = gr.Textbox(
                    label="📋 Résultat",
                    lines=4,
                    interactive=False,
                )
            with gr.Column():
                details_output = gr.JSON(label="📈 Détails")

        # Assembly des inputs
        all_inputs = player_inputs + match_inputs + kick_inputs + weather_inputs

        predict_btn.click(
            fn=predict_wrapper,
            inputs=all_inputs,
            outputs=[prediction_output, details_output],
        )

    return demo


def main():
    """Lance l'interface Gradio."""
    parser = argparse.ArgumentParser(description="Lancer l'interface Gradio")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Partager avec un lien public",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port de l'interface",
    )
    args = parser.parse_args()

    # Charger le modèle au démarrage
    logger.info("Chargement du modèle...")
    try:
        model_manager.load_model(hf_repo_id=settings.hf_repo_id)
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        raise

    logger.info(f"Démarrage de l'interface Gradio sur le port {args.port}...")
    demo = build_interface()
    demo.launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
