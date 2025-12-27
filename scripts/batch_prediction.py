"""Script de prédictions en batch via l'API."""

import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def main(
    batch_size: int = 100, api_pause: float = 0.5, distance_drift: bool = False
) -> None:
    """Effectue des prédictions en batch sur un échantillon du dataset.

    Args:
        batch_size: Nombre de prédictions à effectuer
        api_pause: Temps de pause entre chaque requête (secondes)
        distance_drift: Si True, filtre uniquement les échantillons avec distance > 40m
    """
    HF_API_URI = os.getenv("HF_API_URI")
    if not HF_API_URI:
        print("❌ Erreur : L'URL de l'API Hugging Face n'est pas définie.")
        print("Définis HF_API_URI dans le fichier .env avant de lancer le script.")
        sys.exit(1)

    HF_API_PREDICT_ENDPOINT = f"{HF_API_URI}/predict"
    API_KEY = os.getenv("API_KEY", "default-key-change-me")

    if not API_KEY or API_KEY == "default-key-change-me":
        print(
            "❌ Erreur : La clé API n'est pas définie ou utilise la valeur par défaut."
        )
        print("Définis ta clé API dans le fichier .env avant de lancer le script.")
        sys.exit(1)

    DATA_FILE = "data/kicks_ready_for_model.csv"

    if not os.path.exists(DATA_FILE):
        print(f"❌ Erreur : Le fichier {DATA_FILE} est introuvable.")
        print("Vérifie que tu as bien téléchargé le dataset dans le dossier 'data'.")
        sys.exit(1)

    print(f"📂 Chargement de {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Filtrage optionnel pour drift de distance
    if distance_drift:
        df = df[df["distance"] > 40]
        print(
            f"🔍 Filtrage appliqué : distance > 40m ({len(df)} échantillons disponibles)"
        )

    # On prend un échantillon au hasard
    batch = df.sample(n=batch_size)

    # Drop target
    batch = batch.drop(columns="resultat")

    print(f"✨ Données nettoyées. Colonnes envoyées : {list(batch.columns)}")
    print(f"🚀 Démarrage de l'envoi vers {HF_API_PREDICT_ENDPOINT}...")
    print(f"📦 Taille du batch : {batch_size}")
    print("-" * 50)

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    success_count = 0
    error_count = 0
    start_global = time.time()

    # Conversion en liste de dictionnaires pour l'envoi
    payloads = batch.to_dict(orient="records")

    for i, payload in enumerate(payloads):
        try:
            # Envoi de la requête POST
            response = requests.post(
                HF_API_PREDICT_ENDPOINT, json=payload, headers=headers
            )

            # Analyse de la réponse
            if response.status_code == 200:
                data = response.json()
                pred = data.get("prediction", 0.0)
                conf = data.get("confidence", 0.0)
                print(
                    f"✅ [{i + 1}/{batch_size}] Succès | Pred: {pred: .2f} | Conf: {conf: .2f}"
                )
                success_count += 1
            else:
                print(f"❌ [{i + 1}/{batch_size}] Erreur HTTP {response.status_code}")
                print(f"   👉 Détail : {response.text}")
                error_count += 1

        except Exception as e:
            print(f"💀 Erreur de connexion : {str(e)}")
            error_count += 1

        time.sleep(api_pause)

    duration = time.time() - start_global
    print("-" * 50)
    print(f"🎉 Terminé en {duration: .2f} secondes.")
    print(f"📊 Bilan : {success_count} réussites / {error_count} échecs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Effectue des prédictions en batch via l'API"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Nombre de prédictions à effectuer (défaut: 1000)",
    )
    parser.add_argument(
        "--api-pause",
        type=float,
        default=0.5,
        help="Pause entre les requêtes en secondes (défaut: 0.5)",
    )
    parser.add_argument(
        "--distance-drift",
        action="store_true",
        help="Filtre uniquement les échantillons avec distance > 40m",
    )

    args = parser.parse_args()
    main(
        batch_size=args.batch_size,
        api_pause=args.api_pause,
        distance_drift=args.distance_drift,
    )
