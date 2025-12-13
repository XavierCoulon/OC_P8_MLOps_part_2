import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://xaviercoulon-rugbymlops.hf.space/api/v1/predict"
API_KEY = os.getenv("API_KEY", "default-key-change-me")

if not API_KEY or API_KEY == "default-key-change-me":
    print("❌ Erreur : La clé API n'est pas définie ou utilise la valeur par défaut.")
    print("Définis ta clé API dans le fichier .env avant de lancer le script.")
    exit()

DATA_FILE = "data/kicks_ready_for_model.csv"
BATCH_SIZE = 50
API_PAUSE_SECONDS = 1  # Pause entre les requêtes pour éviter d'être banni

if not os.path.exists(DATA_FILE):
    print(f"❌ Erreur : Le fichier {DATA_FILE} est introuvable.")
    print("Vérifie que tu as bien téléchargé le dataset dans le dossier 'data'.")
    exit()

print(f"📂 Chargement de {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# On prend un échantillon au hasard
batch = df.sample(n=BATCH_SIZE)

# Drop target
batch = batch.drop(columns="resultat")

print(f"✨ Données nettoyées. Colonnes envoyées : {list(batch.columns)}")
print(f"🚀 Démarrage de l'envoi vers {API_URL}...")
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
        response = requests.post(API_URL, json=payload, headers=headers)

        # Analyse de la réponse
        if response.status_code == 200:
            data = response.json()
            pred = data.get("prediction", 0.0)
            conf = data.get("confidence", 0.0)
            print(
                f"✅ [{i + 1}/{BATCH_SIZE}] Succès | Pred: {pred : .2f} | Conf: {conf : .2f}"
            )
            success_count += 1
        else:
            print(f"❌ [{i + 1}/{BATCH_SIZE}] Erreur HTTP {response.status_code}")
            print(f"   👉 Détail : {response.text}")
            error_count += 1

    except Exception as e:
        print(f"💀 Erreur de connexion : {str(e)}")
        error_count += 1

    time.sleep(API_PAUSE_SECONDS)

duration = time.time() - start_global
print("-" * 50)
print(f"🎉 Terminé en {duration : .2f} secondes.")
print(f"📊 Bilan : {success_count} réussites / {error_count} échecs.")
