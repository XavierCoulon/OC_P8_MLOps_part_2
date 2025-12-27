---
title: Rugby Kick Success Predictor
emoji: 🏉
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---

# 🏉 Rugby MLOps - Kick Success Predictor

API de prédiction de réussite de coups de pied au rugby avec monitoring, profiling et déploiement automatisé.

## 🎯 Fonctionnalités

-   **API FastAPI** : Endpoint `/predict` pour prédictions en temps réel
-   **Interface Gradio** : Interface web interactive pour tester les prédictions
-   **Monitoring** : Logging des prédictions en base de données avec métriques de performance
-   **Profiling** : Analyse des performances avec cProfile et SnakeViz
-   **CI/CD** : Tests automatiques et déploiement sur Docker Hub et Hugging Face
-   **Data Drift** : Évaluation de la dérive de données avec Evidently

## 🚀 Démarrage Rapide

### Prérequis

-   Python 3.12+
-   Docker (optionnel)
-   Make

### Installation

```bash
# Cloner le repo
git clone <repo-url>
cd OC_P8_Rugby_MLOps

# Installer les dépendances
pip install -e .

# Configuration
cp .env.example .env
# Éditer .env avec vos clés API
```

### Configuration Requise (.env)

```bash
API_KEY=votre-cle-api
HF_REPO_ID=XavierCoulon/rugby-kicks-model
DATABASE_URL=postgresql://user:pass@localhost:5432/rugby
```

## 💻 Utilisation

### Lancement Local

```bash
# Avec profiling (SQLite local)
make run-local

# Interface Gradio
make ui

# Docker (PostgreSQL)
make up
```

### Tests

```bash
# Tests complets
make test

# Avec couverture
make coverage
```

### Prédictions en Batch

```bash
# Batch standard (1000 échantillons)
make batch

# Batch personnalisé
make batch BATCH_SIZE=500

# Batch avec drift (distance > 40m)
make batch-drift
```

## 📊 Endpoints API

### POST /api/v1/predict

Prédire la réussite d'un coup de pied.

**Paramètres** :

-   `distance` (float) : Distance en mètres (2-100)
-   `angle` (float) : Angle en degrés (0-90)
-   `wind_speed` (float) : Vitesse du vent en km/h (0-50)
-   `time_norm` (float) : Temps normalisé (0-1)
-   `precipitation_probability` (float) : Probabilité de précipitations (0-1)
-   `is_left_footed` (bool) : Gaucher
-   `game_away` (bool) : Match à l'extérieur
-   `is_endgame` (bool) : Fin de match
-   `is_start` (bool) : Début de match
-   `is_left_side` (bool) : Côté gauche
-   `has_previous_attempts` (bool) : Tentatives précédentes

**Réponse** :

```json
{
    "prediction": 0.85,
    "confidence": 0.92
}
```

### GET /api/v1/health

Vérifier l'état de l'API.

### GET /api/v1/predictions

Lister toutes les prédictions enregistrées.

### GET /api/v1/predictions/{id}

Récupérer une prédiction spécifique.

### DELETE /api/v1/predictions/{id}

Supprimer une prédiction.

## 🔍 Profiling

Le profiling est activé en mode debug pour analyser les performances :

```bash
# Lancer en mode profiling
make run-local

# Analyser les résultats
snakeviz profiles/*.prof
```

Les fichiers de profiling sont générés dans `profiles/` avec timestamp et endpoint.

## 📈 Monitoring de Drift

```bash
# Générer un rapport de drift
make evaluate
```

Rapport généré dans `data/drift_reports/`.

## 🏗️ Architecture

```
app/
├── api/          # Routes FastAPI
├── config/       # Configuration
├── db/           # Base de données (CRUD, models)
├── ml/           # Gestion du modèle ML
├── models/       # Schémas Pydantic
├── services/     # Logique métier
├── middleware/   # Profiling middleware
└── utils/        # Utilitaires (logging)

scripts/
├── batch_prediction.py  # Prédictions batch
└── evaluate_drift.py    # Analyse de drift

tests/            # Tests unitaires (91% coverage)
```

## 🔄 CI/CD

### Workflow GitHub Actions

**Sur PR vers main** :

-   ✅ Tests automatiques
-   ✅ Build Docker (validation)

**Sur merge vers main** :

-   ✅ Tests
-   ✅ Build et push Docker Hub
-   ✅ Déploiement Hugging Face

## 🐳 Docker

```bash
# Build
docker compose build

# Lancer
docker compose up -d

# Logs
docker compose logs -f

# Arrêter
docker compose down
```

## 📝 Commandes Make

```bash
make up          # Démarrer Docker
make down        # Arrêter Docker
make rebuild     # Rebuild complet
make test        # Tests
make coverage    # Tests avec couverture
make ui          # Interface Gradio
make batch       # Batch predictions
make batch-drift # Batch avec drift
make evaluate    # Rapport de drift
make run-local   # Lancement local avec profiling
make precommit   # Pre-commit hooks
```

## 🧪 Tests

-   **59 tests** avec **91% de couverture**
-   Tests unitaires pour API, DB, ML, Services
-   Base de données SQLite en mémoire pour tests
-   Mocks pour psutil et model manager

## 🔧 Optimisations Implémentées

1. **Fast Pandas DataFrame** : Construction optimisée pour inférence
2. **Single Inference Call** : `predict_proba` uniquement (pas de double appel)
3. **Background Tasks** : Logging asynchrone en base de données
4. **Profiling Sélectif** : Uniquement sur endpoints `/api/*`

## 📦 Dépendances Principales

-   FastAPI : Framework web
-   Gradio : Interface utilisateur
-   SQLAlchemy : ORM base de données
-   Scikit-learn : Modèle ML
-   Evidently : Monitoring de drift
-   psutil : Métriques système

## 📄 Licence

Projet éducatif OpenClassrooms.
