"""Data drift evaluation script using Evidently.

Compares initial training data with production predictions to detect data drift.
"""

import logging
import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = os.getenv("API_URL", "https://xaviercoulon-rugbymlops.hf.space/api/v1")
API_KEY = os.getenv("API_KEY", "default-key-change-me")
DATA_FILE = "data/kicks_ready_for_model.csv"
OUTPUT_DIR = "data/drift_reports"

# Validate API configuration
if not API_KEY or API_KEY == "default-key-change-me":
    logger.error("❌ API_KEY not defined or using default value.")
    logger.error("Please set API_KEY in your .env file before running this script.")
    raise ValueError("API_KEY is required for production data access")

logger.info(f"✅ Configuration loaded: API_URL={API_URL}")

# Feature columns (exclude target and prediction)
FEATURE_COLUMNS = [
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


def load_reference_data(file_path: str) -> pd.DataFrame:
    """Load reference (training) data from CSV.

    Args:
        file_path: Path to the training data CSV

    Returns:
        DataFrame with reference data
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ Reference data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"📂 Loading reference data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"✅ Loaded {len(df)} rows from reference data")

    return df[FEATURE_COLUMNS + ["resultat"]]


def fetch_production_data(
    api_url: str, api_key: str, limit: int = 1000
) -> pd.DataFrame:
    """Fetch production predictions from API (Neon database via Hugging Face).

    Args:
        api_url: Base API URL (e.g., Hugging Face Space endpoint)
        api_key: API key for authentication
        limit: Maximum number of records to fetch

    Returns:
        DataFrame with production data

    Raises:
        requests.RequestException: If API call fails
        ValueError: If response is invalid
    """
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    predictions_url = f"{api_url}/predictions"
    logger.info(f"🌐 Fetching production predictions from {predictions_url}...")

    try:
        response = requests.get(
            predictions_url,
            headers=headers,
            params={"limit": limit, "skip": 0},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        if not data:
            logger.warning("⚠️ No prediction records found in production database")
            return pd.DataFrame()

        logger.info(f"✅ Fetched {len(data)} prediction records from API")

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Keep only feature columns + prediction
        columns_to_keep = [col for col in FEATURE_COLUMNS if col in df.columns]
        if "prediction" in df.columns:
            columns_to_keep.append("prediction")

        return df[columns_to_keep]

    except requests.exceptions.Timeout:
        logger.error(f"❌ Timeout connecting to {predictions_url}")
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Connection error to {predictions_url}: {e}")
        raise
    except requests.exceptions.HTTPError:
        logger.error(f"❌ HTTP error {response.status_code}: {response.text}")
        raise
    except (ValueError, KeyError) as e:
        logger.error(f"❌ Invalid API response format: {e}")
        raise


def prepare_data_for_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare and align data for drift detection.

    Args:
        reference_df: Reference (training) data
        production_df: Production prediction data

    Returns:
        Tuple of (reference_data, production_data) aligned for comparison
    """
    logger.info("🔄 Preparing data for drift analysis...")

    # Align feature columns
    common_features = [col for col in FEATURE_COLUMNS if col in production_df.columns]

    reference_subset = reference_df[common_features].copy()
    production_subset = production_df[common_features].copy()

    logger.info(
        f"📊 Reference data shape: {reference_subset.shape}, "
        f"Production data shape: {production_subset.shape}"
    )

    return reference_subset, production_subset


def evaluate_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    output_dir: str = "data/drift_reports",
):
    """Evaluate data drift using Evidently.

    Args:
        reference_df: Reference data from training
        production_df: Production data from API
        output_dir: Directory to save drift report

    Returns:
        Dictionary with drift evaluation results
    """
    logger.info("🔍 Running Evidently drift detection...")

    # Create drift report with 50% drift threshold
    # (drift detected if 50% of features are drifting)
    report = Report(metrics=[DataDriftPreset(drift_share=0.5)])

    # Run analysis
    eval = report.run(
        reference_data=reference_df,
        current_data=production_df,
    )

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")

    eval.save_html(report_path)
    logger.info(f"✅ Report saved to {report_path}")


def main():
    """Main execution function."""
    try:
        # Step 1: Load reference data
        logger.info("=" * 50)
        logger.info("🏈 RUGBY KICK DRIFT ANALYSIS")
        logger.info("=" * 50)
        reference_data = load_reference_data(DATA_FILE)

        # Step 2: Fetch production data from Neon (via API)
        logger.info("\n📡 Connecting to production database...")
        production_data = fetch_production_data(API_URL, API_KEY)

        if production_data.empty:
            logger.warning("⚠️ No production data available for drift analysis")
            return None

        # Step 3: Prepare data
        ref_prepared, prod_prepared = prepare_data_for_drift(
            reference_data, production_data
        )

        # Step 4: Run drift detection
        drift_results = evaluate_drift(ref_prepared, prod_prepared, OUTPUT_DIR)

        # Step 5: Display summary
        logger.info("\n" + "=" * 50)
        logger.info("📋 DRIFT ANALYSIS SUMMARY")
        logger.info(f"Report saved at: {OUTPUT_DIR}")
        logger.info("=" * 50 + "\n")

        return drift_results

    except FileNotFoundError as e:
        logger.error(f"❌ File error: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error during drift evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
