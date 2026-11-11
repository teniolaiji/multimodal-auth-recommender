"""Shared paths and settings. Every script imports from here; don't hard-code paths elsewhere.

Tabular section added by the tabular lead. Image/audio/integration leads: add your
own sections below rather than editing these names (other scripts depend on them).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
UNAUTHORIZED_DIR = DATA_DIR / "unauthorized"
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42

# ---------------------------------------------------------------- tabular
SOCIAL_RAW = RAW_DIR / "customer_social_profiles.xlsx"
TRANSACTIONS_RAW = RAW_DIR / "customer_transactions.xlsx"
MERGED_CSV = PROCESSED_DIR / "merged_customer_data.csv"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_model.joblib"
PRODUCT_METRICS_PATH = MODELS_DIR / "product_model_metrics.json"
TARGET = "product_category"

# Which customer profile each team member "is" in the demo. The face/voice models
# identify a member; the app looks up this customer_id and asks the product model.
# Any customer_id present in merged_customer_data.csv works. Integration lead can change these.
MEMBER_TO_CUSTOMER = {
    "kethia": 101,
    "tabitha": 114,
    "towi": 152,
    "harmony": 188,
}
