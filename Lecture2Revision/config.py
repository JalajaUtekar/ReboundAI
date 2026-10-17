"""
Lecture To Revision - Configuration Module
Centralizes paths, environment variables, hyperparameters, and application defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Directory configurations
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / os.getenv("UPLOAD_DIR", "uploads")
MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
DB_PATH = PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/lecture2revision.db")

# Ensure critical directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Transformer / LLM Backend Configuration
TRANSFORMER_BACKEND = os.getenv("TRANSFORMER_BACKEND", "offline_extractive").lower().strip()
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Adaptive Engine & SM-2 Spaced Repetition Defaults
SM2_DEFAULT_EASE_FACTOR = float(os.getenv("SM2_DEFAULT_EASE_FACTOR", "2.5"))
SM2_MIN_EASE_FACTOR = float(os.getenv("SM2_MIN_EASE_FACTOR", "1.3"))

# LSTM Neural Model Parameters
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "5"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "25"))
LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "16"))
LSTM_MODEL_PATH = MODEL_DIR / "lstm_student_predictor.keras"
LSTM_SCALER_PATH = MODEL_DIR / "lstm_feature_scaler.pkl"
LSTM_METRICS_PATH = MODEL_DIR / "lstm_metrics.json"

# UN SDG 4 Information
SDG_TITLE = "UN Sustainable Development Goal 4: Quality Education"
SDG_STATEMENT = (
    "Lecture To Revision targets SDG 4 (Ensure inclusive and equitable quality "
    "education and promote lifelong learning opportunities for all). By converting complex "
    "lecture transcripts into adaptive, bite-sized revision schedules grounded strictly "
    "in primary course texts, the platform empowers students with self-paced, personalized mastery "
    "without fabricating educational outcomes."
)
