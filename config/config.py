"""
Configuration settings for MedExplain AI Pro.

This module contains all configuration parameters, paths, and settings
used throughout the application.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Application Information
APP_NAME = "MedExplain AI Pro"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Advanced Personal Health Assistant with AI Analytics"
APP_AUTHOR = "MedExplain AI Team"
APP_CONTACT = "contact@medexplain-ai.example.com"

# Base Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "ml_models"
STATIC_DIR = PROJECT_ROOT / "static"

# Data File Paths
MEDICAL_DATA_FILE = DATA_DIR / "medical_data.json"
USER_PROFILE_FILE = PROJECT_ROOT / "user_profile.json"
HEALTH_HISTORY_FILE = PROJECT_ROOT / "health_history.json"
SYMPTOM_PREDICTOR_MODEL = MODELS_DIR / "symptom_predictor.pkl"
RISK_ASSESSOR_MODEL = MODELS_DIR / "risk_assessor.pkl"

# Static Assets
CSS_DIR = STATIC_DIR / "css"
IMG_DIR = STATIC_DIR / "img"
CSS_FILE = CSS_DIR / "style.css"
LOGO_FILE = IMG_DIR / "logo.png"

# OpenAI API Settings
DEFAULT_AI_MODEL = "gpt-4"
AI_TEMPERATURE = 0.7
MAX_TOKENS = 1500

# Feature Flags
ENABLE_ML_FEATURES = True
ENABLE_AI_CHAT = True
ENABLE_VOICE_INPUT = False
ENABLE_ADVANCED_ANALYTICS = True
ENABLE_PDF_EXPORT = False  # Placeholder for future feature

# UI Settings
DEFAULT_THEME = "light"  # Options: "light", "dark", "system"
DASHBOARD_THEME = "default"  # Options: "default", "plotly", "streamlit", "minimal"
AVAILABLE_LANGUAGES = ["English", "Spanish", "French", "German", "Turkish"]
DEFAULT_LANGUAGE = "English"

# User Data Settings
MAX_HISTORY_ENTRIES = 100  # Maximum number of history entries to keep
ALLOW_ANONYMOUS_USAGE_STATS = False

# Medical Classification Settings
SYMPTOM_CONFIDENCE_THRESHOLD = 30  # Minimum confidence (%) to include a condition
RISK_LEVELS = {
    "low": {
        "threshold": 0,
        "color": "#28a745",
        "description": "Low risk level indicates that your symptoms are generally mild and/or common."
    },
    "moderate": {
        "threshold": 25,
        "color": "#fd7e14",
        "description": "Moderate risk level indicates that your symptoms may require attention from a healthcare provider."
    },
    "high": {
        "threshold": 50,
        "color": "#dc3545",
        "description": "High risk level indicates that your symptoms should be evaluated promptly by a healthcare provider."
    },
    "unknown": {
        "threshold": -1,
        "color": "#6c757d",
        "description": "Risk level could not be determined due to insufficient data."
    }
}

# Disclaimer Text
MEDICAL_DISCLAIMER = """
This application is for educational and informational purposes only. It is not intended to be a substitute
for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or
other qualified health provider with any questions you may have regarding a medical condition.

If you think you may have a medical emergency, call your doctor or emergency services immediately.
MedExplain AI does not recommend or endorse any specific tests, physicians, products, procedures,
opinions, or other information that may be mentioned in the application.
"""

EMERGENCY_DISCLAIMER = """
If you are experiencing a medical emergency, please call emergency services (such as 911 in the US)
or go to the nearest emergency room immediately.
"""

# Logging Configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "medexplain.log"

# Initialize logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
CSS_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

# Load environment variables if .env file exists
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logging.info("Loaded environment variables from .env file")
except ImportError:
    logging.warning("dotenv package not installed. Environment variables not loaded from .env file.")
except Exception as e:
    logging.error(f"Error loading environment variables: {str(e)}")
