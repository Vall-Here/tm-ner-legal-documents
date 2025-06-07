"""
Configuration settings for Indonesian Legal NER Deployment
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "Code" / "main" / "models" / "model_final"

# Model configuration
MODEL_CONFIG = {
    "model_path": MODEL_DIR / "model_artifacts",
    "config_path": MODEL_DIR,
    "max_length": 512,
    "max_text_length": 5000,  # Maximum text length for processing
    "batch_size": 1,
    "device": "auto",  # auto, cpu, cuda
}

# UI configuration
UI_CONFIG = {
    "title": "Indonesian Legal NER",
    "description": "Extract named entities from Indonesian legal documents",
    "theme": "soft",  # default, soft, monochrome, etc.
    "share": True,
    "server_port": 7860,
    "server_name": "0.0.0.0",
}

# File processing limits
FILE_CONFIG = {
    "max_pdf_size": 10 * 1024 * 1024,  # 10MB
    "supported_formats": [".pdf"],
    "max_pages": 50,
}

# Entity display configuration
ENTITY_CONFIG = {
    "max_entities_display": 100,
    "entity_colors": {
        "PERSON": "#FFE6E6",
        "ORG": "#E6F3FF", 
        "LOC": "#E6FFE6",
        "MISC": "#FFF3E6",
        "DATE": "#F0E6FF",
        "MONEY": "#FFFFE6",
    },
    "confidence_threshold": 0.5,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "app.log",
}
