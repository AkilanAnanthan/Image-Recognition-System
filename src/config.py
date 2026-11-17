"""
Configuration module for Image Recognition System.
Defines project paths, dataset parameters, model hyperparameters, and class labels.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "image_metadata.db"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = DATA_DIR / "images"

# Ensure essential directories exist
for directory in [DATA_DIR, DATABASE_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Classes (10 balanced categories)
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Image Dimensions
IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# SQL Pipeline Benchmarking
BENCHMARK_ITERATIONS = 100
BENCHMARK_BATCH_SIZE = 128
