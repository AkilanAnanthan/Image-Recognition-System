"""
Baseline Image Classification Models using scikit-learn
Implements benchmark models (Random Forest and Support Vector Machine)
to establish the baseline performance before deep learning architectures.
"""

import os
import joblib
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED, CLASS_NAMES
from src.data_loader import get_numpy_dataset


def train_baseline_models():
    """Trains scikit-learn baseline models on flattened image arrays."""
    print("=" * 60)
    print("1. Loading Data for scikit-learn Baseline Models...")
    print("=" * 60)
    
    X_train, y_train = get_numpy_dataset(split="train")
    X_val, y_val = get_numpy_dataset(split="val")
    X_test, y_test = get_numpy_dataset(split="test")

    # Flatten (N, 32, 32, 3) -> (N, 3072)
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]

    X_train_flat = X_train.reshape(N_train, -1)
    X_val_flat = X_val.reshape(N_val, -1)
    X_test_flat = X_test.reshape(N_test, -1)

    print(f"Data shapes: Train={X_train_flat.shape}, Val={X_val_flat.shape}, Test={X_test_flat.shape}")

    # Optional: Dimensionality Reduction using PCA for feature extraction
    print("\nExtracting PCA visual features (50 components)...")
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    # 1. Initial Un-tuned Baseline Model (Before optimization)
    print("\nTraining Initial Baseline: Un-tuned Decision Tree (max_depth=5)...")
    initial_clf = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)
    initial_clf.fit(X_train_pca, y_train)
    acc_initial = float(accuracy_score(y_test, initial_clf.predict(X_test_pca)))
    print(f"--> Initial Un-tuned Baseline Accuracy: {acc_initial * 100:.2f}%")

    # 2. Iterative Tuning: Tuned Random Forest Classifier (100 estimators) with PCA Pipeline
    print("\nTraining Iteratively Tuned Model: Tuned Random Forest with PCA Pipeline...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
    
    # Bundle into an sklearn Pipeline for production inference
    pipeline = Pipeline([
        ('pca', pca),
        ('classifier', rf_clf)
    ])
    pipeline.fit(X_train_flat, y_train)

    y_pred_tuned = pipeline.predict(X_test_flat)
    acc_tuned = float(accuracy_score(y_test, y_pred_tuned))
    print(f"--> Tuned Model Test Accuracy: {acc_tuned * 100:.2f}%")

    tuning_gain_abs = (acc_tuned - acc_initial) * 100
    tuning_gain_rel = ((acc_tuned - acc_initial) / acc_initial) * 100
    print(f"--> Accuracy Gain via Iterative Tuning: +{tuning_gain_abs:.2f}% points (+{tuning_gain_rel:.2f}% relative)")

    # Save pipeline
    model_path = MODELS_DIR / "baseline_random_forest.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved tuned pipeline to: {model_path}")

    # Save metrics
    report = classification_report(y_test, y_pred_tuned, target_names=CLASS_NAMES, output_dict=True)
    baseline_metrics = {
        "model_type": "scikit-learn Tuned Pipeline (PCA + Random Forest)",
        "initial_untuned_accuracy": round(acc_initial, 4),
        "tuned_test_accuracy": round(acc_tuned, 4),
        "tuning_improvement_abs_percent": round(tuning_gain_abs, 2),
        "tuning_improvement_rel_percent": round(tuning_gain_rel, 2),
        "detailed_report": report
    }

    metrics_file = OUTPUTS_DIR / "baseline_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"Saved baseline metrics to: {metrics_file}")

    return acc_tuned


if __name__ == "__main__":
    train_baseline_models()
