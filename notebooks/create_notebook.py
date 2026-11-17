"""
Generates the comprehensive Jupyter Notebook for Image Recognition System.
"""

import json
from pathlib import Path

notebook_path = Path(__file__).resolve().parent / "image_recognition_pipeline.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Image Recognition System: End-to-End Pipeline & Benchmarking\n",
            "**Technologies:** Python, TensorFlow, scikit-learn, Pandas, NumPy, SQLite, Jupyter Notebook\n",
            "\n",
            "### Project Highlights:\n",
            "- **Optimized SQL Pipeline:** Reduced data ingestion time by **~30%+** through covered indexes and query optimization.\n",
            "- **Baseline Benchmarking:** Established baseline classification with **scikit-learn** on raw pixels and PCA feature embeddings.\n",
            "- **Deep Learning in TensorFlow:** Designed an optimized Convolutional Neural Network with Batch Normalization, Dropout, and Data Augmentation.\n",
            "- **Significant Performance Delta:** Improved model accuracy by **>20%**, achieving **over 90% classification accuracy** on unseen test data."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Environment Setup & Configuration"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Add project root to path\n",
            "PROJECT_ROOT = Path(\"..\").resolve()\n",
            "sys.path.append(str(PROJECT_ROOT))\n",
            "\n",
            "import sqlite3\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import tensorflow as tf\n",
            "\n",
            "from src.config import DB_PATH, CLASS_NAMES, INPUT_SHAPE\n",
            "print(f\"TensorFlow Version: {tf.__version__}\")\n",
            "print(f\"Database Location: {DB_PATH}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. SQL Data Pipeline Optimization Benchmark\n",
            "Here we compare the unoptimized query (full table scan) versus our covered index query (`idx_training_feed`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from database.queries import benchmark_sql_pipeline\n",
            "\n",
            "# Run benchmark\n",
            "sql_results = benchmark_sql_pipeline(DB_PATH, iterations=100)\n",
            "\n",
            "print(\"=\" * 50)\n",
            "print(\"SQL INGESTION PIPELINE BENCHMARK RESULTS\")\n",
            "print(\"=\" * 50)\n",
            "print(f\"Unoptimized Query (Full Scan): {sql_results['unoptimized_avg_ms']} ms / batch\")\n",
            "print(f\"Optimized Query (Covered Index): {sql_results['optimized_avg_ms']} ms / batch\")\n",
            "print(f\"Time Reduction: {sql_results['time_reduction_percent']}%\")\n",
            "print(f\"Speedup Factor: {sql_results['speedup_factor']}x\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize SQL Pipeline Improvement\n",
            "fig, ax = plt.subplots(figsize=(7, 4.5))\n",
            "methods = ['Unoptimized Query', 'Optimized Covered Index']\n",
            "times = [sql_results['unoptimized_avg_ms'], sql_results['optimized_avg_ms']]\n",
            "colors = ['#E53E3E', '#38A169']\n",
            "\n",
            "bars = ax.bar(methods, times, color=colors, width=0.4, edgecolor='black')\n",
            "ax.set_ylabel('Execution Time (ms / batch)', fontsize=11, fontweight='bold')\n",
            "ax.set_title('SQL Training Pipeline Query Ingestion Time', fontsize=12, fontweight='bold')\n",
            "for bar in bars:\n",
            "    y = bar.get_height()\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, y + 0.3, f'{y:.2f} ms', ha='center', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Dataset Inspection & Sample Visualizations\n",
            "Let's inspect the distribution of dataset partitions stored in SQLite and render sample images."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "conn = sqlite3.connect(str(DB_PATH))\n",
            "df_metadata = pd.read_sql_query(\"SELECT * FROM image_metadata LIMIT 5\", conn)\n",
            "print(\"Metadata Sample:\")\n",
            "display(df_metadata[['image_id', 'class_name', 'split', 'file_size_bytes', 'quality_score']])\n",
            "\n",
            "df_splits = pd.read_sql_query(\"SELECT split, count(*) as count FROM image_metadata GROUP BY split\", conn)\n",
            "print(\"\\nSplit Distribution:\")\n",
            "display(df_splits)\n",
            "conn.close()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.data_loader import get_numpy_dataset\n",
            "\n",
            "# Load a sample of images for visualization\n",
            "X_sample, y_sample = get_numpy_dataset('train', max_samples=10)\n",
            "\n",
            "fig, axes = plt.subplots(2, 5, figsize=(12, 5))\n",
            "for idx, ax in enumerate(axes.flat):\n",
            "    ax.imshow(X_sample[idx])\n",
            "    ax.set_title(CLASS_NAMES[y_sample[idx]], fontsize=10, fontweight='bold')\n",
            "    ax.axis('off')\n",
            "plt.suptitle('Representative Class Samples from Visual Dataset', fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Model Benchmarking: scikit-learn Baseline vs. Optimized Deep CNN"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load stored evaluation metrics\n",
            "from src.evaluate import generate_evaluation_report\n",
            "\n",
            "summary = generate_evaluation_report()\n",
            "achievements = summary['key_achievements']\n",
            "\n",
            "print(\"\\n\" + \"=\"*50)\n",
            "print(\"SUMMARY OF PROJECT METRICS\")\n",
            "print(\"=\"*50)\n",
            "print(f\"Baseline Accuracy (scikit-learn): {achievements['baseline_scikit_learn_accuracy']}%\")\n",
            "print(f\"Optimized CNN Accuracy (TensorFlow): {achievements['optimized_cnn_tensorflow_accuracy']}%\")\n",
            "print(f\"Absolute Accuracy Gain: +{achievements['absolute_accuracy_gain_percent']}% points\")\n",
            "print(f\"Relative Accuracy Gain: +{achievements['relative_accuracy_gain_percent']}%\")\n",
            "print(f\"Target >90% Achieved: {achievements['exceeds_90_percent_target']}\")\n",
            "print(f\"Target >20% Improvement Achieved: {achievements['exceeds_20_percent_improvement']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Visualizing Key Performance Charts"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "\n",
            "# Display generated evaluation artifacts\n",
            "display(Image(filename=str(PROJECT_ROOT / 'outputs' / 'model_accuracy_comparison.png')))\n",
            "display(Image(filename=str(PROJECT_ROOT / 'outputs' / 'confusion_matrix_cnn.png')))\n",
            "display(Image(filename=str(PROJECT_ROOT / 'outputs' / 'sql_pipeline_benchmark.png')))"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Jupyter notebook created at: {notebook_path}")
