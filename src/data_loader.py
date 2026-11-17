"""
Data Ingestion Pipeline
Queries SQLite database for image metadata and builds high-throughput tf.data pipelines.
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Tuple, List
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DB_PATH, INPUT_SHAPE, BATCH_SIZE


def fetch_metadata_from_db(split: str = "train", min_quality: float = 0.8) -> List[Tuple[str, int]]:
    """
    Executes optimized index query to retrieve filepaths and class labels from SQLite.
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    query = """
        SELECT filepath, class_id
        FROM image_metadata INDEXED BY idx_training_feed
        WHERE split = ? AND quality_score >= ?;
    """
    cursor.execute(query, (split, min_quality))
    rows = cursor.fetchall()
    conn.close()
    return rows


def load_image_and_preprocess(filepath: str, label: int) -> Tuple[np.ndarray, int]:
    """Loads a PNG image from disk and normalizes pixel values to [0.0, 1.0]."""
    img = Image.open(filepath).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, label


def get_numpy_dataset(split: str = "train", max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns image tensors and label arrays as NumPy arrays for scikit-learn baseline models.
    """
    records = fetch_metadata_from_db(split=split)
    if max_samples:
        records = records[:max_samples]

    images = []
    labels = []
    for filepath, class_id in records:
        if Path(filepath).exists():
            img_arr, _ = load_image_and_preprocess(filepath, class_id)
            images.append(img_arr)
            labels.append(class_id)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_tf_dataset(
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = False
):
    """
    Constructs an optimized TensorFlow tf.data.Dataset pipeline:
    - Parallel disk I/O loading
    - In-memory caching
    - Prefetching for GPU/CPU pipeline overlap
    """
    import tensorflow as tf

    records = fetch_metadata_from_db(split=split)
    filepaths = [r[0] for r in records if Path(r[0]).exists()]
    labels = [r[1] for r in records if Path(r[0]).exists()]

    def parse_image_tf(filename, label):
        image_bytes = tf.io.read_file(filename)
        image = tf.io.decode_png(image_bytes, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

    dataset = dataset.map(parse_image_tf, num_parallel_calls=tf.data.AUTOTUNE)

    # Optional augmentation for training split
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ], name="data_augmentation")
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    X_train, y_train = get_numpy_dataset("train", max_samples=100)
    print(f"Loaded NumPy train batch: X={X_train.shape}, y={y_train.shape}")
