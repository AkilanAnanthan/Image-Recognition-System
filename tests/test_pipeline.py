"""
Unit and Integration Tests for Image Recognition System
Verifies database integrity, SQL query indexes, data loading, and model architectures.
Built with Python's standard unittest framework for zero-dependency test execution.
"""

import unittest
import sqlite3
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DB_PATH, INPUT_SHAPE, NUM_CLASSES, CLASS_NAMES
from database.queries import query_optimized_covered_index, query_unoptimized_full_scan, benchmark_sql_pipeline
from src.data_loader import fetch_metadata_from_db, load_image_and_preprocess
from src.cnn_model import build_cnn_model


class TestImageRecognitionPipeline(unittest.TestCase):

    def test_database_exists_and_populated(self):
        """Verify SQLite database exists and has required tables and entries."""
        self.assertTrue(DB_PATH.exists(), f"Database not found at {DB_PATH}")
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("SELECT count(*) FROM image_metadata;")
        total = cursor.fetchone()[0]
        self.assertGreater(total, 0, "image_metadata table is empty")

        cursor.execute("SELECT DISTINCT split FROM image_metadata;")
        splits = {r[0] for r in cursor.fetchall()}
        self.assertEqual(splits, {"train", "val", "test"})
        conn.close()

    def test_sql_indexes_exist(self):
        """Verify performance indexes are created on image_metadata table."""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indexes = {r[0] for r in cursor.fetchall()}
        conn.close()

        self.assertIn("idx_split_quality", indexes)
        self.assertIn("idx_split_class", indexes)
        self.assertIn("idx_training_feed", indexes)

    def test_sql_query_optimization(self):
        """Verify optimized covered query returns correct columns and data."""
        conn = sqlite3.connect(str(DB_PATH))
        rows_opt = query_optimized_covered_index(conn, split="train", min_quality=0.8)
        conn.close()

        self.assertGreater(len(rows_opt), 0, "No records returned by optimized query")
        # Each row should be (filepath, class_id)
        self.assertEqual(len(rows_opt[0]), 2)
        self.assertIsInstance(rows_opt[0][0], str)
        self.assertIsInstance(rows_opt[0][1], int)

    def test_sql_pipeline_benchmark_metric(self):
        """Verify SQL benchmark runs and verifies positive query speedup."""
        metrics = benchmark_sql_pipeline(DB_PATH, iterations=10)
        self.assertIn("time_reduction_percent", metrics)
        self.assertIn("speedup_factor", metrics)
        self.assertGreater(metrics["time_reduction_percent"], 0)
        self.assertGreater(metrics["speedup_factor"], 1.0)

    def test_image_preprocessing(self):
        """Verify image loading produces normalized tensor with correct dimensions."""
        records = fetch_metadata_from_db("train")
        self.assertGreater(len(records), 0)
        filepath, label = records[0]

        img_tensor, ret_label = load_image_and_preprocess(filepath, label)
        self.assertEqual(img_tensor.shape, INPUT_SHAPE)
        self.assertEqual(img_tensor.dtype, np.float32)
        self.assertTrue(0.0 <= img_tensor.min() and img_tensor.max() <= 1.0)
        self.assertEqual(ret_label, label)

    def test_cnn_architecture_forward_pass(self):
        """Verify TensorFlow CNN model instantiates and executes forward pass with dummy tensor."""
        import tensorflow as tf
        model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
        
        dummy_input = tf.random.uniform((2, 32, 32, 3))
        predictions = model(dummy_input)

        self.assertEqual(predictions.shape, (2, NUM_CLASSES))
        sums = tf.reduce_sum(predictions, axis=-1).numpy()
        np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
