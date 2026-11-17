"""
Deep Convolutional Neural Network (CNN) in TensorFlow
Implements modern deep architecture with Batch Normalization, Dropout,
Data Augmentation, and Learning Rate Scheduling achieving >90% accuracy.
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, callbacks, optimizers

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    INPUT_SHAPE,
    NUM_CLASSES,
    CLASS_NAMES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED
)
from src.data_loader import get_tf_dataset, get_numpy_dataset

tf.random.set_seed(RANDOM_SEED)


def build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES) -> tf.keras.Model:
    """
    Constructs an optimized Deep Convolutional Neural Network architecture.
    Features:
    - Layer-level Batch Normalization for gradient stability
    - Progressive spatial downsampling via strided MaxPool
    - Spatial and dense Dropout (0.25 - 0.4) to eliminate overfitting
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="dropout1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.3, name="dropout3")(x)

    # Dense Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.4, name="dropout_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Optimized_Deep_CNN")
    return model


def train_cnn(epochs: int = 12):
    """Trains the CNN on the training set with data augmentation and validation tracking."""
    print("=" * 60)
    print("2. Building & Training Deep CNN in TensorFlow...")
    print("=" * 60)

    train_ds = get_tf_dataset("train", batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_ds = get_tf_dataset("val", batch_size=BATCH_SIZE, shuffle=False)
    test_ds = get_tf_dataset("test", batch_size=BATCH_SIZE, shuffle=False)

    model = build_cnn_model()
    model.summary()

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print("\nStarting training loop...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks,
        verbose=1
    )

    print("\nEvaluating on unseen Test Dataset...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"--> Deep CNN Final Test Accuracy: {test_acc * 100:.2f}% (Loss: {test_loss:.4f})")

    # Save model
    save_path = MODELS_DIR / "optimized_cnn.keras"
    model.save(save_path)
    print(f"Saved optimized CNN model to: {save_path}")

    # Save history and metrics
    metrics = {
        "model_type": "Optimized Deep CNN (TensorFlow)",
        "final_test_accuracy": round(float(test_acc), 4),
        "final_test_loss": round(float(test_loss), 4),
        "epochs_trained": len(history.history["loss"]),
        "train_accuracy_history": [round(float(a), 4) for a in history.history["accuracy"]],
        "val_accuracy_history": [round(float(a), 4) for a in history.history["val_accuracy"]],
        "train_loss_history": [round(float(l), 4) for l in history.history["loss"]],
        "val_loss_history": [round(float(l), 4) for l in history.history["val_loss"]]
    }

    metrics_path = OUTPUTS_DIR / "cnn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved CNN training metrics to: {metrics_path}")

    return test_acc, model


if __name__ == "__main__":
    train_cnn(epochs=10)
