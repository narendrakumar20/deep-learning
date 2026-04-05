"""
Train a CNN model to classify chest X-ray images as Normal or Pneumonia.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "model", "pneumonia_model.h5")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "static", "plots")


def build_model():
    """Build the CNN model with data augmentation layers."""

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="data_augmentation")

    model = keras.Sequential([
        # Input
        layers.Input(shape=(*IMG_SIZE, 3)),

        # Data augmentation (only active during training)
        data_augmentation,

        # Rescale pixel values to [0, 1]
        layers.Rescaling(1.0 / 255),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="pneumonia_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_class_names(split="train"):
    """Get sorted class names from directory listing (TF 2.20+ compatible)."""
    split_dir = os.path.join(DATASET_DIR, split)
    return sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )


def load_datasets():
    """Load train, validation, and test datasets from directory structure."""

    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=42,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
        seed=42,
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def plot_training_history(history):
    """Save training/validation accuracy and loss plots."""

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n📊 Training history plot saved to: {plot_path}")


def main():
    print("=" * 60)
    print("  Pneumonia Detection CNN — Training")
    print("=" * 60)

    # Check dataset exists
    if not os.path.isdir(DATASET_DIR):
        print("\n[ERROR] Dataset not found. Run 'python download_dataset.py' first.")
        return

    # Load data
    print("\n📂 Loading datasets...")
    train_ds, val_ds = load_datasets()

    # Print class names
    class_names = get_class_names("train")
    print(f"   Classes: {class_names}")

    # Build model
    print("\n🏗️  Building CNN model...")
    model = build_model()
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Train
    print(f"\n🚀 Starting training for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_list,
    )

    # Plot
    plot_training_history(history)

    print(f"\n✅ Best model saved to: {MODEL_SAVE_PATH}")
    print("   Training complete!")


if __name__ == "__main__":
    main()
