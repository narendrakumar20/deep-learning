"""
Evaluate the trained pneumonia detection model on the test set.
Produces accuracy, classification report, confusion matrix, and sample predictions.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "pneumonia_model.h5")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "static", "plots")


def get_class_names(split="test"):
    """Get sorted class names from directory listing (TF 2.20+ compatible)."""
    split_dir = os.path.join(DATASET_DIR, split)
    return sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )


def main():
    print("=" * 60)
    print("  Pneumonia Detection CNN — Evaluation")
    print("=" * 60)

    # Load model
    if not os.path.isfile(MODEL_PATH):
        print(f"\n[ERROR] Model not found at {MODEL_PATH}")
        print("Run 'python train.py' first to train the model.")
        return

    print("\n📦 Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)

    # Load test dataset
    print("📂 Loading test dataset...")
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    class_names = get_class_names("test")
    print(f"   Classes: {class_names}")

    # Collect all labels and predictions
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'─' * 40}")
    print(f"  Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"{'─' * 40}")

    print("\n📋 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    with open("eval_results.txt", "w") as f:
        f.write("Pneumonia Detection Evaluation Results\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("True Label", fontsize=13)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n📊 Confusion matrix saved to: {cm_path}")

    # ── Sample Predictions ───────────────────────────────────────────────────
    # Take first batch for visualization
    sample_ds = test_ds.take(1)
    for images, labels in sample_ds:
        preds = model.predict(images, verbose=0)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(min(8, len(images))):
            ax = axes[i]
            img = images[i].numpy().astype("uint8")
            ax.imshow(img)
            ax.axis("off")

            true_label = class_names[int(labels[i].numpy())]
            pred_label = class_names[int(preds[i] >= 0.5)]
            confidence = preds[i][0] if preds[i] >= 0.5 else 1 - preds[i][0]

            color = "green" if true_label == pred_label else "red"
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})",
                fontsize=10, fontweight="bold", color=color,
            )

        plt.suptitle("Sample Predictions on Test Set", fontsize=14, fontweight="bold")
        plt.tight_layout()
        samples_path = os.path.join(PLOTS_DIR, "sample_predictions.png")
        plt.savefig(samples_path, dpi=150)
        plt.close()
        print(f"📊 Sample predictions saved to: {samples_path}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
