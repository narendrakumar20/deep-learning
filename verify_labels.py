import tensorflow as tf
import os

DATASET_DIR = "dataset"
train_dir = os.path.join(DATASET_DIR, "train")

if os.path.isdir(train_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(150, 150),
        batch_size=32,
    )
    print(f"Indices: {train_ds.class_names}")
    # NORMAL should be index 0, PNEUMONIA should be index 1
else:
    print("Dataset not found")
