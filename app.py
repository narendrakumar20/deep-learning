"""
Flask web application for pneumonia detection from chest X-ray images.
Upload an X-ray image and get a prediction with confidence score.
"""

import os
import uuid
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = (150, 150)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.h5")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

# Load model at startup
model = None


def load_model():
    global model
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.isfile(MODEL_PATH):
        app.logger.info("Loading model from %s", MODEL_PATH)
        print(f"📦 Loading model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # Verify classes from dataset if it exists
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "train")
        if os.path.isdir(dataset_path):
            temp_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_path, 
                image_size=IMG_SIZE,
                batch_size=1
            )
            print(f"🔍 Dataset Classes: {temp_ds.class_names}")
            # If temp_ds.class_names[0] is PNEUMONIA, indices are flipped!
    else:
        app.logger.warning("Model not found at %s", MODEL_PATH)
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("   Run 'python train.py' first to train the model.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """Run inference on a single image and return labels with logging."""
    # Use TensorFlow for resizing to match training pipeline exactly
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img_array = tf.expand_dims(img, axis=0)  # Add batch dimension (Rescaling layer handles 1/255)

    prediction_raw = model.predict(img_array, verbose=0)[0][0]
    print(f"🧪 Prediction Raw Score: {prediction_raw:.4f}")

    if prediction_raw >= 0.5:
        label = "PNEUMONIA POSITIVE"
    else:
        label = "PNEUMONIA NEGATIVE"

    print(f"🔬 Diagnosis: {label}")
    return label


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, BMP, or WEBP."}), 400

    # Save uploaded file
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Predict
    label = predict_image(filepath)

    return jsonify({
        "label": label,
        "image_url": f"/static/uploads/{filename}",
    })


if __name__ == "__main__":
    load_model()
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
