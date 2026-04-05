import tensorflow as tf
from tensorflow import keras
import os

model_path = os.path.join("model", "pneumonia_model.h5")
if os.path.isfile(model_path):
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    model.summary()
    
    # Check if first/second layer is rescaling
    for layer in model.layers:
        print(f"Layer: {layer.name}, Config: {layer.get_config()}")
        if "rescaling" in layer.name.lower():
            print("FOUND RESCALING LAYER")
else:
    print("Model file not found")
