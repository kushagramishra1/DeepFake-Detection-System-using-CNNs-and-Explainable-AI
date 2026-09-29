from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import keras
from PIL import Image
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
IMG_SIZE = (128, 128)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "deepfake_detection_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the model
print("📦 Loading model...")
try:
    # Try loading with standalone keras (Keras 3)
    model = keras.saving.load_model(MODEL_PATH)
    print("✅ Model loaded successfully with Keras 3!")
except Exception as e1:
    print(f"Keras 3 load failed: {e1}")
    try:
        # Fallback to tf.keras
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully with tf.keras!")
    except Exception as e2:
        print(f"tf.keras load failed: {e2}")
        raise

# Try to find a valid last conv layer name for Grad-CAM
LAST_CONV_LAYER_NAME = None
for layer in reversed(model.layers):
    if isinstance(layer, (keras.layers.Conv2D, tf.keras.layers.Conv2D)):
        LAST_CONV_LAYER_NAME = layer.name
        break
if LAST_CONV_LAYER_NAME is None:
    LAST_CONV_LAYER_NAME = "conv3"  # fallback
print(f"🔍 Using Grad-CAM layer: {LAST_CONV_LAYER_NAME}")

# Grad-CAM function
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv3"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return np.uint8(255 * heatmap)

def preprocess_image(image_path, target_size=IMG_SIZE):
    """Load and preprocess image using PIL"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{filename}")
        file.save(upload_path)

        try:
            # Process image
            img_array = preprocess_image(upload_path)

            # Predict
            preds = model.predict(img_array, verbose=0)
            pred_value = float(preds.squeeze())
            prediction = "Real" if pred_value > 0.5 else "Fake"

            # Generate Grad-CAM heatmap
            heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME)
            heatmap = cv2.resize(heatmap, IMG_SIZE)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Create overlay
            original = cv2.imread(upload_path)
            original = cv2.resize(original, IMG_SIZE)
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            # Save heatmap image
            heatmap_filename = f"{unique_id}_heatmap.png"
            heatmap_path = os.path.join(OUTPUT_FOLDER, heatmap_filename)
            cv2.imwrite(heatmap_path, overlay)

            # Generate explanation
            if prediction == "Real":
                explanation = (
                    "The model detected this image as REAL. "
                    "Grad-CAM shows strong activation in natural facial areas (eyes, nose, mouth), "
                    "indicating real textures and consistent lighting."
                )
            else:
                explanation = (
                    "The model detected this image as FAKE. "
                    "Grad-CAM highlights unusual patterns or inconsistencies (like blurred patches or unnatural lighting) "
                    "that often appear in deepfakes."
                )

            # Clean up uploaded file
            os.remove(upload_path)

            return jsonify({
                'prediction': prediction,
                'confidence': pred_value,
                'heatmap_path': heatmap_filename,
                'explanation': explanation
            })

        except Exception as e:
            # Clean up on error
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/heatmap/<filename>', methods=['GET'])
def get_heatmap(filename):
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    print(f"Looking for file: {filepath}")  # Debug log
    if os.path.exists(filepath):
        print(f"File found: {filepath}")  # Debug log
        return send_file(filepath, mimetype='image/png')
    print(f"File not found: {filepath}")  # Debug log
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

