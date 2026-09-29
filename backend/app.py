from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

# Configuration
IMG_SIZE = (128, 128)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detection_model.h5")

# Load the model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# Try to find a valid last conv layer name for Grad-CAM
LAST_CONV_LAYER_NAME = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
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

def preprocess_image(image, target_size=IMG_SIZE):
    """Convert an uploaded image to a normalized model input array."""
    img = Image.open(image).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            image_bytes = file.read()
            img_array = preprocess_image(BytesIO(image_bytes))

            # Predict
            preds = model.predict(img_array, verbose=0)
            pred_value = float(preds.squeeze())
            prediction = "Real" if pred_value > 0.5 else "Fake"

            # Generate Grad-CAM heatmap
            heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME)
            heatmap = cv2.resize(heatmap, IMG_SIZE)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Create overlay
            original = cv2.cvtColor(
                np.array(Image.open(BytesIO(image_bytes)).convert('RGB').resize(IMG_SIZE)),
                cv2.COLOR_RGB2BGR,
            )
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            image_buffer = BytesIO()
            Image.fromarray(overlay_rgb).save(image_buffer, format='PNG')
            heatmap_data_url = 'data:image/png;base64,' + base64.b64encode(image_buffer.getvalue()).decode('ascii')

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

            return jsonify({
                'prediction': prediction,
                'confidence': pred_value,
                'heatmap_image': heatmap_data_url,
                'explanation': explanation
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({'error': 'Image uploads must be 4 MiB or smaller.'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

