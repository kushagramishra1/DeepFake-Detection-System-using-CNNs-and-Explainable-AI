import base64
import os
from io import BytesIO

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

IMG_SIZE = (128, 128)
GRID_SIZE = 8
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detection_model.onnx")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image, image_array[np.newaxis, ...]


def get_occlusion_heatmap(image, image_array, original_score):
    patch_size = IMG_SIZE[0] // GRID_SIZE
    occluded_images = np.repeat(image_array, GRID_SIZE * GRID_SIZE, axis=0)
    fill_color = image_array.mean(axis=(1, 2), keepdims=True)[0, 0, 0]

    patch_index = 0
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            top = row * patch_size
            left = column * patch_size
            occluded_images[patch_index, top:top + patch_size, left:left + patch_size] = fill_color
            patch_index += 1

    occluded_scores = session.run([OUTPUT_NAME], {INPUT_NAME: occluded_images})[0].reshape(GRID_SIZE, GRID_SIZE)
    heatmap = np.abs(original_score - occluded_scores)
    maximum = float(heatmap.max())
    if maximum > 0:
        heatmap /= maximum

    heatmap_image = Image.fromarray(np.uint8(heatmap * 255), mode="L").resize(
        IMG_SIZE, Image.Resampling.BILINEAR
    )
    heat = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    colors = np.stack(
        [
            np.clip(1.5 * heat, 0, 1),
            np.clip(1.5 - np.abs(2 * heat - 1) * 1.5, 0, 1),
            np.clip(1.5 * (1 - heat), 0, 1),
        ],
        axis=-1,
    )
    original = np.asarray(image, dtype=np.float32) / 255.0
    overlay = np.uint8(np.clip((original * 0.6 + colors * 0.4) * 255, 0, 255))

    image_buffer = BytesIO()
    Image.fromarray(overlay).save(image_buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    uploaded_file = request.files['file']
    if not uploaded_file.filename:
        return jsonify({'error': 'No file selected'}), 400
    if not uploaded_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        image_bytes = uploaded_file.read()
        image, image_array = preprocess_image(image_bytes)
        prediction_score = float(session.run([OUTPUT_NAME], {INPUT_NAME: image_array})[0].squeeze())
        prediction = "Real" if prediction_score > 0.5 else "Fake"
        heatmap_image = get_occlusion_heatmap(image, image_array, prediction_score)
        explanation = (
            "The model detected this image as REAL. "
            "The overlay highlights regions that most influence the model's prediction."
            if prediction == "Real"
            else "The model detected this image as FAKE. "
            "The overlay highlights regions that most influence the model's prediction."
        )

        return jsonify({
            'prediction': prediction,
            'confidence': prediction_score,
            'heatmap_image': heatmap_image,
            'explanation': explanation,
        })
    except Exception as error:
        app.logger.exception("Image prediction failed")
        return jsonify({'error': str(error)}), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({'error': 'Image uploads must be 4 MiB or smaller.'}), 413


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

