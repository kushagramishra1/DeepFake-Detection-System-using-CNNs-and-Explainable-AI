from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import io
import os
import sys
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Load model - handle if it doesn't exist yet
model = None
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'deepfake_detector.h5')

print(f"[INFO] Looking for model at: {model_path}")
print(f"[INFO] Model exists: {os.path.exists(model_path)}")

try:
    if os.path.exists(model_path):
        print(f"[INFO] Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("[SUCCESS] Model loaded successfully!")
    else:
        print(f"[WARNING] Model not found at {model_path}. Running in demo mode.")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}. Running in demo mode.")
    import traceback
    traceback.print_exc()

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    print(f"[INFO] Predict request received. Files: {request.files.keys()}")
    
    if 'image' not in request.files:
        print("[ERROR] No image in request")
        return jsonify({'error': 'No image provided', 'status': 'error'}), 400

    try:
        file = request.files['image']
        print(f"[INFO] File received: {file.filename}, size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer
        
        # Load and process image
        img = Image.open(io.BytesIO(file.read()))
        print(f"[INFO] Image loaded: size={img.size}, mode={img.mode}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"[INFO] Image array shape: {img_array.shape}")

        if model is None:
            print("[INFO] Using demo mode prediction")
            confidence = np.random.rand()
        else:
            print("[INFO] Running model prediction...")
            preds = model.predict(img_array, verbose=0)
            confidence = float(preds[0][0])
            print(f"[INFO] Model output: {confidence}")

        prediction = 'Fake' if confidence < 0.5 else 'Real'
        
        result = {
            'prediction': prediction,
            'confidence': float(confidence),
            'status': 'success'
        }
        print(f"[SUCCESS] Prediction: {result}")
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 200
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    }), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Deepfake Detection API',
        'version': '1.0',
        'endpoints': {
            'POST /predict': 'Predict if image is fake',
            'GET /health': 'Health check'
        },
        'model_loaded': model is not None
    }), 200

if __name__ == '__main__':
    print("[INFO] Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)