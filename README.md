# DeepFake Detection System Using CNNs and Explainable AI

A web application for classifying an image as real or fake with a CNN and visualizing influential image regions using occlusion sensitivity.

## Project Structure

- `backend/`: Flask API, ONNX inference model, and image preprocessing/inference code.
- `frontend/`: React interface for uploading images and viewing predictions.
- `deepfake_detection_model.h5`: original trained Keras model used to export the ONNX model.
- `Dataset/`: training, validation, and test image folders.

## Requirements

- Python 3.12 for the backend (ONNX Runtime).
- Node.js and npm.
- The ONNX model file at `backend/deepfake_detection_model.onnx`.

## Run Locally

Start the backend in one terminal:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python backend/app.py
```

The Flask API runs at `http://localhost:5000`.

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm start
```

The React app runs at `http://localhost:3000` and sends image predictions to the Flask API.

## API

- `GET /api/health`: Check that the backend service and ONNX model loaded successfully.
- `POST /api/predict`: Upload an image using the multipart form field `file`. Supported by the backend: PNG, JPG, JPEG, BMP, and TIFF. Returns the prediction, confidence, occlusion-sensitivity overlay data URL, and explanation. Uploads are limited to 4 MiB for serverless compatibility.

In the Vercel multi-service deployment, call backend routes on the same public domain as the frontend using the `/api` prefix.

The frontend currently accepts JPG, JPEG, and PNG uploads.