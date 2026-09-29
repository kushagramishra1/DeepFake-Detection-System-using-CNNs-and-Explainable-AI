# DeepFake Detection System Using CNNs and Explainable AI

A web application for classifying an image as real or fake with a CNN and visualizing the model's image regions of interest with Grad-CAM.

## Project Structure

- `backend/`: Flask API and image preprocessing/inference code.
- `frontend/`: React interface for uploading images and viewing predictions.
- `deepfake_detection_model.h5`: trained model loaded by the backend.
- `Dataset/`: training, validation, and test image folders.

## Requirements

- Python 3.9 or compatible with the pinned TensorFlow dependencies.
- Node.js and npm.
- The trained model file at the project root: `deepfake_detection_model.h5`.

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

- `POST /predict`: Upload an image using the multipart form field `file`. Supported by the backend: PNG, JPG, JPEG, BMP, and TIFF. Returns the prediction, confidence, Grad-CAM heatmap filename, and explanation.
- `GET /heatmap/<filename>`: Retrieve the generated Grad-CAM overlay.

The frontend currently accepts JPG, JPEG, and PNG uploads.