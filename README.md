# Deepfake Detection System

A patent-level deepfake detection system using hybrid AI models.

## Architecture

The system combines:
- **CNN Branch**: EfficientNetB4 for spatial features
- **FFT Branch**: Frequency domain analysis
- **Transformer Branch**: Vision Transformer (ViT)
- **Fusion**: Attention-based feature fusion
- **Explainability**: Grad-CAM + Frequency heatmaps

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Training

1. Prepare datasets in `data/train/real/`, `data/train/fake/`, etc.
2. Run training:
   ```bash
   python models/train.py
   ```

## Inference

```bash
python scripts/inference.py path/to/image.jpg
```

## Backend Services

This repo includes deployment support files for public hosting:

- `wsgi.py` — PythonAnywhere / WSGI entrypoint.
- `Procfile` — Heroku process definition.
- `render.yaml` — Render service configuration.

## Deployment

### Backend deployment

The backend is a Flask API and can be deployed to any Python hosting platform.

#### Option 1: Render
1. Sign in to https://dashboard.render.com/
2. Create a new **Web Service** from GitHub.
3. Connect this repository.
4. Use the `render.yaml` file from this repo.
5. Render will build with:
   ```bash
   pip install -r backend/requirements.txt
   ```
   and start with:
   ```bash
   gunicorn backend.app:app --bind 0.0.0.0:$PORT
   ```

#### Option 2: PythonAnywhere
1. Upload or clone this repo into your PythonAnywhere account.
2. Create a new Web App with Flask and Python 3.10+.
3. Set the WSGI file path to `wsgi.py` in the repo root.
4. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

#### Option 3: Heroku
1. Create a Heroku app.
2. Add a `Procfile` in the repo root:
   ```Procfile
   web: gunicorn backend.app:app
   ```
3. Deploy the repo and set the required buildpacks.

#### Verify backend
Once the backend is deployed, verify it is reachable:
```bash
curl https://<your-backend-domain>/health
```

### Frontend production link

The frontend already uses an environment variable for the backend URL:

- `VITE_API_URL` should be set to your deployed backend URL.
- Example: `https://<your-backend-domain>`

If you deploy the frontend to GitHub Pages, add a GitHub secret named `VITE_API_URL` with that backend URL so the static build can call the correct API.

---

1. Start backend:
   ```bash
   cd backend
   python app.py
   ```

2. Start frontend:
   ```bash
   cd frontend
   npm run dev
   ```

## Features

- Modular architecture
- GPU support
- Explainable AI
- Flask API + React frontend
- Optimized for accuracy with transfer learning, augmentation, focal loss