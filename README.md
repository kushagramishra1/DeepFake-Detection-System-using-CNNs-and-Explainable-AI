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

## Deployment

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