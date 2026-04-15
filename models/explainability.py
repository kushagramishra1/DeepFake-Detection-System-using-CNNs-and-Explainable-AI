import tensorflow as tf
import numpy as np
import cv2
from model import create_hybrid_model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_frequency_heatmap(image):
    # Assume image is (224, 224, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift) + 1e-8)
    # Normalize
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
    return magnitude_spectrum

def fuse_heatmaps(gradcam_heatmap, freq_heatmap, method='average'):
    # Resize freq_heatmap to match gradcam
    freq_resized = cv2.resize(freq_heatmap, (gradcam_heatmap.shape[1], gradcam_heatmap.shape[0]))
    if method == 'average':
        fused = (gradcam_heatmap + freq_resized) / 2
    elif method == 'max':
        fused = np.maximum(gradcam_heatmap, freq_resized)
    return fused

def explain_prediction(image_path, model_path='models/deepfake_detector.h5'):
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    preds = model.predict(img_array)
    confidence = preds[0][0]
    prediction = 'Fake' if confidence < 0.5 else 'Real'

    # Grad-CAM for CNN branch
    # Assuming EfficientNetB4 last conv layer is 'top_conv'
    gradcam_heatmap = make_gradcam_heatmap(img_array, model, 'efficientnet-b4/top_conv')

    # Frequency heatmap
    freq_heatmap = generate_frequency_heatmap(img_array[0])

    # Fuse
    fused_heatmap = fuse_heatmaps(gradcam_heatmap, freq_heatmap)

    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'heatmap': fused_heatmap
    }