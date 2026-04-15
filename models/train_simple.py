import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model_simple import create_simplified_model

def create_synthetic_dataset(num_samples=100, batch_size=32, image_size=(224, 224)):
    """Create synthetic dataset for demo/testing"""
    real_images = np.random.rand(num_samples // 2, *image_size, 3).astype(np.float32)
    real_labels = np.ones((num_samples // 2, 1))
    
    fake_images = np.random.rand(num_samples // 2, *image_size, 3).astype(np.float32)
    fake_images += np.random.normal(0.1, 0.05, fake_images.shape)
    fake_images = np.clip(fake_images, 0, 1)
    fake_labels = np.zeros((num_samples // 2, 1))
    
    all_images = np.vstack([real_images, fake_images])
    all_labels = np.vstack([real_labels, fake_labels])
    
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    model_save_path = os.path.join(Path(__file__).parent, 'deepfake_detector.h5')
    
    print("[INFO] Creating simplified hybrid model...")
    model = create_simplified_model()
    
    print("[INFO] Compiling model...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("[INFO] Model summary:")
    model.summary()
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        model_save_path, 
        save_best_only=True, 
        monitor='val_accuracy',
        verbose=1
    )
    
    print("[INFO] Creating training data...")
    train_dataset = create_synthetic_dataset(num_samples=200, batch_size=16)
    val_dataset = create_synthetic_dataset(num_samples=50, batch_size=16)
    
    print("[INFO] Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    print(f"[INFO] Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("[SUCCESS] Model trained and saved!")
    
    return model, history

if __name__ == "__main__":
    model, history = main()