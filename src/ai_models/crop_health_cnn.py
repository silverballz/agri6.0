"""
CNN model for crop health classification.

This module implements the CropHealthCNN class as specified in the design document
for spatial analysis of multispectral satellite imagery.
"""

import numpy as np
from typing import Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)


class CropHealthCNN:
    """
    CNN model for crop health classification using multispectral patches.
    
    This class implements a U-Net style architecture for pixel-level
    classification of crop health conditions using 4-band Sentinel-2A data
    (B02-Blue, B03-Green, B04-Red, B08-NIR).
    
    Classes:
        0: healthy
        1: stressed
        2: diseased
        3: pest
    """
    
    def __init__(self):
        """Initialize CNN model."""
        self.model = None
        self.is_trained = False
        self.model_version = "1.0.0"
        self._build_model()
        
    def _build_model(self):
        """
        Create CNN model architecture with U-Net style.
        
        Architecture:
        - Input: (64, 64, 4) - 4 spectral bands
        - Encoder: 4 convolutional blocks with batch normalization
        - Decoder: 4 upsampling blocks with skip connections
        - Output: (64, 64, 4) - 4 class probabilities per pixel
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            inputs = keras.Input(shape=(64, 64, 4))
            
            # Encoder (downsampling path)
            # Block 1
            conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            conv1 = layers.BatchNormalization()(conv1)
            conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
            conv1 = layers.BatchNormalization()(conv1)
            pool1 = layers.MaxPooling2D(2)(conv1)
            
            # Block 2
            conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
            conv2 = layers.BatchNormalization()(conv2)
            conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
            conv2 = layers.BatchNormalization()(conv2)
            pool2 = layers.MaxPooling2D(2)(conv2)
            
            # Block 3
            conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
            conv3 = layers.BatchNormalization()(conv3)
            conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
            conv3 = layers.BatchNormalization()(conv3)
            pool3 = layers.MaxPooling2D(2)(conv3)
            
            # Block 4 (bottleneck)
            conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
            conv4 = layers.BatchNormalization()(conv4)
            conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
            conv4 = layers.BatchNormalization()(conv4)
            
            # Decoder (upsampling path)
            # Block 5
            up5 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(conv4)
            up5 = layers.concatenate([up5, conv3])
            conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(up5)
            conv5 = layers.BatchNormalization()(conv5)
            conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv5)
            conv5 = layers.BatchNormalization()(conv5)
            
            # Block 6
            up6 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv5)
            up6 = layers.concatenate([up6, conv2])
            conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(up6)
            conv6 = layers.BatchNormalization()(conv6)
            conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv6)
            conv6 = layers.BatchNormalization()(conv6)
            
            # Block 7
            up7 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(conv6)
            up7 = layers.concatenate([up7, conv1])
            conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(up7)
            conv7 = layers.BatchNormalization()(conv7)
            conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv7)
            conv7 = layers.BatchNormalization()(conv7)
            
            # Output layer - 4 classes
            outputs = layers.Conv2D(4, 1, activation='softmax')(conv7)
            
            # Create and compile model
            self.model = keras.Model(inputs, outputs, name='crop_health_cnn')
            
            # Compile with Adam optimizer
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("CropHealthCNN model created successfully")
            logger.info(f"Model has {self.model.count_params()} parameters")
            
        except ImportError:
            logger.warning("TensorFlow not available. Model will not be functional.")
            self.model = None
    
    def predict_with_confidence(self, image_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return predictions with confidence scores.
        
        Args:
            image_patch: Input image patch(es) of shape (N, 64, 64, 4) or (64, 64, 4)
        
        Returns:
            Tuple of (predictions, confidence_scores)
            - predictions: Class labels (0-3) of shape matching input
            - confidence_scores: Confidence values (0-1) of shape matching input
        
        Raises:
            ValueError: If model is not trained or input shape is invalid
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model is None:
            raise ValueError("Model not available (TensorFlow not installed)")
        
        # Handle single patch input
        if len(image_patch.shape) == 3:
            image_patch = np.expand_dims(image_patch, axis=0)
            single_input = True
        else:
            single_input = False
        
        # Validate shape
        if image_patch.shape[1:] != (64, 64, 4):
            raise ValueError(f"Expected shape (N, 64, 64, 4), got {image_patch.shape}")
        
        # Record inference time
        start_time = time.time()
        
        # Make predictions
        probs = self.model.predict(image_patch, verbose=0)
        
        inference_time = time.time() - start_time
        
        # Get class predictions (argmax over class dimension)
        predictions = np.argmax(probs, axis=-1)
        
        # Get confidence scores (max probability)
        confidence = np.max(probs, axis=-1)
        
        # Log inference metrics
        logger.info(f"Inference completed in {inference_time:.3f}s for {len(image_patch)} patches")
        logger.info(f"Model version: {self.model_version}")
        
        # Return single values if single input
        if single_input:
            return predictions[0], confidence[0]
        
        return predictions, confidence

    
    def prepare_training_data(self, 
                             images: np.ndarray, 
                             labels: np.ndarray,
                             augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with patch extraction and augmentation.
        
        Args:
            images: Array of images (N, H, W, 4)
            labels: Array of labels (N, H, W) with class indices 0-3
            augment: Whether to apply data augmentation
        
        Returns:
            Tuple of (patches, patch_labels) ready for training
        """
        patches = []
        patch_labels = []
        
        # Extract 64x64 patches from images
        for img, lbl in zip(images, labels):
            h, w = img.shape[:2]
            
            # Extract patches with stride of 32 (50% overlap)
            for i in range(0, h - 64 + 1, 32):
                for j in range(0, w - 64 + 1, 32):
                    patch = img[i:i+64, j:j+64]
                    label_patch = lbl[i:i+64, j:j+64]
                    
                    patches.append(patch)
                    patch_labels.append(label_patch)
        
        patches = np.array(patches)
        patch_labels = np.array(patch_labels)
        
        logger.info(f"Extracted {len(patches)} patches from {len(images)} images")
        
        # Apply data augmentation if requested
        if augment:
            patches, patch_labels = self._augment_data(patches, patch_labels)
        
        # Convert labels to one-hot encoding
        patch_labels_onehot = self._to_categorical(patch_labels, num_classes=4)
        
        return patches, patch_labels_onehot
    
    def _augment_data(self, patches: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation (rotations and flips).
        
        Args:
            patches: Input patches (N, 64, 64, 4)
            labels: Input labels (N, 64, 64)
        
        Returns:
            Augmented (patches, labels)
        """
        augmented_patches = [patches]
        augmented_labels = [labels]
        
        # Rotation 90 degrees
        augmented_patches.append(np.rot90(patches, k=1, axes=(1, 2)))
        augmented_labels.append(np.rot90(labels, k=1, axes=(1, 2)))
        
        # Rotation 180 degrees
        augmented_patches.append(np.rot90(patches, k=2, axes=(1, 2)))
        augmented_labels.append(np.rot90(labels, k=2, axes=(1, 2)))
        
        # Rotation 270 degrees
        augmented_patches.append(np.rot90(patches, k=3, axes=(1, 2)))
        augmented_labels.append(np.rot90(labels, k=3, axes=(1, 2)))
        
        # Horizontal flip
        augmented_patches.append(np.flip(patches, axis=2))
        augmented_labels.append(np.flip(labels, axis=2))
        
        # Vertical flip
        augmented_patches.append(np.flip(patches, axis=1))
        augmented_labels.append(np.flip(labels, axis=1))
        
        # Concatenate all augmented data
        final_patches = np.concatenate(augmented_patches, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0)
        
        logger.info(f"Augmented data from {len(patches)} to {len(final_patches)} patches")
        
        return final_patches, final_labels
    
    def _to_categorical(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert integer labels to one-hot encoding.
        
        Args:
            labels: Integer labels (N, H, W)
            num_classes: Number of classes
        
        Returns:
            One-hot encoded labels (N, H, W, num_classes)
        """
        shape = labels.shape
        one_hot = np.zeros(shape + (num_classes,), dtype=np.float32)
        
        for i in range(num_classes):
            one_hot[..., i] = (labels == i).astype(np.float32)
        
        return one_hot
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32) -> dict:
        """
        Train the CNN model with early stopping and checkpointing.
        
        Args:
            X_train: Training patches (N, 64, 64, 4)
            y_train: Training labels (N, 64, 64, 4) one-hot encoded
            X_val: Validation patches (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history dictionary
        
        Raises:
            ValueError: If model is not available
        """
        if self.model is None:
            raise ValueError("Model not available (TensorFlow not installed)")
        
        try:
            from tensorflow import keras
            
            # Setup callbacks
            callbacks = [
                # Early stopping
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                # Model checkpointing
                keras.callbacks.ModelCheckpoint(
                    'models/crop_health_cnn_best.h5',
                    monitor='val_accuracy' if X_val is not None else 'accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                # Learning rate reduction
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            logger.info(f"Starting training with {len(X_train)} samples")
            logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            logger.info("Training completed successfully")
            
            return history.history
            
        except ImportError:
            raise ValueError("TensorFlow not available for training")
    
    def save_model(self, filepath: str):
        """
        Save the trained model with versioning.
        
        Args:
            filepath: Path to save the model
        
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if self.model is None:
            raise ValueError("Model not available")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath} (version {self.model_version})")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            from tensorflow import keras
            
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            
        except ImportError:
            raise ValueError("TensorFlow not available for loading model")

    
    def predict_with_uncertainty(self, 
                                image_patch: np.ndarray,
                                n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            image_patch: Input image patch(es) of shape (N, 64, 64, 4) or (64, 64, 4)
            n_samples: Number of Monte Carlo samples for uncertainty estimation
        
        Returns:
            Tuple of (predictions, confidence_scores, uncertainty_estimates)
            - predictions: Class labels (0-3)
            - confidence_scores: Confidence values (0-1)
            - uncertainty_estimates: Uncertainty values (0-1), higher means more uncertain
        
        Raises:
            ValueError: If model is not trained or input shape is invalid
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model is None:
            raise ValueError("Model not available (TensorFlow not installed)")
        
        # Handle single patch input
        if len(image_patch.shape) == 3:
            image_patch = np.expand_dims(image_patch, axis=0)
            single_input = True
        else:
            single_input = False
        
        # Validate shape
        if image_patch.shape[1:] != (64, 64, 4):
            raise ValueError(f"Expected shape (N, 64, 64, 4), got {image_patch.shape}")
        
        # Collect predictions with dropout enabled (Monte Carlo dropout)
        predictions_list = []
        
        logger.info(f"Running Monte Carlo dropout with {n_samples} samples")
        
        for i in range(n_samples):
            # Enable dropout during inference by setting training=True
            pred = self.model(image_patch, training=True)
            predictions_list.append(pred.numpy())
        
        # Stack predictions
        predictions_array = np.array(predictions_list)  # Shape: (n_samples, N, H, W, 4)
        
        # Calculate mean predictions
        mean_probs = np.mean(predictions_array, axis=0)
        
        # Get class predictions
        predictions = np.argmax(mean_probs, axis=-1)
        
        # Get confidence scores (max probability)
        confidence = np.max(mean_probs, axis=-1)
        
        # Calculate uncertainty using entropy
        # Entropy = -sum(p * log(p))
        epsilon = 1e-10  # Small value to avoid log(0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)
        
        # Normalize entropy to 0-1 scale
        max_entropy = np.log(4)  # log(num_classes)
        uncertainty = entropy / max_entropy
        
        logger.info(f"Uncertainty estimation completed")
        logger.info(f"Mean uncertainty: {np.mean(uncertainty):.3f}")
        
        # Return single values if single input
        if single_input:
            return predictions[0], confidence[0], uncertainty[0]
        
        return predictions, confidence, uncertainty
    
    def predict_batch(self, 
                     image_patches: np.ndarray,
                     batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficient batch prediction for multiple patches.
        
        Args:
            image_patches: Input patches (N, 64, 64, 4)
            batch_size: Batch size for prediction
        
        Returns:
            Tuple of (predictions, confidence_scores)
        
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model is None:
            raise ValueError("Model not available (TensorFlow not installed)")
        
        n_patches = len(image_patches)
        all_predictions = []
        all_confidence = []
        
        logger.info(f"Batch prediction for {n_patches} patches with batch_size={batch_size}")
        start_time = time.time()
        
        # Process in batches
        for i in range(0, n_patches, batch_size):
            batch = image_patches[i:i+batch_size]
            
            # Make predictions
            probs = self.model.predict(batch, verbose=0)
            
            # Get class predictions
            preds = np.argmax(probs, axis=-1)
            conf = np.max(probs, axis=-1)
            
            all_predictions.append(preds)
            all_confidence.append(conf)
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        confidence = np.concatenate(all_confidence, axis=0)
        
        inference_time = time.time() - start_time
        logger.info(f"Batch prediction completed in {inference_time:.3f}s")
        logger.info(f"Average time per patch: {inference_time/n_patches*1000:.2f}ms")
        
        return predictions, confidence
