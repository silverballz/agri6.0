"""
CNN model for spatial analysis of multispectral satellite imagery.

This module implements Convolutional Neural Networks for analyzing
spatial patterns in Sentinel-2A multispectral data to classify crop health
and detect diseases or stress conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class CNNConfig:
    """Configuration for CNN model."""
    input_shape: Tuple[int, int, int] = (64, 64, 4)  # Height, Width, Channels (B02,B03,B04,B08)
    num_classes: int = 4  # healthy, stressed, diseased, pest_damage
    conv_filters: List[int] = None
    conv_kernel_size: int = 3
    pool_size: int = 2
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [32, 64, 128, 256]


@dataclass
class ClassificationResult:
    """Result of CNN classification."""
    predictions: np.ndarray
    probabilities: np.ndarray
    confidence_scores: np.ndarray
    class_names: List[str]
    uncertainty_estimates: np.ndarray


class SpatialCNN:
    """
    CNN model for spatial analysis of multispectral imagery.
    
    This class implements a U-Net style architecture for pixel-level
    classification of crop health conditions using Sentinel-2A bands.
    """
    
    def __init__(self, config: CNNConfig = None):
        """Initialize CNN model with configuration."""
        self.config = config or CNNConfig()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.class_names = ['healthy', 'stressed', 'diseased', 'pest_damage']
        
    def _create_model(self) -> keras.Model:
        """
        Create CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.config.input_shape)
        
        # Encoder (downsampling path)
        x = inputs
        skip_connections = []
        
        for i, filters in enumerate(self.config.conv_filters):
            # Convolutional block
            x = layers.Conv2D(
                filters, 
                self.config.conv_kernel_size, 
                activation='relu', 
                padding='same',
                name=f'conv_block_{i}_1'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}_1')(x)
            x = layers.Conv2D(
                filters, 
                self.config.conv_kernel_size, 
                activation='relu', 
                padding='same',
                name=f'conv_block_{i}_2'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}_2')(x)
            
            # Store skip connection (except for the last layer)
            if i < len(self.config.conv_filters) - 1:
                skip_connections.append(x)
                x = layers.MaxPooling2D(self.config.pool_size, name=f'pool_{i}')(x)
                x = layers.Dropout(self.config.dropout_rate, name=f'dropout_{i}')(x)
        
        # Decoder (upsampling path)
        skip_connections.reverse()
        
        for i, filters in enumerate(reversed(self.config.conv_filters[:-1])):
            # Upsampling
            x = layers.Conv2DTranspose(
                filters, 
                2, 
                strides=2, 
                padding='same',
                name=f'upsample_{i}'
            )(x)
            
            # Skip connection
            if i < len(skip_connections):
                x = layers.concatenate([x, skip_connections[i]], name=f'concat_{i}')
            
            # Convolutional block
            x = layers.Conv2D(
                filters, 
                self.config.conv_kernel_size, 
                activation='relu', 
                padding='same',
                name=f'up_conv_block_{i}_1'
            )(x)
            x = layers.BatchNormalization(name=f'up_bn_{i}_1')(x)
            x = layers.Conv2D(
                filters, 
                self.config.conv_kernel_size, 
                activation='relu', 
                padding='same',
                name=f'up_conv_block_{i}_2'
            )(x)
            x = layers.BatchNormalization(name=f'up_bn_{i}_2')(x)
            x = layers.Dropout(self.config.dropout_rate, name=f'up_dropout_{i}')(x)
        
        # Output layer
        outputs = layers.Conv2D(
            self.config.num_classes, 
            1, 
            activation='softmax',
            name='output'
        )(x)
        
        model = keras.Model(inputs, outputs, name='spatial_cnn')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_crossentropy']
        )
        
        return model
    
    def prepare_training_data(self,
                            image_patches: np.ndarray,
                            labels: np.ndarray,
                            normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare image patches and labels for training.
        
        Args:
            image_patches: Array of image patches (N, H, W, C)
            labels: Array of labels for each patch
            normalize: Whether to normalize pixel values
            
        Returns:
            Tuple of (processed_patches, encoded_labels)
        """
        # Normalize pixel values if requested
        if normalize:
            # Assume input values are in 0-10000 range (Sentinel-2A DN values)
            image_patches = image_patches.astype(np.float32) / 10000.0
            image_patches = np.clip(image_patches, 0, 1)
        
        # Encode labels
        if not hasattr(self.label_encoder, 'classes_'):
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            encoded_labels = self.label_encoder.transform(labels)
        
        logger.info(f"Prepared {len(image_patches)} patches with shape {image_patches.shape}")
        logger.info(f"Label distribution: {np.bincount(encoded_labels)}")
        
        return image_patches, encoded_labels
    
    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            X: Training image patches
            y: Training labels
            validation_data: Optional validation data
            
        Returns:
            Training history
        """
        # Create model
        self.model = self._create_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_spatial_cnn.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split if validation_data is None else 0,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("CNN model training completed")
        
        return history.history
    
    def predict(self,
                X: np.ndarray,
                return_uncertainty: bool = True) -> ClassificationResult:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input image patches
            return_uncertainty: Whether to calculate uncertainty estimates
            
        Returns:
            ClassificationResult with predictions and metadata
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Make predictions
        probabilities = self.model.predict(X)
        
        # Get class predictions
        predictions = np.argmax(probabilities, axis=-1)
        
        # Calculate confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=-1)
        
        # Calculate uncertainty estimates
        uncertainty_estimates = None
        if return_uncertainty:
            uncertainty_estimates = self._calculate_uncertainty(X)
        
        # Flatten predictions for patch-level results
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
            confidence_scores = confidence_scores.flatten()
            if uncertainty_estimates is not None:
                uncertainty_estimates = uncertainty_estimates.flatten()
        
        return ClassificationResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=confidence_scores,
            class_names=self.class_names,
            uncertainty_estimates=uncertainty_estimates
        )
    
    def _calculate_uncertainty(self,
                             X: np.ndarray,
                             n_samples: int = 50) -> np.ndarray:
        """
        Calculate uncertainty estimates using Monte Carlo dropout.
        
        Args:
            X: Input image patches
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Uncertainty estimates array
        """
        # Enable dropout during inference
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate entropy as uncertainty measure
        mean_probs = np.mean(predictions, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
        
        # Normalize entropy to 0-1 scale
        max_entropy = np.log(self.config.num_classes)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def evaluate(self,
                X_test: np.ndarray,
                y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test image patches
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        result = self.predict(X_test, return_uncertainty=False)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, result.predictions)
        
        # Classification report
        class_report = classification_report(
            y_test,
            result.predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, result.predictions)
        
        metrics = {
            'accuracy': float(accuracy),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'mean_confidence': float(np.mean(result.confidence_scores))
        }
        
        logger.info(f"Model evaluation - Accuracy: {accuracy:.3f}")
        return metrics
    
    def predict_image(self,
                     image: np.ndarray,
                     patch_size: int = None,
                     overlap: int = 16) -> np.ndarray:
        """
        Predict on a full image using sliding window approach.
        
        Args:
            image: Full image array (H, W, C)
            patch_size: Size of patches to extract
            overlap: Overlap between patches
            
        Returns:
            Prediction map for the full image
        """
        if patch_size is None:
            patch_size = self.config.input_shape[0]
        
        h, w, c = image.shape
        stride = patch_size - overlap
        
        # Calculate output shape
        out_h = ((h - patch_size) // stride) + 1
        out_w = ((w - patch_size) // stride) + 1
        
        prediction_map = np.zeros((out_h, out_w), dtype=np.int32)
        confidence_map = np.zeros((out_h, out_w), dtype=np.float32)
        
        # Extract patches and predict
        patches = []
        positions = []
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                positions.append((i // stride, j // stride))
        
        if patches:
            patches = np.array(patches)
            
            # Normalize patches
            patches = patches.astype(np.float32) / 10000.0
            patches = np.clip(patches, 0, 1)
            
            # Make predictions
            result = self.predict(patches, return_uncertainty=False)
            
            # Fill prediction map
            for (i, j), pred, conf in zip(positions, result.predictions, result.confidence_scores):
                if i < out_h and j < out_w:
                    prediction_map[i, j] = pred
                    confidence_map[i, j] = conf
        
        return prediction_map, confidence_map
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class ImagePatchExtractor:
    """
    Utility class for extracting patches from satellite images.
    
    Handles patch extraction, augmentation, and labeling for CNN training.
    """
    
    def __init__(self, patch_size: int = 64, overlap: int = 32):
        """
        Initialize patch extractor.
        
        Args:
            patch_size: Size of patches to extract
            overlap: Overlap between patches
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
    
    def extract_patches(self,
                       image: np.ndarray,
                       mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches from an image.
        
        Args:
            image: Input image (H, W, C)
            mask: Optional mask for patch labeling (H, W)
            
        Returns:
            Tuple of (patches, labels) if mask provided, else (patches, None)
        """
        h, w, c = image.shape
        patches = []
        labels = []
        
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
                
                if mask is not None:
                    # Use center pixel of patch for labeling
                    center_i = i + self.patch_size // 2
                    center_j = j + self.patch_size // 2
                    label = mask[center_i, center_j]
                    labels.append(label)
        
        patches = np.array(patches)
        labels = np.array(labels) if labels else None
        
        logger.info(f"Extracted {len(patches)} patches of size {self.patch_size}x{self.patch_size}")
        
        return patches, labels
    
    def augment_patches(self,
                       patches: np.ndarray,
                       labels: np.ndarray = None,
                       augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment patches with rotations and flips.
        
        Args:
            patches: Input patches
            labels: Corresponding labels
            augmentation_factor: Number of augmented versions per patch
            
        Returns:
            Tuple of (augmented_patches, augmented_labels)
        """
        augmented_patches = [patches]
        augmented_labels = [labels] if labels is not None else [None]
        
        for _ in range(augmentation_factor - 1):
            # Random rotation (90, 180, 270 degrees)
            rotation = np.random.choice([1, 2, 3])
            rotated = np.rot90(patches, rotation, axes=(1, 2))
            augmented_patches.append(rotated)
            
            if labels is not None:
                augmented_labels.append(labels)
            
            # Random flip
            if np.random.random() > 0.5:
                flipped = np.flip(patches, axis=1)  # Horizontal flip
                augmented_patches.append(flipped)
                
                if labels is not None:
                    augmented_labels.append(labels)
        
        # Combine all augmented data
        final_patches = np.concatenate(augmented_patches, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0) if labels is not None else None
        
        logger.info(f"Augmented to {len(final_patches)} patches")
        
        return final_patches, final_labels
    
    def create_balanced_dataset(self,
                               patches: np.ndarray,
                               labels: np.ndarray,
                               samples_per_class: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a balanced dataset by sampling equal numbers from each class.
        
        Args:
            patches: Input patches
            labels: Corresponding labels
            samples_per_class: Number of samples per class
            
        Returns:
            Tuple of (balanced_patches, balanced_labels)
        """
        unique_labels = np.unique(labels)
        balanced_patches = []
        balanced_labels = []
        
        for label in unique_labels:
            # Find indices for this class
            class_indices = np.where(labels == label)[0]
            
            # Sample with replacement if needed
            if len(class_indices) >= samples_per_class:
                selected_indices = np.random.choice(
                    class_indices, 
                    samples_per_class, 
                    replace=False
                )
            else:
                selected_indices = np.random.choice(
                    class_indices, 
                    samples_per_class, 
                    replace=True
                )
            
            balanced_patches.append(patches[selected_indices])
            balanced_labels.append(labels[selected_indices])
        
        # Combine and shuffle
        final_patches = np.concatenate(balanced_patches, axis=0)
        final_labels = np.concatenate(balanced_labels, axis=0)
        
        # Shuffle
        shuffle_indices = np.random.permutation(len(final_patches))
        final_patches = final_patches[shuffle_indices]
        final_labels = final_labels[shuffle_indices]
        
        logger.info(f"Created balanced dataset with {len(final_patches)} patches")
        logger.info(f"Class distribution: {np.bincount(final_labels)}")
        
        return final_patches, final_labels