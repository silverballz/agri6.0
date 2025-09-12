"""
Training pipeline for CNN spatial analysis.

This module provides utilities for preparing spatial data, training CNN models,
and evaluating performance for crop health classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import joblib
import rasterio
from rasterio.windows import Window

from .spatial_cnn import SpatialCNN, CNNConfig, ImagePatchExtractor
from ..data_processing.sentinel2_parser import Sentinel2Parser
from ..data_processing.vegetation_indices import VegetationIndices
from ..database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class CNNTrainingPipeline:
    """
    Complete training pipeline for CNN spatial analysis.
    
    Handles data preparation, patch extraction, model training, and evaluation.
    """
    
    def __init__(self,
                 config: CNNConfig = None,
                 model_save_path: str = "models/cnn_spatial",
                 patch_size: int = 64):
        """
        Initialize training pipeline.
        
        Args:
            config: CNN configuration
            model_save_path: Path to save trained models
            patch_size: Size of image patches for training
        """
        self.config = config or CNNConfig()
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.cnn_model = SpatialCNN(self.config)
        self.patch_extractor = ImagePatchExtractor(patch_size=patch_size)
        self.sentinel_parser = Sentinel2Parser()
        self.vegetation_indices = VegetationIndices()
        
    def load_satellite_images(self,
                             safe_directories: List[str],
                             target_bands: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load Sentinel-2A images from SAFE directories.
        
        Args:
            safe_directories: List of SAFE directory paths
            target_bands: Bands to load (default: B02, B03, B04, B08, B11, B12)
            
        Returns:
            List of image dictionaries with bands and metadata
        """
        if target_bands is None:
            target_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        
        images = []
        
        for safe_dir in safe_directories:
            try:
                # Parse SAFE directory
                metadata = self.sentinel_parser.parse_safe_directory(safe_dir)
                
                # Load bands
                bands = {}
                for band in target_bands:
                    band_path = self.sentinel_parser.get_band_path(safe_dir, band, resolution='10m')
                    if band_path and band_path.exists():
                        with rasterio.open(band_path) as src:
                            bands[band] = src.read(1)
                    else:
                        logger.warning(f"Band {band} not found in {safe_dir}")
                
                if len(bands) == len(target_bands):
                    # Stack bands into single array
                    band_stack = np.stack([bands[band] for band in target_bands], axis=-1)
                    
                    images.append({
                        'image': band_stack,
                        'bands': bands,
                        'metadata': metadata,
                        'safe_dir': safe_dir
                    })
                    
                    logger.info(f"Loaded image from {safe_dir} with shape {band_stack.shape}")
                else:
                    logger.warning(f"Incomplete band set for {safe_dir}")
                    
            except Exception as e:
                logger.error(f"Error loading {safe_dir}: {e}")
        
        return images
    
    def create_ground_truth_masks(self,
                                 images: List[Dict[str, Any]],
                                 field_boundaries: pd.DataFrame,
                                 health_labels: pd.DataFrame) -> List[np.ndarray]:
        """
        Create ground truth masks for crop health classification.
        
        Args:
            images: List of loaded satellite images
            field_boundaries: DataFrame with field boundary geometries
            health_labels: DataFrame with health labels for each field
            
        Returns:
            List of ground truth masks
        """
        masks = []
        
        for image_data in images:
            image_shape = image_data['image'].shape[:2]
            mask = np.zeros(image_shape, dtype=np.int32)
            
            # Get image metadata for georeferencing
            metadata = image_data['metadata']
            
            # Map health labels to mask
            # This is a simplified version - in practice, you'd need proper
            # coordinate transformation and rasterization
            for _, field in field_boundaries.iterrows():
                field_id = field['field_id']
                
                # Find health label for this field
                health_row = health_labels[health_labels['field_id'] == field_id]
                if not health_row.empty:
                    health_status = health_row.iloc[0]['health_status']
                    
                    # Map health status to class index
                    class_mapping = {
                        'healthy': 0,
                        'stressed': 1,
                        'diseased': 2,
                        'pest_damage': 3
                    }
                    
                    class_idx = class_mapping.get(health_status, 0)
                    
                    # For demonstration, create random regions
                    # In practice, you'd rasterize the field geometry
                    y_start = np.random.randint(0, image_shape[0] - 100)
                    x_start = np.random.randint(0, image_shape[1] - 100)
                    y_end = y_start + np.random.randint(50, 100)
                    x_end = x_start + np.random.randint(50, 100)
                    
                    mask[y_start:y_end, x_start:x_end] = class_idx
            
            masks.append(mask)
            logger.info(f"Created mask with shape {mask.shape}")
        
        return masks
    
    def extract_training_patches(self,
                                images: List[Dict[str, Any]],
                                masks: List[np.ndarray],
                                samples_per_class: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract training patches from images and masks.
        
        Args:
            images: List of satellite images
            masks: List of ground truth masks
            samples_per_class: Number of samples per class
            
        Returns:
            Tuple of (patches, labels)
        """
        all_patches = []
        all_labels = []
        
        for image_data, mask in zip(images, masks):
            image = image_data['image']
            
            # Extract patches
            patches, labels = self.patch_extractor.extract_patches(image, mask)
            
            if patches is not None and labels is not None:
                # Filter out patches with label 0 (background) if too many
                valid_indices = labels > 0
                if np.sum(valid_indices) > 0:
                    patches = patches[valid_indices]
                    labels = labels[valid_indices]
                
                all_patches.append(patches)
                all_labels.append(labels)
        
        if all_patches:
            # Combine all patches
            combined_patches = np.concatenate(all_patches, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
            
            # Create balanced dataset
            balanced_patches, balanced_labels = self.patch_extractor.create_balanced_dataset(
                combined_patches,
                combined_labels,
                samples_per_class=samples_per_class
            )
            
            # Apply augmentation
            augmented_patches, augmented_labels = self.patch_extractor.augment_patches(
                balanced_patches,
                balanced_labels,
                augmentation_factor=3
            )
            
            logger.info(f"Extracted {len(augmented_patches)} training patches")
            return augmented_patches, augmented_labels
        
        else:
            logger.warning("No valid patches extracted")
            return np.array([]), np.array([])
    
    def create_synthetic_training_data(self,
                                      n_samples: int = 5000,
                                      patch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data for testing purposes.
        
        Args:
            n_samples: Number of samples to generate
            patch_size: Size of patches
            
        Returns:
            Tuple of (synthetic_patches, synthetic_labels)
        """
        np.random.seed(42)
        
        patches = []
        labels = []
        
        # Generate samples for each class
        samples_per_class = n_samples // 4
        
        for class_idx in range(4):  # 4 classes
            for _ in range(samples_per_class):
                # Create synthetic spectral signature for each class
                if class_idx == 0:  # Healthy
                    # High NIR, moderate Red, low SWIR
                    base_values = [800, 1200, 900, 3000, 1500, 1000]  # B02,B03,B04,B08,B11,B12
                elif class_idx == 1:  # Stressed
                    # Moderate NIR, higher Red, moderate SWIR
                    base_values = [900, 1300, 1200, 2200, 2000, 1500]
                elif class_idx == 2:  # Diseased
                    # Low NIR, high Red, high SWIR
                    base_values = [1000, 1400, 1500, 1800, 2500, 2000]
                else:  # Pest damage
                    # Very low NIR, very high Red, variable SWIR
                    base_values = [1100, 1500, 1800, 1500, 2200, 1800]
                
                # Create patch with spatial variation
                patch = np.zeros((patch_size, patch_size, 6))
                
                for i in range(6):  # 6 bands
                    # Add spatial structure
                    center_value = base_values[i]
                    noise = np.random.normal(0, center_value * 0.1, (patch_size, patch_size))
                    
                    # Add some spatial correlation
                    from scipy.ndimage import gaussian_filter
                    noise = gaussian_filter(noise, sigma=2)
                    
                    patch[:, :, i] = center_value + noise
                
                # Ensure values are positive and within reasonable range
                patch = np.clip(patch, 0, 10000)
                
                patches.append(patch)
                labels.append(class_idx)
        
        patches = np.array(patches)
        labels = np.array(labels)
        
        # Shuffle data
        shuffle_indices = np.random.permutation(len(patches))
        patches = patches[shuffle_indices]
        labels = labels[shuffle_indices]
        
        logger.info(f"Created {len(patches)} synthetic training samples")
        logger.info(f"Class distribution: {np.bincount(labels)}")
        
        return patches, labels
    
    def train_model(self,
                   patches: np.ndarray,
                   labels: np.ndarray,
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train CNN model on prepared patches.
        
        Args:
            patches: Training patches
            labels: Training labels
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and metrics
        """
        # Prepare training data
        X, y = self.cnn_model.prepare_training_data(patches, labels)
        
        # Split into train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training set: {len(X_train)} patches")
        logger.info(f"Validation set: {len(X_val)} patches")
        
        # Train model
        history = self.cnn_model.train(
            X_train, y_train,
            validation_data=(X_val, y_val)
        )
        
        # Evaluate on validation set
        val_metrics = self.cnn_model.evaluate(X_val, y_val)
        
        results = {
            'history': history,
            'validation_metrics': val_metrics,
            'training_patches': len(X_train),
            'validation_patches': len(X_val)
        }
        
        return results
    
    def evaluate_model_performance(self,
                                  test_patches: np.ndarray,
                                  test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_patches: Test patches
            test_labels: Test labels
            
        Returns:
            Evaluation metrics and results
        """
        # Prepare test data
        X_test, y_test = self.cnn_model.prepare_training_data(test_patches, test_labels)
        
        # Make predictions
        result = self.cnn_model.predict(X_test)
        
        # Evaluate metrics
        metrics = self.cnn_model.evaluate(X_test, y_test)
        
        evaluation_results = {
            'metrics': metrics,
            'predictions': result,
            'test_patches': len(X_test),
            'mean_confidence': float(np.mean(result.confidence_scores)),
            'mean_uncertainty': float(np.mean(result.uncertainty_estimates)) if result.uncertainty_estimates is not None else None
        }
        
        logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.3f}")
        
        return evaluation_results
    
    def predict_on_image(self,
                        image_path: str,
                        output_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply trained model to predict on a full satellite image.
        
        Args:
            image_path: Path to satellite image
            output_path: Optional path to save prediction map
            
        Returns:
            Tuple of (prediction_map, confidence_map)
        """
        if not self.cnn_model.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Load image
        with rasterio.open(image_path) as src:
            # Read bands (assuming 6-band image)
            image = src.read([1, 2, 3, 4, 5, 6])  # Read first 6 bands
            image = np.transpose(image, (1, 2, 0))  # Change to H,W,C format
        
        # Make predictions
        prediction_map, confidence_map = self.cnn_model.predict_image(image)
        
        # Save results if output path provided
        if output_path:
            output_path = Path(output_path)
            
            # Save prediction map
            pred_path = output_path.with_suffix('.pred.tif')
            with rasterio.open(pred_path, 'w', **src.profile) as dst:
                dst.write(prediction_map, 1)
            
            # Save confidence map
            conf_path = output_path.with_suffix('.conf.tif')
            profile = src.profile.copy()
            profile['dtype'] = 'float32'
            with rasterio.open(conf_path, 'w', **profile) as dst:
                dst.write(confidence_map, 1)
            
            logger.info(f"Predictions saved to {pred_path} and {conf_path}")
        
        return prediction_map, confidence_map
    
    def save_pipeline(self, suffix: str = None):
        """
        Save trained model and components.
        
        Args:
            suffix: Optional suffix for model files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = suffix or timestamp
        
        # Save CNN model
        model_path = self.model_save_path / f"cnn_model_{suffix}.h5"
        self.cnn_model.save_model(str(model_path))
        
        # Save label encoder
        encoder_path = self.model_save_path / f"label_encoder_{suffix}.pkl"
        joblib.dump(self.cnn_model.label_encoder, encoder_path)
        
        # Save configuration
        config_path = self.model_save_path / f"config_{suffix}.pkl"
        joblib.dump(self.config, config_path)
        
        logger.info(f"Pipeline saved with suffix: {suffix}")
        
        return {
            'model_path': str(model_path),
            'encoder_path': str(encoder_path),
            'config_path': str(config_path)
        }
    
    def load_pipeline(self, suffix: str):
        """
        Load trained model and components.
        
        Args:
            suffix: Suffix of model files to load
        """
        # Load CNN model
        model_path = self.model_save_path / f"cnn_model_{suffix}.h5"
        self.cnn_model.load_model(str(model_path))
        
        # Load label encoder
        encoder_path = self.model_save_path / f"label_encoder_{suffix}.pkl"
        self.cnn_model.label_encoder = joblib.load(encoder_path)
        
        # Load configuration
        config_path = self.model_save_path / f"config_{suffix}.pkl"
        self.config = joblib.load(config_path)
        
        logger.info(f"Pipeline loaded with suffix: {suffix}")


def create_sample_field_data(n_fields: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sample field boundary and health label data for testing.
    
    Args:
        n_fields: Number of fields to create
        
    Returns:
        Tuple of (field_boundaries, health_labels)
    """
    np.random.seed(42)
    
    # Create field boundaries (simplified)
    field_boundaries = []
    health_labels = []
    
    health_statuses = ['healthy', 'stressed', 'diseased', 'pest_damage']
    
    for i in range(n_fields):
        field_id = f'field_{i+1}'
        
        # Random field boundary (simplified as bounding box)
        x_min = np.random.randint(0, 1000)
        y_min = np.random.randint(0, 1000)
        x_max = x_min + np.random.randint(50, 200)
        y_max = y_min + np.random.randint(50, 200)
        
        field_boundaries.append({
            'field_id': field_id,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        })
        
        # Random health status
        health_status = np.random.choice(health_statuses)
        health_labels.append({
            'field_id': field_id,
            'health_status': health_status,
            'confidence': np.random.uniform(0.7, 1.0),
            'assessment_date': datetime.now()
        })
    
    field_boundaries_df = pd.DataFrame(field_boundaries)
    health_labels_df = pd.DataFrame(health_labels)
    
    logger.info(f"Created sample data for {n_fields} fields")
    
    return field_boundaries_df, health_labels_df