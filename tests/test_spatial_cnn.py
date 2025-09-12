"""
Tests for CNN spatial analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import os

from src.ai_models.spatial_cnn import (
    SpatialCNN, CNNConfig, ImagePatchExtractor, ClassificationResult
)
from src.ai_models.cnn_training_pipeline import (
    CNNTrainingPipeline, create_sample_field_data
)


class TestCNNConfig:
    """Test CNN configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CNNConfig()
        
        assert config.input_shape == (64, 64, 6)
        assert config.num_classes == 4
        assert config.conv_filters == [32, 64, 128, 256]
        assert config.conv_kernel_size == 3
        assert config.pool_size == 2
        assert config.dropout_rate == 0.3
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 50
        assert config.validation_split == 0.2
        assert config.early_stopping_patience == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CNNConfig(
            input_shape=(32, 32, 6),
            num_classes=3,
            conv_filters=[16, 32, 64],
            dropout_rate=0.4,
            learning_rate=0.01
        )
        
        assert config.input_shape == (32, 32, 6)
        assert config.num_classes == 3
        assert config.conv_filters == [16, 32, 64]
        assert config.dropout_rate == 0.4
        assert config.learning_rate == 0.01


class TestSpatialCNN:
    """Test CNN model for spatial analysis."""
    
    @pytest.fixture
    def sample_patches(self):
        """Create sample image patches."""
        np.random.seed(42)
        
        # Create 100 patches of size 32x32x6
        patches = np.random.randint(0, 10000, (100, 32, 32, 6)).astype(np.float32)
        
        # Create labels (4 classes)
        labels = np.random.randint(0, 4, 100)
        
        return patches, labels
    
    @pytest.fixture
    def cnn_model(self):
        """Create CNN model with test configuration."""
        config = CNNConfig(
            input_shape=(32, 32, 6),
            num_classes=4,
            conv_filters=[16, 32],
            epochs=3,
            batch_size=16
        )
        return SpatialCNN(config)
    
    def test_model_initialization(self, cnn_model):
        """Test model initialization."""
        assert cnn_model.config.input_shape == (32, 32, 6)
        assert cnn_model.config.num_classes == 4
        assert cnn_model.model is None
        assert not cnn_model.is_trained
        assert len(cnn_model.class_names) == 4
    
    def test_prepare_training_data(self, cnn_model, sample_patches):
        """Test training data preparation."""
        patches, labels = sample_patches
        
        X, y = cnn_model.prepare_training_data(patches, labels)
        
        # Check shapes
        assert X.shape == patches.shape
        assert y.shape == labels.shape
        
        # Check normalization
        assert X.min() >= 0
        assert X.max() <= 1
        
        # Check label encoding
        assert y.min() >= 0
        assert y.max() < cnn_model.config.num_classes
    
    def test_model_training(self, cnn_model, sample_patches):
        """Test model training."""
        patches, labels = sample_patches
        X, y = cnn_model.prepare_training_data(patches, labels)
        
        # Train model
        history = cnn_model.train(X, y)
        
        # Check training completed
        assert cnn_model.is_trained
        assert cnn_model.model is not None
        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) > 0
    
    def test_model_prediction(self, cnn_model, sample_patches):
        """Test model prediction."""
        patches, labels = sample_patches
        X, y = cnn_model.prepare_training_data(patches, labels)
        
        # Train model
        cnn_model.train(X, y)
        
        # Make predictions
        result = cnn_model.predict(X[:10])
        
        # Check prediction result
        assert isinstance(result, ClassificationResult)
        assert len(result.predictions) == 10
        assert result.probabilities.shape[0] == 10
        assert result.probabilities.shape[-1] == cnn_model.config.num_classes
        assert len(result.confidence_scores) == 10
        assert result.class_names == cnn_model.class_names
        assert result.uncertainty_estimates is not None
        assert len(result.uncertainty_estimates) == 10
    
    def test_model_evaluation(self, cnn_model, sample_patches):
        """Test model evaluation."""
        patches, labels = sample_patches
        X, y = cnn_model.prepare_training_data(patches, labels)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train and evaluate
        cnn_model.train(X_train, y_train)
        metrics = cnn_model.evaluate(X_test, y_test)
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics
        assert 'mean_confidence' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert isinstance(metrics['confusion_matrix'], list)
    
    def test_predict_image(self, cnn_model, sample_patches):
        """Test prediction on full image."""
        patches, labels = sample_patches
        X, y = cnn_model.prepare_training_data(patches, labels)
        
        # Train model
        cnn_model.train(X, y)
        
        # Create a larger test image
        test_image = np.random.randint(0, 10000, (128, 128, 6)).astype(np.float32)
        
        # Predict on image
        prediction_map, confidence_map = cnn_model.predict_image(
            test_image, 
            patch_size=32, 
            overlap=16
        )
        
        # Check results
        assert prediction_map.shape[0] > 0
        assert prediction_map.shape[1] > 0
        assert confidence_map.shape == prediction_map.shape
        assert prediction_map.min() >= 0
        assert prediction_map.max() < cnn_model.config.num_classes
        assert confidence_map.min() >= 0
        assert confidence_map.max() <= 1
    
    def test_model_save_load(self, cnn_model, sample_patches):
        """Test model saving and loading."""
        patches, labels = sample_patches
        X, y = cnn_model.prepare_training_data(patches, labels)
        cnn_model.train(X, y)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            
            # Save model
            cnn_model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new model and load
            new_model = SpatialCNN(cnn_model.config)
            new_model.label_encoder = cnn_model.label_encoder  # Copy encoder
            new_model.load_model(model_path)
            
            # Test loaded model
            assert new_model.is_trained
            result1 = cnn_model.predict(X[:5], return_uncertainty=False)
            result2 = new_model.predict(X[:5], return_uncertainty=False)
            
            # Predictions should be identical
            np.testing.assert_array_equal(result1.predictions, result2.predictions)


class TestImagePatchExtractor:
    """Test image patch extraction functionality."""
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image and mask."""
        np.random.seed(42)
        
        # Create 256x256x6 image
        image = np.random.randint(0, 10000, (256, 256, 6)).astype(np.float32)
        
        # Create mask with different regions
        mask = np.zeros((256, 256), dtype=np.int32)
        mask[50:150, 50:150] = 1  # Class 1 region
        mask[100:200, 100:200] = 2  # Class 2 region
        mask[150:250, 150:250] = 3  # Class 3 region
        
        return image, mask
    
    def test_extractor_initialization(self):
        """Test patch extractor initialization."""
        extractor = ImagePatchExtractor(patch_size=64, overlap=32)
        assert extractor.patch_size == 64
        assert extractor.overlap == 32
        assert extractor.stride == 32
    
    def test_extract_patches(self, sample_image):
        """Test patch extraction."""
        image, mask = sample_image
        extractor = ImagePatchExtractor(patch_size=64, overlap=32)
        
        patches, labels = extractor.extract_patches(image, mask)
        
        # Check results
        assert patches is not None
        assert labels is not None
        assert len(patches) == len(labels)
        assert patches.shape[1:] == (64, 64, 6)
        assert labels.min() >= 0
        assert labels.max() <= 3
    
    def test_extract_patches_without_mask(self, sample_image):
        """Test patch extraction without mask."""
        image, _ = sample_image
        extractor = ImagePatchExtractor(patch_size=64, overlap=32)
        
        patches, labels = extractor.extract_patches(image)
        
        # Check results
        assert patches is not None
        assert labels is None
        assert patches.shape[1:] == (64, 64, 6)
    
    def test_augment_patches(self, sample_image):
        """Test patch augmentation."""
        image, mask = sample_image
        extractor = ImagePatchExtractor(patch_size=64, overlap=32)
        
        patches, labels = extractor.extract_patches(image, mask)
        original_count = len(patches)
        
        # Augment patches
        aug_patches, aug_labels = extractor.augment_patches(
            patches, labels, augmentation_factor=3
        )
        
        # Check augmentation
        assert len(aug_patches) >= original_count * 3
        assert len(aug_labels) == len(aug_patches)
        assert aug_patches.shape[1:] == patches.shape[1:]
    
    def test_create_balanced_dataset(self, sample_image):
        """Test balanced dataset creation."""
        image, mask = sample_image
        extractor = ImagePatchExtractor(patch_size=64, overlap=32)
        
        patches, labels = extractor.extract_patches(image, mask)
        
        # Create balanced dataset
        balanced_patches, balanced_labels = extractor.create_balanced_dataset(
            patches, labels, samples_per_class=50
        )
        
        # Check balance
        unique_labels, counts = np.unique(balanced_labels, return_counts=True)
        assert len(unique_labels) > 1  # Multiple classes
        # All classes should have similar counts (allowing for some variation)
        assert np.std(counts) < np.mean(counts) * 0.5


class TestCNNTrainingPipeline:
    """Test CNN training pipeline."""
    
    @pytest.fixture
    def training_pipeline(self):
        """Create training pipeline with test configuration."""
        config = CNNConfig(
            input_shape=(32, 32, 6),
            num_classes=4,
            conv_filters=[16, 32],
            epochs=2,
            batch_size=16
        )
        return CNNTrainingPipeline(config, patch_size=32)
    
    def test_pipeline_initialization(self, training_pipeline):
        """Test pipeline initialization."""
        assert training_pipeline.config.input_shape == (32, 32, 6)
        assert training_pipeline.cnn_model is not None
        assert training_pipeline.patch_extractor is not None
        assert training_pipeline.patch_extractor.patch_size == 32
    
    def test_create_synthetic_training_data(self, training_pipeline):
        """Test synthetic training data creation."""
        patches, labels = training_pipeline.create_synthetic_training_data(
            n_samples=200, patch_size=32
        )
        
        # Check results
        assert len(patches) == 200
        assert len(labels) == 200
        assert patches.shape[1:] == (32, 32, 6)
        assert labels.min() >= 0
        assert labels.max() < 4
        
        # Check class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        assert len(unique_labels) == 4  # All 4 classes present
        assert all(count > 0 for count in counts)  # All classes have samples
    
    def test_train_model(self, training_pipeline):
        """Test model training."""
        # Create synthetic data
        patches, labels = training_pipeline.create_synthetic_training_data(
            n_samples=100, patch_size=32
        )
        
        # Train model
        results = training_pipeline.train_model(patches, labels, validation_split=0.3)
        
        # Check results
        assert 'history' in results
        assert 'validation_metrics' in results
        assert 'training_patches' in results
        assert 'validation_patches' in results
        
        assert training_pipeline.cnn_model.is_trained
        assert results['training_patches'] > 0
        assert results['validation_patches'] > 0
    
    def test_evaluate_model_performance(self, training_pipeline):
        """Test model performance evaluation."""
        # Create synthetic data
        train_patches, train_labels = training_pipeline.create_synthetic_training_data(
            n_samples=80, patch_size=32
        )
        test_patches, test_labels = training_pipeline.create_synthetic_training_data(
            n_samples=20, patch_size=32
        )
        
        # Train model
        training_pipeline.train_model(train_patches, train_labels, validation_split=0.2)
        
        # Evaluate performance
        results = training_pipeline.evaluate_model_performance(test_patches, test_labels)
        
        # Check results
        assert 'metrics' in results
        assert 'predictions' in results
        assert 'test_patches' in results
        assert 'mean_confidence' in results
        
        assert 0 <= results['metrics']['accuracy'] <= 1
        assert results['test_patches'] == 20
        assert 0 <= results['mean_confidence'] <= 1
    
    def test_save_load_pipeline(self, training_pipeline):
        """Test pipeline saving and loading."""
        # Create and train model
        patches, labels = training_pipeline.create_synthetic_training_data(
            n_samples=50, patch_size=32
        )
        training_pipeline.train_model(patches, labels, validation_split=0.3)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update save path
            training_pipeline.model_save_path = temp_dir
            
            # Save pipeline
            paths = training_pipeline.save_pipeline(suffix='test')
            
            # Check files exist
            assert os.path.exists(paths['model_path'])
            assert os.path.exists(paths['encoder_path'])
            assert os.path.exists(paths['config_path'])
            
            # Create new pipeline and load
            new_pipeline = CNNTrainingPipeline()
            new_pipeline.model_save_path = temp_dir
            new_pipeline.load_pipeline('test')
            
            # Check loaded pipeline
            assert new_pipeline.cnn_model.is_trained


class TestSampleFieldData:
    """Test sample field data generation utilities."""
    
    def test_create_sample_field_data(self):
        """Test sample field data creation."""
        field_boundaries, health_labels = create_sample_field_data(n_fields=10)
        
        # Check field boundaries
        assert len(field_boundaries) == 10
        assert 'field_id' in field_boundaries.columns
        assert 'x_min' in field_boundaries.columns
        assert 'y_min' in field_boundaries.columns
        assert 'x_max' in field_boundaries.columns
        assert 'y_max' in field_boundaries.columns
        
        # Check health labels
        assert len(health_labels) == 10
        assert 'field_id' in health_labels.columns
        assert 'health_status' in health_labels.columns
        assert 'confidence' in health_labels.columns
        assert 'assessment_date' in health_labels.columns
        
        # Check field IDs match
        boundary_ids = set(field_boundaries['field_id'])
        label_ids = set(health_labels['field_id'])
        assert boundary_ids == label_ids
        
        # Check health status values
        valid_statuses = {'healthy', 'stressed', 'diseased', 'pest_damage'}
        assert all(status in valid_statuses for status in health_labels['health_status'])
        
        # Check confidence values
        assert all(0.7 <= conf <= 1.0 for conf in health_labels['confidence'])
    
    def test_field_data_structure(self):
        """Test field data structure and relationships."""
        field_boundaries, health_labels = create_sample_field_data(n_fields=5)
        
        # Check that each field has valid boundaries
        for _, field in field_boundaries.iterrows():
            assert field['x_max'] > field['x_min']
            assert field['y_max'] > field['y_min']
            assert field['x_min'] >= 0
            assert field['y_min'] >= 0
        
        # Check that each field has a health assessment
        for field_id in field_boundaries['field_id']:
            health_row = health_labels[health_labels['field_id'] == field_id]
            assert len(health_row) == 1  # Exactly one health record per field