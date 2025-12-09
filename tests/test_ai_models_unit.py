"""
Unit tests for AI models.

Tests model loading, initialization, inference, fallback activation, and logging.
"""

import numpy as np
import pytest
import logging

# Try to import AI models
try:
    from src.ai_models.crop_health_cnn import CropHealthCNN
    from src.ai_models.vegetation_trend_lstm import VegetationTrendLSTM
    from src.ai_models.rule_based_classifier import RuleBasedClassifier
    from src.ai_models.crop_health_predictor import CropHealthPredictor
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    pytest.skip("TensorFlow or models not available", allow_module_level=True)


class TestCropHealthCNN:
    """Unit tests for CropHealthCNN model."""
    
    def test_model_initialization(self):
        """Test that CNN model initializes correctly."""
        cnn = CropHealthCNN()
        
        assert cnn is not None
        assert cnn.model_version == "1.0.0"
        assert cnn.is_trained == False
        
        if cnn.model is not None:
            # Check model architecture
            assert cnn.model.input_shape == (None, 64, 64, 4)
            assert cnn.model.output_shape == (None, 64, 64, 4)
    
    def test_model_loading_without_training(self):
        """Test that model raises error when predicting without training."""
        cnn = CropHealthCNN()
        
        if cnn.model is None:
            pytest.skip("TensorFlow not available")
        
        test_patch = np.random.rand(1, 64, 64, 4).astype(np.float32)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            cnn.predict_with_confidence(test_patch)
    
    def test_inference_with_sample_data(self):
        """Test inference with sample data after marking as trained."""
        cnn = CropHealthCNN()
        
        if cnn.model is None:
            pytest.skip("TensorFlow not available")
        
        # Mark as trained for testing
        cnn.is_trained = True
        
        # Create sample data
        test_patch = np.random.rand(1, 64, 64, 4).astype(np.float32)
        
        # Make predictions
        predictions, confidence = cnn.predict_with_confidence(test_patch)
        
        # Validate outputs
        assert predictions is not None
        assert confidence is not None
        assert predictions.shape == (1, 64, 64)  # Batch dimension preserved
        assert confidence.shape == (1, 64, 64)
        assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0)
        assert np.all(predictions >= 0) and np.all(predictions <= 3)


class TestVegetationTrendLSTM:
    """Unit tests for VegetationTrendLSTM model."""
    
    def test_model_initialization(self):
        """Test that LSTM model initializes correctly."""
        lstm = VegetationTrendLSTM(sequence_length=30)
        
        assert lstm is not None
        assert lstm.sequence_length == 30
        assert lstm.model_version == "1.0.0"
        assert lstm.is_trained == False
        
        if lstm.model is not None:
            # Check model architecture
            assert lstm.model.input_shape == (None, 30, 4)
            assert lstm.model.output_shape == (None, 1)
    
    def test_model_loading_without_training(self):
        """Test that model raises error when predicting without training."""
        lstm = VegetationTrendLSTM(sequence_length=10)
        
        if lstm.model is None:
            pytest.skip("TensorFlow not available")
        
        test_sequence = np.random.rand(5, 10, 4).astype(np.float32)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            lstm.predict_trend(test_sequence)
    
    def test_inference_with_sample_data(self):
        """Test inference with sample data after marking as trained."""
        lstm = VegetationTrendLSTM(sequence_length=10)
        
        if lstm.model is None:
            pytest.skip("TensorFlow not available")
        
        # Mark as trained for testing
        lstm.is_trained = True
        
        # Create sample data
        test_sequence = np.random.rand(5, 10, 4).astype(np.float32)
        
        # Make predictions
        predictions, ci, trend_dir, trend_str = lstm.predict_trend(test_sequence, return_confidence=False)
        
        # Validate outputs
        assert predictions is not None
        assert len(predictions) == 5
        assert trend_dir in ['increasing', 'decreasing', 'stable']
        assert 0.0 <= trend_str <= 1.0


class TestRuleBasedClassifier:
    """Unit tests for RuleBasedClassifier."""
    
    def test_classifier_initialization(self):
        """Test that rule-based classifier initializes correctly."""
        classifier = RuleBasedClassifier()
        
        assert classifier is not None
        assert classifier.THRESHOLD_HEALTHY == 0.7
        assert classifier.THRESHOLD_MODERATE == 0.5
        assert classifier.THRESHOLD_STRESSED == 0.3
    
    def test_classify_healthy(self):
        """Test classification of healthy vegetation."""
        classifier = RuleBasedClassifier()
        
        # NDVI > 0.7 should be healthy
        ndvi_values = np.array([0.75, 0.8, 0.85, 0.9])
        result = classifier.classify(ndvi_values)
        
        assert np.all(result.predictions == 0)  # All healthy
        assert np.all(result.confidence_scores >= 0.0)
        assert np.all(result.confidence_scores <= 1.0)
    
    def test_classify_moderate(self):
        """Test classification of moderate vegetation."""
        classifier = RuleBasedClassifier()
        
        # 0.5 < NDVI <= 0.7 should be moderate
        ndvi_values = np.array([0.55, 0.6, 0.65])
        result = classifier.classify(ndvi_values)
        
        assert np.all(result.predictions == 1)  # All moderate
    
    def test_classify_stressed(self):
        """Test classification of stressed vegetation."""
        classifier = RuleBasedClassifier()
        
        # 0.3 < NDVI <= 0.5 should be stressed
        ndvi_values = np.array([0.35, 0.4, 0.45])
        result = classifier.classify(ndvi_values)
        
        assert np.all(result.predictions == 2)  # All stressed
    
    def test_classify_critical(self):
        """Test classification of critical vegetation."""
        classifier = RuleBasedClassifier()
        
        # NDVI <= 0.3 should be critical
        ndvi_values = np.array([0.1, 0.2, 0.3])
        result = classifier.classify(ndvi_values)
        
        assert np.all(result.predictions == 3)  # All critical
    
    def test_classify_mixed(self):
        """Test classification of mixed vegetation health."""
        classifier = RuleBasedClassifier()
        
        ndvi_values = np.array([0.8, 0.6, 0.4, 0.2])
        result = classifier.classify(ndvi_values)
        
        assert result.predictions[0] == 0  # Healthy
        assert result.predictions[1] == 1  # Moderate
        assert result.predictions[2] == 2  # Stressed
        assert result.predictions[3] == 3  # Critical


class TestCropHealthPredictor:
    """Unit tests for CropHealthPredictor with fallback."""
    
    def test_predictor_initialization_without_model(self):
        """Test predictor initializes in rule-based mode when model unavailable."""
        predictor = CropHealthPredictor(model_path="nonexistent_model.h5")
        
        assert predictor.mode == 'rule_based'
        assert predictor.rule_classifier is not None
    
    def test_fallback_activation(self):
        """Test that fallback to rule-based works."""
        predictor = CropHealthPredictor(model_path="nonexistent_model.h5")
        
        # Should use rule-based classifier
        ndvi_values = np.array([0.8, 0.6, 0.4, 0.2])
        result = predictor.predict(ndvi_values)
        
        assert result.method == 'rule_based'
        assert len(result.predictions) == 4
    
    def test_model_info(self):
        """Test getting model information."""
        predictor = CropHealthPredictor(model_path="nonexistent_model.h5")
        
        info = predictor.get_model_info()
        
        assert 'mode' in info
        assert 'model_version' in info
        assert 'fallback_available' in info
        assert info['fallback_available'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
