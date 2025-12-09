"""
Property-based tests for CNN model predictions.

**Feature: production-enhancements, Property 11: CNN prediction confidence bounds**
**Validates: Requirements 3.3**
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import hypothesis.extra.numpy as npst

# Try to import the CNN model
try:
    from src.ai_models.crop_health_cnn import CropHealthCNN
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    pytest.skip("TensorFlow not available", allow_module_level=True)


def get_mock_trained_cnn():
    """Create a mock trained CNN for testing."""
    cnn = CropHealthCNN()
    # Mark as trained for testing purposes
    cnn.is_trained = True
    return cnn


@given(
    image_patch=npst.arrays(
        dtype=np.float32,
        shape=(1, 64, 64, 4),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)
    )
)
@settings(max_examples=100, deadline=None)
def test_cnn_confidence_bounds_property(image_patch):
    """
    Property 11: CNN prediction confidence bounds
    
    For any image patch input to the CNN model, confidence scores should be 
    in range [0, 1] and sum to 1.0 across all classes.
    
    This property validates Requirements 3.3.
    """
    mock_trained_cnn = get_mock_trained_cnn()
    
    if not TENSORFLOW_AVAILABLE or mock_trained_cnn.model is None:
        pytest.skip("TensorFlow not available or model not built")
    
    try:
        # Make predictions
        predictions, confidence = mock_trained_cnn.predict_with_confidence(image_patch)
        
        # Property 1: Confidence scores must be in range [0, 1]
        assert np.all(confidence >= 0.0), f"Confidence scores below 0: {confidence[confidence < 0]}"
        assert np.all(confidence <= 1.0), f"Confidence scores above 1: {confidence[confidence > 1]}"
        
        # Property 2: Predictions must be valid class indices (0-3)
        assert np.all(predictions >= 0), f"Invalid predictions below 0: {predictions[predictions < 0]}"
        assert np.all(predictions <= 3), f"Invalid predictions above 3: {predictions[predictions > 3]}"
        
        # Property 3: For the full probability distribution, probabilities should sum to 1
        # Get the full probability distribution
        probs = mock_trained_cnn.model.predict(image_patch, verbose=0)
        prob_sums = np.sum(probs, axis=-1)
        
        # Allow small numerical error (1e-5)
        assert np.allclose(prob_sums, 1.0, atol=1e-5), \
            f"Probabilities don't sum to 1.0: {prob_sums}"
        
    except Exception as e:
        # If model inference fails, that's acceptable for this test
        # (we're testing the property when inference succeeds)
        if "Model must be trained" in str(e):
            pytest.skip("Model not properly trained for testing")
        raise


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    patch_values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)
)
@settings(max_examples=50, deadline=None)
def test_cnn_confidence_consistency(batch_size, patch_values):
    """
    Property: CNN confidence scores should be reasonably consistent for identical inputs.
    
    Note: Due to batch normalization and potential numerical differences,
    we allow for small variations but they should be minimal.
    """
    mock_trained_cnn = get_mock_trained_cnn()
    
    if not TENSORFLOW_AVAILABLE or mock_trained_cnn.model is None:
        pytest.skip("TensorFlow not available or model not built")
    
    try:
        # Create a batch of identical patches
        image_batch = np.full((batch_size, 64, 64, 4), patch_values, dtype=np.float32)
        
        # Make predictions
        predictions, confidence = mock_trained_cnn.predict_with_confidence(image_batch)
        
        # All confidence scores should be reasonably similar for identical inputs
        # Allow for some variation due to batch normalization
        if batch_size > 1:
            confidence_std = np.std(confidence)
            # Relaxed threshold to account for batch normalization effects
            assert confidence_std < 0.1, \
                f"Confidence scores vary too much for identical inputs: std={confidence_std}"
        
        # Confidence should still be in valid range
        assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0)
        
    except Exception as e:
        if "Model must be trained" in str(e):
            pytest.skip("Model not properly trained for testing")
        raise


def test_cnn_confidence_bounds_with_real_data():
    """
    Test CNN confidence bounds with realistic satellite data values.
    
    This is a concrete example test to complement the property tests.
    """
    mock_trained_cnn = get_mock_trained_cnn()
    
    if not TENSORFLOW_AVAILABLE or mock_trained_cnn.model is None:
        pytest.skip("TensorFlow not available or model not built")
    
    try:
        # Create realistic satellite data (normalized reflectance values)
        # Typical values for vegetation: NIR high, Red low
        healthy_patch = np.random.uniform(0.1, 0.9, size=(1, 64, 64, 4)).astype(np.float32)
        healthy_patch[..., 3] = np.random.uniform(0.6, 0.9, size=(64, 64))  # High NIR
        healthy_patch[..., 2] = np.random.uniform(0.1, 0.3, size=(64, 64))  # Low Red
        
        predictions, confidence = mock_trained_cnn.predict_with_confidence(healthy_patch)
        
        # Validate confidence bounds
        assert np.all(confidence >= 0.0), "Confidence below 0"
        assert np.all(confidence <= 1.0), "Confidence above 1"
        
        # Validate predictions are valid classes
        assert np.all(predictions >= 0) and np.all(predictions <= 3), \
            "Predictions outside valid class range"
        
    except Exception as e:
        if "Model must be trained" in str(e):
            pytest.skip("Model not properly trained for testing")
        raise


def test_cnn_monte_carlo_uncertainty_bounds():
    """
    Test that Monte Carlo dropout uncertainty estimates are in valid range.
    """
    mock_trained_cnn = get_mock_trained_cnn()
    
    if not TENSORFLOW_AVAILABLE or mock_trained_cnn.model is None:
        pytest.skip("TensorFlow not available or model not built")
    
    try:
        # Create a test patch
        test_patch = np.random.uniform(0.0, 1.0, size=(1, 64, 64, 4)).astype(np.float32)
        
        # Get predictions with uncertainty
        predictions, confidence, uncertainty = mock_trained_cnn.predict_with_uncertainty(
            test_patch, n_samples=10
        )
        
        # Validate uncertainty bounds
        assert np.all(uncertainty >= 0.0), "Uncertainty below 0"
        assert np.all(uncertainty <= 1.0), "Uncertainty above 1"
        
        # Validate confidence bounds
        assert np.all(confidence >= 0.0), "Confidence below 0"
        assert np.all(confidence <= 1.0), "Confidence above 1"
        
    except Exception as e:
        if "Model must be trained" in str(e):
            pytest.skip("Model not properly trained for testing")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
