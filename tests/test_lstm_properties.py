"""
Property-based tests for LSTM model trend detection.

**Feature: production-enhancements, Property 12: LSTM trend detection consistency**
**Validates: Requirements 6.2**
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst

# Try to import the LSTM model
try:
    from src.ai_models.vegetation_trend_lstm import VegetationTrendLSTM
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    pytest.skip("TensorFlow not available", allow_module_level=True)


def get_mock_trained_lstm():
    """Create a mock trained LSTM for testing."""
    lstm = VegetationTrendLSTM(sequence_length=10)  # Shorter for testing
    # Mark as trained for testing purposes
    lstm.is_trained = True
    return lstm


def create_monotonic_increasing_sequence(length: int, start: float = 0.3, increment: float = 0.01) -> np.ndarray:
    """Create a monotonically increasing sequence."""
    return np.array([start + i * increment for i in range(length)])


def create_monotonic_decreasing_sequence(length: int, start: float = 0.8, decrement: float = 0.01) -> np.ndarray:
    """Create a monotonically decreasing sequence."""
    return np.array([start - i * decrement for i in range(length)])


@given(
    sequence_length=st.integers(min_value=15, max_value=50),
    start_value=st.floats(min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False, allow_subnormal=False),
    increment=st.floats(min_value=0.002, max_value=0.05, allow_nan=False, allow_infinity=False, allow_subnormal=False)
)
@settings(max_examples=100, deadline=None)
def test_lstm_increasing_trend_detection_property(sequence_length, start_value, increment):
    """
    Property 12: LSTM trend detection consistency
    
    For any time series with monotonically increasing values, the trend analysis
    logic should detect trend_direction as 'increasing'.
    
    This property validates Requirements 6.2.
    
    Note: This tests the _analyze_trend method directly rather than the full
    model prediction, since an untrained model produces random predictions.
    """
    lstm = get_mock_trained_lstm()
    
    if not TENSORFLOW_AVAILABLE or lstm.model is None:
        pytest.skip("TensorFlow not available or model not built")
    
    try:
        # Create monotonically increasing predictions (simulating what a trained model would produce)
        predictions = create_monotonic_increasing_sequence(sequence_length, start_value, increment)
        
        # Clip to valid range
        predictions = np.clip(predictions, 0.0, 1.0)
        
        # Test the trend analysis logic directly
        trend_direction, trend_strength = lstm._analyze_trend(predictions)
        
        # Property: For monotonically increasing predictions, trend should be 'increasing'
        assert trend_direction == 'increasing', \
            f"Expected 'increasing' trend for monotonic increase, got '{trend_direction}'"
        
        # Trend strength should be positive
        assert trend_strength >= 0.0, f"Trend strength should be non-negative, got {trend_strength}"
        
    except Exception as e:
        if "Model must be trained" in str(e):
            pytest.skip("Model not properly trained for testing")
        raise


@given(
    sequence_length=st.integers(min_value=15, max_value=50),
    start_value=st.floats(min_value=0.5, max_value=0.9, allow_nan=False, allow_infinity=False, allow_subnormal=False),
    decrement=st.floats(min_value=0.002, max_value=0.05, allow_nan=False, allow_infinity=False, allow_subnormal=False)
)
@settings(max_examples=100, deadline=None)
def test_lstm_decreasing_trend_detection_property(sequence_length, start_value, decrement):
    """
    Property: LSTM should detect decreasing trends.
    
    For any time series with monotonically decreasing values, the trend analysis
    logic should detect trend_direction as 'decreasing'.
    
    Note: This tests the _analyze_trend method directly rather than the full
    model prediction, since an untrained model produces random predictions.
    """
    lstm = get_mock_trained_lstm()
    
    if not TENSORFLOW_AVAILABLE or lstm.model is None:
        pytest.skip("TensorFlow not available or model not built")
    
    try:
        # Create monotonically decreasing predictions (simulating what a trained model would produce)
        predictions = create_monotonic_decreasing_sequence(sequence_length, start_value, decrement)
        
        # Clip to valid range
        predictions = np.clip(predictions, 0.0, 1.0)
        
        # Test the trend analysis logic directly
        trend_direction, trend_strength = lstm._analyze_trend(predictions)
        
        # Property: For monotonically decreasing predictions, trend should be 'decreasing'
        assert trend_direction == 'decreasing', \
            f"Expected 'decreasing' trend for monotonic decrease, got '{trend_direction}'"
        
        # Trend strength should be positive
        assert trend_strength >= 0.0, f"Trend strength should be non-negative, got {trend_strength}"
        
    except Exception as e:
        if "Model must be trained" in str(e):
            pytest.skip("Model not properly trained for testing")
        raise


def test_lstm_stable_trend_detection():
    """
    Test that LSTM detects stable trends for constant values.
    """
    lstm = get_mock_trained_lstm()
    
    if not TENSORFLOW_AVAILABLE or lstm.model is None:
        pytest.skip("TensorFlow not available or model not built")
    
    try:
        # Create constant NDVI values
        sequence_length = 30
        ndvi_values = np.full(sequence_length, 0.6)
        
        # Create dummy features
        temp_values = np.full(sequence_length, 25.0)
        humidity_values = np.full(sequence_length, 65.0)
        soil_moisture_values = np.full(sequence_length, 25.0)
        
        # Create sequences for LSTM
        X = []
        for i in range(lstm.sequence_length, len(ndvi_values)):
            seq = np.column_stack([
                ndvi_values[i-lstm.sequence_length:i],
                temp_values[i-lstm.sequence_length:i],
                humidity_values[i-lstm.sequence_length:i],
                soil_moisture_values[i-lstm.sequence_length:i]
            ])
            X.append(seq)
        
        if len(X) == 0:
            pytest.skip("Not enough data points for sequence")
        
        X = np.array(X)
        
        # Make predictions
        predictions, _, trend_direction, trend_strength = lstm.predict_trend(X, return_confidence=False)
        
        # For constant values, trend should be 'stable'
        assert trend_direction == 'stable', \
            f"Expected 'stable' trend for constant values, got '{trend_direction}'"
        
        # Trend strength should be low for stable trend
        assert trend_strength < 0.5, \
            f"Trend strength should be low for stable trend, got {trend_strength}"
        
    except Exception as e:
        if "Model must be trained" in str(e):
            pytest.skip("Model not properly trained for testing")
        raise


def test_lstm_confidence_intervals():
    """
    Test that LSTM confidence intervals are valid.
    """
    lstm = get_mock_trained_lstm()
    
    if not TENSORFLOW_AVAILABLE or lstm.model is None:
        pytest.skip("TensorFlow not available or model not built")
    
    try:
        # Create test data
        sequence_length = 30
        ndvi_values = create_monotonic_increasing_sequence(sequence_length, 0.4, 0.01)
        temp_values = np.linspace(20, 30, sequence_length)
        humidity_values = np.linspace(60, 70, sequence_length)
        soil_moisture_values = ndvi_values * 30 + 10
        
        # Create sequences
        X = []
        for i in range(lstm.sequence_length, len(ndvi_values)):
            seq = np.column_stack([
                ndvi_values[i-lstm.sequence_length:i],
                temp_values[i-lstm.sequence_length:i],
                humidity_values[i-lstm.sequence_length:i],
                soil_moisture_values[i-lstm.sequence_length:i]
            ])
            X.append(seq)
        
        if len(X) == 0:
            pytest.skip("Not enough data points for sequence")
        
        X = np.array(X)
        
        # Make predictions with confidence intervals
        predictions, confidence_intervals, _, _ = lstm.predict_trend(X, return_confidence=True)
        
        # Validate confidence intervals
        assert confidence_intervals is not None, "Confidence intervals should not be None"
        assert confidence_intervals.shape == (len(predictions), 2), \
            f"Expected shape ({len(predictions)}, 2), got {confidence_intervals.shape}"
        
        # Lower bound should be less than upper bound
        assert np.all(confidence_intervals[:, 0] <= confidence_intervals[:, 1]), \
            "Lower confidence bound should be <= upper bound"
        
        # Predictions should generally be within confidence intervals
        # (allowing some tolerance for Monte Carlo sampling)
        within_ci = np.sum(
            (predictions >= confidence_intervals[:, 0]) & 
            (predictions <= confidence_intervals[:, 1])
        )
        ratio = within_ci / len(predictions)
        assert ratio > 0.5, \
            f"Most predictions should be within confidence intervals, got {ratio:.2%}"
        
    except Exception as e:
        if "Model must be trained" in str(e):
            pytest.skip("Model not properly trained for testing")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
