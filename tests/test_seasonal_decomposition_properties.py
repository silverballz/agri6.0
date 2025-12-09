"""
Property-based tests for seasonal decomposition in temporal analysis.

Feature: production-enhancements, Property 23: Seasonal decomposition completeness
Validates: Requirements 6.4
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.trend_analyzer import TrendAnalyzer


# Custom strategies for generating time series data
@st.composite
def time_series_with_seasonality(draw):
    """
    Generate time series data with seasonal pattern, trend, and noise
    
    Returns:
        tuple: (time_series, period, expected_seasonal_amplitude)
    """
    # Generate parameters
    period = draw(st.integers(min_value=12, max_value=52))  # 12-52 for weekly/monthly patterns
    length = draw(st.integers(min_value=2 * period, max_value=5 * period))
    
    # Trend parameters
    trend_slope = draw(st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False))
    trend_intercept = draw(st.floats(min_value=0.4, max_value=0.7, allow_nan=False, allow_infinity=False))
    
    # Seasonal parameters
    seasonal_amplitude = draw(st.floats(min_value=0.05, max_value=0.15, allow_nan=False, allow_infinity=False))
    
    # Noise parameters
    noise_std = draw(st.floats(min_value=0.01, max_value=0.03, allow_nan=False, allow_infinity=False))
    
    # Generate time series components
    t = np.arange(length)
    
    # Trend component
    trend = trend_slope * t + trend_intercept
    
    # Seasonal component (sinusoidal)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / period)
    
    # Noise component
    noise = np.random.normal(0, noise_std, length)
    
    # Combine components
    y = trend + seasonal + noise
    
    # Clip to valid NDVI range
    y = np.clip(y, 0.0, 1.0)
    
    # Create pandas Series with datetime index
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=length, freq='D')
    time_series = pd.Series(y, index=dates)
    
    return time_series, period, seasonal_amplitude


@st.composite
def time_series_with_known_components(draw):
    """
    Generate time series with known trend, seasonal, and residual components
    for exact verification
    
    Returns:
        tuple: (time_series, trend, seasonal, residual, period)
    """
    period = draw(st.integers(min_value=12, max_value=30))
    length = draw(st.integers(min_value=2 * period, max_value=4 * period))
    
    # Generate components explicitly
    t = np.arange(length)
    
    # Simple linear trend
    trend_slope = draw(st.floats(min_value=-0.0005, max_value=0.0005, allow_nan=False, allow_infinity=False))
    trend_intercept = draw(st.floats(min_value=0.5, max_value=0.6, allow_nan=False, allow_infinity=False))
    trend = trend_slope * t + trend_intercept
    
    # Pure sinusoidal seasonal
    seasonal_amplitude = draw(st.floats(min_value=0.05, max_value=0.1, allow_nan=False, allow_infinity=False))
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / period)
    
    # Small residual
    residual_std = draw(st.floats(min_value=0.005, max_value=0.02, allow_nan=False, allow_infinity=False))
    residual = np.random.normal(0, residual_std, length)
    
    # Combine
    y = trend + seasonal + residual
    y = np.clip(y, 0.0, 1.0)
    
    # Create Series
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=length, freq='D')
    time_series = pd.Series(y, index=dates)
    
    return time_series, trend, seasonal, residual, period


class TestSeasonalDecompositionProperties:
    """
    Property-based tests for seasonal decomposition
    
    Property 23: For any time series with length >= 2*period, decomposition 
    should produce trend, seasonal, and residual components that sum to the original series
    """
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_seasonality())
    def test_property_decomposition_completeness(self, data):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: For any time series with length >= 2*period, decomposition should 
        produce trend, seasonal, and residual components that sum to the original series.
        
        This is the fundamental property of additive decomposition:
        original = trend + seasonal + residual
        """
        time_series, period, seasonal_amplitude = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 2 * period)
        
        # Skip if time series has no variance
        assume(time_series.std() > 0.001)
        
        # Create analyzer and decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        # Check that decomposition succeeded
        if result is None:
            pytest.skip("Seasonal decomposition not available (statsmodels not installed)")
        
        # Extract components
        trend = result['trend']
        seasonal = result['seasonal']
        residual = result['residual']
        
        # Verify all components have same length as original
        assert len(trend) == len(time_series), \
            f"Trend length {len(trend)} should match time series length {len(time_series)}"
        assert len(seasonal) == len(time_series), \
            f"Seasonal length {len(seasonal)} should match time series length {len(time_series)}"
        assert len(residual) == len(time_series), \
            f"Residual length {len(residual)} should match time series length {len(time_series)}"
        
        # Property 23: Components should sum to original series
        # trend + seasonal + residual = original
        reconstructed = trend + seasonal + residual
        
        # Remove NaN values for comparison (decomposition may have NaN at edges)
        valid_mask = ~(np.isnan(trend) | np.isnan(seasonal) | np.isnan(residual))
        
        if valid_mask.sum() > 0:
            original_valid = time_series.values[valid_mask]
            reconstructed_valid = reconstructed.values[valid_mask]
            
            # Check that reconstruction matches original (within numerical tolerance)
            np.testing.assert_allclose(
                reconstructed_valid,
                original_valid,
                rtol=1e-5,
                atol=1e-8,
                err_msg="Decomposition components should sum to original series"
            )
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_seasonality())
    def test_property_seasonal_periodicity(self, data):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: The seasonal component should exhibit periodicity matching 
        the specified period.
        """
        time_series, period, seasonal_amplitude = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 2 * period)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        if result is None:
            pytest.skip("Seasonal decomposition not available")
        
        seasonal = result['seasonal']
        
        # Remove NaN values
        seasonal_clean = seasonal.dropna()
        
        if len(seasonal_clean) >= 2 * period:
            # Check that seasonal component repeats with the specified period
            # Compare values that are one period apart
            for i in range(len(seasonal_clean) - period):
                val1 = seasonal_clean.iloc[i]
                val2 = seasonal_clean.iloc[i + period]
                
                # Values one period apart should be very similar
                # Allow some tolerance due to edge effects and numerical precision
                assert abs(val1 - val2) < 0.1, \
                    f"Seasonal values {period} steps apart should be similar: " \
                    f"{val1:.4f} vs {val2:.4f} at positions {i} and {i + period}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_seasonality())
    def test_property_seasonal_mean_zero(self, data):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: The seasonal component should have mean approximately zero
        (for additive decomposition).
        """
        time_series, period, seasonal_amplitude = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 2 * period)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        if result is None:
            pytest.skip("Seasonal decomposition not available")
        
        seasonal = result['seasonal']
        
        # Remove NaN values
        seasonal_clean = seasonal.dropna()
        
        if len(seasonal_clean) > 0:
            seasonal_mean = seasonal_clean.mean()
            
            # Seasonal component should have mean close to zero
            # Allow tolerance based on the amplitude
            # Use 20% of amplitude to account for edge effects and small sample sizes
            tolerance = 0.2 * seasonal_amplitude if seasonal_amplitude > 0 else 0.02
            
            assert abs(seasonal_mean) < tolerance, \
                f"Seasonal component mean {seasonal_mean:.6f} should be close to zero " \
                f"(tolerance: {tolerance:.6f})"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_seasonality())
    def test_property_residual_mean_zero(self, data):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: The residual component should have mean approximately zero.
        """
        time_series, period, seasonal_amplitude = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 2 * period)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        if result is None:
            pytest.skip("Seasonal decomposition not available")
        
        residual = result['residual']
        
        # Remove NaN values
        residual_clean = residual.dropna()
        
        if len(residual_clean) > 0:
            residual_mean = residual_clean.mean()
            residual_std = residual_clean.std()
            
            # Residual mean should be close to zero
            # Allow tolerance based on standard deviation
            tolerance = 0.5 * residual_std if residual_std > 0 else 0.01
            
            assert abs(residual_mean) < tolerance, \
                f"Residual mean {residual_mean:.6f} should be close to zero " \
                f"(tolerance: {tolerance:.6f}, std: {residual_std:.6f})"
    
    @settings(max_examples=50, deadline=None)
    @given(data=time_series_with_known_components())
    def test_property_trend_extraction(self, data):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: The trend component should capture the long-term direction,
        removing seasonal variations.
        """
        time_series, expected_trend, expected_seasonal, expected_residual, period = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 2 * period)
        
        # Create analyzer and decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        if result is None:
            pytest.skip("Seasonal decomposition not available")
        
        trend = result['trend']
        
        # Remove NaN values
        trend_clean = trend.dropna()
        expected_trend_clean = expected_trend[~np.isnan(trend.values)]
        
        if len(trend_clean) > period:
            # Trend should be smoother than original series
            # Calculate variance of differences
            trend_diff_var = np.var(np.diff(trend_clean))
            original_diff_var = np.var(np.diff(time_series.dropna()))
            
            # Trend should have lower variance in differences (smoother)
            assert trend_diff_var <= original_diff_var, \
                f"Trend should be smoother than original series: " \
                f"trend_var={trend_diff_var:.6f}, original_var={original_diff_var:.6f}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_seasonality())
    def test_property_seasonal_amplitude_reasonable(self, data):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: The seasonal component amplitude should be reasonable relative
        to the original series range.
        """
        time_series, period, expected_seasonal_amplitude = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 2 * period)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        if result is None:
            pytest.skip("Seasonal decomposition not available")
        
        seasonal = result['seasonal']
        seasonal_amplitude = result['seasonal_amplitude']
        
        # Remove NaN values
        seasonal_clean = seasonal.dropna()
        
        if len(seasonal_clean) > 0:
            # Seasonal amplitude should not exceed the range of the original series
            original_range = time_series.max() - time_series.min()
            
            assert seasonal_amplitude <= original_range, \
                f"Seasonal amplitude {seasonal_amplitude:.4f} should not exceed " \
                f"original series range {original_range:.4f}"
            
            # Seasonal amplitude should be positive
            assert seasonal_amplitude >= 0, \
                f"Seasonal amplitude should be non-negative, got {seasonal_amplitude:.4f}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_seasonality())
    def test_property_explanations_provided(self, data):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: Decomposition should provide explanations for each component.
        """
        time_series, period, seasonal_amplitude = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 2 * period)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        if result is None:
            pytest.skip("Seasonal decomposition not available")
        
        # Check that explanations exist
        assert 'explanations' in result, "Result should contain explanations"
        
        explanations = result['explanations']
        
        # Check that all component explanations exist
        assert 'trend' in explanations, "Should have trend explanation"
        assert 'seasonal' in explanations, "Should have seasonal explanation"
        assert 'residual' in explanations, "Should have residual explanation"
        
        # Check that explanations are non-empty strings
        assert isinstance(explanations['trend'], str) and len(explanations['trend']) > 0, \
            "Trend explanation should be a non-empty string"
        assert isinstance(explanations['seasonal'], str) and len(explanations['seasonal']) > 0, \
            "Seasonal explanation should be a non-empty string"
        assert isinstance(explanations['residual'], str) and len(explanations['residual']) > 0, \
            "Residual explanation should be a non-empty string"
        
        # Check that trend direction is provided
        assert 'trend_direction' in result, "Should have trend direction"
        assert result['trend_direction'] in ['increasing', 'decreasing'], \
            f"Trend direction should be 'increasing' or 'decreasing', got {result['trend_direction']}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        period=st.integers(min_value=12, max_value=30),
        length=st.integers(min_value=50, max_value=200),
        base_value=st.floats(min_value=0.4, max_value=0.7, allow_nan=False, allow_infinity=False),
        noise_std=st.floats(min_value=0.01, max_value=0.03, allow_nan=False, allow_infinity=False)
    )
    def test_property_flat_trend_detection(self, period, length, base_value, noise_std):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: For a time series with no trend (flat), the trend component
        should be approximately constant.
        """
        # Skip if too short
        assume(length >= 2 * period)
        
        # Generate flat time series with seasonal pattern
        t = np.arange(length)
        seasonal = 0.1 * np.sin(2 * np.pi * t / period)
        noise = np.random.normal(0, noise_std, length)
        y = base_value + seasonal + noise
        y = np.clip(y, 0.0, 1.0)
        
        # Create Series
        start_date = datetime(2020, 1, 1)
        dates = pd.date_range(start=start_date, periods=length, freq='D')
        time_series = pd.Series(y, index=dates)
        
        # Decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        if result is None:
            pytest.skip("Seasonal decomposition not available")
        
        trend = result['trend']
        trend_clean = trend.dropna()
        
        if len(trend_clean) > period:
            # Trend should be relatively flat (low variance)
            trend_std = trend_clean.std()
            
            # For a flat series, trend std should be small
            assert trend_std < 0.1, \
                f"Trend should be relatively flat for no-trend series, got std={trend_std:.4f}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        period=st.integers(min_value=12, max_value=30),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_property_insufficient_data_handling(self, period, seed):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: When time series length < 2*period, decomposition should
        return None or handle gracefully.
        """
        np.random.seed(seed)
        
        # Generate short time series (less than 2*period)
        length = period + 5  # Intentionally too short
        y = np.random.normal(0.5, 0.1, length)
        y = np.clip(y, 0.0, 1.0)
        
        # Create Series
        start_date = datetime(2020, 1, 1)
        dates = pd.date_range(start=start_date, periods=length, freq='D')
        time_series = pd.Series(y, index=dates)
        
        # Decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        # Should return None for insufficient data
        assert result is None, \
            f"Decomposition should return None for insufficient data (length={length}, need {2*period})"
    
    @settings(max_examples=50, deadline=None)
    @given(
        period=st.integers(min_value=12, max_value=30),
        length=st.integers(min_value=50, max_value=150),
        constant_value=st.floats(min_value=0.4, max_value=0.7, allow_nan=False, allow_infinity=False)
    )
    def test_property_constant_series_handling(self, period, length, constant_value):
        """
        **Feature: production-enhancements, Property 23: Seasonal decomposition completeness**
        **Validates: Requirements 6.4**
        
        Property: For a constant time series (no variation), decomposition
        should handle gracefully.
        """
        # Skip if too short
        assume(length >= 2 * period)
        
        # Generate constant series
        y = np.full(length, constant_value)
        
        # Create Series
        start_date = datetime(2020, 1, 1)
        dates = pd.date_range(start=start_date, periods=length, freq='D')
        time_series = pd.Series(y, index=dates)
        
        # Decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        # Should either return None or handle gracefully
        # If it returns a result, components should be valid
        if result is not None:
            trend = result['trend']
            seasonal = result['seasonal']
            residual = result['residual']
            
            # All components should be defined (may contain NaN)
            assert trend is not None
            assert seasonal is not None
            assert residual is not None
            
            # Seasonal amplitude should be very small for constant series
            assert result['seasonal_amplitude'] < 0.01, \
                f"Seasonal amplitude should be near zero for constant series, " \
                f"got {result['seasonal_amplitude']:.4f}"


class TestSeasonalDecompositionEdgeCases:
    """Edge case tests for seasonal decomposition"""
    
    @settings(max_examples=50, deadline=None)
    @given(
        period=st.integers(min_value=12, max_value=30),
        length=st.integers(min_value=50, max_value=150),
        trend_slope=st.floats(min_value=0.001, max_value=0.005, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_strong_trend_extraction(self, period, length, trend_slope, seed):
        """
        Test that decomposition correctly extracts a strong linear trend.
        """
        np.random.seed(seed)
        
        # Skip if too short
        assume(length >= 2 * period)
        
        # Generate series with strong trend
        t = np.arange(length)
        trend = 0.5 + trend_slope * t
        seasonal = 0.05 * np.sin(2 * np.pi * t / period)
        noise = np.random.normal(0, 0.01, length)
        y = trend + seasonal + noise
        y = np.clip(y, 0.0, 1.0)
        
        # Create Series
        start_date = datetime(2020, 1, 1)
        dates = pd.date_range(start=start_date, periods=length, freq='D')
        time_series = pd.Series(y, index=dates)
        
        # Decompose
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_seasonal(time_series, period=period)
        
        if result is None:
            pytest.skip("Seasonal decomposition not available")
        
        # Check trend direction
        assert result['trend_direction'] == 'increasing', \
            f"Should detect increasing trend, got {result['trend_direction']}"
        
        # Trend should increase from start to end
        trend_component = result['trend'].dropna()
        if len(trend_component) > 10:
            assert trend_component.iloc[-1] > trend_component.iloc[0], \
                "Trend should increase from start to end"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
