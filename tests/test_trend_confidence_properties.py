"""
Property-based tests for trend line confidence intervals

Feature: production-enhancements, Property 25: Trend line confidence intervals
Validates: Requirements 6.2
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
def time_series_with_trend(draw):
    """
    Generate time series data with a linear trend and noise
    
    Returns:
        tuple: (time_series, expected_slope, expected_intercept)
    """
    # Generate parameters
    length = draw(st.integers(min_value=20, max_value=100))
    slope = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False))
    intercept = draw(st.floats(min_value=0.3, max_value=0.8, allow_nan=False, allow_infinity=False))
    noise_std = draw(st.floats(min_value=0.001, max_value=0.05, allow_nan=False, allow_infinity=False))
    
    # Generate time series with linear trend + noise
    x = np.arange(length)
    y = slope * x + intercept + np.random.normal(0, noise_std, length)
    
    # Clip to valid NDVI range
    y = np.clip(y, 0.0, 1.0)
    
    # Create pandas Series
    time_series = pd.Series(y)
    
    return time_series, slope, intercept, noise_std


@st.composite
def time_series_with_known_outliers(draw):
    """
    Generate time series with known outliers to test confidence interval coverage
    
    Returns:
        tuple: (time_series, outlier_indices)
    """
    # Generate base parameters
    length = draw(st.integers(min_value=30, max_value=100))
    slope = draw(st.floats(min_value=-0.005, max_value=0.005, allow_nan=False, allow_infinity=False))
    intercept = draw(st.floats(min_value=0.4, max_value=0.7, allow_nan=False, allow_infinity=False))
    noise_std = draw(st.floats(min_value=0.01, max_value=0.03, allow_nan=False, allow_infinity=False))
    
    # Generate clean time series
    x = np.arange(length)
    y = slope * x + intercept + np.random.normal(0, noise_std, length)
    
    # Add a few outliers (but not too many)
    num_outliers = draw(st.integers(min_value=0, max_value=min(3, length // 10)))
    outlier_indices = []
    
    if num_outliers > 0:
        outlier_positions = draw(st.lists(
            st.integers(min_value=5, max_value=length-5),
            min_size=num_outliers,
            max_size=num_outliers,
            unique=True
        ))
        
        for idx in outlier_positions:
            # Add significant deviation (3-5 standard deviations)
            outlier_magnitude = draw(st.floats(min_value=3.0, max_value=5.0))
            direction = draw(st.sampled_from([-1, 1]))
            y[idx] += direction * outlier_magnitude * noise_std
            outlier_indices.append(idx)
    
    # Clip to valid range
    y = np.clip(y, 0.0, 1.0)
    
    time_series = pd.Series(y)
    
    return time_series, outlier_indices


class TestTrendLineConfidenceProperties:
    """
    Property-based tests for trend line confidence intervals
    
    Property 25: For any fitted regression model, 95% confidence intervals 
    should contain approximately 95% of actual data points
    """
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_trend())
    def test_property_confidence_interval_coverage_rate(self, data):
        """
        **Feature: production-enhancements, Property 25: Trend line confidence intervals**
        **Validates: Requirements 6.2**
        
        Property: For any time series with linear trend and noise, 
        the 95% confidence intervals should contain approximately 95% of data points.
        
        We allow some tolerance since:
        1. Small sample sizes may have slightly different coverage
        2. Random noise can cause variation
        3. We're testing the statistical property holds in general
        """
        time_series, expected_slope, expected_intercept, noise_std = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 10)
        
        # Skip if all values are identical (no variance)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and fit regression
        analyzer = TrendAnalyzer()
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Get confidence intervals
        predictions = result['predictions']
        confidence_lower = result['confidence_lower']
        confidence_upper = result['confidence_upper']
        
        # Check that confidence intervals are valid
        assert len(predictions) == len(time_series), \
            "Predictions length should match time series length"
        assert len(confidence_lower) == len(time_series), \
            "Lower confidence bound length should match time series length"
        assert len(confidence_upper) == len(time_series), \
            "Upper confidence bound length should match time series length"
        
        # Check that lower bound <= prediction <= upper bound
        assert np.all(confidence_lower <= predictions), \
            "Lower confidence bound should be <= predictions"
        assert np.all(predictions <= confidence_upper), \
            "Predictions should be <= upper confidence bound"
        
        # Calculate coverage: percentage of actual points within confidence interval
        actual_values = time_series.values
        within_interval = (actual_values >= confidence_lower) & (actual_values <= confidence_upper)
        coverage_rate = np.sum(within_interval) / len(time_series)
        
        # For 95% confidence intervals, we expect approximately 95% coverage
        # Allow tolerance of ±15% to account for:
        # - Small sample sizes (n=10-100)
        # - Random variation in noise
        # - Non-normal residuals in some cases
        # This means we accept coverage between 80% and 100%
        min_acceptable_coverage = 0.80
        max_acceptable_coverage = 1.00
        
        assert min_acceptable_coverage <= coverage_rate <= max_acceptable_coverage, \
            f"Coverage rate {coverage_rate:.2%} outside acceptable range " \
            f"[{min_acceptable_coverage:.0%}, {max_acceptable_coverage:.0%}]. " \
            f"For 95% CI, expected ~95% coverage. " \
            f"Points within interval: {np.sum(within_interval)}/{len(time_series)}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_trend())
    def test_property_confidence_interval_width_increases_with_noise(self, data):
        """
        **Feature: production-enhancements, Property 25: Trend line confidence intervals**
        **Validates: Requirements 6.2**
        
        Property: Confidence interval width should be proportional to the 
        standard error of residuals. Higher noise should lead to wider intervals.
        """
        time_series, expected_slope, expected_intercept, noise_std = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 10)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and fit regression
        analyzer = TrendAnalyzer()
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Calculate confidence interval width
        confidence_lower = result['confidence_lower']
        confidence_upper = result['confidence_upper']
        ci_width = confidence_upper - confidence_lower
        
        # Calculate residual standard error
        predictions = result['predictions']
        actual_values = time_series.values
        residuals = actual_values - predictions
        residual_std = np.std(residuals)
        
        # Confidence interval width should be approximately 2 * 1.96 * std_error
        # (1.96 is the z-score for 95% CI)
        expected_ci_width = 2 * 1.96 * residual_std
        
        # Check that all CI widths are consistent (should be constant for linear regression)
        ci_width_std = np.std(ci_width)
        ci_width_mean = np.mean(ci_width)
        
        # CI width should be relatively constant across the time series
        # (for simple linear regression with homoscedastic errors)
        assert ci_width_std / ci_width_mean < 0.1, \
            f"Confidence interval width should be relatively constant. " \
            f"Std/Mean ratio: {ci_width_std / ci_width_mean:.3f}"
        
        # Check that CI width is approximately correct
        # Allow 50% tolerance due to small sample effects
        assert 0.5 * expected_ci_width <= ci_width_mean <= 2.0 * expected_ci_width, \
            f"Mean CI width {ci_width_mean:.4f} should be approximately " \
            f"{expected_ci_width:.4f} (2 * 1.96 * {residual_std:.4f})"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_known_outliers())
    def test_property_confidence_interval_handles_outliers(self, data):
        """
        **Feature: production-enhancements, Property 25: Trend line confidence intervals**
        **Validates: Requirements 6.2**
        
        Property: Even with outliers present, confidence intervals should still 
        provide reasonable coverage of the main trend (excluding extreme outliers).
        """
        time_series, outlier_indices = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 15)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and fit regression
        analyzer = TrendAnalyzer()
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Get confidence intervals
        predictions = result['predictions']
        confidence_lower = result['confidence_lower']
        confidence_upper = result['confidence_upper']
        actual_values = time_series.values
        
        # Calculate coverage excluding known outliers
        non_outlier_mask = np.ones(len(time_series), dtype=bool)
        if outlier_indices:
            non_outlier_mask[outlier_indices] = False
        
        # Check coverage on non-outlier points
        within_interval = (actual_values >= confidence_lower) & (actual_values <= confidence_upper)
        non_outlier_coverage = np.sum(within_interval & non_outlier_mask) / np.sum(non_outlier_mask)
        
        # Non-outlier points should have good coverage (>70%)
        # This is lower than 95% because outliers affect the CI width
        assert non_outlier_coverage >= 0.70, \
            f"Non-outlier coverage {non_outlier_coverage:.2%} should be >= 70%. " \
            f"Points within interval: {np.sum(within_interval & non_outlier_mask)}/{np.sum(non_outlier_mask)}"
        
        # Overall coverage (including outliers) should still be reasonable
        overall_coverage = np.sum(within_interval) / len(time_series)
        assert overall_coverage >= 0.60, \
            f"Overall coverage {overall_coverage:.2%} should be >= 60% even with outliers"
    
    @settings(max_examples=100, deadline=None)
    @given(
        length=st.integers(min_value=20, max_value=100),
        slope=st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False),
        intercept=st.floats(min_value=0.3, max_value=0.8, allow_nan=False, allow_infinity=False),
        noise_std=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False)
    )
    def test_property_confidence_interval_symmetry(self, length, slope, intercept, noise_std):
        """
        **Feature: production-enhancements, Property 25: Trend line confidence intervals**
        **Validates: Requirements 6.2**
        
        Property: Confidence intervals should be symmetric around the prediction line.
        Upper bound - prediction should equal prediction - lower bound.
        """
        # Generate time series
        x = np.arange(length)
        y = slope * x + intercept + np.random.normal(0, noise_std, length)
        y = np.clip(y, 0.0, 1.0)
        time_series = pd.Series(y)
        
        # Skip if no variance
        assume(time_series.std() > 0.001)
        
        # Create analyzer and fit regression
        analyzer = TrendAnalyzer()
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Get intervals
        predictions = result['predictions']
        confidence_lower = result['confidence_lower']
        confidence_upper = result['confidence_upper']
        
        # Calculate distances from prediction to bounds
        upper_distance = confidence_upper - predictions
        lower_distance = predictions - confidence_lower
        
        # Check symmetry (should be equal within numerical precision)
        np.testing.assert_allclose(
            upper_distance,
            lower_distance,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Confidence intervals should be symmetric around predictions"
        )
    
    @settings(max_examples=50, deadline=None)
    @given(
        length=st.integers(min_value=30, max_value=100),
        base_value=st.floats(min_value=0.4, max_value=0.7, allow_nan=False, allow_infinity=False),
        noise_std=st.floats(min_value=0.01, max_value=0.03, allow_nan=False, allow_infinity=False)
    )
    def test_property_confidence_interval_for_flat_trend(self, length, base_value, noise_std):
        """
        **Feature: production-enhancements, Property 25: Trend line confidence intervals**
        **Validates: Requirements 6.2**
        
        Property: For a flat trend (slope ≈ 0), confidence intervals should 
        still provide appropriate coverage around the mean value.
        """
        # Generate flat time series (no trend)
        y = base_value + np.random.normal(0, noise_std, length)
        y = np.clip(y, 0.0, 1.0)
        time_series = pd.Series(y)
        
        # Skip if no variance
        assume(time_series.std() > 0.001)
        
        # Create analyzer and fit regression
        analyzer = TrendAnalyzer()
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Check that slope is close to zero
        assert abs(result['slope']) < 0.01, \
            f"Slope should be close to zero for flat trend, got {result['slope']:.4f}"
        
        # Check coverage
        predictions = result['predictions']
        confidence_lower = result['confidence_lower']
        confidence_upper = result['confidence_upper']
        actual_values = time_series.values
        
        within_interval = (actual_values >= confidence_lower) & (actual_values <= confidence_upper)
        coverage_rate = np.sum(within_interval) / len(time_series)
        
        # Should still have good coverage for flat trend
        assert coverage_rate >= 0.80, \
            f"Coverage rate {coverage_rate:.2%} should be >= 80% for flat trend"
    
    @settings(max_examples=50, deadline=None)
    @given(data=time_series_with_trend())
    def test_property_r_squared_correlates_with_coverage(self, data):
        """
        **Feature: production-enhancements, Property 25: Trend line confidence intervals**
        **Validates: Requirements 6.2**
        
        Property: Higher R² values should generally correspond to better 
        confidence interval coverage, as the model fits the data better.
        """
        time_series, expected_slope, expected_intercept, noise_std = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 15)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and fit regression
        analyzer = TrendAnalyzer()
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Get R² and coverage
        r_squared = result['r_squared']
        
        predictions = result['predictions']
        confidence_lower = result['confidence_lower']
        confidence_upper = result['confidence_upper']
        actual_values = time_series.values
        
        within_interval = (actual_values >= confidence_lower) & (actual_values <= confidence_upper)
        coverage_rate = np.sum(within_interval) / len(time_series)
        
        # For high R² (good fit), expect high coverage
        if r_squared > 0.8:
            assert coverage_rate >= 0.85, \
                f"High R² ({r_squared:.3f}) should have high coverage, got {coverage_rate:.2%}"
        
        # For moderate R² (decent fit), expect moderate coverage
        if 0.5 <= r_squared <= 0.8:
            assert coverage_rate >= 0.75, \
                f"Moderate R² ({r_squared:.3f}) should have moderate coverage, got {coverage_rate:.2%}"
        
        # For low R² (poor fit), still expect reasonable coverage
        # (CI should widen to accommodate poor fit)
        if r_squared < 0.5:
            assert coverage_rate >= 0.65, \
                f"Low R² ({r_squared:.3f}) should still have reasonable coverage, got {coverage_rate:.2%}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
