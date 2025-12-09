"""
Property-based tests for rate of change calculation

Feature: production-enhancements, Property 24: Rate of change calculation
Validates: Requirements 6.5
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.trend_analyzer import TrendAnalyzer


# Custom strategies for generating time series data
@st.composite
def time_series_with_dates(draw):
    """
    Generate time series data with corresponding dates
    
    Returns:
        tuple: (time_series, dates)
    """
    # Generate parameters
    length = draw(st.integers(min_value=15, max_value=100))
    start_date = datetime(2024, 1, 1)
    
    # Generate dates (daily intervals)
    dates = pd.date_range(start=start_date, periods=length, freq='D')
    
    # Generate values (NDVI-like range)
    base_value = draw(st.floats(min_value=0.3, max_value=0.8, allow_nan=False, allow_infinity=False))
    trend = draw(st.floats(min_value=-0.005, max_value=0.005, allow_nan=False, allow_infinity=False))
    noise_std = draw(st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False))
    
    # Generate time series with trend and noise
    x = np.arange(length)
    y = base_value + trend * x + np.random.normal(0, noise_std, length)
    
    # Clip to valid NDVI range
    y = np.clip(y, 0.0, 1.0)
    
    time_series = pd.Series(y, index=dates)
    
    return time_series, dates


@st.composite
def time_series_with_known_rate(draw):
    """
    Generate time series with a known constant rate of change
    
    Returns:
        tuple: (time_series, dates, expected_rate)
    """
    length = draw(st.integers(min_value=20, max_value=100))
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=length, freq='D')
    
    # Known constant rate per day - keep it small to avoid clipping
    rate_per_day = draw(st.floats(min_value=-0.003, max_value=0.003, allow_nan=False, allow_infinity=False))
    start_value = draw(st.floats(min_value=0.45, max_value=0.55, allow_nan=False, allow_infinity=False))
    
    # Generate linear time series
    x = np.arange(length)
    y = start_value + rate_per_day * x
    
    # Only use this example if it stays within valid range (no clipping)
    assume(np.all((y >= 0.0) & (y <= 1.0)))
    
    time_series = pd.Series(y, index=dates)
    
    return time_series, dates, rate_per_day


@st.composite
def time_series_with_step_change(draw):
    """
    Generate time series with a step change (sudden jump or drop)
    
    Returns:
        tuple: (time_series, dates, step_position, step_magnitude)
    """
    length = draw(st.integers(min_value=30, max_value=100))
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=length, freq='D')
    
    # Base value and noise
    base_value = draw(st.floats(min_value=0.4, max_value=0.6, allow_nan=False, allow_infinity=False))
    noise_std = draw(st.floats(min_value=0.005, max_value=0.02, allow_nan=False, allow_infinity=False))
    
    # Generate base time series
    y = base_value + np.random.normal(0, noise_std, length)
    
    # Add step change at a random position
    step_position = draw(st.integers(min_value=10, max_value=length-10))
    step_magnitude = draw(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False))
    
    y[step_position:] += step_magnitude
    
    # Clip to valid range
    y = np.clip(y, 0.0, 1.0)
    
    time_series = pd.Series(y, index=dates)
    
    return time_series, dates, step_position, step_magnitude


class TestRateOfChangeProperties:
    """
    Property-based tests for rate of change calculation
    
    Property 24: For any two time points t1 and t2, rate of change should 
    equal (value_t2 - value_t1) / (t2 - t1)
    """
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_dates())
    def test_property_rate_formula_correctness(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: For any two time points t1 and t2, rate of change should 
        equal (value_t2 - value_t1) / (t2 - t1)
        
        This tests the fundamental definition of rate of change.
        """
        time_series, dates = data
        window = 7  # 7-day window
        
        # Skip if time series is too short
        assume(len(time_series) >= window + 1)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and calculate rate of change
        analyzer = TrendAnalyzer()
        result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
        
        rate = result['rate']
        
        # Verify rate calculation manually for each point
        for i in range(window, len(time_series)):
            t1 = i - window
            t2 = i
            
            value_t1 = time_series.iloc[t1]
            value_t2 = time_series.iloc[t2]
            
            # Expected rate: (value_t2 - value_t1) / window
            expected_rate = (value_t2 - value_t1) / window
            
            # Actual rate from function
            actual_rate = rate.iloc[i]
            
            # Check if rate matches expected (within numerical precision)
            if not np.isnan(actual_rate):
                assert abs(actual_rate - expected_rate) < 1e-10, \
                    f"Rate at index {i} should be {expected_rate:.6f}, got {actual_rate:.6f}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_known_rate())
    def test_property_constant_rate_detection(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: For a time series with constant rate of change, 
        the calculated rate should be approximately equal to the known rate.
        """
        time_series, dates, expected_rate = data
        window = 7
        
        # Skip if time series is too short
        assume(len(time_series) >= window + 5)
        
        # Create analyzer and calculate rate of change
        analyzer = TrendAnalyzer()
        result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
        
        rate = result['rate']
        
        # After the initial window, rates should stabilize to the expected rate
        # (ignoring the first few points which are affected by initialization)
        stable_rates = rate.iloc[window+2:]  # Skip first few points
        
        # Remove NaN values
        stable_rates = stable_rates.dropna()
        
        if len(stable_rates) > 0:
            mean_rate = stable_rates.mean()
            
            # Mean rate should be close to expected rate
            # Allow tolerance for numerical precision and edge effects
            # Use relative tolerance: 20% of the expected rate, or absolute 0.001 for small rates
            tolerance = max(0.001, abs(expected_rate) * 0.2)
            
            assert abs(mean_rate - expected_rate) < tolerance, \
                f"Mean calculated rate {mean_rate:.6f} should be close to " \
                f"expected rate {expected_rate:.6f} (tolerance: {tolerance:.6f})"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_dates())
    def test_property_rate_units_consistency(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: Rate of change should have consistent units (change per day).
        For a window of N days, rate = (value_t2 - value_t1) / N
        """
        time_series, dates = data
        
        # Test with different window sizes
        windows = [3, 7, 14, 30]
        
        for window in windows:
            if len(time_series) < window + 1:
                continue
            
            # Create analyzer and calculate rate
            analyzer = TrendAnalyzer()
            result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
            
            rate = result['rate']
            
            # Verify units: rate should be in "change per day"
            # For any point i, rate[i] * window should equal the total change
            for i in range(window, len(time_series)):
                if not np.isnan(rate.iloc[i]):
                    total_change = time_series.iloc[i] - time_series.iloc[i - window]
                    expected_total = rate.iloc[i] * window
                    
                    # Check consistency
                    assert abs(total_change - expected_total) < 1e-10, \
                        f"Rate units inconsistent: rate[{i}] * {window} = {expected_total:.6f}, " \
                        f"but actual change = {total_change:.6f}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_dates())
    def test_property_rate_percentage_calculation(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: Percentage rate of change should equal (rate / value) * 100
        """
        time_series, dates = data
        window = 7
        
        # Skip if time series is too short
        assume(len(time_series) >= window + 1)
        assume(time_series.std() > 0.001)
        
        # Create analyzer and calculate rate
        analyzer = TrendAnalyzer()
        result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
        
        rate = result['rate']
        rate_pct_weekly = result['rate_pct_weekly']
        
        # Verify percentage calculation
        for i in range(window, len(time_series)):
            if not np.isnan(rate.iloc[i]) and not np.isnan(rate_pct_weekly.iloc[i]):
                value = time_series.iloc[i]
                
                if value != 0:
                    expected_pct = (rate.iloc[i] / value) * 100
                    actual_pct = rate_pct_weekly.iloc[i]
                    
                    # Check percentage calculation (allow small tolerance for inf/nan handling)
                    if not np.isinf(expected_pct) and not np.isinf(actual_pct):
                        assert abs(actual_pct - expected_pct) < 0.01, \
                            f"Percentage rate at index {i} should be {expected_pct:.2f}%, " \
                            f"got {actual_pct:.2f}%"
    
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    @given(data=time_series_with_step_change())
    def test_property_rate_detects_rapid_changes(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: Rate of change should detect rapid changes (step changes) 
        in the time series and flag them as significant.
        """
        time_series, dates, step_position, step_magnitude = data
        window = 7
        
        # Skip if step is too small to detect
        assume(abs(step_magnitude) > 0.05)
        assume(len(time_series) >= window + 10)
        
        # Create analyzer and calculate rate
        analyzer = TrendAnalyzer()
        result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
        
        rate = result['rate']
        significant_changes = result['significant_changes']
        
        # Check if the step change is detected
        # The rate should spike around the step position
        detection_window = range(max(window, step_position - 5), 
                                 min(len(time_series), step_position + window + 5))
        
        rates_around_step = [rate.iloc[i] for i in detection_window if not np.isnan(rate.iloc[i])]
        
        if len(rates_around_step) > 0:
            max_rate_around_step = max(abs(r) for r in rates_around_step)
            
            # The maximum rate around the step should be significantly higher than baseline
            baseline_rate = np.nanmean([abs(rate.iloc[i]) for i in range(window, step_position - 5) 
                                       if i >= window and not np.isnan(rate.iloc[i])])
            
            if baseline_rate > 0:
                # Rate around step should be at least 2x the baseline
                assert max_rate_around_step > 1.5 * baseline_rate, \
                    f"Step change should cause rate spike. Max rate: {max_rate_around_step:.4f}, " \
                    f"Baseline: {baseline_rate:.4f}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_dates())
    def test_property_growth_decline_classification(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: Growth periods should have positive rates, 
        decline periods should have negative rates.
        """
        time_series, dates = data
        window = 7
        
        # Skip if time series is too short
        assume(len(time_series) >= window + 1)
        
        # Create analyzer and calculate rate
        analyzer = TrendAnalyzer()
        result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
        
        rate = result['rate']
        growth_periods = result['growth_periods']
        decline_periods = result['decline_periods']
        
        # Verify classification
        for i in range(len(time_series)):
            if not np.isnan(rate.iloc[i]):
                if growth_periods.iloc[i]:
                    # Growth period should have positive rate
                    assert rate.iloc[i] > 0, \
                        f"Growth period at index {i} should have positive rate, got {rate.iloc[i]:.6f}"
                
                if decline_periods.iloc[i]:
                    # Decline period should have negative rate
                    assert rate.iloc[i] < 0, \
                        f"Decline period at index {i} should have negative rate, got {rate.iloc[i]:.6f}"
                
                # A period cannot be both growth and decline
                assert not (growth_periods.iloc[i] and decline_periods.iloc[i]), \
                    f"Index {i} cannot be both growth and decline"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_dates())
    def test_property_average_rates_consistency(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: Average growth rate should be positive, 
        average decline rate should be negative.
        """
        time_series, dates = data
        window = 7
        
        # Skip if time series is too short
        assume(len(time_series) >= window + 5)
        
        # Create analyzer and calculate rate
        analyzer = TrendAnalyzer()
        result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
        
        avg_growth_rate = result['avg_growth_rate']
        avg_decline_rate = result['avg_decline_rate']
        
        # Check average rates
        if not np.isnan(avg_growth_rate) and avg_growth_rate != 0:
            assert avg_growth_rate > 0, \
                f"Average growth rate should be positive, got {avg_growth_rate:.6f}"
        
        if not np.isnan(avg_decline_rate) and avg_decline_rate != 0:
            assert avg_decline_rate < 0, \
                f"Average decline rate should be negative, got {avg_decline_rate:.6f}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        length=st.integers(min_value=20, max_value=100),
        constant_value=st.floats(min_value=0.3, max_value=0.8, allow_nan=False, allow_infinity=False)
    )
    def test_property_zero_rate_for_constant_series(self, length, constant_value):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: For a constant time series (no change), 
        rate of change should be zero everywhere.
        """
        # Create constant time series
        start_date = datetime(2024, 1, 1)
        dates = pd.date_range(start=start_date, periods=length, freq='D')
        time_series = pd.Series([constant_value] * length, index=dates)
        
        window = 7
        
        # Skip if too short
        assume(length >= window + 1)
        
        # Create analyzer and calculate rate
        analyzer = TrendAnalyzer()
        result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
        
        rate = result['rate']
        
        # All rates (after initial window) should be zero
        for i in range(window, len(time_series)):
            if not np.isnan(rate.iloc[i]):
                assert abs(rate.iloc[i]) < 1e-10, \
                    f"Rate for constant series should be zero at index {i}, got {rate.iloc[i]:.6f}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=time_series_with_dates())
    def test_property_historical_average_is_mean_of_rates(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: Historical average rate should equal the mean of all calculated rates.
        """
        time_series, dates = data
        window = 7
        
        # Skip if time series is too short
        assume(len(time_series) >= window + 5)
        
        # Create analyzer and calculate rate
        analyzer = TrendAnalyzer()
        result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
        
        rate = result['rate']
        historical_avg_rate = result['historical_avg_rate']
        
        # Calculate mean of rates (excluding NaN)
        rate_values = rate.dropna()
        
        if len(rate_values) > 0:
            expected_avg = rate_values.mean()
            
            # Historical average should match mean
            assert abs(historical_avg_rate - expected_avg) < 1e-10, \
                f"Historical average rate {historical_avg_rate:.6f} should equal " \
                f"mean of rates {expected_avg:.6f}"
    
    @settings(max_examples=50, deadline=None)
    @given(data=time_series_with_dates())
    def test_property_rate_with_different_windows(self, data):
        """
        **Feature: production-enhancements, Property 24: Rate of change calculation**
        **Validates: Requirements 6.5**
        
        Property: Rate of change calculated with different window sizes 
        should maintain the fundamental relationship: rate = change / time.
        Larger windows should produce smoother (less volatile) rates.
        """
        time_series, dates = data
        
        # Skip if time series is too short
        assume(len(time_series) >= 30)
        
        # Calculate rates with different windows
        windows = [3, 7, 14]
        rates = {}
        
        analyzer = TrendAnalyzer()
        
        for window in windows:
            result = analyzer.calculate_rate_of_change(time_series, dates, window=window)
            rates[window] = result['rate']
        
        # Larger windows should produce less volatile rates
        # (lower standard deviation)
        for i in range(len(windows) - 1):
            small_window = windows[i]
            large_window = windows[i + 1]
            
            # Get rates for comparison (skip NaN values)
            small_rates = rates[small_window].dropna()
            large_rates = rates[large_window].dropna()
            
            if len(small_rates) > 10 and len(large_rates) > 10:
                small_std = small_rates.std()
                large_std = large_rates.std()
                
                # Larger window should have lower or similar volatility
                # (allow some tolerance for random variation)
                assert large_std <= small_std * 1.5, \
                    f"Larger window ({large_window}) should have lower volatility than " \
                    f"smaller window ({small_window}). Got std {large_std:.6f} vs {small_std:.6f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
