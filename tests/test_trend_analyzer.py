"""
Unit tests for TrendAnalyzer class
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.trend_analyzer import TrendAnalyzer


class TestTrendAnalyzer:
    """Test suite for TrendAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create TrendAnalyzer instance"""
        return TrendAnalyzer()
    
    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        # Create upward trend with some noise
        values = np.linspace(0.5, 0.8, 30) + np.random.normal(0, 0.02, 30)
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def sample_dates(self):
        """Create sample dates"""
        return pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    def test_fit_regression_with_known_data(self, analyzer):
        """Test regression fitting with known data and verify explanation generation"""
        # Create known upward trend
        time_series = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9])
        
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Check that result contains expected keys
        assert 'slope' in result
        assert 'intercept' in result
        assert 'predictions' in result
        assert 'confidence_lower' in result
        assert 'confidence_upper' in result
        assert 'weekly_change_pct' in result
        assert 'monthly_change_pct' in result
        assert 'explanation' in result
        assert 'recommendation' in result
        assert 'r_squared' in result
        
        # Check that slope is positive (upward trend)
        assert result['slope'] > 0
        
        # Check that explanation is generated
        assert len(result['explanation']) > 0
        assert 'crops' in result['explanation'] or 'NDVI' in result['explanation']
        
        # Check that recommendation is generated
        assert len(result['recommendation']) > 0
        
        # Check that R² is high for this linear data
        assert result['r_squared'] > 0.9
    
    def test_fit_regression_with_insufficient_data(self, analyzer):
        """Test regression with insufficient data"""
        # Single data point
        time_series = pd.Series([0.7])
        
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Should return default values
        assert result['slope'] == 0
        assert result['weekly_change_pct'] == 0
        assert 'Insufficient data' in result['explanation']
    
    def test_detect_anomalies_with_synthetic_anomalies(self, analyzer):
        """Test anomaly detection with synthetic anomalies and verify descriptions"""
        # Create data with known anomalies
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        values = np.ones(20) * 0.7  # Stable at 0.7
        values[5] = 0.3  # Anomaly: drop
        values[15] = 0.95  # Anomaly: spike
        
        time_series = pd.Series(values)
        
        result = analyzer.detect_anomalies(time_series, dates, threshold_std=2.0)
        
        # Check that anomalies are detected
        assert result['count'] > 0
        assert len(result['descriptions']) > 0
        
        # Check that anomaly at index 5 is detected
        assert result['anomalies'][5] == True
        
        # Check that anomaly at index 15 is detected
        assert result['anomalies'][15] == True
        
        # Check that descriptions are generated
        for desc in result['descriptions']:
            assert 'date' in desc
            assert 'value' in desc
            assert 'z_score' in desc
            assert 'description' in desc
            assert 'direction' in desc
    
    def test_detect_anomalies_with_no_anomalies(self, analyzer):
        """Test anomaly detection with stable data"""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        values = np.ones(20) * 0.7  # Perfectly stable
        
        time_series = pd.Series(values)
        
        result = analyzer.detect_anomalies(time_series, dates, threshold_std=2.0)
        
        # Should detect no anomalies
        assert result['count'] == 0
        assert len(result['descriptions']) == 0
    
    def test_decompose_seasonal_with_periodic_data(self, analyzer):
        """Test seasonal decomposition with periodic data and verify component explanations"""
        # Create data with trend and seasonality (need enough data points)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        trend = np.linspace(0.5, 0.8, 100)
        seasonal = 0.1 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly pattern
        noise = np.random.normal(0, 0.01, 100)
        values = trend + seasonal + noise
        
        time_series = pd.Series(values, index=dates)
        
        result = analyzer.decompose_seasonal(time_series, period=7)
        
        if result is not None:  # Only test if statsmodels is available
            # Check that components are present
            assert 'trend' in result
            assert 'seasonal' in result
            assert 'residual' in result
            assert 'explanations' in result
            
            # Check that explanations are generated
            assert 'trend' in result['explanations']
            assert 'seasonal' in result['explanations']
            assert 'residual' in result['explanations']
            
            # Check that each explanation is non-empty
            assert len(result['explanations']['trend']) > 0
            assert len(result['explanations']['seasonal']) > 0
            assert len(result['explanations']['residual']) > 0
            
            # Check that seasonal amplitude is calculated
            assert 'seasonal_amplitude' in result
            assert result['seasonal_amplitude'] > 0
            
            # Check that trend direction is determined
            assert 'trend_direction' in result
            assert result['trend_direction'] in ['increasing', 'decreasing']
    
    def test_calculate_rate_of_change_accuracy(self, analyzer):
        """Test rate calculation accuracy and historical comparison logic"""
        # Create data with known rate of change
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        # Linear increase: 0.01 per day
        values = 0.5 + 0.01 * np.arange(30)
        
        time_series = pd.Series(values)
        
        result = analyzer.calculate_rate_of_change(time_series, dates, window=7)
        
        # Check that result contains expected keys
        assert 'rate' in result
        assert 'rate_pct_weekly' in result
        assert 'growth_periods' in result
        assert 'decline_periods' in result
        assert 'avg_growth_rate' in result
        assert 'avg_decline_rate' in result
        assert 'historical_avg_rate' in result
        assert 'significant_changes' in result
        
        # Check that rate is positive (growth)
        valid_rates = result['rate'][~result['rate'].isna()]
        assert np.all(valid_rates > 0)
        
        # Check that growth periods are detected
        assert result['growth_periods'].any()
        
        # Check that decline periods are not detected (all growth)
        assert not result['decline_periods'].any()
    
    def test_calculate_rate_of_change_with_decline(self, analyzer):
        """Test rate calculation with declining data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        # Linear decrease: -0.01 per day
        values = 0.8 - 0.01 * np.arange(30)
        
        time_series = pd.Series(values)
        
        result = analyzer.calculate_rate_of_change(time_series, dates, window=7)
        
        # Check that rate is negative (decline)
        valid_rates = result['rate'][~result['rate'].isna()]
        assert np.all(valid_rates < 0)
        
        # Check that decline periods are detected
        assert result['decline_periods'].any()
        
        # Check that growth periods are not detected
        assert not result['growth_periods'].any()
    
    def test_generate_day_wise_comparison_with_mock_data(self, analyzer):
        """Test day-wise comparison with mock data"""
        # Create mock time series data
        data = pd.DataFrame({
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 15)],
            'NDVI': [0.6, 0.8],
            'SAVI': [0.55, 0.75],
            'EVI': [0.5, 0.7],
            'NDWI': [0.3, 0.4]
        })
        
        date1 = datetime(2024, 1, 1)
        date2 = datetime(2024, 1, 15)
        
        result = analyzer.generate_day_wise_comparison(data, date1, date2)
        
        # Check that result is not None
        assert result is not None
        
        # Check that changes are calculated for each index
        assert 'NDVI' in result
        assert 'SAVI' in result
        assert 'EVI' in result
        assert 'NDWI' in result
        
        # Check NDVI change
        ndvi_change = result['NDVI']
        assert ndvi_change['value1'] == 0.6
        assert ndvi_change['value2'] == 0.8
        assert ndvi_change['delta'] == pytest.approx(0.2, abs=0.01)
        assert ndvi_change['pct_change'] > 0  # Positive change
        assert 'interpretation' in ndvi_change
        assert len(ndvi_change['interpretation']) > 0
    
    def test_generate_day_wise_comparison_with_missing_data(self, analyzer):
        """Test day-wise comparison with missing data"""
        data = pd.DataFrame({
            'date': [datetime(2024, 1, 1)],
            'NDVI': [0.6]
        })
        
        date1 = datetime(2024, 1, 1)
        date2 = datetime(2024, 1, 15)  # Date not in data
        
        result = analyzer.generate_day_wise_comparison(data, date1, date2)
        
        # Should return None for missing data
        assert result is None
    
    def test_interpret_change_positive(self, analyzer):
        """Test interpretation of positive change"""
        interpretation = analyzer._interpret_change('NDVI', 0.15, 20.0)
        
        assert 'improvement' in interpretation.lower()
        assert len(interpretation) > 0
    
    def test_interpret_change_negative(self, analyzer):
        """Test interpretation of negative change"""
        interpretation = analyzer._interpret_change('NDVI', -0.15, -20.0)
        
        assert 'decline' in interpretation.lower()
        assert '⚠️' in interpretation
    
    def test_interpret_change_minimal(self, analyzer):
        """Test interpretation of minimal change"""
        interpretation = analyzer._interpret_change('NDVI', 0.01, 1.0)
        
        assert 'minimal' in interpretation.lower() or 'stable' in interpretation.lower()
    
    def test_regression_with_empty_series(self, analyzer):
        """Test regression with empty series"""
        time_series = pd.Series([])
        
        result = analyzer.fit_regression(time_series, index_name='NDVI')
        
        # Should handle gracefully
        assert result['slope'] == 0
        assert result['r_squared'] == 0
    
    def test_anomaly_detection_with_insufficient_data(self, analyzer):
        """Test anomaly detection with insufficient data"""
        dates = pd.date_range(start='2024-01-01', periods=2, freq='D')
        time_series = pd.Series([0.7, 0.8])
        
        result = analyzer.detect_anomalies(time_series, dates)
        
        # Should return no anomalies
        assert result['count'] == 0
    
    def test_rate_of_change_with_insufficient_data(self, analyzer):
        """Test rate of change with insufficient data"""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        time_series = pd.Series([0.7, 0.71, 0.72, 0.73, 0.74])
        
        result = analyzer.calculate_rate_of_change(time_series, dates, window=7)
        
        # Should handle gracefully
        assert 'rate' in result
        assert len(result['significant_changes']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
