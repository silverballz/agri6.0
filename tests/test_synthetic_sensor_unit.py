"""
Unit tests for synthetic sensor data generation.
Tests specific functionality and edge cases.

Feature: production-enhancements
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sensors.synthetic_sensor_generator import SyntheticSensorGenerator, SyntheticSensorReading


class TestSyntheticSensorGenerator:
    """Unit tests for SyntheticSensorGenerator class"""
    
    def test_initialization(self):
        """Test generator initialization"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        assert generator is not None
        assert generator.temporal_autocorrelation == 0.7
        assert 'soil_moisture' in generator.correlation_params
        assert 'temperature' in generator.correlation_params
        assert 'humidity' in generator.correlation_params
        assert 'leaf_wetness' in generator.correlation_params
    
    def test_soil_moisture_noise_characteristics(self):
        """Test noise characteristics (mean, std, distribution) for soil moisture"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        # Generate data
        size = 100
        ndvi_values = np.full(size, 0.6)  # Constant NDVI
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_soil_moisture(ndvi_values, timestamps, locations)
        values = np.array([r.value for r in readings])
        
        # Check mean is reasonable (should be around base value for NDVI=0.6)
        mean_value = np.mean(values)
        assert 15 < mean_value < 35, f"Mean soil moisture {mean_value} outside expected range"
        
        # Check standard deviation is reasonable
        std_value = np.std(values)
        assert std_value > 0, "Standard deviation should be positive"
        
        # Check coefficient of variation is within expected range
        cv = std_value / mean_value
        assert 0.0 < cv < 0.5, f"Coefficient of variation {cv} outside reasonable range"
    
    def test_temperature_noise_characteristics(self):
        """Test noise characteristics for temperature"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 100
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_temperature(timestamps, locations)
        values = np.array([r.value for r in readings])
        
        # Check mean is reasonable
        mean_value = np.mean(values)
        assert 15 < mean_value < 35, f"Mean temperature {mean_value} outside expected range"
        
        # Check all values are within valid range
        assert np.all(values >= 5), "Temperature below minimum"
        assert np.all(values <= 45), "Temperature above maximum"
    
    def test_humidity_noise_characteristics(self):
        """Test noise characteristics for humidity"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 100
        temperature_values = np.full(size, 25.0)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_humidity(temperature_values, None, timestamps, locations)
        values = np.array([r.value for r in readings])
        
        # Check all values are within valid range
        assert np.all(values >= 20), "Humidity below minimum"
        assert np.all(values <= 95), "Humidity above maximum"
        
        # Check standard deviation is reasonable
        std_value = np.std(values)
        assert std_value > 0, "Standard deviation should be positive"
    
    def test_leaf_wetness_noise_characteristics(self):
        """Test noise characteristics for leaf wetness"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 100
        humidity_values = np.full(size, 75.0)
        temperature_values = np.full(size, 22.5)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_leaf_wetness(
            humidity_values, temperature_values, timestamps, locations
        )
        values = np.array([r.value for r in readings])
        
        # Check all values are within valid range [0, 1]
        assert np.all(values >= 0), "Leaf wetness below 0"
        assert np.all(values <= 1), "Leaf wetness above 1"
    
    def test_soil_moisture_correlation_strength(self):
        """Test correlation strength between soil moisture and NDVI"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 100
        # Create NDVI values with good spread
        ndvi_values = np.linspace(-0.5, 0.9, size)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_soil_moisture(ndvi_values, timestamps, locations)
        moisture_values = np.array([r.value for r in readings])
        
        # Calculate correlation
        correlation = np.corrcoef(ndvi_values, moisture_values)[0, 1]
        
        # Should have strong positive correlation
        assert correlation > 0.5, f"Correlation {correlation} below target 0.5"
    
    def test_humidity_temperature_correlation_strength(self):
        """Test correlation strength between humidity and temperature"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 100
        # Create temperature values with good spread
        temperature_values = np.linspace(15.0, 40.0, size)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_humidity(temperature_values, None, timestamps, locations)
        humidity_values = np.array([r.value for r in readings])
        
        # Calculate correlation
        correlation = np.corrcoef(temperature_values, humidity_values)[0, 1]
        
        # Should have negative correlation
        assert correlation < -0.3, f"Correlation {correlation} not sufficiently negative"
    
    def test_temporal_autocorrelation(self):
        """Test temporal autocorrelation in generated data"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 100
        ndvi_values = np.random.uniform(0.3, 0.8, size)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_soil_moisture(ndvi_values, timestamps, locations)
        values = np.array([r.value for r in readings])
        
        # Calculate lag-1 autocorrelation
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
        
        # Should have some positive autocorrelation due to temporal variation
        # But not too strong due to noise
        assert -0.5 < autocorr < 1.0, f"Autocorrelation {autocorr} outside reasonable range"
    
    def test_data_range_validity_soil_moisture(self):
        """Test that soil moisture values are within valid range"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 50
        # Test with extreme NDVI values
        ndvi_values = np.array([-1.0] * 25 + [1.0] * 25)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_soil_moisture(ndvi_values, timestamps, locations)
        values = np.array([r.value for r in readings])
        
        # All values should be within [0, 50]
        assert np.all(values >= 0), f"Soil moisture below 0: min={np.min(values)}"
        assert np.all(values <= 50), f"Soil moisture above 50: max={np.max(values)}"
    
    def test_data_range_validity_temperature(self):
        """Test that temperature values are within valid range"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 365  # Full year
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_temperature(timestamps, locations)
        values = np.array([r.value for r in readings])
        
        # All values should be within [5, 45]
        assert np.all(values >= 5), f"Temperature below 5: min={np.min(values)}"
        assert np.all(values <= 45), f"Temperature above 45: max={np.max(values)}"
    
    def test_data_range_validity_humidity(self):
        """Test that humidity values are within valid range"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 50
        # Test with extreme temperature values
        temperature_values = np.array([5.0] * 25 + [45.0] * 25)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_humidity(temperature_values, None, timestamps, locations)
        values = np.array([r.value for r in readings])
        
        # All values should be within [20, 95]
        assert np.all(values >= 20), f"Humidity below 20: min={np.min(values)}"
        assert np.all(values <= 95), f"Humidity above 95: max={np.max(values)}"
    
    def test_data_range_validity_leaf_wetness(self):
        """Test that leaf wetness values are within valid range"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 50
        # Test with extreme conditions
        humidity_values = np.array([20.0] * 25 + [95.0] * 25)
        temperature_values = np.array([5.0] * 25 + [45.0] * 25)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_leaf_wetness(
            humidity_values, temperature_values, timestamps, locations
        )
        values = np.array([r.value for r in readings])
        
        # All values should be within [0, 1]
        assert np.all(values >= 0), f"Leaf wetness below 0: min={np.min(values)}"
        assert np.all(values <= 1), f"Leaf wetness above 1: max={np.max(values)}"
    
    def test_complete_sensor_suite_generation(self):
        """Test generation of complete sensor suite"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 30
        ndvi_values = np.random.uniform(0.3, 0.8, size)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        sensor_suite = generator.generate_complete_sensor_suite(
            ndvi_values, timestamps, locations
        )
        
        # Check all sensor types are present
        assert 'soil_moisture' in sensor_suite
        assert 'temperature' in sensor_suite
        assert 'humidity' in sensor_suite
        assert 'leaf_wetness' in sensor_suite
        
        # Check all have correct number of readings
        assert len(sensor_suite['soil_moisture']) == size
        assert len(sensor_suite['temperature']) == size
        assert len(sensor_suite['humidity']) == size
        assert len(sensor_suite['leaf_wetness']) == size
    
    def test_validate_correlations(self):
        """Test correlation validation method"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 100
        ndvi_values = np.linspace(0.3, 0.8, size)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        sensor_suite = generator.generate_complete_sensor_suite(
            ndvi_values, timestamps, locations
        )
        
        correlations = generator.validate_correlations(sensor_suite, ndvi_values)
        
        # Check correlation keys are present
        assert 'soil_moisture_ndvi' in correlations
        assert 'temperature_humidity' in correlations
        
        # Check correlation values are reasonable
        assert correlations['soil_moisture_ndvi'] > 0.5
        assert correlations['temperature_humidity'] < -0.3
    
    def test_export_to_dataframe(self):
        """Test export to pandas DataFrame"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 20
        ndvi_values = np.random.uniform(0.3, 0.8, size)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        sensor_suite = generator.generate_complete_sensor_suite(
            ndvi_values, timestamps, locations
        )
        
        df = generator.export_to_dataframe(sensor_suite)
        
        # Check DataFrame structure
        assert len(df) == size * 4  # 4 sensor types
        assert 'timestamp' in df.columns
        assert 'sensor_type' in df.columns
        assert 'value' in df.columns
        assert 'is_synthetic' in df.columns
        
        # Check all values are marked as synthetic
        assert df['is_synthetic'].all()
    
    def test_custom_noise_level(self):
        """Test custom noise level parameter"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 50
        ndvi_values = np.full(size, 0.6)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        # Generate with low noise
        readings_low = generator.generate_soil_moisture(
            ndvi_values, timestamps, locations, noise_level=0.05
        )
        values_low = np.array([r.value for r in readings_low])
        
        # Generate with high noise
        readings_high = generator.generate_soil_moisture(
            ndvi_values, timestamps, locations, noise_level=0.20
        )
        values_high = np.array([r.value for r in readings_high])
        
        # High noise should have higher standard deviation
        std_low = np.std(values_low)
        std_high = np.std(values_high)
        
        assert std_high > std_low, "High noise should have higher standard deviation"
    
    def test_pest_risk_assessment(self):
        """Test pest risk assessment in leaf wetness"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 10
        # Create conditions favorable for pests
        humidity_values = np.full(size, 80.0)
        temperature_values = np.full(size, 25.0)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_leaf_wetness(
            humidity_values, temperature_values, timestamps, locations
        )
        
        # Check pest risk is assessed
        assert all('pest_risk' in r.metadata for r in readings)
        
        # Most should be high risk under these conditions
        high_risk_count = sum(1 for r in readings if r.metadata['pest_risk'] == 'high')
        assert high_risk_count > 0, "Should have some high pest risk readings"
    
    def test_metadata_storage(self):
        """Test that metadata is properly stored in readings"""
        generator = SyntheticSensorGenerator(random_seed=42)
        
        size = 10
        ndvi_values = np.random.uniform(0.3, 0.8, size)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        locations = [(30.95, 75.85)] * size
        
        readings = generator.generate_soil_moisture(ndvi_values, timestamps, locations)
        
        # Check metadata is present
        for i, reading in enumerate(readings):
            assert reading.metadata is not None
            assert 'ndvi_value' in reading.metadata
            assert 'correlation_coefficient' in reading.metadata
            assert reading.metadata['ndvi_value'] == pytest.approx(ndvi_values[i], rel=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
