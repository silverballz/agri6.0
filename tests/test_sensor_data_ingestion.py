"""
Tests for sensor data ingestion system.
"""

import pytest
import tempfile
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.sensors.data_ingestion import SensorDataIngester, SensorReading
from src.sensors.data_validation import SensorDataValidator, ValidationResult
from src.sensors.temporal_alignment import TemporalAligner, AlignedReading
from src.sensors.spatial_interpolation import SpatialInterpolator, InterpolationGrid


class TestSensorDataIngester:
    """Test sensor data ingestion functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ingester = SensorDataIngester()
        
        # Sample sensor data
        self.sample_readings = [
            {
                'sensor_id': 'soil_001',
                'timestamp': '2024-09-23T10:30:00Z',
                'sensor_type': 'soil_moisture',
                'value': 45.2,
                'unit': '%',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'quality_flag': 'good'
            },
            {
                'sensor_id': 'temp_001',
                'timestamp': '2024-09-23T10:30:00Z',
                'sensor_type': 'temperature',
                'value': 22.5,
                'unit': '°C',
                'latitude': 40.7130,
                'longitude': -74.0062,
                'quality_flag': 'good'
            }
        ]
    
    def test_csv_ingestion(self):
        """Test CSV file ingestion."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=self.sample_readings[0].keys())
            writer.writeheader()
            writer.writerows(self.sample_readings)
            csv_path = f.name
        
        try:
            readings = self.ingester.ingest_csv(csv_path)
            
            assert len(readings) == 2
            assert readings[0].sensor_id == 'soil_001'
            assert readings[0].sensor_type == 'soil_moisture'
            assert readings[0].value == 45.2
            assert readings[0].latitude == 40.7128
            assert readings[0].longitude == -74.0060
            
        finally:
            Path(csv_path).unlink()
    
    def test_json_ingestion(self):
        """Test JSON file ingestion."""
        json_data = {'readings': self.sample_readings}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name
        
        try:
            readings = self.ingester.ingest_json(json_path)
            
            assert len(readings) == 2
            assert readings[1].sensor_id == 'temp_001'
            assert readings[1].sensor_type == 'temperature'
            assert readings[1].value == 22.5
            
        finally:
            Path(json_path).unlink()
    
    def test_dataframe_ingestion(self):
        """Test DataFrame ingestion."""
        df = pd.DataFrame(self.sample_readings)
        readings = self.ingester.ingest_dataframe(df)
        
        assert len(readings) == 2
        assert all(isinstance(r, SensorReading) for r in readings)
    
    def test_timestamp_parsing(self):
        """Test various timestamp formats."""
        test_timestamps = [
            '2024-09-23T10:30:00Z',
            '2024-09-23T10:30:00',
            '2024-09-23 10:30:00',
            '2024-09-23 10:30',
            '2024-09-23',
            '23/09/2024 10:30:00'
        ]
        
        for ts in test_timestamps:
            parsed = self.ingester._parse_timestamp(ts)
            assert isinstance(parsed, datetime)
    
    def test_invalid_file_handling(self):
        """Test handling of invalid files."""
        with pytest.raises(FileNotFoundError):
            self.ingester.ingest_csv('nonexistent.csv')
        
        with pytest.raises(FileNotFoundError):
            self.ingester.ingest_json('nonexistent.json')
    
    def test_sensor_type_validation(self):
        """Test sensor type validation."""
        assert self.ingester.validate_sensor_type('soil_moisture')
        assert self.ingester.validate_sensor_type('temperature')
        assert not self.ingester.validate_sensor_type('unknown_sensor')


class TestSensorDataValidator:
    """Test sensor data validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SensorDataValidator()
        
        # Create test readings
        self.valid_reading = SensorReading(
            sensor_id='test_001',
            timestamp=datetime.now(),
            sensor_type='soil_moisture',
            value=45.0,
            unit='%',
            latitude=40.7128,
            longitude=-74.0060
        )
        
        self.invalid_reading = SensorReading(
            sensor_id='test_002',
            timestamp=datetime.now(),
            sensor_type='soil_moisture',
            value=150.0,  # Invalid: > 100%
            unit='%',
            latitude=40.7128,
            longitude=-74.0060
        )
    
    def test_range_validation(self):
        """Test range validation."""
        # Valid reading
        result = self.validator.validate_reading(self.valid_reading)
        assert result.is_valid
        assert result.quality_score > 0.8
        assert result.recommended_flag == 'good'
        
        # Invalid reading
        result = self.validator.validate_reading(self.invalid_reading)
        assert not result.is_valid
        assert result.quality_score < 0.8
        assert len(result.issues) > 0
    
    def test_temporal_validation(self):
        """Test temporal consistency validation."""
        base_time = datetime.now()
        
        # Create historical readings
        historical = [
            SensorReading(
                sensor_id='test_001',
                timestamp=base_time - timedelta(hours=2),
                sensor_type='temperature',
                value=20.0,
                unit='°C'
            ),
            SensorReading(
                sensor_id='test_001',
                timestamp=base_time - timedelta(hours=1),
                sensor_type='temperature',
                value=21.0,
                unit='°C'
            )
        ]
        
        # Valid temporal reading
        current_reading = SensorReading(
            sensor_id='test_001',
            timestamp=base_time,
            sensor_type='temperature',
            value=22.0,
            unit='°C'
        )
        
        result = self.validator.validate_reading(current_reading, historical)
        assert result.is_valid
        
        # Invalid temporal reading (too rapid change)
        rapid_change_reading = SensorReading(
            sensor_id='test_001',
            timestamp=base_time,
            sensor_type='temperature',
            value=50.0,  # Too rapid change
            unit='°C'
        )
        
        result = self.validator.validate_reading(rapid_change_reading, historical)
        assert not result.is_valid
        assert result.corrected_value is not None
    
    def test_batch_validation(self):
        """Test batch validation."""
        readings = [self.valid_reading, self.invalid_reading]
        results = self.validator.validate_batch(readings)
        
        assert len(results) == 2
        assert results[0].is_valid
        assert not results[1].is_valid
    
    def test_quality_statistics(self):
        """Test quality statistics calculation."""
        results = [
            ValidationResult(True, 0.9, [], recommended_flag='good'),
            ValidationResult(False, 0.3, ['Range error'], recommended_flag='bad')
        ]
        
        stats = self.validator.get_quality_statistics(results)
        
        assert stats['total_readings'] == 2
        assert stats['valid_readings'] == 1
        assert stats['validity_rate'] == 0.5
        assert 'flag_distribution' in stats


class TestTemporalAligner:
    """Test temporal alignment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = TemporalAligner()
        
        # Create test readings around satellite overpass time
        overpass_time = datetime(2024, 9, 23, 10, 30, 0)
        
        self.test_readings = [
            SensorReading(
                sensor_id='sensor_001',
                timestamp=overpass_time - timedelta(minutes=30),
                sensor_type='soil_moisture',
                value=40.0,
                unit='%',
                latitude=40.7128,
                longitude=-74.0060
            ),
            SensorReading(
                sensor_id='sensor_001',
                timestamp=overpass_time + timedelta(minutes=30),
                sensor_type='soil_moisture',
                value=42.0,
                unit='%',
                latitude=40.7128,
                longitude=-74.0060
            )
        ]
        
        self.overpass_times = [overpass_time]
    
    def test_alignment_to_overpass(self):
        """Test alignment of readings to overpass times."""
        aligned = self.aligner.align_to_overpass(
            self.test_readings, self.overpass_times, method='linear'
        )
        
        assert len(aligned) == 1
        assert aligned[0].interpolated_value is not None
        assert 40.0 <= aligned[0].interpolated_value <= 42.0
        assert aligned[0].confidence >= 0.5
    
    def test_closest_readings(self):
        """Test finding closest readings to target time."""
        target_time = datetime(2024, 9, 23, 10, 30, 0)
        closest = self.aligner.find_closest_readings(self.test_readings, target_time)
        
        assert len(closest) == 2
        # Should be sorted by proximity
        assert closest[0].timestamp <= target_time <= closest[1].timestamp or \
               closest[1].timestamp <= target_time <= closest[0].timestamp
    
    def test_interpolation_to_time(self):
        """Test interpolation to specific time."""
        target_time = datetime(2024, 9, 23, 10, 30, 0)
        result = self.aligner.interpolate_to_time(
            self.test_readings, target_time, method='linear'
        )
        
        assert result is not None
        interpolated_value, confidence = result
        assert 40.0 <= interpolated_value <= 42.0
        assert 0.0 <= confidence <= 1.0
    
    def test_overpass_time_generation(self):
        """Test generation of overpass times."""
        start_date = datetime(2024, 9, 1)
        end_date = datetime(2024, 9, 30)
        
        overpass_times = self.aligner.generate_overpass_times(
            start_date, end_date, 40.7128, -74.0060
        )
        
        assert len(overpass_times) >= 2  # At least 2 overpasses in September
        assert all(isinstance(t, datetime) for t in overpass_times)


class TestSpatialInterpolator:
    """Test spatial interpolation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interpolator = SpatialInterpolator()
        
        # Create test readings with spatial distribution
        self.test_readings = [
            SensorReading(
                sensor_id='sensor_001',
                timestamp=datetime.now(),
                sensor_type='soil_moisture',
                value=40.0,
                unit='%',
                latitude=40.710,
                longitude=-74.000
            ),
            SensorReading(
                sensor_id='sensor_002',
                timestamp=datetime.now(),
                sensor_type='soil_moisture',
                value=45.0,
                unit='%',
                latitude=40.715,
                longitude=-74.005
            ),
            SensorReading(
                sensor_id='sensor_003',
                timestamp=datetime.now(),
                sensor_type='soil_moisture',
                value=42.0,
                unit='%',
                latitude=40.720,
                longitude=-74.010
            )
        ]
    
    def test_grid_interpolation(self):
        """Test interpolation to regular grid."""
        grid_bounds = (-74.015, 40.705, -73.995, 40.725)
        
        grid = self.interpolator.interpolate_sensors(
            self.test_readings, grid_bounds, grid_resolution=0.005, method='linear'
        )
        
        assert isinstance(grid, InterpolationGrid)
        assert grid.values.shape[0] > 1
        assert grid.values.shape[1] > 1
        assert grid.method == 'linear'
        assert grid.sensor_type == 'soil_moisture'
        assert grid.quality_mask is not None
    
    def test_point_interpolation(self):
        """Test interpolation to specific points."""
        target_points = [(40.712, -74.002), (40.717, -74.007)]
        
        values = self.interpolator.interpolate_to_points(
            self.test_readings, target_points, method='linear'
        )
        
        assert len(values) == 2
        assert all(isinstance(v, float) for v in values)
        assert all(35.0 <= v <= 50.0 for v in values)  # Reasonable range
    
    def test_interpolation_validation(self):
        """Test interpolation validation."""
        validation_result = self.interpolator.validate_interpolation(
            self.test_readings, method='linear'
        )
        
        assert 'mae' in validation_result or 'error' in validation_result
        if 'mae' in validation_result:
            assert validation_result['mae'] >= 0
            assert validation_result['rmse'] >= 0
            assert -1 <= validation_result['r_squared'] <= 1
    
    def test_optimal_method_selection(self):
        """Test optimal method selection."""
        optimal_method = self.interpolator.get_optimal_method(self.test_readings)
        
        assert optimal_method in self.interpolator.supported_methods
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Test with no readings
        with pytest.raises(ValueError):
            self.interpolator.interpolate_sensors([], (-74.015, 40.705, -73.995, 40.725))
        
        # Test with readings without coordinates
        readings_no_coords = [
            SensorReading(
                sensor_id='sensor_001',
                timestamp=datetime.now(),
                sensor_type='soil_moisture',
                value=40.0,
                unit='%'
            )
        ]
        
        with pytest.raises(ValueError):
            self.interpolator.interpolate_sensors(
                readings_no_coords, (-74.015, 40.705, -73.995, 40.725)
            )


class TestSensorDataValidationKnownValues:
    """Test sensor data validation with known input/output values."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SensorDataValidator()
    
    def test_soil_moisture_validation_known_values(self):
        """Test soil moisture validation with known valid/invalid values."""
        # Valid soil moisture readings
        valid_readings = [
            SensorReading(
                sensor_id='test_001',
                timestamp=datetime.now(),
                sensor_type='soil_moisture',
                value=25.0,  # Typical field capacity
                unit='%',
                latitude=40.7128,
                longitude=-74.0060
            ),
            SensorReading(
                sensor_id='test_002',
                timestamp=datetime.now(),
                sensor_type='soil_moisture',
                value=45.0,  # Good moisture level
                unit='%',
                latitude=40.7128,
                longitude=-74.0060
            )
        ]
        
        for reading in valid_readings:
            result = self.validator.validate_reading(reading)
            assert result.is_valid, f"Valid reading {reading.value}% should pass validation"
            assert result.quality_score > 0.8
            assert result.recommended_flag == 'good'
        
        # Invalid soil moisture readings
        invalid_readings = [
            SensorReading(
                sensor_id='test_003',
                timestamp=datetime.now(),
                sensor_type='soil_moisture',
                value=-5.0,  # Negative moisture (impossible)
                unit='%',
                latitude=40.7128,
                longitude=-74.0060
            ),
            SensorReading(
                sensor_id='test_004',
                timestamp=datetime.now(),
                sensor_type='soil_moisture',
                value=120.0,  # Over 100% (impossible)
                unit='%',
                latitude=40.7128,
                longitude=-74.0060
            )
        ]
        
        for reading in invalid_readings:
            result = self.validator.validate_reading(reading)
            assert not result.is_valid, f"Invalid reading {reading.value}% should fail validation"
            assert result.quality_score < 0.5
            assert len(result.issues) > 0
            assert 'range' in result.issues[0].lower()
    
    def test_temperature_validation_known_values(self):
        """Test temperature validation with known valid/invalid values."""
        # Valid temperature readings
        valid_temps = [0.0, 15.0, 25.0, 35.0, 45.0]  # Celsius
        
        for temp in valid_temps:
            reading = SensorReading(
                sensor_id='temp_test',
                timestamp=datetime.now(),
                sensor_type='temperature',
                value=temp,
                unit='°C',
                latitude=40.7128,
                longitude=-74.0060
            )
            
            result = self.validator.validate_reading(reading)
            assert result.is_valid, f"Valid temperature {temp}°C should pass validation"
            assert result.quality_score > 0.7
        
        # Invalid temperature readings
        invalid_temps = [-50.0, 70.0, 100.0]  # Extreme values for air temperature
        
        for temp in invalid_temps:
            reading = SensorReading(
                sensor_id='temp_test',
                timestamp=datetime.now(),
                sensor_type='temperature',
                value=temp,
                unit='°C',
                latitude=40.7128,
                longitude=-74.0060
            )
            
            result = self.validator.validate_reading(reading)
            assert not result.is_valid, f"Invalid temperature {temp}°C should fail validation"
            assert result.quality_score < 0.6
    
    def test_humidity_validation_known_values(self):
        """Test humidity validation with known valid/invalid values."""
        # Valid humidity readings
        valid_humidity = [30.0, 50.0, 75.0, 95.0]  # Percentage
        
        for humidity in valid_humidity:
            reading = SensorReading(
                sensor_id='humidity_test',
                timestamp=datetime.now(),
                sensor_type='humidity',
                value=humidity,
                unit='%',
                latitude=40.7128,
                longitude=-74.0060
            )
            
            result = self.validator.validate_reading(reading)
            assert result.is_valid, f"Valid humidity {humidity}% should pass validation"
            assert result.quality_score > 0.8
        
        # Invalid humidity readings
        invalid_humidity = [-10.0, 110.0, 150.0]  # Outside 0-100% range
        
        for humidity in invalid_humidity:
            reading = SensorReading(
                sensor_id='humidity_test',
                timestamp=datetime.now(),
                sensor_type='humidity',
                value=humidity,
                unit='%',
                latitude=40.7128,
                longitude=-74.0060
            )
            
            result = self.validator.validate_reading(reading)
            assert not result.is_valid, f"Invalid humidity {humidity}% should fail validation"
            assert result.quality_score < 0.5
    
    def test_temporal_consistency_known_patterns(self):
        """Test temporal consistency validation with known patterns."""
        base_time = datetime(2024, 9, 23, 10, 0, 0)
        
        # Create realistic temperature progression (gradual warming)
        historical_temps = [
            SensorReading(
                sensor_id='temp_001',
                timestamp=base_time - timedelta(hours=3),
                sensor_type='temperature',
                value=18.0,
                unit='°C'
            ),
            SensorReading(
                sensor_id='temp_001',
                timestamp=base_time - timedelta(hours=2),
                sensor_type='temperature',
                value=20.0,
                unit='°C'
            ),
            SensorReading(
                sensor_id='temp_001',
                timestamp=base_time - timedelta(hours=1),
                sensor_type='temperature',
                value=22.0,
                unit='°C'
            )
        ]
        
        # Test realistic next reading (gradual increase)
        realistic_reading = SensorReading(
            sensor_id='temp_001',
            timestamp=base_time,
            sensor_type='temperature',
            value=24.0,  # Gradual 2°C increase
            unit='°C'
        )
        
        result = self.validator.validate_reading(realistic_reading, historical_temps)
        assert result.is_valid, "Realistic temperature progression should be valid"
        assert result.quality_score > 0.8
        
        # Test unrealistic jump (sudden spike)
        spike_reading = SensorReading(
            sensor_id='temp_001',
            timestamp=base_time,
            sensor_type='temperature',
            value=45.0,  # Sudden 23°C jump
            unit='°C'
        )
        
        result = self.validator.validate_reading(spike_reading, historical_temps)
        assert not result.is_valid, "Unrealistic temperature spike should be invalid"
        assert result.corrected_value is not None
        assert 20.0 <= result.corrected_value <= 30.0  # Should suggest reasonable value
    
    def test_quality_flag_assignment(self):
        """Test quality flag assignment based on validation results."""
        test_cases = [
            # (value, sensor_type, expected_flag, expected_quality_range)
            (25.0, 'soil_moisture', 'good', (0.9, 1.0)),
            (5.0, 'soil_moisture', 'questionable', (0.5, 0.8)),
            (-5.0, 'soil_moisture', 'bad', (0.0, 0.3)),
            (22.0, 'temperature', 'good', (0.9, 1.0)),
            (55.0, 'temperature', 'bad', (0.0, 0.3)),
            (75.0, 'humidity', 'good', (0.9, 1.0)),
            (110.0, 'humidity', 'bad', (0.0, 0.3))
        ]
        
        for value, sensor_type, expected_flag, quality_range in test_cases:
            reading = SensorReading(
                sensor_id='test',
                timestamp=datetime.now(),
                sensor_type=sensor_type,
                value=value,
                unit='%' if sensor_type != 'temperature' else '°C',
                latitude=40.7128,
                longitude=-74.0060
            )
            
            result = self.validator.validate_reading(reading)
            
            assert result.recommended_flag == expected_flag, \
                f"Value {value} for {sensor_type} should get flag '{expected_flag}', got '{result.recommended_flag}'"
            
            assert quality_range[0] <= result.quality_score <= quality_range[1], \
                f"Quality score {result.quality_score} not in expected range {quality_range}"
    
    def test_coordinate_validation(self):
        """Test validation of sensor coordinates."""
        # Valid coordinates
        valid_coords = [
            (40.7128, -74.0060),  # New York
            (51.5074, -0.1278),   # London
            (35.6762, 139.6503),  # Tokyo
            (-33.8688, 151.2093)  # Sydney
        ]
        
        for lat, lon in valid_coords:
            reading = SensorReading(
                sensor_id='coord_test',
                timestamp=datetime.now(),
                sensor_type='temperature',
                value=20.0,
                unit='°C',
                latitude=lat,
                longitude=lon
            )
            
            result = self.validator.validate_reading(reading)
            assert result.is_valid, f"Valid coordinates ({lat}, {lon}) should pass validation"
        
        # Invalid coordinates
        invalid_coords = [
            (91.0, 0.0),    # Latitude > 90
            (-91.0, 0.0),   # Latitude < -90
            (0.0, 181.0),   # Longitude > 180
            (0.0, -181.0),  # Longitude < -180
            (float('nan'), 0.0),  # NaN latitude
            (0.0, float('inf'))   # Infinite longitude
        ]
        
        for lat, lon in invalid_coords:
            reading = SensorReading(
                sensor_id='coord_test',
                timestamp=datetime.now(),
                sensor_type='temperature',
                value=20.0,
                unit='°C',
                latitude=lat,
                longitude=lon
            )
            
            result = self.validator.validate_reading(reading)
            assert not result.is_valid, f"Invalid coordinates ({lat}, {lon}) should fail validation"
            assert any('coordinate' in issue.lower() or 'location' in issue.lower() 
                      for issue in result.issues)