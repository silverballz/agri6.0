"""
Unit tests for data models.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Polygon
import tempfile
import os

from src.models import (
    SatelliteImage, MonitoringZone, SensorLocation, Alert,
    IndexTimeSeries, TimeSeriesCollection
)


class TestSatelliteImage:
    """Test cases for SatelliteImage model."""
    
    def test_satellite_image_creation(self):
        """Test basic SatelliteImage creation."""
        # Create test data
        bands = {
            'B02': np.random.rand(100, 100),
            'B03': np.random.rand(100, 100),
            'B04': np.random.rand(100, 100),
            'B08': np.random.rand(100, 100),
            'B11': np.random.rand(100, 100),
            'B12': np.random.rand(100, 100)
        }
        
        indices = {
            'NDVI': np.random.rand(100, 100)
        }
        
        geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        
        image = SatelliteImage(
            id="test_image_001",
            acquisition_date=datetime.now(),
            tile_id="T43REQ",
            cloud_coverage=15.5,
            bands=bands,
            indices=indices,
            geometry=geometry,
            quality_flags={'atmospheric_correction': True, 'cloud_mask_applied': True}
        )
        
        assert image.id == "test_image_001"
        assert image.tile_id == "T43REQ"
        assert image.cloud_coverage == 15.5
        assert len(image.bands) == 6
        assert 'NDVI' in image.indices
        assert image.is_quality_flagged('atmospheric_correction')
    
    def test_satellite_image_validation(self):
        """Test SatelliteImage validation."""
        bands = {
            'B02': np.random.rand(10, 10),
            'B03': np.random.rand(10, 10),
            'B04': np.random.rand(10, 10),
            'B08': np.random.rand(10, 10),
            'B11': np.random.rand(10, 10),
            'B12': np.random.rand(10, 10)
        }
        
        geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        
        # Test invalid cloud coverage
        with pytest.raises(ValueError, match="Cloud coverage must be between 0 and 100"):
            SatelliteImage(
                id="test",
                acquisition_date=datetime.now(),
                tile_id="T43REQ",
                cloud_coverage=150,  # Invalid
                bands=bands,
                indices={},
                geometry=geometry,
                quality_flags={}
            )
        
        # Test missing required band
        incomplete_bands = bands.copy()
        del incomplete_bands['B04']
        
        with pytest.raises(ValueError, match="Required band B04 is missing"):
            SatelliteImage(
                id="test",
                acquisition_date=datetime.now(),
                tile_id="T43REQ",
                cloud_coverage=15,
                bands=incomplete_bands,
                indices={},
                geometry=geometry,
                quality_flags={}
            )
    
    def test_satellite_image_serialization(self):
        """Test SatelliteImage serialization and deserialization."""
        bands = {
            'B02': np.random.rand(10, 10),
            'B03': np.random.rand(10, 10),
            'B04': np.random.rand(10, 10),
            'B08': np.random.rand(10, 10),
            'B11': np.random.rand(10, 10),
            'B12': np.random.rand(10, 10)
        }
        
        indices = {'NDVI': np.random.rand(10, 10)}
        geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        
        original_image = SatelliteImage(
            id="test_serialization",
            acquisition_date=datetime.now(),
            tile_id="T43REQ",
            cloud_coverage=25.0,
            bands=bands,
            indices=indices,
            geometry=geometry,
            quality_flags={'test_flag': True}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = os.path.join(temp_dir, "metadata.json")
            arrays_prefix = os.path.join(temp_dir, "arrays")
            
            # Save
            original_image.save_metadata(metadata_path)
            original_image.save_arrays(arrays_prefix)
            
            # Load
            loaded_image = SatelliteImage.from_metadata_file(metadata_path, arrays_prefix)
            
            assert loaded_image.id == original_image.id
            assert loaded_image.tile_id == original_image.tile_id
            assert loaded_image.cloud_coverage == original_image.cloud_coverage
            assert np.array_equal(loaded_image.bands['B04'], original_image.bands['B04'])
            assert np.array_equal(loaded_image.indices['NDVI'], original_image.indices['NDVI'])


class TestMonitoringZone:
    """Test cases for MonitoringZone model."""
    
    def test_monitoring_zone_creation(self):
        """Test basic MonitoringZone creation."""
        geometry = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        
        zone = MonitoringZone(
            id="zone_001",
            name="North Field",
            geometry=geometry,
            crop_type="wheat",
            planting_date=datetime(2024, 3, 15),
            expected_harvest=datetime(2024, 8, 15)
        )
        
        assert zone.id == "zone_001"
        assert zone.name == "North Field"
        assert zone.crop_type == "wheat"
        assert zone.get_area() == 100.0  # 10x10 square
    
    def test_monitoring_zone_validation(self):
        """Test MonitoringZone validation."""
        geometry = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        
        # Test invalid date range
        with pytest.raises(ValueError, match="Expected harvest date must be after planting date"):
            MonitoringZone(
                id="zone_001",
                name="Test Field",
                geometry=geometry,
                crop_type="wheat",
                planting_date=datetime(2024, 8, 15),
                expected_harvest=datetime(2024, 3, 15)  # Before planting
            )
    
    def test_sensor_management(self):
        """Test sensor addition and removal."""
        geometry = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        
        zone = MonitoringZone(
            id="zone_001",
            name="Test Field",
            geometry=geometry,
            crop_type="wheat",
            planting_date=datetime(2024, 3, 15),
            expected_harvest=datetime(2024, 8, 15)
        )
        
        # Add sensor within boundary
        sensor = SensorLocation(
            id="sensor_001",
            sensor_type="soil_moisture",
            latitude=5.0,
            longitude=5.0,
            installation_date=datetime.now()
        )
        
        zone.add_sensor(sensor)
        assert len(zone.sensors) == 1
        assert len(zone.get_active_sensors()) == 1
        
        # Test sensor outside boundary
        outside_sensor = SensorLocation(
            id="sensor_002",
            sensor_type="weather_station",
            latitude=15.0,  # Outside boundary
            longitude=15.0,
            installation_date=datetime.now()
        )
        
        with pytest.raises(ValueError, match="Sensor location is outside the monitoring zone boundary"):
            zone.add_sensor(outside_sensor)
    
    def test_alert_management(self):
        """Test alert addition and management."""
        geometry = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        
        zone = MonitoringZone(
            id="zone_001",
            name="Test Field",
            geometry=geometry,
            crop_type="wheat",
            planting_date=datetime(2024, 3, 15),
            expected_harvest=datetime(2024, 8, 15)
        )
        
        alert = Alert(
            id="alert_001",
            alert_type="pest_risk",
            severity="high",
            message="High pest risk detected",
            created_at=datetime.now()
        )
        
        zone.add_alert(alert)
        assert len(zone.alerts) == 1
        assert len(zone.get_active_alerts()) == 1
        
        # Resolve alert
        alert.resolve()
        assert len(zone.get_active_alerts()) == 0


class TestAlert:
    """Test cases for Alert model."""
    
    def test_alert_creation(self):
        """Test basic Alert creation."""
        alert = Alert(
            id="alert_001",
            alert_type="disease_risk",
            severity="medium",
            message="Potential disease outbreak detected",
            created_at=datetime.now()
        )
        
        assert alert.id == "alert_001"
        assert alert.severity == "medium"
        assert alert.is_active()
        assert not alert.is_acknowledged()
    
    def test_alert_validation(self):
        """Test Alert validation."""
        with pytest.raises(ValueError, match="Severity must be one of"):
            Alert(
                id="alert_001",
                alert_type="test",
                severity="invalid_severity",
                message="Test message",
                created_at=datetime.now()
            )
    
    def test_alert_lifecycle(self):
        """Test Alert acknowledgment and resolution."""
        alert = Alert(
            id="alert_001",
            alert_type="stress",
            severity="low",
            message="Mild stress detected",
            created_at=datetime.now()
        )
        
        # Initially active and not acknowledged
        assert alert.is_active()
        assert not alert.is_acknowledged()
        
        # Acknowledge
        alert.acknowledge()
        assert alert.is_acknowledged()
        assert alert.is_active()  # Still active until resolved
        
        # Resolve
        alert.resolve()
        assert not alert.is_active()
        assert alert.is_acknowledged()


class TestIndexTimeSeries:
    """Test cases for IndexTimeSeries model."""
    
    def test_index_timeseries_creation(self):
        """Test basic IndexTimeSeries creation."""
        measurement = IndexTimeSeries(
            zone_id="zone_001",
            index_type="NDVI",
            timestamp=datetime.now(),
            mean_value=0.75,
            std_deviation=0.15,
            pixel_count=1000,
            quality_score=0.95
        )
        
        assert measurement.zone_id == "zone_001"
        assert measurement.index_type == "NDVI"
        assert measurement.mean_value == 0.75
        assert measurement.is_high_quality()
        assert measurement.has_sufficient_pixels()
    
    def test_index_timeseries_validation(self):
        """Test IndexTimeSeries validation."""
        # Test invalid quality score
        with pytest.raises(ValueError, match="Quality score must be between 0 and 1"):
            IndexTimeSeries(
                zone_id="zone_001",
                index_type="NDVI",
                timestamp=datetime.now(),
                mean_value=0.75,
                std_deviation=0.15,
                pixel_count=1000,
                quality_score=1.5  # Invalid
            )
        
        # Test invalid index type
        with pytest.raises(ValueError, match="Index type must be one of"):
            IndexTimeSeries(
                zone_id="zone_001",
                index_type="INVALID_INDEX",
                timestamp=datetime.now(),
                mean_value=0.75,
                std_deviation=0.15,
                pixel_count=1000,
                quality_score=0.95
            )


class TestTimeSeriesCollection:
    """Test cases for TimeSeriesCollection."""
    
    def test_timeseries_collection_creation(self):
        """Test TimeSeriesCollection creation and basic operations."""
        collection = TimeSeriesCollection("zone_001", "NDVI")
        
        assert collection.zone_id == "zone_001"
        assert collection.index_type == "NDVI"
        assert collection.get_measurement_count() == 0
    
    def test_timeseries_collection_measurements(self):
        """Test adding and retrieving measurements."""
        collection = TimeSeriesCollection("zone_001", "NDVI")
        
        # Add measurements
        for i in range(5):
            measurement = IndexTimeSeries(
                zone_id="zone_001",
                index_type="NDVI",
                timestamp=datetime.now() - timedelta(days=i),
                mean_value=0.7 + i * 0.05,
                std_deviation=0.1,
                pixel_count=1000,
                quality_score=0.9
            )
            collection.add_measurement(measurement)
        
        assert collection.get_measurement_count() == 5
        
        # Test latest measurement
        latest = collection.get_latest_measurement()
        assert latest is not None
        assert latest.mean_value == 0.7  # Most recent (i=0)
        
        # Test date range filtering
        cutoff_date = datetime.now() - timedelta(days=2.5)  # Include measurements from 2 days ago
        recent_measurements = collection.get_measurements(start_date=cutoff_date)
        assert len(recent_measurements) >= 2  # At least last 2 days
    
    def test_timeseries_collection_statistics(self):
        """Test time series statistics calculation."""
        collection = TimeSeriesCollection("zone_001", "NDVI")
        
        # Add test measurements
        values = [0.6, 0.7, 0.8, 0.75, 0.65]
        for i, value in enumerate(values):
            measurement = IndexTimeSeries(
                zone_id="zone_001",
                index_type="NDVI",
                timestamp=datetime.now() - timedelta(days=i),
                mean_value=value,
                std_deviation=0.1,
                pixel_count=1000,
                quality_score=0.9
            )
            collection.add_measurement(measurement)
        
        stats = collection.get_statistics()
        assert stats['count'] == 5
        assert abs(stats['mean'] - np.mean(values)) < 0.001
        assert stats['min'] == min(values)
        assert stats['max'] == max(values)
    
    def test_timeseries_collection_serialization(self):
        """Test TimeSeriesCollection serialization."""
        collection = TimeSeriesCollection("zone_001", "NDVI")
        
        # Add test measurement
        measurement = IndexTimeSeries(
            zone_id="zone_001",
            index_type="NDVI",
            timestamp=datetime.now(),
            mean_value=0.75,
            std_deviation=0.1,
            pixel_count=1000,
            quality_score=0.9
        )
        collection.add_measurement(measurement)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "timeseries.json")
            
            # Save and load
            collection.save_to_file(filepath)
            loaded_collection = TimeSeriesCollection.from_file(filepath)
            
            assert loaded_collection.zone_id == collection.zone_id
            assert loaded_collection.index_type == collection.index_type
            assert loaded_collection.get_measurement_count() == 1
            
            loaded_measurement = loaded_collection.get_latest_measurement()
            assert loaded_measurement.mean_value == 0.75


if __name__ == "__main__":
    pytest.main([__file__])