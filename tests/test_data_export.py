"""
Tests for data export functionality.
"""

import os
import tempfile
import shutil
import json
from datetime import datetime, timedelta
import pytest
import pandas as pd
import numpy as np

from src.dashboard.data_exporter import (
    DataExporter, 
    generate_mock_vegetation_indices_data,
    generate_mock_sensor_data,
    generate_mock_monitoring_zones
)


class TestDataExporter:
    """Test cases for DataExporter class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
        
        # Generate mock data
        self.zones = ['zone_1', 'zone_2', 'zone_3']
        self.vegetation_data = generate_mock_vegetation_indices_data(self.zones, days=30)
        self.sensor_data = generate_mock_sensor_data(self.zones, hours=168)
        self.monitoring_zones = generate_mock_monitoring_zones(len(self.zones))
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_export_vegetation_indices_csv(self):
        """Test vegetation indices CSV export"""
        
        # Test basic export
        filepath = self.exporter.export_vegetation_indices_csv(self.vegetation_data)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.csv')
        
        # Verify CSV content
        df = pd.read_csv(filepath)
        assert len(df) > 0
        assert 'zone_id' in df.columns
        assert 'index_type' in df.columns
        assert 'timestamp' in df.columns
        assert 'mean_value' in df.columns
        
        # Test with zone filter
        filtered_filepath = self.exporter.export_vegetation_indices_csv(
            self.vegetation_data, zones=['zone_1']
        )
        
        filtered_df = pd.read_csv(filtered_filepath)
        assert all(filtered_df['zone_id'] == 'zone_1')
        
        # Test with index filter
        ndvi_filepath = self.exporter.export_vegetation_indices_csv(
            self.vegetation_data, indices=['NDVI']
        )
        
        ndvi_df = pd.read_csv(ndvi_filepath)
        assert all(ndvi_df['index_type'] == 'NDVI')
    
    def test_export_satellite_imagery_geotiff(self):
        """Test satellite imagery GeoTIFF export"""
        
        mock_images = [{"acquisition_date": datetime.now().isoformat()}]
        
        filepath = self.exporter.export_satellite_imagery_geotiff(
            mock_images, index_type='NDVI'
        )
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.tif')
        
        # Verify file is not empty
        assert os.path.getsize(filepath) > 0
    
    def test_export_monitoring_zones_geojson(self):
        """Test monitoring zones GeoJSON export"""
        
        filepath = self.exporter.export_monitoring_zones_geojson(self.monitoring_zones)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.geojson')
        
        # Verify GeoJSON structure
        with open(filepath, 'r') as f:
            geojson = json.load(f)
        
        assert geojson['type'] == 'FeatureCollection'
        assert 'features' in geojson
        assert len(geojson['features']) == len(self.monitoring_zones)
        
        # Verify feature structure
        feature = geojson['features'][0]
        assert feature['type'] == 'Feature'
        assert 'geometry' in feature
        assert 'properties' in feature
        assert 'name' in feature['properties']
    
    def test_export_sensor_data_csv(self):
        """Test sensor data CSV export"""
        
        filepath = self.exporter.export_sensor_data_csv(self.sensor_data)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.csv')
        
        # Verify CSV content
        df = pd.read_csv(filepath)
        assert len(df) > 0
        assert 'zone_id' in df.columns
        assert 'sensor_type' in df.columns
        assert 'timestamp' in df.columns
        assert 'value' in df.columns
        
        # Test with sensor type filter
        moisture_filepath = self.exporter.export_sensor_data_csv(
            self.sensor_data, sensor_types=['soil_moisture']
        )
        
        moisture_df = pd.read_csv(moisture_filepath)
        assert all(moisture_df['sensor_type'] == 'soil_moisture')
    
    def test_export_alerts_csv(self):
        """Test alerts CSV export"""
        
        # Generate mock alert data
        mock_alerts = [
            {
                'id': f'alert_{i}',
                'zone_id': f'zone_{(i % 3) + 1}',
                'severity': ['low', 'medium', 'high', 'critical'][i % 4],
                'message': f'Test alert {i}',
                'status': 'active' if i % 2 == 0 else 'resolved',
                'created_at': datetime.now() - timedelta(days=i)
            }
            for i in range(10)
        ]
        
        filepath = self.exporter.export_alerts_csv(mock_alerts)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.csv')
        
        # Verify CSV content
        df = pd.read_csv(filepath)
        assert len(df) == 10
        assert 'zone_id' in df.columns
        assert 'severity' in df.columns
        assert 'status' in df.columns
        
        # Test with severity filter
        critical_filepath = self.exporter.export_alerts_csv(
            mock_alerts, severities=['critical']
        )
        
        critical_df = pd.read_csv(critical_filepath)
        assert all(critical_df['severity'] == 'critical')
    
    def test_create_batch_export(self):
        """Test batch export functionality"""
        
        # Create batch export configurations
        export_configs = [
            {
                'type': 'vegetation_indices',
                'data': self.vegetation_data,
                'zones': ['zone_1', 'zone_2'],
                'indices': ['NDVI', 'SAVI']
            },
            {
                'type': 'sensor_data',
                'data': self.sensor_data,
                'zones': ['zone_1'],
                'sensor_types': ['soil_moisture']
            },
            {
                'type': 'monitoring_zones',
                'data': self.monitoring_zones
            }
        ]
        
        # Test compressed batch export
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        assert os.path.exists(zip_filepath)
        assert zip_filepath.endswith('.zip')
        assert os.path.getsize(zip_filepath) > 0
        
        # Test uncompressed batch export
        dir_path = self.exporter.create_batch_export(export_configs, compress=False)
        
        assert os.path.exists(dir_path)
        assert os.path.isdir(dir_path)
        
        # Verify manifest file exists
        manifest_path = os.path.join(dir_path, 'export_manifest.json')
        assert os.path.exists(manifest_path)
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        assert 'export_date' in manifest
        assert 'total_files' in manifest
        assert 'files' in manifest
        assert manifest['total_files'] > 0
    
    def test_get_export_history(self):
        """Test export history functionality"""
        
        # Create some export files
        self.exporter.export_vegetation_indices_csv(self.vegetation_data)
        self.exporter.export_sensor_data_csv(self.sensor_data)
        
        history = self.exporter.get_export_history()
        
        assert len(history) >= 2
        
        # Verify history structure
        for item in history:
            assert 'filename' in item
            assert 'filepath' in item
            assert 'size_bytes' in item
            assert 'size_mb' in item
            assert 'created' in item
            assert 'type' in item
    
    def test_cleanup_old_exports(self):
        """Test cleanup of old export files"""
        
        # Create some export files
        filepath1 = self.exporter.export_vegetation_indices_csv(self.vegetation_data)
        filepath2 = self.exporter.export_sensor_data_csv(self.sensor_data)
        
        # Verify files exist
        assert os.path.exists(filepath1)
        assert os.path.exists(filepath2)
        
        # Test cleanup with 0 days (should delete all files)
        deleted_count = self.exporter.cleanup_old_exports(days_old=0)
        
        assert deleted_count >= 2
        assert not os.path.exists(filepath1)
        assert not os.path.exists(filepath2)
    
    def test_date_filtering(self):
        """Test date filtering in exports"""
        
        # Create data with specific date range
        start_date = datetime.now() - timedelta(days=10)
        end_date = datetime.now() - timedelta(days=5)
        
        filepath = self.exporter.export_vegetation_indices_csv(
            self.vegetation_data,
            start_date=start_date,
            end_date=end_date
        )
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Verify all timestamps are within the specified range
        assert all(df['timestamp'] >= start_date)
        assert all(df['timestamp'] <= end_date)
    
    def test_error_handling(self):
        """Test error handling in export functions"""
        
        # Test with empty data
        with pytest.raises(ValueError):
            self.exporter.export_vegetation_indices_csv([])
        
        with pytest.raises(ValueError):
            self.exporter.export_satellite_imagery_geotiff([])
        
        with pytest.raises(ValueError):
            self.exporter.export_monitoring_zones_geojson([])
        
        with pytest.raises(ValueError):
            self.exporter.export_sensor_data_csv([])
        
        with pytest.raises(ValueError):
            self.exporter.export_alerts_csv([])


class TestMockDataGeneration:
    """Test cases for mock data generation functions"""
    
    def test_generate_mock_vegetation_indices_data(self):
        """Test mock vegetation indices data generation"""
        
        zones = ['zone_1', 'zone_2']
        days = 30
        
        data = generate_mock_vegetation_indices_data(zones, days)
        
        assert len(data) > 0
        
        # Verify data structure
        for item in data:
            assert 'zone_id' in item
            assert 'index_type' in item
            assert 'timestamp' in item
            assert 'mean_value' in item
            assert 'std_deviation' in item
            assert 'pixel_count' in item
            assert 'quality_score' in item
            
            # Verify value ranges
            assert item['zone_id'] in zones
            assert item['index_type'] in ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI']
            assert -1 <= item['mean_value'] <= 1
            assert 0 <= item['quality_score'] <= 1
    
    def test_generate_mock_sensor_data(self):
        """Test mock sensor data generation"""
        
        zones = ['zone_1', 'zone_2']
        hours = 24
        
        data = generate_mock_sensor_data(zones, hours)
        
        assert len(data) > 0
        
        # Verify data structure
        for item in data:
            assert 'zone_id' in item
            assert 'sensor_type' in item
            assert 'timestamp' in item
            assert 'value' in item
            assert 'unit' in item
            assert 'sensor_id' in item
            
            # Verify value ranges based on sensor type
            if item['sensor_type'] == 'soil_moisture':
                assert 30 <= item['value'] <= 80
                assert item['unit'] == '%'
            elif item['sensor_type'] == 'air_temperature':
                assert 15 <= item['value'] <= 35
                assert item['unit'] == '°C'
    
    def test_generate_mock_monitoring_zones(self):
        """Test mock monitoring zones data generation"""
        
        count = 3
        zones = generate_mock_monitoring_zones(count)
        
        assert len(zones) == count
        
        # Verify zone structure
        for zone in zones:
            assert 'id' in zone
            assert 'name' in zone
            assert 'crop_type' in zone
            assert 'area' in zone
            assert 'planting_date' in zone
            assert 'expected_harvest' in zone
            assert 'alerts' in zone
            assert 'sensors' in zone
            
            # Verify area is reasonable
            assert 50 <= zone['area'] <= 300


if __name__ == "__main__":
    pytest.main([__file__])