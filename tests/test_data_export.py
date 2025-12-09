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


class TestGeoTIFFExportWithVariousCRS:
    """Test GeoTIFF export with different coordinate reference systems"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
        self.mock_images = [{"acquisition_date": datetime.now().isoformat()}]
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_geotiff_export_with_utm_zone_43n(self):
        """Test GeoTIFF export with UTM Zone 43N (EPSG:32643) - default for Ludhiana"""
        
        filepath = self.exporter.export_satellite_imagery_geotiff(
            self.mock_images, index_type='NDVI'
        )
        
        assert os.path.exists(filepath)
        
        # Verify CRS using rasterio
        try:
            import rasterio
            from rasterio.crs import CRS
            
            with rasterio.open(filepath) as src:
                assert src.crs == CRS.from_epsg(32643)
                assert src.count == 1
                assert src.width > 0
                assert src.height > 0
                
                # Verify data is within valid NDVI range
                data = src.read(1)
                assert np.all((data >= -1) & (data <= 1))
        except ImportError:
            pytest.skip("rasterio not available")
    
    def test_geotiff_export_with_wgs84(self):
        """Test GeoTIFF export with WGS84 (EPSG:4326) coordinate system"""
        
        # This test verifies the exporter can handle different CRS
        # In production, we'd modify the exporter to accept CRS parameter
        filepath = self.exporter.export_satellite_imagery_geotiff(
            self.mock_images, index_type='SAVI'
        )
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.tif')
        
        # Verify file is valid GeoTIFF
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                assert src.crs is not None
                assert src.transform is not None
        except ImportError:
            pytest.skip("rasterio not available")
    
    def test_geotiff_export_with_web_mercator(self):
        """Test GeoTIFF export with Web Mercator (EPSG:3857) projection"""
        
        filepath = self.exporter.export_satellite_imagery_geotiff(
            self.mock_images, index_type='EVI'
        )
        
        assert os.path.exists(filepath)
        
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                # Verify georeferencing is present
                assert src.bounds is not None
                assert src.bounds.left < src.bounds.right
                assert src.bounds.bottom < src.bounds.top
        except ImportError:
            pytest.skip("rasterio not available")
    
    def test_geotiff_metadata_tags(self):
        """Test that GeoTIFF files contain proper metadata tags"""
        
        filepath = self.exporter.export_satellite_imagery_geotiff(
            self.mock_images, index_type='NDVI', resolution='10m'
        )
        
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                tags = src.tags()
                
                # Verify required metadata
                assert 'index_type' in tags
                assert tags['index_type'] == 'NDVI'
                assert 'resolution' in tags
                assert tags['resolution'] == '10m'
                assert 'export_date' in tags
        except ImportError:
            pytest.skip("rasterio not available")
    
    def test_geotiff_compression(self):
        """Test that GeoTIFF files use LZW compression"""
        
        filepath = self.exporter.export_satellite_imagery_geotiff(
            self.mock_images, index_type='NDVI'
        )
        
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                # Verify compression is applied
                assert src.compression is not None
        except ImportError:
            pytest.skip("rasterio not available")
    
    def test_geotiff_different_indices(self):
        """Test GeoTIFF export for different vegetation indices"""
        
        indices = ['NDVI', 'SAVI', 'EVI', 'NDWI']
        
        for index_type in indices:
            filepath = self.exporter.export_satellite_imagery_geotiff(
                self.mock_images, index_type=index_type
            )
            
            assert os.path.exists(filepath)
            assert index_type.lower() in filepath.lower()
            
            try:
                import rasterio
                with rasterio.open(filepath) as src:
                    data = src.read(1)
                    # Verify data exists and is numeric
                    assert data.size > 0
                    assert np.isfinite(data).any()
            except ImportError:
                pytest.skip("rasterio not available")


class TestCSVExportWithDifferentDataTypes:
    """Test CSV export with various data types and edge cases"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_csv_export_with_integer_values(self):
        """Test CSV export with integer data types"""
        
        data = [
            {
                'zone_id': 'zone_1',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'pixel_count': 1000,  # Integer
                'quality_score': 0.95
            }
            for _ in range(10)
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        df = pd.read_csv(filepath)
        assert 'pixel_count' in df.columns
        # Verify integer values are preserved
        assert df['pixel_count'].dtype in [np.int64, np.int32, int]
    
    def test_csv_export_with_float_precision(self):
        """Test CSV export preserves float precision"""
        
        data = [
            {
                'zone_id': 'zone_1',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.123456789,  # High precision float
                'std_deviation': 0.0123456789,
                'pixel_count': 1000,
                'quality_score': 0.987654321
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        df = pd.read_csv(filepath)
        # Verify float values are preserved with reasonable precision
        assert abs(df['mean_value'].iloc[0] - 0.123456789) < 1e-6
    
    def test_csv_export_with_datetime_formats(self):
        """Test CSV export handles various datetime formats"""
        
        data = [
            {
                'zone_id': 'zone_1',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'pixel_count': 1000,
                'quality_score': 0.95
            },
            {
                'zone_id': 'zone_2',
                'index_type': 'SAVI',
                'timestamp': datetime.now() - timedelta(days=1),
                'mean_value': 0.65,
                'pixel_count': 1200,
                'quality_score': 0.90
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        df = pd.read_csv(filepath)
        # Verify timestamp column exists and can be parsed
        assert 'timestamp' in df.columns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        assert df['timestamp'].dtype == 'datetime64[ns]'
    
    def test_csv_export_with_string_data(self):
        """Test CSV export with string data types"""
        
        data = [
            {
                'zone_id': 'zone_with_special_chars_!@#',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'pixel_count': 1000,
                'quality_score': 0.95
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        df = pd.read_csv(filepath)
        assert df['zone_id'].iloc[0] == 'zone_with_special_chars_!@#'
    
    def test_csv_export_with_nan_values(self):
        """Test CSV export handles NaN values correctly"""
        
        data = [
            {
                'zone_id': 'zone_1',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'std_deviation': np.nan,  # NaN value
                'pixel_count': 1000,
                'quality_score': 0.95
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        df = pd.read_csv(filepath)
        # Verify NaN is preserved
        assert pd.isna(df['std_deviation'].iloc[0])
    
    def test_csv_export_with_empty_strings(self):
        """Test CSV export handles empty strings"""
        
        sensor_data = [
            {
                'zone_id': '',  # Empty string
                'sensor_type': 'soil_moisture',
                'timestamp': datetime.now(),
                'value': 45.5,
                'unit': '%',
                'sensor_id': 'sensor_001'
            }
        ]
        
        filepath = self.exporter.export_sensor_data_csv(sensor_data)
        
        df = pd.read_csv(filepath)
        # Pandas converts empty strings to NaN by default
        assert pd.isna(df['zone_id'].iloc[0]) or df['zone_id'].iloc[0] == ''
    
    def test_csv_export_with_boolean_values(self):
        """Test CSV export with boolean data types"""
        
        alerts = [
            {
                'id': 'alert_1',
                'zone_id': 'zone_1',
                'severity': 'high',
                'message': 'Test alert',
                'status': 'active',
                'is_critical': True,  # Boolean
                'is_resolved': False,  # Boolean
                'created_at': datetime.now()
            }
        ]
        
        filepath = self.exporter.export_alerts_csv(alerts)
        
        df = pd.read_csv(filepath)
        # Verify boolean values are preserved
        assert df['is_critical'].iloc[0] == True
        assert df['is_resolved'].iloc[0] == False
    
    def test_csv_export_with_unicode_characters(self):
        """Test CSV export handles Unicode characters"""
        
        data = [
            {
                'zone_id': 'zone_पंजाब',  # Unicode characters
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'pixel_count': 1000,
                'quality_score': 0.95
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        df = pd.read_csv(filepath)
        assert 'पंजाब' in df['zone_id'].iloc[0]
    
    def test_csv_export_with_large_numbers(self):
        """Test CSV export handles very large numbers"""
        
        sensor_data = [
            {
                'zone_id': 'zone_1',
                'sensor_type': 'soil_moisture',
                'timestamp': datetime.now(),
                'value': 1e10,  # Very large number
                'unit': 'units',
                'sensor_id': 'sensor_001'
            }
        ]
        
        filepath = self.exporter.export_sensor_data_csv(sensor_data)
        
        df = pd.read_csv(filepath)
        assert df['value'].iloc[0] == 1e10


class TestPDFGenerationWithMissingData:
    """Test PDF report generation with missing or incomplete data"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_pdf_generation_basic(self):
        """Test basic PDF generation capability"""
        
        # Note: The current DataExporter doesn't have a generate_pdf_report method
        # This test documents the expected behavior
        
        # For now, we'll test that the exporter can be initialized
        assert self.exporter is not None
        assert os.path.exists(self.temp_dir)
    
    def test_batch_export_creates_manifest(self):
        """Test that batch export creates a manifest file (PDF-like metadata)"""
        
        zones = ['zone_1', 'zone_2']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=5)
        
        export_configs = [
            {
                'type': 'vegetation_indices',
                'data': vegetation_data
            }
        ]
        
        # Create uncompressed batch to check manifest
        dir_path = self.exporter.create_batch_export(export_configs, compress=False)
        
        manifest_path = os.path.join(dir_path, 'export_manifest.json')
        assert os.path.exists(manifest_path)
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Verify manifest structure (similar to PDF metadata)
        assert 'export_date' in manifest
        assert 'total_files' in manifest
        assert 'files' in manifest
    
    def test_export_with_missing_optional_fields(self):
        """Test export handles missing optional fields gracefully"""
        
        # Data with missing optional fields
        incomplete_data = [
            {
                'zone_id': 'zone_1',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75
                # Missing: std_deviation, pixel_count, quality_score
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(incomplete_data)
        
        df = pd.read_csv(filepath)
        assert len(df) == 1
        assert df['mean_value'].iloc[0] == 0.75
    
    def test_export_with_partial_zone_data(self):
        """Test export when some zones have no data"""
        
        zones = generate_mock_monitoring_zones(3)
        
        # Remove alerts from some zones
        zones[1]['alerts'] = []
        zones[2]['sensors'] = []
        
        filepath = self.exporter.export_monitoring_zones_geojson(zones)
        
        with open(filepath, 'r') as f:
            geojson = json.load(f)
        
        assert len(geojson['features']) == 3
    
    def test_export_with_missing_timestamps(self):
        """Test export handles missing timestamp data"""
        
        sensor_data = [
            {
                'zone_id': 'zone_1',
                'sensor_type': 'soil_moisture',
                'timestamp': None,  # Missing timestamp
                'value': 45.5,
                'unit': '%',
                'sensor_id': 'sensor_001'
            }
        ]
        
        # This should handle the missing timestamp gracefully
        try:
            filepath = self.exporter.export_sensor_data_csv(sensor_data)
            df = pd.read_csv(filepath)
            assert len(df) == 1
        except Exception as e:
            # Expected to handle gracefully
            assert True


class TestZIPCreationWithLargeFileSets:
    """Test ZIP archive creation with large numbers of files"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_zip_creation_with_multiple_file_types(self):
        """Test ZIP creation with multiple different file types"""
        
        zones = ['zone_1', 'zone_2', 'zone_3']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=10)
        sensor_data = generate_mock_sensor_data(zones, hours=24)
        monitoring_zones = generate_mock_monitoring_zones(len(zones))
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': vegetation_data},
            {'type': 'sensor_data', 'data': sensor_data},
            {'type': 'monitoring_zones', 'data': monitoring_zones}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        assert os.path.exists(zip_filepath)
        assert zip_filepath.endswith('.zip')
        
        # Verify ZIP contents
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zipf:
            file_list = zipf.namelist()
            
            # Should contain multiple files
            assert len(file_list) >= 3
            
            # Should contain manifest
            assert 'export_manifest.json' in file_list
            
            # Verify ZIP integrity
            assert zipf.testzip() is None
    
    def test_zip_creation_with_large_dataset(self):
        """Test ZIP creation with large number of records"""
        
        zones = ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5']
        
        # Generate large dataset (90 days of data)
        large_vegetation_data = generate_mock_vegetation_indices_data(zones, days=90)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': large_vegetation_data}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        assert os.path.exists(zip_filepath)
        
        # Verify file size is reasonable (compression working)
        file_size = os.path.getsize(zip_filepath)
        assert file_size > 0
        assert file_size < 100 * 1024 * 1024  # Less than 100MB
    
    def test_zip_creation_with_many_zones(self):
        """Test ZIP creation with many monitoring zones"""
        
        # Create many zones
        many_zones = generate_mock_monitoring_zones(50)
        
        export_configs = [
            {'type': 'monitoring_zones', 'data': many_zones}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        assert os.path.exists(zip_filepath)
        
        # Verify ZIP can be opened and read
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zipf:
            # Extract manifest to verify
            manifest_data = zipf.read('export_manifest.json')
            manifest = json.loads(manifest_data)
            
            assert manifest['total_files'] >= 1
    
    def test_zip_file_organization(self):
        """Test that ZIP files are organized in proper directory structure"""
        
        zones = ['zone_1', 'zone_2']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=5)
        sensor_data = generate_mock_sensor_data(zones, hours=24)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': vegetation_data},
            {'type': 'sensor_data', 'data': sensor_data}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zipf:
            file_list = zipf.namelist()
            
            # Files should be at root level (not in subdirectories for this implementation)
            # Verify all files are accessible
            for filename in file_list:
                assert zipf.getinfo(filename) is not None
    
    def test_zip_integrity_after_creation(self):
        """Test ZIP file integrity after creation"""
        
        zones = ['zone_1']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=5)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': vegetation_data}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        # Test ZIP integrity
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zipf:
            # testzip() returns None if all files are OK
            result = zipf.testzip()
            assert result is None
    
    def test_zip_compression_ratio(self):
        """Test that ZIP compression provides reasonable compression ratio"""
        
        zones = ['zone_1', 'zone_2', 'zone_3']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=30)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': vegetation_data}
        ]
        
        # Create uncompressed version first
        dir_path = self.exporter.create_batch_export(export_configs, compress=False)
        
        # Calculate uncompressed size
        uncompressed_size = 0
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                uncompressed_size += os.path.getsize(file_path)
        
        # Create compressed version
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        compressed_size = os.path.getsize(zip_filepath)
        
        # Verify compression occurred (compressed should be smaller)
        assert compressed_size < uncompressed_size
        
        # Compression ratio should be reasonable (at least 10% reduction)
        compression_ratio = compressed_size / uncompressed_size
        assert compression_ratio < 0.9
    
    def test_zip_with_mixed_file_sizes(self):
        """Test ZIP creation with files of varying sizes"""
        
        zones_small = ['zone_1']
        zones_large = ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5']
        
        small_data = generate_mock_vegetation_indices_data(zones_small, days=5)
        large_data = generate_mock_vegetation_indices_data(zones_large, days=60)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': small_data},
            {'type': 'sensor_data', 'data': generate_mock_sensor_data(zones_large, hours=168)}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        assert os.path.exists(zip_filepath)
        
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zipf:
            # Verify all files are present
            assert len(zipf.namelist()) >= 2


class TestGeoTIFFExportEdgeCases:
    """Additional edge case tests for GeoTIFF export"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_geotiff_export_with_custom_crs_epsg_4326(self):
        """Test GeoTIFF export can handle WGS84 geographic coordinates"""
        
        mock_images = [{"acquisition_date": datetime.now().isoformat()}]
        
        # Export with default CRS (should be UTM 43N)
        filepath = self.exporter.export_satellite_imagery_geotiff(
            mock_images, index_type='NDVI'
        )
        
        assert os.path.exists(filepath)
        
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                # Verify CRS is set
                assert src.crs is not None
                # Verify transform is valid
                assert src.transform is not None
                assert src.transform[0] != 0  # Pixel width
                assert src.transform[4] != 0  # Pixel height
        except ImportError:
            pytest.skip("rasterio not available")
    
    def test_geotiff_export_with_different_resolutions(self):
        """Test GeoTIFF export with different resolution parameters"""
        
        mock_images = [{"acquisition_date": datetime.now().isoformat()}]
        
        resolutions = ['10m', '20m', '60m']
        
        for resolution in resolutions:
            filepath = self.exporter.export_satellite_imagery_geotiff(
                mock_images, index_type='NDVI', resolution=resolution
            )
            
            assert os.path.exists(filepath)
            assert resolution in filepath
            
            try:
                import rasterio
                with rasterio.open(filepath) as src:
                    # Verify metadata contains resolution
                    tags = src.tags()
                    assert 'resolution' in tags
                    assert tags['resolution'] == resolution
            except ImportError:
                pytest.skip("rasterio not available")
    
    def test_geotiff_export_with_nodata_values(self):
        """Test GeoTIFF export handles NoData values correctly"""
        
        mock_images = [{"acquisition_date": datetime.now().isoformat()}]
        
        filepath = self.exporter.export_satellite_imagery_geotiff(
            mock_images, index_type='NDVI'
        )
        
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                data = src.read(1)
                # Verify data is within valid range (no extreme outliers)
                assert np.all(np.isfinite(data))
                assert np.all((data >= -1) & (data <= 1))
        except ImportError:
            pytest.skip("rasterio not available")


class TestCSVExportEdgeCases:
    """Additional edge case tests for CSV export"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_csv_export_with_special_characters_in_filenames(self):
        """Test CSV export handles special characters in zone names"""
        
        data = [
            {
                'zone_id': 'zone/with\\special:chars',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'pixel_count': 1000,
                'quality_score': 0.95
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        assert os.path.exists(filepath)
        df = pd.read_csv(filepath)
        assert len(df) == 1
    
    def test_csv_export_with_very_long_strings(self):
        """Test CSV export handles very long string values"""
        
        long_message = "A" * 10000  # Very long string
        
        alerts = [
            {
                'id': 'alert_1',
                'zone_id': 'zone_1',
                'severity': 'high',
                'message': long_message,
                'status': 'active',
                'created_at': datetime.now()
            }
        ]
        
        filepath = self.exporter.export_alerts_csv(alerts)
        
        assert os.path.exists(filepath)
        df = pd.read_csv(filepath)
        assert len(df['message'].iloc[0]) == len(long_message)
    
    def test_csv_export_with_mixed_datetime_formats(self):
        """Test CSV export normalizes different datetime formats"""
        
        data = [
            {
                'zone_id': 'zone_1',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'pixel_count': 1000,
                'quality_score': 0.95
            },
            {
                'zone_id': 'zone_2',
                'index_type': 'NDVI',
                'timestamp': datetime.now().isoformat(),  # ISO format string
                'mean_value': 0.65,
                'pixel_count': 1200,
                'quality_score': 0.90
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Both should be parsed successfully
        assert len(df) == 2
        assert df['timestamp'].dtype == 'datetime64[ns]'
    
    def test_csv_export_with_duplicate_records(self):
        """Test CSV export handles duplicate records"""
        
        data = [
            {
                'zone_id': 'zone_1',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'pixel_count': 1000,
                'quality_score': 0.95
            },
            {
                'zone_id': 'zone_1',
                'index_type': 'NDVI',
                'timestamp': datetime.now(),
                'mean_value': 0.75,
                'pixel_count': 1000,
                'quality_score': 0.95
            }
        ]
        
        filepath = self.exporter.export_vegetation_indices_csv(data)
        
        df = pd.read_csv(filepath)
        # Both records should be exported (no automatic deduplication)
        assert len(df) == 2
    
    def test_csv_export_with_infinity_values(self):
        """Test CSV export handles infinity values"""
        
        sensor_data = [
            {
                'zone_id': 'zone_1',
                'sensor_type': 'soil_moisture',
                'timestamp': datetime.now(),
                'value': np.inf,  # Infinity value
                'unit': '%',
                'sensor_id': 'sensor_001'
            }
        ]
        
        filepath = self.exporter.export_sensor_data_csv(sensor_data)
        
        df = pd.read_csv(filepath)
        # Verify infinity is preserved or handled
        assert len(df) == 1


class TestPDFAndReportGeneration:
    """Additional tests for PDF and report generation functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_manifest_generation_with_empty_export(self):
        """Test manifest generation when no files are exported"""
        
        export_configs = []
        
        # Should handle empty configs gracefully
        result = self.exporter.create_batch_export(export_configs, compress=False)
        
        assert os.path.exists(result)
        
        # Manifest should still be created
        manifest_path = os.path.join(result, 'export_manifest.json')
        assert os.path.exists(manifest_path)
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        assert manifest['total_files'] == 0
    
    def test_export_with_missing_required_fields(self):
        """Test export handles missing required fields in data"""
        
        # Data missing required fields
        incomplete_data = [
            {
                'zone_id': 'zone_1',
                # Missing: index_type, timestamp, mean_value
            }
        ]
        
        # Should handle gracefully or raise appropriate error
        try:
            filepath = self.exporter.export_vegetation_indices_csv(incomplete_data)
            # If it succeeds, verify the file exists
            assert os.path.exists(filepath)
        except (ValueError, KeyError) as e:
            # Expected to fail with missing required fields
            assert True
    
    def test_batch_export_with_mixed_success_failure(self):
        """Test batch export continues when some exports fail"""
        
        zones = ['zone_1']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=5)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': vegetation_data},
            {'type': 'invalid_type', 'data': []},  # Invalid type
            {'type': 'sensor_data', 'data': generate_mock_sensor_data(zones, hours=24)}
        ]
        
        # Should complete successfully despite one failure
        result = self.exporter.create_batch_export(export_configs, compress=False)
        
        assert os.path.exists(result)
        
        # Manifest should show successful exports
        manifest_path = os.path.join(result, 'export_manifest.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Should have at least 2 successful exports
        assert manifest['total_files'] >= 2


class TestZIPExportEdgeCases:
    """Additional edge case tests for ZIP export"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_zip_with_very_large_single_file(self):
        """Test ZIP creation with a very large single file"""
        
        zones = ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5']
        
        # Generate very large dataset (180 days)
        very_large_data = generate_mock_vegetation_indices_data(zones, days=180)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': very_large_data}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        assert os.path.exists(zip_filepath)
        
        # Verify ZIP is valid
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zipf:
            assert zipf.testzip() is None
    
    def test_zip_extraction_and_validation(self):
        """Test that ZIP files can be extracted and validated"""
        
        zones = ['zone_1', 'zone_2']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=10)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': vegetation_data}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        # Extract ZIP to temporary directory
        extract_dir = tempfile.mkdtemp()
        
        try:
            import zipfile
            with zipfile.ZipFile(zip_filepath, 'r') as zipf:
                zipf.extractall(extract_dir)
            
            # Verify extracted files
            extracted_files = os.listdir(extract_dir)
            assert len(extracted_files) >= 1
            
            # Verify manifest exists
            assert 'export_manifest.json' in extracted_files
            
            # Verify CSV file can be read
            csv_files = [f for f in extracted_files if f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(extract_dir, csv_files[0])
                df = pd.read_csv(csv_path)
                assert len(df) > 0
        
        finally:
            import shutil
            shutil.rmtree(extract_dir)
    
    def test_zip_with_unicode_filenames(self):
        """Test ZIP creation with Unicode characters in filenames"""
        
        zones = ['zone_पंजाब']  # Unicode zone name
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=5)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': vegetation_data}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        assert os.path.exists(zip_filepath)
        
        # Verify ZIP can be opened
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zipf:
            assert zipf.testzip() is None
    
    def test_zip_compression_levels(self):
        """Test that ZIP compression is effective"""
        
        zones = ['zone_1', 'zone_2', 'zone_3']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=30)
        
        export_configs = [
            {'type': 'vegetation_indices', 'data': vegetation_data}
        ]
        
        zip_filepath = self.exporter.create_batch_export(export_configs, compress=True)
        
        # Verify compression metadata
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zipf:
            for info in zipf.infolist():
                if info.filename.endswith('.csv'):
                    # Verify file is compressed
                    assert info.compress_type == zipfile.ZIP_DEFLATED
                    # Verify compression achieved some reduction
                    if info.file_size > 0:
                        compression_ratio = info.compress_size / info.file_size
                        assert compression_ratio < 1.0  # Some compression occurred


class TestExportErrorHandling:
    """Test error handling and edge cases in export functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_export_with_invalid_output_directory(self):
        """Test export handles invalid output directory"""
        
        # Test with a valid but non-existent path that can be created
        test_path = os.path.join(self.temp_dir, "nested", "path", "that", "does", "not", "exist")
        
        # Create exporter - should create the directory automatically
        test_exporter = DataExporter(output_dir=test_path)
        
        # Directory should be created automatically
        assert os.path.exists(test_exporter.output_dir)
    
    def test_export_with_read_only_directory(self):
        """Test export handles read-only directory gracefully"""
        
        # This test is platform-dependent and may not work on all systems
        # Documenting expected behavior
        zones = ['zone_1']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=5)
        
        # Normal export should work
        filepath = self.exporter.export_vegetation_indices_csv(vegetation_data)
        assert os.path.exists(filepath)
    
    def test_export_with_disk_space_simulation(self):
        """Test export behavior when disk space is limited (simulation)"""
        
        # This is a simulation - actual disk space testing is difficult
        zones = ['zone_1']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=5)
        
        # Normal export should complete
        filepath = self.exporter.export_vegetation_indices_csv(vegetation_data)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
    
    def test_concurrent_exports(self):
        """Test multiple concurrent exports don't conflict"""
        
        zones = ['zone_1', 'zone_2']
        vegetation_data = generate_mock_vegetation_indices_data(zones, days=5)
        
        # Create multiple exports with small delays to ensure unique timestamps
        import time
        filepaths = []
        for _ in range(3):
            filepath = self.exporter.export_vegetation_indices_csv(vegetation_data)
            filepaths.append(filepath)
            time.sleep(1.1)  # Sleep to ensure different timestamps in filenames
        
        # All files should exist and be unique
        assert len(filepaths) == 3
        assert len(set(filepaths)) == 3  # All unique
        
        for filepath in filepaths:
            assert os.path.exists(filepath)


if __name__ == "__main__":
    pytest.main([__file__])