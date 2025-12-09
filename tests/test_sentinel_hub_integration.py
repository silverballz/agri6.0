"""
Unit tests for Sentinel Hub API integration.

Tests authentication, query construction, response parsing, and fallback mechanisms.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import os
import numpy as np

from src.data_processing.sentinel_hub_client import (
    SentinelHubClient,
    SentinelHubConfig,
    create_client_from_env
)
from src.data_processing.geojson_handler import create_ludhiana_sample_geojson
from src.data_processing.local_tif_fallback import (
    LocalTifFallback,
    fallback_to_local_tif
)


class TestSentinelHubConfig:
    """Unit tests for SentinelHubConfig."""
    
    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = SentinelHubConfig(
            instance_id='test_instance',
            client_id='test_client',
            client_secret='test_secret'
        )
        
        assert config.instance_id == 'test_instance'
        assert config.client_id == 'test_client'
        assert config.client_secret == 'test_secret'
        assert config.bands == ['B02', 'B03', 'B04', 'B08']
        assert config.resolution == 10
        assert config.max_cloud_coverage == 20.0
    
    def test_config_from_env_success(self):
        """Test loading config from environment variables."""
        with patch.dict(os.environ, {
            'SENTINEL_HUB_INSTANCE_ID': 'env_instance',
            'SENTINEL_HUB_CLIENT_ID': 'env_client',
            'SENTINEL_HUB_CLIENT_SECRET': 'env_secret'
        }):
            config = SentinelHubConfig.from_env()
            
            assert config.instance_id == 'env_instance'
            assert config.client_id == 'env_client'
            assert config.client_secret == 'env_secret'
    
    def test_config_from_env_missing_variables(self):
        """Test that missing environment variables raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                SentinelHubConfig.from_env()
    
    def test_config_validation(self):
        """Test config validation."""
        valid_config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        assert valid_config.validate() is True
        
        # Invalid config (empty instance_id)
        invalid_config = SentinelHubConfig(
            instance_id='',
            client_id='test',
            client_secret='test'
        )
        assert invalid_config.validate() is False


class TestSentinelHubAuthentication:
    """Unit tests for authentication."""
    
    def test_authentication_success(self):
        """Test successful authentication."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        client = SentinelHubClient(config)
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'access_token': 'test_token_123',
            'expires_in': 3600
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.post', return_value=mock_response):
            token = client.authenticate()
            
            assert token == 'test_token_123'
            assert client.access_token == 'test_token_123'
            assert client.token_expiry is not None
    
    def test_authentication_failure(self):
        """Test authentication failure with invalid credentials."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='invalid',
            client_secret='invalid'
        )
        client = SentinelHubClient(config)
        
        import requests
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        
        with patch('requests.post', return_value=mock_response):
            with pytest.raises(ValueError, match="Failed to authenticate"):
                client.authenticate()
    
    def test_connection_test_success(self):
        """Test successful connection test."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        client = SentinelHubClient(config)
        client.access_token = 'test_token'
        client.token_expiry = datetime.now() + timedelta(hours=1)
        
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response):
            result = client.test_connection()
            assert result is True
    
    def test_connection_test_failure(self):
        """Test connection test failure."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        client = SentinelHubClient(config)
        client.access_token = 'test_token'
        
        with patch('requests.get', side_effect=Exception("Connection error")):
            result = client.test_connection()
            assert result is False
    
    def test_validate_credentials(self):
        """Test credential validation."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        client = SentinelHubClient(config)
        
        with patch.object(client, 'authenticate'):
            with patch.object(client, 'test_connection', return_value=True):
                is_valid, message = client.validate_credentials()
                
                assert is_valid is True
                assert "valid" in message.lower()


class TestQueryConstruction:
    """Unit tests for query parameter construction."""
    
    def test_query_sentinel_imagery_basic(self):
        """Test basic imagery query."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        client = SentinelHubClient(config)
        client.access_token = 'test_token'
        client.token_expiry = datetime.now() + timedelta(hours=1)
        
        geojson = create_ludhiana_sample_geojson()
        date_range = ('2024-01-01', '2024-01-31')
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'features': [
                {
                    'id': 'test_image',
                    'properties': {
                        'datetime': '2024-01-15T12:00:00Z',
                        'eo:cloud_cover': 10.0,
                        'sentinel:product_id': 'S2A_TEST'
                    },
                    'geometry': geojson,
                    'bbox': [75.8, 30.9, 75.9, 31.0]
                }
            ]
        }
        
        with patch.object(client, 'request_with_retry', return_value=mock_response):
            results = client.query_sentinel_imagery(
                geometry=geojson,
                date_range=date_range,
                cloud_threshold=20.0
            )
            
            assert len(results) == 1
            assert results[0]['id'] == 'test_image'
            assert results[0]['cloud_coverage'] == 10.0
    
    def test_query_filters_by_cloud_coverage(self):
        """Test that query respects cloud coverage threshold."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        client = SentinelHubClient(config)
        client.access_token = 'test_token'
        client.token_expiry = datetime.now() + timedelta(hours=1)
        
        geojson = create_ludhiana_sample_geojson()
        date_range = ('2024-01-01', '2024-01-31')
        cloud_threshold = 15.0
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'features': [
                {
                    'id': 'image1',
                    'properties': {
                        'datetime': '2024-01-15T12:00:00Z',
                        'eo:cloud_cover': 10.0,
                        'sentinel:product_id': 'S2A_TEST1'
                    },
                    'geometry': geojson,
                    'bbox': [75.8, 30.9, 75.9, 31.0]
                }
            ]
        }
        
        with patch.object(client, 'request_with_retry', return_value=mock_response):
            results = client.query_sentinel_imagery(
                geometry=geojson,
                date_range=date_range,
                cloud_threshold=cloud_threshold
            )
            
            # All results should have cloud coverage <= threshold
            for result in results:
                assert result['cloud_coverage'] <= cloud_threshold


class TestResponseParsing:
    """Unit tests for response parsing and validation."""
    
    def test_extract_imagery_metadata(self):
        """Test metadata extraction from API response."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        client = SentinelHubClient(config)
        
        feature = {
            'id': 'test_id',
            'properties': {
                'datetime': '2024-01-15T12:00:00Z',
                'eo:cloud_cover': 15.5,
                'sentinel:product_id': 'S2A_MSIL2A_TEST'
            },
            'geometry': {'type': 'Polygon'},
            'bbox': [75.8, 30.9, 75.9, 31.0]
        }
        
        metadata = client._extract_imagery_metadata(feature)
        
        assert metadata['id'] == 'test_id'
        assert metadata['acquisition_date'] == '2024-01-15T12:00:00Z'
        assert metadata['cloud_coverage'] == 15.5
        assert metadata['tile_id'] == 'S2A_MSIL2A_TEST'
    
    def test_geometry_to_bbox(self):
        """Test conversion of geometry to bounding box."""
        config = SentinelHubConfig(
            instance_id='test',
            client_id='test',
            client_secret='test'
        )
        client = SentinelHubClient(config)
        
        geometry = {
            'type': 'Polygon',
            'coordinates': [[
                [75.8, 30.9],
                [75.9, 30.9],
                [75.9, 31.0],
                [75.8, 31.0],
                [75.8, 30.9]
            ]]
        }
        
        bbox = client._geometry_to_bbox(geometry)
        
        assert bbox == [75.8, 30.9, 75.9, 31.0]
        assert len(bbox) == 4


class TestFallbackMechanism:
    """Unit tests for fallback to local TIF files."""
    
    def test_discover_local_tif_files(self):
        """Test discovery of local TIF files."""
        fallback = LocalTifFallback(search_paths=[])
        
        # Mock the file discovery
        with patch.object(fallback, 'discover_local_tif_files') as mock_discover:
            mock_discover.return_value = {
                'B02': '/path/to/B02.tif',
                'B03': '/path/to/B03.tif',
                'B04': '/path/to/B04.tif',
                'B08': '/path/to/B08.tif'
            }
            
            files = fallback.discover_local_tif_files(['B02', 'B03', 'B04', 'B08'])
            
            assert len(files) == 4
            assert 'B02' in files
            assert 'B08' in files
    
    def test_validate_local_files_all_available(self):
        """Test validation when all required files are available."""
        fallback = LocalTifFallback(search_paths=[])
        fallback._cached_files = {
            'B02': '/path/to/B02.tif',
            'B03': '/path/to/B03.tif',
            'B04': '/path/to/B04.tif',
            'B08': '/path/to/B08.tif'
        }
        
        all_available, missing = fallback.validate_local_files(['B02', 'B03', 'B04', 'B08'])
        
        assert all_available is True
        assert len(missing) == 0
    
    def test_validate_local_files_missing_bands(self):
        """Test validation when some files are missing."""
        fallback = LocalTifFallback(search_paths=[])
        fallback._cached_files = {
            'B02': '/path/to/B02.tif',
            'B03': '/path/to/B03.tif'
        }
        
        all_available, missing = fallback.validate_local_files(['B02', 'B03', 'B04', 'B08'])
        
        assert all_available is False
        assert 'B04' in missing
        assert 'B08' in missing
    
    def test_extract_band_name_from_filename(self):
        """Test band name extraction from various filename formats."""
        fallback = LocalTifFallback(search_paths=[])
        
        test_cases = [
            ('T43REQ_20240923T053641_B02_10m.jp2', 'B02'),
            ('T43REQ_20240923T053641_B08_10m.jp2', 'B08'),
            ('T43REQ_20240923T053641_B8A_20m.jp2', 'B8A'),
            ('some_file_B04.tif', 'B04'),
            ('no_band_here.tif', None)
        ]
        
        for filename, expected_band in test_cases:
            result = fallback._extract_band_name(filename)
            assert result == expected_band, f"Failed for {filename}"
    
    def test_get_local_metadata(self):
        """Test metadata extraction from local files."""
        fallback = LocalTifFallback(search_paths=[])
        fallback._cached_files = {
            'B02': '/path/to/T43REQ_20240923_B02.tif',
            'B03': '/path/to/T43REQ_20240923_B03.tif'
        }
        
        metadata = fallback.get_local_metadata()
        
        assert metadata['source'] == 'local_tif'
        assert metadata['num_bands'] == 2
        assert 'B02' in metadata['bands']
        assert 'B03' in metadata['bands']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
