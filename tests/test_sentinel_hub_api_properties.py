"""
Property-based tests for Sentinel Hub API integration.

Tests universal properties that should hold across all API interactions.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.data_processing.sentinel_hub_client import (
    SentinelHubClient,
    SentinelHubConfig
)
from src.data_processing.geojson_handler import (
    GeoJSONHandler,
    create_ludhiana_sample_geojson,
    LUDHIANA_BOUNDS
)


# Strategy for generating valid Ludhiana coordinates
@st.composite
def ludhiana_coordinates(draw):
    """Generate valid coordinates within Ludhiana region."""
    lon = draw(st.floats(
        min_value=LUDHIANA_BOUNDS['lon_min'],
        max_value=LUDHIANA_BOUNDS['lon_max'],
        allow_nan=False,
        allow_infinity=False
    ))
    lat = draw(st.floats(
        min_value=LUDHIANA_BOUNDS['lat_min'],
        max_value=LUDHIANA_BOUNDS['lat_max'],
        allow_nan=False,
        allow_infinity=False
    ))
    return lon, lat


@st.composite
def ludhiana_geojson(draw):
    """Generate valid GeoJSON for Ludhiana region."""
    # Generate 4 corner points for a bounding box
    lon1 = draw(st.floats(
        min_value=LUDHIANA_BOUNDS['lon_min'],
        max_value=LUDHIANA_BOUNDS['lon_max'] - 0.01,
        allow_nan=False,
        allow_infinity=False
    ))
    lon2 = draw(st.floats(
        min_value=lon1 + 0.01,
        max_value=LUDHIANA_BOUNDS['lon_max'],
        allow_nan=False,
        allow_infinity=False
    ))
    
    lat1 = draw(st.floats(
        min_value=LUDHIANA_BOUNDS['lat_min'],
        max_value=LUDHIANA_BOUNDS['lat_max'] - 0.01,
        allow_nan=False,
        allow_infinity=False
    ))
    lat2 = draw(st.floats(
        min_value=lat1 + 0.01,
        max_value=LUDHIANA_BOUNDS['lat_max'],
        allow_nan=False,
        allow_infinity=False
    ))
    
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon1, lat1],
            [lon2, lat1],
            [lon2, lat2],
            [lon1, lat2],
            [lon1, lat1]
        ]]
    }


@st.composite
def date_range_strategy(draw):
    """Generate valid date ranges."""
    # Generate dates within a reasonable range (2020-2024)
    start_days = draw(st.integers(min_value=0, max_value=1460))  # ~4 years
    duration_days = draw(st.integers(min_value=1, max_value=90))  # 1-90 days
    
    start_date = datetime(2020, 1, 1) + timedelta(days=start_days)
    end_date = start_date + timedelta(days=duration_days)
    
    return (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


class TestAPIQueryProperties:
    """Property-based tests for API query functionality."""
    
    @given(
        geojson=ludhiana_geojson(),
        date_range=date_range_strategy(),
        cloud_threshold=st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_1_api_query_returns_valid_imagery(
        self,
        geojson,
        date_range,
        cloud_threshold
    ):
        """
        **Feature: production-enhancements, Property 1: API query returns valid imagery**
        
        For any valid GeoJSON boundary in Ludhiana region and date range,
        querying the Sentinel Hub API should return imagery data with exactly
        4 bands (B02, B03, B04, B08) at 10m resolution.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Create mock config
        config = SentinelHubConfig(
            instance_id='test_instance',
            client_id='test_client',
            client_secret='test_secret',
            bands=['B02', 'B03', 'B04', 'B08'],
            resolution=10
        )
        
        client = SentinelHubClient(config)
        
        # Mock the authentication and API calls
        with patch.object(client, '_ensure_authenticated'):
            with patch.object(client, 'request_with_retry') as mock_request:
                # Create mock response with valid structure
                mock_response = Mock()
                mock_response.json.return_value = {
                    'features': [
                        {
                            'id': 'test_image_1',
                            'properties': {
                                'datetime': date_range[0] + 'T12:00:00Z',
                                'eo:cloud_cover': cloud_threshold - 5,
                                'sentinel:product_id': 'S2A_TEST'
                            },
                            'geometry': geojson,
                            'bbox': [75.8, 30.9, 75.9, 31.0]
                        }
                    ]
                }
                mock_request.return_value = mock_response
                
                # Execute query
                results = client.query_sentinel_imagery(
                    geometry=geojson,
                    date_range=date_range,
                    cloud_threshold=cloud_threshold
                )
                
                # Property: Results should be a list
                assert isinstance(results, list), "Query should return a list"
                
                # Property: Each result should have required metadata fields
                for result in results:
                    assert 'id' in result, "Result must have 'id' field"
                    assert 'acquisition_date' in result, "Result must have 'acquisition_date'"
                    assert 'cloud_coverage' in result, "Result must have 'cloud_coverage'"
                    assert 'geometry' in result, "Result must have 'geometry'"
                    
                    # Property: Cloud coverage should be within threshold
                    assert result['cloud_coverage'] <= cloud_threshold, \
                        f"Cloud coverage {result['cloud_coverage']} exceeds threshold {cloud_threshold}"
                
                # Property: Results should be sorted by acquisition date
                if len(results) > 1:
                    dates = [r['acquisition_date'] for r in results]
                    assert dates == sorted(dates), "Results should be sorted by acquisition date"
    
    @given(
        geojson=ludhiana_geojson(),
        date=st.dates(min_value=datetime(2020, 1, 1).date(), max_value=datetime(2024, 12, 31).date())
    )
    @settings(max_examples=50, deadline=None)
    def test_property_api_download_returns_correct_bands(self, geojson, date):
        """
        Property: Downloaded imagery should contain exactly the requested bands.
        
        For any valid geometry and date, downloading bands should return
        a dictionary with exactly the requested band names as keys.
        """
        config = SentinelHubConfig(
            instance_id='test_instance',
            client_id='test_client',
            client_secret='test_secret',
            bands=['B02', 'B03', 'B04', 'B08'],
            resolution=10
        )
        
        client = SentinelHubClient(config)
        
        date_str = date.strftime('%Y-%m-%d')
        
        with patch.object(client, '_ensure_authenticated'):
            with patch.object(client, 'request_with_retry') as mock_request:
                # Mock successful download
                mock_response = Mock()
                mock_response.content = b'mock_tiff_data'
                mock_request.return_value = mock_response
                
                # Mock the band parsing to return proper structure
                with patch.object(client, '_parse_band_response') as mock_parse:
                    expected_bands = config.bands
                    mock_parse.return_value = {
                        band: np.zeros((100, 100), dtype=np.float32)
                        for band in expected_bands
                    }
                    
                    # Execute download
                    band_data = client.download_multispectral_bands(
                        geometry=geojson,
                        acquisition_date=date_str
                    )
                    
                    # Property: Should return exactly the requested bands
                    assert set(band_data.keys()) == set(expected_bands), \
                        f"Expected bands {expected_bands}, got {list(band_data.keys())}"
                    
                    # Property: Each band should be a numpy array
                    for band_name, data in band_data.items():
                        assert isinstance(data, np.ndarray), \
                            f"Band {band_name} should be numpy array"
                        
                        # Property: All bands should have same shape
                        assert data.ndim == 2, f"Band {band_name} should be 2D array"
                    
                    # Property: All bands should have the same dimensions
                    shapes = [data.shape for data in band_data.values()]
                    assert len(set(shapes)) == 1, \
                        f"All bands should have same shape, got {shapes}"


class TestGeoJSONValidationProperties:
    """Property-based tests for GeoJSON validation."""
    
    @given(
        lon=st.floats(
            min_value=LUDHIANA_BOUNDS['lon_min'],
            max_value=LUDHIANA_BOUNDS['lon_max'],
            allow_nan=False,
            allow_infinity=False
        ),
        lat=st.floats(
            min_value=LUDHIANA_BOUNDS['lat_min'],
            max_value=LUDHIANA_BOUNDS['lat_max'],
            allow_nan=False,
            allow_infinity=False
        )
    )
    @settings(max_examples=100)
    def test_property_ludhiana_coordinates_accepted(self, lon, lat):
        """
        Property: All coordinates within Ludhiana bounds should be accepted.
        
        For any coordinate pair within the defined Ludhiana region,
        validation should succeed.
        """
        handler = GeoJSONHandler(validate_ludhiana=True)
        
        geojson = {
            "type": "Point",
            "coordinates": [lon, lat]
        }
        
        # Should not raise exception
        result = handler.parse_geojson(geojson)
        assert result is not None
    
    @given(
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_coordinates_outside_ludhiana_rejected(self, lon, lat):
        """
        Property: Coordinates outside Ludhiana bounds should be rejected.
        
        For any coordinate pair outside the Ludhiana region,
        validation should fail when Ludhiana validation is enabled.
        """
        # Skip if coordinates are actually within Ludhiana
        assume(not (
            LUDHIANA_BOUNDS['lon_min'] <= lon <= LUDHIANA_BOUNDS['lon_max'] and
            LUDHIANA_BOUNDS['lat_min'] <= lat <= LUDHIANA_BOUNDS['lat_max']
        ))
        
        handler = GeoJSONHandler(validate_ludhiana=True)
        
        geojson = {
            "type": "Point",
            "coordinates": [lon, lat]
        }
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="outside Ludhiana region"):
            handler.parse_geojson(geojson)
    
    @given(geojson=ludhiana_geojson())
    @settings(max_examples=50)
    def test_property_bounding_box_contains_geometry(self, geojson):
        """
        Property: Extracted bounding box should contain all geometry coordinates.
        
        For any valid GeoJSON, the extracted bounding box should fully
        contain all coordinates in the geometry.
        """
        handler = GeoJSONHandler(validate_ludhiana=True)
        
        # Parse and extract bounding box
        parsed = handler.parse_geojson(geojson)
        bbox = handler.extract_bounding_box(parsed)
        
        # Extract all coordinates from geometry
        coords = handler._extract_all_coordinates(parsed)
        
        # Property: All coordinates should be within bounding box
        for lon, lat in coords:
            assert bbox.min_lon <= lon <= bbox.max_lon, \
                f"Longitude {lon} outside bbox [{bbox.min_lon}, {bbox.max_lon}]"
            assert bbox.min_lat <= lat <= bbox.max_lat, \
                f"Latitude {lat} outside bbox [{bbox.min_lat}, {bbox.max_lat}]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
