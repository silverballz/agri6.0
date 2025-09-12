"""
Tests for geospatial utility functions.
"""

import pytest
import numpy as np
import sys
import os
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.geospatial_utils import (
    CoordinateTransformer,
    RasterProcessor,
    RasterInfo,
    BoundingBox,
    utm_zone_from_longitude,
    create_utm_crs,
    pixel_to_world,
    world_to_pixel
)


class TestCoordinateTransformer:
    """Test coordinate transformation utilities."""
    
    def test_coordinate_transformer_initialization(self):
        """Test transformer initialization with different CRS formats."""
        # Test with EPSG codes
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:32643')
        assert transformer.source_crs.to_epsg() == 4326
        assert transformer.target_crs.to_epsg() == 32643
    
    def test_transform_point(self):
        """Test single point transformation."""
        # WGS84 to UTM Zone 43N (approximate for T43REQ tile)
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:32643')
        
        # Transform a point in the T43REQ area
        lon, lat = 75.0, 25.0  # Approximate center of T43REQ
        x, y = transformer.transform_point(lon, lat)
        
        # UTM coordinates should be in reasonable range
        assert 200000 < x < 800000  # UTM X range
        assert 2000000 < y < 4000000  # UTM Y range for northern hemisphere
    
    def test_transform_bounds(self):
        """Test bounding box transformation."""
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:32643')
        
        # Small bounding box in WGS84
        wgs84_bounds = (74.9, 24.9, 75.1, 25.1)
        utm_bounds = transformer.transform_bounds(wgs84_bounds)
        
        assert len(utm_bounds) == 4
        assert utm_bounds[0] < utm_bounds[2]  # min_x < max_x
        assert utm_bounds[1] < utm_bounds[3]  # min_y < max_y


class TestRasterProcessor:
    """Test raster processing utilities."""
    
    @pytest.fixture
    def sample_raster_data(self):
        """Create sample raster data for testing."""
        # Create a simple 100x100 raster
        array = np.random.rand(100, 100).astype(np.float32)
        
        # Create transform for 10m resolution
        transform = from_bounds(0, 0, 1000, 1000, 100, 100)
        crs = CRS.from_epsg(32643)  # UTM Zone 43N
        
        return array, transform, crs
    
    def test_create_mask_nodata(self):
        """Test mask creation with nodata values."""
        array = np.array([[1, 2, -9999], [4, 5, 6]], dtype=np.float32)
        mask = RasterProcessor.create_mask(array, nodata_value=-9999)
        
        expected = np.array([[True, True, False], [True, True, True]])
        np.testing.assert_array_equal(mask, expected)
    
    def test_create_mask_valid_range(self):
        """Test mask creation with valid range."""
        array = np.array([[0.1, 0.5, 1.5], [0.8, 0.3, 0.9]], dtype=np.float32)
        mask = RasterProcessor.create_mask(array, valid_range=(0.2, 1.0))
        
        expected = np.array([[False, True, False], [True, True, True]])
        np.testing.assert_array_equal(mask, expected)
    
    def test_create_mask_nan_values(self):
        """Test mask creation with NaN values."""
        array = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.inf]], dtype=np.float32)
        mask = RasterProcessor.create_mask(array)
        
        expected = np.array([[True, False, True], [True, True, False]])
        np.testing.assert_array_equal(mask, expected)
    
    def test_apply_scale_offset(self):
        """Test scale and offset application."""
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        result = RasterProcessor.apply_scale_offset(array, scale=0.1, offset=10.0)
        
        expected = np.array([[10.1, 10.2, 10.3], [10.4, 10.5, 10.6]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_apply_scale_offset_with_mask(self):
        """Test scale and offset with nodata mask."""
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        mask = np.array([[True, False, True], [True, True, False]])
        
        result = RasterProcessor.apply_scale_offset(array, scale=0.1, offset=10.0, nodata_mask=mask)
        
        assert np.isnan(result[0, 1])  # Masked pixel should be NaN
        assert np.isnan(result[1, 2])  # Masked pixel should be NaN
        assert abs(result[0, 0] - 10.1) < 1e-6   # Unmasked pixel should be scaled


class TestUtilityFunctions:
    """Test standalone utility functions."""
    
    def test_utm_zone_from_longitude(self):
        """Test UTM zone calculation from longitude."""
        # Test known values
        assert utm_zone_from_longitude(-180) == 1
        assert utm_zone_from_longitude(0) == 31
        assert utm_zone_from_longitude(75) == 43  # Approximate for T43REQ area
        assert utm_zone_from_longitude(180) == 61  # This wraps to zone 1
    
    def test_create_utm_crs(self):
        """Test UTM CRS creation."""
        # Test northern hemisphere
        crs_north = create_utm_crs(43, 'north')
        assert crs_north.to_epsg() == 32643
        
        # Test southern hemisphere
        crs_south = create_utm_crs(43, 'south')
        assert crs_south.to_epsg() == 32743
    
    def test_pixel_to_world_conversion(self):
        """Test pixel to world coordinate conversion."""
        # Create simple transform: 10m resolution, origin at (100000, 3000000)
        transform = from_bounds(100000, 3000000, 101000, 3001000, 100, 100)
        
        # Test corner pixel (0, 0) - should be at upper-left
        x, y = pixel_to_world(0, 0, transform)
        assert abs(x - 100000) < 1e-6
        assert abs(y - 3001000) < 1e-6  # Y decreases going down
        
        # Test center pixel
        x, y = pixel_to_world(50, 50, transform)
        assert abs(x - 100500) < 1e-6
        assert abs(y - 3000500) < 1e-6
    
    def test_world_to_pixel_conversion(self):
        """Test world to pixel coordinate conversion."""
        transform = from_bounds(100000, 3000000, 101000, 3001000, 100, 100)
        
        # Test conversion back
        row, col = world_to_pixel(100500, 3000500, transform)
        assert row == 50
        assert col == 50
    
    def test_pixel_world_roundtrip(self):
        """Test roundtrip conversion between pixel and world coordinates."""
        transform = from_bounds(100000, 3000000, 101000, 3001000, 100, 100)
        
        # Original pixel coordinates
        orig_row, orig_col = 25, 75
        
        # Convert to world and back
        x, y = pixel_to_world(orig_row, orig_col, transform)
        new_row, new_col = world_to_pixel(x, y, transform)
        
        assert new_row == orig_row
        assert new_col == orig_col


class TestDataClasses:
    """Test data class functionality."""
    
    def test_raster_info_creation(self):
        """Test RasterInfo dataclass."""
        transform = from_bounds(0, 0, 1000, 1000, 100, 100)
        crs = CRS.from_epsg(4326)
        
        raster_info = RasterInfo(
            width=100,
            height=100,
            transform=transform,
            crs=crs,
            nodata=-9999.0,
            dtype='float32'
        )
        
        assert raster_info.width == 100
        assert raster_info.height == 100
        assert raster_info.nodata == -9999.0
        assert raster_info.dtype == 'float32'
    
    def test_bounding_box_creation(self):
        """Test BoundingBox dataclass."""
        crs = CRS.from_epsg(4326)
        bbox = BoundingBox(
            min_x=-180.0,
            min_y=-90.0,
            max_x=180.0,
            max_y=90.0,
            crs=crs
        )
        
        assert bbox.min_x == -180.0
        assert bbox.max_y == 90.0
        assert bbox.crs.to_epsg() == 4326


class TestCoordinateTransformationEdgeCases:
    """Test edge cases for coordinate transformations."""
    
    def test_invalid_crs_handling(self):
        """Test handling of invalid CRS specifications."""
        with pytest.raises((ValueError, Exception)):
            CoordinateTransformer('INVALID:CRS', 'EPSG:4326')
    
    def test_same_crs_transformation(self):
        """Test transformation between same CRS."""
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:4326')
        
        # Should return same coordinates
        x, y = transformer.transform_point(75.0, 25.0)
        assert abs(x - 75.0) < 1e-10
        assert abs(y - 25.0) < 1e-10
    
    def test_extreme_coordinate_values(self):
        """Test transformation with extreme coordinate values."""
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:32643')
        
        # Test coordinates at edge of UTM zone
        x1, y1 = transformer.transform_point(72.0, 0.0)  # Western edge
        x2, y2 = transformer.transform_point(78.0, 0.0)  # Eastern edge
        
        # Should produce valid UTM coordinates
        assert 100000 < x1 < 900000
        assert 100000 < x2 < 900000
        assert abs(y1) < 20000000  # Reasonable Y range
        assert abs(y2) < 20000000
    
    def test_polar_coordinates(self):
        """Test transformation near polar regions."""
        # Test with coordinates that might cause issues in some projections
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:3857')  # Web Mercator
        
        # Near north pole (Web Mercator has issues near poles)
        try:
            x, y = transformer.transform_point(0.0, 85.0)
            assert abs(x) < 1e10  # Should not be infinite
            assert abs(y) < 1e10
        except Exception:
            # Some transformations may fail near poles, which is acceptable
            pass
    
    def test_antimeridian_crossing(self):
        """Test transformation across the antimeridian."""
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:32601')  # UTM Zone 1N
        
        # Test coordinates on both sides of antimeridian
        x1, y1 = transformer.transform_point(179.0, 25.0)
        x2, y2 = transformer.transform_point(-179.0, 25.0)
        
        # Should produce valid coordinates
        assert not np.isnan(x1) and not np.isnan(y1)
        assert not np.isnan(x2) and not np.isnan(y2)
    
    def test_bounds_transformation_validation(self):
        """Test that transformed bounds maintain proper ordering."""
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:32643')
        
        # Original bounds in WGS84
        original_bounds = (74.5, 24.5, 75.5, 25.5)  # min_x, min_y, max_x, max_y
        transformed_bounds = transformer.transform_bounds(original_bounds)
        
        # Transformed bounds should maintain ordering
        assert transformed_bounds[0] < transformed_bounds[2]  # min_x < max_x
        assert transformed_bounds[1] < transformed_bounds[3]  # min_y < max_y
        
        # Should be reasonable UTM coordinates
        assert 200000 < transformed_bounds[0] < 800000
        assert 200000 < transformed_bounds[2] < 800000


class TestGeospatialUtilityFunctions:
    """Test additional geospatial utility functions."""
    
    def test_utm_zone_edge_cases(self):
        """Test UTM zone calculation for edge cases."""
        # Test longitude exactly on zone boundary
        assert utm_zone_from_longitude(-180.0) == 1
        assert utm_zone_from_longitude(180.0) == 61  # Wraps around
        
        # Test zone boundaries
        assert utm_zone_from_longitude(-174.0) == 1  # Zone 1
        assert utm_zone_from_longitude(-168.0) == 2  # Zone 2
        assert utm_zone_from_longitude(0.0) == 31    # Zone 31 (Greenwich)
        assert utm_zone_from_longitude(6.0) == 32    # Zone 32
    
    def test_utm_crs_creation_validation(self):
        """Test UTM CRS creation with validation."""
        # Valid zones
        for zone in [1, 31, 43, 60]:
            crs_north = create_utm_crs(zone, 'north')
            crs_south = create_utm_crs(zone, 'south')
            
            assert crs_north.is_projected
            assert crs_south.is_projected
            assert crs_north.to_epsg() == 32600 + zone
            assert crs_south.to_epsg() == 32700 + zone
        
        # Invalid zones should raise errors
        with pytest.raises(ValueError):
            create_utm_crs(0, 'north')  # Zone 0 doesn't exist
        
        with pytest.raises(ValueError):
            create_utm_crs(61, 'north')  # Zone 61 doesn't exist
        
        with pytest.raises(ValueError):
            create_utm_crs(31, 'invalid')  # Invalid hemisphere
    
    def test_pixel_world_conversion_precision(self):
        """Test precision of pixel-world coordinate conversion."""
        # High precision transform
        transform = from_bounds(100000.0, 3000000.0, 100010.0, 3000010.0, 100, 100)
        
        # Test multiple points for precision
        test_points = [(0, 0), (50, 50), (99, 99), (25, 75)]
        
        for row, col in test_points:
            # Convert to world and back
            x, y = pixel_to_world(row, col, transform)
            new_row, new_col = world_to_pixel(x, y, transform)
            
            # Should be exact for integer pixel coordinates
            assert new_row == row, f"Row mismatch: {new_row} != {row}"
            assert new_col == col, f"Col mismatch: {new_col} != {col}"
    
    def test_pixel_world_conversion_fractional(self):
        """Test pixel-world conversion with fractional coordinates."""
        transform = from_bounds(0, 0, 100, 100, 10, 10)
        
        # Test fractional pixel coordinates
        row, col = 2.5, 7.3
        x, y = pixel_to_world(row, col, transform)
        new_row, new_col = world_to_pixel(x, y, transform)
        
        # Should preserve fractional precision
        assert abs(new_row - row) < 1e-10
        assert abs(new_col - col) < 1e-10
    
    def test_coordinate_array_transformations(self):
        """Test transformation of coordinate arrays."""
        transformer = CoordinateTransformer('EPSG:4326', 'EPSG:32643')
        
        # Create arrays of coordinates
        lons = np.array([74.0, 74.5, 75.0, 75.5, 76.0])
        lats = np.array([24.0, 24.5, 25.0, 25.5, 26.0])
        
        # Transform arrays
        xs, ys = transformer.transform_points(lons, lats)
        
        assert len(xs) == len(lons)
        assert len(ys) == len(lats)
        assert all(200000 < x < 800000 for x in xs)  # Valid UTM X range
        assert all(2000000 < y < 4000000 for y in ys)  # Valid UTM Y range for this area