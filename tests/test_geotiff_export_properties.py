"""
Property-based tests for GeoTIFF export functionality.
Tests universal properties that should hold across all valid inputs.

Feature: production-enhancements
"""

import pytest
import numpy as np
import tempfile
import os
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


# Strategy for generating valid vegetation index values
# NDVI range: [-1, 1], but we'll use a slightly narrower range for realistic values
vegetation_index_strategy = st.floats(min_value=-0.95, max_value=0.95, allow_nan=False, allow_infinity=False)

# Strategy for generating array shapes (keep small for performance)
shape_strategy = st.tuples(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=10, max_value=50)
)

# Strategy for generating geographic bounds (Ludhiana region)
# Ludhiana is approximately at 30.9-31.0°N, 75.8-75.9°E
# In UTM Zone 43N, this is roughly 500000-510000 E, 3420000-3430000 N
bounds_strategy = st.tuples(
    st.floats(min_value=500000, max_value=505000),  # minx
    st.floats(min_value=3420000, max_value=3425000),  # miny
    st.floats(min_value=505000, max_value=510000),  # maxx
    st.floats(min_value=3425000, max_value=3430000)  # maxy
)


def export_geotiff_with_georef(data: np.ndarray, bounds: tuple, crs: str, output_path: str):
    """
    Export vegetation index array as GeoTIFF with proper georeferencing.
    
    This is the function being tested - it should preserve CRS and transform exactly.
    """
    height, width = data.shape
    minx, miny, maxx, maxy = bounds
    
    # Ensure bounds are valid
    if maxx <= minx or maxy <= miny:
        raise ValueError("Invalid bounds: max must be greater than min")
    
    # Create transform from bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)
        
        # Add metadata tags
        dst.update_tags(
            index_name='NDVI',
            export_date='2024-01-01T00:00:00',
            source='AgriFlux'
        )
    
    return output_path


class TestGeoTIFFRoundTripProperties:
    """Property-based tests for GeoTIFF export and round-trip preservation.
    
    **Feature: production-enhancements, Property 18: GeoTIFF round-trip preservation**
    **Validates: Requirements 5.1**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=shape_strategy,
        bounds=bounds_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_geotiff_roundtrip_crs_preservation(self, shape, bounds, seed):
        """
        Property 18: GeoTIFF round-trip preservation
        
        For any vegetation index array with georeference information, 
        exporting to GeoTIFF and reading back should preserve the CRS exactly.
        """
        np.random.seed(seed)
        
        # Validate bounds
        minx, miny, maxx, maxy = bounds
        assume(maxx > minx + 100)  # Ensure reasonable width
        assume(maxy > miny + 100)  # Ensure reasonable height
        
        # Generate vegetation index data (NDVI-like values)
        data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        # Define CRS (UTM Zone 43N for Ludhiana region)
        original_crs = CRS.from_epsg(32643)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export to GeoTIFF
            export_geotiff_with_georef(data, bounds, original_crs, temp_path)
            
            # Read back from GeoTIFF
            with rasterio.open(temp_path) as src:
                read_crs = src.crs
                read_data = src.read(1)
            
            # Verify CRS preservation
            assert read_crs == original_crs, \
                f"CRS not preserved: expected {original_crs}, got {read_crs}"
            
            # Verify data shape preservation
            assert read_data.shape == data.shape, \
                f"Shape not preserved: expected {data.shape}, got {read_data.shape}"
            
            # Verify data values preservation (within floating point tolerance)
            np.testing.assert_allclose(
                read_data,
                data,
                rtol=1e-5,
                atol=1e-7,
                err_msg="Data values not preserved in round-trip"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=shape_strategy,
        bounds=bounds_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_geotiff_roundtrip_transform_preservation(self, shape, bounds, seed):
        """
        Property 18: GeoTIFF round-trip preservation (Transform)
        
        For any vegetation index array with georeference information, 
        exporting to GeoTIFF and reading back should preserve the transform exactly.
        """
        np.random.seed(seed)
        
        # Validate bounds
        minx, miny, maxx, maxy = bounds
        assume(maxx > minx + 100)  # Ensure reasonable width
        assume(maxy > miny + 100)  # Ensure reasonable height
        
        # Generate vegetation index data
        data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        # Define CRS
        crs = CRS.from_epsg(32643)
        
        # Calculate expected transform
        height, width = shape
        expected_transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export to GeoTIFF
            export_geotiff_with_georef(data, bounds, crs, temp_path)
            
            # Read back from GeoTIFF
            with rasterio.open(temp_path) as src:
                read_transform = src.transform
            
            # Verify transform preservation (all 6 affine parameters)
            assert read_transform.a == pytest.approx(expected_transform.a, rel=1e-9), \
                f"Transform.a not preserved: expected {expected_transform.a}, got {read_transform.a}"
            assert read_transform.b == pytest.approx(expected_transform.b, rel=1e-9), \
                f"Transform.b not preserved: expected {expected_transform.b}, got {read_transform.b}"
            assert read_transform.c == pytest.approx(expected_transform.c, rel=1e-9), \
                f"Transform.c not preserved: expected {expected_transform.c}, got {read_transform.c}"
            assert read_transform.d == pytest.approx(expected_transform.d, rel=1e-9), \
                f"Transform.d not preserved: expected {expected_transform.d}, got {read_transform.d}"
            assert read_transform.e == pytest.approx(expected_transform.e, rel=1e-9), \
                f"Transform.e not preserved: expected {expected_transform.e}, got {read_transform.e}"
            assert read_transform.f == pytest.approx(expected_transform.f, rel=1e-9), \
                f"Transform.f not preserved: expected {expected_transform.f}, got {read_transform.f}"
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=shape_strategy,
        bounds=bounds_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_geotiff_roundtrip_bounds_preservation(self, shape, bounds, seed):
        """
        Property 18: GeoTIFF round-trip preservation (Bounds)
        
        For any vegetation index array with georeference information, 
        exporting to GeoTIFF and reading back should preserve the bounds exactly.
        """
        np.random.seed(seed)
        
        # Validate bounds
        minx, miny, maxx, maxy = bounds
        assume(maxx > minx + 100)  # Ensure reasonable width
        assume(maxy > miny + 100)  # Ensure reasonable height
        
        # Generate vegetation index data
        data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        # Define CRS
        crs = CRS.from_epsg(32643)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export to GeoTIFF
            export_geotiff_with_georef(data, bounds, crs, temp_path)
            
            # Read back from GeoTIFF
            with rasterio.open(temp_path) as src:
                read_bounds = src.bounds
            
            # Verify bounds preservation
            assert read_bounds.left == pytest.approx(minx, rel=1e-9), \
                f"Bounds.left not preserved: expected {minx}, got {read_bounds.left}"
            assert read_bounds.bottom == pytest.approx(miny, rel=1e-9), \
                f"Bounds.bottom not preserved: expected {miny}, got {read_bounds.bottom}"
            assert read_bounds.right == pytest.approx(maxx, rel=1e-9), \
                f"Bounds.right not preserved: expected {maxx}, got {read_bounds.right}"
            assert read_bounds.top == pytest.approx(maxy, rel=1e-9), \
                f"Bounds.top not preserved: expected {maxy}, got {read_bounds.top}"
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @settings(max_examples=50, deadline=None)
    @given(
        shape=shape_strategy,
        bounds=bounds_strategy,
        seed=st.integers(min_value=0, max_value=10000),
        epsg_code=st.sampled_from([32643, 4326, 32644])  # UTM 43N, WGS84, UTM 44N
    )
    def test_geotiff_roundtrip_multiple_crs(self, shape, bounds, seed, epsg_code):
        """
        Property: GeoTIFF round-trip should preserve CRS for different coordinate systems.
        """
        np.random.seed(seed)
        
        # Validate bounds
        minx, miny, maxx, maxy = bounds
        assume(maxx > minx + 100)
        assume(maxy > miny + 100)
        
        # Adjust bounds for WGS84 (lat/lon)
        if epsg_code == 4326:
            minx, miny, maxx, maxy = 75.8, 30.9, 75.9, 31.0
        
        # Generate vegetation index data
        data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        # Define CRS
        original_crs = CRS.from_epsg(epsg_code)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export to GeoTIFF
            export_geotiff_with_georef(data, (minx, miny, maxx, maxy), original_crs, temp_path)
            
            # Read back from GeoTIFF
            with rasterio.open(temp_path) as src:
                read_crs = src.crs
            
            # Verify CRS preservation
            assert read_crs == original_crs, \
                f"CRS not preserved for EPSG:{epsg_code}: expected {original_crs}, got {read_crs}"
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @settings(max_examples=50, deadline=None)
    @given(
        shape=shape_strategy,
        bounds=bounds_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_geotiff_metadata_preservation(self, shape, bounds, seed):
        """
        Property: GeoTIFF export should preserve metadata tags.
        """
        np.random.seed(seed)
        
        # Validate bounds
        minx, miny, maxx, maxy = bounds
        assume(maxx > minx + 100)
        assume(maxy > miny + 100)
        
        # Generate vegetation index data
        data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        # Define CRS
        crs = CRS.from_epsg(32643)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export to GeoTIFF
            export_geotiff_with_georef(data, bounds, crs, temp_path)
            
            # Read back metadata
            with rasterio.open(temp_path) as src:
                tags = src.tags()
            
            # Verify metadata tags exist
            assert 'index_name' in tags, "Metadata tag 'index_name' not preserved"
            assert tags['index_name'] == 'NDVI', \
                f"Metadata value incorrect: expected 'NDVI', got '{tags['index_name']}'"
            assert 'export_date' in tags, "Metadata tag 'export_date' not preserved"
            assert 'source' in tags, "Metadata tag 'source' not preserved"
            assert tags['source'] == 'AgriFlux', \
                f"Metadata value incorrect: expected 'AgriFlux', got '{tags['source']}'"
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @settings(max_examples=50, deadline=None)
    @given(
        shape=shape_strategy,
        bounds=bounds_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_geotiff_compression_integrity(self, shape, bounds, seed):
        """
        Property: GeoTIFF compression should not affect data integrity.
        """
        np.random.seed(seed)
        
        # Validate bounds
        minx, miny, maxx, maxy = bounds
        assume(maxx > minx + 100)
        assume(maxy > miny + 100)
        
        # Generate vegetation index data with specific patterns
        data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        # Define CRS
        crs = CRS.from_epsg(32643)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export to GeoTIFF (with LZW compression)
            export_geotiff_with_georef(data, bounds, crs, temp_path)
            
            # Read back and verify compression is applied
            with rasterio.open(temp_path) as src:
                read_data = src.read(1)
                compression = src.compression
            
            # Verify compression is set
            assert compression is not None, "Compression not applied"
            
            # Verify data integrity despite compression
            np.testing.assert_allclose(
                read_data,
                data,
                rtol=1e-5,
                atol=1e-7,
                err_msg="Compression affected data integrity"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @settings(max_examples=50, deadline=None)
    @given(
        shape=shape_strategy,
        bounds=bounds_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_geotiff_pixel_to_coordinate_mapping(self, shape, bounds, seed):
        """
        Property: Pixel coordinates should map correctly to geographic coordinates.
        """
        np.random.seed(seed)
        
        # Validate bounds
        minx, miny, maxx, maxy = bounds
        assume(maxx > minx + 100)
        assume(maxy > miny + 100)
        
        # Generate vegetation index data
        data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        # Define CRS
        crs = CRS.from_epsg(32643)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export to GeoTIFF
            export_geotiff_with_georef(data, bounds, crs, temp_path)
            
            # Read back and test pixel-to-coordinate mapping
            with rasterio.open(temp_path) as src:
                transform = src.transform
                
                # Test corner pixels
                # Top-left pixel (0, 0) should map near minx, maxy
                x0, y0 = transform * (0, 0)
                assert abs(x0 - minx) < (maxx - minx) / shape[1], \
                    f"Top-left X coordinate incorrect: {x0} vs {minx}"
                assert abs(y0 - maxy) < (maxy - miny) / shape[0], \
                    f"Top-left Y coordinate incorrect: {y0} vs {maxy}"
                
                # Bottom-right pixel should map near maxx, miny
                x1, y1 = transform * (shape[1], shape[0])
                assert abs(x1 - maxx) < (maxx - minx) / shape[1], \
                    f"Bottom-right X coordinate incorrect: {x1} vs {maxx}"
                assert abs(y1 - miny) < (maxy - miny) / shape[0], \
                    f"Bottom-right Y coordinate incorrect: {y1} vs {miny}"
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
