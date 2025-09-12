"""
Utility functions for coordinate transformations and raster operations.
Provides common geospatial operations needed for satellite imagery processing.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from pyproj import Transformer
import warnings


@dataclass
class RasterInfo:
    """Container for raster metadata and properties."""
    width: int
    height: int
    transform: rasterio.Affine
    crs: CRS
    nodata: Optional[float] = None
    dtype: str = 'float32'


@dataclass
class BoundingBox:
    """Geographic bounding box in specified coordinate system."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: CRS


class CoordinateTransformer:
    """Handles coordinate transformations between different CRS."""
    
    def __init__(self, source_crs: Union[str, CRS], target_crs: Union[str, CRS]):
        """
        Initialize coordinate transformer.
        
        Args:
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
        """
        self.source_crs = CRS.from_user_input(source_crs)
        self.target_crs = CRS.from_user_input(target_crs)
        self.transformer = Transformer.from_crs(
            self.source_crs, 
            self.target_crs, 
            always_xy=True
        )
    
    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform a single point from source to target CRS.
        
        Args:
            x: X coordinate in source CRS
            y: Y coordinate in source CRS
        
        Returns:
            Tuple of (x, y) in target CRS
        """
        return self.transformer.transform(x, y)
    
    def transform_bounds(self, bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Transform bounding box from source to target CRS.
        
        Args:
            bounds: Tuple of (min_x, min_y, max_x, max_y) in source CRS
        
        Returns:
            Transformed bounds in target CRS
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Transform corner points
        corners = [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y)
        ]
        
        transformed_corners = [self.transform_point(x, y) for x, y in corners]
        
        # Find new bounds
        x_coords = [corner[0] for corner in transformed_corners]
        y_coords = [corner[1] for corner in transformed_corners]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


class RasterProcessor:
    """Utilities for raster data processing and manipulation."""
    
    @staticmethod
    def resample_raster(
        source_array: np.ndarray,
        source_transform: rasterio.Affine,
        source_crs: CRS,
        target_resolution: float,
        target_crs: Optional[CRS] = None,
        resampling_method: Resampling = Resampling.bilinear
    ) -> Tuple[np.ndarray, rasterio.Affine, CRS]:
        """
        Resample raster to target resolution and optionally reproject.
        
        Args:
            source_array: Input raster array
            source_transform: Affine transform of source raster
            source_crs: CRS of source raster
            target_resolution: Target pixel resolution in target CRS units
            target_crs: Target CRS (if None, uses source CRS)
            resampling_method: Resampling algorithm to use
        
        Returns:
            Tuple of (resampled_array, new_transform, new_crs)
        """
        if target_crs is None:
            target_crs = source_crs
        
        # Calculate target dimensions and transform
        height, width = source_array.shape[-2:]
        bounds = rasterio.transform.array_bounds(height, width, source_transform)
        
        if source_crs != target_crs:
            # Reproject bounds to target CRS
            transformer = CoordinateTransformer(source_crs, target_crs)
            bounds = transformer.transform_bounds(bounds)
        
        # Calculate new dimensions based on target resolution
        new_width = int((bounds[2] - bounds[0]) / target_resolution)
        new_height = int((bounds[3] - bounds[1]) / target_resolution)
        
        # Create target transform
        target_transform = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            new_width, new_height
        )
        
        # Prepare output array
        if source_array.ndim == 2:
            target_array = np.empty((new_height, new_width), dtype=source_array.dtype)
        else:
            target_array = np.empty((source_array.shape[0], new_height, new_width), dtype=source_array.dtype)
        
        # Perform resampling/reprojection
        reproject(
            source=source_array,
            destination=target_array,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=resampling_method
        )
        
        return target_array, target_transform, target_crs
    
    @staticmethod
    def align_rasters(
        arrays: List[np.ndarray],
        transforms: List[rasterio.Affine],
        crs_list: List[CRS],
        target_resolution: float,
        target_crs: Optional[CRS] = None
    ) -> Tuple[List[np.ndarray], rasterio.Affine, CRS]:
        """
        Align multiple rasters to common grid and resolution.
        
        Args:
            arrays: List of raster arrays
            transforms: List of corresponding transforms
            crs_list: List of corresponding CRS
            target_resolution: Target pixel resolution
            target_crs: Target CRS (if None, uses first raster's CRS)
        
        Returns:
            Tuple of (aligned_arrays, common_transform, common_crs)
        """
        if not arrays:
            raise ValueError("No arrays provided")
        
        if target_crs is None:
            target_crs = crs_list[0]
        
        # Calculate common bounds in target CRS
        all_bounds = []
        for i, (array, transform, crs) in enumerate(zip(arrays, transforms, crs_list)):
            height, width = array.shape[-2:]
            bounds = rasterio.transform.array_bounds(height, width, transform)
            
            if crs != target_crs:
                transformer = CoordinateTransformer(crs, target_crs)
                bounds = transformer.transform_bounds(bounds)
            
            all_bounds.append(bounds)
        
        # Find union of all bounds
        min_x = min(bounds[0] for bounds in all_bounds)
        min_y = min(bounds[1] for bounds in all_bounds)
        max_x = max(bounds[2] for bounds in all_bounds)
        max_y = max(bounds[3] for bounds in all_bounds)
        
        common_bounds = (min_x, min_y, max_x, max_y)
        
        # Calculate common grid
        common_width = int((max_x - min_x) / target_resolution)
        common_height = int((max_y - min_y) / target_resolution)
        
        common_transform = from_bounds(
            min_x, min_y, max_x, max_y,
            common_width, common_height
        )
        
        # Resample all arrays to common grid
        aligned_arrays = []
        for array, transform, crs in zip(arrays, transforms, crs_list):
            aligned_array, _, _ = RasterProcessor.resample_raster(
                array, transform, crs, target_resolution, target_crs
            )
            aligned_arrays.append(aligned_array)
        
        return aligned_arrays, common_transform, target_crs
    
    @staticmethod
    def create_mask(
        array: np.ndarray,
        nodata_value: Optional[float] = None,
        valid_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Create boolean mask for valid pixels.
        
        Args:
            array: Input raster array
            nodata_value: Value representing no data
            valid_range: Tuple of (min_valid, max_valid) values
        
        Returns:
            Boolean mask where True indicates valid pixels
        """
        mask = np.ones(array.shape, dtype=bool)
        
        # Mask nodata values
        if nodata_value is not None:
            if np.isnan(nodata_value):
                mask &= ~np.isnan(array)
            else:
                mask &= (array != nodata_value)
        
        # Mask values outside valid range
        if valid_range is not None:
            min_val, max_val = valid_range
            mask &= (array >= min_val) & (array <= max_val)
        
        # Mask infinite values
        mask &= np.isfinite(array)
        
        return mask
    
    @staticmethod
    def apply_scale_offset(
        array: np.ndarray,
        scale: float = 1.0,
        offset: float = 0.0,
        nodata_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply scale and offset transformation to raster data.
        
        Args:
            array: Input raster array
            scale: Scale factor
            offset: Offset value
            nodata_mask: Boolean mask for nodata pixels
        
        Returns:
            Scaled and offset array
        """
        result = array.astype(np.float32) * scale + offset
        
        if nodata_mask is not None:
            result[~nodata_mask] = np.nan
        
        return result


def utm_zone_from_longitude(longitude: float) -> int:
    """
    Calculate UTM zone number from longitude.
    
    Args:
        longitude: Longitude in decimal degrees
    
    Returns:
        UTM zone number (1-60)
    """
    return int((longitude + 180) / 6) + 1


def create_utm_crs(zone: int, hemisphere: str = 'north') -> CRS:
    """
    Create UTM CRS from zone number and hemisphere.
    
    Args:
        zone: UTM zone number (1-60)
        hemisphere: 'north' or 'south'
    
    Returns:
        UTM CRS object
    """
    if hemisphere.lower() == 'north':
        epsg_code = 32600 + zone
    else:
        epsg_code = 32700 + zone
    
    return CRS.from_epsg(epsg_code)


def pixel_to_world(
    row: int, 
    col: int, 
    transform: rasterio.Affine
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to world coordinates.
    
    Args:
        row: Pixel row (y-coordinate)
        col: Pixel column (x-coordinate)
        transform: Rasterio affine transform
    
    Returns:
        Tuple of (x, y) world coordinates
    """
    return transform * (col, row)


def world_to_pixel(
    x: float, 
    y: float, 
    transform: rasterio.Affine
) -> Tuple[int, int]:
    """
    Convert world coordinates to pixel coordinates.
    
    Args:
        x: World x-coordinate
        y: World y-coordinate
        transform: Rasterio affine transform
    
    Returns:
        Tuple of (row, col) pixel coordinates
    """
    inv_transform = ~transform
    col, row = inv_transform * (x, y)
    return int(row), int(col)