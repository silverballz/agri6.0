"""
Sentinel-2A band processing module for reading, calibrating, and resampling JP2 files.
Handles individual spectral bands and provides resampling to common 10m resolution grid.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

from .sentinel2_parser import BandInfo, Sentinel2Metadata


@dataclass
class BandData:
    """Container for processed band data with metadata."""
    band_id: str
    data: np.ndarray
    transform: rasterio.Affine
    crs: str
    nodata_value: Optional[float]
    resolution: float  # in meters
    shape: Tuple[int, int]  # (height, width)
    dtype: np.dtype


class Sentinel2BandProcessor:
    """Processor for Sentinel-2A spectral bands with calibration and resampling."""
    
    # Target bands for vegetation analysis
    TARGET_BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    
    # Digital Number to reflectance conversion factor for L2A products
    # L2A products are already atmospherically corrected and in reflectance units
    REFLECTANCE_SCALE_FACTOR = 10000.0  # Scale factor to convert to 0-1 range
    
    def __init__(self, target_resolution: float = 10.0):
        """
        Initialize band processor.
        
        Args:
            target_resolution: Target resolution in meters for resampling (default: 10.0)
        """
        self.target_resolution = target_resolution
        
    def read_band(self, band_info: BandInfo, apply_scaling: bool = True) -> BandData:
        """
        Read a single JP2 band file and return calibrated data.
        
        Args:
            band_info: BandInfo object containing file path and metadata
            apply_scaling: Whether to apply reflectance scaling (default: True)
            
        Returns:
            BandData object with processed band information
            
        Raises:
            FileNotFoundError: If JP2 file doesn't exist
            rasterio.errors.RasterioIOError: If file cannot be read
        """
        if not band_info.file_path.exists():
            raise FileNotFoundError(f"Band file not found: {band_info.file_path}")
        
        try:
            with rasterio.open(band_info.file_path) as src:
                # Read the band data
                data = src.read(1)  # Read first (and only) band
                
                # Apply reflectance scaling for L2A products
                if apply_scaling:
                    # Convert to float32 for processing
                    data = data.astype(np.float32)
                    
                    # Apply scaling factor (L2A products are scaled by 10000)
                    data = data / self.REFLECTANCE_SCALE_FACTOR
                    
                    # Clip to valid reflectance range [0, 1]
                    data = np.clip(data, 0.0, 1.0)
                
                # Extract resolution from band info (format: "R10m", "R20m", "R60m")
                resolution_str = band_info.resolution.replace('R', '').replace('m', '')
                resolution = float(resolution_str)
                
                return BandData(
                    band_id=band_info.band_id,
                    data=data,
                    transform=src.transform,
                    crs=src.crs.to_string(),
                    nodata_value=src.nodata,
                    resolution=resolution,
                    shape=data.shape,
                    dtype=data.dtype
                )
                
        except Exception as e:
            raise rasterio.errors.RasterioIOError(f"Failed to read band {band_info.band_id}: {str(e)}")
    
    def resample_band_to_target_resolution(self, band_data: BandData, 
                                         target_transform: Optional[rasterio.Affine] = None,
                                         target_shape: Optional[Tuple[int, int]] = None) -> BandData:
        """
        Resample band data to target resolution using bilinear interpolation.
        
        Args:
            band_data: BandData object to resample
            target_transform: Target affine transform (optional)
            target_shape: Target shape (height, width) (optional)
            
        Returns:
            BandData object with resampled data
        """
        # If already at target resolution, return as-is
        if abs(band_data.resolution - self.target_resolution) < 0.1:
            return band_data
        
        # Calculate target transform and shape if not provided
        if target_transform is None or target_shape is None:
            # Calculate new transform for target resolution
            pixel_size = self.target_resolution
            
            # Get bounds from original transform
            left = band_data.transform.c
            top = band_data.transform.f
            right = left + (band_data.shape[1] * band_data.transform.a)
            bottom = top + (band_data.shape[0] * band_data.transform.e)
            
            # Calculate new dimensions
            new_width = int((right - left) / pixel_size)
            new_height = int((top - bottom) / pixel_size)
            
            target_transform = rasterio.Affine(pixel_size, 0.0, left,
                                             0.0, -pixel_size, top)
            target_shape = (new_height, new_width)
        
        # Create output array
        resampled_data = np.zeros(target_shape, dtype=band_data.dtype)
        
        # Perform resampling
        reproject(
            source=band_data.data,
            destination=resampled_data,
            src_transform=band_data.transform,
            src_crs=band_data.crs,
            dst_transform=target_transform,
            dst_crs=band_data.crs,
            resampling=Resampling.bilinear
        )
        
        return BandData(
            band_id=band_data.band_id,
            data=resampled_data,
            transform=target_transform,
            crs=band_data.crs,
            nodata_value=band_data.nodata_value,
            resolution=self.target_resolution,
            shape=target_shape,
            dtype=band_data.dtype
        )
    
    def process_bands(self, band_files: Dict[str, BandInfo], 
                     target_bands: Optional[List[str]] = None) -> Dict[str, BandData]:
        """
        Process multiple bands with calibration and resampling to common grid.
        
        Args:
            band_files: Dictionary of band_id -> BandInfo
            target_bands: List of band IDs to process (default: TARGET_BANDS)
            
        Returns:
            Dictionary of band_id -> BandData with processed bands
        """
        if target_bands is None:
            target_bands = self.TARGET_BANDS
        
        processed_bands = {}
        reference_transform = None
        reference_shape = None
        
        # First pass: read all bands and determine reference grid from 10m bands
        band_data_raw = {}
        for band_id in target_bands:
            if band_id not in band_files:
                warnings.warn(f"Band {band_id} not found in band_files, skipping")
                continue
            
            try:
                band_data = self.read_band(band_files[band_id])
                band_data_raw[band_id] = band_data
                
                # Use 10m bands as reference for resampling grid
                if band_data.resolution == 10.0:
                    reference_transform = band_data.transform
                    reference_shape = band_data.shape
                    
            except Exception as e:
                warnings.warn(f"Failed to read band {band_id}: {str(e)}")
                continue
        
        # If no 10m reference found, calculate from first available band
        if reference_transform is None and band_data_raw:
            first_band = next(iter(band_data_raw.values()))
            
            # Calculate reference grid for 10m resolution
            pixel_size = self.target_resolution
            left = first_band.transform.c
            top = first_band.transform.f
            right = left + (first_band.shape[1] * first_band.transform.a)
            bottom = top + (first_band.shape[0] * first_band.transform.e)
            
            new_width = int((right - left) / pixel_size)
            new_height = int((top - bottom) / pixel_size)
            
            reference_transform = rasterio.Affine(pixel_size, 0.0, left,
                                                0.0, -pixel_size, top)
            reference_shape = (new_height, new_width)
        
        # Second pass: resample all bands to common grid
        for band_id, band_data in band_data_raw.items():
            try:
                resampled_band = self.resample_band_to_target_resolution(
                    band_data, reference_transform, reference_shape
                )
                processed_bands[band_id] = resampled_band
                
            except Exception as e:
                warnings.warn(f"Failed to resample band {band_id}: {str(e)}")
                continue
        
        return processed_bands
    
    def validate_band_data(self, band_data: BandData) -> Dict[str, bool]:
        """
        Validate processed band data for quality and consistency.
        
        Args:
            band_data: BandData object to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Check for valid data range (reflectance should be 0-1)
        if band_data.data.dtype in [np.float32, np.float64]:
            min_val = np.nanmin(band_data.data)
            max_val = np.nanmax(band_data.data)
            validation_results['valid_reflectance_range'] = bool(0.0 <= min_val and max_val <= 1.0)
        else:
            validation_results['valid_reflectance_range'] = True  # Skip for integer data
        
        # Check for reasonable data coverage (not all zeros or nodata)
        if band_data.nodata_value is not None:
            valid_pixels = ~np.isclose(band_data.data, band_data.nodata_value)
        else:
            valid_pixels = ~np.isnan(band_data.data)
        
        valid_pixel_ratio = np.sum(valid_pixels) / band_data.data.size
        validation_results['sufficient_data_coverage'] = bool(valid_pixel_ratio > 0.1)  # At least 10% valid
        
        # Check for expected resolution
        validation_results['correct_resolution'] = bool(abs(band_data.resolution - self.target_resolution) < 0.1)
        
        # Check for non-empty data
        validation_results['non_empty_data'] = bool(band_data.data.size > 0)
        
        # Check for finite values
        validation_results['finite_values'] = bool(np.all(np.isfinite(band_data.data[valid_pixels])))
        
        return validation_results


def read_and_process_bands(band_files: Dict[str, BandInfo], 
                          target_bands: Optional[List[str]] = None,
                          target_resolution: float = 10.0) -> Dict[str, BandData]:
    """
    Convenience function to read and process Sentinel-2A bands.
    
    Args:
        band_files: Dictionary of band_id -> BandInfo from Sentinel2SafeParser
        target_bands: List of band IDs to process
        target_resolution: Target resolution in meters
        
    Returns:
        Dictionary of band_id -> BandData with processed bands
    """
    processor = Sentinel2BandProcessor(target_resolution)
    return processor.process_bands(band_files, target_bands)