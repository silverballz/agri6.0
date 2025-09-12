"""
Cloud masking module using Sentinel-2A Scene Classification Layer (SCL).
Implements cloud detection, masking, and quality flagging for processed imagery.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import IntEnum
import warnings

from .band_processor import BandData


class SCLClass(IntEnum):
    """Sentinel-2A Scene Classification Layer class values."""
    NO_DATA = 0
    SATURATED_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11


@dataclass
class CloudMaskResult:
    """Container for cloud masking results."""
    cloud_mask: np.ndarray  # Boolean mask: True = cloudy/invalid, False = clear
    quality_flags: Dict[str, np.ndarray]  # Individual quality flag layers
    statistics: Dict[str, float]  # Coverage statistics
    scl_data: Optional[np.ndarray] = None  # Original SCL data (optional)
    
    def get_clear_pixel_mask(self) -> np.ndarray:
        """Get mask for clear (non-cloudy) pixels."""
        return ~self.cloud_mask
    
    def get_coverage_statistics(self) -> Dict[str, float]:
        """Get detailed coverage statistics."""
        total_pixels = self.cloud_mask.size
        if total_pixels == 0:
            return {}
        
        stats = {
            'total_pixels': total_pixels,
            'clear_pixels': int(np.sum(~self.cloud_mask)),
            'cloudy_pixels': int(np.sum(self.cloud_mask)),
            'clear_percentage': float(np.sum(~self.cloud_mask) / total_pixels * 100),
            'cloud_percentage': float(np.sum(self.cloud_mask) / total_pixels * 100)
        }
        
        # Add individual flag statistics if available
        for flag_name, flag_data in self.quality_flags.items():
            flag_count = int(np.sum(flag_data))
            stats[f'{flag_name}_pixels'] = flag_count
            stats[f'{flag_name}_percentage'] = float(flag_count / total_pixels * 100)
        
        return stats


class CloudMaskProcessor:
    """Processor for cloud masking using Sentinel-2A Scene Classification Layer."""
    
    # Default cloud classes to mask
    DEFAULT_CLOUD_CLASSES = [
        SCLClass.CLOUD_SHADOWS,
        SCLClass.CLOUD_MEDIUM_PROBABILITY,
        SCLClass.CLOUD_HIGH_PROBABILITY,
        SCLClass.THIN_CIRRUS
    ]
    
    # Additional quality classes that might be masked
    QUALITY_CLASSES = {
        'no_data': [SCLClass.NO_DATA],
        'saturated': [SCLClass.SATURATED_DEFECTIVE],
        'dark_areas': [SCLClass.DARK_AREA_PIXELS],
        'cloud_shadows': [SCLClass.CLOUD_SHADOWS],
        'clouds': [SCLClass.CLOUD_MEDIUM_PROBABILITY, SCLClass.CLOUD_HIGH_PROBABILITY],
        'cirrus': [SCLClass.THIN_CIRRUS],
        'snow_ice': [SCLClass.SNOW_ICE],
        'unclassified': [SCLClass.UNCLASSIFIED]
    }
    
    def __init__(self, 
                 cloud_classes: Optional[List[SCLClass]] = None,
                 include_shadows: bool = True,
                 include_cirrus: bool = True,
                 include_snow: bool = False):
        """
        Initialize cloud mask processor.
        
        Args:
            cloud_classes: List of SCL classes to mask as clouds (default: DEFAULT_CLOUD_CLASSES)
            include_shadows: Whether to include cloud shadows in mask
            include_cirrus: Whether to include thin cirrus in mask
            include_snow: Whether to include snow/ice in mask
        """
        if cloud_classes is None:
            cloud_classes = list(self.DEFAULT_CLOUD_CLASSES)
        
        # Modify cloud classes based on options
        if not include_shadows:
            cloud_classes = [c for c in cloud_classes if c != SCLClass.CLOUD_SHADOWS]
        
        if not include_cirrus:
            cloud_classes = [c for c in cloud_classes if c != SCLClass.THIN_CIRRUS]
        
        if include_snow:
            cloud_classes.append(SCLClass.SNOW_ICE)
        
        self.cloud_classes = cloud_classes
    
    def read_scl_data(self, scl_file_path: Path) -> Optional[np.ndarray]:
        """
        Read Scene Classification Layer data from JP2 file.
        
        Args:
            scl_file_path: Path to SCL JP2 file
            
        Returns:
            SCL data array or None if reading fails
        """
        if not scl_file_path.exists():
            warnings.warn(f"SCL file not found: {scl_file_path}")
            return None
        
        try:
            with rasterio.open(scl_file_path) as src:
                scl_data = src.read(1)  # Read first band
                return scl_data
        except Exception as e:
            warnings.warn(f"Failed to read SCL file {scl_file_path}: {str(e)}")
            return None
    
    def resample_scl_to_target(self, 
                              scl_data: np.ndarray,
                              scl_transform: rasterio.Affine,
                              scl_crs: str,
                              target_transform: rasterio.Affine,
                              target_shape: Tuple[int, int],
                              target_crs: str) -> np.ndarray:
        """
        Resample SCL data to match target band resolution and grid.
        
        Args:
            scl_data: Original SCL data array
            scl_transform: SCL affine transform
            scl_crs: SCL coordinate reference system
            target_transform: Target affine transform
            target_shape: Target shape (height, width)
            target_crs: Target coordinate reference system
            
        Returns:
            Resampled SCL data array
        """
        # Create output array
        resampled_scl = np.zeros(target_shape, dtype=scl_data.dtype)
        
        # Perform resampling using nearest neighbor (appropriate for classification data)
        reproject(
            source=scl_data,
            destination=resampled_scl,
            src_transform=scl_transform,
            src_crs=scl_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )
        
        return resampled_scl
    
    def create_cloud_mask_from_scl(self, scl_data: np.ndarray) -> CloudMaskResult:
        """
        Create cloud mask from Scene Classification Layer data.
        
        Args:
            scl_data: SCL data array with classification values
            
        Returns:
            CloudMaskResult object with mask and statistics
        """
        # Initialize cloud mask (False = clear, True = cloudy/invalid)
        cloud_mask = np.zeros(scl_data.shape, dtype=bool)
        
        # Create individual quality flag layers
        quality_flags = {}
        
        # Apply cloud classes
        for class_value in self.cloud_classes:
            cloud_mask |= (scl_data == class_value)
        
        # Create individual quality flag layers
        for flag_name, class_list in self.QUALITY_CLASSES.items():
            flag_mask = np.zeros(scl_data.shape, dtype=bool)
            for class_value in class_list:
                flag_mask |= (scl_data == class_value)
            quality_flags[flag_name] = flag_mask
        
        # Calculate statistics
        total_pixels = scl_data.size
        clear_pixels = np.sum(~cloud_mask)
        cloudy_pixels = np.sum(cloud_mask)
        
        statistics = {
            'total_pixels': total_pixels,
            'clear_pixels': int(clear_pixels),
            'cloudy_pixels': int(cloudy_pixels),
            'clear_percentage': float(clear_pixels / total_pixels * 100) if total_pixels > 0 else 0.0,
            'cloud_percentage': float(cloudy_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
        }
        
        return CloudMaskResult(
            cloud_mask=cloud_mask,
            quality_flags=quality_flags,
            statistics=statistics,
            scl_data=scl_data
        )
    
    def process_scl_file(self, 
                        scl_file_path: Path,
                        target_band: Optional[BandData] = None) -> Optional[CloudMaskResult]:
        """
        Process SCL file and create cloud mask, optionally resampling to target band.
        
        Args:
            scl_file_path: Path to SCL JP2 file
            target_band: Optional target band for resampling (if None, uses original SCL resolution)
            
        Returns:
            CloudMaskResult object or None if processing fails
        """
        # Read SCL data
        scl_data = self.read_scl_data(scl_file_path)
        if scl_data is None:
            return None
        
        # If target band is provided, resample SCL to match
        if target_band is not None:
            try:
                with rasterio.open(scl_file_path) as scl_src:
                    resampled_scl = self.resample_scl_to_target(
                        scl_data=scl_data,
                        scl_transform=scl_src.transform,
                        scl_crs=scl_src.crs.to_string(),
                        target_transform=target_band.transform,
                        target_shape=target_band.shape,
                        target_crs=target_band.crs
                    )
                    scl_data = resampled_scl
            except Exception as e:
                warnings.warn(f"Failed to resample SCL data: {str(e)}")
                return None
        
        # Create cloud mask
        return self.create_cloud_mask_from_scl(scl_data)
    
    def apply_cloud_mask_to_bands(self, 
                                 bands: Dict[str, BandData],
                                 cloud_mask_result: CloudMaskResult,
                                 fill_value: float = np.nan) -> Dict[str, BandData]:
        """
        Apply cloud mask to band data by setting cloudy pixels to fill_value.
        
        Args:
            bands: Dictionary of band_id -> BandData
            cloud_mask_result: CloudMaskResult with cloud mask
            fill_value: Value to use for masked pixels (default: NaN)
            
        Returns:
            Dictionary of masked band data
        """
        masked_bands = {}
        
        for band_id, band_data in bands.items():
            # Create a copy of the band data
            masked_data = band_data.data.copy()
            
            # Check if cloud mask shape matches band shape
            if cloud_mask_result.cloud_mask.shape != band_data.shape:
                warnings.warn(f"Cloud mask shape {cloud_mask_result.cloud_mask.shape} "
                            f"does not match band {band_id} shape {band_data.shape}")
                # Skip masking for this band
                masked_bands[band_id] = band_data
                continue
            
            # Apply cloud mask
            masked_data[cloud_mask_result.cloud_mask] = fill_value
            
            # Create new BandData object with masked data
            masked_bands[band_id] = BandData(
                band_id=band_data.band_id,
                data=masked_data,
                transform=band_data.transform,
                crs=band_data.crs,
                nodata_value=fill_value,
                resolution=band_data.resolution,
                shape=band_data.shape,
                dtype=band_data.dtype
            )
        
        return masked_bands
    
    def interpolate_cloudy_pixels(self, 
                                 band_data: BandData,
                                 cloud_mask: np.ndarray,
                                 method: str = 'nearest') -> BandData:
        """
        Interpolate cloudy pixels using neighboring clear pixels.
        
        Args:
            band_data: BandData object to interpolate
            cloud_mask: Boolean mask (True = cloudy pixels to interpolate)
            method: Interpolation method ('nearest', 'linear', or 'cubic')
            
        Returns:
            BandData object with interpolated values
        """
        from scipy.interpolate import griddata
        
        # Create a copy of the data
        interpolated_data = band_data.data.copy().astype(np.float32)
        
        # Get coordinates of all pixels
        rows, cols = np.mgrid[0:band_data.shape[0], 0:band_data.shape[1]]
        
        # Find clear pixels (not cloudy and not NaN)
        clear_mask = ~cloud_mask & np.isfinite(band_data.data)
        
        if np.sum(clear_mask) < 3:
            warnings.warn("Not enough clear pixels for interpolation")
            return band_data
        
        # Get coordinates and values of clear pixels
        clear_points = np.column_stack((rows[clear_mask], cols[clear_mask]))
        clear_values = band_data.data[clear_mask]
        
        # Get coordinates of cloudy pixels to interpolate
        cloudy_points = np.column_stack((rows[cloud_mask], cols[cloud_mask]))
        
        if len(cloudy_points) == 0:
            return band_data  # No cloudy pixels to interpolate
        
        try:
            # Perform interpolation
            interpolated_values = griddata(
                points=clear_points,
                values=clear_values,
                xi=cloudy_points,
                method=method,
                fill_value=np.nan
            )
            
            # Fill interpolated values back into the array
            interpolated_data[cloud_mask] = interpolated_values
            
        except Exception as e:
            warnings.warn(f"Interpolation failed: {str(e)}")
            return band_data
        
        # Create new BandData object
        return BandData(
            band_id=band_data.band_id,
            data=interpolated_data,
            transform=band_data.transform,
            crs=band_data.crs,
            nodata_value=band_data.nodata_value,
            resolution=band_data.resolution,
            shape=band_data.shape,
            dtype=interpolated_data.dtype
        )


def create_cloud_mask_from_scl_file(scl_file_path: Path,
                                   target_band: Optional[BandData] = None,
                                   cloud_classes: Optional[List[SCLClass]] = None) -> Optional[CloudMaskResult]:
    """
    Convenience function to create cloud mask from SCL file.
    
    Args:
        scl_file_path: Path to SCL JP2 file
        target_band: Optional target band for resampling
        cloud_classes: List of SCL classes to mask as clouds
        
    Returns:
        CloudMaskResult object or None if processing fails
    """
    processor = CloudMaskProcessor(cloud_classes=cloud_classes)
    return processor.process_scl_file(scl_file_path, target_band)


def apply_cloud_masking(bands: Dict[str, BandData],
                       scl_file_path: Path,
                       mask_clouds: bool = True,
                       interpolate: bool = False) -> Tuple[Dict[str, BandData], Optional[CloudMaskResult]]:
    """
    Convenience function to apply cloud masking to bands.
    
    Args:
        bands: Dictionary of band_id -> BandData
        scl_file_path: Path to SCL JP2 file
        mask_clouds: Whether to mask cloudy pixels with NaN
        interpolate: Whether to interpolate cloudy pixels
        
    Returns:
        Tuple of (processed_bands, cloud_mask_result)
    """
    # Use first band as reference for resampling
    reference_band = next(iter(bands.values())) if bands else None
    
    # Create cloud mask
    processor = CloudMaskProcessor()
    cloud_mask_result = processor.process_scl_file(scl_file_path, reference_band)
    
    if cloud_mask_result is None:
        warnings.warn("Failed to create cloud mask, returning original bands")
        return bands, None
    
    processed_bands = bands
    
    if mask_clouds:
        # Apply cloud mask
        processed_bands = processor.apply_cloud_mask_to_bands(bands, cloud_mask_result)
    
    if interpolate:
        # Interpolate cloudy pixels
        interpolated_bands = {}
        for band_id, band_data in processed_bands.items():
            interpolated_bands[band_id] = processor.interpolate_cloudy_pixels(
                band_data, cloud_mask_result.cloud_mask
            )
        processed_bands = interpolated_bands
    
    return processed_bands, cloud_mask_result