"""
Vegetation index calculation module for Sentinel-2A multispectral data.
Implements standard vegetation and soil indices including NDVI, SAVI, EVI, NDWI, and NDSI.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

from .band_processor import BandData


@dataclass
class IndexResult:
    """Container for vegetation index calculation results."""
    index_name: str
    data: np.ndarray
    valid_range: Tuple[float, float]
    description: str
    formula: str
    
    def get_statistics(self) -> Dict[str, float]:
        """Get basic statistics for the index values."""
        valid_data = self.data[np.isfinite(self.data)]
        if len(valid_data) == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'valid_pixels': 0,
                'total_pixels': self.data.size
            }
        
        return {
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'valid_pixels': len(valid_data),
            'total_pixels': self.data.size
        }


class VegetationIndexCalculator:
    """Calculator for vegetation and soil indices from Sentinel-2A bands."""
    
    # Cache for computed indices to avoid recomputation
    _cache = {}
    
    # Index definitions with metadata
    INDEX_DEFINITIONS = {
        'NDVI': {
            'name': 'Normalized Difference Vegetation Index',
            'formula': '(NIR - Red) / (NIR + Red)',
            'bands': ['B08', 'B04'],
            'range': (-1.0, 1.0),
            'description': 'Measures vegetation greenness and health'
        },
        'SAVI': {
            'name': 'Soil Adjusted Vegetation Index',
            'formula': '((NIR - Red) / (NIR + Red + L)) * (1 + L)',
            'bands': ['B08', 'B04'],
            'range': (-1.5, 1.5),
            'description': 'NDVI adjusted for soil brightness, L=0.5'
        },
        'EVI': {
            'name': 'Enhanced Vegetation Index',
            'formula': '2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))',
            'bands': ['B08', 'B04', 'B02'],
            'range': (-1.0, 1.0),
            'description': 'Enhanced vegetation index with atmospheric correction'
        },
        'NDWI': {
            'name': 'Normalized Difference Water Index',
            'formula': '(Green - NIR) / (Green + NIR)',
            'bands': ['B03', 'B08'],
            'range': (-1.0, 1.0),
            'description': 'Measures vegetation water content and stress'
        },
        'GNDVI': {
            'name': 'Green Normalized Difference Vegetation Index',
            'formula': '(NIR - Green) / (NIR + Green)',
            'bands': ['B08', 'B03'],
            'range': (-1.0, 1.0),
            'description': 'Green-based vegetation index, sensitive to chlorophyll'
        },
        'NDSI': {
            'name': 'Normalized Difference Soil Index',
            'formula': '(SWIR1 - SWIR2) / (SWIR1 + SWIR2)',
            'bands': ['B11', 'B12'],
            'range': (-1.0, 1.0),
            'description': 'Soil moisture and composition indicator'
        }
    }
    
    def __init__(self, nodata_threshold: float = 0.0001, use_cache: bool = True):
        """
        Initialize vegetation index calculator.
        
        Args:
            nodata_threshold: Threshold below which values are considered nodata
            use_cache: Whether to cache computed indices (default: True)
        """
        self.nodata_threshold = nodata_threshold
        self.use_cache = use_cache
    
    def _get_cache_key(self, bands: Dict[str, BandData], index_name: str, **kwargs) -> str:
        """Generate cache key for index calculation."""
        band_ids = sorted([bid for bid in bands.keys()])
        band_shapes = tuple(bands[bid].shape for bid in band_ids)
        # Use id() of data arrays for uniqueness
        band_ids_str = '_'.join(band_ids)
        kwargs_str = '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{index_name}_{band_ids_str}_{band_shapes}_{kwargs_str}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[IndexResult]:
        """Get result from cache if available."""
        if not self.use_cache:
            return None
        return self._cache.get(cache_key)
    
    def _store_in_cache(self, cache_key: str, result: IndexResult):
        """Store result in cache."""
        if self.use_cache:
            # Limit cache size to prevent memory issues
            if len(self._cache) > 50:
                # Remove oldest entries
                keys_to_remove = list(self._cache.keys())[:10]
                for key in keys_to_remove:
                    del self._cache[key]
            self._cache[cache_key] = result
    
    def _validate_bands(self, bands: Dict[str, BandData], required_bands: list) -> bool:
        """
        Validate that required bands are available and compatible.
        
        Args:
            bands: Dictionary of band_id -> BandData
            required_bands: List of required band IDs
            
        Returns:
            True if all required bands are available and compatible
        """
        # Check if all required bands are present
        for band_id in required_bands:
            if band_id not in bands:
                warnings.warn(f"Required band {band_id} not found")
                return False
        
        # Check if all bands have compatible shapes
        shapes = [bands[band_id].shape for band_id in required_bands]
        if not all(shape == shapes[0] for shape in shapes):
            warnings.warn("Band shapes are not compatible")
            return False
        
        return True
    
    def _apply_nodata_mask(self, *arrays: np.ndarray) -> np.ndarray:
        """
        Create a mask for valid pixels across multiple arrays.
        
        Args:
            *arrays: Variable number of numpy arrays
            
        Returns:
            Boolean mask where True indicates valid pixels
        """
        mask = np.ones(arrays[0].shape, dtype=bool)
        
        for array in arrays:
            # Mask out NaN, inf, and very small values
            mask &= np.isfinite(array)
            mask &= np.abs(array) > self.nodata_threshold
        
        return mask
    
    def calculate_ndvi(self, bands: Dict[str, BandData]) -> Optional[IndexResult]:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        Formula: NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            bands: Dictionary containing B08 (NIR) and B04 (Red) bands
            
        Returns:
            IndexResult object or None if calculation fails
        """
        required_bands = ['B08', 'B04']
        if not self._validate_bands(bands, required_bands):
            return None
        
        nir = bands['B08'].data.astype(np.float32)
        red = bands['B04'].data.astype(np.float32)
        
        # Optimized calculation using numpy's where for better performance
        denominator = nir + red
        
        # Use numpy.where for efficient conditional calculation
        # This avoids creating intermediate masks and is faster
        ndvi = np.where(
            np.abs(denominator) > self.nodata_threshold,
            (nir - red) / denominator,
            np.nan
        )
        
        return IndexResult(
            index_name='NDVI',
            data=ndvi,
            valid_range=self.INDEX_DEFINITIONS['NDVI']['range'],
            description=self.INDEX_DEFINITIONS['NDVI']['description'],
            formula=self.INDEX_DEFINITIONS['NDVI']['formula']
        )
    
    def calculate_savi(self, bands: Dict[str, BandData], L: float = 0.5) -> Optional[IndexResult]:
        """
        Calculate Soil Adjusted Vegetation Index (SAVI).
        
        Formula: SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        
        Args:
            bands: Dictionary containing B08 (NIR) and B04 (Red) bands
            L: Soil brightness correction factor (default: 0.5)
            
        Returns:
            IndexResult object or None if calculation fails
        """
        required_bands = ['B08', 'B04']
        if not self._validate_bands(bands, required_bands):
            return None
        
        nir = bands['B08'].data.astype(np.float32)
        red = bands['B04'].data.astype(np.float32)
        
        # Optimized calculation
        denominator = nir + red + L
        savi = np.where(
            np.abs(denominator) > self.nodata_threshold,
            ((nir - red) / denominator) * (1 + L),
            np.nan
        )
        
        return IndexResult(
            index_name='SAVI',
            data=savi,
            valid_range=self.INDEX_DEFINITIONS['SAVI']['range'],
            description=self.INDEX_DEFINITIONS['SAVI']['description'],
            formula=f"((NIR - Red) / (NIR + Red + {L})) * (1 + {L})"
        )
    
    def calculate_evi(self, bands: Dict[str, BandData]) -> Optional[IndexResult]:
        """
        Calculate Enhanced Vegetation Index (EVI).
        
        Formula: EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        
        Args:
            bands: Dictionary containing B08 (NIR), B04 (Red), and B02 (Blue) bands
            
        Returns:
            IndexResult object or None if calculation fails
        """
        required_bands = ['B08', 'B04', 'B02']
        if not self._validate_bands(bands, required_bands):
            return None
        
        nir = bands['B08'].data.astype(np.float32)
        red = bands['B04'].data.astype(np.float32)
        blue = bands['B02'].data.astype(np.float32)
        
        # Optimized calculation - compute denominator once
        denominator = nir + 6.0 * red - 7.5 * blue + 1.0
        
        # Use a larger threshold to prevent extreme values
        # When denominator is too small, the EVI becomes unreliable
        min_denominator = 0.1
        evi = np.where(
            np.abs(denominator) > min_denominator,
            2.5 * ((nir - red) / denominator),
            np.nan
        )
        
        # Clamp EVI values to valid range [-1.0, 1.0]
        # Values outside this range are physically unrealistic
        evi = np.clip(evi, -1.0, 1.0)
        
        return IndexResult(
            index_name='EVI',
            data=evi,
            valid_range=self.INDEX_DEFINITIONS['EVI']['range'],
            description=self.INDEX_DEFINITIONS['EVI']['description'],
            formula=self.INDEX_DEFINITIONS['EVI']['formula']
        )
    
    def calculate_ndwi(self, bands: Dict[str, BandData]) -> Optional[IndexResult]:
        """
        Calculate Normalized Difference Water Index (NDWI).
        
        Formula: NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            bands: Dictionary containing B03 (Green) and B08 (NIR) bands
            
        Returns:
            IndexResult object or None if calculation fails
        """
        required_bands = ['B03', 'B08']
        if not self._validate_bands(bands, required_bands):
            return None
        
        green = bands['B03'].data.astype(np.float32)
        nir = bands['B08'].data.astype(np.float32)
        
        # Optimized calculation
        denominator = green + nir
        ndwi = np.where(
            np.abs(denominator) > self.nodata_threshold,
            (green - nir) / denominator,
            np.nan
        )
        
        return IndexResult(
            index_name='NDWI',
            data=ndwi,
            valid_range=self.INDEX_DEFINITIONS['NDWI']['range'],
            description=self.INDEX_DEFINITIONS['NDWI']['description'],
            formula=self.INDEX_DEFINITIONS['NDWI']['formula']
        )
    
    def calculate_gndvi(self, bands: Dict[str, BandData]) -> Optional[IndexResult]:
        """
        Calculate Green Normalized Difference Vegetation Index (GNDVI).
        
        Formula: GNDVI = (NIR - Green) / (NIR + Green)
        
        Args:
            bands: Dictionary containing B08 (NIR) and B03 (Green) bands
            
        Returns:
            IndexResult object or None if calculation fails
        """
        required_bands = ['B08', 'B03']
        if not self._validate_bands(bands, required_bands):
            return None
        
        nir = bands['B08'].data.astype(np.float32)
        green = bands['B03'].data.astype(np.float32)
        
        # Create mask for valid pixels
        valid_mask = self._apply_nodata_mask(nir, green)
        
        # Calculate GNDVI
        denominator = nir + green
        gndvi = np.full_like(nir, np.nan)
        
        # Avoid division by zero
        valid_calc = valid_mask & (np.abs(denominator) > self.nodata_threshold)
        gndvi[valid_calc] = (nir[valid_calc] - green[valid_calc]) / denominator[valid_calc]
        
        return IndexResult(
            index_name='GNDVI',
            data=gndvi,
            valid_range=self.INDEX_DEFINITIONS['GNDVI']['range'],
            description=self.INDEX_DEFINITIONS['GNDVI']['description'],
            formula=self.INDEX_DEFINITIONS['GNDVI']['formula']
        )
    
    def calculate_ndsi(self, bands: Dict[str, BandData]) -> Optional[IndexResult]:
        """
        Calculate Normalized Difference Soil Index (NDSI).
        
        Formula: NDSI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
        
        Args:
            bands: Dictionary containing B11 (SWIR1) and B12 (SWIR2) bands
            
        Returns:
            IndexResult object or None if calculation fails
        """
        required_bands = ['B11', 'B12']
        if not self._validate_bands(bands, required_bands):
            return None
        
        swir1 = bands['B11'].data.astype(np.float32)
        swir2 = bands['B12'].data.astype(np.float32)
        
        # Create mask for valid pixels
        valid_mask = self._apply_nodata_mask(swir1, swir2)
        
        # Calculate NDSI
        denominator = swir1 + swir2
        ndsi = np.full_like(swir1, np.nan)
        
        # Avoid division by zero
        valid_calc = valid_mask & (np.abs(denominator) > self.nodata_threshold)
        ndsi[valid_calc] = (swir1[valid_calc] - swir2[valid_calc]) / denominator[valid_calc]
        
        return IndexResult(
            index_name='NDSI',
            data=ndsi,
            valid_range=self.INDEX_DEFINITIONS['NDSI']['range'],
            description=self.INDEX_DEFINITIONS['NDSI']['description'],
            formula=self.INDEX_DEFINITIONS['NDSI']['formula']
        )
    
    def calculate_all_indices(self, bands: Dict[str, BandData]) -> Dict[str, IndexResult]:
        """
        Calculate all available vegetation indices based on available bands.
        
        Args:
            bands: Dictionary of band_id -> BandData
            
        Returns:
            Dictionary of index_name -> IndexResult for successfully calculated indices
        """
        results = {}
        
        # Define calculation methods
        calculations = {
            'NDVI': self.calculate_ndvi,
            'SAVI': self.calculate_savi,
            'EVI': self.calculate_evi,
            'NDWI': self.calculate_ndwi,
            'GNDVI': self.calculate_gndvi,
            'NDSI': self.calculate_ndsi
        }
        
        for index_name, calc_method in calculations.items():
            try:
                result = calc_method(bands)
                if result is not None:
                    results[index_name] = result
            except Exception as e:
                warnings.warn(f"Failed to calculate {index_name}: {str(e)}")
        
        return results
    
    def validate_index_values(self, index_result: IndexResult) -> Dict[str, bool]:
        """
        Validate calculated index values against expected ranges.
        
        Args:
            index_result: IndexResult object to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Check if values are within expected range
        valid_data = index_result.data[np.isfinite(index_result.data)]
        if len(valid_data) > 0:
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            expected_min, expected_max = index_result.valid_range
            
            validation_results['within_expected_range'] = bool(
                expected_min <= min_val and max_val <= expected_max
            )
            validation_results['has_valid_data'] = True
        else:
            validation_results['within_expected_range'] = False
            validation_results['has_valid_data'] = False
        
        # Check for reasonable data coverage
        total_pixels = index_result.data.size
        valid_pixels = len(valid_data) if len(valid_data) > 0 else 0
        coverage_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0
        
        validation_results['sufficient_coverage'] = bool(coverage_ratio > 0.1)  # At least 10%
        validation_results['coverage_ratio'] = float(coverage_ratio)
        
        return validation_results


def calculate_vegetation_indices(bands: Dict[str, BandData], 
                               indices: Optional[list] = None) -> Dict[str, IndexResult]:
    """
    Convenience function to calculate vegetation indices from processed bands.
    
    Args:
        bands: Dictionary of band_id -> BandData from band processor
        indices: List of index names to calculate (default: all available)
        
    Returns:
        Dictionary of index_name -> IndexResult
    """
    calculator = VegetationIndexCalculator()
    
    if indices is None:
        return calculator.calculate_all_indices(bands)
    else:
        results = {}
        calculation_methods = {
            'NDVI': calculator.calculate_ndvi,
            'SAVI': calculator.calculate_savi,
            'EVI': calculator.calculate_evi,
            'NDWI': calculator.calculate_ndwi,
            'GNDVI': calculator.calculate_gndvi,
            'NDSI': calculator.calculate_ndsi
        }
        
        for index_name in indices:
            if index_name in calculation_methods:
                try:
                    result = calculation_methods[index_name](bands)
                    if result is not None:
                        results[index_name] = result
                except Exception as e:
                    warnings.warn(f"Failed to calculate {index_name}: {str(e)}")
            else:
                warnings.warn(f"Unknown index: {index_name}")
        
        return results