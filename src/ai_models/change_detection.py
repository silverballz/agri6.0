"""
Multi-temporal Change Detection Module

Analyzes changes between two dates of satellite imagery to identify:
- Vegetation improvement or degradation
- Significant changes in crop health
- Temporal patterns and anomalies

Part of the USP features for AgriFlux dashboard.
"""

import numpy as np
import rasterio
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Classification of change types."""
    SIGNIFICANT_IMPROVEMENT = "significant_improvement"
    MODERATE_IMPROVEMENT = "moderate_improvement"
    NO_CHANGE = "no_change"
    MODERATE_DEGRADATION = "moderate_degradation"
    SIGNIFICANT_DEGRADATION = "significant_degradation"


@dataclass
class ChangeDetectionResult:
    """Container for change detection analysis results."""
    change_magnitude: np.ndarray
    change_type: np.ndarray
    change_percentage: float
    improvement_area: float
    degradation_area: float
    stable_area: float
    statistics: Dict[str, float]
    
    def get_change_summary(self) -> Dict[str, any]:
        """Get a summary of detected changes."""
        return {
            'total_change_percentage': self.change_percentage,
            'improvement_area_percentage': self.improvement_area,
            'degradation_area_percentage': self.degradation_area,
            'stable_area_percentage': self.stable_area,
            'mean_change': self.statistics.get('mean_change', 0),
            'max_improvement': self.statistics.get('max_improvement', 0),
            'max_degradation': self.statistics.get('max_degradation', 0)
        }


class ChangeDetector:
    """
    Multi-temporal change detection for vegetation indices.
    
    Compares two dates of imagery to identify and classify changes.
    """
    
    # Thresholds for change classification (for NDVI)
    THRESHOLDS = {
        'significant_improvement': 0.15,
        'moderate_improvement': 0.05,
        'moderate_degradation': -0.05,
        'significant_degradation': -0.15
    }
    
    def __init__(self, 
                 significant_threshold: float = 0.15,
                 moderate_threshold: float = 0.05):
        """
        Initialize change detector.
        
        Args:
            significant_threshold: Threshold for significant changes (absolute value)
            moderate_threshold: Threshold for moderate changes (absolute value)
        """
        self.significant_threshold = abs(significant_threshold)
        self.moderate_threshold = abs(moderate_threshold)
    
    def load_index_from_geotiff(self, geotiff_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load vegetation index data from GeoTIFF file.
        
        Args:
            geotiff_path: Path to GeoTIFF file
            
        Returns:
            Tuple of (data array, metadata dict)
        """
        try:
            with rasterio.open(geotiff_path) as src:
                data = src.read(1)
                metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'shape': data.shape
                }
                return data, metadata
        except Exception as e:
            logger.error(f"Failed to load GeoTIFF {geotiff_path}: {e}")
            raise
    
    def calculate_change_magnitude(self, 
                                   before: np.ndarray, 
                                   after: np.ndarray) -> np.ndarray:
        """
        Calculate pixel-wise change magnitude.
        
        Args:
            before: Earlier date index values
            after: Later date index values
            
        Returns:
            Change magnitude array (after - before)
        """
        # Ensure arrays have same shape
        if before.shape != after.shape:
            raise ValueError(f"Array shapes don't match: {before.shape} vs {after.shape}")
        
        # Calculate change
        change = after.astype(np.float32) - before.astype(np.float32)
        
        # Mask invalid values
        valid_mask = np.isfinite(before) & np.isfinite(after)
        change[~valid_mask] = np.nan
        
        return change
    
    def classify_changes(self, change_magnitude: np.ndarray) -> np.ndarray:
        """
        Classify changes into categories based on magnitude.
        
        Args:
            change_magnitude: Array of change values
            
        Returns:
            Array of change type classifications (integer codes)
        """
        # Initialize with NO_CHANGE (code 2)
        change_types = np.full_like(change_magnitude, 2, dtype=np.int8)
        
        # Classify based on thresholds
        # Significant improvement (code 0)
        change_types[change_magnitude >= self.significant_threshold] = 0
        
        # Moderate improvement (code 1)
        change_types[(change_magnitude >= self.moderate_threshold) & 
                    (change_magnitude < self.significant_threshold)] = 1
        
        # Moderate degradation (code 3)
        change_types[(change_magnitude <= -self.moderate_threshold) & 
                    (change_magnitude > -self.significant_threshold)] = 3
        
        # Significant degradation (code 4)
        change_types[change_magnitude <= -self.significant_threshold] = 4
        
        # Mark invalid pixels as -1
        change_types[~np.isfinite(change_magnitude)] = -1
        
        return change_types
    
    def calculate_statistics(self, 
                            change_magnitude: np.ndarray,
                            change_types: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for the change detection.
        
        Args:
            change_magnitude: Array of change values
            change_types: Array of change classifications
            
        Returns:
            Dictionary of statistics
        """
        valid_changes = change_magnitude[np.isfinite(change_magnitude)]
        
        if len(valid_changes) == 0:
            return {
                'mean_change': 0.0,
                'std_change': 0.0,
                'max_improvement': 0.0,
                'max_degradation': 0.0,
                'total_valid_pixels': 0
            }
        
        stats = {
            'mean_change': float(np.mean(valid_changes)),
            'std_change': float(np.std(valid_changes)),
            'median_change': float(np.median(valid_changes)),
            'max_improvement': float(np.max(valid_changes)),
            'max_degradation': float(np.min(valid_changes)),
            'total_valid_pixels': int(len(valid_changes))
        }
        
        # Calculate area percentages for each change type
        total_pixels = len(valid_changes)
        for i, change_type in enumerate([
            'significant_improvement',
            'moderate_improvement', 
            'no_change',
            'moderate_degradation',
            'significant_degradation'
        ]):
            count = np.sum(change_types == i)
            stats[f'{change_type}_pixels'] = int(count)
            stats[f'{change_type}_percentage'] = float(count / total_pixels * 100)
        
        return stats
    
    def detect_changes(self,
                      before_path: str,
                      after_path: str,
                      index_name: str = "NDVI") -> ChangeDetectionResult:
        """
        Perform complete change detection analysis.
        
        Args:
            before_path: Path to earlier date GeoTIFF
            after_path: Path to later date GeoTIFF
            index_name: Name of the vegetation index being analyzed
            
        Returns:
            ChangeDetectionResult object with complete analysis
        """
        logger.info(f"Detecting changes between {before_path} and {after_path}")
        
        # Load data
        before_data, before_meta = self.load_index_from_geotiff(before_path)
        after_data, after_meta = self.load_index_from_geotiff(after_path)
        
        # Calculate change magnitude
        change_magnitude = self.calculate_change_magnitude(before_data, after_data)
        
        # Classify changes
        change_types = self.classify_changes(change_magnitude)
        
        # Calculate statistics
        stats = self.calculate_statistics(change_magnitude, change_types)
        
        # Calculate area percentages
        valid_pixels = stats['total_valid_pixels']
        if valid_pixels > 0:
            improvement_area = (
                stats.get('significant_improvement_percentage', 0) +
                stats.get('moderate_improvement_percentage', 0)
            )
            degradation_area = (
                stats.get('significant_degradation_percentage', 0) +
                stats.get('moderate_degradation_percentage', 0)
            )
            stable_area = stats.get('no_change_percentage', 0)
            
            # Overall change percentage (pixels with any change)
            change_percentage = 100.0 - stable_area
        else:
            improvement_area = 0.0
            degradation_area = 0.0
            stable_area = 0.0
            change_percentage = 0.0
        
        result = ChangeDetectionResult(
            change_magnitude=change_magnitude,
            change_type=change_types,
            change_percentage=change_percentage,
            improvement_area=improvement_area,
            degradation_area=degradation_area,
            stable_area=stable_area,
            statistics=stats
        )
        
        logger.info(f"Change detection complete: {change_percentage:.1f}% changed")
        return result
    
    def get_change_hotspots(self,
                           change_magnitude: np.ndarray,
                           percentile: float = 95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify hotspots of significant change.
        
        Args:
            change_magnitude: Array of change values
            percentile: Percentile threshold for hotspot identification
            
        Returns:
            Tuple of (improvement_hotspots, degradation_hotspots) as boolean masks
        """
        valid_changes = change_magnitude[np.isfinite(change_magnitude)]
        
        if len(valid_changes) == 0:
            return (
                np.zeros_like(change_magnitude, dtype=bool),
                np.zeros_like(change_magnitude, dtype=bool)
            )
        
        # Calculate thresholds
        improvement_threshold = np.percentile(valid_changes, percentile)
        degradation_threshold = np.percentile(valid_changes, 100 - percentile)
        
        # Create hotspot masks
        improvement_hotspots = change_magnitude >= improvement_threshold
        degradation_hotspots = change_magnitude <= degradation_threshold
        
        return improvement_hotspots, degradation_hotspots
    
    def export_change_map(self,
                         result: ChangeDetectionResult,
                         output_path: str,
                         reference_geotiff: str):
        """
        Export change detection results as a GeoTIFF.
        
        Args:
            result: ChangeDetectionResult object
            output_path: Path for output GeoTIFF
            reference_geotiff: Path to reference GeoTIFF for georeferencing
        """
        try:
            with rasterio.open(reference_geotiff) as src:
                profile = src.profile.copy()
                profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    compress='lzw'
                )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(result.change_magnitude.astype(np.float32), 1)
            
            logger.info(f"Change map exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export change map: {e}")
            raise


def compare_imagery_dates(before_imagery_id: int,
                          after_imagery_id: int,
                          db_manager,
                          index_name: str = "NDVI") -> Optional[ChangeDetectionResult]:
    """
    Convenience function to compare two imagery dates from database.
    
    Args:
        before_imagery_id: Database ID of earlier imagery
        after_imagery_id: Database ID of later imagery
        db_manager: DatabaseManager instance
        index_name: Name of index to compare (default: NDVI)
        
    Returns:
        ChangeDetectionResult or None if comparison fails
    """
    try:
        # Get imagery records
        before_record = db_manager.get_processed_imagery(before_imagery_id)
        after_record = db_manager.get_processed_imagery(after_imagery_id)
        
        if not before_record or not after_record:
            logger.error("One or both imagery records not found")
            return None
        
        # Get paths for the specified index
        index_path_key = f"{index_name.lower()}_path"
        before_path = before_record.get(index_path_key)
        after_path = after_record.get(index_path_key)
        
        if not before_path or not after_path:
            logger.error(f"Index {index_name} not available for both dates")
            return None
        
        # Perform change detection
        detector = ChangeDetector()
        result = detector.detect_changes(before_path, after_path, index_name)
        
        return result
        
    except Exception as e:
        logger.error(f"Change detection failed: {e}")
        return None


def get_change_type_color(change_type_code: int) -> str:
    """
    Get color code for visualization based on change type.
    
    Args:
        change_type_code: Integer code for change type
        
    Returns:
        Hex color code
    """
    color_map = {
        0: '#00ff00',  # Significant improvement - bright green
        1: '#90ee90',  # Moderate improvement - light green
        2: '#ffff00',  # No change - yellow
        3: '#ffa500',  # Moderate degradation - orange
        4: '#ff0000',  # Significant degradation - red
        -1: '#808080'  # Invalid - gray
    }
    return color_map.get(change_type_code, '#808080')


def get_change_type_label(change_type_code: int) -> str:
    """
    Get human-readable label for change type.
    
    Args:
        change_type_code: Integer code for change type
        
    Returns:
        Label string
    """
    label_map = {
        0: 'Significant Improvement',
        1: 'Moderate Improvement',
        2: 'No Change',
        3: 'Moderate Degradation',
        4: 'Significant Degradation',
        -1: 'Invalid'
    }
    return label_map.get(change_type_code, 'Unknown')
