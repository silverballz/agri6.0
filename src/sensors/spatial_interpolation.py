"""
Spatial interpolation of point sensor data.

This module provides functions for spatially interpolating point sensor
measurements to create continuous surfaces that can be compared with
satellite imagery pixels.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, Rbf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import logging

from .data_ingestion import SensorReading

logger = logging.getLogger(__name__)


@dataclass
class InterpolationGrid:
    """Represents an interpolated grid of sensor values."""
    values: np.ndarray  # 2D array of interpolated values
    x_coords: np.ndarray  # X coordinates of grid points
    y_coords: np.ndarray  # Y coordinates of grid points
    method: str  # Interpolation method used
    sensor_type: str  # Type of sensor data
    timestamp: Optional[str] = None
    quality_mask: Optional[np.ndarray] = None  # Mask indicating interpolation quality


class SpatialInterpolator:
    """Handles spatial interpolation of point sensor data."""
    
    def __init__(self):
        self.supported_methods = [
            'linear', 'nearest', 'cubic', 'rbf', 'kriging', 'idw'
        ]
        
        # Default parameters for different interpolation methods
        self.method_params = {
            'rbf': {'function': 'multiquadric', 'epsilon': 1.0},
            'kriging': {'kernel': RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)},
            'idw': {'power': 2.0}  # Inverse Distance Weighting power
        }
    
    def interpolate_sensors(self, readings: List[SensorReading],
                          grid_bounds: Tuple[float, float, float, float],
                          grid_resolution: float = 0.001,  # degrees
                          method: str = 'linear') -> InterpolationGrid:
        """
        Interpolate sensor readings to a regular grid.
        
        Args:
            readings: List of sensor readings with coordinates
            grid_bounds: (min_lon, min_lat, max_lon, max_lat)
            grid_resolution: Grid resolution in degrees
            method: Interpolation method
            
        Returns:
            InterpolationGrid object with interpolated values
        """
        if not readings:
            raise ValueError("No sensor readings provided")
        
        # Filter readings with valid coordinates
        valid_readings = [r for r in readings if r.latitude is not None and r.longitude is not None]
        
        if not valid_readings:
            raise ValueError("No sensor readings with valid coordinates")
        
        # Extract coordinates and values
        coords = np.array([[r.longitude, r.latitude] for r in valid_readings])
        values = np.array([r.value for r in valid_readings])
        
        # Create interpolation grid
        min_lon, min_lat, max_lon, max_lat = grid_bounds
        
        x_coords = np.arange(min_lon, max_lon + grid_resolution, grid_resolution)
        y_coords = np.arange(min_lat, max_lat + grid_resolution, grid_resolution)
        
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # Perform interpolation
        if method in ['linear', 'nearest', 'cubic']:
            interpolated_values = self._scipy_interpolation(
                coords, values, grid_points, method
            )
        elif method == 'rbf':
            interpolated_values = self._rbf_interpolation(
                coords, values, grid_points
            )
        elif method == 'kriging':
            interpolated_values = self._kriging_interpolation(
                coords, values, grid_points
            )
        elif method == 'idw':
            interpolated_values = self._idw_interpolation(
                coords, values, grid_points
            )
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
        
        # Reshape to grid
        interpolated_grid = interpolated_values.reshape(grid_x.shape)
        
        # Calculate quality mask
        quality_mask = self._calculate_quality_mask(
            coords, grid_points, grid_x.shape
        )
        
        # Get sensor type from first reading
        sensor_type = valid_readings[0].sensor_type
        
        return InterpolationGrid(
            values=interpolated_grid,
            x_coords=x_coords,
            y_coords=y_coords,
            method=method,
            sensor_type=sensor_type,
            quality_mask=quality_mask
        )
    
    def interpolate_to_points(self, readings: List[SensorReading],
                            target_points: List[Tuple[float, float]],
                            method: str = 'linear') -> List[float]:
        """
        Interpolate sensor values to specific target points.
        
        Args:
            readings: List of sensor readings with coordinates
            target_points: List of (longitude, latitude) tuples
            method: Interpolation method
            
        Returns:
            List of interpolated values at target points
        """
        if not readings or not target_points:
            return []
        
        # Filter readings with valid coordinates
        valid_readings = [r for r in readings if r.latitude is not None and r.longitude is not None]
        
        if not valid_readings:
            return [np.nan] * len(target_points)
        
        # Extract coordinates and values
        coords = np.array([[r.longitude, r.latitude] for r in valid_readings])
        values = np.array([r.value for r in valid_readings])
        target_coords = np.array(target_points)
        
        # Perform interpolation
        if method in ['linear', 'nearest', 'cubic']:
            interpolated_values = self._scipy_interpolation(
                coords, values, target_coords, method
            )
        elif method == 'rbf':
            interpolated_values = self._rbf_interpolation(
                coords, values, target_coords
            )
        elif method == 'kriging':
            interpolated_values = self._kriging_interpolation(
                coords, values, target_coords
            )
        elif method == 'idw':
            interpolated_values = self._idw_interpolation(
                coords, values, target_coords
            )
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
        
        return interpolated_values.tolist()
    
    def _scipy_interpolation(self, coords: np.ndarray, values: np.ndarray,
                           target_points: np.ndarray, method: str) -> np.ndarray:
        """Perform interpolation using scipy.interpolate.griddata."""
        try:
            interpolated = griddata(
                coords, values, target_points, method=method, fill_value=np.nan
            )
            return interpolated
        except Exception as e:
            logger.warning(f"Scipy interpolation failed: {e}")
            # Fall back to nearest neighbor
            return griddata(coords, values, target_points, method='nearest')
    
    def _rbf_interpolation(self, coords: np.ndarray, values: np.ndarray,
                          target_points: np.ndarray) -> np.ndarray:
        """Perform Radial Basis Function interpolation."""
        try:
            params = self.method_params['rbf']
            rbf = Rbf(coords[:, 0], coords[:, 1], values, 
                     function=params['function'], epsilon=params['epsilon'])
            interpolated = rbf(target_points[:, 0], target_points[:, 1])
            return interpolated
        except Exception as e:
            logger.warning(f"RBF interpolation failed: {e}")
            # Fall back to linear
            return self._scipy_interpolation(coords, values, target_points, 'linear')
    
    def _kriging_interpolation(self, coords: np.ndarray, values: np.ndarray,
                             target_points: np.ndarray) -> np.ndarray:
        """Perform Kriging interpolation using Gaussian Process."""
        try:
            kernel = self.method_params['kriging']['kernel']
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
            gp.fit(coords, values)
            
            interpolated, _ = gp.predict(target_points, return_std=True)
            return interpolated
        except Exception as e:
            logger.warning(f"Kriging interpolation failed: {e}")
            # Fall back to RBF
            return self._rbf_interpolation(coords, values, target_points)
    
    def _idw_interpolation(self, coords: np.ndarray, values: np.ndarray,
                          target_points: np.ndarray) -> np.ndarray:
        """Perform Inverse Distance Weighting interpolation."""
        try:
            power = self.method_params['idw']['power']
            
            # Calculate distances from each target point to all sensor points
            distances = cdist(target_points, coords)
            
            # Avoid division by zero for exact matches
            distances = np.maximum(distances, 1e-10)
            
            # Calculate weights (inverse distance with power)
            weights = 1.0 / (distances ** power)
            
            # Normalize weights
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            weights_normalized = weights / weights_sum
            
            # Calculate interpolated values
            interpolated = np.sum(weights_normalized * values, axis=1)
            
            return interpolated
        except Exception as e:
            logger.warning(f"IDW interpolation failed: {e}")
            # Fall back to nearest neighbor
            return self._scipy_interpolation(coords, values, target_points, 'nearest')
    
    def _calculate_quality_mask(self, sensor_coords: np.ndarray,
                              grid_points: np.ndarray,
                              grid_shape: Tuple[int, int]) -> np.ndarray:
        """Calculate quality mask based on distance to nearest sensors."""
        # Calculate distance from each grid point to nearest sensor
        distances = cdist(grid_points, sensor_coords)
        min_distances = np.min(distances, axis=1)
        
        # Create quality mask (1.0 = high quality, 0.0 = low quality)
        # Quality decreases with distance from sensors
        max_distance = np.percentile(min_distances, 95)  # Use 95th percentile as max
        quality = 1.0 - (min_distances / max_distance)
        quality = np.clip(quality, 0.0, 1.0)
        
        return quality.reshape(grid_shape)
    
    def validate_interpolation(self, readings: List[SensorReading],
                             method: str = 'linear',
                             validation_fraction: float = 0.2) -> Dict[str, float]:
        """
        Validate interpolation accuracy using cross-validation.
        
        Args:
            readings: List of sensor readings
            method: Interpolation method to validate
            validation_fraction: Fraction of readings to use for validation
            
        Returns:
            Dictionary with validation metrics
        """
        if len(readings) < 5:
            return {'error': 'Insufficient readings for validation'}
        
        # Filter readings with valid coordinates
        valid_readings = [r for r in readings if r.latitude is not None and r.longitude is not None]
        
        if len(valid_readings) < 5:
            return {'error': 'Insufficient readings with coordinates for validation'}
        
        # Randomly split into training and validation sets
        np.random.shuffle(valid_readings)
        n_validation = max(1, int(len(valid_readings) * validation_fraction))
        
        training_readings = valid_readings[:-n_validation]
        validation_readings = valid_readings[-n_validation:]
        
        # Get validation points and true values
        validation_points = [(r.longitude, r.latitude) for r in validation_readings]
        true_values = [r.value for r in validation_readings]
        
        # Interpolate to validation points using training data
        try:
            predicted_values = self.interpolate_to_points(
                training_readings, validation_points, method
            )
            
            # Calculate validation metrics
            true_values = np.array(true_values)
            predicted_values = np.array(predicted_values)
            
            # Remove NaN predictions
            valid_mask = ~np.isnan(predicted_values)
            if np.sum(valid_mask) == 0:
                return {'error': 'All predictions are NaN'}
            
            true_values = true_values[valid_mask]
            predicted_values = predicted_values[valid_mask]
            
            # Calculate metrics
            mae = np.mean(np.abs(true_values - predicted_values))
            rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
            
            # R-squared
            ss_res = np.sum((true_values - predicted_values) ** 2)
            ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r_squared': r_squared,
                'n_validation_points': len(true_values),
                'method': method
            }
            
        except Exception as e:
            return {'error': f'Validation failed: {str(e)}'}
    
    def get_optimal_method(self, readings: List[SensorReading]) -> str:
        """
        Determine the optimal interpolation method for given sensor readings.
        
        Args:
            readings: List of sensor readings
            
        Returns:
            Name of the optimal interpolation method
        """
        if len(readings) < 5:
            return 'nearest'
        
        # Test different methods
        methods_to_test = ['linear', 'rbf', 'idw']
        if len(readings) >= 10:
            methods_to_test.append('kriging')
        
        best_method = 'linear'
        best_score = float('inf')
        
        for method in methods_to_test:
            try:
                validation_result = self.validate_interpolation(readings, method)
                if 'rmse' in validation_result:
                    if validation_result['rmse'] < best_score:
                        best_score = validation_result['rmse']
                        best_method = method
            except Exception as e:
                logger.warning(f"Method {method} failed validation: {e}")
                continue
        
        return best_method