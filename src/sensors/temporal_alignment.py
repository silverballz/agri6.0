"""
Temporal alignment with satellite overpass times.

This module provides functionality to align environmental sensor data
with Sentinel-2A satellite overpass times for synchronized analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .data_ingestion import SensorReading

logger = logging.getLogger(__name__)


@dataclass
class AlignedReading:
    """Sensor reading aligned to satellite overpass time."""
    original_reading: SensorReading
    satellite_timestamp: datetime
    time_offset: timedelta  # Difference between sensor and satellite time
    interpolated_value: Optional[float] = None
    confidence: float = 1.0  # Confidence in alignment (0.0 to 1.0)


class TemporalAligner:
    """Aligns sensor data with satellite overpass times."""
    
    def __init__(self):
        # Sentinel-2A typical overpass times for different regions
        # These are approximate local solar times
        self.default_overpass_times = {
            'morning': {'hour': 10, 'minute': 30},  # ~10:30 AM local solar time
            'afternoon': {'hour': 14, 'minute': 30}  # Some regions have afternoon passes
        }
        
        # Maximum time window for considering readings as "close" to overpass
        self.max_alignment_window = timedelta(hours=2)
        
        # Interpolation methods
        self.interpolation_methods = ['linear', 'nearest', 'cubic']
    
    def align_to_overpass(self, readings: List[SensorReading], 
                         overpass_times: List[datetime],
                         method: str = 'linear') -> List[AlignedReading]:
        """
        Align sensor readings to satellite overpass times.
        
        Args:
            readings: List of sensor readings to align
            overpass_times: List of satellite overpass timestamps
            method: Interpolation method ('linear', 'nearest', 'cubic')
            
        Returns:
            List of AlignedReading objects
        """
        if not readings or not overpass_times:
            return []
        
        aligned_readings = []
        
        # Group readings by sensor_id for individual alignment
        sensor_groups = self._group_by_sensor(readings)
        
        for sensor_id, sensor_readings in sensor_groups.items():
            # Sort readings by timestamp
            sensor_readings.sort(key=lambda x: x.timestamp)
            
            for overpass_time in overpass_times:
                aligned_reading = self._align_sensor_to_overpass(
                    sensor_readings, overpass_time, method
                )
                if aligned_reading:
                    aligned_readings.append(aligned_reading)
        
        return aligned_readings
    
    def find_closest_readings(self, readings: List[SensorReading], 
                            target_time: datetime,
                            max_window: Optional[timedelta] = None) -> List[SensorReading]:
        """
        Find sensor readings closest to a target time.
        
        Args:
            readings: List of sensor readings
            target_time: Target timestamp to find readings near
            max_window: Maximum time window to consider
            
        Returns:
            List of readings within the time window, sorted by proximity
        """
        if max_window is None:
            max_window = self.max_alignment_window
        
        # Filter readings within the time window
        close_readings = []
        for reading in readings:
            time_diff = abs((reading.timestamp - target_time).total_seconds())
            if time_diff <= max_window.total_seconds():
                close_readings.append((reading, time_diff))
        
        # Sort by time difference and return readings
        close_readings.sort(key=lambda x: x[1])
        return [reading for reading, _ in close_readings]
    
    def interpolate_to_time(self, readings: List[SensorReading], 
                          target_time: datetime,
                          method: str = 'linear') -> Optional[Tuple[float, float]]:
        """
        Interpolate sensor value to a specific target time.
        
        Args:
            readings: List of sensor readings from the same sensor
            target_time: Target time for interpolation
            method: Interpolation method
            
        Returns:
            Tuple of (interpolated_value, confidence) or None if not possible
        """
        if len(readings) < 2:
            if len(readings) == 1:
                # Use nearest value if only one reading available
                time_diff = abs((readings[0].timestamp - target_time).total_seconds())
                confidence = max(0.1, 1.0 - time_diff / 3600)  # Decay over 1 hour
                return readings[0].value, confidence
            return None
        
        # Sort readings by timestamp
        readings = sorted(readings, key=lambda x: x.timestamp)
        
        # Extract timestamps and values
        timestamps = [r.timestamp for r in readings]
        values = [r.value for r in readings]
        
        # Convert to pandas for easier interpolation
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        df.set_index('timestamp', inplace=True)
        
        # Add target time to the series
        df.loc[target_time] = np.nan
        df.sort_index(inplace=True)
        
        # Interpolate
        if method == 'linear':
            df['value'] = df['value'].interpolate(method='linear')
        elif method == 'nearest':
            df['value'] = df['value'].interpolate(method='nearest')
        elif method == 'cubic' and len(readings) >= 4:
            df['value'] = df['value'].interpolate(method='cubic')
        else:
            # Fall back to linear if cubic not possible
            df['value'] = df['value'].interpolate(method='linear')
        
        # Get interpolated value
        if target_time in df.index:
            interpolated_value = df.loc[target_time, 'value']
            
            # Calculate confidence based on temporal distance to nearest readings
            confidence = self._calculate_interpolation_confidence(
                readings, target_time, method
            )
            
            return float(interpolated_value), confidence
        
        return None
    
    def generate_overpass_times(self, start_date: datetime, end_date: datetime,
                              latitude: float, longitude: float,
                              overpass_type: str = 'morning') -> List[datetime]:
        """
        Generate approximate satellite overpass times for a location and date range.
        
        Args:
            start_date: Start date for overpass generation
            end_date: End date for overpass generation
            latitude: Latitude of the location
            longitude: Longitude of the location
            overpass_type: Type of overpass ('morning' or 'afternoon')
            
        Returns:
            List of estimated overpass times
        """
        overpass_times = []
        
        # Get base overpass time
        base_time = self.default_overpass_times.get(overpass_type, 
                                                   self.default_overpass_times['morning'])
        
        # Sentinel-2A has a 10-day repeat cycle
        repeat_cycle = 10
        
        current_date = start_date.replace(hour=base_time['hour'], 
                                        minute=base_time['minute'], 
                                        second=0, microsecond=0)
        
        while current_date <= end_date:
            # Adjust for longitude (rough approximation)
            # Each 15 degrees of longitude = 1 hour time difference
            time_adjustment = timedelta(minutes=int(longitude * 4))  # 4 minutes per degree
            adjusted_time = current_date + time_adjustment
            
            overpass_times.append(adjusted_time)
            current_date += timedelta(days=repeat_cycle)
        
        return overpass_times
    
    def _group_by_sensor(self, readings: List[SensorReading]) -> Dict[str, List[SensorReading]]:
        """Group readings by sensor ID."""
        groups = {}
        for reading in readings:
            if reading.sensor_id not in groups:
                groups[reading.sensor_id] = []
            groups[reading.sensor_id].append(reading)
        return groups
    
    def _align_sensor_to_overpass(self, sensor_readings: List[SensorReading],
                                 overpass_time: datetime,
                                 method: str) -> Optional[AlignedReading]:
        """Align a single sensor's readings to an overpass time."""
        # Find readings within the alignment window
        close_readings = self.find_closest_readings(sensor_readings, overpass_time)
        
        if not close_readings:
            return None
        
        # If we have an exact match or very close reading, use it directly
        closest_reading = close_readings[0]
        time_offset = closest_reading.timestamp - overpass_time
        
        if abs(time_offset.total_seconds()) <= 300:  # Within 5 minutes
            return AlignedReading(
                original_reading=closest_reading,
                satellite_timestamp=overpass_time,
                time_offset=time_offset,
                confidence=1.0
            )
        
        # Otherwise, interpolate
        interpolation_result = self.interpolate_to_time(close_readings, overpass_time, method)
        
        if interpolation_result:
            interpolated_value, confidence = interpolation_result
            
            return AlignedReading(
                original_reading=closest_reading,  # Use closest as reference
                satellite_timestamp=overpass_time,
                time_offset=time_offset,
                interpolated_value=interpolated_value,
                confidence=confidence
            )
        
        return None
    
    def _calculate_interpolation_confidence(self, readings: List[SensorReading],
                                          target_time: datetime,
                                          method: str) -> float:
        """Calculate confidence score for interpolation."""
        if not readings:
            return 0.0
        
        # Find the closest readings before and after target time
        before_readings = [r for r in readings if r.timestamp <= target_time]
        after_readings = [r for r in readings if r.timestamp > target_time]
        
        if not before_readings or not after_readings:
            # Extrapolation - lower confidence
            closest = min(readings, key=lambda x: abs((x.timestamp - target_time).total_seconds()))
            time_diff = abs((closest.timestamp - target_time).total_seconds())
            return max(0.1, 1.0 - time_diff / 7200)  # Decay over 2 hours
        
        # Interpolation - higher confidence
        closest_before = max(before_readings, key=lambda x: x.timestamp)
        closest_after = min(after_readings, key=lambda x: x.timestamp)
        
        # Calculate time gaps
        gap_before = (target_time - closest_before.timestamp).total_seconds()
        gap_after = (closest_after.timestamp - target_time).total_seconds()
        total_gap = gap_before + gap_after
        
        # Confidence decreases with larger gaps
        base_confidence = max(0.3, 1.0 - total_gap / 7200)  # Decay over 2 hours
        
        # Adjust based on method
        if method == 'linear':
            return base_confidence
        elif method == 'cubic' and len(readings) >= 4:
            return base_confidence * 1.1  # Slight boost for cubic with enough points
        else:
            return base_confidence * 0.9  # Slight penalty for nearest neighbor
    
    def get_alignment_statistics(self, aligned_readings: List[AlignedReading]) -> Dict[str, Any]:
        """Calculate statistics for temporal alignment results."""
        if not aligned_readings:
            return {}
        
        time_offsets = [abs(ar.time_offset.total_seconds()) for ar in aligned_readings]
        confidences = [ar.confidence for ar in aligned_readings]
        
        interpolated_count = sum(1 for ar in aligned_readings if ar.interpolated_value is not None)
        
        return {
            'total_alignments': len(aligned_readings),
            'interpolated_readings': interpolated_count,
            'direct_readings': len(aligned_readings) - interpolated_count,
            'mean_time_offset_seconds': np.mean(time_offsets),
            'median_time_offset_seconds': np.median(time_offsets),
            'max_time_offset_seconds': np.max(time_offsets),
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences)
        }