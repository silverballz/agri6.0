"""
Data validation functions for sensor readings.

This module provides comprehensive validation for environmental sensor data
including range checks, quality flagging, and anomaly detection.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

from .data_ingestion import SensorReading

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of sensor data validation."""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    issues: List[str]
    corrected_value: Optional[float] = None
    recommended_flag: str = 'good'  # 'good', 'suspect', 'bad'


class SensorDataValidator:
    """Validates environmental sensor data for quality and consistency."""
    
    def __init__(self):
        # Define acceptable ranges for different sensor types
        self.sensor_ranges = {
            'soil_moisture': {'min': 0, 'max': 100, 'unit': '%'},
            'temperature': {'min': -50, 'max': 60, 'unit': '°C'},
            'humidity': {'min': 0, 'max': 100, 'unit': '%'},
            'leaf_wetness': {'min': 0, 'max': 1440, 'unit': 'minutes'},
            'solar_radiation': {'min': 0, 'max': 1500, 'unit': 'W/m²'},
            'precipitation': {'min': 0, 'max': 500, 'unit': 'mm'}
        }
        
        # Define typical rates of change (per hour)
        self.max_change_rates = {
            'soil_moisture': 10.0,  # % per hour
            'temperature': 15.0,    # °C per hour
            'humidity': 30.0,       # % per hour
            'leaf_wetness': 60.0,   # minutes per hour
            'solar_radiation': 200.0,  # W/m² per hour
            'precipitation': 50.0   # mm per hour
        }
    
    def validate_reading(self, reading: SensorReading, 
                        historical_readings: Optional[List[SensorReading]] = None) -> ValidationResult:
        """
        Validate a single sensor reading.
        
        Args:
            reading: The sensor reading to validate
            historical_readings: Previous readings from the same sensor for temporal validation
            
        Returns:
            ValidationResult with validation outcome
        """
        issues = []
        quality_score = 1.0
        corrected_value = None
        
        # Basic range validation
        range_result = self._validate_range(reading)
        if not range_result['valid']:
            issues.extend(range_result['issues'])
            quality_score *= 0.3  # More severe penalty for range violations
        
        # Temporal validation if historical data available
        if historical_readings:
            temporal_result = self._validate_temporal_consistency(reading, historical_readings)
            if not temporal_result['valid']:
                issues.extend(temporal_result['issues'])
                quality_score *= 0.5  # More severe penalty for temporal inconsistency
                if temporal_result.get('corrected_value'):
                    corrected_value = temporal_result['corrected_value']
        
        # Physical consistency validation
        physics_result = self._validate_physical_consistency(reading)
        if not physics_result['valid']:
            issues.extend(physics_result['issues'])
            quality_score *= 0.8
        
        # Determine overall validity and recommended flag
        is_valid = quality_score >= 0.6  # Stricter threshold for validity
        if quality_score >= 0.8:
            recommended_flag = 'good'
        elif quality_score >= 0.5:
            recommended_flag = 'suspect'
        else:
            recommended_flag = 'bad'
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            corrected_value=corrected_value,
            recommended_flag=recommended_flag
        )
    
    def validate_batch(self, readings: List[SensorReading]) -> List[ValidationResult]:
        """
        Validate a batch of sensor readings with cross-validation.
        
        Args:
            readings: List of sensor readings to validate
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        # Group readings by sensor_id for temporal validation
        sensor_groups = {}
        for reading in readings:
            if reading.sensor_id not in sensor_groups:
                sensor_groups[reading.sensor_id] = []
            sensor_groups[reading.sensor_id].append(reading)
        
        # Sort each group by timestamp
        for sensor_id in sensor_groups:
            sensor_groups[sensor_id].sort(key=lambda x: x.timestamp)
        
        # Validate each reading with its historical context
        for reading in readings:
            historical = [r for r in sensor_groups[reading.sensor_id] 
                         if r.timestamp < reading.timestamp]
            result = self.validate_reading(reading, historical)
            results.append(result)
        
        return results
    
    def _validate_range(self, reading: SensorReading) -> Dict[str, Any]:
        """Validate that sensor value is within acceptable range."""
        issues = []
        valid = True
        
        sensor_type = reading.sensor_type.lower()
        if sensor_type not in self.sensor_ranges:
            issues.append(f"Unknown sensor type: {sensor_type}")
            return {'valid': False, 'issues': issues}
        
        range_info = self.sensor_ranges[sensor_type]
        min_val, max_val = range_info['min'], range_info['max']
        
        if reading.value < min_val:
            issues.append(f"Value {reading.value} below minimum {min_val} for {sensor_type}")
            valid = False
        elif reading.value > max_val:
            issues.append(f"Value {reading.value} above maximum {max_val} for {sensor_type}")
            valid = False
        
        # Check for extreme outliers (beyond 3 standard deviations of typical range)
        range_span = max_val - min_val
        if reading.value < min_val - 0.1 * range_span or reading.value > max_val + 0.1 * range_span:
            issues.append(f"Value {reading.value} is an extreme outlier for {sensor_type}")
            valid = False
        
        return {'valid': valid, 'issues': issues}
    
    def _validate_temporal_consistency(self, reading: SensorReading, 
                                     historical_readings: List[SensorReading]) -> Dict[str, Any]:
        """Validate temporal consistency with previous readings."""
        issues = []
        valid = True
        corrected_value = None
        
        if not historical_readings:
            return {'valid': True, 'issues': []}
        
        # Get the most recent reading
        recent_reading = max(historical_readings, key=lambda x: x.timestamp)
        time_diff = (reading.timestamp - recent_reading.timestamp).total_seconds() / 3600  # hours
        
        if time_diff <= 0:
            issues.append("Reading timestamp is not after previous reading")
            return {'valid': False, 'issues': issues}
        
        # Check rate of change
        value_diff = abs(reading.value - recent_reading.value)
        rate_of_change = value_diff / time_diff if time_diff > 0 else float('inf')
        
        sensor_type = reading.sensor_type.lower()
        max_rate = self.max_change_rates.get(sensor_type, float('inf'))
        
        if rate_of_change > max_rate:
            issues.append(f"Rate of change {rate_of_change:.2f} exceeds maximum {max_rate} for {sensor_type}")
            valid = False
            
            # Suggest corrected value based on maximum allowed change
            max_change = max_rate * time_diff
            if reading.value > recent_reading.value:
                corrected_value = recent_reading.value + max_change
            else:
                corrected_value = recent_reading.value - max_change
        
        # Check for stuck sensor (no change over extended period)
        if len(historical_readings) >= 3:
            recent_values = [r.value for r in historical_readings[-3:]] + [reading.value]
            if len(set(recent_values)) == 1:  # All values identical
                total_time = (reading.timestamp - historical_readings[-3].timestamp).total_seconds() / 3600
                if total_time > 6:  # No change for more than 6 hours
                    issues.append("Sensor appears to be stuck (no change for extended period)")
                    valid = False
        
        return {'valid': valid, 'issues': issues, 'corrected_value': corrected_value}
    
    def _validate_physical_consistency(self, reading: SensorReading) -> Dict[str, Any]:
        """Validate physical consistency of the reading."""
        issues = []
        valid = True
        
        sensor_type = reading.sensor_type.lower()
        
        # Specific physical consistency checks
        if sensor_type == 'humidity' and reading.value > 100:
            issues.append("Humidity cannot exceed 100%")
            valid = False
        
        if sensor_type == 'soil_moisture' and reading.value < 0:
            issues.append("Soil moisture cannot be negative")
            valid = False
        
        if sensor_type == 'temperature':
            # Check for physically impossible temperatures
            if reading.value < -273.15:  # Absolute zero
                issues.append("Temperature below absolute zero")
                valid = False
            elif reading.value > 100 and reading.unit == '°C':
                # Very high temperatures are suspicious for agricultural sensors
                issues.append("Temperature unusually high for agricultural monitoring")
                valid = False
        
        if sensor_type == 'solar_radiation':
            # Solar radiation should be zero at night (rough check)
            hour = reading.timestamp.hour
            if 20 <= hour or hour <= 5:  # Rough night hours
                if reading.value > 50:  # Some threshold for moonlight/artificial light
                    issues.append("High solar radiation detected during night hours")
                    valid = False
        
        return {'valid': valid, 'issues': issues}
    
    def get_quality_statistics(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Calculate quality statistics for a batch of validation results."""
        if not validation_results:
            return {}
        
        total_readings = len(validation_results)
        valid_readings = sum(1 for r in validation_results if r.is_valid)
        
        quality_scores = [r.quality_score for r in validation_results]
        
        flag_counts = {}
        for result in validation_results:
            flag = result.recommended_flag
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        all_issues = []
        for result in validation_results:
            all_issues.extend(result.issues)
        
        return {
            'total_readings': total_readings,
            'valid_readings': valid_readings,
            'validity_rate': valid_readings / total_readings,
            'mean_quality_score': np.mean(quality_scores),
            'median_quality_score': np.median(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'flag_distribution': flag_counts,
            'common_issues': self._get_common_issues(all_issues)
        }
    
    def _get_common_issues(self, issues: List[str]) -> Dict[str, int]:
        """Get frequency count of common validation issues."""
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort by frequency
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))