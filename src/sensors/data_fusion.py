"""
Data fusion layer for correlating spectral anomalies with environmental conditions.

This module provides functionality to correlate satellite-derived spectral
anomalies with environmental sensor data and generate threshold-based alerts.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging

from .data_ingestion import SensorReading
from .temporal_alignment import AlignedReading

logger = logging.getLogger(__name__)


@dataclass
class SpectralAnomaly:
    """Represents a detected spectral anomaly."""
    location: Tuple[float, float]  # (latitude, longitude)
    timestamp: datetime
    anomaly_type: str  # 'vegetation_stress', 'soil_moisture_deficit', etc.
    severity: float  # 0.0 to 1.0
    affected_indices: List[str]  # ['NDVI', 'SAVI', etc.]
    confidence: float  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CorrelationResult:
    """Result of correlation analysis between spectral and environmental data."""
    correlation_coefficient: float
    p_value: float
    is_significant: bool
    environmental_factor: str
    spectral_indicator: str
    sample_size: int
    confidence_interval: Tuple[float, float]


@dataclass
class Alert:
    """Represents a generated alert."""
    alert_id: str
    timestamp: datetime
    alert_type: str  # 'pest_risk', 'drought_stress', 'disease_risk', etc.
    severity: str  # 'low', 'medium', 'high', 'critical'
    location: Tuple[float, float]
    description: str
    contributing_factors: List[str]
    recommended_actions: List[str]
    confidence: float
    expires_at: Optional[datetime] = None


class DataFusionEngine:
    """Handles fusion of spectral and environmental sensor data."""
    
    def __init__(self):
        # Correlation thresholds for different environmental factors
        self.correlation_thresholds = {
            'soil_moisture': {
                'NDVI': {'strong': 0.7, 'moderate': 0.5, 'weak': 0.3},
                'SAVI': {'strong': 0.6, 'moderate': 0.4, 'weak': 0.2},
                'NDWI': {'strong': 0.8, 'moderate': 0.6, 'weak': 0.4}
            },
            'temperature': {
                'NDVI': {'strong': -0.6, 'moderate': -0.4, 'weak': -0.2},
                'EVI': {'strong': -0.5, 'moderate': -0.3, 'weak': -0.1}
            },
            'humidity': {
                'NDVI': {'strong': 0.4, 'moderate': 0.3, 'weak': 0.2},
                'leaf_wetness': {'strong': 0.8, 'moderate': 0.6, 'weak': 0.4}
            }
        }
        
        # Alert generation thresholds
        self.alert_thresholds = {
            'drought_stress': {
                'soil_moisture_max': 20.0,  # %
                'ndvi_decline_threshold': 0.15,  # Relative decline
                'temperature_min': 30.0,  # °C
                'days_without_rain': 7
            },
            'pest_risk': {
                'temperature_range': (20.0, 30.0),  # °C
                'humidity_min': 60.0,  # %
                'leaf_wetness_min': 6.0,  # hours
                'vegetation_stress_threshold': 0.1
            },
            'disease_risk': {
                'humidity_min': 80.0,  # %
                'leaf_wetness_min': 8.0,  # hours
                'temperature_range': (15.0, 25.0),  # °C
                'ndvi_anomaly_threshold': 0.2
            }
        }
    
    def correlate_spectral_environmental(self, 
                                       spectral_data: List[Dict[str, Any]],
                                       environmental_data: List[AlignedReading],
                                       correlation_window: timedelta = timedelta(days=7)) -> List[CorrelationResult]:
        """
        Correlate spectral anomalies with environmental conditions.
        
        Args:
            spectral_data: List of spectral measurements with indices
            environmental_data: List of aligned environmental sensor readings
            correlation_window: Time window for correlation analysis
            
        Returns:
            List of correlation results
        """
        correlations = []
        
        # Group environmental data by sensor type
        env_groups = {}
        for reading in environmental_data:
            sensor_type = reading.original_reading.sensor_type
            if sensor_type not in env_groups:
                env_groups[sensor_type] = []
            env_groups[sensor_type].append(reading)
        
        # Extract spectral indices
        spectral_indices = self._extract_spectral_indices(spectral_data)
        
        # Perform correlation analysis for each combination
        for sensor_type, readings in env_groups.items():
            for index_name, index_values in spectral_indices.items():
                correlation = self._calculate_correlation(
                    readings, index_values, sensor_type, index_name, correlation_window
                )
                if correlation:
                    correlations.append(correlation)
        
        return correlations
    
    def detect_spectral_anomalies(self, 
                                spectral_data: List[Dict[str, Any]],
                                baseline_period: int = 30) -> List[SpectralAnomaly]:
        """
        Detect anomalies in spectral data using statistical methods.
        
        Args:
            spectral_data: List of spectral measurements
            baseline_period: Number of days to use for baseline calculation
            
        Returns:
            List of detected spectral anomalies
        """
        anomalies = []
        
        # Group data by location
        location_groups = {}
        for data in spectral_data:
            location = (data['latitude'], data['longitude'])
            if location not in location_groups:
                location_groups[location] = []
            location_groups[location].append(data)
        
        # Detect anomalies for each location
        for location, location_data in location_groups.items():
            location_anomalies = self._detect_location_anomalies(
                location_data, baseline_period
            )
            anomalies.extend(location_anomalies)
        
        return anomalies
    
    def generate_alerts(self, 
                       spectral_anomalies: List[SpectralAnomaly],
                       environmental_data: List[AlignedReading],
                       correlations: List[CorrelationResult]) -> List[Alert]:
        """
        Generate threshold-based alerts from fused data.
        
        Args:
            spectral_anomalies: Detected spectral anomalies
            environmental_data: Environmental sensor data
            correlations: Correlation analysis results
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Group environmental data by location and type
        env_by_location = self._group_environmental_by_location(environmental_data)
        
        # Check each anomaly against alert conditions
        for anomaly in spectral_anomalies:
            location_env_data = env_by_location.get(anomaly.location, {})
            
            # Check for different alert types
            drought_alert = self._check_drought_stress(anomaly, location_env_data)
            if drought_alert:
                alerts.append(drought_alert)
            
            pest_alert = self._check_pest_risk(anomaly, location_env_data)
            if pest_alert:
                alerts.append(pest_alert)
            
            disease_alert = self._check_disease_risk(anomaly, location_env_data)
            if disease_alert:
                alerts.append(disease_alert)
        
        return alerts
    
    def calculate_data_quality_score(self, 
                                   spectral_data: List[Dict[str, Any]],
                                   environmental_data: List[AlignedReading]) -> Dict[str, float]:
        """
        Calculate quality scores for fused datasets.
        
        Args:
            spectral_data: Spectral measurement data
            environmental_data: Environmental sensor data
            
        Returns:
            Dictionary with quality scores for different aspects
        """
        scores = {}
        
        # Temporal coverage score
        scores['temporal_coverage'] = self._calculate_temporal_coverage_score(
            spectral_data, environmental_data
        )
        
        # Spatial coverage score
        scores['spatial_coverage'] = self._calculate_spatial_coverage_score(
            spectral_data, environmental_data
        )
        
        # Data completeness score
        scores['data_completeness'] = self._calculate_completeness_score(
            spectral_data, environmental_data
        )
        
        # Alignment quality score
        scores['alignment_quality'] = self._calculate_alignment_quality_score(
            environmental_data
        )
        
        # Overall fusion quality
        scores['overall_quality'] = np.mean(list(scores.values()))
        
        return scores
    
    def _extract_spectral_indices(self, spectral_data: List[Dict[str, Any]]) -> Dict[str, List[Tuple[datetime, float, Tuple[float, float]]]]:
        """Extract spectral indices with timestamps and locations."""
        indices = {}
        
        for data in spectral_data:
            timestamp = data['timestamp']
            location = (data['latitude'], data['longitude'])
            
            for index_name, value in data.get('indices', {}).items():
                if index_name not in indices:
                    indices[index_name] = []
                indices[index_name].append((timestamp, value, location))
        
        return indices
    
    def _calculate_correlation(self, 
                             env_readings: List[AlignedReading],
                             spectral_values: List[Tuple[datetime, float, Tuple[float, float]]],
                             sensor_type: str,
                             index_name: str,
                             window: timedelta) -> Optional[CorrelationResult]:
        """Calculate correlation between environmental and spectral data."""
        # Align data within the correlation window
        aligned_pairs = []
        
        for env_reading in env_readings:
            env_time = env_reading.satellite_timestamp
            env_value = env_reading.interpolated_value or env_reading.original_reading.value
            env_location = (env_reading.original_reading.latitude, env_reading.original_reading.longitude)
            
            # Find spectral values within window and location
            for spec_time, spec_value, spec_location in spectral_values:
                time_diff = abs((spec_time - env_time).total_seconds())
                location_diff = self._calculate_distance(env_location, spec_location)
                
                if time_diff <= window.total_seconds() and location_diff <= 0.01:  # ~1km
                    aligned_pairs.append((env_value, spec_value))
        
        if len(aligned_pairs) < 5:  # Need minimum samples for correlation
            return None
        
        env_values, spec_values = zip(*aligned_pairs)
        
        # Calculate correlation
        correlation_coeff, p_value = stats.pearsonr(env_values, spec_values)
        
        # Calculate confidence interval
        n = len(aligned_pairs)
        confidence_interval = self._calculate_correlation_confidence_interval(
            correlation_coeff, n
        )
        
        return CorrelationResult(
            correlation_coefficient=correlation_coeff,
            p_value=p_value,
            is_significant=p_value < 0.05,
            environmental_factor=sensor_type,
            spectral_indicator=index_name,
            sample_size=n,
            confidence_interval=confidence_interval
        )
    
    def _detect_location_anomalies(self, 
                                 location_data: List[Dict[str, Any]],
                                 baseline_period: int) -> List[SpectralAnomaly]:
        """Detect anomalies for a specific location."""
        anomalies = []
        
        if len(location_data) < baseline_period:
            return anomalies
        
        # Sort by timestamp
        location_data.sort(key=lambda x: x['timestamp'])
        
        # Calculate baseline statistics for each index
        baseline_data = location_data[-baseline_period:]
        location = (location_data[0]['latitude'], location_data[0]['longitude'])
        
        for index_name in ['NDVI', 'SAVI', 'EVI', 'NDWI']:
            values = [d['indices'].get(index_name) for d in baseline_data if d['indices'].get(index_name) is not None]
            
            if len(values) < 10:  # Need sufficient baseline
                continue
            
            baseline_mean = np.mean(values)
            baseline_std = np.std(values)
            
            # Check recent values for anomalies
            recent_data = location_data[-5:]  # Last 5 measurements
            for data in recent_data:
                current_value = data['indices'].get(index_name)
                if current_value is None:
                    continue
                
                # Calculate z-score
                z_score = abs((current_value - baseline_mean) / baseline_std) if baseline_std > 0 else 0
                
                if z_score > 2.0:  # Significant anomaly
                    severity = min(1.0, z_score / 4.0)  # Scale to 0-1
                    confidence = min(1.0, z_score / 3.0)
                    
                    anomaly_type = self._classify_anomaly_type(index_name, current_value, baseline_mean)
                    
                    anomaly = SpectralAnomaly(
                        location=location,
                        timestamp=data['timestamp'],
                        anomaly_type=anomaly_type,
                        severity=severity,
                        affected_indices=[index_name],
                        confidence=confidence,
                        metadata={'z_score': z_score, 'baseline_mean': baseline_mean}
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _check_drought_stress(self, 
                            anomaly: SpectralAnomaly,
                            env_data: Dict[str, List[AlignedReading]]) -> Optional[Alert]:
        """Check for drought stress conditions."""
        thresholds = self.alert_thresholds['drought_stress']
        
        # Check soil moisture
        soil_moisture_readings = env_data.get('soil_moisture', [])
        if soil_moisture_readings:
            recent_moisture = soil_moisture_readings[-1]
            moisture_value = recent_moisture.interpolated_value or recent_moisture.original_reading.value
            
            if moisture_value <= thresholds['soil_moisture_max']:
                # Check temperature
                temp_readings = env_data.get('temperature', [])
                high_temp = False
                if temp_readings:
                    recent_temp = temp_readings[-1]
                    temp_value = recent_temp.interpolated_value or recent_temp.original_reading.value
                    high_temp = temp_value >= thresholds['temperature_min']
                
                # Check NDVI decline
                ndvi_decline = anomaly.anomaly_type == 'vegetation_stress' and 'NDVI' in anomaly.affected_indices
                
                if ndvi_decline and (high_temp or moisture_value <= 15.0):
                    severity = 'high' if moisture_value <= 10.0 else 'medium'
                    
                    return Alert(
                        alert_id=f"drought_{anomaly.location[0]:.4f}_{anomaly.location[1]:.4f}_{int(anomaly.timestamp.timestamp())}",
                        timestamp=datetime.now(),
                        alert_type='drought_stress',
                        severity=severity,
                        location=anomaly.location,
                        description=f"Drought stress detected: soil moisture {moisture_value:.1f}%, NDVI decline",
                        contributing_factors=['low_soil_moisture', 'vegetation_stress', 'high_temperature'] if high_temp else ['low_soil_moisture', 'vegetation_stress'],
                        recommended_actions=['increase_irrigation', 'monitor_soil_moisture', 'consider_drought_resistant_varieties'],
                        confidence=min(anomaly.confidence, recent_moisture.confidence),
                        expires_at=datetime.now() + timedelta(days=3)
                    )
        
        return None
    
    def _check_pest_risk(self, 
                       anomaly: SpectralAnomaly,
                       env_data: Dict[str, List[AlignedReading]]) -> Optional[Alert]:
        """Check for pest risk conditions."""
        thresholds = self.alert_thresholds['pest_risk']
        
        temp_readings = env_data.get('temperature', [])
        humidity_readings = env_data.get('humidity', [])
        
        if not temp_readings or not humidity_readings:
            return None
        
        recent_temp = temp_readings[-1]
        recent_humidity = humidity_readings[-1]
        
        temp_value = recent_temp.interpolated_value or recent_temp.original_reading.value
        humidity_value = recent_humidity.interpolated_value or recent_humidity.original_reading.value
        
        # Check temperature range
        temp_in_range = thresholds['temperature_range'][0] <= temp_value <= thresholds['temperature_range'][1]
        high_humidity = humidity_value >= thresholds['humidity_min']
        
        if temp_in_range and high_humidity and anomaly.severity > thresholds['vegetation_stress_threshold']:
            severity = 'high' if humidity_value >= 80.0 and anomaly.severity > 0.3 else 'medium'
            
            return Alert(
                alert_id=f"pest_{anomaly.location[0]:.4f}_{anomaly.location[1]:.4f}_{int(anomaly.timestamp.timestamp())}",
                timestamp=datetime.now(),
                alert_type='pest_risk',
                severity=severity,
                location=anomaly.location,
                description=f"Pest risk conditions: temp {temp_value:.1f}°C, humidity {humidity_value:.1f}%",
                contributing_factors=['favorable_temperature', 'high_humidity', 'vegetation_stress'],
                recommended_actions=['increase_monitoring', 'consider_preventive_treatment', 'check_for_pest_signs'],
                confidence=min(anomaly.confidence, recent_temp.confidence, recent_humidity.confidence),
                expires_at=datetime.now() + timedelta(days=5)
            )
        
        return None
    
    def _check_disease_risk(self, 
                          anomaly: SpectralAnomaly,
                          env_data: Dict[str, List[AlignedReading]]) -> Optional[Alert]:
        """Check for disease risk conditions."""
        thresholds = self.alert_thresholds['disease_risk']
        
        humidity_readings = env_data.get('humidity', [])
        leaf_wetness_readings = env_data.get('leaf_wetness', [])
        
        if not humidity_readings:
            return None
        
        recent_humidity = humidity_readings[-1]
        humidity_value = recent_humidity.interpolated_value or recent_humidity.original_reading.value
        
        high_humidity = humidity_value >= thresholds['humidity_min']
        prolonged_wetness = False
        
        if leaf_wetness_readings:
            recent_wetness = leaf_wetness_readings[-1]
            wetness_value = recent_wetness.interpolated_value or recent_wetness.original_reading.value
            prolonged_wetness = wetness_value >= thresholds['leaf_wetness_min']
        
        if high_humidity and anomaly.severity > thresholds['ndvi_anomaly_threshold']:
            severity = 'high' if prolonged_wetness and humidity_value >= 90.0 else 'medium'
            
            return Alert(
                alert_id=f"disease_{anomaly.location[0]:.4f}_{anomaly.location[1]:.4f}_{int(anomaly.timestamp.timestamp())}",
                timestamp=datetime.now(),
                alert_type='disease_risk',
                severity=severity,
                location=anomaly.location,
                description=f"Disease risk conditions: humidity {humidity_value:.1f}%, vegetation anomaly detected",
                contributing_factors=['high_humidity', 'vegetation_anomaly'] + (['prolonged_leaf_wetness'] if prolonged_wetness else []),
                recommended_actions=['increase_air_circulation', 'monitor_for_disease_symptoms', 'consider_fungicide_application'],
                confidence=min(anomaly.confidence, recent_humidity.confidence),
                expires_at=datetime.now() + timedelta(days=7)
            )
        
        return None
    
    def _group_environmental_by_location(self, 
                                       environmental_data: List[AlignedReading]) -> Dict[Tuple[float, float], Dict[str, List[AlignedReading]]]:
        """Group environmental data by location and sensor type."""
        grouped = {}
        
        for reading in environmental_data:
            location = (reading.original_reading.latitude, reading.original_reading.longitude)
            sensor_type = reading.original_reading.sensor_type
            
            if location not in grouped:
                grouped[location] = {}
            if sensor_type not in grouped[location]:
                grouped[location][sensor_type] = []
            
            grouped[location][sensor_type].append(reading)
        
        return grouped
    
    def _classify_anomaly_type(self, index_name: str, current_value: float, baseline_mean: float) -> str:
        """Classify the type of spectral anomaly."""
        if current_value < baseline_mean:
            if index_name in ['NDVI', 'SAVI', 'EVI']:
                return 'vegetation_stress'
            elif index_name == 'NDWI':
                return 'water_stress'
        else:
            if index_name in ['NDVI', 'SAVI', 'EVI']:
                return 'vegetation_enhancement'
            elif index_name == 'NDWI':
                return 'water_excess'
        
        return 'unknown_anomaly'
    
    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate approximate distance between two lat/lon points in degrees."""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def _calculate_correlation_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient."""
        if n < 3:
            return (r, r)
        
        # Fisher z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        # Critical value for 95% confidence
        z_critical = 1.96 if confidence == 0.95 else 2.576
        
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        # Transform back to correlation scale
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _calculate_temporal_coverage_score(self, 
                                         spectral_data: List[Dict[str, Any]],
                                         environmental_data: List[AlignedReading]) -> float:
        """Calculate temporal coverage quality score."""
        if not spectral_data or not environmental_data:
            return 0.0
        
        # Get time ranges
        spec_times = [d['timestamp'] for d in spectral_data]
        env_times = [r.satellite_timestamp for r in environmental_data]
        
        spec_range = max(spec_times) - min(spec_times)
        env_range = max(env_times) - min(env_times)
        
        # Calculate overlap
        overlap_start = max(min(spec_times), min(env_times))
        overlap_end = min(max(spec_times), max(env_times))
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        total_duration = max(spec_range, env_range)
        
        return min(1.0, overlap_duration.total_seconds() / total_duration.total_seconds())
    
    def _calculate_spatial_coverage_score(self, 
                                        spectral_data: List[Dict[str, Any]],
                                        environmental_data: List[AlignedReading]) -> float:
        """Calculate spatial coverage quality score."""
        if not spectral_data or not environmental_data:
            return 0.0
        
        # Get spatial extents
        spec_lats = [d['latitude'] for d in spectral_data]
        spec_lons = [d['longitude'] for d in spectral_data]
        
        env_lats = [r.original_reading.latitude for r in environmental_data if r.original_reading.latitude]
        env_lons = [r.original_reading.longitude for r in environmental_data if r.original_reading.longitude]
        
        if not env_lats or not env_lons:
            return 0.0
        
        # Calculate overlap area (simplified)
        spec_area = (max(spec_lats) - min(spec_lats)) * (max(spec_lons) - min(spec_lons))
        env_area = (max(env_lats) - min(env_lats)) * (max(env_lons) - min(env_lons))
        
        if spec_area == 0 or env_area == 0:
            return 0.0
        
        # Rough overlap calculation
        overlap_score = min(1.0, min(spec_area, env_area) / max(spec_area, env_area))
        
        return overlap_score
    
    def _calculate_completeness_score(self, 
                                    spectral_data: List[Dict[str, Any]],
                                    environmental_data: List[AlignedReading]) -> float:
        """Calculate data completeness score."""
        if not spectral_data or not environmental_data:
            return 0.0
        
        # Check spectral data completeness
        spec_completeness = 0.0
        if spectral_data:
            total_indices = len(['NDVI', 'SAVI', 'EVI', 'NDWI'])
            available_indices = 0
            for data in spectral_data:
                available_indices += len([idx for idx in ['NDVI', 'SAVI', 'EVI', 'NDWI'] 
                                        if data.get('indices', {}).get(idx) is not None])
            spec_completeness = available_indices / (len(spectral_data) * total_indices)
        
        # Check environmental data completeness
        env_completeness = 0.0
        if environmental_data:
            valid_readings = sum(1 for r in environmental_data if r.interpolated_value is not None or r.original_reading.value is not None)
            env_completeness = valid_readings / len(environmental_data)
        
        return (spec_completeness + env_completeness) / 2
    
    def _calculate_alignment_quality_score(self, environmental_data: List[AlignedReading]) -> float:
        """Calculate alignment quality score."""
        if not environmental_data:
            return 0.0
        
        confidence_scores = [r.confidence for r in environmental_data]
        return np.mean(confidence_scores)