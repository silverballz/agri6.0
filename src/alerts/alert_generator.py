"""
Alert Generation Module for AgriFlux Dashboard

Generates actionable alerts based on vegetation indices and environmental conditions.
Implements threshold-based rules for vegetation stress, pest risk, and other agricultural alerts.
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertType(Enum):
    """Types of alerts"""
    VEGETATION_STRESS = "vegetation_stress"
    PEST_RISK = "pest_risk"
    DISEASE_RISK = "disease_risk"
    WATER_STRESS = "water_stress"
    ENVIRONMENTAL = "environmental"


@dataclass
class Alert:
    """
    Alert data structure with enhanced contextual information.
    
    Attributes:
        alert_type: Type of alert
        severity: Severity level
        message: Human-readable alert message
        recommendation: Recommended action
        affected_area_percentage: Percentage of area affected (0-100)
        affected_area_geojson: Optional GeoJSON of affected area
        metadata: Additional metadata dictionary
        field_name: Optional field/location name
        coordinates: Optional coordinates (lat, lon) tuple
        historical_context: Optional historical comparison text
        rate_of_change: Optional rate of change value
        priority_score: Calculated priority score (0-100)
    """
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    recommendation: str
    affected_area_percentage: float
    affected_area_geojson: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    field_name: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    historical_context: Optional[str] = None
    rate_of_change: Optional[float] = None
    priority_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for database storage."""
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'recommendation': self.recommendation,
            'affected_area': self.affected_area_geojson,
            'metadata': json.dumps({
                **(self.metadata or {}),
                'field_name': self.field_name,
                'coordinates': self.coordinates,
                'historical_context': self.historical_context,
                'rate_of_change': self.rate_of_change,
                'priority_score': self.priority_score
            })
        }


class AlertGenerator:
    """
    Generates alerts based on vegetation indices and environmental conditions.
    
    Implements threshold-based rules for:
    - Vegetation stress (NDVI, SAVI, EVI)
    - Water stress (NDWI)
    - Pest risk (environmental conditions)
    - Disease risk (humidity, temperature, leaf wetness)
    """
    
    # NDVI thresholds for vegetation stress
    NDVI_CRITICAL = 0.3
    NDVI_HIGH_STRESS = 0.4
    NDVI_MEDIUM_STRESS = 0.5
    NDVI_LOW_STRESS = 0.6
    
    # SAVI thresholds (similar to NDVI but adjusted for soil)
    SAVI_CRITICAL = 0.25
    SAVI_HIGH_STRESS = 0.35
    SAVI_MEDIUM_STRESS = 0.45
    
    # EVI thresholds
    EVI_CRITICAL = 0.2
    EVI_HIGH_STRESS = 0.3
    EVI_MEDIUM_STRESS = 0.4
    
    # NDWI thresholds for water stress
    NDWI_CRITICAL = -0.2
    NDWI_HIGH_STRESS = -0.1
    NDWI_MEDIUM_STRESS = 0.0
    
    # Environmental thresholds for pest/disease risk
    TEMP_HIGH = 32.0  # Celsius
    TEMP_OPTIMAL_MIN = 20.0
    TEMP_OPTIMAL_MAX = 28.0
    
    HUMIDITY_LOW = 40.0  # Percentage
    HUMIDITY_HIGH = 80.0
    HUMIDITY_FUNGAL_RISK = 75.0
    
    # Affected area thresholds (percentage)
    AREA_CRITICAL_THRESHOLD = 30.0
    AREA_HIGH_THRESHOLD = 20.0
    AREA_MEDIUM_THRESHOLD = 10.0
    
    def __init__(self):
        """Initialize the alert generator."""
        self.historical_data = {}  # Store historical values for comparison
        logger.info("Initialized AlertGenerator")
    
    def calculate_priority_score(self, 
                                 severity: AlertSeverity,
                                 affected_area_pct: float,
                                 rate_of_change: Optional[float] = None) -> float:
        """
        Calculate priority score for alert ranking.
        
        Priority score is based on:
        - Severity level (40% weight)
        - Affected area percentage (30% weight)
        - Rate of change/trend (30% weight)
        
        Args:
            severity: Alert severity level
            affected_area_pct: Percentage of area affected (0-100)
            rate_of_change: Optional rate of change value
            
        Returns:
            Priority score from 0-100
        """
        # Severity scoring (0-40 points)
        severity_scores = {
            AlertSeverity.CRITICAL: 40,
            AlertSeverity.HIGH: 30,
            AlertSeverity.MEDIUM: 20,
            AlertSeverity.LOW: 10
        }
        severity_score = severity_scores.get(severity, 10)
        
        # Affected area scoring (0-30 points)
        # Linear scale: 0% = 0 points, 100% = 30 points
        area_score = min(30, (affected_area_pct / 100) * 30)
        
        # Rate of change scoring (0-30 points)
        # Higher absolute rate of change = higher score
        if rate_of_change is not None:
            # Normalize rate of change to 0-30 scale
            # Assume rate of change is in percentage points per day
            # Rapid changes (>5% per day) get max score
            rate_score = min(30, abs(rate_of_change) * 6)
        else:
            rate_score = 15  # Default middle score if no rate data
        
        total_score = severity_score + area_score + rate_score
        return round(total_score, 2)
    
    def add_historical_context(self,
                              current_value: float,
                              index_name: str,
                              field_name: Optional[str] = None) -> Optional[str]:
        """
        Generate historical context text by comparing to previous values.
        
        Args:
            current_value: Current index value
            index_name: Name of the index (e.g., 'NDVI')
            field_name: Optional field identifier
            
        Returns:
            Historical context string or None if no history available
        """
        key = f"{field_name or 'default'}_{index_name}"
        
        if key in self.historical_data:
            previous_value = self.historical_data[key]
            change = current_value - previous_value
            change_pct = (change / previous_value * 100) if previous_value != 0 else 0
            
            if abs(change_pct) < 2:
                context = f"{index_name} stable at {current_value:.3f} (no significant change from previous {previous_value:.3f})"
            elif change_pct > 0:
                context = f"{index_name} increased {abs(change_pct):.1f}% from {previous_value:.3f} to {current_value:.3f}"
            else:
                context = f"{index_name} dropped {abs(change_pct):.1f}% from {previous_value:.3f} to {current_value:.3f}"
            
            # Update historical data
            self.historical_data[key] = current_value
            return context
        else:
            # First time seeing this field/index combination
            self.historical_data[key] = current_value
            return None
    
    def calculate_rate_of_change(self,
                                 current_value: float,
                                 previous_value: float,
                                 days_elapsed: float = 1.0) -> float:
        """
        Calculate rate of change per day.
        
        Args:
            current_value: Current value
            previous_value: Previous value
            days_elapsed: Number of days between measurements
            
        Returns:
            Rate of change per day (percentage points)
        """
        if previous_value == 0 or days_elapsed == 0:
            return 0.0
        
        change = current_value - previous_value
        rate = (change / days_elapsed)
        return rate
    
    def generate_alerts(self,
                       ndvi: Optional[np.ndarray] = None,
                       savi: Optional[np.ndarray] = None,
                       evi: Optional[np.ndarray] = None,
                       ndwi: Optional[np.ndarray] = None,
                       temperature: Optional[float] = None,
                       humidity: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       field_name: Optional[str] = None,
                       coordinates: Optional[Tuple[float, float]] = None,
                       previous_values: Optional[Dict[str, float]] = None,
                       days_since_last: float = 7.0) -> List[Alert]:
        """
        Generate alerts based on vegetation indices and environmental data with enhanced context.
        
        Args:
            ndvi: NDVI array (optional)
            savi: SAVI array (optional)
            evi: EVI array (optional)
            ndwi: NDWI array (optional)
            temperature: Temperature in Celsius (optional)
            humidity: Relative humidity percentage (optional)
            metadata: Additional metadata (optional)
            field_name: Name/identifier of the field (optional)
            coordinates: (latitude, longitude) tuple (optional)
            previous_values: Dictionary of previous index values for comparison (optional)
            days_since_last: Days elapsed since last measurement (default: 7)
        
        Returns:
            List of Alert objects with enhanced contextual information
        """
        alerts = []
        
        # Generate vegetation stress alerts
        if ndvi is not None:
            alerts.extend(self._check_ndvi_stress(
                ndvi, metadata, field_name, coordinates, 
                previous_values.get('ndvi') if previous_values else None,
                days_since_last
            ))
        
        if savi is not None:
            alerts.extend(self._check_savi_stress(
                savi, metadata, field_name, coordinates,
                previous_values.get('savi') if previous_values else None,
                days_since_last
            ))
        
        if evi is not None:
            alerts.extend(self._check_evi_stress(
                evi, metadata, field_name, coordinates,
                previous_values.get('evi') if previous_values else None,
                days_since_last
            ))
        
        # Generate water stress alerts
        if ndwi is not None:
            alerts.extend(self._check_water_stress(
                ndwi, metadata, field_name, coordinates,
                previous_values.get('ndwi') if previous_values else None,
                days_since_last
            ))
        
        # Generate environmental alerts
        if temperature is not None or humidity is not None:
            alerts.extend(self._check_environmental_conditions(
                temperature, humidity, metadata, field_name, coordinates
            ))
        
        # Generate pest/disease risk alerts
        if temperature is not None and humidity is not None:
            alerts.extend(self._check_pest_disease_risk(
                temperature, humidity, metadata, field_name, coordinates
            ))
        
        logger.info(f"Generated {len(alerts)} alerts for field '{field_name or 'unknown'}'")
        return alerts
    
    def _check_ndvi_stress(self, ndvi: np.ndarray, 
                          metadata: Optional[Dict] = None,
                          field_name: Optional[str] = None,
                          coordinates: Optional[Tuple[float, float]] = None,
                          previous_ndvi: Optional[float] = None,
                          days_since_last: float = 7.0) -> List[Alert]:
        """Check for vegetation stress based on NDVI values with enhanced context."""
        alerts = []
        
        # Filter out invalid NDVI values (< -1 or > 1)
        valid_ndvi = ndvi[(ndvi >= -1) & (ndvi <= 1)]
        
        if valid_ndvi.size == 0:
            return alerts
        
        # Calculate statistics
        mean_ndvi = np.mean(valid_ndvi)
        
        # Calculate rate of change if previous value available
        rate_of_change = None
        historical_context = None
        if previous_ndvi is not None:
            rate_of_change = self.calculate_rate_of_change(mean_ndvi, previous_ndvi, days_since_last)
            historical_context = self.add_historical_context(mean_ndvi, 'NDVI', field_name)
        
        # Count pixels in each stress category
        critical_pixels = np.sum(valid_ndvi <= self.NDVI_CRITICAL)
        high_stress_pixels = np.sum((valid_ndvi > self.NDVI_CRITICAL) & 
                                    (valid_ndvi <= self.NDVI_HIGH_STRESS))
        medium_stress_pixels = np.sum((valid_ndvi > self.NDVI_HIGH_STRESS) & 
                                      (valid_ndvi <= self.NDVI_MEDIUM_STRESS))
        low_stress_pixels = np.sum((valid_ndvi > self.NDVI_MEDIUM_STRESS) & 
                                   (valid_ndvi <= self.NDVI_LOW_STRESS))
        
        total_pixels = valid_ndvi.size
        
        # Calculate percentages
        critical_pct = (critical_pixels / total_pixels) * 100
        high_stress_pct = (high_stress_pixels / total_pixels) * 100
        medium_stress_pct = (medium_stress_pixels / total_pixels) * 100
        low_stress_pct = (low_stress_pixels / total_pixels) * 100
        
        # Generate alerts based on severity and affected area
        if critical_pct > self.AREA_CRITICAL_THRESHOLD:
            severity = AlertSeverity.CRITICAL
            location_text = f" at {field_name}" if field_name else ""
            coord_text = f" (coordinates: {coordinates[0]:.4f}, {coordinates[1]:.4f})" if coordinates else ""
            history_text = f". {historical_context}" if historical_context else ""
            
            message = f"Severe vegetation stress detected{location_text}: {critical_pct:.1f}% of area has NDVI ≤ {self.NDVI_CRITICAL}{history_text}"
            
            alerts.append(Alert(
                alert_type=AlertType.VEGETATION_STRESS,
                severity=severity,
                message=message,
                recommendation="IMMEDIATE ACTION REQUIRED: Inspect affected areas for pest damage, disease, or irrigation failure. Consider emergency irrigation and soil testing.",
                affected_area_percentage=critical_pct,
                field_name=field_name,
                coordinates=coordinates,
                historical_context=historical_context,
                rate_of_change=rate_of_change,
                priority_score=self.calculate_priority_score(severity, critical_pct, rate_of_change),
                metadata={
                    'mean_ndvi': float(mean_ndvi),
                    'index_type': 'NDVI',
                    'threshold': self.NDVI_CRITICAL,
                    **(metadata or {})
                }
            ))
        elif critical_pct > 0:
            severity = AlertSeverity.HIGH
            location_text = f" at {field_name}" if field_name else ""
            coord_text = f" (coordinates: {coordinates[0]:.4f}, {coordinates[1]:.4f})" if coordinates else ""
            history_text = f". {historical_context}" if historical_context else ""
            
            message = f"Critical vegetation stress{location_text}: {critical_pct:.1f}% of area (NDVI ≤ {self.NDVI_CRITICAL}){history_text}"
            
            alerts.append(Alert(
                alert_type=AlertType.VEGETATION_STRESS,
                severity=severity,
                message=message,
                recommendation="Urgent inspection required. Check irrigation systems, test soil moisture, and look for signs of pest or disease damage.",
                affected_area_percentage=critical_pct,
                field_name=field_name,
                coordinates=coordinates,
                historical_context=historical_context,
                rate_of_change=rate_of_change,
                priority_score=self.calculate_priority_score(severity, critical_pct, rate_of_change),
                metadata={
                    'mean_ndvi': float(mean_ndvi),
                    'index_type': 'NDVI',
                    'threshold': self.NDVI_CRITICAL,
                    **(metadata or {})
                }
            ))
        
        if high_stress_pct > self.AREA_HIGH_THRESHOLD:
            severity = AlertSeverity.HIGH
            location_text = f" at {field_name}" if field_name else ""
            history_text = f". {historical_context}" if historical_context else ""
            
            alerts.append(Alert(
                alert_type=AlertType.VEGETATION_STRESS,
                severity=severity,
                message=f"High vegetation stress detected{location_text}: {high_stress_pct:.1f}% of area has NDVI between {self.NDVI_CRITICAL} and {self.NDVI_HIGH_STRESS}{history_text}",
                recommendation="Increase irrigation frequency, monitor closely for pest activity, and consider nutrient supplementation.",
                affected_area_percentage=high_stress_pct,
                field_name=field_name,
                coordinates=coordinates,
                historical_context=historical_context,
                rate_of_change=rate_of_change,
                priority_score=self.calculate_priority_score(severity, high_stress_pct, rate_of_change),
                metadata={
                    'mean_ndvi': float(mean_ndvi),
                    'index_type': 'NDVI',
                    'threshold_range': f"{self.NDVI_CRITICAL}-{self.NDVI_HIGH_STRESS}",
                    **(metadata or {})
                }
            ))
        
        if medium_stress_pct > self.AREA_MEDIUM_THRESHOLD:
            severity = AlertSeverity.MEDIUM
            location_text = f" at {field_name}" if field_name else ""
            history_text = f". {historical_context}" if historical_context else ""
            
            alerts.append(Alert(
                alert_type=AlertType.VEGETATION_STRESS,
                severity=severity,
                message=f"Moderate vegetation stress{location_text}: {medium_stress_pct:.1f}% of area has NDVI between {self.NDVI_HIGH_STRESS} and {self.NDVI_MEDIUM_STRESS}{history_text}",
                recommendation="Review irrigation schedule, monitor weather conditions, and prepare for potential intervention.",
                affected_area_percentage=medium_stress_pct,
                field_name=field_name,
                coordinates=coordinates,
                historical_context=historical_context,
                rate_of_change=rate_of_change,
                priority_score=self.calculate_priority_score(severity, medium_stress_pct, rate_of_change),
                metadata={
                    'mean_ndvi': float(mean_ndvi),
                    'index_type': 'NDVI',
                    'threshold_range': f"{self.NDVI_HIGH_STRESS}-{self.NDVI_MEDIUM_STRESS}",
                    **(metadata or {})
                }
            ))
        
        if low_stress_pct > self.AREA_MEDIUM_THRESHOLD:
            severity = AlertSeverity.LOW
            location_text = f" at {field_name}" if field_name else ""
            history_text = f". {historical_context}" if historical_context else ""
            
            alerts.append(Alert(
                alert_type=AlertType.VEGETATION_STRESS,
                severity=severity,
                message=f"Minor vegetation stress{location_text}: {low_stress_pct:.1f}% of area has NDVI between {self.NDVI_MEDIUM_STRESS} and {self.NDVI_LOW_STRESS}{history_text}",
                recommendation="Continue routine monitoring. Consider optimizing irrigation and fertilization schedules.",
                affected_area_percentage=low_stress_pct,
                field_name=field_name,
                coordinates=coordinates,
                historical_context=historical_context,
                rate_of_change=rate_of_change,
                priority_score=self.calculate_priority_score(severity, low_stress_pct, rate_of_change),
                metadata={
                    'mean_ndvi': float(mean_ndvi),
                    'index_type': 'NDVI',
                    'threshold_range': f"{self.NDVI_MEDIUM_STRESS}-{self.NDVI_LOW_STRESS}",
                    **(metadata or {})
                }
            ))
        
        return alerts
    
    def _check_savi_stress(self, savi: np.ndarray,
                          metadata: Optional[Dict] = None,
                          field_name: Optional[str] = None,
                          coordinates: Optional[Tuple[float, float]] = None,
                          previous_savi: Optional[float] = None,
                          days_since_last: float = 7.0) -> List[Alert]:
        """Check for vegetation stress based on SAVI values with enhanced context."""
        alerts = []
        
        # Filter valid SAVI values
        valid_savi = savi[(savi >= -1) & (savi <= 1)]
        
        if valid_savi.size == 0:
            return alerts
        
        mean_savi = np.mean(valid_savi)
        total_pixels = valid_savi.size
        
        # Calculate rate of change if previous value available
        rate_of_change = None
        historical_context = None
        if previous_savi is not None:
            rate_of_change = self.calculate_rate_of_change(mean_savi, previous_savi, days_since_last)
            historical_context = self.add_historical_context(mean_savi, 'SAVI', field_name)
        
        # Check critical stress
        critical_pixels = np.sum(valid_savi <= self.SAVI_CRITICAL)
        critical_pct = (critical_pixels / total_pixels) * 100
        
        if critical_pct > self.AREA_CRITICAL_THRESHOLD:
            severity = AlertSeverity.CRITICAL
            location_text = f" at {field_name}" if field_name else ""
            history_text = f". {historical_context}" if historical_context else ""
            
            alerts.append(Alert(
                alert_type=AlertType.VEGETATION_STRESS,
                severity=severity,
                message=f"Severe soil-adjusted vegetation stress{location_text}: {critical_pct:.1f}% of area has SAVI ≤ {self.SAVI_CRITICAL}{history_text}",
                recommendation="Critical soil and vegetation conditions detected. Immediate soil testing and irrigation assessment required.",
                affected_area_percentage=critical_pct,
                field_name=field_name,
                coordinates=coordinates,
                historical_context=historical_context,
                rate_of_change=rate_of_change,
                priority_score=self.calculate_priority_score(severity, critical_pct, rate_of_change),
                metadata={
                    'mean_savi': float(mean_savi),
                    'index_type': 'SAVI',
                    'threshold': self.SAVI_CRITICAL,
                    **(metadata or {})
                }
            ))
        
        return alerts
    
    def _check_evi_stress(self, evi: np.ndarray,
                         metadata: Optional[Dict] = None,
                         field_name: Optional[str] = None,
                         coordinates: Optional[Tuple[float, float]] = None,
                         previous_evi: Optional[float] = None,
                         days_since_last: float = 7.0) -> List[Alert]:
        """Check for vegetation stress based on EVI values."""
        alerts = []
        
        # Filter valid EVI values
        valid_evi = evi[(evi >= -1) & (evi <= 1)]
        
        if valid_evi.size == 0:
            return alerts
        
        mean_evi = np.mean(valid_evi)
        total_pixels = valid_evi.size
        
        # Check critical stress
        critical_pixels = np.sum(valid_evi <= self.EVI_CRITICAL)
        critical_pct = (critical_pixels / total_pixels) * 100
        
        if critical_pct > self.AREA_CRITICAL_THRESHOLD:
            alerts.append(Alert(
                alert_type=AlertType.VEGETATION_STRESS,
                severity=AlertSeverity.CRITICAL,
                message=f"Severe enhanced vegetation stress: {critical_pct:.1f}% of area has EVI ≤ {self.EVI_CRITICAL}",
                recommendation="Critical vegetation health detected. Comprehensive field assessment needed including canopy density and leaf area index.",
                affected_area_percentage=critical_pct,
                metadata={
                    'mean_evi': float(mean_evi),
                    'index_type': 'EVI',
                    'threshold': self.EVI_CRITICAL,
                    **(metadata or {})
                }
            ))
        
        return alerts
    
    def _check_water_stress(self, ndwi: np.ndarray,
                           metadata: Optional[Dict] = None,
                           field_name: Optional[str] = None,
                           coordinates: Optional[Tuple[float, float]] = None,
                           previous_ndwi: Optional[float] = None,
                           days_since_last: float = 7.0) -> List[Alert]:
        """Check for water stress based on NDWI values."""
        alerts = []
        
        # Filter valid NDWI values
        valid_ndwi = ndwi[(ndwi >= -1) & (ndwi <= 1)]
        
        if valid_ndwi.size == 0:
            return alerts
        
        mean_ndwi = np.mean(valid_ndwi)
        total_pixels = valid_ndwi.size
        
        # Count pixels in stress categories
        critical_pixels = np.sum(valid_ndwi <= self.NDWI_CRITICAL)
        high_stress_pixels = np.sum((valid_ndwi > self.NDWI_CRITICAL) & 
                                    (valid_ndwi <= self.NDWI_HIGH_STRESS))
        medium_stress_pixels = np.sum((valid_ndwi > self.NDWI_HIGH_STRESS) & 
                                      (valid_ndwi <= self.NDWI_MEDIUM_STRESS))
        
        critical_pct = (critical_pixels / total_pixels) * 100
        high_stress_pct = (high_stress_pixels / total_pixels) * 100
        medium_stress_pct = (medium_stress_pixels / total_pixels) * 100
        
        # Generate water stress alerts
        if critical_pct > self.AREA_CRITICAL_THRESHOLD:
            alerts.append(Alert(
                alert_type=AlertType.WATER_STRESS,
                severity=AlertSeverity.CRITICAL,
                message=f"Severe water stress: {critical_pct:.1f}% of area has NDWI ≤ {self.NDWI_CRITICAL}",
                recommendation="IMMEDIATE IRRIGATION REQUIRED. Check irrigation system functionality and increase watering frequency.",
                affected_area_percentage=critical_pct,
                metadata={
                    'mean_ndwi': float(mean_ndwi),
                    'index_type': 'NDWI',
                    'threshold': self.NDWI_CRITICAL,
                    **(metadata or {})
                }
            ))
        elif high_stress_pct > self.AREA_HIGH_THRESHOLD:
            alerts.append(Alert(
                alert_type=AlertType.WATER_STRESS,
                severity=AlertSeverity.HIGH,
                message=f"High water stress: {high_stress_pct:.1f}% of area has low water content (NDWI: {self.NDWI_CRITICAL} to {self.NDWI_HIGH_STRESS})",
                recommendation="Increase irrigation immediately. Monitor soil moisture levels and adjust watering schedule.",
                affected_area_percentage=high_stress_pct,
                metadata={
                    'mean_ndwi': float(mean_ndwi),
                    'index_type': 'NDWI',
                    'threshold_range': f"{self.NDWI_CRITICAL}-{self.NDWI_HIGH_STRESS}",
                    **(metadata or {})
                }
            ))
        elif medium_stress_pct > self.AREA_MEDIUM_THRESHOLD:
            alerts.append(Alert(
                alert_type=AlertType.WATER_STRESS,
                severity=AlertSeverity.MEDIUM,
                message=f"Moderate water stress: {medium_stress_pct:.1f}% of area showing reduced water content",
                recommendation="Review and optimize irrigation schedule. Consider soil moisture sensors for precision watering.",
                affected_area_percentage=medium_stress_pct,
                metadata={
                    'mean_ndwi': float(mean_ndwi),
                    'index_type': 'NDWI',
                    'threshold_range': f"{self.NDWI_HIGH_STRESS}-{self.NDWI_MEDIUM_STRESS}",
                    **(metadata or {})
                }
            ))
        
        return alerts
    
    def _check_environmental_conditions(self,
                                       temperature: Optional[float],
                                       humidity: Optional[float],
                                       metadata: Optional[Dict] = None,
                                       field_name: Optional[str] = None,
                                       coordinates: Optional[Tuple[float, float]] = None) -> List[Alert]:
        """Check environmental conditions for alerts."""
        alerts = []
        
        # Temperature alerts
        if temperature is not None:
            if temperature > self.TEMP_HIGH:
                alerts.append(Alert(
                    alert_type=AlertType.ENVIRONMENTAL,
                    severity=AlertSeverity.HIGH,
                    message=f"High temperature alert: {temperature:.1f}°C exceeds optimal range",
                    recommendation="Monitor crop stress closely. Consider shade structures or increased irrigation to mitigate heat stress.",
                    affected_area_percentage=100.0,  # Affects entire area
                    metadata={
                        'temperature': temperature,
                        'threshold': self.TEMP_HIGH,
                        **(metadata or {})
                    }
                ))
            elif temperature < self.TEMP_OPTIMAL_MIN:
                alerts.append(Alert(
                    alert_type=AlertType.ENVIRONMENTAL,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Low temperature alert: {temperature:.1f}°C below optimal range",
                    recommendation="Monitor for cold stress. Consider frost protection measures if temperature continues to drop.",
                    affected_area_percentage=100.0,
                    metadata={
                        'temperature': temperature,
                        'threshold': self.TEMP_OPTIMAL_MIN,
                        **(metadata or {})
                    }
                ))
        
        # Humidity alerts
        if humidity is not None:
            if humidity < self.HUMIDITY_LOW:
                alerts.append(Alert(
                    alert_type=AlertType.ENVIRONMENTAL,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Low humidity alert: {humidity:.1f}% may increase water stress",
                    recommendation="Increase irrigation frequency to compensate for high evapotranspiration rates.",
                    affected_area_percentage=100.0,
                    metadata={
                        'humidity': humidity,
                        'threshold': self.HUMIDITY_LOW,
                        **(metadata or {})
                    }
                ))
            elif humidity > self.HUMIDITY_HIGH:
                alerts.append(Alert(
                    alert_type=AlertType.ENVIRONMENTAL,
                    severity=AlertSeverity.MEDIUM,
                    message=f"High humidity alert: {humidity:.1f}% may increase disease risk",
                    recommendation="Monitor for fungal diseases. Ensure adequate air circulation and consider preventive fungicide application.",
                    affected_area_percentage=100.0,
                    metadata={
                        'humidity': humidity,
                        'threshold': self.HUMIDITY_HIGH,
                        **(metadata or {})
                    }
                ))
        
        return alerts
    
    def _check_pest_disease_risk(self,
                                temperature: float,
                                humidity: float,
                                metadata: Optional[Dict] = None,
                                field_name: Optional[str] = None,
                                coordinates: Optional[Tuple[float, float]] = None) -> List[Alert]:
        """Check for pest and disease risk based on environmental conditions."""
        alerts = []
        
        # Fungal disease risk (high humidity + moderate temperature)
        if humidity > self.HUMIDITY_FUNGAL_RISK and 20 <= temperature <= 30:
            severity = AlertSeverity.HIGH if humidity > 85 else AlertSeverity.MEDIUM
            alerts.append(Alert(
                alert_type=AlertType.DISEASE_RISK,
                severity=severity,
                message=f"High fungal disease risk: Humidity {humidity:.1f}% and temperature {temperature:.1f}°C favor fungal growth",
                recommendation="Inspect crops for early signs of fungal infection (leaf spots, mildew). Consider preventive fungicide application and improve air circulation.",
                affected_area_percentage=100.0,
                metadata={
                    'temperature': temperature,
                    'humidity': humidity,
                    'risk_type': 'fungal',
                    **(metadata or {})
                }
            ))
        
        # Pest risk (high temperature + moderate humidity)
        if temperature > 28 and 50 <= humidity <= 75:
            alerts.append(Alert(
                alert_type=AlertType.PEST_RISK,
                severity=AlertSeverity.MEDIUM,
                message=f"Elevated pest activity risk: Temperature {temperature:.1f}°C and humidity {humidity:.1f}% favor insect reproduction",
                recommendation="Increase pest monitoring frequency. Check for aphids, thrips, and other common pests. Consider integrated pest management strategies.",
                affected_area_percentage=100.0,
                metadata={
                    'temperature': temperature,
                    'humidity': humidity,
                    'risk_type': 'insect',
                    **(metadata or {})
                }
            ))
        
        # Bacterial disease risk (high temperature + high humidity)
        if temperature > 30 and humidity > 80:
            alerts.append(Alert(
                alert_type=AlertType.DISEASE_RISK,
                severity=AlertSeverity.HIGH,
                message=f"Bacterial disease risk: High temperature ({temperature:.1f}°C) and humidity ({humidity:.1f}%) create favorable conditions",
                recommendation="Monitor for bacterial wilt, leaf blight, and soft rot. Ensure proper sanitation and avoid overhead irrigation.",
                affected_area_percentage=100.0,
                metadata={
                    'temperature': temperature,
                    'humidity': humidity,
                    'risk_type': 'bacterial',
                    **(metadata or {})
                }
            ))
        
        return alerts
    
    def _create_alert_with_context(self,
                                   alert_type: AlertType,
                                   severity: AlertSeverity,
                                   base_message: str,
                                   recommendation: str,
                                   affected_area_pct: float,
                                   field_name: Optional[str],
                                   coordinates: Optional[Tuple[float, float]],
                                   historical_context: Optional[str],
                                   rate_of_change: Optional[float],
                                   metadata: Optional[Dict]) -> Alert:
        """
        Helper method to create an alert with full contextual information.
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            base_message: Base alert message
            recommendation: Recommended action
            affected_area_pct: Percentage of area affected
            field_name: Optional field name
            coordinates: Optional coordinates
            historical_context: Optional historical context text
            rate_of_change: Optional rate of change value
            metadata: Optional metadata dictionary
            
        Returns:
            Alert object with full context
        """
        # Enhance message with location and historical context
        location_text = f" at {field_name}" if field_name else ""
        history_text = f". {historical_context}" if historical_context else ""
        enhanced_message = f"{base_message}{location_text}{history_text}"
        
        return Alert(
            alert_type=alert_type,
            severity=severity,
            message=enhanced_message,
            recommendation=recommendation,
            affected_area_percentage=affected_area_pct,
            field_name=field_name,
            coordinates=coordinates,
            historical_context=historical_context,
            rate_of_change=rate_of_change,
            priority_score=self.calculate_priority_score(severity, affected_area_pct, rate_of_change),
            metadata=metadata
        )
    
    def rank_alerts_by_priority(self, alerts: List[Alert]) -> List[Alert]:
        """
        Rank alerts by priority score in descending order.
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            Sorted list of alerts (highest priority first)
        """
        return sorted(alerts, key=lambda a: a.priority_score, reverse=True)
    
    def get_top_priority_alerts(self, alerts: List[Alert], top_n: int = 5) -> List[Alert]:
        """
        Get the top N highest priority alerts.
        
        Args:
            alerts: List of Alert objects
            top_n: Number of top alerts to return (default: 5)
            
        Returns:
            List of top N alerts sorted by priority
        """
        ranked = self.rank_alerts_by_priority(alerts)
        return ranked[:top_n]
    
    def categorize_alerts(self, alerts: List[Alert]) -> Dict[str, List[Alert]]:
        """
        Categorize alerts into "Needs Attention" and "For Information" categories.
        
        Needs Attention: Critical/High severity OR priority score >= 60
        For Information: Medium/Low severity AND priority score < 60
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            Dictionary with 'needs_attention' and 'for_information' lists
        """
        needs_attention = []
        for_information = []
        
        for alert in alerts:
            # Critical and High severity always need attention
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                needs_attention.append(alert)
            # High priority score also needs attention
            elif alert.priority_score >= 60:
                needs_attention.append(alert)
            else:
                for_information.append(alert)
        
        # Sort each category by priority
        needs_attention = self.rank_alerts_by_priority(needs_attention)
        for_information = self.rank_alerts_by_priority(for_information)
        
        return {
            'needs_attention': needs_attention,
            'for_information': for_information
        }
    
    def get_alert_summary(self, alerts: List[Alert]) -> Dict[str, Any]:
        """
        Generate summary statistics for a list of alerts.
        
        Args:
            alerts: List of Alert objects
        
        Returns:
            Dictionary with alert summary statistics
        """
        if not alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_type': {},
                'by_category': {'needs_attention': 0, 'for_information': 0},
                'max_affected_area': 0.0,
                'avg_priority_score': 0.0,
                'top_5_alerts': []
            }
        
        # Categorize alerts
        categorized = self.categorize_alerts(alerts)
        
        summary = {
            'total_alerts': len(alerts),
            'by_severity': {},
            'by_type': {},
            'by_category': {
                'needs_attention': len(categorized['needs_attention']),
                'for_information': len(categorized['for_information'])
            },
            'max_affected_area': max(a.affected_area_percentage for a in alerts),
            'avg_priority_score': sum(a.priority_score for a in alerts) / len(alerts),
            'top_5_alerts': self.get_top_priority_alerts(alerts, 5)
        }
        
        # Count by severity
        for severity in AlertSeverity:
            count = sum(1 for a in alerts if a.severity == severity)
            if count > 0:
                summary['by_severity'][severity.value] = count
        
        # Count by type
        for alert_type in AlertType:
            count = sum(1 for a in alerts if a.alert_type == alert_type)
            if count > 0:
                summary['by_type'][alert_type.value] = count
        
        return summary
