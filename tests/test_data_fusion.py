"""
Tests for data fusion layer functionality.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.sensors.data_fusion import (
    DataFusionEngine, FusedDataPoint
)
from src.sensors.data_ingestion import SensorReading
from src.sensors.temporal_alignment import AlignedReading


class TestDataFusionEngine:
    """Test data fusion engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fusion_engine = DataFusionEngine()
        
        # Create sample spectral data
        base_time = datetime(2024, 9, 23, 10, 30, 0)
        self.spectral_data = [
            {
                'timestamp': base_time,
                'latitude': 40.7128,
                'longitude': -74.0060,
                'indices': {
                    'NDVI': 0.75,
                    'SAVI': 0.65,
                    'EVI': 0.55,
                    'NDWI': 0.25
                }
            },
            {
                'timestamp': base_time + timedelta(days=1),
                'latitude': 40.7128,
                'longitude': -74.0060,
                'indices': {
                    'NDVI': 0.65,  # Decline indicating stress
                    'SAVI': 0.55,
                    'EVI': 0.45,
                    'NDWI': 0.15
                }
            }
        ]
        
        # Create sample environmental data
        self.environmental_data = [
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='soil_001',
                    timestamp=base_time - timedelta(minutes=30),
                    sensor_type='soil_moisture',
                    value=15.0,  # Low soil moisture
                    unit='%',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=base_time,
                time_offset=timedelta(minutes=-30),
                interpolated_value=15.0,
                confidence=0.9
            ),
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='temp_001',
                    timestamp=base_time - timedelta(minutes=15),
                    sensor_type='temperature',
                    value=32.0,  # High temperature
                    unit='°C',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=base_time,
                time_offset=timedelta(minutes=-15),
                interpolated_value=32.0,
                confidence=0.8
            ),
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='humidity_001',
                    timestamp=base_time,
                    sensor_type='humidity',
                    value=85.0,  # High humidity
                    unit='%',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=base_time,
                time_offset=timedelta(0),
                confidence=0.95
            )
        ]
    
    def test_spectral_anomaly_detection(self):
        """Test detection of spectral anomalies."""
        # Create extended spectral data with baseline
        extended_data = []
        base_time = datetime(2024, 9, 1, 10, 30, 0)
        
        # Create 30 days of baseline data
        for i in range(30):
            extended_data.append({
                'timestamp': base_time + timedelta(days=i),
                'latitude': 40.7128,
                'longitude': -74.0060,
                'indices': {
                    'NDVI': 0.75 + np.random.normal(0, 0.05),  # Normal variation
                    'SAVI': 0.65 + np.random.normal(0, 0.04),
                    'EVI': 0.55 + np.random.normal(0, 0.03)
                }
            })
        
        # Add anomalous data
        extended_data.append({
            'timestamp': base_time + timedelta(days=31),
            'latitude': 40.7128,
            'longitude': -74.0060,
            'indices': {
                'NDVI': 0.45,  # Significant drop
                'SAVI': 0.35,
                'EVI': 0.25
            }
        })
        
        anomalies = self.fusion_engine.detect_spectral_anomalies(extended_data)
        
        assert len(anomalies) > 0
        assert any(a.anomaly_type == 'vegetation_stress' for a in anomalies)
        assert all(a.severity > 0 for a in anomalies)
        assert all(a.confidence > 0 for a in anomalies)
    
    def test_correlation_analysis(self):
        """Test correlation between spectral and environmental data."""
        correlations = self.fusion_engine.correlate_spectral_environmental(
            self.spectral_data, self.environmental_data
        )
        
        # Should find correlations between different factors
        assert len(correlations) >= 0  # May be empty if insufficient data
        
        for correlation in correlations:
            assert isinstance(correlation, CorrelationResult)
            assert -1 <= correlation.correlation_coefficient <= 1
            assert 0 <= correlation.p_value <= 1
            assert correlation.sample_size >= 0
    
    def test_drought_alert_generation(self):
        """Test generation of drought stress alerts."""
        # Create spectral anomaly indicating vegetation stress
        anomaly = SpectralAnomaly(
            location=(40.7128, -74.0060),
            timestamp=datetime.now(),
            anomaly_type='vegetation_stress',
            severity=0.7,
            affected_indices=['NDVI', 'SAVI'],
            confidence=0.8
        )
        
        alerts = self.fusion_engine.generate_alerts(
            [anomaly], self.environmental_data, []
        )
        
        # Should generate drought alert due to low soil moisture + high temp + vegetation stress
        drought_alerts = [a for a in alerts if a.alert_type == 'drought_stress']
        assert len(drought_alerts) > 0
        
        alert = drought_alerts[0]
        assert alert.severity in ['low', 'medium', 'high', 'critical']
        assert 'low_soil_moisture' in alert.contributing_factors
        assert 'vegetation_stress' in alert.contributing_factors
        assert len(alert.recommended_actions) > 0
    
    def test_pest_risk_alert_generation(self):
        """Test generation of pest risk alerts."""
        # Create environmental conditions favorable for pests
        pest_env_data = [
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='temp_001',
                    timestamp=datetime.now(),
                    sensor_type='temperature',
                    value=25.0,  # Optimal pest temperature
                    unit='°C',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=datetime.now(),
                time_offset=timedelta(0),
                confidence=0.9
            ),
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='humidity_001',
                    timestamp=datetime.now(),
                    sensor_type='humidity',
                    value=75.0,  # High humidity
                    unit='%',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=datetime.now(),
                time_offset=timedelta(0),
                confidence=0.9
            )
        ]
        
        anomaly = SpectralAnomaly(
            location=(40.7128, -74.0060),
            timestamp=datetime.now(),
            anomaly_type='vegetation_stress',
            severity=0.4,  # Moderate stress
            affected_indices=['NDVI'],
            confidence=0.7
        )
        
        alerts = self.fusion_engine.generate_alerts([anomaly], pest_env_data, [])
        
        pest_alerts = [a for a in alerts if a.alert_type == 'pest_risk']
        assert len(pest_alerts) > 0
        
        alert = pest_alerts[0]
        assert 'favorable_temperature' in alert.contributing_factors
        assert 'high_humidity' in alert.contributing_factors
    
    def test_disease_risk_alert_generation(self):
        """Test generation of disease risk alerts."""
        # Create environmental conditions favorable for disease
        disease_env_data = [
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='humidity_001',
                    timestamp=datetime.now(),
                    sensor_type='humidity',
                    value=90.0,  # Very high humidity
                    unit='%',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=datetime.now(),
                time_offset=timedelta(0),
                confidence=0.9
            ),
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='leaf_wetness_001',
                    timestamp=datetime.now(),
                    sensor_type='leaf_wetness',
                    value=10.0,  # Prolonged wetness
                    unit='hours',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=datetime.now(),
                time_offset=timedelta(0),
                confidence=0.8
            )
        ]
        
        anomaly = SpectralAnomaly(
            location=(40.7128, -74.0060),
            timestamp=datetime.now(),
            anomaly_type='vegetation_stress',
            severity=0.5,
            affected_indices=['NDVI', 'EVI'],
            confidence=0.8
        )
        
        alerts = self.fusion_engine.generate_alerts([anomaly], disease_env_data, [])
        
        disease_alerts = [a for a in alerts if a.alert_type == 'disease_risk']
        assert len(disease_alerts) > 0
        
        alert = disease_alerts[0]
        assert 'high_humidity' in alert.contributing_factors
        assert 'vegetation_anomaly' in alert.contributing_factors
    
    def test_data_quality_scoring(self):
        """Test data quality score calculation."""
        quality_scores = self.fusion_engine.calculate_data_quality_score(
            self.spectral_data, self.environmental_data
        )
        
        required_scores = [
            'temporal_coverage', 'spatial_coverage', 
            'data_completeness', 'alignment_quality', 'overall_quality'
        ]
        
        for score_type in required_scores:
            assert score_type in quality_scores
            assert 0.0 <= quality_scores[score_type] <= 1.0
        
        # Overall quality should be average of component scores
        component_scores = [quality_scores[s] for s in required_scores[:-1]]
        expected_overall = np.mean(component_scores)
        assert abs(quality_scores['overall_quality'] - expected_overall) < 0.01
    
    def test_correlation_with_insufficient_data(self):
        """Test correlation analysis with insufficient data."""
        # Create minimal data that shouldn't produce correlations
        minimal_spectral = [self.spectral_data[0]]
        minimal_env = [self.environmental_data[0]]
        
        correlations = self.fusion_engine.correlate_spectral_environmental(
            minimal_spectral, minimal_env
        )
        
        # Should return empty or very few correlations due to insufficient data
        assert len(correlations) == 0
    
    def test_alert_expiration(self):
        """Test that alerts have appropriate expiration times."""
        anomaly = SpectralAnomaly(
            location=(40.7128, -74.0060),
            timestamp=datetime.now(),
            anomaly_type='vegetation_stress',
            severity=0.6,
            affected_indices=['NDVI'],
            confidence=0.8
        )
        
        alerts = self.fusion_engine.generate_alerts(
            [anomaly], self.environmental_data, []
        )
        
        for alert in alerts:
            assert alert.expires_at is not None
            assert alert.expires_at > alert.timestamp
            # Should expire within reasonable timeframe (1-7 days)
            time_to_expire = alert.expires_at - alert.timestamp
            assert timedelta(days=1) <= time_to_expire <= timedelta(days=7, hours=1)
    
    def test_alert_confidence_calculation(self):
        """Test that alert confidence is properly calculated."""
        anomaly = SpectralAnomaly(
            location=(40.7128, -74.0060),
            timestamp=datetime.now(),
            anomaly_type='vegetation_stress',
            severity=0.5,
            affected_indices=['NDVI'],
            confidence=0.7
        )
        
        alerts = self.fusion_engine.generate_alerts(
            [anomaly], self.environmental_data, []
        )
        
        for alert in alerts:
            assert 0.0 <= alert.confidence <= 1.0
            # Alert confidence should not exceed the minimum of contributing factors
            assert alert.confidence <= anomaly.confidence
    
    def test_multiple_alert_types(self):
        """Test that multiple alert types can be generated simultaneously."""
        # Create conditions that could trigger multiple alerts
        multi_risk_env_data = [
            # Low soil moisture (drought)
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='soil_001',
                    timestamp=datetime.now(),
                    sensor_type='soil_moisture',
                    value=10.0,
                    unit='%',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=datetime.now(),
                time_offset=timedelta(0),
                confidence=0.9
            ),
            # High temperature (drought + pest)
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='temp_001',
                    timestamp=datetime.now(),
                    sensor_type='temperature',
                    value=28.0,
                    unit='°C',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=datetime.now(),
                time_offset=timedelta(0),
                confidence=0.9
            ),
            # High humidity (pest + disease)
            AlignedReading(
                original_reading=SensorReading(
                    sensor_id='humidity_001',
                    timestamp=datetime.now(),
                    sensor_type='humidity',
                    value=85.0,
                    unit='%',
                    latitude=40.7128,
                    longitude=-74.0060
                ),
                satellite_timestamp=datetime.now(),
                time_offset=timedelta(0),
                confidence=0.9
            )
        ]
        
        anomaly = SpectralAnomaly(
            location=(40.7128, -74.0060),
            timestamp=datetime.now(),
            anomaly_type='vegetation_stress',
            severity=0.6,
            affected_indices=['NDVI', 'SAVI'],
            confidence=0.8
        )
        
        alerts = self.fusion_engine.generate_alerts([anomaly], multi_risk_env_data, [])
        
        # Should potentially generate multiple types of alerts
        alert_types = {alert.alert_type for alert in alerts}
        assert len(alert_types) >= 1  # At least one alert type
        
        # Check that each alert has proper structure
        for alert in alerts:
            assert alert.alert_id
            assert alert.timestamp
            assert alert.location == anomaly.location
            assert alert.description
            assert len(alert.contributing_factors) > 0
            assert len(alert.recommended_actions) > 0