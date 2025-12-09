"""
Synthetic sensor data generation module.

This module generates realistic synthetic environmental sensor data
correlated with satellite-derived vegetation indices for demonstration
and testing purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SyntheticSensorReading:
    """Represents a synthetic sensor reading."""
    timestamp: datetime
    location: Tuple[float, float]  # (latitude, longitude)
    sensor_type: str  # 'soil_moisture', 'temperature', 'humidity', 'leaf_wetness'
    value: float
    unit: str
    is_synthetic: bool = True
    correlation_source: Optional[str] = None  # e.g., 'ndvi_based'
    noise_level: float = 0.1
    metadata: Optional[Dict[str, Any]] = None


class SyntheticSensorGenerator:
    """
    Generate realistic synthetic sensor data correlated with satellite imagery.
    
    This class implements correlation algorithms and noise generation to create
    synthetic environmental sensor data that mimics real IoT sensor behavior
    while maintaining realistic correlations with vegetation indices.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the synthetic sensor generator.
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Correlation parameters
        self.correlation_params = {
            'soil_moisture': {
                'ndvi_correlation': 0.65,  # Target correlation coefficient
                'base_range': (10.0, 40.0),  # % moisture
                'noise_cv': 0.10  # Coefficient of variation
            },
            'temperature': {
                'seasonal_amplitude': 10.0,  # °C variation
                'base_temp': 25.0,  # °C average
                'daily_variation': 5.0,  # °C
                'noise_cv': 0.05
            },
            'humidity': {
                'temp_correlation': -0.45,  # Inverse correlation with temperature
                'moisture_influence': 0.3,  # Influence of soil moisture
                'base_range': (40.0, 90.0),  # %
                'noise_cv': 0.08
            },
            'leaf_wetness': {
                'humidity_threshold': 75.0,  # % humidity for wetness
                'optimal_temp_range': (20.0, 25.0),  # °C
                'noise_cv': 0.12
            }
        }
        
        # Temporal autocorrelation factor (0-1)
        self.temporal_autocorrelation = 0.7
        
        logger.info("SyntheticSensorGenerator initialized")
    
    def generate_soil_moisture(self,
                              ndvi_values: np.ndarray,
                              timestamps: List[datetime],
                              locations: List[Tuple[float, float]],
                              noise_level: Optional[float] = None) -> List[SyntheticSensorReading]:
        """
        Generate synthetic soil moisture data correlated with NDVI.
        
        Higher NDVI values indicate healthier vegetation, which typically
        correlates with adequate soil moisture.
        
        Args:
            ndvi_values: Array of NDVI values (range: -1 to 1)
            timestamps: List of timestamps for each reading
            locations: List of (lat, lon) tuples for each reading
            noise_level: Optional custom noise level (coefficient of variation)
            
        Returns:
            List of synthetic soil moisture readings
        """
        params = self.correlation_params['soil_moisture']
        noise_cv = noise_level if noise_level is not None else params['noise_cv']
        
        # Normalize NDVI to 0-1 range for correlation
        ndvi_normalized = (ndvi_values + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Base soil moisture correlated with NDVI
        # NDVI 0.3 -> ~15% moisture, NDVI 0.8 -> ~35% moisture
        min_moisture, max_moisture = params['base_range']
        base_moisture = min_moisture + ndvi_normalized * (max_moisture - min_moisture)
        
        # Add realistic noise
        noise = np.random.normal(0, noise_cv * base_moisture, size=len(base_moisture))
        moisture_values = np.clip(base_moisture + noise, 0, 50)
        
        # Add temporal variation (autocorrelation)
        moisture_values = self.add_temporal_variation(moisture_values)
        
        # Create readings
        readings = []
        for i, (timestamp, location, moisture) in enumerate(zip(timestamps, locations, moisture_values)):
            reading = SyntheticSensorReading(
                timestamp=timestamp,
                location=location,
                sensor_type='soil_moisture',
                value=float(moisture),
                unit='%',
                is_synthetic=True,
                correlation_source='ndvi_based',
                noise_level=noise_cv,
                metadata={
                    'ndvi_value': float(ndvi_values[i]),
                    'correlation_coefficient': params['ndvi_correlation']
                }
            )
            readings.append(reading)
        
        # Validate correlation
        actual_correlation = np.corrcoef(ndvi_values, moisture_values)[0, 1]
        logger.info(f"Generated soil moisture data with correlation to NDVI: {actual_correlation:.3f}")
        
        if actual_correlation < 0.5:
            logger.warning(f"Soil moisture correlation ({actual_correlation:.3f}) below target (0.5)")
        
        return readings
    
    def generate_temperature(self,
                           timestamps: List[datetime],
                           locations: List[Tuple[float, float]],
                           location_lat: float = 30.95,  # Ludhiana latitude
                           noise_level: Optional[float] = None) -> List[SyntheticSensorReading]:
        """
        Generate synthetic temperature data with seasonal patterns.
        
        Temperature follows a sinusoidal seasonal pattern based on day of year,
        with daily variation and realistic noise.
        
        Args:
            timestamps: List of timestamps for each reading
            locations: List of (lat, lon) tuples for each reading
            location_lat: Latitude for location-based adjustments
            noise_level: Optional custom noise level (coefficient of variation)
            
        Returns:
            List of synthetic temperature readings
        """
        params = self.correlation_params['temperature']
        noise_cv = noise_level if noise_level is not None else params['noise_cv']
        
        # Extract day of year for seasonal pattern
        day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])
        
        # Seasonal pattern (sine wave, peak around day 172 = June 21)
        seasonal = params['base_temp'] + params['seasonal_amplitude'] * np.sin(
            2 * np.pi * (day_of_year - 80) / 365
        )
        
        # Add daily variation
        daily_variation = np.random.normal(0, params['daily_variation'], size=len(timestamps))
        
        # Add noise
        noise = np.random.normal(0, noise_cv * seasonal, size=len(timestamps))
        
        # Combine components
        temperature_values = seasonal + daily_variation + noise
        temperature_values = np.clip(temperature_values, 5, 45)
        
        # Add temporal variation
        temperature_values = self.add_temporal_variation(temperature_values)
        
        # Create readings
        readings = []
        for timestamp, location, temp in zip(timestamps, locations, temperature_values):
            reading = SyntheticSensorReading(
                timestamp=timestamp,
                location=location,
                sensor_type='temperature',
                value=float(temp),
                unit='°C',
                is_synthetic=True,
                correlation_source='seasonal_pattern',
                noise_level=noise_cv,
                metadata={
                    'day_of_year': int(timestamp.timetuple().tm_yday),
                    'seasonal_component': float(seasonal[len(readings)]),
                    'location_lat': location_lat
                }
            )
            readings.append(reading)
        
        logger.info(f"Generated temperature data: mean={np.mean(temperature_values):.1f}°C, "
                   f"std={np.std(temperature_values):.1f}°C")
        
        return readings
    
    def generate_humidity(self,
                         temperature_values: np.ndarray,
                         soil_moisture_values: Optional[np.ndarray] = None,
                         timestamps: Optional[List[datetime]] = None,
                         locations: Optional[List[Tuple[float, float]]] = None,
                         noise_level: Optional[float] = None) -> List[SyntheticSensorReading]:
        """
        Generate synthetic humidity data inversely correlated with temperature.
        
        Humidity is inversely related to temperature and positively influenced
        by soil moisture.
        
        Args:
            temperature_values: Array of temperature values (°C)
            soil_moisture_values: Optional array of soil moisture values (%)
            timestamps: List of timestamps for each reading
            locations: List of (lat, lon) tuples for each reading
            noise_level: Optional custom noise level (coefficient of variation)
            
        Returns:
            List of synthetic humidity readings
        """
        params = self.correlation_params['humidity']
        noise_cv = noise_level if noise_level is not None else params['noise_cv']
        
        # Base humidity inversely related to temperature
        # Temp 20°C -> ~80% humidity, Temp 35°C -> ~50% humidity
        base_humidity = 95 - (temperature_values - 15) * 1.5
        
        # Adjust for soil moisture if provided
        if soil_moisture_values is not None:
            moisture_effect = soil_moisture_values * params['moisture_influence']
            base_humidity += moisture_effect
        
        # Add noise
        noise = np.random.normal(0, noise_cv * base_humidity, size=len(base_humidity))
        humidity_values = np.clip(base_humidity + noise, 20, 95)
        
        # Add temporal variation
        humidity_values = self.add_temporal_variation(humidity_values)
        
        # Create readings
        readings = []
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(humidity_values))]
        if locations is None:
            locations = [(30.95, 75.85)] * len(humidity_values)
        
        for i, (timestamp, location, humidity) in enumerate(zip(timestamps, locations, humidity_values)):
            metadata = {
                'temperature_value': float(temperature_values[i]),
                'temp_correlation': params['temp_correlation']
            }
            if soil_moisture_values is not None:
                metadata['soil_moisture_value'] = float(soil_moisture_values[i])
            
            reading = SyntheticSensorReading(
                timestamp=timestamp,
                location=location,
                sensor_type='humidity',
                value=float(humidity),
                unit='%',
                is_synthetic=True,
                correlation_source='temperature_based',
                noise_level=noise_cv,
                metadata=metadata
            )
            readings.append(reading)
        
        # Validate correlation
        actual_correlation = np.corrcoef(temperature_values, humidity_values)[0, 1]
        logger.info(f"Generated humidity data with correlation to temperature: {actual_correlation:.3f}")
        
        if actual_correlation > -0.3:
            logger.warning(f"Humidity-temperature correlation ({actual_correlation:.3f}) "
                         f"not sufficiently negative (target < -0.3)")
        
        return readings
    
    def generate_leaf_wetness(self,
                             humidity_values: np.ndarray,
                             temperature_values: np.ndarray,
                             timestamps: Optional[List[datetime]] = None,
                             locations: Optional[List[Tuple[float, float]]] = None,
                             noise_level: Optional[float] = None) -> List[SyntheticSensorReading]:
        """
        Generate synthetic leaf wetness data based on humidity and temperature.
        
        Leaf wetness increases with high humidity and is optimal at moderate
        temperatures (20-25°C). Used for pest risk assessment.
        
        Args:
            humidity_values: Array of humidity values (%)
            temperature_values: Array of temperature values (°C)
            timestamps: List of timestamps for each reading
            locations: List of (lat, lon) tuples for each reading
            noise_level: Optional custom noise level (coefficient of variation)
            
        Returns:
            List of synthetic leaf wetness readings
        """
        params = self.correlation_params['leaf_wetness']
        noise_cv = noise_level if noise_level is not None else params['noise_cv']
        
        # Leaf wetness increases with humidity (above threshold)
        humidity_factor = np.clip((humidity_values - 50) / 50, 0, 1)
        
        # Temperature effect (optimal around 20-25°C)
        optimal_temp = np.mean(params['optimal_temp_range'])
        temp_range = params['optimal_temp_range'][1] - params['optimal_temp_range'][0]
        temp_factor = 1 - np.abs(temperature_values - optimal_temp) / (optimal_temp + 5)
        temp_factor = np.clip(temp_factor, 0, 1)
        
        # Combine factors
        base_wetness = humidity_factor * temp_factor
        
        # Add noise
        noise = np.random.normal(0, noise_cv, size=len(base_wetness))
        wetness_values = np.clip(base_wetness + noise, 0, 1)
        
        # Add temporal variation
        wetness_values = self.add_temporal_variation(wetness_values)
        
        # Create readings
        readings = []
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(wetness_values))]
        if locations is None:
            locations = [(30.95, 75.85)] * len(wetness_values)
        
        for i, (timestamp, location, wetness) in enumerate(zip(timestamps, locations, wetness_values)):
            # Calculate pest risk based on conditions
            pest_risk = 'high' if (humidity_values[i] > 75 and 
                                  20 <= temperature_values[i] <= 30 and 
                                  wetness > 0.6) else 'low'
            
            reading = SyntheticSensorReading(
                timestamp=timestamp,
                location=location,
                sensor_type='leaf_wetness',
                value=float(wetness),
                unit='0-1 scale',
                is_synthetic=True,
                correlation_source='humidity_temperature_based',
                noise_level=noise_cv,
                metadata={
                    'humidity_value': float(humidity_values[i]),
                    'temperature_value': float(temperature_values[i]),
                    'pest_risk': pest_risk
                }
            )
            readings.append(reading)
        
        logger.info(f"Generated leaf wetness data: mean={np.mean(wetness_values):.3f}, "
                   f"std={np.std(wetness_values):.3f}")
        
        return readings
    
    def add_temporal_variation(self, data: np.ndarray, variation_factor: Optional[float] = None) -> np.ndarray:
        """
        Add realistic temporal variation to sensor data using autocorrelation.
        
        Current values depend partially on previous values, simulating the
        temporal continuity of real environmental measurements.
        
        Args:
            data: Input data array
            variation_factor: Autocorrelation factor (0-1), defaults to self.temporal_autocorrelation
            
        Returns:
            Data array with temporal variation applied
        """
        if variation_factor is None:
            variation_factor = self.temporal_autocorrelation
        
        result = data.copy()
        
        # Apply autocorrelation: current = factor * current + (1-factor) * previous
        for i in range(1, len(result)):
            result[i] = variation_factor * result[i] + (1 - variation_factor) * result[i-1]
        
        return result
    
    def generate_complete_sensor_suite(self,
                                      ndvi_values: np.ndarray,
                                      timestamps: List[datetime],
                                      locations: List[Tuple[float, float]],
                                      location_lat: float = 30.95) -> Dict[str, List[SyntheticSensorReading]]:
        """
        Generate a complete suite of correlated synthetic sensor data.
        
        This method generates all sensor types (soil moisture, temperature,
        humidity, leaf wetness) with proper correlations between them.
        
        Args:
            ndvi_values: Array of NDVI values
            timestamps: List of timestamps
            locations: List of (lat, lon) tuples
            location_lat: Latitude for location-based adjustments
            
        Returns:
            Dictionary mapping sensor type to list of readings
        """
        logger.info(f"Generating complete sensor suite for {len(ndvi_values)} data points")
        
        # Generate soil moisture (correlated with NDVI)
        soil_moisture_readings = self.generate_soil_moisture(
            ndvi_values, timestamps, locations
        )
        soil_moisture_values = np.array([r.value for r in soil_moisture_readings])
        
        # Generate temperature (seasonal pattern)
        temperature_readings = self.generate_temperature(
            timestamps, locations, location_lat
        )
        temperature_values = np.array([r.value for r in temperature_readings])
        
        # Generate humidity (inversely correlated with temperature, influenced by soil moisture)
        humidity_readings = self.generate_humidity(
            temperature_values, soil_moisture_values, timestamps, locations
        )
        humidity_values = np.array([r.value for r in humidity_readings])
        
        # Generate leaf wetness (based on humidity and temperature)
        leaf_wetness_readings = self.generate_leaf_wetness(
            humidity_values, temperature_values, timestamps, locations
        )
        
        sensor_suite = {
            'soil_moisture': soil_moisture_readings,
            'temperature': temperature_readings,
            'humidity': humidity_readings,
            'leaf_wetness': leaf_wetness_readings
        }
        
        # Log summary statistics
        logger.info("Sensor suite generation complete:")
        logger.info(f"  Soil Moisture: {np.mean(soil_moisture_values):.1f}% ± {np.std(soil_moisture_values):.1f}%")
        logger.info(f"  Temperature: {np.mean(temperature_values):.1f}°C ± {np.std(temperature_values):.1f}°C")
        logger.info(f"  Humidity: {np.mean(humidity_values):.1f}% ± {np.std(humidity_values):.1f}%")
        logger.info(f"  Leaf Wetness: {np.mean([r.value for r in leaf_wetness_readings]):.3f} ± "
                   f"{np.std([r.value for r in leaf_wetness_readings]):.3f}")
        
        return sensor_suite
    
    def validate_correlations(self, sensor_suite: Dict[str, List[SyntheticSensorReading]],
                            ndvi_values: np.ndarray) -> Dict[str, float]:
        """
        Validate that generated sensor data meets correlation requirements.
        
        Args:
            sensor_suite: Dictionary of sensor readings
            ndvi_values: Original NDVI values
            
        Returns:
            Dictionary of correlation coefficients
        """
        correlations = {}
        
        # Soil moisture - NDVI correlation (should be > 0.5)
        if 'soil_moisture' in sensor_suite:
            moisture_values = np.array([r.value for r in sensor_suite['soil_moisture']])
            correlations['soil_moisture_ndvi'] = np.corrcoef(ndvi_values, moisture_values)[0, 1]
        
        # Temperature - Humidity correlation (should be < -0.3)
        if 'temperature' in sensor_suite and 'humidity' in sensor_suite:
            temp_values = np.array([r.value for r in sensor_suite['temperature']])
            humidity_values = np.array([r.value for r in sensor_suite['humidity']])
            correlations['temperature_humidity'] = np.corrcoef(temp_values, humidity_values)[0, 1]
        
        # Log validation results
        logger.info("Correlation validation:")
        for key, value in correlations.items():
            logger.info(f"  {key}: {value:.3f}")
        
        return correlations
    
    def export_to_dataframe(self, sensor_suite: Dict[str, List[SyntheticSensorReading]]) -> pd.DataFrame:
        """
        Export sensor suite to a pandas DataFrame for analysis.
        
        Args:
            sensor_suite: Dictionary of sensor readings
            
        Returns:
            DataFrame with all sensor data
        """
        data = []
        
        for sensor_type, readings in sensor_suite.items():
            for reading in readings:
                data.append({
                    'timestamp': reading.timestamp,
                    'latitude': reading.location[0],
                    'longitude': reading.location[1],
                    'sensor_type': reading.sensor_type,
                    'value': reading.value,
                    'unit': reading.unit,
                    'is_synthetic': reading.is_synthetic,
                    'correlation_source': reading.correlation_source,
                    'noise_level': reading.noise_level
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Exported {len(df)} sensor readings to DataFrame")
        
        return df
