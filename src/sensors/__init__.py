"""
Environmental sensor data integration module.

This module provides functionality for ingesting, validating, and processing
environmental sensor data including soil moisture, temperature, humidity,
and other agricultural monitoring sensors.
"""

from .data_ingestion import SensorDataIngester, SensorReading
from .data_validation import SensorDataValidator, ValidationResult
from .temporal_alignment import TemporalAligner, AlignedReading
from .spatial_interpolation import SpatialInterpolator, InterpolationGrid
from .data_fusion import DataFusionEngine, FusedDataPoint
from .synthetic_sensor_generator import SyntheticSensorGenerator, SyntheticSensorReading

__all__ = [
    'SensorDataIngester',
    'SensorReading', 
    'SensorDataValidator',
    'ValidationResult',
    'TemporalAligner',
    'AlignedReading',
    'SpatialInterpolator',
    'InterpolationGrid',
    'DataFusionEngine',
    'FusedDataPoint',
    'SyntheticSensorGenerator',
    'SyntheticSensorReading'
]