"""
Example demonstrating environmental sensor data integration system.

This example shows how to:
1. Ingest sensor data from CSV/JSON files
2. Validate sensor readings for quality
3. Align sensor data with satellite overpass times
4. Perform spatial interpolation of sensor measurements
5. Correlate spectral anomalies with environmental conditions
6. Generate threshold-based alerts
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
import csv

from src.sensors import (
    SensorDataIngester, SensorDataValidator, TemporalAligner,
    SpatialInterpolator, DataFusionEngine, SpectralAnomaly
)


def create_sample_sensor_data():
    """Create sample sensor data files for demonstration."""
    
    # Create sample CSV data
    csv_data = [
        {
            'sensor_id': 'soil_moisture_001',
            'timestamp': '2024-09-23T10:00:00Z',
            'sensor_type': 'soil_moisture',
            'value': 45.2,
            'unit': '%',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'quality_flag': 'good'
        },
        {
            'sensor_id': 'soil_moisture_001',
            'timestamp': '2024-09-23T11:00:00Z',
            'sensor_type': 'soil_moisture',
            'value': 43.8,
            'unit': '%',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'quality_flag': 'good'
        },
        {
            'sensor_id': 'temperature_001',
            'timestamp': '2024-09-23T10:30:00Z',
            'sensor_type': 'temperature',
            'value': 28.5,
            'unit': '°C',
            'latitude': 40.7130,
            'longitude': -74.0062,
            'quality_flag': 'good'
        },
        {
            'sensor_id': 'humidity_001',
            'timestamp': '2024-09-23T10:30:00Z',
            'sensor_type': 'humidity',
            'value': 75.0,
            'unit': '%',
            'latitude': 40.7132,
            'longitude': -74.0064,
            'quality_flag': 'good'
        }
    ]
    
    # Write CSV file
    csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    writer = csv.DictWriter(csv_file, fieldnames=csv_data[0].keys())
    writer.writeheader()
    writer.writerows(csv_data)
    csv_file.close()
    
    # Create sample JSON data
    json_data = {
        'readings': [
            {
                'sensor_id': 'leaf_wetness_001',
                'timestamp': '2024-09-23T10:15:00Z',
                'sensor_type': 'leaf_wetness',
                'value': 4.5,
                'unit': 'hours',
                'latitude': 40.7125,
                'longitude': -74.0058,
                'quality_flag': 'good',
                'metadata': {'sensor_height': '1.5m'}
            },
            {
                'sensor_id': 'solar_radiation_001',
                'timestamp': '2024-09-23T10:30:00Z',
                'sensor_type': 'solar_radiation',
                'value': 850.0,
                'unit': 'W/m²',
                'latitude': 40.7127,
                'longitude': -74.0061,
                'quality_flag': 'good'
            }
        ]
    }
    
    # Write JSON file
    json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(json_data, json_file)
    json_file.close()
    
    return csv_file.name, json_file.name


def create_sample_spectral_data():
    """Create sample spectral data for demonstration."""
    base_time = datetime(2024, 9, 23, 10, 30, 0)
    
    return [
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
                'NDVI': 0.65,  # Decline indicating potential stress
                'SAVI': 0.55,
                'EVI': 0.45,
                'NDWI': 0.15
            }
        },
        {
            'timestamp': base_time,
            'latitude': 40.7130,
            'longitude': -74.0062,
            'indices': {
                'NDVI': 0.72,
                'SAVI': 0.62,
                'EVI': 0.52,
                'NDWI': 0.22
            }
        }
    ]


def demonstrate_sensor_ingestion():
    """Demonstrate sensor data ingestion from different formats."""
    print("=== Sensor Data Ingestion Demo ===")
    
    # Create sample data files
    csv_file, json_file = create_sample_sensor_data()
    
    try:
        # Initialize ingester
        ingester = SensorDataIngester()
        
        # Ingest CSV data
        print(f"\n1. Ingesting CSV data from {Path(csv_file).name}")
        csv_readings = ingester.ingest_csv(csv_file)
        print(f"   Ingested {len(csv_readings)} readings from CSV")
        
        for reading in csv_readings[:2]:  # Show first 2
            print(f"   - {reading.sensor_type}: {reading.value} {reading.unit} "
                  f"at {reading.timestamp} (ID: {reading.sensor_id})")
        
        # Ingest JSON data
        print(f"\n2. Ingesting JSON data from {Path(json_file).name}")
        json_readings = ingester.ingest_json(json_file)
        print(f"   Ingested {len(json_readings)} readings from JSON")
        
        for reading in json_readings:
            print(f"   - {reading.sensor_type}: {reading.value} {reading.unit} "
                  f"at {reading.timestamp} (ID: {reading.sensor_id})")
        
        # Combine all readings
        all_readings = csv_readings + json_readings
        print(f"\n3. Total readings ingested: {len(all_readings)}")
        
        # Show supported sensor types
        print(f"\n4. Supported sensor types: {list(ingester.get_supported_sensor_types().keys())}")
        
        return all_readings
        
    finally:
        # Clean up temporary files
        Path(csv_file).unlink()
        Path(json_file).unlink()


def demonstrate_data_validation(readings):
    """Demonstrate sensor data validation."""
    print("\n=== Data Validation Demo ===")
    
    # Initialize validator
    validator = SensorDataValidator()
    
    # Add some invalid readings for demonstration
    invalid_reading = readings[0]
    invalid_reading.value = 150.0  # Invalid soil moisture > 100%
    
    print("\n1. Validating individual readings:")
    
    # Validate each reading
    validation_results = []
    for i, reading in enumerate(readings[:3]):  # Validate first 3
        result = validator.validate_reading(reading)
        validation_results.append(result)
        
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"   Reading {i+1}: {status} (Quality: {result.quality_score:.2f}, "
              f"Flag: {result.recommended_flag})")
        
        if result.issues:
            for issue in result.issues:
                print(f"     Issue: {issue}")
    
    # Batch validation
    print("\n2. Batch validation:")
    batch_results = validator.validate_batch(readings)
    
    # Calculate statistics
    stats = validator.get_quality_statistics(batch_results)
    print(f"   Total readings: {stats['total_readings']}")
    print(f"   Valid readings: {stats['valid_readings']}")
    print(f"   Validity rate: {stats['validity_rate']:.1%}")
    print(f"   Mean quality score: {stats['mean_quality_score']:.2f}")
    print(f"   Flag distribution: {stats['flag_distribution']}")
    
    return [r for r, result in zip(readings, batch_results) if result.is_valid]


def demonstrate_temporal_alignment(readings):
    """Demonstrate temporal alignment with satellite overpass times."""
    print("\n=== Temporal Alignment Demo ===")
    
    # Initialize aligner
    aligner = TemporalAligner()
    
    # Generate satellite overpass times
    start_date = datetime(2024, 9, 23)
    end_date = datetime(2024, 9, 25)
    
    print("\n1. Generating satellite overpass times:")
    overpass_times = aligner.generate_overpass_times(
        start_date, end_date, 40.7128, -74.0060
    )
    
    for i, overpass_time in enumerate(overpass_times):
        print(f"   Overpass {i+1}: {overpass_time}")
    
    # Align readings to overpass times
    print("\n2. Aligning sensor readings to overpass times:")
    aligned_readings = aligner.align_to_overpass(readings, overpass_times)
    
    for i, aligned in enumerate(aligned_readings[:3]):  # Show first 3
        print(f"   Alignment {i+1}:")
        print(f"     Sensor: {aligned.original_reading.sensor_id}")
        print(f"     Original time: {aligned.original_reading.timestamp}")
        print(f"     Satellite time: {aligned.satellite_timestamp}")
        print(f"     Time offset: {aligned.time_offset}")
        print(f"     Value: {aligned.interpolated_value or aligned.original_reading.value}")
        print(f"     Confidence: {aligned.confidence:.2f}")
    
    # Calculate alignment statistics
    stats = aligner.get_alignment_statistics(aligned_readings)
    print(f"\n3. Alignment statistics:")
    if stats:
        print(f"   Total alignments: {stats['total_alignments']}")
        print(f"   Interpolated readings: {stats['interpolated_readings']}")
        print(f"   Direct readings: {stats['direct_readings']}")
        print(f"   Mean time offset: {stats['mean_time_offset_seconds']:.0f} seconds")
        print(f"   Mean confidence: {stats['mean_confidence']:.2f}")
    else:
        print("   No alignment statistics available (no alignments found)")
    
    return aligned_readings


def demonstrate_spatial_interpolation(readings):
    """Demonstrate spatial interpolation of sensor data."""
    print("\n=== Spatial Interpolation Demo ===")
    
    # Initialize interpolator
    interpolator = SpatialInterpolator()
    
    # Filter readings with coordinates
    readings_with_coords = [r for r in readings if r.latitude and r.longitude]
    
    if len(readings_with_coords) < 3:
        print("   Insufficient readings with coordinates for interpolation")
        return
    
    print(f"\n1. Interpolating {len(readings_with_coords)} sensor readings:")
    
    # Define interpolation grid bounds
    lats = [r.latitude for r in readings_with_coords]
    lons = [r.longitude for r in readings_with_coords]
    
    grid_bounds = (
        min(lons) - 0.01, min(lats) - 0.01,
        max(lons) + 0.01, max(lats) + 0.01
    )
    
    print(f"   Grid bounds: {grid_bounds}")
    
    # Group by sensor type for interpolation
    sensor_groups = {}
    for reading in readings_with_coords:
        if reading.sensor_type not in sensor_groups:
            sensor_groups[reading.sensor_type] = []
        sensor_groups[reading.sensor_type].append(reading)
    
    # Interpolate each sensor type
    for sensor_type, sensor_readings in sensor_groups.items():
        if len(sensor_readings) < 2:
            continue
            
        print(f"\n2. Interpolating {sensor_type} data:")
        
        try:
            # Create interpolation grid
            grid = interpolator.interpolate_sensors(
                sensor_readings, grid_bounds, grid_resolution=0.005
            )
            
            print(f"   Grid shape: {grid.values.shape}")
            print(f"   Method: {grid.method}")
            print(f"   Value range: {np.nanmin(grid.values):.2f} - {np.nanmax(grid.values):.2f}")
            print(f"   Quality mask range: {np.nanmin(grid.quality_mask):.2f} - {np.nanmax(grid.quality_mask):.2f}")
            
            # Validate interpolation
            validation = interpolator.validate_interpolation(sensor_readings)
            if 'mae' in validation:
                print(f"   Validation MAE: {validation['mae']:.2f}")
                print(f"   Validation RMSE: {validation['rmse']:.2f}")
                print(f"   R-squared: {validation['r_squared']:.3f}")
            
        except Exception as e:
            print(f"   Interpolation failed: {e}")
    
    # Demonstrate point interpolation
    print(f"\n3. Point interpolation example:")
    target_points = [(40.7129, -74.0061), (40.7131, -74.0063)]
    
    for sensor_type, sensor_readings in sensor_groups.items():
        if len(sensor_readings) >= 2:
            try:
                interpolated_values = interpolator.interpolate_to_points(
                    sensor_readings, target_points
                )
                print(f"   {sensor_type} at target points: {interpolated_values}")
                break
            except Exception as e:
                print(f"   Point interpolation failed for {sensor_type}: {e}")


def demonstrate_data_fusion(aligned_readings):
    """Demonstrate data fusion and alert generation."""
    print("\n=== Data Fusion and Alert Generation Demo ===")
    
    # Initialize fusion engine
    fusion_engine = DataFusionEngine()
    
    # Create sample spectral data
    spectral_data = create_sample_spectral_data()
    
    print(f"\n1. Spectral data: {len(spectral_data)} measurements")
    for i, data in enumerate(spectral_data):
        print(f"   Measurement {i+1}: NDVI={data['indices']['NDVI']:.2f}, "
              f"SAVI={data['indices']['SAVI']:.2f} at {data['timestamp']}")
    
    # Detect spectral anomalies
    print(f"\n2. Detecting spectral anomalies:")
    
    # Create extended data for anomaly detection
    extended_spectral_data = []
    base_time = datetime(2024, 9, 1, 10, 30, 0)
    
    # Add baseline data (30 days)
    for i in range(30):
        extended_spectral_data.append({
            'timestamp': base_time + timedelta(days=i),
            'latitude': 40.7128,
            'longitude': -74.0060,
            'indices': {
                'NDVI': 0.75 + np.random.normal(0, 0.03),
                'SAVI': 0.65 + np.random.normal(0, 0.02),
                'EVI': 0.55 + np.random.normal(0, 0.02)
            }
        })
    
    # Add anomalous data
    extended_spectral_data.extend(spectral_data)
    
    anomalies = fusion_engine.detect_spectral_anomalies(extended_spectral_data)
    print(f"   Detected {len(anomalies)} spectral anomalies")
    
    for anomaly in anomalies:
        print(f"     - {anomaly.anomaly_type} at {anomaly.location} "
              f"(severity: {anomaly.severity:.2f}, confidence: {anomaly.confidence:.2f})")
    
    # Correlate spectral and environmental data
    print(f"\n3. Correlating spectral and environmental data:")
    correlations = fusion_engine.correlate_spectral_environmental(
        spectral_data, aligned_readings
    )
    
    print(f"   Found {len(correlations)} correlations")
    for correlation in correlations:
        print(f"     - {correlation.environmental_factor} vs {correlation.spectral_indicator}: "
              f"r={correlation.correlation_coefficient:.3f} "
              f"(p={correlation.p_value:.3f}, n={correlation.sample_size})")
    
    # Generate alerts
    print(f"\n4. Generating alerts:")
    alerts = fusion_engine.generate_alerts(anomalies, aligned_readings, correlations)
    
    print(f"   Generated {len(alerts)} alerts")
    for alert in alerts:
        print(f"     - {alert.alert_type.upper()} ({alert.severity})")
        print(f"       Location: {alert.location}")
        print(f"       Description: {alert.description}")
        print(f"       Contributing factors: {', '.join(alert.contributing_factors)}")
        print(f"       Recommended actions: {', '.join(alert.recommended_actions[:2])}...")
        print(f"       Confidence: {alert.confidence:.2f}")
        print(f"       Expires: {alert.expires_at}")
    
    # Calculate data quality scores
    print(f"\n5. Data quality assessment:")
    quality_scores = fusion_engine.calculate_data_quality_score(spectral_data, aligned_readings)
    
    for score_type, score_value in quality_scores.items():
        print(f"   {score_type.replace('_', ' ').title()}: {score_value:.2f}")


def main():
    """Run the complete sensor integration demonstration."""
    print("Environmental Sensor Data Integration System Demo")
    print("=" * 50)
    
    try:
        # Step 1: Ingest sensor data
        readings = demonstrate_sensor_ingestion()
        
        # Step 2: Validate data quality
        valid_readings = demonstrate_data_validation(readings)
        
        # Step 3: Temporal alignment
        aligned_readings = demonstrate_temporal_alignment(valid_readings)
        
        # Step 4: Spatial interpolation
        demonstrate_spatial_interpolation(valid_readings)
        
        # Step 5: Data fusion and alerts
        demonstrate_data_fusion(aligned_readings)
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nThis demonstration showed:")
        print("✓ Sensor data ingestion from CSV/JSON formats")
        print("✓ Data validation and quality assessment")
        print("✓ Temporal alignment with satellite overpass times")
        print("✓ Spatial interpolation of point measurements")
        print("✓ Spectral anomaly detection")
        print("✓ Correlation analysis between spectral and environmental data")
        print("✓ Threshold-based alert generation")
        print("✓ Data quality scoring for fused datasets")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()