"""
Example demonstrating database functionality for the Agricultural Monitoring Platform.
"""

import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Polygon

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import DatabaseConnection, DatabaseModels, DatabaseMigrations
from src.models import SatelliteImage, MonitoringZone, SensorLocation, Alert, IndexTimeSeries


def main():
    """Demonstrate database operations."""
    print("Agricultural Monitoring Platform - Database Example")
    print("=" * 50)
    
    # Initialize database
    db_conn = DatabaseConnection("example_agricultural_monitoring.db")
    
    # Apply migrations to create schema
    print("\n1. Setting up database schema...")
    migrations = DatabaseMigrations(db_conn)
    migrations.migrate_to_latest()
    
    # Initialize models
    db_models = DatabaseModels(db_conn)
    
    # Create a monitoring zone
    print("\n2. Creating monitoring zone...")
    field_boundary = Polygon([
        (43.123, -80.456),  # Southwest corner
        (43.125, -80.456),  # Southeast corner
        (43.125, -80.454),  # Northeast corner
        (43.123, -80.454),  # Northwest corner
        (43.123, -80.456)   # Close polygon
    ])
    
    # Add sensors to the zone
    soil_sensor = SensorLocation(
        id="soil_001",
        sensor_type="soil_moisture",
        latitude=43.124,
        longitude=-80.455,
        installation_date=datetime(2024, 3, 1)
    )
    
    weather_sensor = SensorLocation(
        id="weather_001", 
        sensor_type="weather_station",
        latitude=43.124,
        longitude=-80.455,
        installation_date=datetime(2024, 3, 1)
    )
    
    # Create an alert
    pest_alert = Alert(
        id="alert_001",
        alert_type="pest_risk",
        severity="medium",
        message="Increased aphid activity detected in field sensors",
        created_at=datetime.now()
    )
    
    zone = MonitoringZone(
        id="field_north_40",
        name="North Field 40 Acres",
        geometry=field_boundary,
        crop_type="corn",
        planting_date=datetime(2024, 4, 15),
        expected_harvest=datetime(2024, 10, 1),
        sensors=[soil_sensor, weather_sensor],
        alerts=[pest_alert],
        metadata={"field_notes": "High yield potential area"}
    )
    
    zone_id = db_models.create_monitoring_zone(zone)
    print(f"Created monitoring zone: {zone_id}")
    
    # Create satellite image data
    print("\n3. Creating satellite image...")
    
    # Simulate Sentinel-2A bands (small arrays for example)
    bands = {
        'B02': np.random.rand(50, 50) * 0.3,  # Blue
        'B03': np.random.rand(50, 50) * 0.4,  # Green  
        'B04': np.random.rand(50, 50) * 0.3,  # Red
        'B08': np.random.rand(50, 50) * 0.8,  # NIR
        'B11': np.random.rand(50, 50) * 0.2,  # SWIR1
        'B12': np.random.rand(50, 50) * 0.1   # SWIR2
    }
    
    # Calculate vegetation indices
    ndvi = (bands['B08'] - bands['B04']) / (bands['B08'] + bands['B04'])
    savi = ((bands['B08'] - bands['B04']) / (bands['B08'] + bands['B04'] + 0.5)) * 1.5
    
    indices = {
        'NDVI': ndvi,
        'SAVI': savi
    }
    
    image = SatelliteImage(
        id="S2A_20240923_T43REQ",
        acquisition_date=datetime(2024, 9, 23, 10, 30),
        tile_id="T43REQ",
        cloud_coverage=12.5,
        bands=bands,
        indices=indices,
        geometry=field_boundary,
        quality_flags={
            'atmospheric_correction': True,
            'cloud_mask_applied': True,
            'geometric_correction': True
        }
    )
    
    image_id = db_models.create_satellite_image(image)
    print(f"Created satellite image: {image_id}")
    
    # Create time series data
    print("\n4. Creating vegetation index time series...")
    
    # Simulate weekly NDVI measurements over growing season
    base_date = datetime(2024, 5, 1)
    ndvi_values = [0.3, 0.45, 0.62, 0.78, 0.85, 0.82, 0.79, 0.75, 0.68, 0.55]
    
    for i, ndvi_val in enumerate(ndvi_values):
        measurement = IndexTimeSeries(
            zone_id=zone_id,
            index_type="NDVI",
            timestamp=base_date + timedelta(weeks=i),
            mean_value=ndvi_val,
            std_deviation=0.08,
            pixel_count=2500,  # 50x50 pixels
            quality_score=0.92,
            metadata={"processing_version": "1.0", "cloud_coverage": 5.0}
        )
        
        db_models.create_index_measurement(measurement)
    
    print(f"Created {len(ndvi_values)} NDVI measurements")
    
    # Query and display data
    print("\n5. Querying database...")
    
    # Get monitoring zones
    zones = db_models.list_monitoring_zones()
    print(f"Total monitoring zones: {len(zones)}")
    for zone_info in zones:
        print(f"  - {zone_info['name']} ({zone_info['crop_type']})")
    
    # Get satellite images
    images = db_models.list_satellite_images(max_cloud_coverage=20.0)
    print(f"Satellite images (cloud coverage < 20%): {len(images)}")
    for img_info in images:
        print(f"  - {img_info['id']} ({img_info['cloud_coverage']}% clouds)")
    
    # Get time series data
    timeseries = db_models.get_index_timeseries(zone_id, "NDVI")
    print(f"NDVI time series points: {len(timeseries)}")
    
    # Show trend
    if len(timeseries) >= 2:
        first_val = timeseries[0].mean_value
        last_val = timeseries[-1].mean_value
        trend = "increasing" if last_val > first_val else "decreasing"
        print(f"  NDVI trend: {first_val:.3f} → {last_val:.3f} ({trend})")
    
    # Get alerts
    retrieved_zone = db_models.get_monitoring_zone(zone_id)
    active_alerts = [alert for alert in retrieved_zone.alerts if alert.is_active()]
    print(f"Active alerts: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"  - {alert.severity.upper()}: {alert.message}")
    
    # Database statistics
    print("\n6. Database information...")
    db_info = db_conn.get_database_info()
    print(f"Database file: {db_info['db_path']}")
    print(f"File size: {db_info['file_size']} bytes")
    print("Tables:")
    for table in db_info['tables']:
        print(f"  - {table['name']}: {table['row_count']} rows")
    
    # Cleanup
    db_conn.disconnect()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()