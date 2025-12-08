#!/usr/bin/env python3
"""
Demo: Database Usage Examples

Shows how to use the database in the dashboard and other components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database.db_manager import DatabaseManager


def demo_basic_queries():
    """Demonstrate basic database queries."""
    print("="*60)
    print("DEMO: Basic Database Queries")
    print("="*60)
    
    db = DatabaseManager('data/agriflux.db')
    
    # Get latest imagery
    print("\n1. Get Latest Imagery:")
    latest = db.get_latest_imagery()
    print(f"   Tile: {latest['tile_id']}")
    print(f"   Date: {latest['acquisition_date']}")
    print(f"   Cloud Coverage: {latest['cloud_coverage']:.2f}%")
    
    # Access GeoTIFF paths
    print("\n2. Access Vegetation Index Files:")
    print(f"   NDVI: {Path(latest['ndvi_path']).name}")
    print(f"   SAVI: {Path(latest['savi_path']).name}")
    print(f"   EVI: {Path(latest['evi_path']).name}")
    print(f"   NDWI: {Path(latest['ndwi_path']).name}")
    
    # Get metadata
    print("\n3. Access Metadata:")
    import json
    metadata = json.loads(latest['metadata_json'])
    print(f"   Spacecraft: {metadata['spacecraft_name']}")
    print(f"   Processing Level: {metadata['processing_level']}")
    print(f"   EPSG Code: {metadata['epsg_code']}")


def demo_alert_workflow():
    """Demonstrate alert creation and management."""
    print("\n" + "="*60)
    print("DEMO: Alert Workflow")
    print("="*60)
    
    db = DatabaseManager('data/agriflux.db')
    
    # Get latest imagery
    latest = db.get_latest_imagery()
    
    # Create a sample alert
    print("\n1. Creating Sample Alert...")
    alert_id = db.save_alert(
        imagery_id=latest['id'],
        alert_type='vegetation_stress',
        severity='medium',
        message='Moderate vegetation stress detected in northern section',
        recommendation='Increase irrigation frequency and monitor closely'
    )
    print(f"   ✓ Alert created with ID: {alert_id}")
    
    # Get active alerts
    print("\n2. Retrieving Active Alerts...")
    active_alerts = db.get_active_alerts()
    print(f"   Found {len(active_alerts)} active alert(s)")
    for alert in active_alerts:
        print(f"   - [{alert['severity'].upper()}] {alert['message']}")
    
    # Acknowledge alert
    print("\n3. Acknowledging Alert...")
    success = db.acknowledge_alert(alert_id)
    print(f"   ✓ Alert acknowledged: {success}")
    
    # Check active alerts again
    active_alerts = db.get_active_alerts()
    print(f"   Active alerts now: {len(active_alerts)}")


def demo_prediction_workflow():
    """Demonstrate AI prediction storage."""
    print("\n" + "="*60)
    print("DEMO: AI Prediction Workflow")
    print("="*60)
    
    db = DatabaseManager('data/agriflux.db')
    
    # Get latest imagery
    latest = db.get_latest_imagery()
    
    # Save a sample prediction
    print("\n1. Saving Sample Prediction...")
    prediction_id = db.save_prediction(
        imagery_id=latest['id'],
        model_version='rule_based_v1.0',
        prediction_type='crop_health',
        predictions={
            'healthy_pixels': 85000,
            'moderate_pixels': 12000,
            'stressed_pixels': 3000,
            'critical_pixels': 500
        },
        confidence_scores={
            'mean_confidence': 0.87,
            'min_confidence': 0.45,
            'max_confidence': 0.99
        }
    )
    print(f"   ✓ Prediction saved with ID: {prediction_id}")
    
    # Retrieve predictions
    print("\n2. Retrieving Predictions...")
    predictions = db.get_predictions_for_imagery(latest['id'])
    print(f"   Found {len(predictions)} prediction(s)")
    for pred in predictions:
        print(f"   - Type: {pred['prediction_type']}")
        print(f"     Model: {pred['model_version']}")
        print(f"     Created: {pred['created_at']}")


def demo_temporal_analysis():
    """Demonstrate temporal series queries."""
    print("\n" + "="*60)
    print("DEMO: Temporal Analysis")
    print("="*60)
    
    db = DatabaseManager('data/agriflux.db')
    
    # Get latest imagery
    latest = db.get_latest_imagery()
    
    # Get temporal series
    print("\n1. Temporal Series Query:")
    series = db.get_temporal_series(latest['tile_id'])
    print(f"   Found {len(series)} record(s) for tile {latest['tile_id']}")
    
    for record in series:
        print(f"\n   Date: {record['acquisition_date']}")
        print(f"   Cloud Coverage: {record['cloud_coverage']:.2f}%")
        
        # Count available indices
        indices = []
        for field in ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path', 'ndsi_path']:
            if record.get(field):
                indices.append(field.replace('_path', '').upper())
        print(f"   Indices: {', '.join(indices)}")


def demo_statistics():
    """Demonstrate database statistics."""
    print("\n" + "="*60)
    print("DEMO: Database Statistics")
    print("="*60)
    
    db = DatabaseManager('data/agriflux.db')
    
    stats = db.get_database_stats()
    
    print(f"\n📊 Current Database State:")
    print(f"   Imagery Records: {stats['imagery_count']}")
    print(f"   Total Alerts: {stats['total_alerts']}")
    print(f"   Active Alerts: {stats['active_alerts']}")
    print(f"   AI Predictions: {stats['predictions_count']}")
    print(f"\n📅 Date Coverage:")
    print(f"   Earliest: {stats['date_range']['earliest']}")
    print(f"   Latest: {stats['date_range']['latest']}")


def main():
    """Run all demos."""
    try:
        demo_basic_queries()
        demo_alert_workflow()
        demo_prediction_workflow()
        demo_temporal_analysis()
        demo_statistics()
        
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        print("\nThe database is ready for use in the dashboard!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
