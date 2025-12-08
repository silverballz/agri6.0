"""
Test script for alert generation functionality.

This script tests the alert generation system with real data from the database.
"""

import sys
import os
import numpy as np
import rasterio
from pathlib import Path

# Direct imports to avoid package issues
sys.path.insert(0, os.path.dirname(__file__))

from src.database.db_manager import DatabaseManager
from src.alerts.alert_generator import AlertGenerator, AlertSeverity, AlertType


def test_alert_generation():
    """Test alert generation with real data."""
    
    print("=" * 60)
    print("Testing Alert Generation System")
    print("=" * 60)
    
    # Initialize components
    db_manager = DatabaseManager()
    alert_generator = AlertGenerator()
    
    # Get latest imagery from database
    print("\n1. Loading latest imagery from database...")
    latest_imagery = db_manager.get_latest_imagery()
    
    if not latest_imagery:
        print("❌ No imagery found in database. Run populate_database.py first.")
        return False
    
    print(f"✅ Found imagery: {latest_imagery['tile_id']} from {latest_imagery['acquisition_date']}")
    
    # Load NDVI data
    print("\n2. Loading NDVI data...")
    ndvi_path = latest_imagery.get('ndvi_path')
    
    if not ndvi_path or not Path(ndvi_path).exists():
        print(f"❌ NDVI file not found: {ndvi_path}")
        return False
    
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)
    
    print(f"✅ Loaded NDVI data: shape={ndvi.shape}, mean={np.mean(ndvi):.3f}")
    
    # Load NDWI data if available
    ndwi = None
    ndwi_path = latest_imagery.get('ndwi_path')
    if ndwi_path and Path(ndwi_path).exists():
        print("\n3. Loading NDWI data...")
        try:
            with rasterio.open(ndwi_path) as src:
                ndwi = src.read(1)
            print(f"✅ Loaded NDWI data: shape={ndwi.shape}, mean={np.mean(ndwi):.3f}")
        except Exception as e:
            print(f"⚠️  Could not load NDWI data: {str(e)}")
            print("   Continuing without NDWI data...")
    
    # Generate alerts
    print("\n4. Generating alerts...")
    alerts = alert_generator.generate_alerts(
        ndvi=ndvi,
        ndwi=ndwi,
        temperature=28.5,  # Example temperature
        humidity=65.0,     # Example humidity
        metadata={
            'tile_id': latest_imagery['tile_id'],
            'acquisition_date': latest_imagery['acquisition_date']
        }
    )
    
    print(f"✅ Generated {len(alerts)} alerts")
    
    # Display alerts
    print("\n5. Alert Details:")
    print("-" * 60)
    
    for i, alert in enumerate(alerts, 1):
        print(f"\nAlert #{i}:")
        print(f"  Type: {alert.alert_type.value}")
        print(f"  Severity: {alert.severity.value.upper()}")
        print(f"  Message: {alert.message}")
        print(f"  Affected Area: {alert.affected_area_percentage:.1f}%")
        print(f"  Recommendation: {alert.recommendation}")
    
    # Save alerts to database
    print("\n6. Saving alerts to database...")
    saved_count = 0
    
    for alert in alerts:
        alert_dict = alert.to_dict()
        alert_id = db_manager.save_alert(
            imagery_id=latest_imagery['id'],
            alert_type=alert_dict['alert_type'],
            severity=alert_dict['severity'],
            message=alert_dict['message'],
            recommendation=alert_dict['recommendation'],
            affected_area=alert_dict['affected_area']
        )
        saved_count += 1
        print(f"  ✅ Saved alert #{alert_id}")
    
    print(f"\n✅ Saved {saved_count} alerts to database")
    
    # Get alert summary
    print("\n7. Alert Summary:")
    summary = alert_generator.get_alert_summary(alerts)
    print(f"  Total Alerts: {summary['total_alerts']}")
    print(f"  By Severity: {summary['by_severity']}")
    print(f"  By Type: {summary['by_type']}")
    print(f"  Max Affected Area: {summary['max_affected_area']:.1f}%")
    
    # Test database retrieval
    print("\n8. Testing database retrieval...")
    active_alerts = db_manager.get_active_alerts()
    print(f"✅ Retrieved {len(active_alerts)} active alerts from database")
    
    # Test acknowledgment
    if active_alerts:
        print("\n9. Testing alert acknowledgment...")
        test_alert_id = active_alerts[0]['id']
        success = db_manager.acknowledge_alert(test_alert_id)
        if success:
            print(f"✅ Successfully acknowledged alert #{test_alert_id}")
        else:
            print(f"❌ Failed to acknowledge alert #{test_alert_id}")
    
    print("\n" + "=" * 60)
    print("✅ Alert Generation Test Complete!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_alert_generation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
