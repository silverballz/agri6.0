"""
Test script to verify dashboard pages work with real data
"""

import sys
sys.path.append('src')

from database.db_manager import DatabaseManager
import rasterio
from pathlib import Path

def test_database_connection():
    """Test database connection and data availability"""
    print("Testing database connection...")
    
    db = DatabaseManager()
    stats = db.get_database_stats()
    
    print(f"✓ Database connected")
    print(f"  - Imagery records: {stats['imagery_count']}")
    print(f"  - Total alerts: {stats['total_alerts']}")
    print(f"  - Active alerts: {stats['active_alerts']}")
    print(f"  - Predictions: {stats['predictions_count']}")
    
    return stats['imagery_count'] > 0

def test_imagery_files():
    """Test that imagery files exist and are readable"""
    print("\nTesting imagery files...")
    
    db = DatabaseManager()
    latest = db.get_latest_imagery()
    
    if not latest:
        print("✗ No imagery found in database")
        return False
    
    print(f"✓ Latest imagery: {latest['acquisition_date']}")
    
    # Check NDVI file
    ndvi_path = latest.get('ndvi_path')
    if ndvi_path and Path(ndvi_path).exists():
        try:
            with rasterio.open(ndvi_path) as src:
                data = src.read(1)
                print(f"✓ NDVI file readable: {ndvi_path}")
                print(f"  - Shape: {data.shape}")
                print(f"  - Valid pixels: {(data != src.nodata).sum()}")
        except Exception as e:
            print(f"✗ Error reading NDVI file: {e}")
            return False
    else:
        print(f"✗ NDVI file not found: {ndvi_path}")
        return False
    
    return True

def test_alerts():
    """Test alert retrieval"""
    print("\nTesting alerts...")
    
    db = DatabaseManager()
    alerts = db.get_active_alerts()
    
    print(f"✓ Retrieved {len(alerts)} active alerts")
    
    if alerts:
        alert = alerts[0]
        print(f"  - Sample alert: {alert['alert_type']} ({alert['severity']})")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Dashboard Pages Integration Test")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Imagery Files", test_imagery_files),
        ("Alerts", test_alerts)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Dashboard pages are ready to use.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
