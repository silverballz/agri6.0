#!/usr/bin/env python3
"""
Quick test to verify database queries work correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database.db_manager import DatabaseManager


def test_database_queries():
    """Test various database query operations."""
    print("Testing database queries...")
    
    db = DatabaseManager('data/agriflux.db')
    
    # Test 1: Get latest imagery
    print("\n1. Testing get_latest_imagery()...")
    latest = db.get_latest_imagery()
    assert latest is not None, "No imagery found"
    assert 'id' in latest, "Missing id field"
    assert 'tile_id' in latest, "Missing tile_id field"
    assert 'acquisition_date' in latest, "Missing acquisition_date field"
    print(f"   ✓ Found imagery: ID={latest['id']}, Tile={latest['tile_id']}")
    
    # Test 2: Get specific imagery by ID
    print("\n2. Testing get_processed_imagery()...")
    imagery = db.get_processed_imagery(latest['id'])
    assert imagery is not None, "Failed to retrieve imagery by ID"
    assert imagery['id'] == latest['id'], "ID mismatch"
    print(f"   ✓ Retrieved imagery by ID: {imagery['id']}")
    
    # Test 3: List all imagery
    print("\n3. Testing list_processed_imagery()...")
    imagery_list = db.list_processed_imagery()
    assert len(imagery_list) > 0, "No imagery in list"
    assert imagery_list[0]['id'] == latest['id'], "Latest not in list"
    print(f"   ✓ Listed {len(imagery_list)} imagery records")
    
    # Test 4: Get temporal series
    print("\n4. Testing get_temporal_series()...")
    series = db.get_temporal_series(latest['tile_id'])
    assert len(series) > 0, "No temporal series data"
    print(f"   ✓ Retrieved {len(series)} temporal records")
    
    # Test 5: Verify GeoTIFF paths
    print("\n5. Testing GeoTIFF path integrity...")
    geotiff_count = 0
    for field in ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path', 'ndsi_path']:
        if latest.get(field):
            path = Path(latest[field])
            assert path.exists(), f"GeoTIFF not found: {path}"
            geotiff_count += 1
    assert geotiff_count > 0, "No GeoTIFF paths found"
    print(f"   ✓ Verified {geotiff_count} GeoTIFF files exist")
    
    # Test 6: Database statistics
    print("\n6. Testing get_database_stats()...")
    stats = db.get_database_stats()
    assert stats['imagery_count'] > 0, "No imagery in stats"
    assert 'date_range' in stats, "Missing date_range in stats"
    print(f"   ✓ Stats: {stats['imagery_count']} imagery, {stats['total_alerts']} alerts")
    
    # Test 7: Test alert operations (should be empty)
    print("\n7. Testing alert operations...")
    active_alerts = db.get_active_alerts()
    assert isinstance(active_alerts, list), "Active alerts should be a list"
    print(f"   ✓ Active alerts: {len(active_alerts)}")
    
    # Test 8: Test prediction operations (should be empty)
    print("\n8. Testing prediction operations...")
    predictions = db.get_predictions_for_imagery(latest['id'])
    assert isinstance(predictions, list), "Predictions should be a list"
    print(f"   ✓ Predictions: {len(predictions)}")
    
    print("\n" + "="*60)
    print("✅ All database query tests passed!")
    print("="*60)


if __name__ == '__main__':
    try:
        test_database_queries()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
