#!/usr/bin/env python3
"""
Test the updated database queries with real production data.
"""

from src.database.db_manager import DatabaseManager


def test_real_database_queries():
    """Test database queries with real production data."""
    
    print("=" * 60)
    print("Testing Database Queries with Real Production Data")
    print("=" * 60)
    
    db = DatabaseManager('data/agriflux.db')
    
    # Test 1: Get database stats
    print("\n--- Test 1: Database Statistics ---")
    stats = db.get_database_stats()
    print(f"Total imagery: {stats['imagery_count']}")
    print(f"Real imagery: {stats['real_imagery_count']}")
    print(f"Synthetic imagery: {stats['synthetic_imagery_count']}")
    print(f"Date range (all): {stats['date_range']}")
    print(f"Date range (real): {stats['real_date_range']}")
    
    # Test 2: Get latest imagery (prefer real)
    print("\n--- Test 2: Get Latest Imagery (prefer real) ---")
    latest = db.get_latest_imagery(prefer_real=True)
    if latest:
        print(f"Latest imagery ID: {latest['id']}")
        print(f"Acquisition date: {latest['acquisition_date']}")
        print(f"Tile ID: {latest['tile_id']}")
        print(f"Synthetic: {latest['synthetic']}")
        print(f"Cloud coverage: {latest['cloud_coverage']}%")
    
    # Test 3: List real imagery only
    print("\n--- Test 3: List Real Imagery Only ---")
    real_imagery = db.get_real_imagery(limit=5)
    print(f"Found {len(real_imagery)} real imagery records (showing first 5)")
    for img in real_imagery[:3]:
        print(f"  - ID {img['id']}: {img['acquisition_date']} (cloud: {img['cloud_coverage']}%)")
    
    # Test 4: List synthetic imagery only
    print("\n--- Test 4: List Synthetic Imagery Only ---")
    synthetic_imagery = db.get_synthetic_imagery(limit=5)
    print(f"Found {len(synthetic_imagery)} synthetic imagery records")
    
    # Test 5: Count real vs synthetic
    print("\n--- Test 5: Count Real vs Synthetic ---")
    real_count = db.count_real_imagery()
    synthetic_count = db.count_synthetic_imagery()
    print(f"Real imagery count: {real_count}")
    print(f"Synthetic imagery count: {synthetic_count}")
    print(f"Total: {real_count + synthetic_count}")
    
    # Test 6: Get temporal series (real only)
    print("\n--- Test 6: Temporal Series (Real Only) ---")
    temporal = db.get_temporal_series('43REQ', synthetic=False)
    print(f"Found {len(temporal)} real temporal records for tile 43REQ")
    if temporal:
        print(f"Date range: {temporal[0]['acquisition_date']} to {temporal[-1]['acquisition_date']}")
    
    # Test 7: List with synthetic filter
    print("\n--- Test 7: List with Synthetic Filter ---")
    all_data = db.list_processed_imagery(limit=100)
    real_data = db.list_processed_imagery(synthetic=False, limit=100)
    synthetic_data = db.list_processed_imagery(synthetic=True, limit=100)
    print(f"All data: {len(all_data)} records")
    print(f"Real data: {len(real_data)} records")
    print(f"Synthetic data: {len(synthetic_data)} records")
    
    print("\n" + "=" * 60)
    print("✓ All database query tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_real_database_queries()
