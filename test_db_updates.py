#!/usr/bin/env python3
"""
Test script for database query updates to prioritize real data.
"""

import sys
import sqlite3
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.db_manager import DatabaseManager


def test_database_updates():
    """Test all database updates for real data prioritization."""
    
    print("=" * 60)
    print("Testing Database Updates for Real Data Prioritization")
    print("=" * 60)
    
    # Test initialization
    db = DatabaseManager('/tmp/test_agriflux.db')
    print('\n✓ Database initialized successfully')
    
    # Check schema
    conn = sqlite3.connect('/tmp/test_agriflux.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(processed_imagery)")
    columns = cursor.fetchall()
    print('\nColumns in processed_imagery:')
    for col in columns:
        print(f'  {col[1]} ({col[2]})')
    
    # Verify synthetic column exists
    column_names = [col[1] for col in columns]
    assert 'synthetic' in column_names, "synthetic column not found!"
    print('\n✓ Synthetic column exists in schema')
    
    # Test save with synthetic flag
    print('\n--- Testing save_processed_imagery ---')
    
    print('\n1. Saving real imagery (synthetic=False)...')
    imagery_id1 = db.save_processed_imagery(
        acquisition_date='2024-01-01',
        tile_id='43REQ',
        cloud_coverage=10.5,
        geotiff_paths={'NDVI': '/path/to/ndvi.tif'},
        metadata={'test': 'data'},
        synthetic=False
    )
    print(f'   Saved real imagery with ID: {imagery_id1}')
    
    print('\n2. Saving synthetic imagery (synthetic=True)...')
    imagery_id2 = db.save_processed_imagery(
        acquisition_date='2024-01-02',
        tile_id='43REQ',
        cloud_coverage=5.0,
        geotiff_paths={'NDVI': '/path/to/ndvi2.tif'},
        metadata={'test': 'data2'},
        synthetic=True
    )
    print(f'   Saved synthetic imagery with ID: {imagery_id2}')
    
    print('\n3. Saving another real imagery (more recent)...')
    imagery_id3 = db.save_processed_imagery(
        acquisition_date='2024-01-03',
        tile_id='43REQ',
        cloud_coverage=8.0,
        geotiff_paths={'NDVI': '/path/to/ndvi3.tif'},
        metadata={'test': 'data3'},
        synthetic=False
    )
    print(f'   Saved real imagery with ID: {imagery_id3}')
    
    # Test get_latest_imagery with prefer_real
    print('\n--- Testing get_latest_imagery ---')
    
    print('\n1. Getting latest with prefer_real=True...')
    latest = db.get_latest_imagery(prefer_real=True)
    print(f'   Latest (prefer real): ID={latest["id"]}, date={latest["acquisition_date"]}, synthetic={latest["synthetic"]}')
    assert latest["synthetic"] == 0, "Should return real data!"
    assert latest["id"] == imagery_id3, "Should return most recent real data!"
    print('   ✓ Correctly returned most recent real data')
    
    print('\n2. Getting latest with prefer_real=False...')
    latest_any = db.get_latest_imagery(prefer_real=False)
    print(f'   Latest (any): ID={latest_any["id"]}, date={latest_any["acquisition_date"]}, synthetic={latest_any["synthetic"]}')
    assert latest_any["id"] == imagery_id3, "Should return most recent overall!"
    print('   ✓ Correctly returned most recent data overall')
    
    # Test list_processed_imagery with synthetic filter
    print('\n--- Testing list_processed_imagery ---')
    
    print('\n1. Listing all imagery...')
    all_imagery = db.list_processed_imagery()
    print(f'   Total imagery count: {len(all_imagery)}')
    assert len(all_imagery) == 3, "Should have 3 records!"
    
    print('\n2. Listing real imagery only (synthetic=False)...')
    real_only = db.list_processed_imagery(synthetic=False)
    print(f'   Real imagery count: {len(real_only)}')
    assert len(real_only) == 2, "Should have 2 real records!"
    for img in real_only:
        assert img['synthetic'] == 0, "All should be real!"
    print('   ✓ Correctly filtered real imagery')
    
    print('\n3. Listing synthetic imagery only (synthetic=True)...')
    synthetic_only = db.list_processed_imagery(synthetic=True)
    print(f'   Synthetic imagery count: {len(synthetic_only)}')
    assert len(synthetic_only) == 1, "Should have 1 synthetic record!"
    for img in synthetic_only:
        assert img['synthetic'] == 1, "All should be synthetic!"
    print('   ✓ Correctly filtered synthetic imagery')
    
    # Test new methods
    print('\n--- Testing new query methods ---')
    
    print('\n1. Testing get_real_imagery()...')
    real_imgs = db.get_real_imagery()
    print(f'   Real imagery: {len(real_imgs)} records')
    assert len(real_imgs) == 2, "Should have 2 real records!"
    print('   ✓ get_real_imagery works correctly')
    
    print('\n2. Testing get_synthetic_imagery()...')
    synth_imgs = db.get_synthetic_imagery()
    print(f'   Synthetic imagery: {len(synth_imgs)} records')
    assert len(synth_imgs) == 1, "Should have 1 synthetic record!"
    print('   ✓ get_synthetic_imagery works correctly')
    
    print('\n3. Testing count_real_imagery()...')
    real_count = db.count_real_imagery()
    print(f'   Real imagery count: {real_count}')
    assert real_count == 2, "Should count 2 real records!"
    print('   ✓ count_real_imagery works correctly')
    
    print('\n4. Testing count_synthetic_imagery()...')
    synthetic_count = db.count_synthetic_imagery()
    print(f'   Synthetic imagery count: {synthetic_count}')
    assert synthetic_count == 1, "Should count 1 synthetic record!"
    print('   ✓ count_synthetic_imagery works correctly')
    
    # Test get_temporal_series with synthetic filter
    print('\n--- Testing get_temporal_series ---')
    
    print('\n1. Getting temporal series (all data)...')
    all_series = db.get_temporal_series('43REQ')
    print(f'   Total series count: {len(all_series)}')
    assert len(all_series) == 3, "Should have 3 records!"
    
    print('\n2. Getting temporal series (real only)...')
    real_series = db.get_temporal_series('43REQ', synthetic=False)
    print(f'   Real series count: {len(real_series)}')
    assert len(real_series) == 2, "Should have 2 real records!"
    print('   ✓ get_temporal_series filtering works correctly')
    
    # Test database stats
    print('\n--- Testing get_database_stats ---')
    stats = db.get_database_stats()
    print(f'\nDatabase Statistics:')
    print(f'  Total imagery: {stats["imagery_count"]}')
    print(f'  Real imagery: {stats["real_imagery_count"]}')
    print(f'  Synthetic imagery: {stats["synthetic_imagery_count"]}')
    print(f'  Date range (all): {stats["date_range"]}')
    print(f'  Date range (real): {stats["real_date_range"]}')
    
    assert stats['imagery_count'] == 3, "Should have 3 total records!"
    assert stats['real_imagery_count'] == 2, "Should have 2 real records!"
    assert stats['synthetic_imagery_count'] == 1, "Should have 1 synthetic record!"
    print('\n✓ get_database_stats includes real data counts')
    
    # Clean up
    conn.close()
    Path('/tmp/test_agriflux.db').unlink()
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    test_database_updates()
