#!/usr/bin/env python3
"""
Verification script for Task 18: Update database queries to prioritize real data

This script demonstrates that all requirements are met:
- Requirement 3.4: System SHALL distinguish between real and synthetic data sources
- Requirement 3.5: System SHALL prioritize real data over synthetic data

Task 18 Requirements:
✓ Modify get_latest_imagery to prefer real data
✓ Update list_processed_imagery to filter by synthetic flag
✓ Add query methods to distinguish real vs synthetic
✓ Update database statistics to show real data count
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.db_manager import DatabaseManager


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def setup_test_database():
    """Create a test database with both real and synthetic data."""
    # Create temporary database
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_path = temp_file.name
    temp_file.close()
    
    db = DatabaseManager(db_path)
    db.init_database()
    
    print_section("Setting up test database")
    
    # Add synthetic data (older dates)
    print("Adding 5 synthetic imagery records...")
    for i in range(5):
        date = (datetime.now() - timedelta(days=30 + i)).strftime('%Y-%m-%d')
        db.save_processed_imagery(
            acquisition_date=date,
            tile_id='43REQ',
            cloud_coverage=15.0 + i,
            geotiff_paths={
                'NDVI': f'/data/synthetic/ndvi_{date}.tif',
                'SAVI': f'/data/synthetic/savi_{date}.tif',
                'EVI': f'/data/synthetic/evi_{date}.tif',
            },
            metadata={'source': 'Synthetic Generator', 'type': 'synthetic'},
            synthetic=True
        )
        print(f"  ✓ Synthetic imagery: {date} (cloud: {15.0 + i}%)")
    
    # Add real data (more recent dates)
    print("\nAdding 3 real imagery records...")
    for i in range(3):
        date = (datetime.now() - timedelta(days=5 + i)).strftime('%Y-%m-%d')
        db.save_processed_imagery(
            acquisition_date=date,
            tile_id='43REQ',
            cloud_coverage=8.0 + i,
            geotiff_paths={
                'NDVI': f'/data/real/ndvi_{date}.tif',
                'SAVI': f'/data/real/savi_{date}.tif',
                'EVI': f'/data/real/evi_{date}.tif',
            },
            metadata={'source': 'Sentinel Hub API', 'type': 'real'},
            synthetic=False
        )
        print(f"  ✓ Real imagery: {date} (cloud: {8.0 + i}%)")
    
    print(f"\n✓ Test database created at: {db_path}")
    return db, db_path


def test_get_latest_imagery(db):
    """Test Requirement 3.5: Prioritize real data over synthetic data."""
    print_section("Test 1: get_latest_imagery() prioritizes real data")
    
    print("Testing with prefer_real=True (default):")
    latest = db.get_latest_imagery(prefer_real=True)
    
    if latest:
        is_real = latest['synthetic'] == 0
        print(f"  ✓ Latest imagery ID: {latest['id']}")
        print(f"  ✓ Acquisition date: {latest['acquisition_date']}")
        print(f"  ✓ Data type: {'REAL' if is_real else 'SYNTHETIC'}")
        print(f"  ✓ Cloud coverage: {latest['cloud_coverage']}%")
        
        if is_real:
            print(f"\n  ✅ SUCCESS: Real data prioritized (Requirement 3.5)")
        else:
            print(f"\n  ❌ FAILED: Synthetic data returned instead of real")
            return False
    else:
        print("  ❌ No imagery found")
        return False
    
    return True


def test_list_processed_imagery(db):
    """Test Requirement 3.4: Distinguish between real and synthetic data."""
    print_section("Test 2: list_processed_imagery() filters by synthetic flag")
    
    # Test filtering for real data only
    print("Testing synthetic=False (real data only):")
    real_imagery = db.list_processed_imagery(synthetic=False)
    print(f"  ✓ Found {len(real_imagery)} real imagery records")
    
    all_real = all(img['synthetic'] == 0 for img in real_imagery)
    if all_real and len(real_imagery) == 3:
        print(f"  ✅ SUCCESS: All records are real data (Requirement 3.4)")
    else:
        print(f"  ❌ FAILED: Expected 3 real records, got {len(real_imagery)}")
        return False
    
    # Test filtering for synthetic data only
    print("\nTesting synthetic=True (synthetic data only):")
    synthetic_imagery = db.list_processed_imagery(synthetic=True)
    print(f"  ✓ Found {len(synthetic_imagery)} synthetic imagery records")
    
    all_synthetic = all(img['synthetic'] == 1 for img in synthetic_imagery)
    if all_synthetic and len(synthetic_imagery) == 5:
        print(f"  ✅ SUCCESS: All records are synthetic data (Requirement 3.4)")
    else:
        print(f"  ❌ FAILED: Expected 5 synthetic records, got {len(synthetic_imagery)}")
        return False
    
    # Test no filter returns all
    print("\nTesting synthetic=None (all data):")
    all_imagery = db.list_processed_imagery(synthetic=None)
    print(f"  ✓ Found {len(all_imagery)} total imagery records")
    
    if len(all_imagery) == 8:
        print(f"  ✅ SUCCESS: Returns all data when no filter applied")
    else:
        print(f"  ❌ FAILED: Expected 8 total records, got {len(all_imagery)}")
        return False
    
    return True


def test_query_methods(db):
    """Test query methods to distinguish real vs synthetic."""
    print_section("Test 3: Query methods distinguish real vs synthetic")
    
    # Test get_real_imagery()
    print("Testing get_real_imagery():")
    real_imagery = db.get_real_imagery()
    print(f"  ✓ Found {len(real_imagery)} real imagery records")
    
    if len(real_imagery) == 3 and all(img['synthetic'] == 0 for img in real_imagery):
        print(f"  ✅ SUCCESS: get_real_imagery() returns only real data")
    else:
        print(f"  ❌ FAILED: get_real_imagery() returned incorrect data")
        return False
    
    # Test get_synthetic_imagery()
    print("\nTesting get_synthetic_imagery():")
    synthetic_imagery = db.get_synthetic_imagery()
    print(f"  ✓ Found {len(synthetic_imagery)} synthetic imagery records")
    
    if len(synthetic_imagery) == 5 and all(img['synthetic'] == 1 for img in synthetic_imagery):
        print(f"  ✅ SUCCESS: get_synthetic_imagery() returns only synthetic data")
    else:
        print(f"  ❌ FAILED: get_synthetic_imagery() returned incorrect data")
        return False
    
    # Test count methods
    print("\nTesting count_real_imagery():")
    real_count = db.count_real_imagery()
    print(f"  ✓ Real imagery count: {real_count}")
    
    print("\nTesting count_synthetic_imagery():")
    synthetic_count = db.count_synthetic_imagery()
    print(f"  ✓ Synthetic imagery count: {synthetic_count}")
    
    if real_count == 3 and synthetic_count == 5:
        print(f"\n  ✅ SUCCESS: Count methods return correct values")
    else:
        print(f"\n  ❌ FAILED: Count methods returned incorrect values")
        return False
    
    return True


def test_database_statistics(db):
    """Test database statistics show real data count."""
    print_section("Test 4: Database statistics show real data count")
    
    stats = db.get_database_stats()
    
    print("Database Statistics:")
    print(f"  Total imagery: {stats['imagery_count']}")
    print(f"  Real imagery: {stats['real_imagery_count']}")
    print(f"  Synthetic imagery: {stats['synthetic_imagery_count']}")
    print(f"\n  Overall date range:")
    print(f"    Earliest: {stats['date_range']['earliest']}")
    print(f"    Latest: {stats['date_range']['latest']}")
    print(f"\n  Real data date range:")
    print(f"    Earliest: {stats['real_date_range']['earliest']}")
    print(f"    Latest: {stats['real_date_range']['latest']}")
    
    # Verify statistics
    checks = [
        (stats['imagery_count'] == 8, "Total imagery count"),
        (stats['real_imagery_count'] == 3, "Real imagery count"),
        (stats['synthetic_imagery_count'] == 5, "Synthetic imagery count"),
        ('real_date_range' in stats, "Real date range included"),
        (stats['real_date_range']['earliest'] is not None, "Real earliest date"),
        (stats['real_date_range']['latest'] is not None, "Real latest date"),
    ]
    
    all_passed = True
    print("\nVerification:")
    for passed, description in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {description}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n  ✅ SUCCESS: Database statistics correctly show real data count")
    else:
        print(f"\n  ❌ FAILED: Database statistics incomplete or incorrect")
    
    return all_passed


def test_temporal_series_filtering(db):
    """Test temporal series can filter by synthetic flag."""
    print_section("Test 5: Temporal series filtering")
    
    print("Testing get_temporal_series() with synthetic=False:")
    real_series = db.get_temporal_series(tile_id='43REQ', synthetic=False)
    print(f"  ✓ Found {len(real_series)} real imagery records")
    
    # Verify all are real and ordered by date
    all_real = all(img['synthetic'] == 0 for img in real_series)
    dates = [img['acquisition_date'] for img in real_series]
    is_ordered = dates == sorted(dates)
    
    if all_real and is_ordered and len(real_series) == 3:
        print(f"  ✅ SUCCESS: Temporal series correctly filters and orders real data")
    else:
        print(f"  ❌ FAILED: Temporal series filtering or ordering incorrect")
        return False
    
    print("\nTesting get_temporal_series() with synthetic=True:")
    synthetic_series = db.get_temporal_series(tile_id='43REQ', synthetic=True)
    print(f"  ✓ Found {len(synthetic_series)} synthetic imagery records")
    
    all_synthetic = all(img['synthetic'] == 1 for img in synthetic_series)
    if all_synthetic and len(synthetic_series) == 5:
        print(f"  ✅ SUCCESS: Temporal series correctly filters synthetic data")
    else:
        print(f"  ❌ FAILED: Temporal series synthetic filtering incorrect")
        return False
    
    return True


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("  TASK 18 VERIFICATION: Database Queries Prioritize Real Data")
    print("="*70)
    
    # Setup test database
    db, db_path = setup_test_database()
    
    # Run all tests
    tests = [
        ("get_latest_imagery prioritizes real data", test_get_latest_imagery),
        ("list_processed_imagery filters by synthetic flag", test_list_processed_imagery),
        ("Query methods distinguish real vs synthetic", test_query_methods),
        ("Database statistics show real data count", test_database_statistics),
        ("Temporal series filtering", test_temporal_series_filtering),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func(db)
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n  ❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_section("VERIFICATION SUMMARY")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✅ ALL TESTS PASSED - Task 18 Complete!")
        print("\n  Requirements Validated:")
        print("    ✓ Requirement 3.4: System distinguishes real vs synthetic data")
        print("    ✓ Requirement 3.5: System prioritizes real data over synthetic")
        print("\n  Task 18 Deliverables:")
        print("    ✓ get_latest_imagery() prefers real data")
        print("    ✓ list_processed_imagery() filters by synthetic flag")
        print("    ✓ Query methods distinguish real vs synthetic")
        print("    ✓ Database statistics show real data count")
    else:
        print("  ❌ SOME TESTS FAILED - Review output above")
    print("="*70 + "\n")
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
