#!/usr/bin/env python3
"""
Database Verification Script

Performs comprehensive verification of the AgriFlux database.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager


def main():
    """Verify database contents."""
    db_manager = DatabaseManager('data/agriflux.db')
    
    print("="*60)
    print("DATABASE VERIFICATION")
    print("="*60)
    
    # Get statistics
    stats = db_manager.get_database_stats()
    print(f"\n📊 Database Statistics:")
    print(f"  Imagery Records: {stats['imagery_count']}")
    print(f"  Total Alerts: {stats['total_alerts']}")
    print(f"  Active Alerts: {stats['active_alerts']}")
    print(f"  AI Predictions: {stats['predictions_count']}")
    print(f"  Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    
    # List all imagery
    print(f"\n📷 Processed Imagery Records:")
    imagery_list = db_manager.list_processed_imagery()
    for img in imagery_list:
        print(f"\n  ID: {img['id']}")
        print(f"  Tile: {img['tile_id']}")
        print(f"  Date: {img['acquisition_date']}")
        print(f"  Cloud Coverage: {img['cloud_coverage']:.2f}%")
        print(f"  Processed: {img['processed_at']}")
        
        # Check which indices are available
        indices = []
        for field in ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path', 'ndsi_path']:
            if img.get(field):
                index_name = field.replace('_path', '').upper()
                indices.append(index_name)
        print(f"  Indices: {', '.join(indices)}")
        
        # Verify files exist
        print(f"  File Verification:")
        for field in ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path', 'ndsi_path']:
            if img.get(field):
                path = Path(img[field])
                exists = "✓" if path.exists() else "✗"
                size_mb = path.stat().st_size / (1024*1024) if path.exists() else 0
                print(f"    {exists} {path.name} ({size_mb:.1f} MB)")
    
    # Test temporal series query
    print(f"\n📈 Temporal Series Query Test:")
    if imagery_list:
        tile_id = imagery_list[0]['tile_id']
        series = db_manager.get_temporal_series(tile_id)
        print(f"  Found {len(series)} records for tile {tile_id}")
    
    # Test latest imagery query
    print(f"\n🔍 Latest Imagery Query Test:")
    latest = db_manager.get_latest_imagery()
    if latest:
        print(f"  Latest imagery: Tile {latest['tile_id']}, Date {latest['acquisition_date']}")
    
    print("\n" + "="*60)
    print("✅ Verification Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
