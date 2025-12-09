#!/usr/bin/env python3
"""
Migration script to update synthetic flag in database based on metadata.
"""

import sqlite3
import json
from pathlib import Path


def migrate_synthetic_flag(db_path: str = 'data/agriflux.db'):
    """
    Update synthetic flag in processed_imagery table based on metadata.
    """
    print("=" * 60)
    print("Migrating synthetic flag from metadata to column")
    print("=" * 60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all records
    cursor.execute('SELECT id, metadata_json FROM processed_imagery')
    rows = cursor.fetchall()
    
    print(f"\nFound {len(rows)} records to process")
    
    updated_count = 0
    real_count = 0
    synthetic_count = 0
    
    for record_id, metadata_json in rows:
        if not metadata_json:
            print(f"  Record {record_id}: No metadata, keeping as synthetic")
            synthetic_count += 1
            continue
        
        try:
            metadata = json.loads(metadata_json)
            
            # Check if metadata has synthetic flag
            if 'synthetic' in metadata:
                is_synthetic = metadata['synthetic']
                
                # Update database column
                cursor.execute(
                    'UPDATE processed_imagery SET synthetic = ? WHERE id = ?',
                    (1 if is_synthetic else 0, record_id)
                )
                
                if is_synthetic:
                    synthetic_count += 1
                    print(f"  Record {record_id}: Marked as SYNTHETIC")
                else:
                    real_count += 1
                    print(f"  Record {record_id}: Marked as REAL")
                
                updated_count += 1
            else:
                print(f"  Record {record_id}: No synthetic flag in metadata, keeping as synthetic")
                synthetic_count += 1
                
        except json.JSONDecodeError:
            print(f"  Record {record_id}: Invalid JSON metadata, keeping as synthetic")
            synthetic_count += 1
    
    conn.commit()
    
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total records processed: {len(rows)}")
    print(f"Records updated: {updated_count}")
    print(f"Real data records: {real_count}")
    print(f"Synthetic data records: {synthetic_count}")
    
    # Verify the migration
    cursor.execute('SELECT COUNT(*) FROM processed_imagery WHERE synthetic = 0')
    real_in_db = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM processed_imagery WHERE synthetic = 1')
    synthetic_in_db = cursor.fetchone()[0]
    
    print("\nVerification:")
    print(f"Real records in database: {real_in_db}")
    print(f"Synthetic records in database: {synthetic_in_db}")
    
    conn.close()
    
    print("\n✓ Migration completed successfully!")


if __name__ == '__main__':
    migrate_synthetic_flag()
