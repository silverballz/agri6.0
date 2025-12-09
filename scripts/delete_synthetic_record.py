"""
Delete Synthetic Record

Deletes the old synthetic record (ID 1) from the database.

Usage:
    python scripts/delete_synthetic_record.py --db-path data/agriflux.db
"""

import sys
import logging
import argparse
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_synthetic_record(db_path: Path):
    """Delete the synthetic record from database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if record exists
    cursor.execute(
        "SELECT id, acquisition_date, tile_id FROM processed_imagery WHERE id = 1"
    )
    record = cursor.fetchone()
    
    if record:
        logger.info(f"Found record: ID={record[0]}, Date={record[1]}, Tile={record[2]}")
        
        # Delete the record
        cursor.execute("DELETE FROM processed_imagery WHERE id = 1")
        conn.commit()
        
        logger.info(f"✓ Deleted record ID 1")
    else:
        logger.info("Record ID 1 not found")
    
    conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Delete synthetic record from database'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/agriflux.db',
        help='Path to SQLite database (default: data/agriflux.db)'
    )
    
    args = parser.parse_args()
    
    try:
        delete_synthetic_record(Path(args.db_path))
        logger.info("\n✓ Synthetic record deleted successfully")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
