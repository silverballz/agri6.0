"""
Database migration scripts for schema creation and updates.
"""

from typing import List, Dict, Any
import logging
from .connection import DatabaseConnection

logger = logging.getLogger(__name__)


class DatabaseMigrations:
    """
    Manages database schema migrations and updates.
    """
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize migrations with database connection.
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db = db_connection
    
    def get_current_version(self) -> int:
        """
        Get the current database schema version.
        
        Returns:
            Current schema version number
        """
        # Create migrations table if it doesn't exist
        if not self.db.table_exists('schema_migrations'):
            self._create_migrations_table()
            return 0
        
        query = "SELECT MAX(version) as version FROM schema_migrations"
        result = self.db.execute_query(query)
        
        if result and result[0]['version'] is not None:
            return result[0]['version']
        return 0
    
    def _create_migrations_table(self):
        """Create the schema migrations tracking table."""
        query = """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.db.execute_update(query)
        logger.info("Created schema_migrations table")
    
    def _record_migration(self, version: int, description: str):
        """
        Record a completed migration.
        
        Args:
            version: Migration version number
            description: Migration description
        """
        query = """
        INSERT INTO schema_migrations (version, description)
        VALUES (?, ?)
        """
        self.db.execute_update(query, (version, description))
        logger.info(f"Recorded migration v{version}: {description}")
    
    def migrate_to_latest(self):
        """Apply all pending migrations to bring database to latest schema."""
        current_version = self.get_current_version()
        migrations = self._get_migrations()
        
        for version, description, migration_func in migrations:
            if version > current_version:
                logger.info(f"Applying migration v{version}: {description}")
                try:
                    migration_func()
                    self._record_migration(version, description)
                    logger.info(f"Successfully applied migration v{version}")
                except Exception as e:
                    logger.error(f"Failed to apply migration v{version}: {e}")
                    raise
        
        final_version = self.get_current_version()
        logger.info(f"Database schema is now at version {final_version}")
    
    def _get_migrations(self) -> List[tuple]:
        """
        Get list of all available migrations.
        
        Returns:
            List of (version, description, function) tuples
        """
        return [
            (1, "Create initial schema", self._migration_001_initial_schema),
            (2, "Add indexes for performance", self._migration_002_add_indexes),
            (3, "Add spatial indexes", self._migration_003_spatial_indexes),
        ]
    
    def _migration_001_initial_schema(self):
        """Migration 001: Create initial database schema."""
        
        # Satellite Images table
        satellite_images_sql = """
        CREATE TABLE satellite_images (
            id TEXT PRIMARY KEY,
            acquisition_date TIMESTAMP NOT NULL,
            tile_id TEXT NOT NULL,
            cloud_coverage REAL NOT NULL CHECK (cloud_coverage >= 0 AND cloud_coverage <= 100),
            geometry_wkt TEXT NOT NULL,
            quality_flags TEXT NOT NULL,
            bands_data BLOB NOT NULL,
            indices_data BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Monitoring Zones table
        monitoring_zones_sql = """
        CREATE TABLE monitoring_zones (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            geometry_wkt TEXT NOT NULL,
            crop_type TEXT NOT NULL,
            planting_date TIMESTAMP NOT NULL,
            expected_harvest TIMESTAMP NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CHECK (expected_harvest > planting_date)
        )
        """
        
        # Sensor Locations table
        sensor_locations_sql = """
        CREATE TABLE sensor_locations (
            id TEXT PRIMARY KEY,
            zone_id TEXT NOT NULL,
            sensor_type TEXT NOT NULL,
            latitude REAL NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
            longitude REAL NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
            installation_date TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (zone_id) REFERENCES monitoring_zones (id) ON DELETE CASCADE
        )
        """
        
        # Alerts table
        alerts_sql = """
        CREATE TABLE alerts (
            id TEXT PRIMARY KEY,
            zone_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
            message TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            acknowledged_at TIMESTAMP,
            resolved_at TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (zone_id) REFERENCES monitoring_zones (id) ON DELETE CASCADE,
            CHECK (acknowledged_at IS NULL OR acknowledged_at >= created_at),
            CHECK (resolved_at IS NULL OR resolved_at >= created_at)
        )
        """
        
        # Index Time Series table
        index_timeseries_sql = """
        CREATE TABLE index_timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id TEXT NOT NULL,
            index_type TEXT NOT NULL CHECK (index_type IN ('NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI', 'GNDVI')),
            timestamp TIMESTAMP NOT NULL,
            mean_value REAL NOT NULL,
            std_deviation REAL NOT NULL CHECK (std_deviation >= 0),
            pixel_count INTEGER NOT NULL CHECK (pixel_count >= 0),
            quality_score REAL NOT NULL CHECK (quality_score >= 0 AND quality_score <= 1),
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (zone_id) REFERENCES monitoring_zones (id) ON DELETE CASCADE,
            UNIQUE (zone_id, index_type, timestamp)
        )
        """
        
        # Execute all table creation statements
        tables = [
            ("satellite_images", satellite_images_sql),
            ("monitoring_zones", monitoring_zones_sql),
            ("sensor_locations", sensor_locations_sql),
            ("alerts", alerts_sql),
            ("index_timeseries", index_timeseries_sql)
        ]
        
        for table_name, sql in tables:
            self.db.execute_update(sql)
            logger.info(f"Created table: {table_name}")
    
    def _migration_002_add_indexes(self):
        """Migration 002: Add performance indexes."""
        
        indexes = [
            # Satellite images indexes
            "CREATE INDEX idx_satellite_images_tile_date ON satellite_images (tile_id, acquisition_date)",
            "CREATE INDEX idx_satellite_images_cloud_coverage ON satellite_images (cloud_coverage)",
            
            # Monitoring zones indexes
            "CREATE INDEX idx_monitoring_zones_crop_type ON monitoring_zones (crop_type)",
            "CREATE INDEX idx_monitoring_zones_planting_date ON monitoring_zones (planting_date)",
            
            # Sensor locations indexes
            "CREATE INDEX idx_sensor_locations_zone_id ON sensor_locations (zone_id)",
            "CREATE INDEX idx_sensor_locations_type ON sensor_locations (sensor_type)",
            "CREATE INDEX idx_sensor_locations_active ON sensor_locations (is_active)",
            
            # Alerts indexes
            "CREATE INDEX idx_alerts_zone_id ON alerts (zone_id)",
            "CREATE INDEX idx_alerts_severity ON alerts (severity)",
            "CREATE INDEX idx_alerts_created_at ON alerts (created_at)",
            "CREATE INDEX idx_alerts_active ON alerts (resolved_at) WHERE resolved_at IS NULL",
            
            # Index timeseries indexes
            "CREATE INDEX idx_index_timeseries_zone_type ON index_timeseries (zone_id, index_type)",
            "CREATE INDEX idx_index_timeseries_timestamp ON index_timeseries (timestamp)",
            "CREATE INDEX idx_index_timeseries_quality ON index_timeseries (quality_score)",
        ]
        
        for index_sql in indexes:
            self.db.execute_update(index_sql)
            logger.info(f"Created index: {index_sql.split()[2]}")
    
    def _migration_003_spatial_indexes(self):
        """Migration 003: Add spatial indexes (if SpatiaLite is available)."""
        
        try:
            # Check if SpatiaLite functions are available
            test_query = "SELECT InitSpatialMetaData(1)"
            self.db.execute_query(test_query)
            
            # Create spatial indexes for geometry columns
            spatial_indexes = [
                "SELECT CreateSpatialIndex('satellite_images', 'geometry_wkt')",
                "SELECT CreateSpatialIndex('monitoring_zones', 'geometry_wkt')",
            ]
            
            for index_sql in spatial_indexes:
                try:
                    self.db.execute_query(index_sql)
                    logger.info(f"Created spatial index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Could not create spatial index: {e}")
            
        except Exception as e:
            logger.warning(f"SpatiaLite not available, skipping spatial indexes: {e}")
    
    def reset_database(self):
        """
        Drop all tables and recreate the schema.
        WARNING: This will delete all data!
        """
        logger.warning("Resetting database - all data will be lost!")
        
        # Get list of all tables
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = self.db.execute_query(query)
        
        # Drop all tables
        for table in tables:
            table_name = table['name']
            if table_name != 'sqlite_sequence':  # Don't drop SQLite system table
                drop_sql = f"DROP TABLE IF EXISTS {table_name}"
                self.db.execute_update(drop_sql)
                logger.info(f"Dropped table: {table_name}")
        
        # Recreate schema
        self.migrate_to_latest()
        logger.info("Database reset complete")
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of applied migrations.
        
        Returns:
            List of migration records
        """
        if not self.db.table_exists('schema_migrations'):
            return []
        
        query = """
        SELECT version, description, applied_at
        FROM schema_migrations
        ORDER BY version
        """
        
        results = self.db.execute_query(query)
        
        return [
            {
                'version': row['version'],
                'description': row['description'],
                'applied_at': row['applied_at']
            }
            for row in results
        ]