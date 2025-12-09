"""
Unit tests for database functionality.
"""

import pytest
import tempfile
import os
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Polygon

from src.database import DatabaseConnection, DatabaseMigrations
from src.models import (
    SatelliteImage, MonitoringZone, SensorLocation, Alert, IndexTimeSeries
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def db_connection(temp_db):
    """Create a database connection for testing."""
    conn = DatabaseConnection(temp_db)
    yield conn
    conn.disconnect()


@pytest.fixture
def db_models(db_connection):
    """Create database models with migrated schema."""
    migrations = DatabaseMigrations(db_connection)
    migrations.migrate_to_latest()
    return DatabaseModels(db_connection)


@pytest.fixture
def sample_satellite_image():
    """Create a sample satellite image for testing."""
    bands = {
        'B02': np.random.rand(10, 10),
        'B03': np.random.rand(10, 10),
        'B04': np.random.rand(10, 10),
        'B08': np.random.rand(10, 10),
        'B11': np.random.rand(10, 10),
        'B12': np.random.rand(10, 10)
    }
    
    indices = {
        'NDVI': np.random.rand(10, 10),
        'SAVI': np.random.rand(10, 10)
    }
    
    geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    
    return SatelliteImage(
        id="test_image_001",
        acquisition_date=datetime.now(),
        tile_id="T43REQ",
        cloud_coverage=15.5,
        bands=bands,
        indices=indices,
        geometry=geometry,
        quality_flags={'atmospheric_correction': True}
    )


@pytest.fixture
def sample_monitoring_zone():
    """Create a sample monitoring zone for testing."""
    geometry = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    sensor = SensorLocation(
        id="sensor_001",
        sensor_type="soil_moisture",
        latitude=5.0,
        longitude=5.0,
        installation_date=datetime.now()
    )
    
    alert = Alert(
        id="alert_001",
        alert_type="pest_risk",
        severity="medium",
        message="Test alert",
        created_at=datetime.now()
    )
    
    return MonitoringZone(
        id="zone_001",
        name="Test Field",
        geometry=geometry,
        crop_type="wheat",
        planting_date=datetime(2024, 3, 15),
        expected_harvest=datetime(2024, 8, 15),
        sensors=[sensor],
        alerts=[alert]
    )


class TestDatabaseConnection:
    """Test database connection functionality."""
    
    def test_connection_creation(self, temp_db):
        """Test database connection creation."""
        conn = DatabaseConnection(temp_db)
        
        # Test connection
        db_conn = conn.connect()
        assert db_conn is not None
        
        # Test database file creation
        assert os.path.exists(temp_db)
        
        conn.disconnect()
    
    def test_table_operations(self, db_connection):
        """Test table existence checking."""
        # Initially no tables should exist
        assert not db_connection.table_exists('test_table')
        
        # Create a test table
        create_sql = "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
        db_connection.execute_update(create_sql)
        
        # Now table should exist
        assert db_connection.table_exists('test_table')
        
        # Test schema retrieval
        schema = db_connection.get_table_schema('test_table')
        assert len(schema) == 2
        assert schema[0]['name'] == 'id'
        assert schema[1]['name'] == 'name'
    
    def test_query_operations(self, db_connection):
        """Test query execution."""
        # Create test table
        create_sql = "CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)"
        db_connection.execute_update(create_sql)
        
        # Insert data
        insert_sql = "INSERT INTO test_data (value) VALUES (?)"
        affected = db_connection.execute_update(insert_sql, ("test_value",))
        assert affected == 1
        
        # Query data
        select_sql = "SELECT * FROM test_data WHERE value = ?"
        results = db_connection.execute_query(select_sql, ("test_value",))
        assert len(results) == 1
        assert results[0]['value'] == "test_value"
    
    def test_transaction_rollback(self, db_connection):
        """Test transaction rollback on error."""
        # Create test table
        create_sql = "CREATE TABLE test_rollback (id INTEGER PRIMARY KEY UNIQUE)"
        db_connection.execute_update(create_sql)
        
        # Insert initial data
        db_connection.execute_update("INSERT INTO test_rollback (id) VALUES (1)")
        
        # Try to insert duplicate (should fail and rollback)
        with pytest.raises(Exception):
            with db_connection.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO test_rollback (id) VALUES (2)")
                cursor.execute("INSERT INTO test_rollback (id) VALUES (1)")  # Duplicate, should fail
        
        # Verify rollback - only original record should exist
        results = db_connection.execute_query("SELECT COUNT(*) as count FROM test_rollback")
        assert results[0]['count'] == 1


class TestDatabaseMigrations:
    """Test database migration functionality."""
    
    def test_initial_migration(self, db_connection):
        """Test initial schema migration."""
        migrations = DatabaseMigrations(db_connection)
        
        # Initially version should be 0
        assert migrations.get_current_version() == 0
        
        # Apply migrations
        migrations.migrate_to_latest()
        
        # Version should be updated
        current_version = migrations.get_current_version()
        assert current_version > 0
        
        # All required tables should exist
        required_tables = [
            'satellite_images', 'monitoring_zones', 'sensor_locations',
            'alerts', 'index_timeseries', 'schema_migrations'
        ]
        
        for table in required_tables:
            assert db_connection.table_exists(table)
    
    def test_migration_history(self, db_connection):
        """Test migration history tracking."""
        migrations = DatabaseMigrations(db_connection)
        migrations.migrate_to_latest()
        
        history = migrations.get_migration_history()
        assert len(history) > 0
        
        # Check first migration
        first_migration = history[0]
        assert first_migration['version'] == 1
        assert 'initial schema' in first_migration['description'].lower()
        assert first_migration['applied_at'] is not None


class TestDatabaseModels:
    """Test database model CRUD operations."""
    
    def test_satellite_image_crud(self, db_models, sample_satellite_image):
        """Test satellite image CRUD operations."""
        # Create
        image_id = db_models.create_satellite_image(sample_satellite_image)
        assert image_id == sample_satellite_image.id
        
        # Read
        retrieved_image = db_models.get_satellite_image(image_id)
        assert retrieved_image is not None
        assert retrieved_image.id == sample_satellite_image.id
        assert retrieved_image.tile_id == sample_satellite_image.tile_id
        assert retrieved_image.cloud_coverage == sample_satellite_image.cloud_coverage
        
        # Verify numpy arrays are preserved
        assert np.array_equal(retrieved_image.bands['B04'], sample_satellite_image.bands['B04'])
        assert np.array_equal(retrieved_image.indices['NDVI'], sample_satellite_image.indices['NDVI'])
        
        # List
        images = db_models.list_satellite_images()
        assert len(images) == 1
        assert images[0]['id'] == image_id
        
        # List with filters
        filtered_images = db_models.list_satellite_images(
            tile_id="T43REQ",
            max_cloud_coverage=20.0
        )
        assert len(filtered_images) == 1
        
        # Delete
        deleted = db_models.delete_satellite_image(image_id)
        assert deleted is True
        
        # Verify deletion
        retrieved_image = db_models.get_satellite_image(image_id)
        assert retrieved_image is None
    
    def test_monitoring_zone_crud(self, db_models, sample_monitoring_zone):
        """Test monitoring zone CRUD operations."""
        # Create
        zone_id = db_models.create_monitoring_zone(sample_monitoring_zone)
        assert zone_id == sample_monitoring_zone.id
        
        # Read
        retrieved_zone = db_models.get_monitoring_zone(zone_id)
        assert retrieved_zone is not None
        assert retrieved_zone.id == sample_monitoring_zone.id
        assert retrieved_zone.name == sample_monitoring_zone.name
        assert retrieved_zone.crop_type == sample_monitoring_zone.crop_type
        
        # Verify sensors and alerts were created
        assert len(retrieved_zone.sensors) == 1
        assert len(retrieved_zone.alerts) == 1
        assert retrieved_zone.sensors[0].id == "sensor_001"
        assert retrieved_zone.alerts[0].id == "alert_001"
        
        # List
        zones = db_models.list_monitoring_zones()
        assert len(zones) == 1
        assert zones[0]['id'] == zone_id
        
        # Update
        sample_monitoring_zone.name = "Updated Field Name"
        updated = db_models.update_monitoring_zone(sample_monitoring_zone)
        assert updated is True
        
        # Verify update
        retrieved_zone = db_models.get_monitoring_zone(zone_id)
        assert retrieved_zone.name == "Updated Field Name"
        
        # Delete
        deleted = db_models.delete_monitoring_zone(zone_id)
        assert deleted is True
        
        # Verify deletion
        retrieved_zone = db_models.get_monitoring_zone(zone_id)
        assert retrieved_zone is None
    
    def test_index_timeseries_crud(self, db_models, sample_monitoring_zone):
        """Test index time series CRUD operations."""
        # First create a monitoring zone
        db_models.create_monitoring_zone(sample_monitoring_zone)
        
        # Create index measurements
        measurements = []
        for i in range(5):
            measurement = IndexTimeSeries(
                zone_id=sample_monitoring_zone.id,
                index_type="NDVI",
                timestamp=datetime.now() - timedelta(days=i),
                mean_value=0.7 + i * 0.05,
                std_deviation=0.1,
                pixel_count=1000,
                quality_score=0.9
            )
            measurements.append(measurement)
            db_models.create_index_measurement(measurement)
        
        # Read time series
        timeseries = db_models.get_index_timeseries(
            sample_monitoring_zone.id, 
            "NDVI"
        )
        assert len(timeseries) == 5
        
        # Test date filtering
        cutoff_date = datetime.now() - timedelta(days=2)
        recent_timeseries = db_models.get_index_timeseries(
            sample_monitoring_zone.id,
            "NDVI",
            start_date=cutoff_date
        )
        assert len(recent_timeseries) <= 3
        
        # Get latest measurement
        latest = db_models.get_latest_index_measurement(
            sample_monitoring_zone.id,
            "NDVI"
        )
        assert latest is not None
        assert latest.mean_value == 0.7  # Most recent (i=0)
    
    def test_alert_operations(self, db_models, sample_monitoring_zone):
        """Test alert-specific operations."""
        # Create monitoring zone
        db_models.create_monitoring_zone(sample_monitoring_zone)
        
        # Get alerts for zone
        alerts = db_models.get_alerts_for_zone(sample_monitoring_zone.id)
        assert len(alerts) == 1
        
        # Get only active alerts
        active_alerts = db_models.get_alerts_for_zone(sample_monitoring_zone.id, active_only=True)
        assert len(active_alerts) == 1
        
        # Update alert (resolve it)
        alert = alerts[0]
        alert.resolve()
        updated = db_models.update_alert(alert)
        assert updated is True
        
        # Verify no active alerts remain
        active_alerts = db_models.get_alerts_for_zone(sample_monitoring_zone.id, active_only=True)
        assert len(active_alerts) == 0


class TestDatabaseIntegration:
    """Test database integration scenarios."""
    
    def test_complete_workflow(self, db_models):
        """Test a complete workflow with all data types."""
        # Create monitoring zone
        geometry = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        zone = MonitoringZone(
            id="integration_zone",
            name="Integration Test Field",
            geometry=geometry,
            crop_type="corn",
            planting_date=datetime(2024, 4, 1),
            expected_harvest=datetime(2024, 9, 1)
        )
        
        db_models.create_monitoring_zone(zone)
        
        # Create satellite image
        bands = {f'B{i:02d}': np.random.rand(5, 5) for i in [2, 3, 4, 8, 11, 12]}
        indices = {'NDVI': np.random.rand(5, 5)}
        
        image = SatelliteImage(
            id="integration_image",
            acquisition_date=datetime.now(),
            tile_id="T43REQ",
            cloud_coverage=10.0,
            bands=bands,
            indices=indices,
            geometry=geometry,
            quality_flags={'test': True}
        )
        
        db_models.create_satellite_image(image)
        
        # Create time series data
        for i in range(3):
            measurement = IndexTimeSeries(
                zone_id=zone.id,
                index_type="NDVI",
                timestamp=datetime.now() - timedelta(days=i*7),
                mean_value=0.8 - i*0.1,
                std_deviation=0.05,
                pixel_count=500,
                quality_score=0.95
            )
            db_models.create_index_measurement(measurement)
        
        # Verify all data exists
        retrieved_zone = db_models.get_monitoring_zone(zone.id)
        assert retrieved_zone is not None
        
        retrieved_image = db_models.get_satellite_image(image.id)
        assert retrieved_image is not None
        
        timeseries = db_models.get_index_timeseries(zone.id, "NDVI")
        assert len(timeseries) == 3
        
        # Test cleanup (cascading deletes)
        db_models.delete_monitoring_zone(zone.id)
        
        # Verify zone and associated data are deleted
        assert db_models.get_monitoring_zone(zone.id) is None
        timeseries_after_delete = db_models.get_index_timeseries(zone.id, "NDVI")
        assert len(timeseries_after_delete) == 0


if __name__ == "__main__":
    pytest.main([__file__])