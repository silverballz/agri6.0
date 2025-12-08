"""
Database models and CRUD operations for the Agricultural Monitoring Platform.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json
import pickle
import numpy as np
from shapely.geometry import Polygon
from shapely import wkt

from .connection import DatabaseConnection
# from ..models import SatelliteImage, MonitoringZone, SensorLocation, Alert, IndexTimeSeries


class DatabaseModels:
    """
    Database access layer with CRUD operations for all data models.
    """
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize database models with connection.
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db = db_connection
    
    # Satellite Image CRUD Operations
    
    def create_satellite_image(self, image: SatelliteImage) -> str:
        """
        Insert a new satellite image record.
        
        Args:
            image: SatelliteImage instance
            
        Returns:
            Image ID
        """
        query = """
        INSERT INTO satellite_images (
            id, acquisition_date, tile_id, cloud_coverage, 
            geometry_wkt, quality_flags, bands_data, indices_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Serialize numpy arrays
        bands_data = pickle.dumps(image.bands)
        indices_data = pickle.dumps(image.indices)
        quality_flags_json = json.dumps(image.quality_flags)
        
        params = (
            image.id,
            image.acquisition_date,
            image.tile_id,
            image.cloud_coverage,
            image.geometry.wkt,
            quality_flags_json,
            bands_data,
            indices_data
        )
        
        self.db.execute_update(query, params)
        return image.id
    
    def get_satellite_image(self, image_id: str) -> Optional[SatelliteImage]:
        """
        Retrieve a satellite image by ID.
        
        Args:
            image_id: Image identifier
            
        Returns:
            SatelliteImage instance or None if not found
        """
        query = """
        SELECT id, acquisition_date, tile_id, cloud_coverage, 
               geometry_wkt, quality_flags, bands_data, indices_data
        FROM satellite_images WHERE id = ?
        """
        
        result = self.db.execute_query(query, (image_id,))
        if not result:
            return None
        
        row = result[0]
        
        # Deserialize data
        bands = pickle.loads(row['bands_data'])
        indices = pickle.loads(row['indices_data'])
        quality_flags = json.loads(row['quality_flags'])
        geometry = wkt.loads(row['geometry_wkt'])
        
        return SatelliteImage(
            id=row['id'],
            acquisition_date=row['acquisition_date'],
            tile_id=row['tile_id'],
            cloud_coverage=row['cloud_coverage'],
            bands=bands,
            indices=indices,
            geometry=geometry,
            quality_flags=quality_flags
        )
    
    def list_satellite_images(self, tile_id: Optional[str] = None, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            max_cloud_coverage: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        List satellite images with optional filters.
        
        Args:
            tile_id: Filter by tile ID
            start_date: Filter by start date
            end_date: Filter by end date
            max_cloud_coverage: Maximum cloud coverage percentage
            
        Returns:
            List of image metadata dictionaries
        """
        query = """
        SELECT id, acquisition_date, tile_id, cloud_coverage, quality_flags
        FROM satellite_images WHERE 1=1
        """
        params = []
        
        if tile_id:
            query += " AND tile_id = ?"
            params.append(tile_id)
        
        if start_date:
            query += " AND acquisition_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND acquisition_date <= ?"
            params.append(end_date)
        
        if max_cloud_coverage is not None:
            query += " AND cloud_coverage <= ?"
            params.append(max_cloud_coverage)
        
        query += " ORDER BY acquisition_date DESC"
        
        results = self.db.execute_query(query, tuple(params))
        
        return [
            {
                'id': row['id'],
                'acquisition_date': row['acquisition_date'],
                'tile_id': row['tile_id'],
                'cloud_coverage': row['cloud_coverage'],
                'quality_flags': json.loads(row['quality_flags'])
            }
            for row in results
        ]
    
    def delete_satellite_image(self, image_id: str) -> bool:
        """
        Delete a satellite image record.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM satellite_images WHERE id = ?"
        affected_rows = self.db.execute_update(query, (image_id,))
        return affected_rows > 0
    
    # Monitoring Zone CRUD Operations
    
    def create_monitoring_zone(self, zone: MonitoringZone) -> str:
        """
        Insert a new monitoring zone record.
        
        Args:
            zone: MonitoringZone instance
            
        Returns:
            Zone ID
        """
        query = """
        INSERT INTO monitoring_zones (
            id, name, geometry_wkt, crop_type, planting_date, 
            expected_harvest, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        metadata_json = json.dumps(zone.metadata)
        
        params = (
            zone.id,
            zone.name,
            zone.geometry.wkt,
            zone.crop_type,
            zone.planting_date,
            zone.expected_harvest,
            metadata_json
        )
        
        self.db.execute_update(query, params)
        
        # Insert sensors
        for sensor in zone.sensors:
            self.create_sensor_location(sensor, zone.id)
        
        # Insert alerts
        for alert in zone.alerts:
            self.create_alert(alert, zone.id)
        
        return zone.id
    
    def get_monitoring_zone(self, zone_id: str) -> Optional[MonitoringZone]:
        """
        Retrieve a monitoring zone by ID.
        
        Args:
            zone_id: Zone identifier
            
        Returns:
            MonitoringZone instance or None if not found
        """
        query = """
        SELECT id, name, geometry_wkt, crop_type, planting_date, 
               expected_harvest, metadata
        FROM monitoring_zones WHERE id = ?
        """
        
        result = self.db.execute_query(query, (zone_id,))
        if not result:
            return None
        
        row = result[0]
        
        # Get sensors and alerts
        sensors = self.get_sensors_for_zone(zone_id)
        alerts = self.get_alerts_for_zone(zone_id)
        
        geometry = wkt.loads(row['geometry_wkt'])
        metadata = json.loads(row['metadata'])
        
        return MonitoringZone(
            id=row['id'],
            name=row['name'],
            geometry=geometry,
            crop_type=row['crop_type'],
            planting_date=row['planting_date'],
            expected_harvest=row['expected_harvest'],
            sensors=sensors,
            alerts=alerts,
            metadata=metadata
        )
    
    def list_monitoring_zones(self) -> List[Dict[str, Any]]:
        """
        List all monitoring zones.
        
        Returns:
            List of zone metadata dictionaries
        """
        query = """
        SELECT id, name, crop_type, planting_date, expected_harvest
        FROM monitoring_zones ORDER BY name
        """
        
        results = self.db.execute_query(query)
        
        return [
            {
                'id': row['id'],
                'name': row['name'],
                'crop_type': row['crop_type'],
                'planting_date': row['planting_date'],
                'expected_harvest': row['expected_harvest']
            }
            for row in results
        ]
    
    def update_monitoring_zone(self, zone: MonitoringZone) -> bool:
        """
        Update a monitoring zone record.
        
        Args:
            zone: MonitoringZone instance
            
        Returns:
            True if updated, False if not found
        """
        query = """
        UPDATE monitoring_zones SET
            name = ?, geometry_wkt = ?, crop_type = ?, 
            planting_date = ?, expected_harvest = ?, metadata = ?
        WHERE id = ?
        """
        
        metadata_json = json.dumps(zone.metadata)
        
        params = (
            zone.name,
            zone.geometry.wkt,
            zone.crop_type,
            zone.planting_date,
            zone.expected_harvest,
            metadata_json,
            zone.id
        )
        
        affected_rows = self.db.execute_update(query, params)
        return affected_rows > 0
    
    def delete_monitoring_zone(self, zone_id: str) -> bool:
        """
        Delete a monitoring zone and all associated data.
        
        Args:
            zone_id: Zone identifier
            
        Returns:
            True if deleted, False if not found
        """
        # Delete associated data first (due to foreign key constraints)
        self.db.execute_update("DELETE FROM sensor_locations WHERE zone_id = ?", (zone_id,))
        self.db.execute_update("DELETE FROM alerts WHERE zone_id = ?", (zone_id,))
        self.db.execute_update("DELETE FROM index_timeseries WHERE zone_id = ?", (zone_id,))
        
        # Delete the zone
        query = "DELETE FROM monitoring_zones WHERE id = ?"
        affected_rows = self.db.execute_update(query, (zone_id,))
        return affected_rows > 0
    
    # Sensor Location CRUD Operations
    
    def create_sensor_location(self, sensor: SensorLocation, zone_id: str) -> str:
        """
        Insert a new sensor location record.
        
        Args:
            sensor: SensorLocation instance
            zone_id: Associated zone ID
            
        Returns:
            Sensor ID
        """
        query = """
        INSERT INTO sensor_locations (
            id, zone_id, sensor_type, latitude, longitude, 
            installation_date, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            sensor.id,
            zone_id,
            sensor.sensor_type,
            sensor.latitude,
            sensor.longitude,
            sensor.installation_date,
            sensor.is_active
        )
        
        self.db.execute_update(query, params)
        return sensor.id
    
    def get_sensors_for_zone(self, zone_id: str) -> List[SensorLocation]:
        """
        Get all sensors for a monitoring zone.
        
        Args:
            zone_id: Zone identifier
            
        Returns:
            List of SensorLocation instances
        """
        query = """
        SELECT id, sensor_type, latitude, longitude, installation_date, is_active
        FROM sensor_locations WHERE zone_id = ?
        """
        
        results = self.db.execute_query(query, (zone_id,))
        
        return [
            SensorLocation(
                id=row['id'],
                sensor_type=row['sensor_type'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                installation_date=row['installation_date'],
                is_active=bool(row['is_active'])
            )
            for row in results
        ]
    
    # Alert CRUD Operations
    
    def create_alert(self, alert: Alert, zone_id: str) -> str:
        """
        Insert a new alert record.
        
        Args:
            alert: Alert instance
            zone_id: Associated zone ID
            
        Returns:
            Alert ID
        """
        query = """
        INSERT INTO alerts (
            id, zone_id, alert_type, severity, message, created_at,
            acknowledged_at, resolved_at, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        metadata_json = json.dumps(alert.metadata)
        
        params = (
            alert.id,
            zone_id,
            alert.alert_type,
            alert.severity,
            alert.message,
            alert.created_at,
            alert.acknowledged_at,
            alert.resolved_at,
            metadata_json
        )
        
        self.db.execute_update(query, params)
        return alert.id
    
    def get_alerts_for_zone(self, zone_id: str, active_only: bool = False) -> List[Alert]:
        """
        Get alerts for a monitoring zone.
        
        Args:
            zone_id: Zone identifier
            active_only: If True, only return unresolved alerts
            
        Returns:
            List of Alert instances
        """
        query = """
        SELECT id, alert_type, severity, message, created_at,
               acknowledged_at, resolved_at, metadata
        FROM alerts WHERE zone_id = ?
        """
        
        if active_only:
            query += " AND resolved_at IS NULL"
        
        query += " ORDER BY created_at DESC"
        
        results = self.db.execute_query(query, (zone_id,))
        
        return [
            Alert(
                id=row['id'],
                alert_type=row['alert_type'],
                severity=row['severity'],
                message=row['message'],
                created_at=row['created_at'],
                acknowledged_at=row['acknowledged_at'],
                resolved_at=row['resolved_at'],
                metadata=json.loads(row['metadata'])
            )
            for row in results
        ]
    
    def update_alert(self, alert: Alert) -> bool:
        """
        Update an alert record.
        
        Args:
            alert: Alert instance
            
        Returns:
            True if updated, False if not found
        """
        query = """
        UPDATE alerts SET
            acknowledged_at = ?, resolved_at = ?, metadata = ?
        WHERE id = ?
        """
        
        metadata_json = json.dumps(alert.metadata)
        
        params = (
            alert.acknowledged_at,
            alert.resolved_at,
            metadata_json,
            alert.id
        )
        
        affected_rows = self.db.execute_update(query, params)
        return affected_rows > 0
    
    # Index Time Series CRUD Operations
    
    def create_index_measurement(self, measurement: IndexTimeSeries) -> None:
        """
        Insert a new index time series measurement.
        
        Args:
            measurement: IndexTimeSeries instance
        """
        query = """
        INSERT INTO index_timeseries (
            zone_id, index_type, timestamp, mean_value, std_deviation,
            pixel_count, quality_score, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        metadata_json = json.dumps(measurement.metadata)
        
        params = (
            measurement.zone_id,
            measurement.index_type,
            measurement.timestamp,
            measurement.mean_value,
            measurement.std_deviation,
            measurement.pixel_count,
            measurement.quality_score,
            metadata_json
        )
        
        self.db.execute_update(query, params)
    
    def get_index_timeseries(self, zone_id: str, index_type: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[IndexTimeSeries]:
        """
        Get index time series data for a zone and index type.
        
        Args:
            zone_id: Zone identifier
            index_type: Type of vegetation index
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of IndexTimeSeries instances
        """
        query = """
        SELECT zone_id, index_type, timestamp, mean_value, std_deviation,
               pixel_count, quality_score, metadata
        FROM index_timeseries 
        WHERE zone_id = ? AND index_type = ?
        """
        params = [zone_id, index_type]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        results = self.db.execute_query(query, tuple(params))
        
        return [
            IndexTimeSeries(
                zone_id=row['zone_id'],
                index_type=row['index_type'],
                timestamp=row['timestamp'],
                mean_value=row['mean_value'],
                std_deviation=row['std_deviation'],
                pixel_count=row['pixel_count'],
                quality_score=row['quality_score'],
                metadata=json.loads(row['metadata'])
            )
            for row in results
        ]
    
    def get_latest_index_measurement(self, zone_id: str, index_type: str) -> Optional[IndexTimeSeries]:
        """
        Get the most recent index measurement for a zone and index type.
        
        Args:
            zone_id: Zone identifier
            index_type: Type of vegetation index
            
        Returns:
            IndexTimeSeries instance or None if not found
        """
        query = """
        SELECT zone_id, index_type, timestamp, mean_value, std_deviation,
               pixel_count, quality_score, metadata
        FROM index_timeseries 
        WHERE zone_id = ? AND index_type = ?
        ORDER BY timestamp DESC LIMIT 1
        """
        
        result = self.db.execute_query(query, (zone_id, index_type))
        if not result:
            return None
        
        row = result[0]
        return IndexTimeSeries(
            zone_id=row['zone_id'],
            index_type=row['index_type'],
            timestamp=row['timestamp'],
            mean_value=row['mean_value'],
            std_deviation=row['std_deviation'],
            pixel_count=row['pixel_count'],
            quality_score=row['quality_score'],
            metadata=json.loads(row['metadata'])
        )