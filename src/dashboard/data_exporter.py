"""
Data Export Module for Agricultural Monitoring Platform
Handles CSV, GeoTIFF, GeoJSON, and batch export functionality.
"""

import os
import io
import json
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
import logging

# Import data models
try:
    from ..models.satellite_image import SatelliteImage
    from ..models.monitoring_zone import MonitoringZone, Alert
    from ..models.index_timeseries import IndexTimeSeries, TimeSeriesCollection
except ImportError:
    # Fallback for development
    SatelliteImage = None
    MonitoringZone = None
    Alert = None
    IndexTimeSeries = None
    TimeSeriesCollection = None


class DataExporter:
    """Main data export class handling multiple formats and batch operations"""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def export_vegetation_indices_csv(self, time_series_data: List[Dict[str, Any]], 
                                    zones: Optional[List[str]] = None,
                                    indices: Optional[List[str]] = None,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> str:
        """
        Export vegetation index time series to CSV format.
        
        Args:
            time_series_data: List of time series data dictionaries
            zones: Optional list of zone IDs to filter
            indices: Optional list of index types to include
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Path to the exported CSV file
        """
        
        # Filter data
        filtered_data = self._filter_time_series_data(
            time_series_data, zones, indices, start_date, end_date
        )
        
        if not filtered_data:
            raise ValueError("No data available for export after filtering")
        
        # Convert to DataFrame
        df = pd.DataFrame(filtered_data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp and zone
        df = df.sort_values(['zone_id', 'index_type', 'timestamp'])
        
        # Add derived columns
        df['date'] = df['timestamp'].dt.date
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"vegetation_indices_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Exported {len(df)} vegetation index records to {filepath}")
        return filepath
    
    def export_satellite_imagery_geotiff(self, satellite_images: List[Dict[str, Any]], 
                                       index_type: str = 'NDVI',
                                       zones: Optional[List[str]] = None,
                                       resolution: str = '10m') -> str:
        """
        Export processed satellite imagery as GeoTIFF format.
        
        Args:
            satellite_images: List of satellite image data
            index_type: Vegetation index to export (NDVI, SAVI, etc.)
            zones: Optional list of zone IDs to filter
            resolution: Output resolution (10m, 20m, 60m)
            
        Returns:
            Path to the exported GeoTIFF file
        """
        
        if not satellite_images:
            raise ValueError("No satellite images provided for export")
        
        # Use the most recent image for export
        latest_image = max(satellite_images, key=lambda x: x.get('acquisition_date', ''))
        
        # Generate mock raster data (in production, this would use actual satellite data)
        height, width = 1000, 1000  # Mock dimensions
        
        # Create mock index data
        if index_type == 'NDVI':
            # NDVI typically ranges from -1 to 1, with vegetation around 0.2-0.8
            data = np.random.uniform(0.2, 0.8, (height, width))
            # Add some spatial patterns
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            data = data * (1 - distance / np.max(distance) * 0.3)  # Gradient effect
        else:
            # Other indices
            data = np.random.uniform(0.1, 0.9, (height, width))
        
        # Define spatial bounds (mock coordinates)
        bounds = (500000, 4000000, 510000, 4010000)  # UTM coordinates
        transform = from_bounds(*bounds, width, height)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"satellite_{index_type.lower()}_{resolution}_{timestamp}.tif"
        filepath = os.path.join(self.output_dir, filename)
        
        # Write GeoTIFF
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=CRS.from_epsg(32643),  # UTM Zone 43N
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)
            
            # Add metadata
            dst.update_tags(
                index_type=index_type,
                resolution=resolution,
                export_date=datetime.now().isoformat(),
                description=f"Agricultural monitoring {index_type} index"
            )
        
        self.logger.info(f"Exported {index_type} GeoTIFF to {filepath}")
        return filepath
    
    def export_monitoring_zones_geojson(self, zones: List[Dict[str, Any]], 
                                      include_alerts: bool = True,
                                      include_sensors: bool = True) -> str:
        """
        Export monitoring zone boundaries as GeoJSON format.
        
        Args:
            zones: List of monitoring zone data
            include_alerts: Whether to include alert information
            include_sensors: Whether to include sensor locations
            
        Returns:
            Path to the exported GeoJSON file
        """
        
        if not zones:
            raise ValueError("No zones provided for export")
        
        features = []
        
        for zone in zones:
            # Create mock geometry if not provided
            if 'geometry' not in zone or not zone['geometry']:
                # Create a mock polygon around a center point
                center_x, center_y = 500000 + np.random.uniform(-5000, 5000), 4005000 + np.random.uniform(-5000, 5000)
                size = np.random.uniform(500, 2000)
                
                # Create a roughly rectangular polygon with some variation
                coords = [
                    [center_x - size, center_y - size],
                    [center_x + size, center_y - size], 
                    [center_x + size, center_y + size],
                    [center_x - size, center_y + size],
                    [center_x - size, center_y - size]
                ]
                
                geometry = {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            else:
                geometry = zone['geometry']
            
            # Prepare properties
            properties = {
                "id": zone.get('id', ''),
                "name": zone.get('name', ''),
                "crop_type": zone.get('crop_type', ''),
                "area_hectares": zone.get('area', 0),
                "planting_date": zone.get('planting_date', ''),
                "expected_harvest": zone.get('expected_harvest', ''),
                "export_date": datetime.now().isoformat()
            }
            
            # Add alert information if requested
            if include_alerts and 'alerts' in zone:
                active_alerts = [alert for alert in zone['alerts'] if alert.get('status') == 'active']
                properties.update({
                    "active_alerts_count": len(active_alerts),
                    "critical_alerts_count": len([a for a in active_alerts if a.get('severity') == 'critical']),
                    "latest_alert": active_alerts[0].get('message', '') if active_alerts else ''
                })
            
            # Add sensor information if requested
            if include_sensors and 'sensors' in zone:
                active_sensors = [sensor for sensor in zone['sensors'] if sensor.get('is_active', True)]
                properties.update({
                    "sensor_count": len(active_sensors),
                    "sensor_types": list(set(sensor.get('sensor_type', '') for sensor in active_sensors))
                })
            
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": properties
            })
        
        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::32643"  # UTM Zone 43N
                }
            },
            "features": features
        }
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"monitoring_zones_{timestamp}.geojson"
        filepath = os.path.join(self.output_dir, filename)
        
        # Write GeoJSON
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        self.logger.info(f"Exported {len(features)} monitoring zones to {filepath}")
        return filepath
    
    def export_sensor_data_csv(self, sensor_data: List[Dict[str, Any]], 
                             zones: Optional[List[str]] = None,
                             sensor_types: Optional[List[str]] = None,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> str:
        """
        Export sensor data to CSV format.
        
        Args:
            sensor_data: List of sensor measurement data
            zones: Optional list of zone IDs to filter
            sensor_types: Optional list of sensor types to include
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Path to the exported CSV file
        """
        
        # Filter data
        filtered_data = sensor_data.copy()
        
        if zones:
            filtered_data = [d for d in filtered_data if d.get('zone_id') in zones]
        
        if sensor_types:
            filtered_data = [d for d in filtered_data if d.get('sensor_type') in sensor_types]
        
        if start_date:
            filtered_data = [d for d in filtered_data 
                           if pd.to_datetime(d.get('timestamp', '')) >= start_date]
        
        if end_date:
            filtered_data = [d for d in filtered_data 
                           if pd.to_datetime(d.get('timestamp', '')) <= end_date]
        
        if not filtered_data:
            raise ValueError("No sensor data available for export after filtering")
        
        # Convert to DataFrame
        df = pd.DataFrame(filtered_data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp and sensor
        df = df.sort_values(['zone_id', 'sensor_type', 'timestamp'])
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sensor_data_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Exported {len(df)} sensor records to {filepath}")
        return filepath
    
    def export_alerts_csv(self, alerts: List[Dict[str, Any]], 
                         zones: Optional[List[str]] = None,
                         severities: Optional[List[str]] = None,
                         status_filter: Optional[str] = None) -> str:
        """
        Export alert data to CSV format.
        
        Args:
            alerts: List of alert data
            zones: Optional list of zone IDs to filter
            severities: Optional list of severities to include
            status_filter: Optional status filter ('active', 'resolved', etc.)
            
        Returns:
            Path to the exported CSV file
        """
        
        # Filter data
        filtered_alerts = alerts.copy()
        
        if zones:
            filtered_alerts = [a for a in filtered_alerts if a.get('zone_id') in zones]
        
        if severities:
            filtered_alerts = [a for a in filtered_alerts if a.get('severity') in severities]
        
        if status_filter:
            filtered_alerts = [a for a in filtered_alerts if a.get('status') == status_filter]
        
        if not filtered_alerts:
            raise ValueError("No alerts available for export after filtering")
        
        # Convert to DataFrame
        df = pd.DataFrame(filtered_alerts)
        
        # Ensure datetime columns
        datetime_columns = ['created_at', 'acknowledged_at', 'resolved_at']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Sort by creation date
        if 'created_at' in df.columns:
            df = df.sort_values('created_at', ascending=False)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"alerts_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Exported {len(df)} alert records to {filepath}")
        return filepath
    
    def create_batch_export(self, export_configs: List[Dict[str, Any]], 
                          compress: bool = True) -> str:
        """
        Create a batch export with multiple datasets.
        
        Args:
            export_configs: List of export configuration dictionaries
            compress: Whether to compress the output into a ZIP file
            
        Returns:
            Path to the batch export (ZIP file if compressed, directory if not)
        """
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = os.path.join(self.output_dir, f"batch_export_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        
        exported_files = []
        
        for config in export_configs:
            try:
                export_type = config.get('type', '')
                
                if export_type == 'vegetation_indices':
                    filepath = self.export_vegetation_indices_csv(
                        config.get('data', []),
                        zones=config.get('zones'),
                        indices=config.get('indices'),
                        start_date=config.get('start_date'),
                        end_date=config.get('end_date')
                    )
                
                elif export_type == 'satellite_imagery':
                    filepath = self.export_satellite_imagery_geotiff(
                        config.get('data', []),
                        index_type=config.get('index_type', 'NDVI'),
                        zones=config.get('zones'),
                        resolution=config.get('resolution', '10m')
                    )
                
                elif export_type == 'monitoring_zones':
                    filepath = self.export_monitoring_zones_geojson(
                        config.get('data', []),
                        include_alerts=config.get('include_alerts', True),
                        include_sensors=config.get('include_sensors', True)
                    )
                
                elif export_type == 'sensor_data':
                    filepath = self.export_sensor_data_csv(
                        config.get('data', []),
                        zones=config.get('zones'),
                        sensor_types=config.get('sensor_types'),
                        start_date=config.get('start_date'),
                        end_date=config.get('end_date')
                    )
                
                elif export_type == 'alerts':
                    filepath = self.export_alerts_csv(
                        config.get('data', []),
                        zones=config.get('zones'),
                        severities=config.get('severities'),
                        status_filter=config.get('status_filter')
                    )
                
                else:
                    self.logger.warning(f"Unknown export type: {export_type}")
                    continue
                
                # Move file to batch directory
                filename = os.path.basename(filepath)
                new_filepath = os.path.join(batch_dir, filename)
                os.rename(filepath, new_filepath)
                exported_files.append(new_filepath)
                
            except Exception as e:
                self.logger.error(f"Error exporting {config.get('type', 'unknown')}: {e}")
        
        # Create manifest file
        manifest = {
            "export_date": datetime.now().isoformat(),
            "total_files": len(exported_files),
            "files": [
                {
                    "filename": os.path.basename(f),
                    "size_bytes": os.path.getsize(f),
                    "type": self._get_file_type(f)
                }
                for f in exported_files
            ]
        }
        
        manifest_path = os.path.join(batch_dir, "export_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        if compress:
            # Create ZIP file
            zip_filename = f"batch_export_{timestamp}.zip"
            zip_filepath = os.path.join(self.output_dir, zip_filename)
            
            with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(batch_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, batch_dir)
                        zipf.write(file_path, arcname)
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(batch_dir)
            
            self.logger.info(f"Created batch export ZIP: {zip_filepath}")
            return zip_filepath
        else:
            self.logger.info(f"Created batch export directory: {batch_dir}")
            return batch_dir
    
    def _filter_time_series_data(self, data: List[Dict[str, Any]], 
                                zones: Optional[List[str]] = None,
                                indices: Optional[List[str]] = None,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Filter time series data based on criteria"""
        
        filtered_data = data.copy()
        
        if zones:
            filtered_data = [d for d in filtered_data if d.get('zone_id') in zones]
        
        if indices:
            filtered_data = [d for d in filtered_data if d.get('index_type') in indices]
        
        if start_date:
            filtered_data = [d for d in filtered_data 
                           if pd.to_datetime(d.get('timestamp', '')) >= start_date]
        
        if end_date:
            filtered_data = [d for d in filtered_data 
                           if pd.to_datetime(d.get('timestamp', '')) <= end_date]
        
        return filtered_data
    
    def _get_file_type(self, filepath: str) -> str:
        """Get file type from extension"""
        ext = os.path.splitext(filepath)[1].lower()
        type_map = {
            '.csv': 'CSV',
            '.tif': 'GeoTIFF',
            '.geojson': 'GeoJSON',
            '.json': 'JSON',
            '.xlsx': 'Excel'
        }
        return type_map.get(ext, 'Unknown')
    
    def get_export_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get history of recent exports"""
        
        history = []
        
        if not os.path.exists(self.output_dir):
            return history
        
        # Get all files in export directory
        for filename in os.listdir(self.output_dir):
            filepath = os.path.join(self.output_dir, filename)
            
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                
                history.append({
                    "filename": filename,
                    "filepath": filepath,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "type": self._get_file_type(filepath)
                })
        
        # Sort by creation date (newest first)
        history.sort(key=lambda x: x['created'], reverse=True)
        
        return history[:limit]
    
    def cleanup_old_exports(self, days_old: int = 30) -> int:
        """Clean up export files older than specified days"""
        
        if not os.path.exists(self.output_dir):
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for filename in os.listdir(self.output_dir):
            filepath = os.path.join(self.output_dir, filename)
            
            if os.path.isfile(filepath):
                file_date = datetime.fromtimestamp(os.path.getctime(filepath))
                
                if file_date < cutoff_date:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        self.logger.info(f"Deleted old export file: {filename}")
                    except Exception as e:
                        self.logger.error(f"Error deleting file {filename}: {e}")
        
        return deleted_count


# Utility functions for generating mock data
def generate_mock_vegetation_indices_data(zones: List[str], days: int = 30) -> List[Dict[str, Any]]:
    """Generate mock vegetation indices data for testing"""
    
    data = []
    indices = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI']
    
    for zone_id in zones:
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            for index_type in indices:
                # Generate realistic values for each index
                base_values = {'NDVI': 0.7, 'SAVI': 0.65, 'EVI': 0.6, 'NDWI': 0.3, 'NDSI': 0.4}
                base_value = base_values[index_type]
                
                value = base_value + np.random.normal(0, 0.05)
                value = max(-1, min(1, value))  # Clamp to valid range
                
                data.append({
                    'zone_id': zone_id,
                    'index_type': index_type,
                    'timestamp': date,
                    'mean_value': round(value, 4),
                    'std_deviation': round(np.random.uniform(0.01, 0.1), 4),
                    'pixel_count': np.random.randint(500, 2000),
                    'quality_score': round(np.random.uniform(0.7, 1.0), 3)
                })
    
    return data


def generate_mock_sensor_data(zones: List[str], hours: int = 168) -> List[Dict[str, Any]]:
    """Generate mock sensor data for testing (default 1 week)"""
    
    data = []
    sensor_types = ['soil_moisture', 'air_temperature', 'humidity', 'leaf_wetness']
    
    for zone_id in zones:
        for i in range(hours):
            timestamp = datetime.now() - timedelta(hours=i)
            
            for sensor_type in sensor_types:
                if sensor_type == 'soil_moisture':
                    value = np.random.uniform(30, 80)  # Percentage
                elif sensor_type == 'air_temperature':
                    value = np.random.uniform(15, 35)  # Celsius
                elif sensor_type == 'humidity':
                    value = np.random.uniform(40, 90)  # Percentage
                elif sensor_type == 'leaf_wetness':
                    value = np.random.uniform(0, 100)  # Percentage
                else:
                    value = np.random.uniform(0, 100)
                
                data.append({
                    'zone_id': zone_id,
                    'sensor_type': sensor_type,
                    'timestamp': timestamp,
                    'value': round(value, 2),
                    'unit': {'soil_moisture': '%', 'air_temperature': '°C', 
                            'humidity': '%', 'leaf_wetness': '%'}[sensor_type],
                    'sensor_id': f"{zone_id}_{sensor_type}_01"
                })
    
    return data


def generate_mock_monitoring_zones(count: int = 5) -> List[Dict[str, Any]]:
    """Generate mock monitoring zone data"""
    
    zones = []
    crop_types = ['Corn', 'Soybeans', 'Wheat', 'Barley', 'Apple Trees']
    
    for i in range(count):
        zone_id = f"zone_{i+1}"
        
        zones.append({
            'id': zone_id,
            'name': f"Field {chr(65+i)}",  # Field A, Field B, etc.
            'crop_type': crop_types[i % len(crop_types)],
            'area': round(np.random.uniform(50, 300), 1),
            'planting_date': (datetime.now() - timedelta(days=np.random.randint(60, 200))).strftime('%Y-%m-%d'),
            'expected_harvest': (datetime.now() + timedelta(days=np.random.randint(30, 120))).strftime('%Y-%m-%d'),
            'alerts': [
                {
                    'id': f"alert_{zone_id}_{j}",
                    'severity': np.random.choice(['low', 'medium', 'high', 'critical']),
                    'message': f"Mock alert {j+1} for {zone_id}",
                    'status': np.random.choice(['active', 'resolved'])
                }
                for j in range(np.random.randint(0, 3))
            ],
            'sensors': [
                {
                    'id': f"sensor_{zone_id}_{k}",
                    'sensor_type': sensor_type,
                    'is_active': np.random.choice([True, False], p=[0.9, 0.1])
                }
                for k, sensor_type in enumerate(['soil_moisture', 'weather_station'])
            ]
        })
    
    return zones