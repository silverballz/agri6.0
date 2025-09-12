"""
Sensor data ingestion system for CSV/JSON formats.

This module handles parsing and ingestion of environmental sensor data
from various formats including CSV and JSON files.
"""

import csv
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Represents a single sensor reading with metadata."""
    sensor_id: str
    timestamp: datetime
    sensor_type: str  # 'soil_moisture', 'temperature', 'humidity', 'leaf_wetness'
    value: float
    unit: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    quality_flag: str = 'good'  # 'good', 'suspect', 'bad'
    metadata: Optional[Dict[str, Any]] = None


class SensorDataIngester:
    """Handles ingestion of sensor data from CSV and JSON formats."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json']
        self.sensor_types = {
            'soil_moisture': {'unit': '%', 'range': (0, 100)},
            'temperature': {'unit': '°C', 'range': (-50, 60)},
            'humidity': {'unit': '%', 'range': (0, 100)},
            'leaf_wetness': {'unit': 'minutes', 'range': (0, 1440)},
            'solar_radiation': {'unit': 'W/m²', 'range': (0, 1500)},
            'precipitation': {'unit': 'mm', 'range': (0, 500)}
        }
    
    def ingest_csv(self, file_path: Union[str, Path]) -> List[SensorReading]:
        """
        Ingest sensor data from CSV file.
        
        Expected CSV format:
        sensor_id,timestamp,sensor_type,value,unit,latitude,longitude,quality_flag
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of SensorReading objects
        """
        readings = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                    try:
                        reading = self._parse_csv_row(row)
                        readings.append(reading)
                    except Exception as e:
                        logger.warning(f"Error parsing CSV row {row_num}: {e}")
                        continue
                        
        except Exception as e:
            raise ValueError(f"Error reading CSV file {file_path}: {e}")
        
        logger.info(f"Successfully ingested {len(readings)} readings from {file_path}")
        return readings
    
    def ingest_json(self, file_path: Union[str, Path]) -> List[SensorReading]:
        """
        Ingest sensor data from JSON file.
        
        Expected JSON format:
        {
            "readings": [
                {
                    "sensor_id": "sensor_001",
                    "timestamp": "2024-09-23T10:30:00Z",
                    "sensor_type": "soil_moisture",
                    "value": 45.2,
                    "unit": "%",
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "quality_flag": "good",
                    "metadata": {"depth": "10cm"}
                }
            ]
        }
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of SensorReading objects
        """
        readings = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
                
                if 'readings' not in data:
                    raise ValueError("JSON file must contain 'readings' array")
                
                for idx, reading_data in enumerate(data['readings']):
                    try:
                        reading = self._parse_json_reading(reading_data)
                        readings.append(reading)
                    except Exception as e:
                        logger.warning(f"Error parsing JSON reading {idx}: {e}")
                        continue
                        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file {file_path}: {e}")
        
        logger.info(f"Successfully ingested {len(readings)} readings from {file_path}")
        return readings
    
    def ingest_dataframe(self, df: pd.DataFrame) -> List[SensorReading]:
        """
        Ingest sensor data from pandas DataFrame.
        
        Args:
            df: DataFrame with sensor data columns
            
        Returns:
            List of SensorReading objects
        """
        readings = []
        
        required_columns = ['sensor_id', 'timestamp', 'sensor_type', 'value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
        for idx, row in df.iterrows():
            try:
                reading = self._parse_dataframe_row(row)
                readings.append(reading)
            except Exception as e:
                logger.warning(f"Error parsing DataFrame row {idx}: {e}")
                continue
        
        logger.info(f"Successfully ingested {len(readings)} readings from DataFrame")
        return readings
    
    def _parse_csv_row(self, row: Dict[str, str]) -> SensorReading:
        """Parse a single CSV row into SensorReading."""
        return SensorReading(
            sensor_id=row['sensor_id'].strip(),
            timestamp=self._parse_timestamp(row['timestamp']),
            sensor_type=row['sensor_type'].strip().lower(),
            value=float(row['value']),
            unit=row.get('unit', '').strip(),
            latitude=float(row['latitude']) if row.get('latitude') else None,
            longitude=float(row['longitude']) if row.get('longitude') else None,
            quality_flag=row.get('quality_flag', 'good').strip().lower(),
            metadata=json.loads(row['metadata']) if row.get('metadata') else None
        )
    
    def _parse_json_reading(self, data: Dict[str, Any]) -> SensorReading:
        """Parse a single JSON reading into SensorReading."""
        return SensorReading(
            sensor_id=data['sensor_id'],
            timestamp=self._parse_timestamp(data['timestamp']),
            sensor_type=data['sensor_type'].lower(),
            value=float(data['value']),
            unit=data.get('unit', ''),
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            quality_flag=data.get('quality_flag', 'good').lower(),
            metadata=data.get('metadata')
        )
    
    def _parse_dataframe_row(self, row: pd.Series) -> SensorReading:
        """Parse a single DataFrame row into SensorReading."""
        return SensorReading(
            sensor_id=str(row['sensor_id']),
            timestamp=self._parse_timestamp(row['timestamp']),
            sensor_type=str(row['sensor_type']).lower(),
            value=float(row['value']),
            unit=str(row.get('unit', '')),
            latitude=row.get('latitude'),
            longitude=row.get('longitude'),
            quality_flag=str(row.get('quality_flag', 'good')).lower(),
            metadata=row.get('metadata')
        )
    
    def _parse_timestamp(self, timestamp_str: Union[str, datetime]) -> datetime:
        """Parse timestamp string into datetime object."""
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        
        # Try common timestamp formats
        formats = [
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%d/%m/%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(timestamp_str).strip(), fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")
    
    def get_supported_sensor_types(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported sensor types."""
        return self.sensor_types.copy()
    
    def validate_sensor_type(self, sensor_type: str) -> bool:
        """Check if sensor type is supported."""
        return sensor_type.lower() in self.sensor_types