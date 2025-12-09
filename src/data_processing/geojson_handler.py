"""
GeoJSON boundary handling for agricultural field boundaries.
Validates and processes GeoJSON for Sentinel Hub API queries.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Ludhiana region coordinate bounds
LUDHIANA_BOUNDS = {
    'lat_min': 30.9,
    'lat_max': 31.0,
    'lon_min': 75.8,
    'lon_max': 75.9
}


@dataclass
class BoundingBox:
    """Bounding box for a geographic region."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    
    def to_bbox_list(self) -> List[float]:
        """Convert to [min_lon, min_lat, max_lon, max_lat] format."""
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
    
    def to_polygon_coords(self) -> List[List[float]]:
        """Convert to polygon coordinate list."""
        return [
            [self.min_lon, self.min_lat],
            [self.max_lon, self.min_lat],
            [self.max_lon, self.max_lat],
            [self.min_lon, self.max_lat],
            [self.min_lon, self.min_lat]  # Close the polygon
        ]
    
    def area(self) -> float:
        """Calculate approximate area in square degrees."""
        return (self.max_lon - self.min_lon) * (self.max_lat - self.min_lat)
    
    def center(self) -> Tuple[float, float]:
        """Get center point as (lon, lat)."""
        return (
            (self.min_lon + self.max_lon) / 2,
            (self.min_lat + self.max_lat) / 2
        )


class GeoJSONHandler:
    """Handler for GeoJSON field boundaries."""
    
    def __init__(self, validate_ludhiana: bool = True):
        """
        Initialize GeoJSON handler.
        
        Args:
            validate_ludhiana: If True, validate coordinates are within Ludhiana region
        """
        self.validate_ludhiana = validate_ludhiana
    
    def parse_geojson(self, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate GeoJSON data.
        
        Args:
            geojson_data: GeoJSON dictionary
            
        Returns:
            Validated GeoJSON dictionary
            
        Raises:
            ValueError: If GeoJSON is invalid
        """
        # Check for required fields
        if 'type' not in geojson_data:
            raise ValueError("GeoJSON must have a 'type' field")
        
        geojson_type = geojson_data['type']
        
        # Handle different GeoJSON types
        if geojson_type == 'FeatureCollection':
            if 'features' not in geojson_data:
                raise ValueError("FeatureCollection must have 'features' field")
            
            if not geojson_data['features']:
                raise ValueError("FeatureCollection has no features")
            
            # Validate each feature
            for feature in geojson_data['features']:
                self._validate_feature(feature)
        
        elif geojson_type == 'Feature':
            self._validate_feature(geojson_data)
        
        elif geojson_type in ['Polygon', 'MultiPolygon', 'Point', 'LineString']:
            # Direct geometry object
            self._validate_geometry(geojson_data)
        
        else:
            raise ValueError(f"Unsupported GeoJSON type: {geojson_type}")
        
        logger.info(f"Successfully parsed GeoJSON of type: {geojson_type}")
        return geojson_data
    
    def _validate_feature(self, feature: Dict[str, Any]):
        """Validate a GeoJSON feature."""
        if 'type' not in feature or feature['type'] != 'Feature':
            raise ValueError("Feature must have type='Feature'")
        
        if 'geometry' not in feature:
            raise ValueError("Feature must have 'geometry' field")
        
        self._validate_geometry(feature['geometry'])
    
    def _validate_geometry(self, geometry: Dict[str, Any]):
        """Validate a GeoJSON geometry."""
        if 'type' not in geometry:
            raise ValueError("Geometry must have 'type' field")
        
        if 'coordinates' not in geometry:
            raise ValueError("Geometry must have 'coordinates' field")
        
        geom_type = geometry['type']
        coords = geometry['coordinates']
        
        if geom_type == 'Polygon':
            self._validate_polygon_coords(coords)
        elif geom_type == 'MultiPolygon':
            for polygon_coords in coords:
                self._validate_polygon_coords(polygon_coords)
        elif geom_type == 'Point':
            self._validate_point_coords(coords)
        elif geom_type == 'LineString':
            self._validate_linestring_coords(coords)
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")
    
    def _validate_polygon_coords(self, coords: List):
        """Validate polygon coordinates."""
        if not coords or not isinstance(coords, list):
            raise ValueError("Polygon coordinates must be a non-empty list")
        
        # Polygon is a list of linear rings (first is exterior, rest are holes)
        for ring in coords:
            if not isinstance(ring, list) or len(ring) < 4:
                raise ValueError("Polygon ring must have at least 4 coordinate pairs")
            
            # Check each coordinate pair
            for coord_pair in ring:
                self._validate_coordinate_pair(coord_pair)
            
            # First and last coordinates should be the same (closed ring)
            if ring[0] != ring[-1]:
                logger.warning("Polygon ring is not closed (first != last coordinate)")
    
    def _validate_linestring_coords(self, coords: List):
        """Validate linestring coordinates."""
        if not coords or not isinstance(coords, list) or len(coords) < 2:
            raise ValueError("LineString must have at least 2 coordinate pairs")
        
        for coord_pair in coords:
            self._validate_coordinate_pair(coord_pair)
    
    def _validate_point_coords(self, coords: List):
        """Validate point coordinates."""
        self._validate_coordinate_pair(coords)
    
    def _validate_coordinate_pair(self, coord_pair: List):
        """
        Validate a single coordinate pair [lon, lat].
        
        Args:
            coord_pair: [longitude, latitude] pair
            
        Raises:
            ValueError: If coordinates are invalid
        """
        if not isinstance(coord_pair, (list, tuple)) or len(coord_pair) < 2:
            raise ValueError(f"Coordinate pair must have at least 2 values: {coord_pair}")
        
        lon, lat = coord_pair[0], coord_pair[1]
        
        # Validate longitude range
        if not isinstance(lon, (int, float)) or not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180")
        
        # Validate latitude range
        if not isinstance(lat, (int, float)) or not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90")
        
        # Validate Ludhiana region if enabled
        if self.validate_ludhiana:
            if not (LUDHIANA_BOUNDS['lon_min'] <= lon <= LUDHIANA_BOUNDS['lon_max']):
                raise ValueError(
                    f"Longitude {lon} is outside Ludhiana region "
                    f"({LUDHIANA_BOUNDS['lon_min']}-{LUDHIANA_BOUNDS['lon_max']})"
                )
            
            if not (LUDHIANA_BOUNDS['lat_min'] <= lat <= LUDHIANA_BOUNDS['lat_max']):
                raise ValueError(
                    f"Latitude {lat} is outside Ludhiana region "
                    f"({LUDHIANA_BOUNDS['lat_min']}-{LUDHIANA_BOUNDS['lat_max']})"
                )
    
    def load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse GeoJSON from file.
        
        Args:
            file_path: Path to GeoJSON file
            
        Returns:
            Parsed GeoJSON dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If GeoJSON is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                geojson_data = json.load(f)
            
            return self.parse_geojson(geojson_data)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
    
    def extract_bounding_box(self, geojson_data: Dict[str, Any]) -> BoundingBox:
        """
        Extract bounding box from GeoJSON.
        
        Args:
            geojson_data: GeoJSON dictionary
            
        Returns:
            BoundingBox object
        """
        coords = self._extract_all_coordinates(geojson_data)
        
        if not coords:
            raise ValueError("No coordinates found in GeoJSON")
        
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        return BoundingBox(
            min_lon=min(lons),
            min_lat=min(lats),
            max_lon=max(lons),
            max_lat=max(lats)
        )
    
    def _extract_all_coordinates(self, geojson_data: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract all coordinate pairs from GeoJSON."""
        coords = []
        
        geojson_type = geojson_data.get('type')
        
        if geojson_type == 'FeatureCollection':
            for feature in geojson_data.get('features', []):
                coords.extend(self._extract_geometry_coordinates(feature.get('geometry', {})))
        
        elif geojson_type == 'Feature':
            coords.extend(self._extract_geometry_coordinates(geojson_data.get('geometry', {})))
        
        elif geojson_type in ['Polygon', 'MultiPolygon', 'Point', 'LineString']:
            coords.extend(self._extract_geometry_coordinates(geojson_data))
        
        return coords
    
    def _extract_geometry_coordinates(self, geometry: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract coordinates from a geometry object."""
        coords = []
        geom_type = geometry.get('type')
        geom_coords = geometry.get('coordinates', [])
        
        if geom_type == 'Point':
            coords.append(tuple(geom_coords[:2]))
        
        elif geom_type == 'LineString':
            coords.extend([tuple(c[:2]) for c in geom_coords])
        
        elif geom_type == 'Polygon':
            for ring in geom_coords:
                coords.extend([tuple(c[:2]) for c in ring])
        
        elif geom_type == 'MultiPolygon':
            for polygon in geom_coords:
                for ring in polygon:
                    coords.extend([tuple(c[:2]) for c in ring])
        
        return coords
    
    def to_sentinel_hub_geometry(self, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert GeoJSON to Sentinel Hub API geometry format.
        
        Args:
            geojson_data: GeoJSON dictionary
            
        Returns:
            Geometry in Sentinel Hub format
        """
        # For Sentinel Hub, we typically use the bounding box
        bbox = self.extract_bounding_box(geojson_data)
        
        return {
            "type": "Polygon",
            "coordinates": [bbox.to_polygon_coords()]
        }


def create_ludhiana_sample_geojson() -> Dict[str, Any]:
    """
    Create a sample GeoJSON for Ludhiana region.
    
    Returns:
        Sample GeoJSON dictionary
    """
    return {
        "type": "Polygon",
        "coordinates": [[
            [75.80, 30.90],
            [75.90, 30.90],
            [75.90, 31.00],
            [75.80, 31.00],
            [75.80, 30.90]
        ]]
    }


def validate_ludhiana_coordinates(lon: float, lat: float) -> bool:
    """
    Check if coordinates are within Ludhiana region.
    
    Args:
        lon: Longitude
        lat: Latitude
        
    Returns:
        True if within Ludhiana bounds
    """
    return (
        LUDHIANA_BOUNDS['lon_min'] <= lon <= LUDHIANA_BOUNDS['lon_max'] and
        LUDHIANA_BOUNDS['lat_min'] <= lat <= LUDHIANA_BOUNDS['lat_max']
    )
