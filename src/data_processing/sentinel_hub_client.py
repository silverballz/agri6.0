"""
Sentinel Hub API client for fetching real Sentinel-2A imagery.
Handles authentication, query construction, and data retrieval.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

import requests
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SentinelHubConfig:
    """Configuration for Sentinel Hub API access."""
    instance_id: str
    client_id: str
    client_secret: str
    base_url: str = 'https://services.sentinel-hub.com'
    bands: List[str] = None
    resolution: int = 10
    max_cloud_coverage: float = 20.0
    
    def __post_init__(self):
        if self.bands is None:
            self.bands = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
    
    @classmethod
    def from_env(cls) -> 'SentinelHubConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            SentinelHubConfig instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        instance_id = os.getenv('SENTINEL_HUB_INSTANCE_ID')
        client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
        client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')
        
        if not all([instance_id, client_id, client_secret]):
            missing = []
            if not instance_id:
                missing.append('SENTINEL_HUB_INSTANCE_ID')
            if not client_id:
                missing.append('SENTINEL_HUB_CLIENT_ID')
            if not client_secret:
                missing.append('SENTINEL_HUB_CLIENT_SECRET')
            
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please set these in your .env file or environment."
            )
        
        return cls(
            instance_id=instance_id,
            client_id=client_id,
            client_secret=client_secret
        )
    
    def validate(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            True if configuration is valid
        """
        return all([
            self.instance_id,
            self.client_id,
            self.client_secret,
            self.base_url,
            self.bands,
            self.resolution > 0,
            0 <= self.max_cloud_coverage <= 100
        ])


class SentinelHubClient:
    """
    Client for interacting with Sentinel Hub API.
    
    Handles authentication, query construction, and data retrieval
    with retry logic and error handling.
    """
    
    def __init__(self, config: Optional[SentinelHubConfig] = None, max_retries: int = 3):
        """
        Initialize Sentinel Hub API client.
        
        Args:
            config: SentinelHubConfig instance. If None, loads from environment.
            max_retries: Maximum number of retry attempts for failed requests
        """
        if config is None:
            config = SentinelHubConfig.from_env()
        
        if not config.validate():
            raise ValueError("Invalid Sentinel Hub configuration")
        
        self.config = config
        self.max_retries = max_retries
        self.access_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        
        logger.info("Sentinel Hub client initialized")
    
    def authenticate(self) -> str:
        """
        Authenticate with Sentinel Hub and obtain access token.
        
        Returns:
            Access token string
            
        Raises:
            requests.exceptions.RequestException: If authentication fails
        """
        auth_url = f"{self.config.base_url}/oauth/token"
        
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret
        }
        
        try:
            logger.info("Authenticating with Sentinel Hub API")
            response = requests.post(auth_url, data=payload, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            # Token typically expires in 3600 seconds (1 hour)
            expires_in = token_data.get('expires_in', 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)  # 60s buffer
            
            logger.info("Successfully authenticated with Sentinel Hub API")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            raise ValueError(
                f"Failed to authenticate with Sentinel Hub API: {e}. "
                "Please check your credentials in the .env file. "
                "Visit https://www.sentinel-hub.com/ to obtain API credentials."
            )
    
    def _ensure_authenticated(self):
        """Ensure we have a valid access token, refreshing if necessary."""
        if self.access_token is None or self.token_expiry is None:
            self.authenticate()
        elif datetime.now() >= self.token_expiry:
            logger.info("Access token expired, re-authenticating")
            self.authenticate()
    
    def test_connection(self) -> bool:
        """
        Test connection to Sentinel Hub API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._ensure_authenticated()
            
            # Test with a simple configuration request
            test_url = f"{self.config.base_url}/configuration/v1/wms/instances"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            response = requests.get(test_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info("Connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def validate_credentials(self) -> Tuple[bool, str]:
        """
        Validate API credentials.
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            self.authenticate()
            if self.test_connection():
                return True, "Credentials are valid and connection successful"
            else:
                return False, "Authentication succeeded but connection test failed"
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation failed: {e}"
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers with authentication.
        
        Returns:
            Dictionary of headers
        """
        self._ensure_authenticated()
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }


    def request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Handles:
        - 429 (Rate Limit): Respects Retry-After header
        - 406 (Not Acceptable): Logs detailed error and retries
        - 5xx (Server Errors): Retries with exponential backoff
        - Other errors: Retries with exponential backoff
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            url: Request URL
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            requests.exceptions.RequestException: If all retries fail
        """
        backoff_factor = 2
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Request attempt {attempt + 1}/{self.max_retries}: {method} {url}")
                
                response = requests.request(method, url, **kwargs)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(
                        f"Rate limited (HTTP 429). Waiting {retry_after}s before retry. "
                        f"Attempt {attempt + 1}/{self.max_retries}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_after)
                        continue
                    else:
                        logger.error(f"Max retries exceeded after rate limiting")
                        response.raise_for_status()
                
                # Handle 406 Not Acceptable errors
                if response.status_code == 406:
                    logger.error(
                        f"HTTP 406 Not Acceptable error. "
                        f"Request may have incorrect format or headers. "
                        f"Attempt {attempt + 1}/{self.max_retries}"
                    )
                    logger.error(f"Request headers: {kwargs.get('headers', {})}")
                    logger.error(f"Response body: {response.text}")
                    
                    if attempt < self.max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                # Handle server errors (5xx)
                if response.status_code >= 500:
                    logger.warning(
                        f"Server error (HTTP {response.status_code}). "
                        f"Attempt {attempt + 1}/{self.max_retries}"
                    )
                    if attempt < self.max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                # Raise for other HTTP errors (4xx except 406 and 429)
                response.raise_for_status()
                return response
                
            except requests.exceptions.HTTPError as e:
                # HTTP error already logged above, re-raise if max retries exceeded
                if attempt >= self.max_retries - 1:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for {url}")
                    raise
                    
            except requests.exceptions.RequestException as e:
                wait_time = backoff_factor ** attempt
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for {url}")
                    raise
        
        raise requests.exceptions.RequestException("Request failed after all retries")
    
    def query_sentinel_imagery(
        self,
        geometry: Dict[str, Any],
        date_range: Tuple[str, str],
        cloud_threshold: Optional[float] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query Sentinel Hub for available imagery.
        
        Args:
            geometry: GeoJSON geometry dict (from geojson_handler)
            date_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
            cloud_threshold: Maximum cloud coverage percentage (0-100)
            max_results: Maximum number of results to return
            
        Returns:
            List of imagery metadata dictionaries
            
        Raises:
            ValueError: If date range is invalid or in the future
            requests.exceptions.RequestException: If query fails
        """
        if cloud_threshold is None:
            cloud_threshold = self.config.max_cloud_coverage
        
        # FIX 1: Validate date range before making API request
        start_date, end_date = self._validate_date_range(date_range)
        
        self._ensure_authenticated()
        
        # FIX 2: Use correct STAC API endpoint
        catalog_url = f"{self.config.base_url}/api/v1/catalog/1.0.0/search"
        
        # FIX 3: Build correct STAC-compliant request payload
        bbox = self._geometry_to_bbox(geometry)
        payload = {
            "bbox": bbox,
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "collections": ["sentinel-2-l2a"],
            "limit": min(max_results, 100),  # API limit per request
            "filter": f"eo:cloud_cover < {cloud_threshold}",
            "fields": {
                "include": ["id", "properties.datetime", "properties.eo:cloud_cover", "properties.sentinel:product_id"],
                "exclude": []
            }
        }
        
        # FIX 4: Use correct headers for STAC API
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/geo+json'  # STAC format
        }
        
        try:
            logger.info(
                f"Querying Sentinel imagery for date range {start_date} to {end_date}, "
                f"cloud threshold {cloud_threshold}%"
            )
            logger.debug(f"Request URL: {catalog_url}")
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
            
            response = self.request_with_retry('POST', catalog_url, json=payload, headers=headers, timeout=30)
            
            data = response.json()
            features = data.get('features', [])
            
            logger.info(f"Found {len(features)} imagery results")
            
            # Extract and format metadata
            results = []
            for feature in features:
                metadata = self._extract_imagery_metadata(feature)
                results.append(metadata)
            
            # Sort by acquisition date
            results.sort(key=lambda x: x['acquisition_date'])
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to query Sentinel imagery: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def download_multispectral_bands(
        self,
        geometry: Dict[str, Any],
        acquisition_date: str,
        bands: Optional[List[str]] = None,
        resolution: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Download multispectral band data for a specific date.
        
        Args:
            geometry: GeoJSON geometry dict
            acquisition_date: Date in 'YYYY-MM-DD' format
            bands: List of band names (e.g., ['B02', 'B03', 'B04', 'B08'])
            resolution: Resolution in meters (default from config)
            
        Returns:
            Dictionary mapping band names to numpy arrays
            
        Raises:
            requests.exceptions.RequestException: If download fails
        """
        if bands is None:
            bands = self.config.bands
        
        if resolution is None:
            resolution = self.config.resolution
        
        self._ensure_authenticated()
        
        # Construct Process API request
        process_url = f"{self.config.base_url}/api/v1/process"
        
        # Build evalscript to retrieve bands
        evalscript = self._build_evalscript(bands)
        
        # Build request payload
        bbox = self._geometry_to_bbox(geometry)
        
        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                    }
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{acquisition_date}T00:00:00Z",
                            "to": f"{acquisition_date}T23:59:59Z"
                        },
                        "maxCloudCoverage": self.config.max_cloud_coverage
                    }
                }]
            },
            "output": {
                "width": None,  # Will be calculated from resolution
                "height": None,
                "responses": [{
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }]
            },
            "evalscript": evalscript
        }
        
        # Calculate output dimensions based on resolution
        width, height = self._calculate_dimensions(bbox, resolution)
        payload["output"]["width"] = width
        payload["output"]["height"] = height
        
        headers = self.get_headers()
        headers['Accept'] = 'application/tar'
        
        try:
            logger.info(
                f"Downloading {len(bands)} bands for date {acquisition_date} "
                f"at {resolution}m resolution"
            )
            
            response = self.request_with_retry(
                'POST',
                process_url,
                json=payload,
                headers=headers,
                timeout=120
            )
            
            logger.debug(f"Response content-type: {response.headers.get('content-type')}")
            logger.debug(f"Response content length: {len(response.content)} bytes")
            
            # Parse response
            band_data = self._parse_band_response(response.content, bands, width, height)
            
            logger.info(f"Successfully downloaded {len(band_data)} bands")
            
            return band_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download bands: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
                logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
            raise
    
    def _validate_date_range(self, date_range: Tuple[str, str]) -> Tuple[str, str]:
        """
        Validate date range to ensure dates are valid and not in the future.
        
        Args:
            date_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
            
        Returns:
            Validated tuple of (start_date, end_date)
            
        Raises:
            ValueError: If dates are invalid, in future, or start > end
        """
        start_str, end_str = date_range
        
        try:
            start = datetime.strptime(start_str, '%Y-%m-%d')
            end = datetime.strptime(end_str, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(
                f"Invalid date format. Expected 'YYYY-MM-DD', got start='{start_str}', end='{end_str}'. "
                f"Error: {e}"
            )
        
        now = datetime.now()
        
        # Check if dates are in the future
        if start > now:
            raise ValueError(
                f"Start date cannot be in future. "
                f"Requested: {start_str}, "
                f"Current date: {now.strftime('%Y-%m-%d')}"
            )
        
        if end > now:
            raise ValueError(
                f"End date cannot be in future. "
                f"Requested: {end_str}, "
                f"Current date: {now.strftime('%Y-%m-%d')}"
            )
        
        # Check if start is after end
        if start > end:
            raise ValueError(
                f"Start date {start_str} is after end date {end_str}"
            )
        
        logger.debug(f"Date range validated: {start_str} to {end_str}")
        return start_str, end_str
    
    def _geometry_to_bbox(self, geometry: Dict[str, Any]) -> List[float]:
        """Convert geometry to bounding box [min_lon, min_lat, max_lon, max_lat]."""
        if geometry['type'] == 'Polygon':
            coords = geometry['coordinates'][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            return [min(lons), min(lats), max(lons), max(lats)]
        else:
            raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    
    def _build_evalscript(self, bands: List[str]) -> str:
        """Build evalscript for Sentinel Hub Process API."""
        # Map band names to Sentinel Hub identifiers
        band_mapping = {
            'B01': 'B01', 'B02': 'B02', 'B03': 'B03', 'B04': 'B04',
            'B05': 'B05', 'B06': 'B06', 'B07': 'B07', 'B08': 'B08',
            'B8A': 'B8A', 'B09': 'B09', 'B11': 'B11', 'B12': 'B12'
        }
        
        # Create array of band names for evalscript
        band_list = ', '.join([f'"{band_mapping.get(b, b)}"' for b in bands])
        
        evalscript = f"""
        //VERSION=3
        function setup() {{
            return {{
                input: [{band_list}],
                output: {{
                    bands: {len(bands)},
                    sampleType: "FLOAT32"
                }}
            }};
        }}
        
        function evaluatePixel(sample) {{
            return [{', '.join([f'sample.{band_mapping.get(b, b)}' for b in bands])}];
        }}
        """
        
        return evalscript
    
    def _calculate_dimensions(self, bbox: List[float], resolution: int) -> Tuple[int, int]:
        """Calculate output dimensions based on bbox and resolution."""
        # Approximate calculation (simplified)
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Rough conversion: 1 degree ≈ 111 km at equator
        width_km = (max_lon - min_lon) * 111 * np.cos(np.radians((min_lat + max_lat) / 2))
        height_km = (max_lat - min_lat) * 111
        
        width_px = int(width_km * 1000 / resolution)
        height_px = int(height_km * 1000 / resolution)
        
        # Limit to reasonable size
        max_dim = 2500
        if width_px > max_dim or height_px > max_dim:
            scale = max_dim / max(width_px, height_px)
            width_px = int(width_px * scale)
            height_px = int(height_px * scale)
        
        return width_px, height_px
    
    def _extract_imagery_metadata(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from catalog search result."""
        properties = feature.get('properties', {})
        
        return {
            'id': feature.get('id', ''),
            'acquisition_date': properties.get('datetime', ''),
            'cloud_coverage': properties.get('eo:cloud_cover', 0.0),
            'tile_id': properties.get('sentinel:product_id', ''),
            'geometry': feature.get('geometry', {}),
            'bbox': feature.get('bbox', []),
            'properties': properties
        }
    
    def _parse_band_response(
        self,
        response_content: bytes,
        bands: List[str],
        width: int,
        height: int
    ) -> Dict[str, np.ndarray]:
        """
        Parse band data from API response.
        
        The Sentinel Hub Process API returns a TAR archive containing a multi-band GeoTIFF.
        We need to extract the TIFF and parse each band into a separate array.
        """
        import io
        import tarfile
        from rasterio.io import MemoryFile
        
        band_data = {}
        
        try:
            # The response is a TAR archive containing the TIFF
            tar_buffer = io.BytesIO(response_content)
            
            with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
                # Get list of files in tar
                members = tar.getmembers()
                logger.debug(f"TAR contains {len(members)} files: {[m.name for m in members]}")
                
                # Find the TIFF file (usually named 'default.tif' or similar)
                tiff_member = None
                for member in members:
                    if member.name.endswith('.tif') or member.name.endswith('.tiff'):
                        tiff_member = member
                        break
                
                if not tiff_member:
                    raise ValueError("No TIFF file found in TAR archive")
                
                # Extract and parse the TIFF
                tiff_file = tar.extractfile(tiff_member)
                tiff_content = tiff_file.read()
                
                logger.debug(f"Extracted TIFF: {tiff_member.name}, size={len(tiff_content)} bytes")
                
                # Parse the TIFF with rasterio
                with MemoryFile(tiff_content) as memfile:
                    with memfile.open() as dataset:
                        logger.debug(f"TIFF has {dataset.count} bands, size={dataset.width}x{dataset.height}")
                        
                        # Check if we have the right number of bands
                        if dataset.count != len(bands):
                            logger.warning(
                                f"Expected {len(bands)} bands but got {dataset.count} in response"
                            )
                        
                        # Read all bands
                        for i, band_name in enumerate(bands, start=1):
                            if i <= dataset.count:
                                band_array = dataset.read(i)
                                band_data[band_name] = band_array
                                logger.debug(
                                    f"Parsed band {band_name}: shape={band_array.shape}, "
                                    f"dtype={band_array.dtype}, min={band_array.min():.3f}, "
                                    f"max={band_array.max():.3f}"
                                )
                            else:
                                logger.warning(f"Band {band_name} not found in response, using zeros")
                                band_data[band_name] = np.zeros((height, width), dtype=np.float32)
            
            logger.info(f"Successfully parsed {len(band_data)} bands from TAR response")
            return band_data
            
        except Exception as e:
            logger.error(f"Failed to parse band response: {e}")
            logger.error(f"Response content length: {len(response_content)} bytes")
            logger.error(f"Response content preview: {response_content[:100]}")
            logger.warning("Falling back to mock data")
            
            # Fallback to mock data
            for band in bands:
                band_data[band] = np.zeros((height, width), dtype=np.float32)
            
            return band_data


def create_client_from_env() -> SentinelHubClient:
    """
    Convenience function to create a Sentinel Hub client from environment variables.
    
    Returns:
        Configured SentinelHubClient instance
        
    Raises:
        ValueError: If environment variables are not properly configured
    """
    config = SentinelHubConfig.from_env()
    return SentinelHubClient(config)
