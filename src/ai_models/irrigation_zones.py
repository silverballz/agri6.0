"""
Precision Irrigation Zone Recommender

Analyzes water stress indicators to generate irrigation zone recommendations.
Uses NDWI (water content) and NDSI (soil moisture) to cluster fields into
irrigation management zones with specific recommendations.

Part of the USP features for AgriFlux dashboard.
"""

import numpy as np
import rasterio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class IrrigationZone:
    """Container for irrigation zone information."""
    zone_id: int
    zone_name: str
    water_stress_level: str  # 'low', 'moderate', 'high', 'severe'
    priority: int  # 1 (highest) to 4 (lowest)
    area_percentage: float
    pixel_count: int
    mean_ndwi: float
    mean_ndsi: float
    recommendation: str
    irrigation_frequency: str
    water_amount: str
    color: str  # For visualization


@dataclass
class IrrigationPlan:
    """Complete irrigation plan with zones and recommendations."""
    zones: List[IrrigationZone]
    zone_map: np.ndarray  # Spatial map of zone assignments
    total_area: int
    high_priority_area: float
    water_savings_estimate: float
    summary: str
    
    def get_zone_by_id(self, zone_id: int) -> Optional[IrrigationZone]:
        """Get zone information by ID."""
        for zone in self.zones:
            if zone.zone_id == zone_id:
                return zone
        return None


class IrrigationZoneRecommender:
    """
    Generates precision irrigation zones based on water stress indicators.
    """
    
    # Water stress classification thresholds
    NDWI_THRESHOLDS = {
        'severe_stress': -0.3,
        'high_stress': -0.1,
        'moderate_stress': 0.1,
        'low_stress': 0.3
    }
    
    # Zone colors for visualization
    ZONE_COLORS = {
        0: '#8B0000',  # Dark red - Severe stress
        1: '#FF4500',  # Orange red - High stress
        2: '#FFA500',  # Orange - Moderate stress
        3: '#90EE90'   # Light green - Low stress
    }
    
    def __init__(self, n_zones: int = 4):
        """
        Initialize irrigation zone recommender.
        
        Args:
            n_zones: Number of irrigation zones to create (default: 4)
        """
        self.n_zones = n_zones
        self.scaler = StandardScaler()
    
    def load_index_data(self, geotiff_path: str) -> np.ndarray:
        """
        Load vegetation index data from GeoTIFF.
        
        Args:
            geotiff_path: Path to GeoTIFF file
            
        Returns:
            Data array
        """
        try:
            with rasterio.open(geotiff_path) as src:
                data = src.read(1)
                return data
        except Exception as e:
            logger.error(f"Failed to load {geotiff_path}: {e}")
            raise
    
    def calculate_water_stress_index(self,
                                     ndwi: np.ndarray,
                                     ndsi: np.ndarray) -> np.ndarray:
        """
        Calculate combined water stress index from NDWI and NDSI.
        
        Formula: WSI = (NDWI * 0.7) + (NDSI * 0.3)
        NDWI weighted more heavily as it directly measures water content.
        
        Args:
            ndwi: Normalized Difference Water Index array
            ndsi: Normalized Difference Soil Index array
            
        Returns:
            Water stress index array
        """
        if ndwi.shape != ndsi.shape:
            raise ValueError(f"Shape mismatch: NDWI {ndwi.shape} vs NDSI {ndsi.shape}")
        
        # Calculate weighted combination
        wsi = (ndwi * 0.7) + (ndsi * 0.3)
        
        # Mask invalid values
        valid_mask = np.isfinite(ndwi) & np.isfinite(ndsi)
        wsi[~valid_mask] = np.nan
        
        return wsi
    
    def prepare_features(self,
                        ndwi: np.ndarray,
                        ndsi: np.ndarray,
                        additional_features: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix for clustering.
        
        Args:
            ndwi: NDWI array
            ndsi: NDSI array
            additional_features: Optional dict of additional feature arrays
            
        Returns:
            Tuple of (feature_matrix, valid_mask)
        """
        # Create valid mask
        valid_mask = np.isfinite(ndwi) & np.isfinite(ndsi)
        
        # Extract valid pixels
        features = []
        features.append(ndwi[valid_mask].reshape(-1, 1))
        features.append(ndsi[valid_mask].reshape(-1, 1))
        
        # Add additional features if provided
        if additional_features:
            for name, data in additional_features.items():
                if data.shape == ndwi.shape:
                    valid_data = data[valid_mask].reshape(-1, 1)
                    features.append(valid_data)
        
        # Combine features
        feature_matrix = np.hstack(features)
        
        return feature_matrix, valid_mask
    
    def cluster_zones(self,
                     feature_matrix: np.ndarray,
                     random_state: int = 42) -> np.ndarray:
        """
        Cluster pixels into irrigation zones using K-means.
        
        Args:
            feature_matrix: Feature matrix (n_samples, n_features)
            random_state: Random seed for reproducibility
            
        Returns:
            Cluster labels array
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_zones,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(features_scaled)
        
        logger.info(f"Clustered into {self.n_zones} irrigation zones")
        return labels
    
    def classify_water_stress(self, mean_ndwi: float) -> Tuple[str, int]:
        """
        Classify water stress level based on mean NDWI.
        
        Args:
            mean_ndwi: Mean NDWI value for zone
            
        Returns:
            Tuple of (stress_level, priority)
        """
        if mean_ndwi < self.NDWI_THRESHOLDS['severe_stress']:
            return 'severe', 1
        elif mean_ndwi < self.NDWI_THRESHOLDS['high_stress']:
            return 'high', 2
        elif mean_ndwi < self.NDWI_THRESHOLDS['moderate_stress']:
            return 'moderate', 3
        else:
            return 'low', 4
    
    def generate_recommendation(self,
                               stress_level: str,
                               mean_ndwi: float,
                               mean_ndsi: float) -> Tuple[str, str, str]:
        """
        Generate irrigation recommendation based on stress level.
        
        Args:
            stress_level: Water stress classification
            mean_ndwi: Mean NDWI value
            mean_ndsi: Mean NDSI value
            
        Returns:
            Tuple of (recommendation, frequency, amount)
        """
        recommendations = {
            'severe': {
                'text': 'URGENT: Immediate irrigation required. Crops showing severe water stress.',
                'frequency': 'Daily',
                'amount': '25-30mm per application'
            },
            'high': {
                'text': 'High priority irrigation needed. Schedule within 24-48 hours.',
                'frequency': 'Every 2-3 days',
                'amount': '20-25mm per application'
            },
            'moderate': {
                'text': 'Moderate water stress detected. Plan irrigation within this week.',
                'frequency': 'Every 4-5 days',
                'amount': '15-20mm per application'
            },
            'low': {
                'text': 'Adequate water content. Monitor and irrigate as needed.',
                'frequency': 'Every 7-10 days',
                'amount': '10-15mm per application'
            }
        }
        
        rec = recommendations.get(stress_level, recommendations['moderate'])
        return rec['text'], rec['frequency'], rec['amount']
    
    def create_irrigation_zones(self,
                               ndwi_path: str,
                               ndsi_path: str,
                               additional_features: Optional[Dict[str, str]] = None) -> IrrigationPlan:
        """
        Create complete irrigation plan with zones and recommendations.
        
        Args:
            ndwi_path: Path to NDWI GeoTIFF
            ndsi_path: Path to NDSI GeoTIFF
            additional_features: Optional dict of feature_name -> geotiff_path
            
        Returns:
            IrrigationPlan object
        """
        logger.info("Creating irrigation zones...")
        
        # Load data
        ndwi = self.load_index_data(ndwi_path)
        ndsi = self.load_index_data(ndsi_path)
        
        # Load additional features if provided
        extra_features = {}
        if additional_features:
            for name, path in additional_features.items():
                extra_features[name] = self.load_index_data(path)
        
        # Prepare features
        feature_matrix, valid_mask = self.prepare_features(
            ndwi, ndsi, extra_features
        )
        
        # Cluster into zones
        labels = self.cluster_zones(feature_matrix)
        
        # Create zone map
        zone_map = np.full(ndwi.shape, -1, dtype=np.int8)
        zone_map[valid_mask] = labels
        
        # Calculate zone statistics and create zone objects
        zones = []
        total_valid_pixels = np.sum(valid_mask)
        
        # Sort zones by mean NDWI (lowest first = highest stress)
        zone_stats = []
        for zone_id in range(self.n_zones):
            zone_mask = (zone_map == zone_id)
            zone_ndwi = ndwi[zone_mask]
            zone_ndsi = ndsi[zone_mask]
            
            mean_ndwi = float(np.mean(zone_ndwi))
            zone_stats.append((zone_id, mean_ndwi, zone_ndsi))
        
        # Sort by NDWI (ascending = most stressed first)
        zone_stats.sort(key=lambda x: x[1])
        
        # Reassign zone IDs based on stress level
        zone_id_mapping = {old_id: new_id for new_id, (old_id, _, _) in enumerate(zone_stats)}
        
        # Remap zone_map
        remapped_zone_map = np.full_like(zone_map, -1)
        for old_id, new_id in zone_id_mapping.items():
            remapped_zone_map[zone_map == old_id] = new_id
        
        # Create zone objects with remapped IDs
        for new_id, (old_id, _, zone_ndsi) in enumerate(zone_stats):
            zone_mask = (remapped_zone_map == new_id)
            zone_ndwi_vals = ndwi[zone_mask]
            zone_ndsi_vals = ndsi[zone_mask]
            
            pixel_count = int(np.sum(zone_mask))
            area_percentage = float(pixel_count / total_valid_pixels * 100)
            
            mean_ndwi = float(np.mean(zone_ndwi_vals))
            mean_ndsi = float(np.mean(zone_ndsi_vals))
            
            # Classify stress and generate recommendation
            stress_level, priority = self.classify_water_stress(mean_ndwi)
            recommendation, frequency, amount = self.generate_recommendation(
                stress_level, mean_ndwi, mean_ndsi
            )
            
            zone = IrrigationZone(
                zone_id=new_id,
                zone_name=f"Zone {new_id + 1}",
                water_stress_level=stress_level,
                priority=priority,
                area_percentage=area_percentage,
                pixel_count=pixel_count,
                mean_ndwi=mean_ndwi,
                mean_ndsi=mean_ndsi,
                recommendation=recommendation,
                irrigation_frequency=frequency,
                water_amount=amount,
                color=self.ZONE_COLORS.get(new_id, '#808080')
            )
            zones.append(zone)
        
        # Calculate high priority area (severe + high stress)
        high_priority_area = sum(
            z.area_percentage for z in zones 
            if z.water_stress_level in ['severe', 'high']
        )
        
        # Estimate water savings from precision irrigation
        # Assume 20-30% water savings compared to uniform irrigation
        water_savings_estimate = 25.0
        
        # Generate summary
        summary = self._generate_summary(zones, high_priority_area)
        
        plan = IrrigationPlan(
            zones=zones,
            zone_map=remapped_zone_map,
            total_area=total_valid_pixels,
            high_priority_area=high_priority_area,
            water_savings_estimate=water_savings_estimate,
            summary=summary
        )
        
        logger.info(f"Irrigation plan created with {len(zones)} zones")
        return plan
    
    def _generate_summary(self, zones: List[IrrigationZone], high_priority_area: float) -> str:
        """Generate text summary of irrigation plan."""
        severe_zones = [z for z in zones if z.water_stress_level == 'severe']
        high_zones = [z for z in zones if z.water_stress_level == 'high']
        
        summary_parts = []
        
        if severe_zones:
            severe_area = sum(z.area_percentage for z in severe_zones)
            summary_parts.append(
                f"⚠️ URGENT: {severe_area:.1f}% of field shows severe water stress"
            )
        
        if high_zones:
            high_area = sum(z.area_percentage for z in high_zones)
            summary_parts.append(
                f"🔴 {high_area:.1f}% requires high-priority irrigation"
            )
        
        if high_priority_area > 0:
            summary_parts.append(
                f"📊 Total high-priority area: {high_priority_area:.1f}%"
            )
        else:
            summary_parts.append(
                "✅ Field water status is generally adequate"
            )
        
        summary_parts.append(
            "💧 Precision irrigation can save ~25% water compared to uniform application"
        )
        
        return " | ".join(summary_parts)
    
    def export_zone_map(self,
                       plan: IrrigationPlan,
                       output_path: str,
                       reference_geotiff: str):
        """
        Export irrigation zone map as GeoTIFF.
        
        Args:
            plan: IrrigationPlan object
            output_path: Path for output file
            reference_geotiff: Reference GeoTIFF for georeferencing
        """
        try:
            with rasterio.open(reference_geotiff) as src:
                profile = src.profile.copy()
                profile.update(
                    dtype=rasterio.int8,
                    count=1,
                    compress='lzw'
                )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(plan.zone_map.astype(np.int8), 1)
            
            logger.info(f"Zone map exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export zone map: {e}")
            raise


def create_irrigation_plan_from_db(imagery_id: int,
                                   db_manager,
                                   n_zones: int = 4) -> Optional[IrrigationPlan]:
    """
    Convenience function to create irrigation plan from database imagery.
    
    Args:
        imagery_id: Database ID of imagery record
        db_manager: DatabaseManager instance
        n_zones: Number of zones to create
        
    Returns:
        IrrigationPlan or None if creation fails
    """
    try:
        # Get imagery record
        record = db_manager.get_processed_imagery(imagery_id)
        
        if not record:
            logger.error(f"Imagery record {imagery_id} not found")
            return None
        
        # Get NDWI and NDSI paths
        ndwi_path = record.get('ndwi_path')
        ndsi_path = record.get('ndsi_path')
        
        if not ndwi_path or not ndsi_path:
            logger.error("NDWI or NDSI not available for this imagery")
            return None
        
        # Create irrigation plan
        recommender = IrrigationZoneRecommender(n_zones=n_zones)
        plan = recommender.create_irrigation_zones(ndwi_path, ndsi_path)
        
        return plan
        
    except Exception as e:
        logger.error(f"Failed to create irrigation plan: {e}")
        return None
