"""
Carbon Sequestration Calculator

Estimates carbon sequestration and biomass from vegetation indices.
Calculates potential carbon credits and environmental impact metrics.

Part of the USP features for AgriFlux dashboard.
"""

import numpy as np
import rasterio
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CarbonEstimate:
    """Container for carbon sequestration estimates."""
    total_biomass: float  # tons
    above_ground_biomass: float  # tons
    below_ground_biomass: float  # tons
    carbon_sequestered: float  # tons of CO2
    carbon_credits: float  # potential credits (tons CO2e)
    credit_value_usd: float  # estimated monetary value
    area_hectares: float
    biomass_per_hectare: float
    carbon_per_hectare: float
    environmental_impact: Dict[str, str]
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of carbon estimates."""
        return {
            'total_biomass_tons': round(self.total_biomass, 2),
            'carbon_sequestered_tons': round(self.carbon_sequestered, 2),
            'carbon_credits': round(self.carbon_credits, 2),
            'credit_value_usd': round(self.credit_value_usd, 2),
            'area_hectares': round(self.area_hectares, 2),
            'carbon_per_hectare': round(self.carbon_per_hectare, 2)
        }


class CarbonCalculator:
    """
    Calculates carbon sequestration from vegetation indices.
    
    Uses empirical relationships between NDVI and biomass,
    then converts to carbon sequestration estimates.
    """
    
    # Biomass estimation coefficients
    # Based on research: Biomass (kg/m²) ≈ a * NDVI + b
    BIOMASS_COEFFICIENTS = {
        'cropland': {'a': 2.5, 'b': -0.5},
        'grassland': {'a': 2.0, 'b': -0.3},
        'forest': {'a': 4.0, 'b': -1.0},
        'generic': {'a': 2.5, 'b': -0.5}
    }
    
    # Carbon content factors
    CARBON_FRACTION = 0.45  # 45% of dry biomass is carbon
    CO2_TO_C_RATIO = 3.67  # Molecular weight ratio CO2/C
    
    # Root-to-shoot ratios (below-ground to above-ground biomass)
    ROOT_SHOOT_RATIOS = {
        'cropland': 0.25,  # 25% below ground
        'grassland': 0.40,  # 40% below ground
        'forest': 0.30,    # 30% below ground
        'generic': 0.25
    }
    
    # Carbon credit pricing (USD per ton CO2e)
    # Using conservative voluntary market prices
    CARBON_CREDIT_PRICE = 15.0  # USD per ton CO2e
    
    def __init__(self, land_type: str = 'cropland'):
        """
        Initialize carbon calculator.
        
        Args:
            land_type: Type of land ('cropland', 'grassland', 'forest', 'generic')
        """
        self.land_type = land_type.lower()
        self.biomass_coef = self.BIOMASS_COEFFICIENTS.get(
            self.land_type,
            self.BIOMASS_COEFFICIENTS['generic']
        )
        self.root_shoot_ratio = self.ROOT_SHOOT_RATIOS.get(
            self.land_type,
            self.ROOT_SHOOT_RATIOS['generic']
        )
    
    def load_ndvi_data(self, geotiff_path: str) -> Tuple[np.ndarray, float]:
        """
        Load NDVI data and calculate pixel area.
        
        Args:
            geotiff_path: Path to NDVI GeoTIFF
            
        Returns:
            Tuple of (ndvi_array, pixel_area_m2)
        """
        try:
            with rasterio.open(geotiff_path) as src:
                ndvi = src.read(1)
                
                # Calculate pixel area from transform
                transform = src.transform
                pixel_width = abs(transform[0])
                pixel_height = abs(transform[4])
                pixel_area_m2 = pixel_width * pixel_height
                
                return ndvi, pixel_area_m2
        except Exception as e:
            logger.error(f"Failed to load NDVI data: {e}")
            raise
    
    def estimate_biomass_from_ndvi(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Estimate above-ground biomass from NDVI.
        
        Formula: Biomass (kg/m²) = a * NDVI + b
        
        Args:
            ndvi: NDVI array
            
        Returns:
            Biomass array (kg/m²)
        """
        a = self.biomass_coef['a']
        b = self.biomass_coef['b']
        
        # Calculate biomass
        biomass = a * ndvi + b
        
        # Constrain to non-negative values
        biomass = np.maximum(biomass, 0)
        
        # Mask invalid NDVI values
        biomass[~np.isfinite(ndvi)] = np.nan
        
        return biomass
    
    def calculate_total_biomass(self,
                               above_ground_biomass: np.ndarray,
                               pixel_area_m2: float) -> Tuple[float, float, float]:
        """
        Calculate total biomass including below-ground component.
        
        Args:
            above_ground_biomass: Above-ground biomass array (kg/m²)
            pixel_area_m2: Area of each pixel in m²
            
        Returns:
            Tuple of (total_biomass_tons, above_ground_tons, below_ground_tons)
        """
        # Get valid biomass values
        valid_biomass = above_ground_biomass[np.isfinite(above_ground_biomass)]
        
        if len(valid_biomass) == 0:
            return 0.0, 0.0, 0.0
        
        # Calculate total above-ground biomass
        # Sum of (biomass per m² * pixel area) for all pixels
        above_ground_kg = np.sum(valid_biomass) * pixel_area_m2
        above_ground_tons = above_ground_kg / 1000.0
        
        # Estimate below-ground biomass using root-to-shoot ratio
        below_ground_tons = above_ground_tons * self.root_shoot_ratio
        
        # Total biomass
        total_biomass_tons = above_ground_tons + below_ground_tons
        
        return total_biomass_tons, above_ground_tons, below_ground_tons
    
    def calculate_carbon_sequestration(self,
                                      total_biomass_tons: float) -> Tuple[float, float]:
        """
        Calculate carbon sequestration from biomass.
        
        Args:
            total_biomass_tons: Total biomass in tons
            
        Returns:
            Tuple of (carbon_tons, co2_equivalent_tons)
        """
        # Calculate carbon content
        carbon_tons = total_biomass_tons * self.CARBON_FRACTION
        
        # Convert to CO2 equivalent
        co2_equivalent_tons = carbon_tons * self.CO2_TO_C_RATIO
        
        return carbon_tons, co2_equivalent_tons
    
    def estimate_carbon_credits(self,
                               co2_tons: float,
                               sequestration_period_years: float = 1.0) -> Tuple[float, float]:
        """
        Estimate potential carbon credits and monetary value.
        
        Args:
            co2_tons: CO2 equivalent tons sequestered
            sequestration_period_years: Time period for sequestration
            
        Returns:
            Tuple of (carbon_credits, value_usd)
        """
        # Annual sequestration rate
        annual_co2_tons = co2_tons / sequestration_period_years
        
        # Carbon credits (1 credit = 1 ton CO2e)
        carbon_credits = annual_co2_tons
        
        # Monetary value
        value_usd = carbon_credits * self.CARBON_CREDIT_PRICE
        
        return carbon_credits, value_usd
    
    def calculate_environmental_impact(self,
                                      co2_tons: float) -> Dict[str, str]:
        """
        Calculate environmental impact equivalents.
        
        Args:
            co2_tons: CO2 equivalent tons
            
        Returns:
            Dictionary of impact metrics with descriptions
        """
        # Conversion factors (approximate)
        CARS_PER_YEAR = 4.6  # tons CO2 per car per year
        TREES_PER_TON = 16.5  # trees needed to sequester 1 ton CO2 per year
        HOMES_PER_YEAR = 7.5  # tons CO2 per home per year
        
        cars_equivalent = co2_tons / CARS_PER_YEAR
        trees_equivalent = co2_tons * TREES_PER_TON
        homes_equivalent = co2_tons / HOMES_PER_YEAR
        
        impact = {
            'cars_off_road': f"{cars_equivalent:.1f} cars driven for one year",
            'trees_planted': f"{trees_equivalent:.0f} tree seedlings grown for 10 years",
            'homes_powered': f"{homes_equivalent:.1f} homes' energy use for one year",
            'co2_removed': f"{co2_tons:.2f} tons of CO2 removed from atmosphere"
        }
        
        return impact
    
    def calculate_carbon_estimate(self,
                                  ndvi_path: str,
                                  sequestration_period_years: float = 1.0) -> CarbonEstimate:
        """
        Calculate complete carbon sequestration estimate.
        
        Args:
            ndvi_path: Path to NDVI GeoTIFF
            sequestration_period_years: Time period for sequestration
            
        Returns:
            CarbonEstimate object
        """
        logger.info(f"Calculating carbon sequestration for {ndvi_path}")
        
        # Load NDVI data
        ndvi, pixel_area_m2 = self.load_ndvi_data(ndvi_path)
        
        # Estimate biomass
        above_ground_biomass = self.estimate_biomass_from_ndvi(ndvi)
        
        # Calculate total biomass
        total_biomass, above_ground, below_ground = self.calculate_total_biomass(
            above_ground_biomass, pixel_area_m2
        )
        
        # Calculate carbon sequestration
        carbon_tons, co2_tons = self.calculate_carbon_sequestration(total_biomass)
        
        # Estimate carbon credits
        credits, value_usd = self.estimate_carbon_credits(
            co2_tons, sequestration_period_years
        )
        
        # Calculate environmental impact
        impact = self.calculate_environmental_impact(co2_tons)
        
        # Calculate area
        valid_pixels = np.sum(np.isfinite(ndvi))
        area_m2 = valid_pixels * pixel_area_m2
        area_hectares = area_m2 / 10000.0
        
        # Per-hectare metrics
        biomass_per_ha = total_biomass / area_hectares if area_hectares > 0 else 0
        carbon_per_ha = co2_tons / area_hectares if area_hectares > 0 else 0
        
        estimate = CarbonEstimate(
            total_biomass=round(total_biomass, 2),
            above_ground_biomass=round(above_ground, 2),
            below_ground_biomass=round(below_ground, 2),
            carbon_sequestered=round(co2_tons, 2),
            carbon_credits=round(credits, 2),
            credit_value_usd=round(value_usd, 2),
            area_hectares=round(area_hectares, 2),
            biomass_per_hectare=round(biomass_per_ha, 2),
            carbon_per_hectare=round(carbon_per_ha, 2),
            environmental_impact=impact
        )
        
        logger.info(f"Carbon estimate: {co2_tons:.2f} tons CO2, ${value_usd:.2f} value")
        return estimate
    
    def compare_carbon_over_time(self,
                                ndvi_paths: List[Tuple[str, str]]) -> Dict[str, any]:
        """
        Compare carbon sequestration across multiple time points.
        
        Args:
            ndvi_paths: List of (date, path) tuples
            
        Returns:
            Dictionary with temporal comparison data
        """
        estimates = []
        dates = []
        
        for date, path in ndvi_paths:
            try:
                estimate = self.calculate_carbon_estimate(path)
                estimates.append(estimate)
                dates.append(date)
            except Exception as e:
                logger.warning(f"Failed to process {date}: {e}")
        
        if not estimates:
            return {}
        
        # Calculate trends
        carbon_values = [e.carbon_sequestered for e in estimates]
        
        if len(carbon_values) > 1:
            trend = "increasing" if carbon_values[-1] > carbon_values[0] else "decreasing"
            change = carbon_values[-1] - carbon_values[0]
            change_percent = (change / carbon_values[0] * 100) if carbon_values[0] > 0 else 0
        else:
            trend = "stable"
            change = 0
            change_percent = 0
        
        return {
            'dates': dates,
            'estimates': estimates,
            'trend': trend,
            'total_change': round(change, 2),
            'change_percent': round(change_percent, 1),
            'latest_estimate': estimates[-1] if estimates else None
        }


def calculate_carbon_from_imagery(imagery_id: int,
                                  db_manager,
                                  land_type: str = 'cropland') -> Optional[CarbonEstimate]:
    """
    Convenience function to calculate carbon from database imagery.
    
    Args:
        imagery_id: Database ID of imagery record
        db_manager: DatabaseManager instance
        land_type: Type of land
        
    Returns:
        CarbonEstimate or None if calculation fails
    """
    try:
        # Get imagery record
        record = db_manager.get_processed_imagery(imagery_id)
        
        if not record:
            logger.error(f"Imagery record {imagery_id} not found")
            return None
        
        # Get NDVI path
        ndvi_path = record.get('ndvi_path')
        if not ndvi_path:
            logger.error("NDVI not available for this imagery")
            return None
        
        # Calculate carbon estimate
        calculator = CarbonCalculator(land_type=land_type)
        estimate = calculator.calculate_carbon_estimate(ndvi_path)
        
        return estimate
        
    except Exception as e:
        logger.error(f"Failed to calculate carbon: {e}")
        return None


def calculate_carbon_trend(tile_id: str,
                          db_manager,
                          land_type: str = 'cropland') -> Optional[Dict[str, any]]:
    """
    Calculate carbon sequestration trend over time.
    
    Args:
        tile_id: Tile identifier
        db_manager: DatabaseManager instance
        land_type: Type of land
        
    Returns:
        Dictionary with trend data or None if calculation fails
    """
    try:
        # Get temporal series
        records = db_manager.get_temporal_series(tile_id)
        
        if not records:
            logger.error(f"No imagery found for tile {tile_id}")
            return None
        
        # Prepare NDVI paths
        ndvi_paths = []
        for record in records:
            date = record.get('acquisition_date')
            path = record.get('ndvi_path')
            if date and path:
                ndvi_paths.append((date, path))
        
        if not ndvi_paths:
            logger.error("No valid NDVI data found")
            return None
        
        # Calculate trend
        calculator = CarbonCalculator(land_type=land_type)
        trend_data = calculator.compare_carbon_over_time(ndvi_paths)
        
        return trend_data
        
    except Exception as e:
        logger.error(f"Failed to calculate carbon trend: {e}")
        return None
