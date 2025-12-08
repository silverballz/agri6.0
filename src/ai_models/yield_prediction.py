"""
Yield Prediction Module

Estimates crop yield based on NDVI trends and historical patterns.
Provides confidence intervals and uncertainty estimates.

Part of the USP features for AgriFlux dashboard.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class YieldEstimate:
    """Container for yield prediction results."""
    predicted_yield: float  # tons/hectare
    confidence_interval: Tuple[float, float]  # (lower, upper)
    confidence_level: float  # 0-100%
    prediction_date: str
    growth_stage: str
    ndvi_trend: str  # 'increasing', 'stable', 'decreasing'
    yield_category: str  # 'excellent', 'good', 'average', 'below_average', 'poor'
    factors: Dict[str, float]
    recommendations: List[str]
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of yield prediction."""
        return {
            'predicted_yield': self.predicted_yield,
            'lower_bound': self.confidence_interval[0],
            'upper_bound': self.confidence_interval[1],
            'confidence': self.confidence_level,
            'category': self.yield_category,
            'trend': self.ndvi_trend
        }


class YieldPredictor:
    """
    Predicts crop yield based on vegetation index trends.
    
    Uses empirical relationships between NDVI and crop yield,
    with adjustments for growth stage and temporal patterns.
    """
    
    # Baseline yield parameters (tons/hectare) for different crops
    CROP_BASELINES = {
        'wheat': {'baseline': 4.5, 'max': 8.0, 'min': 1.5},
        'rice': {'baseline': 5.0, 'max': 9.0, 'min': 2.0},
        'corn': {'baseline': 7.0, 'max': 12.0, 'min': 2.5},
        'soybean': {'baseline': 3.0, 'max': 5.5, 'min': 1.0},
        'generic': {'baseline': 4.0, 'max': 7.0, 'min': 1.5}
    }
    
    # NDVI-yield relationship coefficients
    # Yield multiplier = baseline * (1 + coef * (NDVI - 0.6))
    NDVI_YIELD_COEFFICIENT = 2.5
    
    # Growth stage multipliers
    GROWTH_STAGE_FACTORS = {
        'early': 0.3,      # Early vegetative
        'vegetative': 0.6,  # Active growth
        'reproductive': 1.0, # Flowering/grain fill
        'maturity': 0.9,    # Near harvest
        'unknown': 0.7
    }
    
    def __init__(self, crop_type: str = 'generic'):
        """
        Initialize yield predictor.
        
        Args:
            crop_type: Type of crop ('wheat', 'rice', 'corn', 'soybean', 'generic')
        """
        self.crop_type = crop_type.lower()
        self.baseline_params = self.CROP_BASELINES.get(
            self.crop_type,
            self.CROP_BASELINES['generic']
        )
    
    def determine_growth_stage(self,
                              acquisition_date: str,
                              planting_date: Optional[str] = None) -> str:
        """
        Determine crop growth stage based on date.
        
        Args:
            acquisition_date: Date of satellite acquisition (ISO format)
            planting_date: Optional planting date (ISO format)
            
        Returns:
            Growth stage identifier
        """
        if not planting_date:
            # Use month as proxy if planting date unknown
            try:
                month = datetime.fromisoformat(acquisition_date).month
                
                # Northern hemisphere growing season approximation
                if month in [4, 5]:
                    return 'early'
                elif month in [6, 7]:
                    return 'vegetative'
                elif month in [8, 9]:
                    return 'reproductive'
                elif month in [10, 11]:
                    return 'maturity'
                else:
                    return 'unknown'
            except:
                return 'unknown'
        
        try:
            # Calculate days since planting
            plant_dt = datetime.fromisoformat(planting_date)
            acq_dt = datetime.fromisoformat(acquisition_date)
            days_since_planting = (acq_dt - plant_dt).days
            
            # Approximate growth stages (adjust for specific crops)
            if days_since_planting < 30:
                return 'early'
            elif days_since_planting < 60:
                return 'vegetative'
            elif days_since_planting < 100:
                return 'reproductive'
            elif days_since_planting < 130:
                return 'maturity'
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    def analyze_ndvi_trend(self,
                          ndvi_values: List[float],
                          dates: List[str]) -> Tuple[str, float]:
        """
        Analyze NDVI temporal trend.
        
        Args:
            ndvi_values: List of mean NDVI values
            dates: Corresponding dates
            
        Returns:
            Tuple of (trend_description, trend_slope)
        """
        if len(ndvi_values) < 2:
            return 'unknown', 0.0
        
        # Calculate simple linear trend
        x = np.arange(len(ndvi_values))
        y = np.array(ndvi_values)
        
        # Remove NaN values
        valid_mask = np.isfinite(y)
        if np.sum(valid_mask) < 2:
            return 'unknown', 0.0
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Linear regression
        slope = np.polyfit(x_valid, y_valid, 1)[0]
        
        # Classify trend
        if slope > 0.02:
            trend = 'increasing'
        elif slope < -0.02:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return trend, float(slope)
    
    def calculate_base_yield(self, mean_ndvi: float) -> float:
        """
        Calculate base yield estimate from NDVI.
        
        Args:
            mean_ndvi: Mean NDVI value for field
            
        Returns:
            Base yield estimate (tons/hectare)
        """
        baseline = self.baseline_params['baseline']
        
        # NDVI-yield relationship
        # Optimal NDVI around 0.7-0.8 for most crops
        ndvi_factor = 1 + self.NDVI_YIELD_COEFFICIENT * (mean_ndvi - 0.6)
        
        # Constrain to reasonable range
        ndvi_factor = max(0.3, min(1.8, ndvi_factor))
        
        base_yield = baseline * ndvi_factor
        
        return base_yield
    
    def apply_adjustments(self,
                         base_yield: float,
                         growth_stage: str,
                         trend_slope: float,
                         additional_factors: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Apply adjustments to base yield estimate.
        
        Args:
            base_yield: Base yield from NDVI
            growth_stage: Current growth stage
            trend_slope: NDVI trend slope
            additional_factors: Optional additional adjustment factors
            
        Returns:
            Tuple of (adjusted_yield, factors_dict)
        """
        factors = {}
        adjusted_yield = base_yield
        
        # Growth stage adjustment
        stage_factor = self.GROWTH_STAGE_FACTORS.get(growth_stage, 0.7)
        factors['growth_stage'] = stage_factor
        
        # Trend adjustment
        if trend_slope > 0.02:
            trend_factor = 1.1  # Positive trend bonus
        elif trend_slope < -0.02:
            trend_factor = 0.9  # Negative trend penalty
        else:
            trend_factor = 1.0
        factors['trend'] = trend_factor
        
        # Apply factors
        adjusted_yield = base_yield * stage_factor * trend_factor
        
        # Apply additional factors if provided
        if additional_factors:
            for name, value in additional_factors.items():
                adjusted_yield *= value
                factors[name] = value
        
        # Constrain to crop-specific range
        adjusted_yield = max(
            self.baseline_params['min'],
            min(self.baseline_params['max'], adjusted_yield)
        )
        
        return adjusted_yield, factors
    
    def calculate_confidence_interval(self,
                                     predicted_yield: float,
                                     data_quality: float = 0.8,
                                     n_observations: int = 1) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for yield prediction.
        
        Args:
            predicted_yield: Point estimate of yield
            data_quality: Quality score 0-1 (based on cloud cover, etc.)
            n_observations: Number of temporal observations
            
        Returns:
            Tuple of (lower_bound, upper_bound, confidence_level)
        """
        # Base uncertainty (as percentage of predicted yield)
        base_uncertainty = 0.25  # ±25%
        
        # Adjust based on data quality
        quality_factor = 1.0 - (data_quality * 0.3)  # Better quality = less uncertainty
        
        # Adjust based on number of observations
        obs_factor = 1.0 / np.sqrt(max(1, n_observations))
        
        # Combined uncertainty
        uncertainty = base_uncertainty * quality_factor * obs_factor
        
        # Calculate bounds
        margin = predicted_yield * uncertainty
        lower_bound = max(0, predicted_yield - margin)
        upper_bound = predicted_yield + margin
        
        # Confidence level (higher with more data and better quality)
        confidence_level = min(95, 60 + (data_quality * 20) + (min(n_observations, 5) * 3))
        
        return lower_bound, upper_bound, confidence_level
    
    def categorize_yield(self, predicted_yield: float) -> str:
        """
        Categorize yield prediction.
        
        Args:
            predicted_yield: Predicted yield value
            
        Returns:
            Category string
        """
        baseline = self.baseline_params['baseline']
        max_yield = self.baseline_params['max']
        
        if predicted_yield >= max_yield * 0.9:
            return 'excellent'
        elif predicted_yield >= baseline * 1.2:
            return 'good'
        elif predicted_yield >= baseline * 0.8:
            return 'average'
        elif predicted_yield >= baseline * 0.6:
            return 'below_average'
        else:
            return 'poor'
    
    def generate_recommendations(self,
                                yield_category: str,
                                trend: str,
                                growth_stage: str) -> List[str]:
        """
        Generate recommendations based on yield prediction.
        
        Args:
            yield_category: Yield category
            trend: NDVI trend
            growth_stage: Growth stage
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Category-based recommendations
        if yield_category in ['poor', 'below_average']:
            recommendations.append(
                "⚠️ Below-average yield predicted. Consider interventions to improve crop health."
            )
            if growth_stage in ['early', 'vegetative']:
                recommendations.append(
                    "💧 Optimize irrigation and nutrient application during critical growth phase."
                )
        elif yield_category == 'excellent':
            recommendations.append(
                "✅ Excellent yield potential. Maintain current management practices."
            )
        
        # Trend-based recommendations
        if trend == 'decreasing':
            recommendations.append(
                "📉 Declining NDVI trend detected. Investigate potential stress factors."
            )
            recommendations.append(
                "🔍 Check for water stress, nutrient deficiency, or pest/disease issues."
            )
        elif trend == 'increasing':
            recommendations.append(
                "📈 Positive growth trend. Continue monitoring to maintain trajectory."
            )
        
        # Growth stage recommendations
        if growth_stage == 'reproductive':
            recommendations.append(
                "🌾 Critical reproductive stage. Ensure adequate water and nutrients."
            )
        elif growth_stage == 'maturity':
            recommendations.append(
                "⏰ Approaching maturity. Plan harvest timing for optimal yield."
            )
        
        if not recommendations:
            recommendations.append(
                "📊 Continue regular monitoring and maintain standard practices."
            )
        
        return recommendations
    
    def predict_yield(self,
                     mean_ndvi: float,
                     acquisition_date: str,
                     ndvi_history: Optional[List[Tuple[str, float]]] = None,
                     planting_date: Optional[str] = None,
                     data_quality: float = 0.8) -> YieldEstimate:
        """
        Generate complete yield prediction.
        
        Args:
            mean_ndvi: Current mean NDVI value
            acquisition_date: Date of current observation
            ndvi_history: Optional list of (date, ndvi) tuples for trend analysis
            planting_date: Optional planting date
            data_quality: Data quality score 0-1
            
        Returns:
            YieldEstimate object
        """
        logger.info(f"Predicting yield for NDVI={mean_ndvi:.3f}")
        
        # Determine growth stage
        growth_stage = self.determine_growth_stage(acquisition_date, planting_date)
        
        # Analyze trend if history available
        if ndvi_history and len(ndvi_history) > 1:
            dates = [d for d, _ in ndvi_history]
            values = [v for _, v in ndvi_history]
            trend, trend_slope = self.analyze_ndvi_trend(values, dates)
            n_observations = len(ndvi_history)
        else:
            trend = 'unknown'
            trend_slope = 0.0
            n_observations = 1
        
        # Calculate base yield
        base_yield = self.calculate_base_yield(mean_ndvi)
        
        # Apply adjustments
        predicted_yield, factors = self.apply_adjustments(
            base_yield, growth_stage, trend_slope
        )
        
        # Calculate confidence interval
        lower, upper, confidence = self.calculate_confidence_interval(
            predicted_yield, data_quality, n_observations
        )
        
        # Categorize yield
        category = self.categorize_yield(predicted_yield)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(category, trend, growth_stage)
        
        estimate = YieldEstimate(
            predicted_yield=round(predicted_yield, 2),
            confidence_interval=(round(lower, 2), round(upper, 2)),
            confidence_level=round(confidence, 1),
            prediction_date=acquisition_date,
            growth_stage=growth_stage,
            ndvi_trend=trend,
            yield_category=category,
            factors=factors,
            recommendations=recommendations
        )
        
        logger.info(f"Yield prediction: {predicted_yield:.2f} t/ha ({category})")
        return estimate


def predict_yield_from_imagery(imagery_id: int,
                               db_manager,
                               crop_type: str = 'generic',
                               planting_date: Optional[str] = None) -> Optional[YieldEstimate]:
    """
    Convenience function to predict yield from database imagery.
    
    Args:
        imagery_id: Database ID of imagery record
        db_manager: DatabaseManager instance
        crop_type: Type of crop
        planting_date: Optional planting date
        
    Returns:
        YieldEstimate or None if prediction fails
    """
    try:
        import rasterio
        
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
        
        # Load NDVI and calculate mean
        with rasterio.open(ndvi_path) as src:
            ndvi_data = src.read(1)
            valid_ndvi = ndvi_data[np.isfinite(ndvi_data)]
            mean_ndvi = float(np.mean(valid_ndvi))
        
        # Get acquisition date
        acquisition_date = record.get('acquisition_date')
        
        # Get data quality (inverse of cloud coverage)
        cloud_coverage = record.get('cloud_coverage', 20.0)
        data_quality = 1.0 - (cloud_coverage / 100.0)
        
        # Get historical NDVI if available
        tile_id = record.get('tile_id')
        history_records = db_manager.get_temporal_series(tile_id)
        
        ndvi_history = []
        for hist_record in history_records:
            hist_ndvi_path = hist_record.get('ndvi_path')
            hist_date = hist_record.get('acquisition_date')
            
            if hist_ndvi_path and hist_date:
                try:
                    with rasterio.open(hist_ndvi_path) as src:
                        hist_data = src.read(1)
                        valid_hist = hist_data[np.isfinite(hist_data)]
                        if len(valid_hist) > 0:
                            hist_mean = float(np.mean(valid_hist))
                            ndvi_history.append((hist_date, hist_mean))
                except:
                    pass
        
        # Create predictor and generate estimate
        predictor = YieldPredictor(crop_type=crop_type)
        estimate = predictor.predict_yield(
            mean_ndvi=mean_ndvi,
            acquisition_date=acquisition_date,
            ndvi_history=ndvi_history if ndvi_history else None,
            planting_date=planting_date,
            data_quality=data_quality
        )
        
        return estimate
        
    except Exception as e:
        logger.error(f"Failed to predict yield: {e}")
        return None
