"""
Test suite for USP (Unique Selling Proposition) features.

Tests the following modules:
- Multi-temporal change detection
- Precision irrigation zone recommender
- Yield prediction
- Carbon sequestration calculator
- Before/after comparison widget
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds

# Import USP modules
from src.ai_models.change_detection import (
    ChangeDetector,
    ChangeType,
    get_change_type_color,
    get_change_type_label
)
from src.ai_models.irrigation_zones import (
    IrrigationZoneRecommender,
    IrrigationZone
)
from src.ai_models.yield_prediction import (
    YieldPredictor,
    YieldEstimate
)
from src.ai_models.carbon_calculator import (
    CarbonCalculator,
    CarbonEstimate
)
from src.dashboard.components.comparison_widget import ComparisonWidget


# Fixtures for test data
@pytest.fixture
def sample_ndvi_data():
    """Create sample NDVI data."""
    # Create 100x100 array with realistic NDVI values
    np.random.seed(42)
    ndvi = np.random.uniform(0.3, 0.8, (100, 100))
    return ndvi.astype(np.float32)


@pytest.fixture
def sample_geotiff(sample_ndvi_data, tmp_path):
    """Create a sample GeoTIFF file."""
    filepath = tmp_path / "test_ndvi.tif"
    
    # Define transform and CRS
    transform = from_bounds(0, 0, 1000, 1000, 100, 100)
    
    # Write GeoTIFF
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=100,
        width=100,
        count=1,
        dtype=rasterio.float32,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(sample_ndvi_data, 1)
    
    return str(filepath)


@pytest.fixture
def two_date_geotiffs(tmp_path):
    """Create two GeoTIFF files for temporal comparison."""
    np.random.seed(42)
    
    # Before data (lower NDVI)
    before_data = np.random.uniform(0.3, 0.6, (100, 100)).astype(np.float32)
    before_path = tmp_path / "before.tif"
    
    # After data (higher NDVI - improvement)
    after_data = before_data + np.random.uniform(0.0, 0.2, (100, 100)).astype(np.float32)
    after_path = tmp_path / "after.tif"
    
    transform = from_bounds(0, 0, 1000, 1000, 100, 100)
    
    for path, data in [(before_path, before_data), (after_path, after_data)]:
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=100,
            width=100,
            count=1,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
    
    return str(before_path), str(after_path)


@pytest.fixture
def sample_ndwi_ndsi_geotiffs(tmp_path):
    """Create sample NDWI and NDSI GeoTIFF files."""
    np.random.seed(42)
    
    # NDWI data (water content indicator)
    ndwi_data = np.random.uniform(-0.2, 0.3, (100, 100)).astype(np.float32)
    ndwi_path = tmp_path / "ndwi.tif"
    
    # NDSI data (soil moisture indicator)
    ndsi_data = np.random.uniform(-0.1, 0.2, (100, 100)).astype(np.float32)
    ndsi_path = tmp_path / "ndsi.tif"
    
    transform = from_bounds(0, 0, 1000, 1000, 100, 100)
    
    for path, data in [(ndwi_path, ndwi_data), (ndsi_path, ndsi_data)]:
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=100,
            width=100,
            count=1,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
    
    return str(ndwi_path), str(ndsi_path)


# Tests for Change Detection
class TestChangeDetection:
    """Test multi-temporal change detection."""
    
    def test_change_detector_initialization(self):
        """Test ChangeDetector initialization."""
        detector = ChangeDetector()
        assert detector.significant_threshold == 0.15
        assert detector.moderate_threshold == 0.05
    
    def test_load_geotiff(self, sample_geotiff):
        """Test loading GeoTIFF data."""
        detector = ChangeDetector()
        data, metadata = detector.load_index_from_geotiff(sample_geotiff)
        
        assert data.shape == (100, 100)
        assert 'crs' in metadata
        assert 'transform' in metadata
    
    def test_calculate_change_magnitude(self):
        """Test change magnitude calculation."""
        detector = ChangeDetector()
        
        before = np.array([[0.5, 0.6], [0.7, 0.8]])
        after = np.array([[0.6, 0.7], [0.6, 0.9]])
        
        change = detector.calculate_change_magnitude(before, after)
        
        assert change.shape == before.shape
        assert change[0, 0] == pytest.approx(0.1)
        assert change[1, 0] == pytest.approx(-0.1)
    
    def test_classify_changes(self):
        """Test change classification."""
        detector = ChangeDetector()
        
        change_magnitude = np.array([
            [0.2, 0.1, 0.0],    # significant, moderate, no change
            [-0.1, -0.2, np.nan]  # moderate deg, significant deg, invalid
        ])
        
        change_types = detector.classify_changes(change_magnitude)
        
        assert change_types[0, 0] == 0  # Significant improvement
        assert change_types[0, 1] == 1  # Moderate improvement
        assert change_types[0, 2] == 2  # No change
        assert change_types[1, 0] == 3  # Moderate degradation
        assert change_types[1, 1] == 4  # Significant degradation
        assert change_types[1, 2] == -1  # Invalid
    
    def test_detect_changes_complete(self, two_date_geotiffs):
        """Test complete change detection workflow."""
        before_path, after_path = two_date_geotiffs
        
        detector = ChangeDetector()
        result = detector.detect_changes(before_path, after_path)
        
        assert result.change_magnitude.shape == (100, 100)
        assert result.change_type.shape == (100, 100)
        assert 0 <= result.change_percentage <= 100
        assert result.improvement_area >= 0
        assert result.degradation_area >= 0
        assert 'mean_change' in result.statistics
    
    def test_get_change_hotspots(self, two_date_geotiffs):
        """Test hotspot identification."""
        before_path, after_path = two_date_geotiffs
        
        detector = ChangeDetector()
        result = detector.detect_changes(before_path, after_path)
        
        improvement_hotspots, degradation_hotspots = detector.get_change_hotspots(
            result.change_magnitude
        )
        
        assert improvement_hotspots.shape == result.change_magnitude.shape
        assert degradation_hotspots.shape == result.change_magnitude.shape
        assert improvement_hotspots.dtype == bool
    
    def test_change_type_utilities(self):
        """Test utility functions for change types."""
        assert get_change_type_color(0) == '#00ff00'
        assert get_change_type_color(4) == '#ff0000'
        
        assert get_change_type_label(0) == 'Significant Improvement'
        assert get_change_type_label(4) == 'Significant Degradation'


# Tests for Irrigation Zones
class TestIrrigationZones:
    """Test precision irrigation zone recommender."""
    
    def test_recommender_initialization(self):
        """Test IrrigationZoneRecommender initialization."""
        recommender = IrrigationZoneRecommender(n_zones=4)
        assert recommender.n_zones == 4
    
    def test_calculate_water_stress_index(self):
        """Test water stress index calculation."""
        recommender = IrrigationZoneRecommender()
        
        ndwi = np.array([[0.2, 0.1], [0.0, -0.1]])
        ndsi = np.array([[0.1, 0.0], [-0.1, -0.2]])
        
        wsi = recommender.calculate_water_stress_index(ndwi, ndsi)
        
        assert wsi.shape == ndwi.shape
        assert np.all(np.isfinite(wsi))
    
    def test_classify_water_stress(self):
        """Test water stress classification."""
        recommender = IrrigationZoneRecommender()
        
        # Test different stress levels
        stress_severe, priority_severe = recommender.classify_water_stress(-0.4)
        assert stress_severe == 'severe'
        assert priority_severe == 1
        
        stress_low, priority_low = recommender.classify_water_stress(0.4)
        assert stress_low == 'low'
        assert priority_low == 4
    
    def test_generate_recommendation(self):
        """Test recommendation generation."""
        recommender = IrrigationZoneRecommender()
        
        rec, freq, amount = recommender.generate_recommendation('severe', -0.4, 0.0)
        
        assert 'URGENT' in rec or 'urgent' in rec.lower()
        assert freq is not None
        assert amount is not None
    
    def test_create_irrigation_zones(self, sample_ndwi_ndsi_geotiffs):
        """Test complete irrigation zone creation."""
        ndwi_path, ndsi_path = sample_ndwi_ndsi_geotiffs
        
        recommender = IrrigationZoneRecommender(n_zones=4)
        plan = recommender.create_irrigation_zones(ndwi_path, ndsi_path)
        
        assert len(plan.zones) == 4
        assert plan.zone_map.shape == (100, 100)
        assert plan.total_area > 0
        assert 0 <= plan.high_priority_area <= 100
        assert plan.water_savings_estimate > 0
        
        # Check zone properties
        for zone in plan.zones:
            assert isinstance(zone, IrrigationZone)
            assert zone.zone_id >= 0
            assert zone.water_stress_level in ['severe', 'high', 'moderate', 'low']
            assert 1 <= zone.priority <= 4
            assert 0 <= zone.area_percentage <= 100


# Tests for Yield Prediction
class TestYieldPrediction:
    """Test yield prediction module."""
    
    def test_predictor_initialization(self):
        """Test YieldPredictor initialization."""
        predictor = YieldPredictor(crop_type='wheat')
        assert predictor.crop_type == 'wheat'
        assert 'baseline' in predictor.baseline_params
    
    def test_determine_growth_stage(self):
        """Test growth stage determination."""
        predictor = YieldPredictor()
        
        # Test with month-based approximation
        stage_may = predictor.determine_growth_stage('2024-05-15')
        assert stage_may in ['early', 'vegetative', 'reproductive', 'maturity', 'unknown']
        
        stage_aug = predictor.determine_growth_stage('2024-08-15')
        assert stage_aug in ['early', 'vegetative', 'reproductive', 'maturity', 'unknown']
    
    def test_analyze_ndvi_trend(self):
        """Test NDVI trend analysis."""
        predictor = YieldPredictor()
        
        # Increasing trend
        ndvi_values = [0.5, 0.55, 0.6, 0.65, 0.7]
        dates = ['2024-05-01', '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01']
        
        trend, slope = predictor.analyze_ndvi_trend(ndvi_values, dates)
        
        assert trend in ['increasing', 'stable', 'decreasing']
        assert isinstance(slope, float)
    
    def test_calculate_base_yield(self):
        """Test base yield calculation."""
        predictor = YieldPredictor(crop_type='wheat')
        
        yield_high = predictor.calculate_base_yield(0.8)
        yield_low = predictor.calculate_base_yield(0.4)
        
        assert yield_high > yield_low
        assert yield_high > 0
    
    def test_categorize_yield(self):
        """Test yield categorization."""
        predictor = YieldPredictor(crop_type='wheat')
        
        baseline = predictor.baseline_params['baseline']
        
        category_high = predictor.categorize_yield(baseline * 1.5)
        category_low = predictor.categorize_yield(baseline * 0.5)
        
        assert category_high in ['excellent', 'good']
        assert category_low in ['below_average', 'poor']
    
    def test_predict_yield(self):
        """Test complete yield prediction."""
        predictor = YieldPredictor(crop_type='wheat')
        
        estimate = predictor.predict_yield(
            mean_ndvi=0.7,
            acquisition_date='2024-08-15',
            data_quality=0.9
        )
        
        assert isinstance(estimate, YieldEstimate)
        assert estimate.predicted_yield > 0
        assert len(estimate.confidence_interval) == 2
        assert estimate.confidence_interval[0] < estimate.predicted_yield
        assert estimate.confidence_interval[1] > estimate.predicted_yield
        assert 0 <= estimate.confidence_level <= 100
        assert estimate.yield_category in ['excellent', 'good', 'average', 'below_average', 'poor']
        assert len(estimate.recommendations) > 0


# Tests for Carbon Calculator
class TestCarbonCalculator:
    """Test carbon sequestration calculator."""
    
    def test_calculator_initialization(self):
        """Test CarbonCalculator initialization."""
        calculator = CarbonCalculator(land_type='cropland')
        assert calculator.land_type == 'cropland'
        assert 'a' in calculator.biomass_coef
    
    def test_estimate_biomass_from_ndvi(self):
        """Test biomass estimation."""
        calculator = CarbonCalculator()
        
        ndvi = np.array([[0.7, 0.6], [0.5, 0.4]])
        biomass = calculator.estimate_biomass_from_ndvi(ndvi)
        
        assert biomass.shape == ndvi.shape
        assert np.all(biomass >= 0)
        assert biomass[0, 0] > biomass[1, 1]  # Higher NDVI = higher biomass
    
    def test_calculate_carbon_sequestration(self):
        """Test carbon sequestration calculation."""
        calculator = CarbonCalculator()
        
        carbon_tons, co2_tons = calculator.calculate_carbon_sequestration(100.0)
        
        assert carbon_tons > 0
        assert co2_tons > carbon_tons  # CO2 is heavier than C
        assert co2_tons == pytest.approx(carbon_tons * calculator.CO2_TO_C_RATIO)
    
    def test_estimate_carbon_credits(self):
        """Test carbon credit estimation."""
        calculator = CarbonCalculator()
        
        credits, value = calculator.estimate_carbon_credits(100.0)
        
        assert credits > 0
        assert value > 0
        assert value == pytest.approx(credits * calculator.CARBON_CREDIT_PRICE)
    
    def test_calculate_environmental_impact(self):
        """Test environmental impact calculation."""
        calculator = CarbonCalculator()
        
        impact = calculator.calculate_environmental_impact(100.0)
        
        assert 'cars_off_road' in impact
        assert 'trees_planted' in impact
        assert 'homes_powered' in impact
        assert 'co2_removed' in impact
    
    def test_calculate_carbon_estimate(self, sample_geotiff):
        """Test complete carbon estimation."""
        calculator = CarbonCalculator(land_type='cropland')
        
        estimate = calculator.calculate_carbon_estimate(sample_geotiff)
        
        assert isinstance(estimate, CarbonEstimate)
        assert estimate.total_biomass > 0
        assert estimate.carbon_sequestered > 0
        assert estimate.carbon_credits > 0
        assert estimate.credit_value_usd > 0
        assert estimate.area_hectares > 0
        assert len(estimate.environmental_impact) > 0


# Tests for Comparison Widget
class TestComparisonWidget:
    """Test before/after comparison widget."""
    
    def test_widget_initialization(self):
        """Test ComparisonWidget initialization."""
        widget = ComparisonWidget()
        assert 'NDVI' in widget.colorscales
    
    def test_load_image_data(self, sample_geotiff):
        """Test image data loading."""
        widget = ComparisonWidget()
        data, metadata = widget.load_image_data(sample_geotiff)
        
        assert data.shape == (100, 100)
        assert 'bounds' in metadata
        assert 'crs' in metadata
    
    def test_create_side_by_side_view(self, two_date_geotiffs):
        """Test side-by-side view creation."""
        before_path, after_path = two_date_geotiffs
        
        widget = ComparisonWidget()
        fig = widget.create_side_by_side_view(
            before_path, after_path,
            '2024-01-01', '2024-06-01',
            'NDVI'
        )
        
        assert fig is not None
        assert len(fig.data) == 2  # Two heatmaps
    
    def test_create_difference_map(self, two_date_geotiffs):
        """Test difference map creation."""
        before_path, after_path = two_date_geotiffs
        
        widget = ComparisonWidget()
        fig = widget.create_difference_map(
            before_path, after_path,
            '2024-01-01', '2024-06-01',
            'NDVI'
        )
        
        assert fig is not None
        assert len(fig.data) == 1  # One difference heatmap
    
    def test_create_histogram_comparison(self, two_date_geotiffs):
        """Test histogram comparison creation."""
        before_path, after_path = two_date_geotiffs
        
        widget = ComparisonWidget()
        fig = widget.create_histogram_comparison(
            before_path, after_path,
            '2024-01-01', '2024-06-01',
            'NDVI'
        )
        
        assert fig is not None
        assert len(fig.data) == 2  # Two histograms
    
    def test_create_statistics_comparison(self, two_date_geotiffs):
        """Test statistics comparison."""
        before_path, after_path = two_date_geotiffs
        
        widget = ComparisonWidget()
        stats = widget.create_statistics_comparison(before_path, after_path, 'NDVI')
        
        assert 'before' in stats
        assert 'after' in stats
        assert 'change' in stats
        assert 'mean' in stats['before']
        assert 'mean' in stats['after']
        assert 'mean' in stats['change']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
