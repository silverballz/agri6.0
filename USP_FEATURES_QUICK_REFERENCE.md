# USP Features Quick Reference Guide

## Quick Start

### Installation
All USP features are already included in the AgriFlux codebase. No additional installation required.

### Import Statements
```python
# Change Detection
from src.ai_models.change_detection import ChangeDetector, compare_imagery_dates

# Irrigation Zones
from src.ai_models.irrigation_zones import IrrigationZoneRecommender, create_irrigation_plan_from_db

# Yield Prediction
from src.ai_models.yield_prediction import YieldPredictor, predict_yield_from_imagery

# Carbon Calculator
from src.ai_models.carbon_calculator import CarbonCalculator, calculate_carbon_from_imagery

# Comparison Widget
from src.dashboard.components import render_comparison_widget, render_multi_date_slider
```

## 1. Change Detection

### Basic Usage
```python
from src.ai_models.change_detection import ChangeDetector

detector = ChangeDetector()
result = detector.detect_changes(
    before_path="data/processed/ndvi_before.tif",
    after_path="data/processed/ndvi_after.tif",
    index_name="NDVI"
)

print(f"Change: {result.change_percentage:.1f}%")
print(f"Improvement: {result.improvement_area:.1f}%")
```

### With Database
```python
from src.ai_models.change_detection import compare_imagery_dates
from src.database.db_manager import DatabaseManager

db = DatabaseManager()
result = compare_imagery_dates(
    before_imagery_id=1,
    after_imagery_id=2,
    db_manager=db,
    index_name="NDVI"
)
```

### Get Change Summary
```python
summary = result.get_change_summary()
# Returns: {
#   'total_change_percentage': 45.2,
#   'improvement_area_percentage': 30.1,
#   'degradation_area_percentage': 15.1,
#   'stable_area_percentage': 54.8,
#   'mean_change': 0.05,
#   'max_improvement': 0.35,
#   'max_degradation': -0.25
# }
```

### Export Change Map
```python
detector.export_change_map(
    result=result,
    output_path="data/processed/change_map.tif",
    reference_geotiff="data/processed/ndvi_before.tif"
)
```

## 2. Irrigation Zones

### Basic Usage
```python
from src.ai_models.irrigation_zones import IrrigationZoneRecommender

recommender = IrrigationZoneRecommender(n_zones=4)
plan = recommender.create_irrigation_zones(
    ndwi_path="data/processed/ndwi.tif",
    ndsi_path="data/processed/ndsi.tif"
)

print(f"High priority area: {plan.high_priority_area:.1f}%")
print(f"Water savings: {plan.water_savings_estimate:.0f}%")
```

### With Database
```python
from src.ai_models.irrigation_zones import create_irrigation_plan_from_db
from src.database.db_manager import DatabaseManager

db = DatabaseManager()
plan = create_irrigation_plan_from_db(
    imagery_id=1,
    db_manager=db,
    n_zones=4
)
```

### Access Zone Information
```python
for zone in plan.zones:
    print(f"Zone {zone.zone_id + 1}:")
    print(f"  Stress Level: {zone.water_stress_level}")
    print(f"  Priority: {zone.priority}")
    print(f"  Area: {zone.area_percentage:.1f}%")
    print(f"  Frequency: {zone.irrigation_frequency}")
    print(f"  Amount: {zone.water_amount}")
    print(f"  Recommendation: {zone.recommendation}")
```

### Export Zone Map
```python
recommender.export_zone_map(
    plan=plan,
    output_path="data/processed/irrigation_zones.tif",
    reference_geotiff="data/processed/ndwi.tif"
)
```

## 3. Yield Prediction

### Basic Usage
```python
from src.ai_models.yield_prediction import YieldPredictor

predictor = YieldPredictor(crop_type='wheat')
estimate = predictor.predict_yield(
    mean_ndvi=0.7,
    acquisition_date='2024-08-15',
    data_quality=0.9
)

print(f"Yield: {estimate.predicted_yield:.2f} t/ha")
print(f"Confidence: {estimate.confidence_level:.1f}%")
print(f"Category: {estimate.yield_category}")
```

### With Database
```python
from src.ai_models.yield_prediction import predict_yield_from_imagery
from src.database.db_manager import DatabaseManager

db = DatabaseManager()
estimate = predict_yield_from_imagery(
    imagery_id=1,
    db_manager=db,
    crop_type='wheat',
    planting_date='2024-04-15'  # Optional
)
```

### With Historical Data
```python
ndvi_history = [
    ('2024-05-01', 0.5),
    ('2024-06-01', 0.6),
    ('2024-07-01', 0.7),
    ('2024-08-01', 0.75)
]

estimate = predictor.predict_yield(
    mean_ndvi=0.75,
    acquisition_date='2024-08-01',
    ndvi_history=ndvi_history,
    planting_date='2024-04-15',
    data_quality=0.9
)
```

### Get Summary
```python
summary = estimate.get_summary()
# Returns: {
#   'predicted_yield': 5.2,
#   'lower_bound': 4.5,
#   'upper_bound': 5.9,
#   'confidence': 85.0,
#   'category': 'good',
#   'trend': 'increasing'
# }
```

### Access Recommendations
```python
for rec in estimate.recommendations:
    print(f"• {rec}")
```

## 4. Carbon Calculator

### Basic Usage
```python
from src.ai_models.carbon_calculator import CarbonCalculator

calculator = CarbonCalculator(land_type='cropland')
estimate = calculator.calculate_carbon_estimate(
    ndvi_path="data/processed/ndvi.tif"
)

print(f"Carbon: {estimate.carbon_sequestered:.2f} tons CO2")
print(f"Credits: {estimate.carbon_credits:.2f}")
print(f"Value: ${estimate.credit_value_usd:.2f}")
```

### With Database
```python
from src.ai_models.carbon_calculator import calculate_carbon_from_imagery
from src.database.db_manager import DatabaseManager

db = DatabaseManager()
estimate = calculate_carbon_from_imagery(
    imagery_id=1,
    db_manager=db,
    land_type='cropland'
)
```

### Get Summary
```python
summary = estimate.get_summary()
# Returns: {
#   'total_biomass_tons': 1000.5,
#   'carbon_sequestered_tons': 1650.8,
#   'carbon_credits': 1650.8,
#   'credit_value_usd': 24762.0,
#   'area_hectares': 100.0,
#   'carbon_per_hectare': 16.5
# }
```

### Environmental Impact
```python
for key, value in estimate.environmental_impact.items():
    print(f"{key}: {value}")
# Output:
# cars_off_road: 358.9 cars driven for one year
# trees_planted: 27258 tree seedlings grown for 10 years
# homes_powered: 220.1 homes' energy use for one year
# co2_removed: 1650.82 tons of CO2 removed from atmosphere
```

### Temporal Trend
```python
from src.ai_models.carbon_calculator import calculate_carbon_trend

trend = calculate_carbon_trend(
    tile_id='43REQ',
    db_manager=db,
    land_type='cropland'
)

print(f"Trend: {trend['trend']}")
print(f"Change: {trend['total_change']:.2f} tons CO2")
print(f"Change %: {trend['change_percent']:.1f}%")
```

## 5. Comparison Widget

### In Streamlit Dashboard
```python
import streamlit as st
from src.dashboard.components import render_comparison_widget

# Get imagery records from database
before_imagery = db.get_processed_imagery(1)
after_imagery = db.get_processed_imagery(2)

# Render comparison widget
render_comparison_widget(
    before_imagery=before_imagery,
    after_imagery=after_imagery,
    index_name="NDVI"
)
```

### Multi-Date Slider
```python
from src.dashboard.components import render_multi_date_slider

# Get all imagery records
imagery_records = db.list_processed_imagery(limit=10)

# Render slider
render_multi_date_slider(
    imagery_records=imagery_records,
    index_name="NDVI"
)
```

### Programmatic Comparison
```python
from src.dashboard.components import ComparisonWidget

widget = ComparisonWidget()

# Side-by-side view
fig = widget.create_side_by_side_view(
    before_path="data/processed/ndvi_before.tif",
    after_path="data/processed/ndvi_after.tif",
    before_date="2024-01-01",
    after_date="2024-06-01",
    index_name="NDVI"
)

# Difference map
fig = widget.create_difference_map(
    before_path="data/processed/ndvi_before.tif",
    after_path="data/processed/ndvi_after.tif",
    before_date="2024-01-01",
    after_date="2024-06-01",
    index_name="NDVI"
)

# Statistics
stats = widget.create_statistics_comparison(
    before_path="data/processed/ndvi_before.tif",
    after_path="data/processed/ndvi_after.tif",
    index_name="NDVI"
)
```

## Common Patterns

### Error Handling
```python
try:
    result = detector.detect_changes(before_path, after_path)
    if result:
        # Process result
        pass
    else:
        print("Detection failed")
except Exception as e:
    print(f"Error: {e}")
```

### Batch Processing
```python
# Process multiple imagery pairs
imagery_list = db.list_processed_imagery()

for i in range(len(imagery_list) - 1):
    before = imagery_list[i + 1]
    after = imagery_list[i]
    
    result = compare_imagery_dates(
        before['id'], after['id'], db
    )
    
    if result:
        print(f"{before['acquisition_date']} → {after['acquisition_date']}")
        print(f"  Change: {result.change_percentage:.1f}%")
```

### Combining Features
```python
# Get latest imagery
latest = db.get_latest_imagery()

# Run all analyses
change_result = compare_imagery_dates(1, latest['id'], db)
irrigation_plan = create_irrigation_plan_from_db(latest['id'], db)
yield_estimate = predict_yield_from_imagery(latest['id'], db, 'wheat')
carbon_estimate = calculate_carbon_from_imagery(latest['id'], db)

# Create comprehensive report
print("=== Field Analysis Report ===")
print(f"\nChange Detection:")
print(f"  Change: {change_result.change_percentage:.1f}%")

print(f"\nIrrigation:")
print(f"  High Priority: {irrigation_plan.high_priority_area:.1f}%")

print(f"\nYield:")
print(f"  Predicted: {yield_estimate.predicted_yield:.2f} t/ha")

print(f"\nCarbon:")
print(f"  Sequestered: {carbon_estimate.carbon_sequestered:.2f} tons CO2")
```

## Configuration

### Change Detection Thresholds
```python
detector = ChangeDetector(
    significant_threshold=0.15,  # ±0.15 for significant changes
    moderate_threshold=0.05      # ±0.05 for moderate changes
)
```

### Irrigation Zones
```python
recommender = IrrigationZoneRecommender(
    n_zones=4  # Number of zones (2-6 recommended)
)
```

### Yield Prediction
```python
predictor = YieldPredictor(
    crop_type='wheat'  # 'wheat', 'rice', 'corn', 'soybean', 'generic'
)
```

### Carbon Calculator
```python
calculator = CarbonCalculator(
    land_type='cropland'  # 'cropland', 'grassland', 'forest', 'generic'
)
```

## Performance Tips

1. **Cache Results:** Use `@st.cache_data` in Streamlit for expensive operations
2. **Batch Processing:** Process multiple images in parallel when possible
3. **Lazy Loading:** Load data only when needed
4. **Optimize GeoTIFFs:** Use compression (LZW) for storage
5. **Database Indexing:** Ensure proper indexes on acquisition_date and tile_id

## Troubleshooting

### "NDVI not available"
- Ensure NDVI has been calculated and saved to database
- Check that the GeoTIFF path is correct

### "Need at least 2 imagery records"
- Change detection requires multiple dates
- Process additional imagery or use demo mode

### "NDWI or NDSI not available"
- Irrigation zones require both NDWI and NDSI
- Ensure these indices are calculated during processing

### "Invalid shape"
- Ensure all input arrays have the same dimensions
- Check that GeoTIFF files are properly formatted

## Support

For issues or questions:
1. Check the comprehensive documentation in `USP_FEATURES_IMPLEMENTATION.md`
2. Review test cases in `test_usp_features.py`
3. Run the demo script: `python demo_usp_features.py`
4. Check logs in `logs/` directory

## Version Information

- **Version:** 1.0.0
- **Date:** December 7, 2024
- **Python:** 3.9+
- **Dependencies:** numpy, rasterio, scikit-learn, plotly, streamlit
