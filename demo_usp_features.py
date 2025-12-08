"""
Demo script showcasing USP features integration.

This script demonstrates how to use all 5 USP features:
1. Multi-temporal change detection
2. Precision irrigation zones
3. Yield prediction
4. Carbon sequestration
5. Before/after comparison

Run this after populating the database with processed imagery.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.db_manager import DatabaseManager
from src.ai_models.change_detection import compare_imagery_dates
from src.ai_models.irrigation_zones import create_irrigation_plan_from_db
from src.ai_models.yield_prediction import predict_yield_from_imagery
from src.ai_models.carbon_calculator import calculate_carbon_from_imagery


def demo_change_detection(db_manager):
    """Demonstrate multi-temporal change detection."""
    print("\n" + "="*60)
    print("1. MULTI-TEMPORAL CHANGE DETECTION")
    print("="*60)
    
    # Get available imagery
    imagery_list = db_manager.list_processed_imagery(limit=10)
    
    if len(imagery_list) < 2:
        print("❌ Need at least 2 imagery records for change detection")
        return
    
    # Compare first two dates
    before = imagery_list[1]  # Older
    after = imagery_list[0]   # Newer
    
    print(f"\n📅 Comparing:")
    print(f"  Before: {before['acquisition_date']} (ID: {before['id']})")
    print(f"  After:  {after['acquisition_date']} (ID: {after['id']})")
    
    # Perform change detection
    result = compare_imagery_dates(
        before_imagery_id=before['id'],
        after_imagery_id=after['id'],
        db_manager=db_manager,
        index_name='NDVI'
    )
    
    if result:
        print(f"\n📊 Change Detection Results:")
        print(f"  Total Change: {result.change_percentage:.1f}%")
        print(f"  Improvement Area: {result.improvement_area:.1f}%")
        print(f"  Degradation Area: {result.degradation_area:.1f}%")
        print(f"  Stable Area: {result.stable_area:.1f}%")
        print(f"\n  Mean Change: {result.statistics['mean_change']:.3f}")
        print(f"  Max Improvement: {result.statistics['max_improvement']:.3f}")
        print(f"  Max Degradation: {result.statistics['max_degradation']:.3f}")
        
        # Interpretation
        if result.improvement_area > result.degradation_area:
            print("\n✅ Overall trend: IMPROVEMENT")
        elif result.degradation_area > result.improvement_area:
            print("\n⚠️  Overall trend: DEGRADATION")
        else:
            print("\n➡️  Overall trend: STABLE")
    else:
        print("❌ Change detection failed")


def demo_irrigation_zones(db_manager):
    """Demonstrate precision irrigation zone recommender."""
    print("\n" + "="*60)
    print("2. PRECISION IRRIGATION ZONE RECOMMENDER")
    print("="*60)
    
    # Get latest imagery
    latest = db_manager.get_latest_imagery()
    
    if not latest:
        print("❌ No imagery available")
        return
    
    print(f"\n📅 Using imagery from: {latest['acquisition_date']}")
    
    # Create irrigation plan
    plan = create_irrigation_plan_from_db(
        imagery_id=latest['id'],
        db_manager=db_manager,
        n_zones=4
    )
    
    if plan:
        print(f"\n💧 Irrigation Plan Summary:")
        print(f"  Total Area: {plan.total_area:,} pixels")
        print(f"  High Priority Area: {plan.high_priority_area:.1f}%")
        print(f"  Estimated Water Savings: {plan.water_savings_estimate:.0f}%")
        print(f"\n  {plan.summary}")
        
        print(f"\n📋 Zone Details:")
        for zone in plan.zones:
            print(f"\n  Zone {zone.zone_id + 1} ({zone.water_stress_level.upper()}):")
            print(f"    Priority: {zone.priority} (1=highest)")
            print(f"    Area: {zone.area_percentage:.1f}%")
            print(f"    Mean NDWI: {zone.mean_ndwi:.3f}")
            print(f"    Frequency: {zone.irrigation_frequency}")
            print(f"    Amount: {zone.water_amount}")
            print(f"    💡 {zone.recommendation}")
    else:
        print("❌ Irrigation plan creation failed")


def demo_yield_prediction(db_manager):
    """Demonstrate yield prediction."""
    print("\n" + "="*60)
    print("3. YIELD PREDICTION")
    print("="*60)
    
    # Get latest imagery
    latest = db_manager.get_latest_imagery()
    
    if not latest:
        print("❌ No imagery available")
        return
    
    print(f"\n📅 Using imagery from: {latest['acquisition_date']}")
    
    # Predict yield for different crop types
    crop_types = ['wheat', 'rice', 'corn']
    
    for crop_type in crop_types:
        print(f"\n🌾 {crop_type.upper()} Yield Prediction:")
        
        estimate = predict_yield_from_imagery(
            imagery_id=latest['id'],
            db_manager=db_manager,
            crop_type=crop_type
        )
        
        if estimate:
            print(f"  Predicted Yield: {estimate.predicted_yield:.2f} tons/hectare")
            print(f"  Confidence Interval: {estimate.confidence_interval[0]:.2f} - {estimate.confidence_interval[1]:.2f} t/ha")
            print(f"  Confidence Level: {estimate.confidence_level:.1f}%")
            print(f"  Category: {estimate.yield_category.upper()}")
            print(f"  Growth Stage: {estimate.growth_stage}")
            print(f"  NDVI Trend: {estimate.ndvi_trend}")
            
            print(f"\n  💡 Recommendations:")
            for rec in estimate.recommendations:
                print(f"    • {rec}")
        else:
            print(f"  ❌ Prediction failed for {crop_type}")


def demo_carbon_calculator(db_manager):
    """Demonstrate carbon sequestration calculator."""
    print("\n" + "="*60)
    print("4. CARBON SEQUESTRATION CALCULATOR")
    print("="*60)
    
    # Get latest imagery
    latest = db_manager.get_latest_imagery()
    
    if not latest:
        print("❌ No imagery available")
        return
    
    print(f"\n📅 Using imagery from: {latest['acquisition_date']}")
    
    # Calculate carbon for different land types
    land_types = ['cropland', 'grassland']
    
    for land_type in land_types:
        print(f"\n🌱 {land_type.upper()} Carbon Estimate:")
        
        estimate = calculate_carbon_from_imagery(
            imagery_id=latest['id'],
            db_manager=db_manager,
            land_type=land_type
        )
        
        if estimate:
            print(f"  Total Biomass: {estimate.total_biomass:.2f} tons")
            print(f"  Above-ground: {estimate.above_ground_biomass:.2f} tons")
            print(f"  Below-ground: {estimate.below_ground_biomass:.2f} tons")
            print(f"\n  Carbon Sequestered: {estimate.carbon_sequestered:.2f} tons CO2")
            print(f"  Carbon Credits: {estimate.carbon_credits:.2f}")
            print(f"  Estimated Value: ${estimate.credit_value_usd:.2f} USD")
            print(f"\n  Area: {estimate.area_hectares:.2f} hectares")
            print(f"  Carbon per Hectare: {estimate.carbon_per_hectare:.2f} tons CO2/ha")
            
            print(f"\n  🌍 Environmental Impact Equivalents:")
            for key, value in estimate.environmental_impact.items():
                print(f"    • {value}")
        else:
            print(f"  ❌ Calculation failed for {land_type}")


def demo_comparison_info(db_manager):
    """Show information about comparison widget."""
    print("\n" + "="*60)
    print("5. BEFORE/AFTER COMPARISON WIDGET")
    print("="*60)
    
    imagery_list = db_manager.list_processed_imagery(limit=10)
    
    print(f"\n📊 Available for comparison: {len(imagery_list)} imagery records")
    
    if len(imagery_list) >= 2:
        print("\n📅 Available dates:")
        for img in imagery_list[:5]:
            print(f"  • {img['acquisition_date']} (ID: {img['id']})")
        
        print("\n💡 Comparison widget features:")
        print("  • Side-by-side view")
        print("  • Difference map (change visualization)")
        print("  • Distribution histograms")
        print("  • Statistical comparison tables")
        print("  • Multi-date slider")
        
        print("\n🎯 To use in Streamlit dashboard:")
        print("  from src.dashboard.components import render_comparison_widget")
        print("  render_comparison_widget(before_imagery, after_imagery, 'NDVI')")
    else:
        print("\n❌ Need at least 2 imagery records for comparison")


def main():
    """Run all USP feature demos."""
    print("\n" + "="*60)
    print("AGRIFLUX USP FEATURES DEMONSTRATION")
    print("="*60)
    
    # Initialize database
    db_path = "data/agriflux.db"
    db_manager = DatabaseManager(db_path)
    
    # Check if database has data
    stats = db_manager.get_database_stats()
    
    print(f"\n📊 Database Statistics:")
    print(f"  Imagery Records: {stats['imagery_count']}")
    print(f"  Total Alerts: {stats['total_alerts']}")
    print(f"  Active Alerts: {stats['active_alerts']}")
    print(f"  Predictions: {stats['predictions_count']}")
    
    if stats['imagery_count'] == 0:
        print("\n❌ No imagery data in database!")
        print("💡 Run 'python scripts/populate_database.py' first")
        return
    
    # Run demos
    try:
        demo_change_detection(db_manager)
        demo_irrigation_zones(db_manager)
        demo_yield_prediction(db_manager)
        demo_carbon_calculator(db_manager)
        demo_comparison_info(db_manager)
        
        print("\n" + "="*60)
        print("✅ ALL USP FEATURES DEMONSTRATED SUCCESSFULLY")
        print("="*60)
        print("\n💡 Next steps:")
        print("  1. Integrate these features into dashboard pages")
        print("  2. Add visualizations using Plotly/Folium")
        print("  3. Create user-friendly UI components")
        print("  4. Test with real user workflows")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
