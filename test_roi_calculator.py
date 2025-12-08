"""
Test script for ROI Calculator functionality
"""

import sys
import os
sys.path.append('src')

from utils.roi_calculator import (
    ROICalculator, 
    FarmParameters, 
    format_currency, 
    format_percentage, 
    format_quantity
)


def test_default_parameters():
    """Test ROI calculator with default parameters"""
    print("=" * 60)
    print("TEST 1: Default Parameters")
    print("=" * 60)
    
    calculator = ROICalculator()
    metrics = calculator.calculate_full_roi()
    
    print(f"\n💰 Financial Metrics:")
    print(f"  Total Annual Savings: {format_currency(metrics.total_annual_savings)}")
    print(f"  Net Benefit: {format_currency(metrics.net_benefit)}")
    print(f"  ROI: {format_percentage(metrics.roi_pct)}")
    print(f"  Payback Period: {metrics.payback_period_years:.2f} years")
    
    print(f"\n📈 Yield Improvement:")
    print(f"  Improvement: {format_percentage(metrics.yield_improvement_pct)}")
    print(f"  Additional Yield: {format_quantity(metrics.yield_improvement_kg, 'kg')}")
    print(f"  Revenue Increase: {format_currency(metrics.revenue_increase)}")
    
    print(f"\n♻️ Resource Efficiency:")
    print(f"  Water Savings: {format_percentage(metrics.water_savings_pct)} ({format_quantity(metrics.water_savings_m3, 'm³')})")
    print(f"  Water Cost Savings: {format_currency(metrics.water_cost_savings)}")
    print(f"  Fertilizer Reduction: {format_percentage(metrics.fertilizer_reduction_pct)}")
    print(f"  Fertilizer Savings: {format_currency(metrics.fertilizer_cost_savings)}")
    print(f"  Pesticide Reduction: {format_percentage(metrics.pesticide_reduction_pct)}")
    print(f"  Pesticide Savings: {format_currency(metrics.pesticide_cost_savings)}")
    
    print(f"\n🌍 Environmental Impact:")
    print(f"  Carbon Sequestered: {format_quantity(metrics.carbon_sequestration_tons, 'tons CO₂')}")
    print(f"  Carbon Value: {format_currency(metrics.carbon_value)}")
    
    # Verify calculations are reasonable
    assert metrics.total_annual_savings > 0, "Total savings should be positive"
    assert metrics.roi_pct > 0, "ROI should be positive"
    assert metrics.payback_period_years > 0, "Payback period should be positive"
    assert 0 < metrics.yield_improvement_pct < 50, "Yield improvement should be reasonable"
    assert 0 < metrics.water_savings_pct < 100, "Water savings should be percentage"
    
    print("\n✅ All assertions passed!")


def test_custom_parameters():
    """Test ROI calculator with custom farm parameters"""
    print("\n" + "=" * 60)
    print("TEST 2: Custom Parameters (Large Farm)")
    print("=" * 60)
    
    # Large farm scenario
    params = FarmParameters(
        farm_size_ha=500.0,
        crop_type="corn",
        baseline_yield_kg_ha=8000.0,
        crop_price_per_kg=0.20,
        water_cost_per_m3=0.75,
        fertilizer_cost_per_ha=200.0,
        pesticide_cost_per_ha=100.0,
        agriflux_annual_cost=10000.0
    )
    
    calculator = ROICalculator(params)
    metrics = calculator.calculate_full_roi(
        health_index=0.75,
        alert_response_rate=0.95,
        irrigation_zones_used=True,
        precision_application=True,
        mean_ndvi=0.75
    )
    
    print(f"\n🚜 Farm: {params.farm_size_ha} ha {params.crop_type} farm")
    print(f"💰 Total Annual Savings: {format_currency(metrics.total_annual_savings)}")
    print(f"📊 Net Benefit: {format_currency(metrics.net_benefit)}")
    print(f"📈 ROI: {format_percentage(metrics.roi_pct)}")
    print(f"⏱️ Payback Period: {metrics.payback_period_years:.2f} years")
    
    # Verify larger farm has larger absolute savings
    assert metrics.total_annual_savings > 50000, "Large farm should have significant savings"
    assert metrics.net_benefit > 0, "Net benefit should be positive"
    
    print("\n✅ Custom parameters test passed!")


def test_cost_savings_component():
    """Test cost savings calculation separately"""
    print("\n" + "=" * 60)
    print("TEST 3: Cost Savings Component")
    print("=" * 60)
    
    calculator = ROICalculator()
    
    # Test with different health indices
    health_scenarios = [
        (0.5, "Poor Health"),
        (0.7, "Good Health"),
        (0.85, "Excellent Health")
    ]
    
    for health_index, label in health_scenarios:
        savings = calculator.calculate_cost_savings(
            health_index=health_index,
            alert_response_rate=0.9
        )
        
        print(f"\n{label} (NDVI={health_index}):")
        print(f"  Yield Improvement: {savings['yield_improvement_pct']:.2f}%")
        print(f"  Additional Yield: {format_quantity(savings['yield_improvement_kg'], 'kg')}")
        print(f"  Revenue Increase: {format_currency(savings['revenue_increase'])}")
    
    print("\n✅ Cost savings component test passed!")


def test_resource_efficiency_component():
    """Test resource efficiency calculation separately"""
    print("\n" + "=" * 60)
    print("TEST 4: Resource Efficiency Component")
    print("=" * 60)
    
    calculator = ROICalculator()
    
    # Test with precision features enabled
    efficiency_enabled = calculator.calculate_resource_efficiency(
        irrigation_zones_used=True,
        precision_application=True
    )
    
    print("\nWith Precision Features:")
    print(f"  Water Savings: {format_percentage(efficiency_enabled['water_savings_pct'])}")
    print(f"  Water Volume Saved: {format_quantity(efficiency_enabled['water_savings_m3'], 'm³')}")
    print(f"  Water Cost Savings: {format_currency(efficiency_enabled['water_cost_savings'])}")
    print(f"  Fertilizer Reduction: {format_percentage(efficiency_enabled['fertilizer_reduction_pct'])}")
    print(f"  Fertilizer Savings: {format_currency(efficiency_enabled['fertilizer_cost_savings'])}")
    print(f"  Pesticide Reduction: {format_percentage(efficiency_enabled['pesticide_reduction_pct'])}")
    print(f"  Pesticide Savings: {format_currency(efficiency_enabled['pesticide_cost_savings'])}")
    
    # Test with precision features disabled
    efficiency_disabled = calculator.calculate_resource_efficiency(
        irrigation_zones_used=False,
        precision_application=False
    )
    
    print("\nWithout Precision Features:")
    print(f"  Water Savings: {format_percentage(efficiency_disabled['water_savings_pct'])}")
    print(f"  Fertilizer Reduction: {format_percentage(efficiency_disabled['fertilizer_reduction_pct'])}")
    print(f"  Pesticide Reduction: {format_percentage(efficiency_disabled['pesticide_reduction_pct'])}")
    
    # Verify precision features make a difference
    assert efficiency_enabled['water_savings_pct'] > efficiency_disabled['water_savings_pct']
    assert efficiency_enabled['fertilizer_reduction_pct'] > efficiency_disabled['fertilizer_reduction_pct']
    
    print("\n✅ Resource efficiency component test passed!")


def test_carbon_impact_component():
    """Test carbon impact calculation separately"""
    print("\n" + "=" * 60)
    print("TEST 5: Carbon Impact Component")
    print("=" * 60)
    
    calculator = ROICalculator()
    
    # Test with different NDVI values
    ndvi_scenarios = [
        (0.5, "Low Vegetation"),
        (0.7, "Moderate Vegetation"),
        (0.85, "High Vegetation")
    ]
    
    for ndvi, label in ndvi_scenarios:
        carbon = calculator.calculate_carbon_impact(
            mean_ndvi=ndvi,
            biomass_increase_pct=0.08
        )
        
        print(f"\n{label} (NDVI={ndvi}):")
        print(f"  Biomass: {carbon['biomass_tons_ha']:.2f} tons/ha")
        print(f"  Total Biomass: {format_quantity(carbon['total_biomass_tons'], 'tons')}")
        print(f"  Carbon Sequestered: {format_quantity(carbon['carbon_sequestered_tons'], 'tons')}")
        print(f"  CO₂ Equivalent: {format_quantity(carbon['co2_equivalent_tons'], 'tons')}")
        print(f"  Carbon Value: {format_currency(carbon['carbon_value'])}")
    
    print("\n✅ Carbon impact component test passed!")


def test_assumptions():
    """Test assumptions display"""
    print("\n" + "=" * 60)
    print("TEST 6: Assumptions")
    print("=" * 60)
    
    calculator = ROICalculator()
    assumptions = calculator.get_assumptions()
    
    print("\nCalculation Assumptions:")
    for category, description in assumptions.items():
        print(f"\n{category}:")
        print(f"  {description}")
    
    assert len(assumptions) > 0, "Should have assumptions"
    
    print("\n✅ Assumptions test passed!")


def test_formatting_functions():
    """Test formatting helper functions"""
    print("\n" + "=" * 60)
    print("TEST 7: Formatting Functions")
    print("=" * 60)
    
    # Test currency formatting
    assert format_currency(1234.56) == "$1,234.56 USD"
    assert format_currency(1000000) == "$1,000,000.00 USD"
    print("✓ Currency formatting works")
    
    # Test percentage formatting
    assert format_percentage(25.5) == "25.5%"
    assert format_percentage(100.0) == "100.0%"
    print("✓ Percentage formatting works")
    
    # Test quantity formatting
    assert format_quantity(1234.5, "kg") == "1,234 kg"
    assert format_quantity(1000000, "m³") == "1,000,000 m³"
    print("✓ Quantity formatting works")
    
    print("\n✅ All formatting tests passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ROI CALCULATOR TEST SUITE")
    print("=" * 60)
    
    try:
        test_default_parameters()
        test_custom_parameters()
        test_cost_savings_component()
        test_resource_efficiency_component()
        test_carbon_impact_component()
        test_assumptions()
        test_formatting_functions()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
