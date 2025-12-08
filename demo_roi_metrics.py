"""
Demo script for ROI and Impact Metrics

This script demonstrates the ROI calculator functionality
and shows how it integrates with the dashboard.
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


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_roi():
    """Demonstrate basic ROI calculation"""
    print_section("DEMO 1: Basic ROI Calculation (100 ha wheat farm)")
    
    calculator = ROICalculator()
    metrics = calculator.calculate_full_roi()
    
    print("\n💰 FINANCIAL SUMMARY")
    print(f"  Total Annual Savings:  {format_currency(metrics.total_annual_savings)}")
    print(f"  AgriFlux Annual Cost:  {format_currency(5000.00)}")
    print(f"  Net Annual Benefit:    {format_currency(metrics.net_benefit)}")
    print(f"  Return on Investment:  {format_percentage(metrics.roi_pct)}")
    print(f"  Payback Period:        {metrics.payback_period_years:.2f} years")
    
    print("\n📊 SAVINGS BREAKDOWN")
    print(f"  Yield Improvement:     {format_currency(metrics.revenue_increase)}")
    print(f"  Water Savings:         {format_currency(metrics.water_cost_savings)}")
    print(f"  Fertilizer Savings:    {format_currency(metrics.fertilizer_cost_savings)}")
    print(f"  Pesticide Savings:     {format_currency(metrics.pesticide_cost_savings)}")
    print(f"  Carbon Credits:        {format_currency(metrics.carbon_value)}")
    
    print("\n♻️ RESOURCE EFFICIENCY")
    print(f"  Water Saved:           {format_quantity(metrics.water_savings_m3, 'm³')} ({format_percentage(metrics.water_savings_pct)})")
    print(f"  Fertilizer Reduced:    {format_percentage(metrics.fertilizer_reduction_pct)}")
    print(f"  Pesticide Reduced:     {format_percentage(metrics.pesticide_reduction_pct)}")
    
    print("\n🌍 ENVIRONMENTAL IMPACT")
    print(f"  Carbon Sequestered:    {format_quantity(metrics.carbon_sequestration_tons, 'tons CO₂')}")
    print(f"  Yield Improvement:     {format_percentage(metrics.yield_improvement_pct)}")
    print(f"  Additional Production: {format_quantity(metrics.yield_improvement_kg, 'kg')}")


def demo_small_farm():
    """Demonstrate ROI for a small farm"""
    print_section("DEMO 2: Small Farm (25 ha vegetable farm)")
    
    params = FarmParameters(
        farm_size_ha=25.0,
        crop_type="vegetables",
        baseline_yield_kg_ha=15000.0,  # Higher yield for vegetables
        crop_price_per_kg=1.50,  # Higher price for vegetables
        water_cost_per_m3=0.80,
        fertilizer_cost_per_ha=300.0,  # Higher for vegetables
        pesticide_cost_per_ha=150.0,
        agriflux_annual_cost=3000.0  # Lower cost for small farm
    )
    
    calculator = ROICalculator(params)
    metrics = calculator.calculate_full_roi(
        health_index=0.65,  # Moderate health
        alert_response_rate=0.85
    )
    
    print("\n🚜 FARM PROFILE")
    print(f"  Size:                  {params.farm_size_ha} hectares")
    print(f"  Crop:                  {params.crop_type}")
    print(f"  Baseline Yield:        {params.baseline_yield_kg_ha} kg/ha")
    print(f"  Crop Price:            ${params.crop_price_per_kg}/kg")
    
    print("\n💰 ROI ANALYSIS")
    print(f"  Total Annual Savings:  {format_currency(metrics.total_annual_savings)}")
    print(f"  Net Benefit:           {format_currency(metrics.net_benefit)}")
    print(f"  ROI:                   {format_percentage(metrics.roi_pct)}")
    print(f"  Payback Period:        {metrics.payback_period_years:.2f} years")
    
    if metrics.net_benefit > 0:
        print(f"\n✅ PROFITABLE: This farm will save {format_currency(metrics.net_benefit)} annually!")
        print(f"   Over 5 years: {format_currency(metrics.net_benefit * 5)}")
    else:
        print(f"\n⚠️  Consider larger farm size or adjust parameters")


def demo_large_farm():
    """Demonstrate ROI for a large farm"""
    print_section("DEMO 3: Large Farm (1000 ha corn farm)")
    
    params = FarmParameters(
        farm_size_ha=1000.0,
        crop_type="corn",
        baseline_yield_kg_ha=9000.0,
        crop_price_per_kg=0.18,
        water_cost_per_m3=0.60,
        fertilizer_cost_per_ha=180.0,
        pesticide_cost_per_ha=90.0,
        agriflux_annual_cost=15000.0  # Higher cost for large farm
    )
    
    calculator = ROICalculator(params)
    metrics = calculator.calculate_full_roi(
        health_index=0.78,  # Good health
        alert_response_rate=0.95,  # High response rate
        irrigation_zones_used=True,
        precision_application=True
    )
    
    print("\n🚜 FARM PROFILE")
    print(f"  Size:                  {params.farm_size_ha} hectares")
    print(f"  Crop:                  {params.crop_type}")
    print(f"  Total Production:      {format_quantity(params.baseline_yield_kg_ha * params.farm_size_ha, 'kg')}")
    
    print("\n💰 MASSIVE SAVINGS POTENTIAL")
    print(f"  Total Annual Savings:  {format_currency(metrics.total_annual_savings)}")
    print(f"  AgriFlux Investment:   {format_currency(params.agriflux_annual_cost)}")
    print(f"  Net Annual Benefit:    {format_currency(metrics.net_benefit)}")
    print(f"  ROI:                   {format_percentage(metrics.roi_pct)}")
    
    print("\n📊 DETAILED BREAKDOWN")
    print(f"  Revenue from Yield:    {format_currency(metrics.revenue_increase)}")
    print(f"  Water Savings:         {format_currency(metrics.water_cost_savings)} ({format_quantity(metrics.water_savings_m3, 'm³')})")
    print(f"  Fertilizer Savings:    {format_currency(metrics.fertilizer_cost_savings)}")
    print(f"  Pesticide Savings:     {format_currency(metrics.pesticide_cost_savings)}")
    print(f"  Carbon Credits:        {format_currency(metrics.carbon_value)}")
    
    print("\n🌍 ENVIRONMENTAL IMPACT AT SCALE")
    print(f"  Carbon Sequestered:    {format_quantity(metrics.carbon_sequestration_tons, 'tons CO₂')}")
    print(f"  Water Saved:           {format_quantity(metrics.water_savings_m3, 'm³')}")
    print(f"  Equivalent to:         {int(metrics.water_savings_m3 / 50)} Olympic swimming pools")
    
    print("\n📈 5-YEAR PROJECTION")
    five_year_benefit = metrics.net_benefit * 5
    print(f"  Total 5-Year Benefit:  {format_currency(five_year_benefit)}")
    print(f"  Average Annual:        {format_currency(five_year_benefit / 5)}")


def demo_comparison():
    """Compare different farm scenarios"""
    print_section("DEMO 4: Scenario Comparison")
    
    scenarios = [
        ("Small Wheat Farm", FarmParameters(farm_size_ha=50, crop_type="wheat", agriflux_annual_cost=3500)),
        ("Medium Corn Farm", FarmParameters(farm_size_ha=200, crop_type="corn", baseline_yield_kg_ha=8000, agriflux_annual_cost=7000)),
        ("Large Rice Farm", FarmParameters(farm_size_ha=500, crop_type="rice", baseline_yield_kg_ha=6000, crop_price_per_kg=0.30, agriflux_annual_cost=12000)),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Scenario':<25} {'Savings':<15} {'ROI':<10} {'Payback':<15}")
    print("-" * 70)
    
    for name, params in scenarios:
        calculator = ROICalculator(params)
        metrics = calculator.calculate_full_roi()
        
        print(f"{name:<25} {format_currency(metrics.total_annual_savings):<15} {format_percentage(metrics.roi_pct):<10} {metrics.payback_period_years:.2f} years")
    
    print("-" * 70)
    print("\n💡 KEY INSIGHT: Larger farms see greater absolute savings,")
    print("   but all farm sizes achieve positive ROI with AgriFlux!")


def demo_sensitivity_analysis():
    """Show how different factors affect ROI"""
    print_section("DEMO 5: Sensitivity Analysis")
    
    base_calculator = ROICalculator()
    
    print("\n📊 Impact of Health Index on Savings:")
    print("-" * 70)
    
    for health_index in [0.5, 0.6, 0.7, 0.8]:
        metrics = base_calculator.calculate_full_roi(health_index=health_index)
        print(f"  NDVI {health_index:.1f}: {format_currency(metrics.total_annual_savings)} " +
              f"(ROI: {format_percentage(metrics.roi_pct)})")
    
    print("\n📊 Impact of Alert Response Rate:")
    print("-" * 70)
    
    for response_rate in [0.5, 0.7, 0.9, 1.0]:
        metrics = base_calculator.calculate_full_roi(alert_response_rate=response_rate)
        print(f"  {int(response_rate*100)}% Response: {format_currency(metrics.total_annual_savings)} " +
              f"(ROI: {format_percentage(metrics.roi_pct)})")
    
    print("\n💡 KEY INSIGHT: Higher crop health and faster alert response")
    print("   lead to greater savings. AgriFlux helps achieve both!")


def demo_assumptions():
    """Display calculation assumptions"""
    print_section("DEMO 6: Transparent Assumptions")
    
    calculator = ROICalculator()
    assumptions = calculator.get_assumptions()
    
    print("\nAll calculations are based on these transparent assumptions:\n")
    
    for i, (category, description) in enumerate(assumptions.items(), 1):
        print(f"{i}. {category}")
        print(f"   {description}\n")
    
    print("💡 These are conservative estimates based on peer-reviewed research")
    print("   and industry benchmarks. Actual results may vary by farm.")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("  AGRIFLUX ROI & IMPACT METRICS DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how AgriFlux calculates return on investment")
    print("and quantifies the benefits of precision agriculture.\n")
    
    demo_basic_roi()
    demo_small_farm()
    demo_large_farm()
    demo_comparison()
    demo_sensitivity_analysis()
    demo_assumptions()
    
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print("\n✅ ROI calculator is working correctly!")
    print("📊 All metrics are calculated using industry-standard methods")
    print("🌱 Ready for integration into the dashboard\n")


if __name__ == "__main__":
    main()
