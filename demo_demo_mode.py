"""
Demo script showing how to use the demo mode system

This script demonstrates the key features of the demo mode system
and how to access demo data programmatically.
"""

import sys
sys.path.append('src')

from utils.demo_data_manager import get_demo_manager
import numpy as np


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_basic_usage():
    """Demonstrate basic demo mode usage."""
    print_section("1. Basic Demo Mode Usage")
    
    # Get demo manager instance (singleton)
    manager = get_demo_manager()
    print("✓ Demo manager instance created")
    
    # Check if demo data is available
    if not manager.is_demo_data_available():
        print("❌ Demo data not available. Run 'python scripts/generate_demo_data.py' first.")
        return False
    
    print("✓ Demo data files found")
    
    # Load demo data
    if manager.load_demo_data():
        print("✓ Demo data loaded successfully")
    else:
        print("❌ Failed to load demo data")
        return False
    
    return True


def demo_scenarios():
    """Demonstrate scenario access."""
    print_section("2. Accessing Field Scenarios")
    
    manager = get_demo_manager()
    
    # Get available scenarios
    scenarios = manager.get_scenario_names()
    print(f"\nAvailable scenarios: {len(scenarios)}")
    
    for scenario_name in scenarios:
        print(f"\n📊 {scenario_name.upper().replace('_', ' ')}")
        
        # Get scenario description
        description = manager.get_scenario_description(scenario_name)
        print(f"   {description}")
        
        # Get scenario data
        scenario = manager.get_scenario(scenario_name)
        if scenario:
            ndvi = scenario['ndvi']
            print(f"   - NDVI range: [{np.min(ndvi):.3f}, {np.max(ndvi):.3f}]")
            print(f"   - Mean NDVI: {np.mean(ndvi):.3f}")
            print(f"   - Health status: {scenario['health_status']}")


def demo_time_series():
    """Demonstrate time series access."""
    print_section("3. Time Series Analysis")
    
    manager = get_demo_manager()
    
    # Pick a scenario
    scenario_name = 'stressed_field'
    print(f"\nAnalyzing time series for: {scenario_name}")
    
    time_series = manager.get_time_series(scenario_name)
    
    if time_series:
        print(f"\nTime points: {len(time_series)}")
        print("\nTemporal evolution:")
        print(f"{'Date':<12} {'Mean NDVI':<12} {'Cloud %':<10}")
        print("-" * 40)
        
        for point in time_series:
            date = point['acquisition_date']
            ndvi = point['mean_ndvi']
            cloud = point['cloud_coverage']
            print(f"{date:<12} {ndvi:<12.3f} {cloud:<10.1f}")
        
        # Show trend
        first_ndvi = time_series[0]['mean_ndvi']
        last_ndvi = time_series[-1]['mean_ndvi']
        change = last_ndvi - first_ndvi
        trend = "↗ improving" if change > 0 else "↘ declining"
        print(f"\nTrend: {trend} ({change:+.3f})")


def demo_alerts():
    """Demonstrate alert access."""
    print_section("4. Alert Management")
    
    manager = get_demo_manager()
    
    # Get all alerts
    all_alerts = manager.get_alerts()
    print(f"\nTotal alerts: {len(all_alerts)}")
    
    # Get active alerts
    active_alerts = manager.get_active_alerts()
    print(f"Active (unacknowledged) alerts: {len(active_alerts)}")
    
    # Show alerts by severity
    print("\nAlerts by severity:")
    for severity in ['critical', 'high', 'medium', 'low']:
        alerts = manager.get_alerts(severity=severity)
        print(f"  {severity.upper():<10}: {len(alerts)} alerts")
    
    # Show a sample critical alert
    critical_alerts = manager.get_alerts(severity='critical')
    if critical_alerts:
        print("\n🚨 Sample Critical Alert:")
        alert = critical_alerts[0]
        print(f"   Type: {alert['alert_type']}")
        print(f"   Message: {alert['message']}")
        print(f"   Recommendation: {alert['recommendation']}")
        print(f"   Affected area: {alert['affected_area_percentage']:.1f}%")


def demo_predictions():
    """Demonstrate prediction access."""
    print_section("5. AI Predictions")
    
    manager = get_demo_manager()
    
    for scenario_name in ['healthy_field', 'stressed_field', 'mixed_field']:
        predictions = manager.get_predictions(scenario_name)
        
        if predictions:
            print(f"\n🤖 {scenario_name.upper().replace('_', ' ')}")
            print(f"   Model: {predictions['model_version']}")
            print(f"   Type: {predictions['prediction_type']}")
            
            # Get class distribution
            if 'metadata' in predictions and 'class_distribution' in predictions['metadata']:
                dist = predictions['metadata']['class_distribution']
                total = sum(dist.values())
                
                print(f"   Class distribution:")
                for class_name, count in dist.items():
                    percentage = (count / total) * 100
                    print(f"     - {class_name.capitalize():<10}: {percentage:>5.1f}%")


def demo_dashboard_formatting():
    """Demonstrate dashboard data formatting."""
    print_section("6. Dashboard Data Formatting")
    
    manager = get_demo_manager()
    
    scenario_name = 'mixed_field'
    print(f"\nFormatting data for dashboard: {scenario_name}")
    
    dashboard_data = manager.format_for_dashboard(scenario_name)
    
    if dashboard_data:
        print("\n✓ Dashboard data formatted successfully")
        print(f"\nData sections:")
        print(f"  - Imagery: {dashboard_data['imagery']['acquisition_date']}")
        print(f"  - Alerts: {len(dashboard_data['alerts'])} active")
        print(f"  - Predictions: {dashboard_data['predictions']['model_version']}")
        print(f"  - Scenario: {dashboard_data['scenario_info']['name']}")
        
        # Show available indices
        imagery = dashboard_data['imagery']
        indices = ['ndvi', 'savi', 'evi', 'ndwi', 'ndsi']
        print(f"\n  Available indices:")
        for idx in indices:
            if idx in imagery:
                mean_val = imagery.get(f'mean_{idx}', np.mean(imagery[idx]))
                print(f"    - {idx.upper()}: {mean_val:.3f}")


def demo_summary_stats():
    """Demonstrate summary statistics."""
    print_section("7. Summary Statistics")
    
    manager = get_demo_manager()
    
    for scenario_name in manager.get_scenario_names():
        stats = manager.get_summary_stats(scenario_name)
        
        if stats:
            print(f"\n📈 {scenario_name.upper().replace('_', ' ')}")
            print(f"   Health status: {stats['health_status']}")
            print(f"   Current NDVI: {stats['current_ndvi']['mean']:.3f} "
                  f"(±{stats['current_ndvi']['std']:.3f})")
            print(f"   Time points: {stats['time_points']}")
            print(f"   Total alerts: {stats['total_alerts']}")
            print(f"   Active alerts: {stats['active_alerts']}")
            
            if stats['alert_breakdown']:
                print(f"   Alert breakdown: {stats['alert_breakdown']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AgriFlux Demo Mode - Feature Demonstration")
    print("=" * 60)
    
    # Basic usage
    if not demo_basic_usage():
        print("\n❌ Demo data not available. Exiting.")
        return 1
    
    # Demonstrate features
    demo_scenarios()
    demo_time_series()
    demo_alerts()
    demo_predictions()
    demo_dashboard_formatting()
    demo_summary_stats()
    
    # Final summary
    print_section("Summary")
    print("\n✅ Demo mode system is fully functional!")
    print("\nKey features demonstrated:")
    print("  1. ✓ Load and access 3 field scenarios")
    print("  2. ✓ Access time series data (5 points per scenario)")
    print("  3. ✓ Filter and manage alerts")
    print("  4. ✓ Access AI predictions")
    print("  5. ✓ Format data for dashboard consumption")
    print("  6. ✓ Generate summary statistics")
    
    print("\nTo use in dashboard:")
    print("  1. Run: streamlit run src/dashboard/main.py")
    print("  2. Enable 'Demo Mode' in sidebar")
    print("  3. Select a scenario to explore")
    
    print("\n" + "=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
