"""
Test script for demo mode functionality

Verifies that demo data can be loaded and accessed correctly.
"""

import sys
import os

# Add src to path
sys.path.append('src')

from utils.demo_data_manager import DemoDataManager, get_demo_manager
import numpy as np


def test_demo_data_availability():
    """Test if demo data files exist."""
    print("=" * 60)
    print("Test 1: Demo Data Availability")
    print("=" * 60)
    
    manager = DemoDataManager()
    available = manager.is_demo_data_available()
    
    print(f"Demo data available: {available}")
    
    if not available:
        print("❌ Demo data not found. Run 'python scripts/generate_demo_data.py' first.")
        return False
    
    print("✅ Demo data files found")
    return True


def test_demo_data_loading():
    """Test loading demo data."""
    print("\n" + "=" * 60)
    print("Test 2: Demo Data Loading")
    print("=" * 60)
    
    manager = DemoDataManager()
    success = manager.load_demo_data()
    
    if not success:
        print("❌ Failed to load demo data")
        return False
    
    print("✅ Demo data loaded successfully")
    return True


def test_scenario_access():
    """Test accessing scenario data."""
    print("\n" + "=" * 60)
    print("Test 3: Scenario Access")
    print("=" * 60)
    
    manager = DemoDataManager()
    manager.load_demo_data()
    
    scenarios = manager.get_scenario_names()
    print(f"Available scenarios: {scenarios}")
    
    if len(scenarios) != 3:
        print(f"❌ Expected 3 scenarios, got {len(scenarios)}")
        return False
    
    expected_scenarios = ['healthy_field', 'stressed_field', 'mixed_field']
    for scenario in expected_scenarios:
        if scenario not in scenarios:
            print(f"❌ Missing scenario: {scenario}")
            return False
    
    print("✅ All expected scenarios found")
    
    # Test accessing each scenario
    for scenario_name in scenarios:
        scenario = manager.get_scenario(scenario_name)
        if scenario is None:
            print(f"❌ Failed to load scenario: {scenario_name}")
            return False
        
        # Check required keys
        required_keys = ['ndvi', 'savi', 'evi', 'ndwi', 'ndsi', 'description', 'health_status']
        for key in required_keys:
            if key not in scenario:
                print(f"❌ Missing key '{key}' in scenario {scenario_name}")
                return False
        
        # Check data shapes
        ndvi = scenario['ndvi']
        if not isinstance(ndvi, np.ndarray):
            print(f"❌ NDVI is not a numpy array in {scenario_name}")
            return False
        
        print(f"  ✓ {scenario_name}: {scenario['description']}")
        print(f"    - NDVI shape: {ndvi.shape}")
        print(f"    - NDVI range: [{np.min(ndvi):.3f}, {np.max(ndvi):.3f}]")
        print(f"    - Health status: {scenario['health_status']}")
    
    print("✅ All scenarios accessible and valid")
    return True


def test_time_series_access():
    """Test accessing time series data."""
    print("\n" + "=" * 60)
    print("Test 4: Time Series Access")
    print("=" * 60)
    
    manager = DemoDataManager()
    manager.load_demo_data()
    
    for scenario_name in manager.get_scenario_names():
        time_series = manager.get_time_series(scenario_name)
        
        if time_series is None:
            print(f"❌ Failed to load time series for {scenario_name}")
            return False
        
        if len(time_series) != 5:
            print(f"❌ Expected 5 time points for {scenario_name}, got {len(time_series)}")
            return False
        
        print(f"  ✓ {scenario_name}: {len(time_series)} time points")
        
        # Check first and last time points
        first = time_series[0]
        last = time_series[-1]
        print(f"    - Date range: {first['acquisition_date']} to {last['acquisition_date']}")
        print(f"    - Mean NDVI: {first['mean_ndvi']:.3f} → {last['mean_ndvi']:.3f}")
    
    print("✅ Time series data accessible and valid")
    return True


def test_alerts_access():
    """Test accessing alerts data."""
    print("\n" + "=" * 60)
    print("Test 5: Alerts Access")
    print("=" * 60)
    
    manager = DemoDataManager()
    manager.load_demo_data()
    
    all_alerts = manager.get_alerts()
    print(f"Total alerts: {len(all_alerts)}")
    
    if len(all_alerts) < 8:
        print(f"❌ Expected at least 8 alerts, got {len(all_alerts)}")
        return False
    
    # Check severity distribution
    severities = {}
    for alert in all_alerts:
        severity = alert.get('severity', 'unknown')
        severities[severity] = severities.get(severity, 0) + 1
    
    print(f"Alert severity distribution:")
    for severity, count in severities.items():
        print(f"  - {severity}: {count}")
    
    # Check for all severity levels
    expected_severities = ['critical', 'high', 'medium', 'low']
    for severity in expected_severities:
        if severity not in severities:
            print(f"❌ Missing alerts with severity: {severity}")
            return False
    
    # Test active alerts
    active_alerts = manager.get_active_alerts()
    print(f"Active (unacknowledged) alerts: {len(active_alerts)}")
    
    print("✅ Alerts data accessible and valid")
    return True


def test_predictions_access():
    """Test accessing predictions data."""
    print("\n" + "=" * 60)
    print("Test 6: Predictions Access")
    print("=" * 60)
    
    manager = DemoDataManager()
    manager.load_demo_data()
    
    for scenario_name in manager.get_scenario_names():
        predictions = manager.get_predictions(scenario_name)
        
        if predictions is None:
            print(f"❌ Failed to load predictions for {scenario_name}")
            return False
        
        # Check required keys
        required_keys = ['predictions', 'confidence_scores', 'class_names', 'model_version']
        for key in required_keys:
            if key not in predictions:
                print(f"❌ Missing key '{key}' in predictions for {scenario_name}")
                return False
        
        pred_array = predictions['predictions']
        conf_array = predictions['confidence_scores']
        
        print(f"  ✓ {scenario_name}:")
        print(f"    - Predictions shape: {pred_array.shape}")
        print(f"    - Confidence shape: {conf_array.shape}")
        print(f"    - Model version: {predictions['model_version']}")
        print(f"    - Classes: {predictions['class_names']}")
        
        # Check class distribution
        if 'metadata' in predictions and 'class_distribution' in predictions['metadata']:
            dist = predictions['metadata']['class_distribution']
            print(f"    - Class distribution: {dist}")
    
    print("✅ Predictions data accessible and valid")
    return True


def test_dashboard_formatting():
    """Test formatting data for dashboard."""
    print("\n" + "=" * 60)
    print("Test 7: Dashboard Data Formatting")
    print("=" * 60)
    
    manager = DemoDataManager()
    manager.load_demo_data()
    
    for scenario_name in manager.get_scenario_names():
        dashboard_data = manager.format_for_dashboard(scenario_name)
        
        if not dashboard_data:
            print(f"❌ Failed to format data for {scenario_name}")
            return False
        
        # Check required sections
        required_sections = ['imagery', 'alerts', 'predictions', 'scenario_info']
        for section in required_sections:
            if section not in dashboard_data:
                print(f"❌ Missing section '{section}' in dashboard data for {scenario_name}")
                return False
        
        print(f"  ✓ {scenario_name}: Dashboard data formatted correctly")
        print(f"    - Imagery date: {dashboard_data['imagery']['acquisition_date']}")
        print(f"    - Active alerts: {len(dashboard_data['alerts'])}")
        print(f"    - Scenario: {dashboard_data['scenario_info']['description']}")
    
    print("✅ Dashboard formatting working correctly")
    return True


def test_singleton_pattern():
    """Test singleton pattern for demo manager."""
    print("\n" + "=" * 60)
    print("Test 8: Singleton Pattern")
    print("=" * 60)
    
    manager1 = get_demo_manager()
    manager2 = get_demo_manager()
    
    if manager1 is not manager2:
        print("❌ Singleton pattern not working - different instances returned")
        return False
    
    print("✅ Singleton pattern working correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AgriFlux Demo Mode Test Suite")
    print("=" * 60)
    
    tests = [
        test_demo_data_availability,
        test_demo_data_loading,
        test_scenario_access,
        test_time_series_access,
        test_alerts_access,
        test_predictions_access,
        test_dashboard_formatting,
        test_singleton_pattern
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
