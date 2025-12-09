"""
Test script for alert system enhancements
"""

import numpy as np
from src.alerts.alert_generator import AlertGenerator, AlertSeverity, AlertType
from src.alerts.alert_preferences import AlertPreferencesManager
from src.alerts.alert_export import AlertExporter

def test_alert_generation_with_context():
    """Test alert generation with enhanced context"""
    print("Testing alert generation with context...")
    
    generator = AlertGenerator()
    
    # Create sample NDVI data with stress
    ndvi = np.random.uniform(0.2, 0.8, (100, 100))
    ndvi[0:30, 0:30] = 0.25  # Critical stress area
    
    # Generate alerts with context
    alerts = generator.generate_alerts(
        ndvi=ndvi,
        field_name="North Field",
        coordinates=(30.95, 75.85),
        previous_values={'ndvi': 0.65},
        days_since_last=7.0
    )
    
    print(f"✓ Generated {len(alerts)} alerts")
    
    if alerts:
        alert = alerts[0]
        print(f"  - Alert type: {alert.alert_type.value}")
        print(f"  - Severity: {alert.severity.value}")
        print(f"  - Field name: {alert.field_name}")
        print(f"  - Coordinates: {alert.coordinates}")
        print(f"  - Priority score: {alert.priority_score:.2f}")
        print(f"  - Historical context: {alert.historical_context}")
        print(f"  - Message: {alert.message[:100]}...")
    
    return True

def test_priority_ranking():
    """Test alert priority ranking"""
    print("\nTesting alert priority ranking...")
    
    generator = AlertGenerator()
    
    # Create sample alerts with different severities
    ndvi1 = np.full((100, 100), 0.25)  # Critical
    ndvi2 = np.full((100, 100), 0.45)  # High stress
    ndvi3 = np.full((100, 100), 0.55)  # Medium stress
    
    alerts = []
    alerts.extend(generator.generate_alerts(ndvi=ndvi1, field_name="Field A"))
    alerts.extend(generator.generate_alerts(ndvi=ndvi2, field_name="Field B"))
    alerts.extend(generator.generate_alerts(ndvi=ndvi3, field_name="Field C"))
    
    # Rank alerts
    ranked = generator.rank_alerts_by_priority(alerts)
    
    print(f"✓ Ranked {len(ranked)} alerts by priority")
    print(f"  - Highest priority: {ranked[0].priority_score:.2f} ({ranked[0].severity.value})")
    print(f"  - Lowest priority: {ranked[-1].priority_score:.2f} ({ranked[-1].severity.value})")
    
    # Get top 5
    top_5 = generator.get_top_priority_alerts(alerts, 5)
    print(f"✓ Retrieved top 5 alerts")
    
    # Categorize alerts
    categorized = generator.categorize_alerts(alerts)
    print(f"✓ Categorized alerts:")
    print(f"  - Needs Attention: {len(categorized['needs_attention'])}")
    print(f"  - For Information: {len(categorized['for_information'])}")
    
    return True

def test_alert_preferences():
    """Test alert preferences management"""
    print("\nTesting alert preferences...")
    
    prefs_manager = AlertPreferencesManager(preferences_file="data/test_alert_preferences.json")
    
    # Test severity threshold
    prefs_manager.update_severity_threshold('high')
    print(f"✓ Updated severity threshold to: {prefs_manager.preferences.severity_threshold}")
    
    # Test alert type filtering
    prefs_manager.update_alert_type_filter({'vegetation_stress', 'water_stress'})
    print(f"✓ Updated alert types: {prefs_manager.preferences.enabled_alert_types}")
    
    # Test snooze functionality
    alert_id = 123
    expiry = prefs_manager.snooze_alert(alert_id, hours=24)
    print(f"✓ Snoozed alert {alert_id} until {expiry}")
    
    is_snoozed = prefs_manager.is_alert_snoozed(alert_id)
    print(f"✓ Alert {alert_id} is snoozed: {is_snoozed}")
    
    # Test unsnooze
    prefs_manager.unsnooze_alert(alert_id)
    is_snoozed = prefs_manager.is_alert_snoozed(alert_id)
    print(f"✓ Alert {alert_id} is snoozed after unsnooze: {is_snoozed}")
    
    return True

def test_alert_export():
    """Test alert export functionality"""
    print("\nTesting alert export...")
    
    exporter = AlertExporter()
    
    # Create sample alerts
    sample_alerts = [
        {
            'id': 1,
            'alert_type': 'vegetation_stress',
            'severity': 'critical',
            'message': 'Severe vegetation stress detected',
            'recommendation': 'Immediate irrigation required',
            'created_at': '2024-12-09T10:00:00',
            'acknowledged': 0,
            'metadata': '{"field_name": "North Field", "priority_score": 85.5}'
        },
        {
            'id': 2,
            'alert_type': 'water_stress',
            'severity': 'high',
            'message': 'High water stress detected',
            'recommendation': 'Increase irrigation frequency',
            'created_at': '2024-12-09T11:00:00',
            'acknowledged': 0,
            'metadata': '{"field_name": "South Field", "priority_score": 72.3}'
        }
    ]
    
    # Test CSV export
    csv_content = exporter.export_to_csv(sample_alerts)
    print(f"✓ Generated CSV export ({len(csv_content)} characters)")
    
    # Test summary report
    report = exporter.generate_summary_report(sample_alerts)
    print(f"✓ Generated summary report ({len(report)} characters)")
    
    # Test email template
    email_html = exporter.generate_email_template(sample_alerts, recipient_name="Test User")
    print(f"✓ Generated email template ({len(email_html)} characters)")
    
    return True

def main():
    """Run all tests"""
    print("=" * 80)
    print("ALERT SYSTEM ENHANCEMENTS TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Alert Generation with Context", test_alert_generation_with_context),
        ("Priority Ranking", test_priority_ranking),
        ("Alert Preferences", test_alert_preferences),
        ("Alert Export", test_alert_export)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
