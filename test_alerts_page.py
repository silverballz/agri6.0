"""
Test script to verify the alerts dashboard page can be imported and initialized.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def test_alerts_page_import():
    """Test that the alerts page can be imported without errors."""
    
    print("=" * 60)
    print("Testing Alerts Dashboard Page")
    print("=" * 60)
    
    try:
        print("\n1. Testing imports...")
        from src.dashboard.pages import alerts
        print("✅ Successfully imported alerts page module")
        
        print("\n2. Checking required functions...")
        required_functions = [
            'show_page',
            'display_alert_metrics',
            'display_active_alerts',
            'display_alert_card',
            'display_alert_history',
            'display_alert_filters',
            'display_alert_analytics',
            'get_time_ago',
            'display_affected_area_map'
        ]
        
        for func_name in required_functions:
            if hasattr(alerts, func_name):
                print(f"  ✅ Found function: {func_name}")
            else:
                print(f"  ❌ Missing function: {func_name}")
                return False
        
        print("\n3. Testing AlertGenerator...")
        from src.alerts.alert_generator import AlertGenerator, AlertSeverity, AlertType, Alert
        
        generator = AlertGenerator()
        print("✅ AlertGenerator initialized successfully")
        
        # Test alert summary with empty list
        summary = generator.get_alert_summary([])
        print(f"✅ Empty alert summary: {summary}")
        
        # Test creating an alert
        test_alert = Alert(
            alert_type=AlertType.VEGETATION_STRESS,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            recommendation="Test recommendation",
            affected_area_percentage=25.0
        )
        print(f"✅ Created test alert: {test_alert.alert_type.value}")
        
        # Test alert to dict
        alert_dict = test_alert.to_dict()
        print(f"✅ Alert to dict: {list(alert_dict.keys())}")
        
        print("\n4. Testing database operations...")
        from src.database.db_manager import DatabaseManager
        
        db = DatabaseManager()
        print("✅ DatabaseManager initialized")
        
        # Test getting active alerts (should not crash even if empty)
        active = db.get_active_alerts(limit=5)
        print(f"✅ Retrieved {len(active)} active alerts")
        
        # Test getting alert history
        history = db.get_alert_history(limit=10)
        print(f"✅ Retrieved {len(history)} historical alerts")
        
        # Test database stats
        stats = db.get_database_stats()
        print(f"✅ Database stats: {stats.get('total_alerts', 0)} total alerts")
        
        print("\n" + "=" * 60)
        print("✅ All Alerts Page Tests Passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_alerts_page_import()
    sys.exit(0 if success else 1)
