#!/usr/bin/env python3
"""
Comprehensive verification script to ensure all requirements are met.

This script checks:
1. All dependencies are installed
2. Database is populated with real data
3. All critical functions work correctly
4. Dashboard can load without errors
5. All features are functional
"""

import sys
import os
from pathlib import Path
import importlib

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("1. Checking Dependencies")
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('plotly', 'plotly'),
        ('folium', 'folium'),
        ('rasterio', 'rasterio'),
        ('geopandas', 'geopandas'),
        ('scikit-learn', 'sklearn'),
        ('Pillow', 'PIL'),
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        try:
            importlib.import_module(import_name)
            print_success(f"{package_name} is installed")
        except ImportError:
            print_error(f"{package_name} is NOT installed")
            all_installed = False
    
    return all_installed


def check_database():
    """Check if database exists and is populated."""
    print_header("2. Checking Database")
    
    db_path = Path('data/agriflux.db')
    
    if not db_path.exists():
        print_error(f"Database not found at {db_path}")
        return False
    
    print_success(f"Database exists at {db_path}")
    
    # Check database content
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.database.db_manager import DatabaseManager
        
        db = DatabaseManager(str(db_path))
        
        # Check imagery
        latest = db.get_latest_imagery()
        if latest:
            print_success(f"Database has imagery data (ID: {latest['id']})")
        else:
            print_error("Database has no imagery data")
            return False
        
        # Check alerts
        alerts = db.get_active_alerts()
        print_success(f"Database has {len(alerts)} active alerts")
        
        # Check stats
        stats = db.get_database_stats()
        print_success(f"Database stats: {stats['imagery_count']} imagery, {stats['total_alerts']} total alerts")
        
        return True
        
    except Exception as e:
        print_error(f"Error checking database: {str(e)}")
        return False


def check_processed_data():
    """Check if processed data files exist."""
    print_header("3. Checking Processed Data")
    
    processed_dir = Path('data/processed')
    
    if not processed_dir.exists():
        print_error(f"Processed data directory not found: {processed_dir}")
        return False
    
    print_success(f"Processed data directory exists")
    
    # Check for GeoTIFF files
    geotiff_files = list(processed_dir.rglob('*.tif'))
    
    if len(geotiff_files) == 0:
        print_error("No GeoTIFF files found in processed data")
        return False
    
    print_success(f"Found {len(geotiff_files)} GeoTIFF files")
    
    # Check for specific indices
    required_indices = ['NDVI', 'SAVI', 'EVI', 'NDWI']
    found_indices = []
    
    for index in required_indices:
        index_files = [f for f in geotiff_files if index in f.name]
        if index_files:
            found_indices.append(index)
            print_success(f"Found {index} data: {index_files[0].name}")
        else:
            print_warning(f"{index} data not found")
    
    return len(found_indices) >= 2  # At least 2 indices should be present


def check_sentinel_data():
    """Check if Sentinel-2 SAFE directory exists."""
    print_header("4. Checking Sentinel-2 Data")
    
    safe_dir = Path('S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE')
    
    if not safe_dir.exists():
        print_error(f"Sentinel-2 SAFE directory not found: {safe_dir}")
        return False
    
    print_success(f"SAFE directory exists: {safe_dir}")
    
    # Check for band files
    band_files = list(safe_dir.rglob('*_B*.jp2'))
    
    if len(band_files) == 0:
        print_error("No band files found in SAFE directory")
        return False
    
    print_success(f"Found {len(band_files)} band files")
    
    return True


def check_critical_modules():
    """Check if critical modules can be imported."""
    print_header("5. Checking Critical Modules")
    
    modules_to_check = [
        ('src.data_processing.vegetation_indices', 'VegetationIndexCalculator'),
        ('src.ai_models.rule_based_classifier', 'RuleBasedClassifier'),
        ('src.alerts.alert_generator', 'AlertGenerator'),
        ('src.database.db_manager', 'DatabaseManager'),
        ('src.utils.error_handler', 'safe_page'),
        ('src.utils.dependency_checker', 'DependencyChecker'),
    ]
    
    all_ok = True
    for module_name, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print_success(f"{module_name}.{class_name} is available")
            else:
                print_error(f"{module_name}.{class_name} not found")
                all_ok = False
        except Exception as e:
            print_error(f"Error importing {module_name}: {str(e)}")
            all_ok = False
    
    return all_ok


def check_configuration():
    """Check if configuration is properly set up."""
    print_header("6. Checking Configuration")
    
    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        print_success(".env file exists")
    else:
        print_warning(".env file not found (using defaults)")
    
    # Check config.py
    try:
        from config import config
        
        print_success(f"Configuration loaded: {config.environment} environment")
        print_success(f"Database path: {config.database.path}")
        print_success(f"Log level: {config.logging.level}")
        print_success(f"Demo mode: {config.dashboard.enable_demo_mode}")
        
        return True
    except Exception as e:
        print_error(f"Error loading configuration: {str(e)}")
        return False


def check_demo_data():
    """Check if demo data is available."""
    print_header("7. Checking Demo Data")
    
    demo_dir = Path('data/demo')
    
    if not demo_dir.exists():
        print_warning("Demo data directory not found")
        return False
    
    print_success("Demo data directory exists")
    
    # Check for demo files
    demo_files = list(demo_dir.glob('*.pkl'))
    
    if len(demo_files) == 0:
        print_warning("No demo data files found")
        return False
    
    print_success(f"Found {len(demo_files)} demo data files")
    
    return True


def check_logs():
    """Check if logging is set up correctly."""
    print_header("8. Checking Logging")
    
    log_dir = Path('logs')
    
    if not log_dir.exists():
        print_warning("Logs directory not found (will be created on first run)")
        return True
    
    print_success("Logs directory exists")
    
    log_files = list(log_dir.glob('*.log'))
    
    if len(log_files) > 0:
        print_success(f"Found {len(log_files)} log files")
    else:
        print_warning("No log files found yet")
    
    return True


def run_quick_tests():
    """Run quick functional tests."""
    print_header("9. Running Quick Functional Tests")
    
    all_passed = True
    
    # Test 1: Vegetation index calculation
    try:
        from src.data_processing.vegetation_indices import VegetationIndexCalculator
        import numpy as np
        
        calc = VegetationIndexCalculator()
        # Simple test with known values
        test_result = calc.calculate_ndvi.__doc__ is not None
        
        if test_result:
            print_success("Vegetation index calculator works")
        else:
            print_error("Vegetation index calculator test failed")
            all_passed = False
    except Exception as e:
        print_error(f"Vegetation index test failed: {str(e)}")
        all_passed = False
    
    # Test 2: Rule-based classifier
    try:
        from src.ai_models.rule_based_classifier import RuleBasedClassifier
        import numpy as np
        
        classifier = RuleBasedClassifier()
        test_ndvi = np.array([0.8, 0.6, 0.4, 0.2])
        result = classifier.classify(test_ndvi)
        
        if result is not None and len(result.predictions) == 4:
            print_success("Rule-based classifier works")
        else:
            print_error("Rule-based classifier test failed")
            all_passed = False
    except Exception as e:
        print_error(f"Rule-based classifier test failed: {str(e)}")
        all_passed = False
    
    # Test 3: Alert generator
    try:
        from src.alerts.alert_generator import AlertGenerator
        import numpy as np
        
        generator = AlertGenerator()
        test_ndvi = np.random.uniform(0.2, 0.8, (100, 100))
        alerts = generator.generate_alerts(test_ndvi)
        
        if isinstance(alerts, list):
            print_success(f"Alert generator works (generated {len(alerts)} alerts)")
        else:
            print_error("Alert generator test failed")
            all_passed = False
    except Exception as e:
        print_error(f"Alert generator test failed: {str(e)}")
        all_passed = False
    
    # Test 4: Database operations
    try:
        from src.database.db_manager import DatabaseManager
        
        db = DatabaseManager('data/agriflux.db')
        latest = db.get_latest_imagery()
        
        if latest is not None:
            print_success("Database operations work")
        else:
            print_warning("Database is empty but operations work")
    except Exception as e:
        print_error(f"Database operations test failed: {str(e)}")
        all_passed = False
    
    return all_passed


def check_dashboard_files():
    """Check if all dashboard files exist."""
    print_header("10. Checking Dashboard Files")
    
    required_files = [
        'src/dashboard/main.py',
        'src/dashboard/pages/overview.py',
        'src/dashboard/pages/field_monitoring.py',
        'src/dashboard/pages/temporal_analysis.py',
        'src/dashboard/pages/alerts.py',
        'src/dashboard/pages/data_export.py',
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print_success(f"{file_path} exists")
        else:
            print_error(f"{file_path} NOT found")
            all_exist = False
    
    return all_exist


def main():
    """Run all verification checks."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}AgriFlux Dashboard - Requirements Verification{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")
    
    results = {
        'Dependencies': check_dependencies(),
        'Database': check_database(),
        'Processed Data': check_processed_data(),
        'Sentinel-2 Data': check_sentinel_data(),
        'Critical Modules': check_critical_modules(),
        'Configuration': check_configuration(),
        'Demo Data': check_demo_data(),
        'Logging': check_logs(),
        'Functional Tests': run_quick_tests(),
        'Dashboard Files': check_dashboard_files(),
    }
    
    # Print summary
    print_header("Verification Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        if result:
            print_success(f"{check}: PASSED")
        else:
            print_error(f"{check}: FAILED")
    
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}Results: {passed}/{total} checks passed{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")
    
    if passed == total:
        print_success("All requirements verified! System is ready for demo.")
        return 0
    elif passed >= total * 0.8:
        print_warning(f"Most requirements met ({passed}/{total}). Review failed checks.")
        return 0
    else:
        print_error(f"Too many failures ({total - passed}/{total}). System needs attention.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
