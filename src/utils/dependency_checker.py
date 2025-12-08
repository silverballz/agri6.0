"""
Dependency checker for AgriFlux dashboard
Verifies required packages and system components on startup
"""

import streamlit as st
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger('agriflux')


class DependencyChecker:
    """Check and validate system dependencies"""
    
    # Required packages with their import names and display names
    REQUIRED_PACKAGES = {
        'streamlit': ('streamlit', 'Streamlit'),
        'pandas': ('pandas', 'Pandas'),
        'numpy': ('numpy', 'NumPy'),
        'plotly': ('plotly', 'Plotly'),
        'folium': ('folium', 'Folium'),
        'streamlit_folium': ('streamlit_folium', 'Streamlit-Folium'),
    }
    
    # Optional packages for full functionality
    OPTIONAL_PACKAGES = {
        'rasterio': ('rasterio', 'Rasterio (for GeoTIFF processing)'),
        'geopandas': ('geopandas', 'GeoPandas (for geospatial operations)'),
        'sklearn': ('sklearn', 'Scikit-learn (for ML features)'),
        'PIL': ('PIL', 'Pillow (for image processing)'),
        'tensorflow': ('tensorflow', 'TensorFlow (for AI models)'),
    }
    
    # Critical paths that should exist
    CRITICAL_PATHS = {
        'data': 'data',
        'processed': 'data/processed',
        'logs': 'logs',
    }
    
    # Optional paths
    OPTIONAL_PATHS = {
        'models': 'models',
        'database': 'data/agriflux.db',
        'demo': 'data/demo',
    }
    
    def __init__(self):
        self.results = {
            'required_packages': {},
            'optional_packages': {},
            'critical_paths': {},
            'optional_paths': {},
            'overall_status': 'unknown'
        }
    
    def check_all(self) -> Dict:
        """Run all dependency checks"""
        
        logger.info("Starting dependency check...")
        
        # Check required packages
        self.check_required_packages()
        
        # Check optional packages
        self.check_optional_packages()
        
        # Check critical paths
        self.check_critical_paths()
        
        # Check optional paths
        self.check_optional_paths()
        
        # Determine overall status
        self.determine_overall_status()
        
        logger.info(f"Dependency check complete. Status: {self.results['overall_status']}")
        
        return self.results
    
    def check_required_packages(self):
        """Check if all required packages are installed"""
        
        for package_key, (import_name, display_name) in self.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(import_name)
                self.results['required_packages'][package_key] = {
                    'status': 'installed',
                    'display_name': display_name,
                    'import_name': import_name
                }
                logger.debug(f"Required package {display_name} is installed")
            except ImportError as e:
                self.results['required_packages'][package_key] = {
                    'status': 'missing',
                    'display_name': display_name,
                    'import_name': import_name,
                    'error': str(e)
                }
                logger.error(f"Required package {display_name} is missing: {e}")
    
    def check_optional_packages(self):
        """Check if optional packages are installed"""
        
        for package_key, (import_name, display_name) in self.OPTIONAL_PACKAGES.items():
            try:
                importlib.import_module(import_name)
                self.results['optional_packages'][package_key] = {
                    'status': 'installed',
                    'display_name': display_name,
                    'import_name': import_name
                }
                logger.debug(f"Optional package {display_name} is installed")
            except ImportError:
                self.results['optional_packages'][package_key] = {
                    'status': 'missing',
                    'display_name': display_name,
                    'import_name': import_name
                }
                logger.debug(f"Optional package {display_name} is not installed")
    
    def check_critical_paths(self):
        """Check if critical paths exist"""
        
        for path_key, path_str in self.CRITICAL_PATHS.items():
            path = Path(path_str)
            
            if path.exists():
                self.results['critical_paths'][path_key] = {
                    'status': 'exists',
                    'path': path_str,
                    'is_dir': path.is_dir()
                }
                logger.debug(f"Critical path {path_str} exists")
            else:
                self.results['critical_paths'][path_key] = {
                    'status': 'missing',
                    'path': path_str
                }
                logger.warning(f"Critical path {path_str} is missing")
    
    def check_optional_paths(self):
        """Check if optional paths exist"""
        
        for path_key, path_str in self.OPTIONAL_PATHS.items():
            path = Path(path_str)
            
            if path.exists():
                self.results['optional_paths'][path_key] = {
                    'status': 'exists',
                    'path': path_str,
                    'is_dir': path.is_dir() if not path.suffix else False
                }
                logger.debug(f"Optional path {path_str} exists")
            else:
                self.results['optional_paths'][path_key] = {
                    'status': 'missing',
                    'path': path_str
                }
                logger.debug(f"Optional path {path_str} is missing")
    
    def determine_overall_status(self):
        """Determine overall system status"""
        
        # Check if any required packages are missing
        missing_required = [
            pkg for pkg, info in self.results['required_packages'].items()
            if info['status'] == 'missing'
        ]
        
        # Check if any critical paths are missing
        missing_critical_paths = [
            path for path, info in self.results['critical_paths'].items()
            if info['status'] == 'missing'
        ]
        
        if missing_required:
            self.results['overall_status'] = 'critical'
            self.results['status_message'] = f"Missing required packages: {', '.join(missing_required)}"
        elif missing_critical_paths:
            self.results['overall_status'] = 'warning'
            self.results['status_message'] = f"Missing critical paths: {', '.join(missing_critical_paths)}"
        else:
            # Check optional components
            missing_optional = [
                pkg for pkg, info in self.results['optional_packages'].items()
                if info['status'] == 'missing'
            ]
            
            if missing_optional:
                self.results['overall_status'] = 'good'
                self.results['status_message'] = f"All required components available. Optional: {', '.join(missing_optional)} not installed"
            else:
                self.results['overall_status'] = 'excellent'
                self.results['status_message'] = "All components available"
    
    def create_missing_paths(self):
        """Create missing critical paths"""
        
        created = []
        failed = []
        
        for path_key, info in self.results['critical_paths'].items():
            if info['status'] == 'missing':
                try:
                    path = Path(info['path'])
                    path.mkdir(parents=True, exist_ok=True)
                    created.append(info['path'])
                    logger.info(f"Created missing path: {info['path']}")
                except Exception as e:
                    failed.append((info['path'], str(e)))
                    logger.error(f"Failed to create path {info['path']}: {e}")
        
        return created, failed


def display_dependency_status(results: Dict):
    """Display dependency check results in Streamlit"""
    
    status = results['overall_status']
    
    # Status indicator
    status_icons = {
        'excellent': '✅',
        'good': '🟢',
        'warning': '⚠️',
        'critical': '🔴',
        'unknown': '❓'
    }
    
    status_colors = {
        'excellent': 'green',
        'good': 'green',
        'warning': 'orange',
        'critical': 'red',
        'unknown': 'gray'
    }
    
    icon = status_icons.get(status, '❓')
    color = status_colors.get(status, 'gray')
    
    st.sidebar.markdown(f"**System Status:** {icon} {status.title()}")
    
    # Show details in expander
    with st.sidebar.expander("🔍 System Health Details", expanded=(status in ['warning', 'critical'])):
        st.markdown(f"**Status:** {results.get('status_message', 'Unknown')}")
        
        # Required packages
        st.markdown("**Required Packages:**")
        for pkg_key, info in results['required_packages'].items():
            if info['status'] == 'installed':
                st.markdown(f"✅ {info['display_name']}")
            else:
                st.markdown(f"❌ {info['display_name']}")
                st.code(f"pip install {info['import_name']}", language="bash")
        
        # Optional packages
        missing_optional = [
            info for info in results['optional_packages'].values()
            if info['status'] == 'missing'
        ]
        
        if missing_optional:
            st.markdown("**Optional Packages (not installed):**")
            for info in missing_optional:
                st.markdown(f"⚪ {info['display_name']}")
        
        # Critical paths
        st.markdown("**Critical Paths:**")
        for path_key, info in results['critical_paths'].items():
            if info['status'] == 'exists':
                st.markdown(f"✅ {info['path']}")
            else:
                st.markdown(f"❌ {info['path']}")
        
        # Action buttons
        if status in ['warning', 'critical']:
            st.markdown("---")
            
            if st.button("🔧 Auto-Fix Issues", key="auto_fix_dependencies"):
                fix_dependency_issues(results)


def fix_dependency_issues(results: Dict):
    """Attempt to automatically fix dependency issues"""
    
    checker = DependencyChecker()
    
    # Try to create missing paths
    created, failed = checker.create_missing_paths()
    
    if created:
        st.success(f"✅ Created missing paths: {', '.join(created)}")
    
    if failed:
        st.error(f"❌ Failed to create paths:")
        for path, error in failed:
            st.text(f"  - {path}: {error}")
    
    # Check for missing packages
    missing_required = [
        info for info in results['required_packages'].values()
        if info['status'] == 'missing'
    ]
    
    if missing_required:
        st.warning("⚠️ Missing required packages must be installed manually:")
        st.code("pip install -r requirements.txt", language="bash")
    
    # Rerun check
    if created and not missing_required:
        st.info("🔄 Refreshing system status...")
        st.rerun()


def check_dependencies_on_startup():
    """
    Check dependencies on dashboard startup
    Returns True if system is ready, False if critical issues exist
    """
    
    checker = DependencyChecker()
    results = checker.check_all()
    
    # Store results in session state
    if 'dependency_check_results' not in st.session_state:
        st.session_state.dependency_check_results = results
    
    # Display status in sidebar
    display_dependency_status(results)
    
    # Return whether system is ready
    return results['overall_status'] not in ['critical']
