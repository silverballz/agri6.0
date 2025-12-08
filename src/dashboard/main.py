"""
AgriFlux - Smart Agricultural Intelligence Platform
Multi-page Streamlit application for crop health monitoring and analysis
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import error handling and dependency checking
from utils.error_handler import setup_logging, logger, display_error_summary
from utils.dependency_checker import check_dependencies_on_startup

# Import UI components and help system
# Temporarily disabled to fix import issues
# try:
#     from dashboard.ui_components import (
#         ColorScheme, Icons, HelpText,
#         show_vegetation_index_help, show_feature_help
#     )
#     from dashboard.help_system import show_quick_help, show_faq_section
# except ImportError:
#     pass

# Import dashboard pages
from dashboard.pages import (
    overview,
    field_monitoring,
    temporal_analysis,
    alerts,
    data_export
)

# Page configuration
st.set_page_config(
    page_title="🌱 AgriFlux - Smart Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional AgriFlux Color Scheme
st.markdown("""
<style>
    /* Clean professional theme */
    .stApp {
        background-color: #1a1d29 !important;
        color: #e0e0e0 !important;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
        background-color: #1a1d29 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #252936 !important;
    }
    
    /* AgriFlux branding */
    .agriflux-brand {
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 0.8rem;
        color: #66bb6a !important;
        background-color: #2d3748;
        border-radius: 10px;
        border: 1px solid #66bb6a;
    }
    
    .agriflux-tagline {
        font-size: 0.85rem;
        text-align: center;
        margin-bottom: 1.5rem;
        font-style: italic;
        color: #9ca3af !important;
    }
    
    /* Metric cards */
    .metric-container {
        background-color: #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid #3d4a5c;
        transition: all 0.2s ease;
    }
    
    .metric-container:hover {
        border-color: #66bb6a;
        box-shadow: 0 4px 12px rgba(102, 187, 106, 0.15);
    }
    
    .metric-title {
        color: #9ca3af !important;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #ffffff !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .metric-delta-positive {
        color: #66bb6a !important;
    }
    
    .metric-delta-negative {
        color: #ef5350 !important;
    }
    
    /* Sidebar elements */
    [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #66bb6a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #81c784 !important;
        box-shadow: 0 4px 12px rgba(102, 187, 106, 0.3) !important;
    }
    
    /* Text elements */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Status indicators */
    .status-healthy { 
        color: #66bb6a !important; 
        font-weight: 600;
    }
    .status-warning { 
        color: #ffa726 !important; 
        font-weight: 600;
    }
    .status-critical { 
        color: #ef5350 !important; 
        font-weight: 600;
    }
    
    /* Alert styling */
    .alert-high {
        background-color: #3d2626 !important;
        border-left: 4px solid #ef5350 !important;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ffcdd2 !important;
        border-radius: 8px;
    }
    
    .alert-medium {
        background-color: #3d3226 !important;
        border-left: 4px solid #ffa726 !important;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ffe0b2 !important;
        border-radius: 8px;
    }
    
    .alert-low {
        background-color: #263d26 !important;
        border-left: 4px solid #66bb6a !important;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #b9f6ca !important;
        border-radius: 8px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background-color: #252936;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #3d4a5c;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #66bb6a;
    }
    
    /* Responsive design for tablets and mobile */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .agriflux-brand {
            font-size: 1.3rem;
        }
        
        .metric-container {
            padding: 1rem;
            margin: 0.25rem;
        }
        
        .metric-value {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables for user interactions"""
    
    # User preferences
    if 'selected_zones' not in st.session_state:
        st.session_state.selected_zones = []
    
    if 'date_range' not in st.session_state:
        st.session_state.date_range = (
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
    
    if 'selected_indices' not in st.session_state:
        st.session_state.selected_indices = ['NDVI', 'SAVI']
    
    # Demo mode settings
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    
    if 'demo_scenario' not in st.session_state:
        st.session_state.demo_scenario = 'healthy_field'
    
    if 'demo_data' not in st.session_state:
        st.session_state.demo_data = None
    
    # Alert management
    if 'acknowledged_alerts' not in st.session_state:
        st.session_state.acknowledged_alerts = set()
    
    if 'alert_filters' not in st.session_state:
        st.session_state.alert_filters = {
            'severity': ['High', 'Medium', 'Low'],
            'type': ['Vegetation Stress', 'Pest Risk', 'Disease Risk', 'Environmental']
        }
    
    # Map settings - Default to Ludhiana, Punjab, India
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [31.1, 75.81]  # Ludhiana coordinates
    
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = 12
    
    # Data refresh settings
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

def create_sidebar_navigation():
    """Create sidebar navigation with page selection and filters"""
    
    # AgriFlux branding
    st.sidebar.markdown("""
    <div class="agriflux-brand">
        🌱 AgriFlux
    </div>
    <div class="agriflux-tagline">
        Smart Agricultural Intelligence
    </div>
    """, unsafe_allow_html=True)
    
    # Help button with improved functionality
    if st.sidebar.button("❓ Help & Documentation", help="Access user guides, tutorials, and support resources"):
        st.session_state.show_help = not st.session_state.get('show_help', False)
    
    # Display help if toggled
    if st.session_state.get('show_help', False):
        st.sidebar.markdown("### 📚 Quick Help")
        with st.sidebar.expander("🚀 Getting Started"):
            st.markdown("1. Select fields\n2. Choose date range\n3. Pick indices\n4. Navigate pages")
        with st.sidebar.expander("📊 NDVI Values"):
            st.markdown("• 0.8-1.0: Excellent 🟢\n• 0.6-0.8: Healthy 🟢\n• 0.4-0.6: Moderate 🟡\n• <0.4: Stressed 🔴")
        
        # Link to full documentation
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📖 Full Documentation")
        st.sidebar.markdown("[Quick Start Guide](../docs/QUICK_START_GUIDE.md)")
        st.sidebar.markdown("[Interpretation Guide](../docs/INTERPRETATION_GUIDE.md)")
        st.sidebar.markdown("[User Guide](../docs/user-guide.md)")
        st.sidebar.markdown("[FAQ](../docs/faq.md)")
    
    st.sidebar.markdown("---")
    
    # Page navigation with descriptions
    pages = {
        "📊 Overview": "overview",
        "🗺️ Field Monitoring": "field_monitoring", 
        "📈 Temporal Analysis": "temporal_analysis",
        "🚨 Alerts & Notifications": "alerts",
        "📤 Data Export": "data_export"
    }
    
    # Create selectbox with help text
    page_options = list(pages.keys())
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        page_options,
        key="page_selector",
        help="Select a page to navigate to different features of the platform"
    )
    
    # Show page description based on selection
    descriptions = {
        "📊 Overview": "Main dashboard with key metrics and field overview",
        "🗺️ Field Monitoring": "Interactive maps and real-time field health analysis", 
        "📈 Temporal Analysis": "Time series charts and vegetation trend analysis",
        "🚨 Alerts & Notifications": "Active alerts, warnings, and notification management",
        "📤 Data Export": "Download reports, data, and generate custom exports"
    }
    st.sidebar.caption(f"ℹ️ {descriptions[selected_page]}")
    
    st.sidebar.markdown("---")
    
    # Global filters
    st.sidebar.subheader("🔧 Global Filters")
    
    # Date range selector with help
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=st.session_state.date_range,
        key="global_date_range",
        help="Select the time period for data analysis. Satellite data is typically available every 5-10 days depending on cloud coverage."
    )
    
    if len(date_range) == 2:
        st.session_state.date_range = date_range
    
    # Monitoring zones selector with help
    available_zones = get_available_zones()
    selected_zones = st.sidebar.multiselect(
        "🗺️ Monitoring Zones",
        available_zones,
        default=st.session_state.selected_zones,
        key="global_zones",
        help="Select one or more field zones to analyze. Leave empty to show all zones. You can create new zones in the Field Monitoring page."
    )
    st.session_state.selected_zones = selected_zones
    
    # Vegetation indices selector with detailed help
    available_indices = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI']
    selected_indices = st.sidebar.multiselect(
        "📊 Vegetation Indices",
        available_indices,
        default=st.session_state.selected_indices,
        key="global_indices",
        help="""Select vegetation indices to display:
        • NDVI: General vegetation health (most common)
        • SAVI: Better for sparse vegetation and exposed soil
        • EVI: Enhanced sensitivity for dense vegetation
        • NDWI: Water content and irrigation monitoring
        • NDSI: Soil moisture and bare soil detection"""
    )
    st.session_state.selected_indices = selected_indices
    
    # Add help links for each index
    if selected_indices:
        with st.sidebar.expander("ℹ️ Learn About Selected Indices"):
            for index in selected_indices:
                st.markdown(f"**{index}**: Vegetation health indicator")
    
    st.sidebar.markdown("---")
    
    # Demo mode section
    st.sidebar.subheader("🎬 Demo Mode")
    
    # Import demo manager with absolute path to demo data
    from utils.demo_data_manager import get_demo_manager
    import os
    # Get the project root directory (2 levels up from dashboard)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    demo_data_path = os.path.join(project_root, 'data', 'demo')
    demo_manager = get_demo_manager(demo_data_dir=demo_data_path)
    
    # Check if demo data is available
    if demo_manager.is_demo_data_available():
        # Demo mode toggle
        demo_mode_enabled = st.sidebar.checkbox(
            "Enable Demo Mode",
            value=st.session_state.demo_mode,
            key="demo_mode_toggle",
            help="Load pre-configured demo data for quick demonstrations. Includes 3 field scenarios with time series, alerts, and predictions."
        )
        
        # If demo mode was just enabled, load the data
        if demo_mode_enabled and not st.session_state.demo_mode:
            with st.spinner("Loading demo data..."):
                if demo_manager.load_demo_data():
                    st.session_state.demo_mode = True
                    st.session_state.demo_data = demo_manager
                    st.sidebar.success("✅ Demo data loaded!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to load demo data")
        
        # If demo mode was just disabled
        elif not demo_mode_enabled and st.session_state.demo_mode:
            st.session_state.demo_mode = False
            st.session_state.demo_data = None
            st.sidebar.info("Demo mode disabled")
            st.rerun()
        
        # If demo mode is active, show scenario selector
        if st.session_state.demo_mode and st.session_state.demo_data:
            st.sidebar.markdown("**Select Demo Scenario:**")
            
            scenario_options = {
                'healthy_field': '🌱 Healthy Field',
                'stressed_field': '⚠️ Stressed Field',
                'mixed_field': '🔄 Mixed Field'
            }
            
            selected_scenario = st.sidebar.selectbox(
                "Scenario",
                options=list(scenario_options.keys()),
                format_func=lambda x: scenario_options[x],
                index=list(scenario_options.keys()).index(st.session_state.demo_scenario),
                key="demo_scenario_selector",
                help="Choose a field scenario to demonstrate different health conditions"
            )
            
            if selected_scenario != st.session_state.demo_scenario:
                st.session_state.demo_scenario = selected_scenario
                st.rerun()
            
            # Show scenario description
            if st.session_state.demo_data:
                description = st.session_state.demo_data.get_scenario_description(selected_scenario)
                st.sidebar.caption(description)
            
            # Exit demo mode button
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Exit Demo Mode", help="Return to real data mode"):
                st.session_state.demo_mode = False
                st.session_state.demo_data = None
                st.sidebar.success("Exited demo mode")
                st.rerun()
    else:
        st.sidebar.info("📦 Demo data not available. Run `python scripts/generate_demo_data.py` to generate demo data.")
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("📡 System Status")
    
    # Auto-refresh toggle with help
    auto_refresh = st.sidebar.checkbox(
        "🔄 Auto-refresh data",
        value=st.session_state.auto_refresh,
        key="auto_refresh_toggle",
        help="Automatically refresh data every 30 seconds. Useful for monitoring real-time changes but may impact performance."
    )
    st.session_state.auto_refresh = auto_refresh
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now", help="Manually refresh all data from the database and sensors"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Last refresh time
    st.sidebar.caption(f"⏰ Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # System health indicators
    display_system_status()
    
    return pages[selected_page]

def get_available_zones():
    """Get list of available monitoring zones in Ludhiana area"""
    # This would typically query the database
    # For now, return mock data for Ludhiana agricultural zones
    return [
        "Ludhiana North Farm",
        "Pakhowal Road Fields", 
        "Sidhwan Bet Area",
        "Raikot Agricultural Zone",
        "Khanna District Fields"
    ]

def display_system_status():
    """Display system health status in sidebar"""
    
    st.sidebar.markdown("**System Health:**")
    
    # Mock system status - would be real data in production
    status_items = [
        ("Satellite Data", "✅", "status-healthy"),
        ("Sensor Network", "⚠️", "status-warning"), 
        ("AI Models", "✅", "status-healthy"),
        ("Database", "✅", "status-healthy")
    ]
    
    for item, icon, css_class in status_items:
        st.sidebar.markdown(
            f"{icon} <span class='{css_class}'>{item}</span>",
            unsafe_allow_html=True
        )

# Help modal function removed - now using help_system.py module

def display_header():
    """Display main header with key metrics in dark theme"""
    
    # Show demo mode badge if active
    if st.session_state.get('demo_mode', False):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                    padding: 1rem; border-radius: 10px; text-align: center; 
                    margin-bottom: 1rem; border: 2px solid #ff8787;">
            <h3 style="color: #ffffff; margin: 0;">🎬 DEMO MODE ACTIVE</h3>
            <p style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Showing pre-configured demo data for demonstration purposes
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show current scenario info
        scenario_name = st.session_state.get('demo_scenario', 'healthy_field')
        if st.session_state.demo_data:
            description = st.session_state.demo_data.get_scenario_description(scenario_name)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%); 
                        padding: 0.75rem; border-radius: 8px; text-align: center; 
                        margin-bottom: 1.5rem; border: 1px solid #4caf50;">
                <p style="color: #ffffff; margin: 0; font-size: 1rem;">
                    <strong>Current Scenario:</strong> {description}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<h2 style="color: #ffffff; text-align: center; margin-bottom: 2rem;">🌱 AgriFlux Intelligence Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_fields = len(st.session_state.selected_zones) if st.session_state.selected_zones else 5
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">🗺️ Active Fields</div>
            <div class="metric-value">{active_fields}</div>
            <div class="metric-delta metric-delta-positive">↗ +2 new this week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        alert_count = get_active_alert_count()
        alert_color = "metric-delta-positive" if alert_count < 5 else "metric-delta-negative"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">🚨 Smart Alerts</div>
            <div class="metric-value">{alert_count}</div>
            <div class="metric-delta {alert_color}">↘ -3 from yesterday</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        health_index = 0.72
        health_color = "metric-delta-positive" if health_index > 0.7 else "metric-delta-negative"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">🌱 Health Index</div>
            <div class="metric-value">{health_index:.2f}</div>
            <div class="metric-delta {health_color}">↗ +0.05 from last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        data_quality = 94
        quality_color = "metric-delta-positive" if data_quality > 90 else "metric-delta-negative"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">📡 Data Quality</div>
            <div class="metric-value">{data_quality}%</div>
            <div class="metric-delta {quality_color}">↗ +2% improvement</div>
        </div>
        """, unsafe_allow_html=True)

def get_active_alert_count():
    """Get count of active alerts"""
    # Mock data - would query database in production
    return 7

def main():
    """Main application entry point"""
    
    # Set up logging
    setup_logging()
    logger.info("AgriFlux Dashboard starting...")
    
    # Check dependencies on startup
    system_ready = check_dependencies_on_startup()
    
    # Display error summary in sidebar
    display_error_summary()
    
    # If critical issues exist, show warning and limited functionality
    if not system_ready:
        st.error("⚠️ **Critical System Issues Detected**")
        st.warning("Some required dependencies are missing. Please install them to use the full dashboard.")
        st.info("Check the System Health Details in the sidebar for more information.")
        
        # Show installation instructions
        with st.expander("📦 Installation Instructions", expanded=True):
            st.markdown("""
            ### Install Required Dependencies
            
            Run the following command to install all required packages:
            
            ```bash
            pip install -r requirements.txt
            ```
            
            ### Verify Installation
            
            After installation, refresh this page to verify all dependencies are installed correctly.
            """)
        
        # Still allow limited navigation
        st.markdown("---")
        st.info("You can still navigate the dashboard, but some features may not work correctly.")
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar navigation
    current_page = create_sidebar_navigation()
    
    # Display header metrics only
    display_header()
    
    # Add some spacing
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Show selected page info
    page_descriptions = {
        "overview": "📊 Overview - Main dashboard with key metrics and field overview",
        "field_monitoring": "🗺️ Field Monitoring - Interactive maps and real-time field health analysis", 
        "temporal_analysis": "📈 Temporal Analysis - Time series charts and vegetation trend analysis",
        "alerts": "🚨 Alerts & Notifications - Active alerts, warnings, and notification management",
        "data_export": "📤 Data Export - Download reports, data, and generate custom exports"
    }
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); 
                border-radius: 15px; border: 1px solid #4a5568; margin: 1rem 0;">
        <h3 style="color: #4caf50; margin-bottom: 1rem;">Selected Page</h3>
        <p style="color: #ffffff; font-size: 1.1rem; margin: 0;">
            {page_descriptions.get(current_page, "Select a page from the sidebar")}
        </p>
        <p style="color: #a0aec0; font-size: 0.9rem; margin-top: 1rem;">
            Use the dropdown menu in the sidebar to navigate between different features
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route to selected page
    if current_page == "overview":
        overview.show_page()
    elif current_page == "field_monitoring":
        field_monitoring.show_page()
    elif current_page == "temporal_analysis":
        temporal_analysis.show_page()
    elif current_page == "alerts":
        alerts.show_page()
    elif current_page == "data_export":
        data_export.show_page()
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        # Refresh every 30 seconds
        time_since_refresh = datetime.now() - st.session_state.last_refresh
        if time_since_refresh.total_seconds() > 30:
            st.session_state.last_refresh = datetime.now()
            st.rerun()

if __name__ == "__main__":
    main()