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
    page_title="AgriFlux - Smart Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode CSS for AgriFlux
st.markdown("""
<style>
    /* Dark mode global styling */
    .stApp {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    
    /* Main container dark styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar dark styling */
    .css-1d391kg {
        background-color: #1e2329 !important;
        color: #ffffff !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #1e2329 !important;
        padding: 1rem;
        color: #ffffff !important;
    }
    
    /* AgriFlux branding dark */
    .agriflux-brand {
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        color: #4caf50 !important;
        background: linear-gradient(135deg, #1e2329 0%, #2d3748 100%);
        border-radius: 10px;
    }
    
    .agriflux-tagline {
        font-size: 0.85rem;
        text-align: center;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
        font-style: italic;
        color: #a0aec0 !important;
    }
    
    /* Dark metric cards */
    .metric-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid #4a5568;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .metric-title {
        color: #4caf50 !important;
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
    }
    
    .metric-delta-positive {
        color: #4caf50 !important;
    }
    
    .metric-delta-negative {
        color: #f44336 !important;
    }
    
    /* Dark sidebar elements */
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stMultiSelect label,
    .css-1d391kg .stDateInput label,
    .css-1d391kg .stCheckbox label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    .css-1d391kg .stMultiSelect > div > div {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    /* Dark buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #66bb6a 0%, #388e3c 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Dark text elements */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p {
        color: #ffffff !important;
    }
    
    /* Status indicators dark */
    .status-healthy { color: #4caf50 !important; font-weight: bold; }
    .status-warning { color: #ff9800 !important; font-weight: bold; }
    .status-critical { color: #f44336 !important; font-weight: bold; }
    
    /* Dark alert styling */
    .alert-high {
        background-color: #2d1b1b !important;
        border-left: 4px solid #f44336 !important;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ffffff !important;
        border-radius: 8px;
    }
    
    .alert-medium {
        background-color: #2d2419 !important;
        border-left: 4px solid #ff9800 !important;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ffffff !important;
        border-radius: 8px;
    }
    
    .alert-low {
        background-color: #1b2d1b !important;
        border-left: 4px solid #4caf50 !important;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ffffff !important;
        border-radius: 8px;
    }
    
    /* Dark scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background-color: #1e2329;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #4a5568;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #4caf50;
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
            font-size: 1.5rem;
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
    
    # Help button
    if st.sidebar.button("❓ Help & Documentation", help="Access user guides, tutorials, and support resources"):
        show_help_modal()
    
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

def show_help_modal():
    """Display help modal with quick start guide"""
    st.sidebar.markdown("### 📚 AgriFlux Quick Start")
    
    with st.sidebar.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        **Welcome to AgriFlux!**
        
        1. **Select your fields** in the Monitoring Zones filter
        2. **Choose a date range** for analysis
        3. **Pick vegetation indices** to display
        4. **Navigate between pages** using the menu above
        
        💡 **Tip**: Start with the Overview page for a comprehensive summary.
        """)
    
    with st.sidebar.expander("📊 Understanding Vegetation Health"):
        st.markdown("""
        **NDVI (Normalized Difference Vegetation Index)**
        - Range: -1 to +1
        - 0.8-1.0: Excellent health 🟢
        - 0.6-0.8: Good health 🟡
        - 0.4-0.6: Moderate stress 🟠
        - <0.4: High stress 🔴
        
        **AgriFlux Color System:**
        - 🟢 **Healthy**: Thriving vegetation
        - 🟡 **Moderate**: Monitor closely
        - 🟠 **Stressed**: Investigate soon
        - 🔴 **Critical**: Immediate attention
        """)
    
    with st.sidebar.expander("🚨 Smart Alert System"):
        st.markdown("""
        **AgriFlux Alert Levels:**
        - 🔴 **Critical**: Act immediately
        - 🟠 **High**: Action within 24h
        - 🟡 **Medium**: Monitor (48h)
        - 🔵 **Info**: Awareness only
        
        **AI-Powered Alerts:**
        - Vegetation stress detection
        - Pest risk predictions
        - Irrigation optimization
        - Weather impact warnings
        """)
    
    with st.sidebar.expander("📞 Support & Resources"):
        st.markdown("""
        **Get Help:**
        - 📖 [User Guide](docs/user-guide.md)
        - 🔧 [Technical Docs](docs/technical-documentation.md)
        - 💬 Contact: support@agriflux.com
        - 📞 Phone: 1-800-AGRIFLUX
        
        **Learning Resources:**
        - Interactive tutorials
        - Weekly training webinars
        - Personalized onboarding
        """)

def display_header():
    """Display main header with key metrics in dark theme"""
    
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