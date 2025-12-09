#!/usr/bin/env python3
"""
AgriFlux Dashboard Runner - Production Version
Runs the full production dashboard with all features
"""

import streamlit as st
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="🌱 AgriFlux - Smart Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after config
from utils.error_handler import setup_logging, logger
from dashboard.pages import overview, field_monitoring, temporal_analysis, alerts, data_export, model_performance

# Try to import custom theme
try:
    from dashboard.ui_components import apply_custom_theme
    apply_custom_theme()
except:
    pass

# Set up logging
setup_logging()
logger.info("AgriFlux Dashboard starting...")

# Apply custom CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #1a1d29;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #252936;
    }
    .metric-container {
        background-color: #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid #4a5568;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_zones' not in st.session_state:
    st.session_state.selected_zones = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False
if 'date_range' not in st.session_state:
    st.session_state.date_range = (datetime.now() - timedelta(days=30), datetime.now())
if 'selected_indices' not in st.session_state:
    st.session_state.selected_indices = ['NDVI', 'SAVI']
if 'acknowledged_alerts' not in st.session_state:
    st.session_state.acknowledged_alerts = set()
if 'map_center' not in st.session_state:
    st.session_state.map_center = [31.1, 75.81]  # Ludhiana coordinates
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 12

# Sidebar with branding
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%); 
            border-radius: 10px; border: 1px solid #4caf50; margin-bottom: 1rem;">
    <h2 style="color: #4caf50; margin: 0;">🌱 AgriFlux</h2>
    <p style="color: #a0aec0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Smart Agricultural Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Help section
if st.sidebar.button("❓ Help & Documentation"):
    st.session_state.show_help = not st.session_state.get('show_help', False)

if st.session_state.get('show_help', False):
    with st.sidebar.expander("🚀 Getting Started", expanded=True):
        st.markdown("1. Select fields\n2. Choose date range\n3. Pick indices\n4. Navigate pages")
    with st.sidebar.expander("📊 NDVI Values"):
        st.markdown("• 0.8-1.0: Excellent 🟢\n• 0.6-0.8: Healthy 🟢\n• 0.4-0.6: Moderate 🟡\n• <0.4: Stressed 🔴")

st.sidebar.markdown("---")

# Page navigation
page = st.sidebar.selectbox(
    "Navigate to:",
    ["📊 Overview", "🗺️ Field Monitoring", "📈 Temporal Analysis", 
     "🚨 Alerts & Notifications", "🤖 AI Model Performance", "📤 Data Export"],
    help="Select a page to navigate to different features"
)

# System status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Fields", "12", "+2")
with col2:
    st.metric("Alerts", "3", "-1")

st.sidebar.markdown("---")
st.sidebar.caption("AgriFlux v1.0.0 - Production Ready 🎉")

# Main content header
st.markdown('<h1 style="text-align: center; color: #4caf50;">🌱 AgriFlux Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #a0aec0; margin-bottom: 2rem;">Real-time Agricultural Monitoring & Analysis</p>', unsafe_allow_html=True)

# Quick metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🗺️ Active Fields", "12", "+2")
with col2:
    st.metric("🚨 Alerts", "3", "-1")
with col3:
    st.metric("🌱 Avg Health", "0.75", "+0.05")
with col4:
    st.metric("📡 Data Quality", "94%", "+2%")

st.markdown("---")

# Route to pages with error handling
try:
    if "Overview" in page:
        overview.show_page()
    elif "Field Monitoring" in page:
        field_monitoring.show_page()
    elif "Temporal Analysis" in page:
        temporal_analysis.show_page()
    elif "Alerts" in page:
        alerts.show_page()
    elif "Model Performance" in page:
        model_performance.show_page()
    elif "Data Export" in page:
        data_export.show_page()
except Exception as e:
    st.error(f"❌ Error loading page: {str(e)}")
    st.exception(e)
    logger.error(f"Page error: {e}", exc_info=True)
    
    # Show recovery options
    st.info("💡 Try refreshing the page or selecting a different page from the sidebar.")
