#!/usr/bin/env python3
"""AgriFlux Production Dashboard - Stable Version with Modern Theme"""

import streamlit as st
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(
    page_title="🌱 AgriFlux - Smart Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Agricultural Theme - Dark with Emerald Green Accents
st.markdown("""
<style>
    /* Main background - Very dark with subtle green tint */
    .stApp {
        background: linear-gradient(135deg, #050a05 0%, #0a120a 100%) !important;
        background-attachment: fixed !important;
        color: #e8f5e9 !important;
    }
    
    /* Subtle green grid overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: 
            linear-gradient(rgba(76, 175, 80, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(76, 175, 80, 0.05) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050a05 0%, #0a120a 100%) !important;
        border-right: 2px solid #4caf50 !important;
    }
    
    /* Metrics - Emerald green gradient text */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a5d6a7 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Metric containers - Glass effect with green tint */
    [data-testid="metric-container"] {
        background: rgba(10, 18, 10, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5), 0 0 20px rgba(76, 175, 80, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: rgba(76, 175, 80, 0.6) !important;
        box-shadow: 0 8px 16px rgba(76, 175, 80, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Buttons - Green gradient */
    .stButton > button {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%) !important;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Alerts */
    .stAlert {
        background: rgba(10, 18, 10, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border-left: 4px solid #4caf50 !important;
        border-radius: 8px !important;
        color: #e8f5e9 !important;
    }
    
    .stSuccess { border-left-color: #66bb6a !important; }
    .stWarning { border-left-color: #ffa726 !important; }
    .stError { border-left-color: #ef5350 !important; }
    .stInfo { border-left-color: #4caf50 !important; }
    
    /* Headers - Green gradient */
    h1 {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    h2, h3 { color: #e8f5e9 !important; font-weight: 700 !important; }
    h4, h5, h6 { color: #c8e6c9 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        color: #a5d6a7 !important;
        background: rgba(10, 18, 10, 0.8) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%) !important;
        color: #ffffff !important;
        border-color: #4caf50 !important;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(10, 18, 10, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: rgba(76, 175, 80, 0.5) !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: rgba(10, 18, 10, 0.9) !important;
        color: #e8f5e9 !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4caf50 !important;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div > div {
        background: rgba(10, 18, 10, 0.9) !important;
        color: #e8f5e9 !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: rgba(10, 18, 10, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
    }
    
    /* Markdown text */
    .stMarkdown { color: #e8f5e9 !important; }
    
    /* Links */
    a { color: #66bb6a !important; }
    a:hover { color: #81c784 !important; }
    
    /* Scrollbar - Green theme */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0a120a; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
    }
    
    /* Dividers */
    hr { border-color: rgba(76, 175, 80, 0.3) !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_zones' not in st.session_state:
    st.session_state.selected_zones = []
if 'demo_mode' not in st.session_state:
    # Auto-enable demo mode if no real data is available
    from database.db_manager import DatabaseManager
    try:
        db = DatabaseManager()
        imagery_list = db.list_processed_imagery(limit=1)
        st.session_state.demo_mode = len(imagery_list) == 0
    except:
        st.session_state.demo_mode = True  # Enable demo mode on any error

# Auto-load demo data if demo mode is enabled
if 'demo_data' not in st.session_state and st.session_state.demo_mode:
    from dashboard.demo_manager import DemoManager
    demo_manager = DemoManager()
    if demo_manager.is_demo_data_available():
        if demo_manager.load_demo_data():
            st.session_state.demo_data = demo_manager

# Sidebar branding
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem; 
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(102, 187, 106, 0.15) 100%);
            backdrop-filter: blur(10px);
            border-radius: 12px; 
            border: 2px solid rgba(76, 175, 80, 0.4); 
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);">
    <h2 style="background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               margin: 0; font-weight: 800;">🌱 AgriFlux</h2>
    <p style="color: #a5d6a7; margin: 0.5rem 0 0 0; font-weight: 500;">Smart Agricultural Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Page navigation
page = st.sidebar.selectbox(
    "Navigate to:",
    ["📊 Overview", "🗺️ Field Monitoring", "📈 Temporal Analysis", 
     "🚨 Alerts & Notifications", "🤖 AI Model Performance", "📤 Data Export", "📚 Documentation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Fields", "12")
with col2:
    st.metric("Alerts", "6")

st.sidebar.markdown("---")
st.sidebar.caption("AgriFlux v1.0.0 🎉")

# Main header
st.markdown('''
<h1 style="text-align: center; font-size: 3rem; margin-bottom: 0.5rem;">
    🌱 AgriFlux Intelligence Dashboard
</h1>
<p style="text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;">
    Real-time Agricultural Monitoring & AI-Powered Analytics
</p>
''', unsafe_allow_html=True)

# Quick metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🗺️ Active Fields", "12", "+2")
with col2:
    st.metric("🚨 Alerts", "6", "-1")
with col3:
    st.metric("🌱 Health Index", "0.75", "+0.05")
with col4:
    st.metric("📡 Data Quality", "94%", "+2%")

st.markdown("---")

# Route to pages with error handling
try:
    if "Overview" in page:
        from utils.error_handler import setup_logging, logger
        from dashboard.pages import overview
        setup_logging()
        logger.info("Loading Overview page...")
        overview.show_page()
        
    elif "Field Monitoring" in page:
        from utils.error_handler import setup_logging, logger
        from dashboard.pages import field_monitoring
        setup_logging()
        logger.info("Loading Field Monitoring page...")
        field_monitoring.show_page()
        
    elif "Temporal Analysis" in page:
        from utils.error_handler import setup_logging, logger
        from dashboard.pages import temporal_analysis
        setup_logging()
        logger.info("Loading Temporal Analysis page...")
        temporal_analysis.show_page()
        
    elif "Alerts" in page:
        from utils.error_handler import setup_logging, logger
        from dashboard.pages import alerts
        setup_logging()
        logger.info("Loading Alerts page...")
        alerts.show_page()
        
    elif "Model Performance" in page:
        from utils.error_handler import setup_logging, logger
        from dashboard.pages import model_performance
        setup_logging()
        logger.info("Loading Model Performance page...")
        model_performance.show_page()
        
    elif "Data Export" in page:
        from utils.error_handler import setup_logging, logger
        from dashboard.pages import data_export
        setup_logging()
        logger.info("Loading Data Export page...")
        data_export.show_page()
        
    elif "Documentation" in page:
        from utils.error_handler import setup_logging, logger
        from dashboard.pages import documentation
        setup_logging()
        logger.info("Loading Documentation page...")
        documentation.show_page()
        
except Exception as e:
    st.error(f"❌ Error loading page: {str(e)}")
    st.exception(e)
    st.info("💡 Try refreshing the page or selecting a different section.")
