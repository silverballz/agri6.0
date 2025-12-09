#!/usr/bin/env python3
"""AgriFlux Production Dashboard - Working Version"""

import streamlit as st
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(
    page_title="🌱 AgriFlux - Smart Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports
from utils.error_handler import setup_logging, logger
from dashboard.pages import overview, field_monitoring, temporal_analysis, alerts, data_export, model_performance

setup_logging()
logger.info("AgriFlux Dashboard starting...")

# Modern Professional Theme - Dark with Teal/Cyan Accents
st.markdown("""
<style>
    /* Main background - Dark slate with subtle grid */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        background-attachment: fixed !important;
        position: relative;
        color: #e2e8f0 !important;
    }
    
    /* Subtle grid overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(6, 182, 212, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(6, 182, 212, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Sidebar - Darker with cyan accent */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 2px solid #06b6d4 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: transparent !important;
    }
    
    /* Metrics - Cyan/Teal gradient */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #06b6d4 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #10b981 !important;
    }
    
    /* Metric containers with glass effect */
    [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(6, 182, 212, 0.2) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: rgba(6, 182, 212, 0.5) !important;
        box-shadow: 0 8px 16px rgba(6, 182, 212, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Alert/Info boxes */
    .stAlert {
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-left: 4px solid #06b6d4 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Success alerts */
    .stSuccess {
        border-left-color: #10b981 !important;
    }
    
    /* Warning alerts */
    .stWarning {
        border-left-color: #f59e0b !important;
    }
    
    /* Error alerts */
    .stError {
        border-left-color: #ef4444 !important;
    }
    
    /* Headers with gradient */
    h1 {
        background: linear-gradient(135deg, #06b6d4 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
    }
    
    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    
    h4, h5, h6 {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
    
    /* Expanders - Glass morphism */
    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(6, 182, 212, 0.2) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: rgba(6, 182, 212, 0.4) !important;
    }
    
    /* Buttons - Cyan gradient */
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Dataframes and tables */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(6, 182, 212, 0.2) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
        color: #ffffff !important;
        border-color: #06b6d4 !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Links */
    a {
        color: #06b6d4 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #0891b2 !important;
        text-decoration: underline !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
    }
    
    /* Container spacing */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(6, 182, 212, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_zones' not in st.session_state:
    st.session_state.selected_zones = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem; 
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%);
            backdrop-filter: blur(10px);
            border-radius: 12px; 
            border: 2px solid rgba(6, 182, 212, 0.3); 
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.2);">
    <h2 style="background: linear-gradient(135deg, #06b6d4 0%, #14b8a6 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               margin: 0; 
               font-weight: 800;">🌱 AgriFlux</h2>
    <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-weight: 500;">Smart Agricultural Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Page navigation
page = st.sidebar.selectbox(
    "Navigate to:",
    ["📊 Overview", "🗺️ Field Monitoring", "📈 Temporal Analysis", 
     "🚨 Alerts & Notifications", "🤖 AI Model Performance", "📤 Data Export"]
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
<h1 style="text-align: center; 
           background: linear-gradient(135deg, #06b6d4 0%, #14b8a6 100%);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           background-clip: text;
           font-weight: 800;
           font-size: 3rem;
           margin-bottom: 0.5rem;">
    🌱 AgriFlux Intelligence Dashboard
</h1>
<p style="text-align: center; 
          color: #94a3b8; 
          font-size: 1.1rem;
          margin-bottom: 2rem;">
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

# Route to pages
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
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)
