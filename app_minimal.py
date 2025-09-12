#!/usr/bin/env python3
"""
🌱 AgriFlux - Minimal Version for Guaranteed Deployment
Ultra-lightweight version that works on any platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="🌱 AgriFlux - Agricultural Intelligence",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .metric-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #4a5568;
    }
    .metric-title { color: #4caf50; font-weight: bold; }
    .metric-value { color: #ffffff; font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Header
    st.markdown("# 🌱 AgriFlux - Smart Agricultural Intelligence")
    st.markdown("**Real-time crop health monitoring using satellite imagery and AI**")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Fields", "5", delta="2 new")
    
    with col2:
        st.metric("Health Index", "0.72", delta="0.05")
    
    with col3:
        st.metric("Smart Alerts", "7", delta="-3")
    
    with col4:
        st.metric("Data Quality", "94%", delta="2%")
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌱 Vegetation Health Overview")
        
        # Sample data
        zones = ['North Field A', 'South Field B', 'East Pasture C', 'West Orchard D', 'Central Plot E']
        ndvi_values = [0.75, 0.68, 0.82, 0.71, 0.59]
        
        # Create chart
        fig = go.Figure()
        colors = ['#4CAF50' if v >= 0.7 else '#FF9800' if v >= 0.5 else '#F44336' for v in ndvi_values]
        
        fig.add_trace(go.Bar(
            x=zones,
            y=ndvi_values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in ndvi_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="NDVI by Zone",
            xaxis_title="Monitoring Zones",
            yaxis_title="NDVI Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend chart
        st.subheader("📈 30-Day Trend")
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        trend_data = []
        
        for i, date in enumerate(dates):
            ndvi = 0.7 + 0.1 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 0.02)
            trend_data.append({'Date': date, 'NDVI': max(0, min(1, ndvi))})
        
        df = pd.DataFrame(trend_data)
        
        fig2 = px.line(df, x='Date', y='NDVI', title='Average NDVI Trend')
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Smart Alerts")
        
        alerts = [
            {"type": "Vegetation Stress", "zone": "Central Plot E", "severity": "High"},
            {"type": "Pest Risk", "zone": "South Field B", "severity": "Medium"},
            {"type": "Low Soil Moisture", "zone": "North Field A", "severity": "Medium"},
        ]
        
        for alert in alerts:
            severity_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            st.markdown(f"""
            <div class="metric-card">
                {severity_icon[alert['severity']]} <strong>{alert['type']}</strong><br>
                📍 {alert['zone']}<br>
                ⚠️ {alert['severity']} Priority
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("🌤️ Weather Summary")
        
        weather = {
            "Temperature": "24.5°C",
            "Humidity": "72%",
            "Wind Speed": "12 km/h",
            "Precipitation": "2.3mm"
        }
        
        for metric, value in weather.items():
            st.markdown(f"**{metric}:** {value}")
    
    # Zone comparison table
    st.subheader("🗺️ Zone Comparison")
    
    comparison_data = {
        'Zone': zones,
        'Area (ha)': [245, 312, 189, 278, 223],
        'NDVI': ndvi_values,
        'Soil Moisture (%)': [72, 65, 78, 69, 58],
        'Health Status': ['Healthy', 'Healthy', 'Excellent', 'Healthy', 'Stressed'],
        'Active Alerts': [1, 1, 0, 0, 2]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**🌱 AgriFlux** - Empowering farmers with AI-driven agricultural intelligence")
    st.markdown("*Sample data from Punjab agricultural zones, India*")

if __name__ == "__main__":
    main()