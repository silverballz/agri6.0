"""
Overview page - Main dashboard with summary metrics and key insights
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def show_page():
    """Display the overview page"""
    
    st.title("🌱 AgriFlux Overview")
    st.markdown("Real-time insights and AI-powered analytics across all monitoring zones")
    
    # Quick stats row
    display_quick_stats()
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_health_summary_chart()
        display_recent_trends()
    
    with col2:
        display_alert_summary()
        display_weather_summary()
    
    # Bottom section - zone comparison
    display_zone_comparison()

def display_quick_stats():
    """Display quick statistics cards"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Area",
            "1,247 ha",
            help="Total monitored agricultural area"
        )
    
    with col2:
        st.metric(
            "Healthy Zones",
            "4/5",
            delta="Same as yesterday",
            help="Zones with NDVI > 0.6"
        )
    
    with col3:
        st.metric(
            "Avg Temperature",
            "24.5°C",
            delta="1.2°C",
            help="Average temperature across all zones"
        )
    
    with col4:
        st.metric(
            "Soil Moisture",
            "68%",
            delta="-5%",
            delta_color="inverse",
            help="Average soil moisture level"
        )
    
    with col5:
        st.metric(
            "Yield Forecast",
            "94%",
            delta="2%",
            help="Predicted yield vs. historical average"
        )

def display_health_summary_chart():
    """Display vegetation health summary chart"""
    
    st.subheader("🌱 Vegetation Health Summary")
    
    # Mock data for demonstration
    zones = ['North Field A', 'South Field B', 'East Pasture C', 'West Orchard D', 'Central Plot E']
    ndvi_values = [0.75, 0.68, 0.82, 0.71, 0.59]
    savi_values = [0.71, 0.64, 0.78, 0.67, 0.55]
    
    # Create health status based on NDVI
    health_status = []
    colors = []
    for ndvi in ndvi_values:
        if ndvi >= 0.7:
            health_status.append('Healthy')
            colors.append('#4CAF50')
        elif ndvi >= 0.5:
            health_status.append('Moderate')
            colors.append('#FF9800')
        else:
            health_status.append('Stressed')
            colors.append('#F44336')
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='NDVI',
        x=zones,
        y=ndvi_values,
        marker_color=colors,
        text=[f'{v:.2f}' for v in ndvi_values],
        textposition='auto',
    ))
    
    fig.add_trace(go.Scatter(
        name='SAVI',
        x=zones,
        y=savi_values,
        mode='markers+lines',
        marker=dict(size=8, color='blue'),
        line=dict(color='blue', dash='dash')
    ))
    
    fig.update_layout(
        title="Vegetation Indices by Zone",
        xaxis_title="Monitoring Zones",
        yaxis_title="Index Value",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_recent_trends():
    """Display recent trend analysis"""
    
    st.subheader("📈 Recent Trends (Last 30 Days)")
    
    # Generate mock time series data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    # Mock NDVI trend with some seasonal variation
    base_ndvi = 0.7
    trend_data = []
    
    for i, date in enumerate(dates):
        # Add seasonal variation and some noise
        seasonal = 0.1 * np.sin(2 * np.pi * i / 30)
        noise = np.random.normal(0, 0.02)
        ndvi = base_ndvi + seasonal + noise
        trend_data.append({
            'Date': date,
            'NDVI': max(0, min(1, ndvi)),
            'Zone': 'Average'
        })
    
    df = pd.DataFrame(trend_data)
    
    fig = px.line(
        df, 
        x='Date', 
        y='NDVI',
        title='Average NDVI Trend',
        color_discrete_sequence=['#2E8B57']
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_title="Date",
        yaxis_title="NDVI Value"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_alert_summary():
    """Display alert summary widget"""
    
    st.subheader("🚨 Smart Alerts")
    
    # Mock alert data
    alerts = [
        {"type": "Vegetation Stress", "zone": "Central Plot E", "severity": "High", "time": "2 hours ago"},
        {"type": "Pest Risk", "zone": "South Field B", "severity": "Medium", "time": "5 hours ago"},
        {"type": "Low Soil Moisture", "zone": "North Field A", "severity": "Medium", "time": "1 day ago"},
    ]
    
    for alert in alerts:
        severity_color = {
            "High": "🔴",
            "Medium": "🟡", 
            "Low": "🟢"
        }
        
        with st.container():
            st.markdown(f"""
            <div class="alert-{alert['severity'].lower()}">
                {severity_color[alert['severity']]} <strong>{alert['type']}</strong><br>
                📍 {alert['zone']}<br>
                🕒 {alert['time']}
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("View All Alerts", key="view_all_alerts"):
        st.session_state.page_selector = "🚨 Alerts & Notifications"
        st.rerun()

def display_weather_summary():
    """Display weather summary widget"""
    
    st.subheader("🌤️ Weather Summary")
    
    # Mock weather data
    weather_data = {
        "Temperature": "24.5°C",
        "Humidity": "72%",
        "Wind Speed": "12 km/h",
        "Precipitation": "2.3mm (today)",
        "UV Index": "6 (High)"
    }
    
    for metric, value in weather_data.items():
        st.markdown(f"**{metric}:** {value}")
    
    # Weather forecast
    st.markdown("**3-Day Forecast:**")
    forecast = ["☀️ 26°C", "🌤️ 23°C", "🌧️ 19°C"]
    st.markdown(" | ".join(forecast))

def display_zone_comparison():
    """Display zone comparison table"""
    
    st.subheader("🗺️ Zone Comparison")
    
    # Mock comparison data
    comparison_data = {
        'Zone': ['North Field A', 'South Field B', 'East Pasture C', 'West Orchard D', 'Central Plot E'],
        'Area (ha)': [245, 312, 189, 278, 223],
        'NDVI': [0.75, 0.68, 0.82, 0.71, 0.59],
        'SAVI': [0.71, 0.64, 0.78, 0.67, 0.55],
        'Soil Moisture (%)': [72, 65, 78, 69, 58],
        'Health Status': ['Healthy', 'Healthy', 'Excellent', 'Healthy', 'Stressed'],
        'Active Alerts': [1, 1, 0, 0, 2]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Style the dataframe
    def style_health_status(val):
        if val == 'Excellent':
            return 'background-color: #c8e6c9'
        elif val == 'Healthy':
            return 'background-color: #e8f5e8'
        elif val == 'Stressed':
            return 'background-color: #ffcdd2'
        return ''
    
    def style_alerts(val):
        if val > 1:
            return 'background-color: #ffcdd2; font-weight: bold'
        elif val == 1:
            return 'background-color: #fff3e0'
        return 'background-color: #e8f5e8'
    
    styled_df = df.style.applymap(style_health_status, subset=['Health Status']) \
                       .applymap(style_alerts, subset=['Active Alerts']) \
                       .format({
                           'NDVI': '{:.2f}',
                           'SAVI': '{:.2f}',
                           'Area (ha)': '{:.0f}'
                       })
    
    st.dataframe(styled_df, use_container_width=True)