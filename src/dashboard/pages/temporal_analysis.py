"""
Temporal Analysis page - Time series visualization and trend analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def show_page():
    """Display the temporal analysis page"""
    
    st.title("📈 Temporal Analysis")
    st.markdown("Time series visualization and trend analysis for vegetation indices")
    
    # Analysis controls
    display_analysis_controls()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_time_series_chart()
        display_comparison_chart()
    
    with col2:
        display_statistics_panel()
        display_trend_analysis()

def display_analysis_controls():
    """Display analysis control panel"""
    
    st.subheader("🎛️ Analysis Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "Custom"],
            key="temporal_time_period"
        )
    
    with col2:
        selected_zones = st.multiselect(
            "Zones to Analyze",
            ["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"],
            default=["North Field A", "South Field B"],
            key="temporal_zones"
        )
    
    with col3:
        selected_indices = st.multiselect(
            "Vegetation Indices",
            ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"],
            default=["NDVI", "SAVI"],
            key="temporal_indices"
        )
    
    with col4:
        aggregation = st.selectbox(
            "Data Aggregation",
            ["Daily", "Weekly", "Monthly"],
            key="temporal_aggregation"
        )
    
    # Custom date range if selected
    if time_period == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=90),
                key="temporal_start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=datetime.now(),
                key="temporal_end_date"
            )

def display_time_series_chart():
    """Display main time series chart"""
    
    st.subheader("📊 Vegetation Index Time Series")
    
    # Generate mock time series data
    df = generate_time_series_data()
    
    # Filter data based on selections
    selected_zones = st.session_state.get('temporal_zones', ['North Field A', 'South Field B'])
    selected_indices = st.session_state.get('temporal_indices', ['NDVI', 'SAVI'])
    
    if not selected_zones or not selected_indices:
        st.warning("Please select at least one zone and one vegetation index.")
        return
    
    # Create subplot for each index
    fig = make_subplots(
        rows=len(selected_indices),
        cols=1,
        subplot_titles=selected_indices,
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, index in enumerate(selected_indices):
        for j, zone in enumerate(selected_zones):
            zone_data = df[(df['Zone'] == zone) & (df['Index'] == index)]
            
            if not zone_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=zone_data['Date'],
                        y=zone_data['Value'],
                        mode='lines+markers',
                        name=f"{zone} - {index}",
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 0),  # Only show legend for first subplot
                        hovertemplate=f"<b>{zone}</b><br>" +
                                    f"{index}: %{{y:.3f}}<br>" +
                                    "Date: %{x}<br>" +
                                    "<extra></extra>"
                    ),
                    row=i+1, col=1
                )
                
                # Add confidence intervals
                if 'Upper_CI' in zone_data.columns and 'Lower_CI' in zone_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=zone_data['Date'].tolist() + zone_data['Date'].tolist()[::-1],
                            y=zone_data['Upper_CI'].tolist() + zone_data['Lower_CI'].tolist()[::-1],
                            fill='toself',
                            fillcolor=colors[j % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.2)'),
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            hoverinfo="skip"
                        ),
                        row=i+1, col=1
                    )
    
    fig.update_layout(
        height=300 * len(selected_indices),
        title="Vegetation Index Trends with Confidence Intervals",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=len(selected_indices), col=1)
    
    for i in range(len(selected_indices)):
        fig.update_yaxes(title_text="Index Value", row=i+1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_comparison_chart():
    """Display zone comparison chart"""
    
    st.subheader("🔄 Zone Comparison")
    
    # Generate comparison data
    comparison_data = generate_comparison_data()
    
    selected_zones = st.session_state.get('temporal_zones', ['North Field A', 'South Field B'])
    selected_index = st.selectbox(
        "Index for Comparison",
        st.session_state.get('temporal_indices', ['NDVI', 'SAVI']),
        key="comparison_index"
    )
    
    if not selected_index:
        return
    
    # Filter data
    filtered_data = comparison_data[
        (comparison_data['Zone'].isin(selected_zones)) & 
        (comparison_data['Index'] == selected_index)
    ]
    
    if filtered_data.empty:
        st.warning("No data available for selected zones and index.")
        return
    
    # Create comparison chart
    fig = px.box(
        filtered_data,
        x='Zone',
        y='Value',
        color='Zone',
        title=f"{selected_index} Distribution by Zone (Last 30 Days)",
        points="all"
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_statistics_panel():
    """Display statistics panel"""
    
    st.subheader("📊 Statistics")
    
    # Generate statistics
    stats = calculate_statistics()
    
    selected_zones = st.session_state.get('temporal_zones', ['North Field A', 'South Field B'])
    
    for zone in selected_zones:
        if zone in stats:
            st.markdown(f"**{zone}:**")
            zone_stats = stats[zone]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Avg NDVI",
                    f"{zone_stats['ndvi_mean']:.3f}",
                    delta=f"{zone_stats['ndvi_change']:+.3f}"
                )
                
                st.metric(
                    "Std Dev",
                    f"{zone_stats['ndvi_std']:.3f}"
                )
            
            with col2:
                st.metric(
                    "Trend",
                    zone_stats['trend'],
                    delta=f"{zone_stats['trend_strength']:.1f}%"
                )
                
                st.metric(
                    "R²",
                    f"{zone_stats['r_squared']:.3f}"
                )
            
            st.markdown("---")

def display_trend_analysis():
    """Display trend analysis results"""
    
    st.subheader("📈 Trend Analysis")
    
    # Mock trend analysis results
    trends = {
        "North Field A": {
            "direction": "Increasing",
            "strength": "Moderate",
            "significance": "p < 0.05",
            "seasonal": "Yes"
        },
        "South Field B": {
            "direction": "Stable", 
            "strength": "Weak",
            "significance": "p > 0.05",
            "seasonal": "No"
        },
        "East Pasture C": {
            "direction": "Increasing",
            "strength": "Strong", 
            "significance": "p < 0.01",
            "seasonal": "Yes"
        },
        "West Orchard D": {
            "direction": "Decreasing",
            "strength": "Weak",
            "significance": "p > 0.05", 
            "seasonal": "No"
        },
        "Central Plot E": {
            "direction": "Decreasing",
            "strength": "Strong",
            "significance": "p < 0.01",
            "seasonal": "No"
        }
    }
    
    selected_zones = st.session_state.get('temporal_zones', ['North Field A', 'South Field B'])
    
    for zone in selected_zones:
        if zone in trends:
            trend = trends[zone]
            
            # Direction indicator
            direction_icon = {
                "Increasing": "📈",
                "Decreasing": "📉", 
                "Stable": "➡️"
            }
            
            st.markdown(f"**{zone}:**")
            st.markdown(f"{direction_icon[trend['direction']]} {trend['direction']} ({trend['strength']})")
            st.markdown(f"🔬 {trend['significance']}")
            st.markdown(f"🌊 Seasonal: {trend['seasonal']}")
            st.markdown("---")
    
    # Export trend analysis
    if st.button("📊 Export Trend Analysis", key="export_trends"):
        st.success("Trend analysis exported to CSV!")

def generate_time_series_data():
    """Generate mock time series data"""
    
    zones = ["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"]
    indices = ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"]
    
    # Generate 90 days of data
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
    
    data = []
    
    for zone in zones:
        for index in indices:
            # Base values for different indices
            base_values = {
                "NDVI": 0.7,
                "SAVI": 0.65,
                "EVI": 0.6,
                "NDWI": 0.3,
                "NDSI": 0.4
            }
            
            base_value = base_values[index]
            
            for i, date in enumerate(dates):
                # Add seasonal trend
                seasonal = 0.1 * np.sin(2 * np.pi * i / 365)
                
                # Add zone-specific variation
                zone_factor = {
                    "North Field A": 1.0,
                    "South Field B": 0.95,
                    "East Pasture C": 1.1,
                    "West Orchard D": 1.05,
                    "Central Plot E": 0.85
                }[zone]
                
                # Add noise
                noise = np.random.normal(0, 0.02)
                
                # Calculate value
                value = base_value * zone_factor + seasonal + noise
                value = max(0, min(1, value))  # Clamp to valid range
                
                # Add confidence intervals
                ci_width = 0.05
                upper_ci = min(1, value + ci_width)
                lower_ci = max(0, value - ci_width)
                
                data.append({
                    'Date': date,
                    'Zone': zone,
                    'Index': index,
                    'Value': value,
                    'Upper_CI': upper_ci,
                    'Lower_CI': lower_ci
                })
    
    return pd.DataFrame(data)

def generate_comparison_data():
    """Generate mock comparison data for box plots"""
    
    zones = ["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"]
    indices = ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"]
    
    data = []
    
    for zone in zones:
        for index in indices:
            # Generate 30 random samples for each zone/index combination
            base_values = {
                "NDVI": 0.7,
                "SAVI": 0.65, 
                "EVI": 0.6,
                "NDWI": 0.3,
                "NDSI": 0.4
            }
            
            base_value = base_values[index]
            
            zone_factor = {
                "North Field A": 1.0,
                "South Field B": 0.95,
                "East Pasture C": 1.1,
                "West Orchard D": 1.05,
                "Central Plot E": 0.85
            }[zone]
            
            for _ in range(30):
                noise = np.random.normal(0, 0.05)
                value = base_value * zone_factor + noise
                value = max(0, min(1, value))
                
                data.append({
                    'Zone': zone,
                    'Index': index,
                    'Value': value
                })
    
    return pd.DataFrame(data)

def calculate_statistics():
    """Calculate statistics for each zone"""
    
    # Mock statistics
    stats = {
        "North Field A": {
            "ndvi_mean": 0.752,
            "ndvi_std": 0.045,
            "ndvi_change": 0.023,
            "trend": "Increasing",
            "trend_strength": 65.4,
            "r_squared": 0.734
        },
        "South Field B": {
            "ndvi_mean": 0.684,
            "ndvi_std": 0.038,
            "ndvi_change": -0.012,
            "trend": "Stable",
            "trend_strength": 23.1,
            "r_squared": 0.234
        },
        "East Pasture C": {
            "ndvi_mean": 0.823,
            "ndvi_std": 0.029,
            "ndvi_change": 0.045,
            "trend": "Increasing",
            "trend_strength": 87.2,
            "r_squared": 0.856
        },
        "West Orchard D": {
            "ndvi_mean": 0.715,
            "ndvi_std": 0.041,
            "ndvi_change": -0.008,
            "trend": "Stable",
            "trend_strength": 18.7,
            "r_squared": 0.187
        },
        "Central Plot E": {
            "ndvi_mean": 0.592,
            "ndvi_std": 0.052,
            "ndvi_change": -0.034,
            "trend": "Decreasing",
            "trend_strength": 78.9,
            "r_squared": 0.789
        }
    }
    
    return stats