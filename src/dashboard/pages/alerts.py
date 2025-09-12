"""
Alerts and Notifications page - Alert management and notification system
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def show_page():
    """Display the alerts and notifications page"""
    
    st.title("🚨 Alerts & Notifications")
    st.markdown("Monitor and manage active alerts and system notifications")
    
    # Alert summary metrics
    display_alert_metrics()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_active_alerts()
        display_alert_history()
    
    with col2:
        display_alert_filters()
        display_alert_analytics()

def display_alert_metrics():
    """Display alert summary metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Alerts",
            "7",
            delta="-3 from yesterday",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "High Priority",
            "2",
            delta="+1 from yesterday",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Avg Response Time",
            "2.3 hrs",
            delta="-0.5 hrs",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Resolution Rate",
            "94%",
            delta="+2%",
            delta_color="inverse"
        )

def display_active_alerts():
    """Display active alerts table"""
    
    st.subheader("🔴 Active Alerts")
    
    # Generate mock active alerts
    active_alerts = generate_active_alerts()
    
    if active_alerts.empty:
        st.info("No active alerts at this time.")
        return
    
    # Alert filters from session state
    severity_filter = st.session_state.get('alert_severity_filter', ['High', 'Medium', 'Low'])
    type_filter = st.session_state.get('alert_type_filter', ['Vegetation Stress', 'Pest Risk', 'Disease Risk', 'Environmental'])
    
    # Filter alerts
    filtered_alerts = active_alerts[
        (active_alerts['Severity'].isin(severity_filter)) &
        (active_alerts['Type'].isin(type_filter))
    ]
    
    # Display alerts as cards
    for idx, alert in filtered_alerts.iterrows():
        display_alert_card(alert, idx)
    
    # Bulk actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Acknowledge All", key="ack_all_alerts"):
            st.success("All visible alerts acknowledged!")
    
    with col2:
        if st.button("📧 Send Summary", key="send_summary"):
            st.success("Alert summary sent to stakeholders!")
    
    with col3:
        if st.button("🔄 Refresh Alerts", key="refresh_alerts"):
            st.rerun()

def display_alert_card(alert, idx):
    """Display individual alert card"""
    
    # Severity styling
    severity_styles = {
        'High': {'color': '#f44336', 'bg': '#ffebee', 'icon': '🔴'},
        'Medium': {'color': '#ff9800', 'bg': '#fff3e0', 'icon': '🟡'},
        'Low': {'color': '#4caf50', 'bg': '#e8f5e8', 'icon': '🟢'}
    }
    
    style = severity_styles.get(alert['Severity'], severity_styles['Medium'])
    
    # Check if alert is acknowledged
    alert_id = f"alert_{idx}"
    is_acknowledged = alert_id in st.session_state.get('acknowledged_alerts', set())
    
    # Card container
    with st.container():
        st.markdown(f"""
        <div style="
            border: 2px solid {style['color']};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: {style['bg']};
            opacity: {'0.6' if is_acknowledged else '1.0'};
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {style['color']};">
                    {style['icon']} {alert['Type']}
                </h4>
                <span style="
                    background-color: {style['color']};
                    color: white;
                    padding: 4px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                ">
                    {alert['Severity']}
                </span>
            </div>
            <p style="margin: 10px 0;"><strong>Zone:</strong> {alert['Zone']}</p>
            <p style="margin: 10px 0;"><strong>Description:</strong> {alert['Description']}</p>
            <p style="margin: 10px 0;"><strong>Time:</strong> {alert['Time']}</p>
            <p style="margin: 10px 0;"><strong>Recommendation:</strong> {alert['Recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not is_acknowledged:
                if st.button("✅ Acknowledge", key=f"ack_{idx}"):
                    if 'acknowledged_alerts' not in st.session_state:
                        st.session_state.acknowledged_alerts = set()
                    st.session_state.acknowledged_alerts.add(alert_id)
                    st.rerun()
            else:
                st.success("Acknowledged")
        
        with col2:
            if st.button("📍 View on Map", key=f"map_{idx}"):
                st.session_state.page_selector = "🗺️ Field Monitoring"
                st.session_state.selected_zone = alert['Zone']
                st.rerun()
        
        with col3:
            if st.button("📊 View Trends", key=f"trends_{idx}"):
                st.session_state.page_selector = "📈 Temporal Analysis"
                st.rerun()
        
        with col4:
            if st.button("❌ Dismiss", key=f"dismiss_{idx}"):
                st.warning(f"Alert {idx+1} dismissed")

def display_alert_history():
    """Display alert history and trends"""
    
    st.subheader("📜 Alert History")
    
    # Time period selector
    history_period = st.selectbox(
        "History Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
        key="alert_history_period"
    )
    
    # Generate historical data
    history_data = generate_alert_history()
    
    # Alert trend chart
    fig = px.line(
        history_data,
        x='Date',
        y='Count',
        color='Severity',
        title="Alert Trends Over Time",
        color_discrete_map={
            'High': '#f44336',
            'Medium': '#ff9800', 
            'Low': '#4caf50'
        }
    )
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert type distribution
    type_data = history_data.groupby(['Type', 'Severity'])['Count'].sum().reset_index()
    
    fig2 = px.bar(
        type_data,
        x='Type',
        y='Count',
        color='Severity',
        title="Alert Distribution by Type",
        color_discrete_map={
            'High': '#f44336',
            'Medium': '#ff9800',
            'Low': '#4caf50'
        }
    )
    
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

def display_alert_filters():
    """Display alert filtering controls"""
    
    st.subheader("🔧 Alert Filters")
    
    # Severity filter
    severity_options = ['High', 'Medium', 'Low']
    selected_severity = st.multiselect(
        "Severity Levels",
        severity_options,
        default=st.session_state.get('alert_severity_filter', severity_options),
        key="alert_severity_filter"
    )
    
    # Type filter
    type_options = ['Vegetation Stress', 'Pest Risk', 'Disease Risk', 'Environmental']
    selected_types = st.multiselect(
        "Alert Types",
        type_options,
        default=st.session_state.get('alert_type_filter', type_options),
        key="alert_type_filter"
    )
    
    # Zone filter
    zone_options = ['North Field A', 'South Field B', 'East Pasture C', 'West Orchard D', 'Central Plot E']
    selected_zones = st.multiselect(
        "Zones",
        zone_options,
        default=st.session_state.get('alert_zone_filter', []),
        key="alert_zone_filter"
    )
    
    # Time filter
    time_filter = st.selectbox(
        "Time Range",
        ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        key="alert_time_filter"
    )
    
    # Acknowledgment filter
    ack_filter = st.selectbox(
        "Acknowledgment Status",
        ["All", "Acknowledged", "Unacknowledged"],
        key="alert_ack_filter"
    )
    
    st.markdown("---")
    
    # Notification settings
    st.subheader("🔔 Notification Settings")
    
    email_notifications = st.checkbox(
        "Email Notifications",
        value=True,
        key="email_notifications"
    )
    
    sms_notifications = st.checkbox(
        "SMS Notifications",
        value=False,
        key="sms_notifications"
    )
    
    push_notifications = st.checkbox(
        "Push Notifications",
        value=True,
        key="push_notifications"
    )
    
    # Notification frequency
    notification_frequency = st.selectbox(
        "Notification Frequency",
        ["Immediate", "Every 15 minutes", "Hourly", "Daily"],
        key="notification_frequency"
    )

def display_alert_analytics():
    """Display alert analytics and insights"""
    
    st.subheader("📊 Alert Analytics")
    
    # Response time metrics
    st.markdown("**Response Time Analysis:**")
    
    response_metrics = {
        "Average Response": "2.3 hours",
        "Fastest Response": "15 minutes",
        "Slowest Response": "8.2 hours",
        "Target Response": "< 4 hours"
    }
    
    for metric, value in response_metrics.items():
        st.markdown(f"- **{metric}:** {value}")
    
    st.markdown("---")
    
    # Alert patterns
    st.markdown("**Alert Patterns:**")
    
    patterns = [
        "🕐 Peak alert time: 2-4 PM",
        "📅 Most alerts on Wednesdays",
        "🌡️ Temperature > 30°C increases alerts by 40%",
        "🌧️ Low rainfall correlates with stress alerts"
    ]
    
    for pattern in patterns:
        st.markdown(f"- {pattern}")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("**System Recommendations:**")
    
    recommendations = [
        "Consider increasing monitoring frequency for Central Plot E",
        "Review irrigation schedule for North Field A",
        "Pest monitoring recommended for South Field B"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Export analytics
    if st.button("📊 Export Analytics Report", key="export_analytics"):
        st.success("Analytics report exported!")

def generate_active_alerts():
    """Generate mock active alerts data"""
    
    alerts_data = [
        {
            'Type': 'Vegetation Stress',
            'Severity': 'High',
            'Zone': 'Central Plot E',
            'Description': 'NDVI values below 0.6 detected in 40% of the zone',
            'Time': '2 hours ago',
            'Recommendation': 'Increase irrigation and check for pest activity'
        },
        {
            'Type': 'Pest Risk',
            'Severity': 'Medium',
            'Zone': 'South Field B',
            'Description': 'Environmental conditions favorable for aphid outbreak',
            'Time': '5 hours ago',
            'Recommendation': 'Apply preventive pest control measures'
        },
        {
            'Type': 'Environmental',
            'Severity': 'Medium',
            'Zone': 'North Field A',
            'Description': 'Soil moisture levels below optimal range',
            'Time': '1 day ago',
            'Recommendation': 'Adjust irrigation schedule'
        },
        {
            'Type': 'Disease Risk',
            'Severity': 'Low',
            'Zone': 'West Orchard D',
            'Description': 'Leaf wetness duration exceeding threshold',
            'Time': '2 days ago',
            'Recommendation': 'Monitor for fungal disease symptoms'
        },
        {
            'Type': 'Vegetation Stress',
            'Severity': 'Low',
            'Zone': 'East Pasture C',
            'Description': 'Minor NDVI decline in northern section',
            'Time': '3 days ago',
            'Recommendation': 'Continue monitoring, no immediate action required'
        }
    ]
    
    return pd.DataFrame(alerts_data)

def generate_alert_history():
    """Generate mock alert history data"""
    
    # Generate 30 days of historical data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    data = []
    
    for date in dates:
        # Generate random alert counts with some patterns
        day_of_week = date.weekday()
        
        # More alerts mid-week
        base_multiplier = 1.5 if day_of_week in [2, 3, 4] else 1.0
        
        for severity in ['High', 'Medium', 'Low']:
            for alert_type in ['Vegetation Stress', 'Pest Risk', 'Disease Risk', 'Environmental']:
                # Different base rates for different severities
                base_rate = {'High': 0.5, 'Medium': 1.5, 'Low': 2.0}[severity]
                
                # Random variation
                count = max(0, int(np.random.poisson(base_rate * base_multiplier)))
                
                data.append({
                    'Date': date,
                    'Severity': severity,
                    'Type': alert_type,
                    'Count': count
                })
    
    return pd.DataFrame(data)