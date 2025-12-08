"""
Alerts and Notifications page - Alert management and notification system
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.error_handler import safe_page, handle_data_loading, logger
from database.db_manager import DatabaseManager
from alerts.alert_generator import AlertGenerator, AlertSeverity, AlertType

@safe_page
def show_page():
    """Display the alerts and notifications page"""
    
    st.title("🚨 Alerts & Notifications")
    st.markdown("Monitor and manage active alerts and system notifications")
    
    # Check if demo mode is active
    if st.session_state.get('demo_mode', False) and st.session_state.get('demo_data'):
        show_demo_alerts()
        return
    
    # Initialize database manager
    try:
        db_manager = DatabaseManager()
        
        # Get database stats
        db_stats = db_manager.get_database_stats()
        
        # Alert summary metrics
        display_alert_metrics(db_manager, db_stats)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_active_alerts(db_manager)
            display_alert_history(db_manager)
        
        with col2:
            display_alert_filters()
            display_alert_analytics(db_manager)
    
    except Exception as e:
        st.error(f"Error loading alerts: {str(e)}")
        logger.error(f"Alerts page error: {e}", exc_info=True)
        st.info("💡 Make sure the database has been initialized and populated with data.")

def display_alert_metrics(db_manager: DatabaseManager, db_stats: dict):
    """Display alert summary metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get active alerts
    active_alerts = db_manager.get_active_alerts()
    total_alerts = db_stats.get('total_alerts', 0)
    active_count = len(active_alerts)
    
    # Count high priority alerts
    high_priority_count = sum(1 for alert in active_alerts 
                             if alert.get('severity') in ['critical', 'high'])
    
    # Calculate acknowledgment rate
    acknowledged_count = total_alerts - active_count
    ack_rate = (acknowledged_count / total_alerts * 100) if total_alerts > 0 else 0
    
    with col1:
        st.metric(
            "Active Alerts",
            active_count,
            delta=f"{total_alerts} total",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "High Priority",
            high_priority_count,
            delta="Critical/High",
            delta_color="normal" if high_priority_count > 0 else "off"
        )
    
    with col3:
        st.metric(
            "Total Alerts",
            total_alerts,
            delta="All time",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            "Acknowledgment Rate",
            f"{ack_rate:.0f}%",
            delta=f"{acknowledged_count} resolved",
            delta_color="inverse"
        )

def display_active_alerts(db_manager: DatabaseManager):
    """Display active alerts table"""
    
    st.subheader("🔴 Active Alerts")
    
    # Get active alerts from database
    active_alerts = db_manager.get_active_alerts(limit=100)
    
    if not active_alerts:
        st.info("✅ No active alerts at this time. All systems operating normally.")
        return
    
    # Alert filters from session state
    severity_filter = st.session_state.get('alert_severity_filter', 
                                          ['critical', 'high', 'medium', 'low'])
    type_filter = st.session_state.get('alert_type_filter', 
                                      ['vegetation_stress', 'pest_risk', 'disease_risk', 
                                       'water_stress', 'environmental'])
    
    # Filter alerts
    filtered_alerts = [
        alert for alert in active_alerts
        if alert.get('severity') in severity_filter 
        and alert.get('alert_type') in type_filter
    ]
    
    if not filtered_alerts:
        st.info("No alerts match the current filters.")
        return
    
    st.markdown(f"**Showing {len(filtered_alerts)} of {len(active_alerts)} active alerts**")
    
    # Display alerts as cards
    for idx, alert in enumerate(filtered_alerts):
        display_alert_card(alert, idx, db_manager)
    
    # Bulk actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Acknowledge All Visible", key="ack_all_alerts"):
            for alert in filtered_alerts:
                db_manager.acknowledge_alert(alert['id'])
            st.success(f"Acknowledged {len(filtered_alerts)} alerts!")
            st.rerun()
    
    with col2:
        if st.button("📧 Export Alert Report", key="export_alerts"):
            # Create CSV export
            df = pd.DataFrame(filtered_alerts)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔄 Refresh Alerts", key="refresh_alerts"):
            st.rerun()

def display_alert_card(alert: dict, idx: int, db_manager: DatabaseManager):
    """Display individual alert card"""
    
    # Severity styling
    severity_styles = {
        'critical': {'color': '#d32f2f', 'bg': '#ffebee', 'icon': '🔴', 'label': 'CRITICAL'},
        'high': {'color': '#f44336', 'bg': '#fff3e0', 'icon': '🟠', 'label': 'HIGH'},
        'medium': {'color': '#ff9800', 'bg': '#fff8e1', 'icon': '🟡', 'label': 'MEDIUM'},
        'low': {'color': '#4caf50', 'bg': '#e8f5e9', 'icon': '🟢', 'label': 'LOW'}
    }
    
    severity = alert.get('severity', 'medium')
    style = severity_styles.get(severity, severity_styles['medium'])
    
    # Format alert type for display
    alert_type_display = alert.get('alert_type', '').replace('_', ' ').title()
    
    # Parse created_at timestamp
    created_at = alert.get('created_at', '')
    try:
        created_dt = datetime.fromisoformat(created_at)
        time_ago = get_time_ago(created_dt)
    except:
        time_ago = "Unknown time"
    
    # Get affected area info
    affected_area = alert.get('affected_area')
    has_map = affected_area is not None
    
    # Card container
    with st.container():
        st.markdown(f"""
        <div style="
            border: 2px solid {style['color']};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: {style['bg']};
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {style['color']};">
                    {style['icon']} {alert_type_display}
                </h4>
                <span style="
                    background-color: {style['color']};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                ">
                    {style['label']}
                </span>
            </div>
            <p style="margin: 10px 0;"><strong>Alert ID:</strong> #{alert.get('id')}</p>
            <p style="margin: 10px 0;"><strong>Message:</strong> {alert.get('message', 'No message')}</p>
            <p style="margin: 10px 0;"><strong>Time:</strong> {time_ago}</p>
            <p style="margin: 10px 0; padding: 10px; background-color: rgba(255,255,255,0.5); border-radius: 5px;">
                <strong>💡 Recommendation:</strong> {alert.get('recommendation', 'No recommendation available')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show affected area map if available
        if has_map:
            with st.expander("📍 View Affected Area on Map"):
                try:
                    display_affected_area_map(affected_area)
                except Exception as e:
                    st.error(f"Could not display map: {str(e)}")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("✅ Acknowledge", key=f"ack_{alert['id']}_{idx}"):
                if db_manager.acknowledge_alert(alert['id']):
                    st.success("Alert acknowledged!")
                    st.rerun()
                else:
                    st.error("Failed to acknowledge alert")
        
        with col2:
            if st.button("📍 View Field", key=f"map_{alert['id']}_{idx}"):
                st.session_state.page_selector = "🗺️ Field Monitoring"
                st.session_state.selected_imagery_id = alert.get('imagery_id')
                st.rerun()
        
        with col3:
            if st.button("📊 View Trends", key=f"trends_{alert['id']}_{idx}"):
                st.session_state.page_selector = "📈 Temporal Analysis"
                st.rerun()
        
        with col4:
            # Show metadata
            if st.button("ℹ️ Details", key=f"details_{alert['id']}_{idx}"):
                st.json(alert)

def display_alert_history(db_manager: DatabaseManager):
    """Display alert history and trends"""
    
    st.subheader("📜 Alert History")
    
    # Time period selector
    history_period = st.selectbox(
        "History Period",
        ["Last 10 Alerts", "Last 50 Alerts", "All Alerts"],
        key="alert_history_period"
    )
    
    # Get limit based on selection
    limit_map = {
        "Last 10 Alerts": 10,
        "Last 50 Alerts": 50,
        "All Alerts": 1000
    }
    limit = limit_map.get(history_period, 50)
    
    # Get alert history from database
    all_alerts = db_manager.get_alert_history(limit=limit)
    
    if not all_alerts:
        st.info("No alert history available.")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_alerts)
    
    # Parse timestamps
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    
    # Alert trend chart by severity
    severity_counts = df.groupby(['date', 'severity']).size().reset_index(name='count')
    
    if not severity_counts.empty:
        fig = px.line(
            severity_counts,
            x='date',
            y='count',
            color='severity',
            title="Alert Trends Over Time",
            color_discrete_map={
                'critical': '#d32f2f',
                'high': '#f44336',
                'medium': '#ff9800',
                'low': '#4caf50'
            },
            labels={'date': 'Date', 'count': 'Number of Alerts', 'severity': 'Severity'}
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Alert type distribution
    type_counts = df.groupby(['alert_type', 'severity']).size().reset_index(name='count')
    
    if not type_counts.empty:
        # Format alert types for display
        type_counts['alert_type_display'] = type_counts['alert_type'].str.replace('_', ' ').str.title()
        
        fig2 = px.bar(
            type_counts,
            x='alert_type_display',
            y='count',
            color='severity',
            title="Alert Distribution by Type",
            color_discrete_map={
                'critical': '#d32f2f',
                'high': '#f44336',
                'medium': '#ff9800',
                'low': '#4caf50'
            },
            labels={'alert_type_display': 'Alert Type', 'count': 'Number of Alerts', 'severity': 'Severity'}
        )
        
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Show acknowledgment statistics
    ack_stats = df['acknowledged'].value_counts()
    if not ack_stats.empty:
        st.markdown("**Acknowledgment Statistics:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Acknowledged", ack_stats.get(1, 0))
        with col2:
            st.metric("Pending", ack_stats.get(0, 0))

def display_alert_filters():
    """Display alert filtering controls"""
    
    st.subheader("🔧 Alert Filters")
    
    # Severity filter
    severity_options = ['critical', 'high', 'medium', 'low']
    severity_display = {
        'critical': '🔴 Critical',
        'high': '🟠 High',
        'medium': '🟡 Medium',
        'low': '🟢 Low'
    }
    
    selected_severity = st.multiselect(
        "Severity Levels",
        severity_options,
        default=st.session_state.get('alert_severity_filter', severity_options),
        format_func=lambda x: severity_display.get(x, x),
        key="alert_severity_filter"
    )
    
    # Type filter
    type_options = ['vegetation_stress', 'pest_risk', 'disease_risk', 'water_stress', 'environmental']
    type_display = {
        'vegetation_stress': '🌱 Vegetation Stress',
        'pest_risk': '🐛 Pest Risk',
        'disease_risk': '🦠 Disease Risk',
        'water_stress': '💧 Water Stress',
        'environmental': '🌡️ Environmental'
    }
    
    selected_types = st.multiselect(
        "Alert Types",
        type_options,
        default=st.session_state.get('alert_type_filter', type_options),
        format_func=lambda x: type_display.get(x, x),
        key="alert_type_filter"
    )
    
    st.markdown("---")
    
    # Notification settings
    st.subheader("🔔 Notification Settings")
    
    st.info("💡 Notification settings are for display purposes in this demo.")
    
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

def display_alert_analytics(db_manager: DatabaseManager):
    """Display alert analytics and insights"""
    
    st.subheader("📊 Alert Analytics")
    
    # Get all alerts for analysis
    all_alerts = db_manager.get_alert_history(limit=1000)
    
    if not all_alerts:
        st.info("No alert data available for analytics.")
        return
    
    df = pd.DataFrame(all_alerts)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Calculate response time for acknowledged alerts
    acknowledged = df[df['acknowledged'] == 1].copy()
    if not acknowledged.empty and 'acknowledged_at' in acknowledged.columns:
        acknowledged['acknowledged_at'] = pd.to_datetime(acknowledged['acknowledged_at'])
        acknowledged['response_time'] = (acknowledged['acknowledged_at'] - acknowledged['created_at']).dt.total_seconds() / 3600
        
        st.markdown("**Response Time Analysis:**")
        
        avg_response = acknowledged['response_time'].mean()
        min_response = acknowledged['response_time'].min()
        max_response = acknowledged['response_time'].max()
        
        st.markdown(f"- **Average Response:** {avg_response:.1f} hours")
        st.markdown(f"- **Fastest Response:** {min_response:.1f} hours")
        st.markdown(f"- **Slowest Response:** {max_response:.1f} hours")
        st.markdown(f"- **Target Response:** < 4 hours")
    else:
        st.markdown("**Response Time Analysis:**")
        st.markdown("- No acknowledged alerts yet")
    
    st.markdown("---")
    
    # Alert patterns
    st.markdown("**Alert Patterns:**")
    
    # Most common alert type
    if not df.empty:
        most_common_type = df['alert_type'].value_counts().index[0]
        most_common_count = df['alert_type'].value_counts().values[0]
        st.markdown(f"- 📊 Most common: {most_common_type.replace('_', ' ').title()} ({most_common_count} alerts)")
        
        # Most common severity
        most_common_severity = df['severity'].value_counts().index[0]
        st.markdown(f"- ⚠️ Most common severity: {most_common_severity.title()}")
        
        # Time-based patterns
        df['hour'] = df['created_at'].dt.hour
        peak_hour = df['hour'].mode()[0] if not df['hour'].empty else 12
        st.markdown(f"- 🕐 Peak alert time: {peak_hour}:00")
        
        df['day_of_week'] = df['created_at'].dt.day_name()
        if not df['day_of_week'].empty:
            peak_day = df['day_of_week'].mode()[0]
            st.markdown(f"- 📅 Most alerts on: {peak_day}")
    
    st.markdown("---")
    
    # System insights
    st.markdown("**System Insights:**")
    
    total_alerts = len(df)
    active_alerts = len(df[df['acknowledged'] == 0])
    
    insights = [
        f"Total alerts generated: {total_alerts}",
        f"Currently active: {active_alerts}",
        f"Acknowledgment rate: {((total_alerts - active_alerts) / total_alerts * 100):.1f}%" if total_alerts > 0 else "No alerts yet"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Export analytics
    if st.button("📊 Export Analytics Report", key="export_analytics"):
        # Create detailed analytics report
        report_df = df[['created_at', 'alert_type', 'severity', 'message', 'acknowledged']]
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Analytics CSV",
            data=csv,
            file_name=f"alert_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def get_time_ago(dt: datetime) -> str:
    """Convert datetime to human-readable 'time ago' string."""
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "Just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"


def display_affected_area_map(geojson_str: str):
    """Display affected area on a map using GeoJSON."""
    try:
        # Parse GeoJSON
        geojson_data = json.loads(geojson_str) if isinstance(geojson_str, str) else geojson_str
        
        # Create a simple map centered on the affected area
        # For now, use a default center (can be improved with actual coordinates)
        m = folium.Map(location=[20.0, 77.0], zoom_start=10)
        
        # Add GeoJSON layer
        folium.GeoJson(
            geojson_data,
            style_function=lambda x: {
                'fillColor': '#ff0000',
                'color': '#ff0000',
                'weight': 2,
                'fillOpacity': 0.3
            }
        ).add_to(m)
        
        # Display map
        folium_static(m, width=600, height=400)
        
    except Exception as e:
        st.error(f"Could not parse GeoJSON: {str(e)}")



def show_demo_alerts():
    """Display alerts page with demo data"""
    
    demo_manager = st.session_state.demo_data
    scenario_name = st.session_state.get('demo_scenario', 'healthy_field')
    
    # Get demo alerts
    all_alerts = demo_manager.get_alerts(scenario_name=scenario_name)
    active_alerts = demo_manager.get_active_alerts(scenario_name=scenario_name)
    
    st.info(f"**Demo Scenario:** {demo_manager.get_scenario_description(scenario_name)}")
    
    # Alert metrics
    st.subheader("📊 Alert Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Alerts", len(active_alerts))
    
    with col2:
        critical_count = sum(1 for a in active_alerts if a.get('severity') == 'critical')
        st.metric("High Priority", critical_count, help="Critical/High severity")
    
    with col3:
        st.metric("Total Alerts", len(all_alerts), help="All time")
    
    with col4:
        ack_rate = ((len(all_alerts) - len(active_alerts)) / len(all_alerts) * 100) if all_alerts else 0
        st.metric("Acknowledgment Rate", f"{ack_rate:.0f}%", help=f"{len(all_alerts) - len(active_alerts)} resolved")
    
    # Active alerts
    st.subheader("🔴 Active Alerts")
    
    if active_alerts:
        severity_icons = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }
        
        for alert in active_alerts:
            severity = alert.get('severity', 'medium')
            alert_type = alert.get('alert_type', 'Unknown')
            message = alert.get('message', 'No message')
            recommendation = alert.get('recommendation', '')
            
            with st.expander(f"{severity_icons.get(severity, '⚪')} {alert_type.replace('_', ' ').title()} - {severity.upper()}", expanded=True):
                st.markdown(f"**Message:** {message}")
                if recommendation:
                    st.markdown(f"**Recommendation:** {recommendation}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Acknowledge", key=f"ack_{alert.get('id', hash(message))}"):
                        st.success("Alert acknowledged!")
                with col_b:
                    if st.button("ℹ️ More Info", key=f"info_{alert.get('id', hash(message))}"):
                        st.info("This is demo data. In production, this would show detailed alert information.")
    else:
        st.success("✅ No active alerts at this time. All systems operating normally.")
    
    # Alert history
    st.subheader("📜 Alert History")
    
    if all_alerts:
        # Create DataFrame
        alert_data = []
        for alert in all_alerts:
            alert_data.append({
                'Type': alert.get('alert_type', 'Unknown').replace('_', ' ').title(),
                'Severity': alert.get('severity', 'medium').title(),
                'Message': alert.get('message', 'No message'),
                'Status': 'Resolved' if alert.get('acknowledged', False) else 'Active'
            })
        
        df = pd.DataFrame(alert_data)
        st.dataframe(df, use_container_width=True)
        
        # Alert distribution chart
        st.subheader("📊 Alert Distribution")
        
        severity_counts = df['Severity'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                marker=dict(colors=['#ef5350', '#ffa726', '#ffeb3b', '#66bb6a'])
            )
        ])
        
        fig.update_layout(
            title='Alerts by Severity',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alert history available.")
