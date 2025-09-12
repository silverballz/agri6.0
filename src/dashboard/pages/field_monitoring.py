"""
Field Monitoring page - Interactive maps and spatial analysis
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from datetime import datetime

def show_page():
    """Display the field monitoring page"""
    
    st.title("🗺️ Field Monitoring")
    st.markdown("Interactive maps showing spectral health zones and monitoring data")
    
    # Page controls
    display_map_controls()
    
    # Main map display
    display_interactive_map()
    
    # Zone details panel
    if 'selected_zone' in st.session_state and st.session_state.selected_zone:
        display_zone_details()

def display_map_controls():
    """Display map control panel"""
    
    st.subheader("🎛️ Map Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        map_layer = st.selectbox(
            "Base Layer",
            ["Satellite", "Terrain", "OpenStreetMap"],
            key="map_base_layer"
        )
    
    with col2:
        vegetation_index = st.selectbox(
            "Vegetation Index",
            ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"],
            key="map_vegetation_index"
        )
    
    with col3:
        show_alerts = st.checkbox(
            "Show Alerts",
            value=True,
            key="map_show_alerts"
        )
    
    with col4:
        show_sensors = st.checkbox(
            "Show Sensors",
            value=True,
            key="map_show_sensors"
        )
    
    # Color scale selector
    col1, col2 = st.columns(2)
    
    with col1:
        color_scale = st.selectbox(
            "Color Scale",
            ["RdYlGn", "Viridis", "Spectral", "RdBu"],
            key="map_color_scale"
        )
    
    with col2:
        opacity = st.slider(
            "Layer Opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="map_opacity"
        )

def display_interactive_map():
    """Display the main interactive map"""
    
    st.subheader(f"📍 {st.session_state.get('map_vegetation_index', 'NDVI')} Health Map")
    
    # Create base map
    center_lat = st.session_state.get('map_center', [40.7128, -74.0060])[0]
    center_lon = st.session_state.get('map_center', [40.7128, -74.0060])[1]
    
    # Create base map with proper tile configuration
    base_layer = st.session_state.get('map_base_layer', 'OpenStreetMap')
    
    if base_layer == 'OpenStreetMap':
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=st.session_state.get('map_zoom', 12),
            tiles='OpenStreetMap'
        )
    elif base_layer == 'Satellite':
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=st.session_state.get('map_zoom', 12),
            tiles=None
        )
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
    elif base_layer == 'Terrain':
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=st.session_state.get('map_zoom', 12),
            tiles=None
        )
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Terrain',
            overlay=False,
            control=True
        ).add_to(m)
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=st.session_state.get('map_zoom', 12),
            tiles='OpenStreetMap'
        )
    
    # Add monitoring zones with vegetation index coloring
    add_monitoring_zones(m)
    
    # Add alerts if enabled
    if st.session_state.get('map_show_alerts', True):
        add_alert_markers(m)
    
    # Add sensor locations if enabled
    if st.session_state.get('map_show_sensors', True):
        add_sensor_markers(m)
    
    # Add legend
    add_map_legend(m)
    
    # Display map and capture interactions
    map_data = st_folium(
        m,
        width=None,
        height=600,
        returned_objects=["last_object_clicked_popup", "last_clicked", "bounds"]
    )
    
    # Handle map interactions
    handle_map_interactions(map_data)



def add_monitoring_zones(map_obj):
    """Add monitoring zones with vegetation index coloring"""
    
    # Load Ludhiana GeoJSON data and create mock agricultural zones
    import json
    import os
    
    # Try to load the test.geojson file
    try:
        geojson_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'test.geojson')
        with open(geojson_path, 'r') as f:
            ludhiana_data = json.load(f)
        
        # Extract Ludhiana boundary coordinates
        ludhiana_coords = ludhiana_data['features'][0]['geometry']['coordinates'][0]
        
        # Create agricultural zones within Ludhiana area
        zones = [
            {
                "name": "Ludhiana North Farm",
                "coordinates": [[31.12, 75.78], [31.13, 75.78], [31.13, 75.80], [31.12, 75.80]],
                "ndvi": 0.78,
                "area": 450,
                "crop": "Wheat"
            },
            {
                "name": "Pakhowal Road Fields", 
                "coordinates": [[31.10, 75.82], [31.11, 75.82], [31.11, 75.84], [31.10, 75.84]],
                "ndvi": 0.72,
                "area": 380,
                "crop": "Rice"
            },
            {
                "name": "Sidhwan Bet Area",
                "coordinates": [[31.08, 75.80], [31.09, 75.80], [31.09, 75.82], [31.08, 75.82]],
                "ndvi": 0.85,
                "area": 320,
                "crop": "Sugarcane"
            },
            {
                "name": "Raikot Agricultural Zone",
                "coordinates": [[31.11, 75.76], [31.12, 75.76], [31.12, 75.78], [31.11, 75.78]],
                "ndvi": 0.69,
                "area": 520,
                "crop": "Cotton"
            },
            {
                "name": "Khanna District Fields",
                "coordinates": [[31.09, 75.84], [31.10, 75.84], [31.10, 75.86], [31.09, 75.86]],
                "ndvi": 0.63,
                "area": 290,
                "crop": "Maize"
            }
        ]
        
        # Add the main Ludhiana AOI boundary
        folium.Polygon(
            locations=[[coord[1], coord[0]] for coord in ludhiana_coords],  # Swap lat/lon
            color='blue',
            weight=3,
            fillColor='lightblue',
            fillOpacity=0.1,
            popup=folium.Popup("Ludhiana 10km x 10km AOI", max_width=200),
            tooltip="Ludhiana Area of Interest"
        ).add_to(map_obj)
        
    except Exception as e:
        st.error(f"Could not load Ludhiana GeoJSON data: {e}")
        # Fallback to default zones
        zones = [
            {
                "name": "Ludhiana North Farm",
                "coordinates": [[31.12, 75.78], [31.13, 75.78], [31.13, 75.80], [31.12, 75.80]],
                "ndvi": 0.78,
                "area": 450,
                "crop": "Wheat"
            },
            {
                "name": "Pakhowal Road Fields", 
                "coordinates": [[31.10, 75.82], [31.11, 75.82], [31.11, 75.84], [31.10, 75.84]],
                "ndvi": 0.72,
                "area": 380,
                "crop": "Rice"
            },
            {
                "name": "Sidhwan Bet Area",
                "coordinates": [[31.08, 75.80], [31.09, 75.80], [31.09, 75.82], [31.08, 75.82]],
                "ndvi": 0.85,
                "area": 320,
                "crop": "Sugarcane"
            },
            {
                "name": "Raikot Agricultural Zone",
                "coordinates": [[31.11, 75.76], [31.12, 75.76], [31.12, 75.78], [31.11, 75.78]],
                "ndvi": 0.69,
                "area": 520,
                "crop": "Cotton"
            },
            {
                "name": "Khanna District Fields",
                "coordinates": [[31.09, 75.84], [31.10, 75.84], [31.10, 75.86], [31.09, 75.86]],
                "ndvi": 0.63,
                "area": 290,
                "crop": "Maize"
            }
        ]
    
    # Color mapping based on NDVI values
    def get_zone_color(ndvi):
        if ndvi >= 0.8:
            return '#2E8B57'  # Dark green - excellent
        elif ndvi >= 0.7:
            return '#32CD32'  # Lime green - healthy
        elif ndvi >= 0.6:
            return '#FFD700'  # Gold - moderate
        elif ndvi >= 0.5:
            return '#FF8C00'  # Dark orange - stressed
        else:
            return '#DC143C'  # Crimson - critical
    
    # Add each zone as a polygon
    for zone in zones:
        color = get_zone_color(zone['ndvi'])
        
        # Create popup content
        popup_content = f"""
        <div style="width: 200px;">
            <h4>{zone['name']}</h4>
            <p><strong>Crop:</strong> {zone['crop']}</p>
            <p><strong>Area:</strong> {zone['area']} ha</p>
            <p><strong>NDVI:</strong> {zone['ndvi']:.2f}</p>
            <p><strong>Status:</strong> {get_health_status(zone['ndvi'])}</p>
            <button onclick="selectZone('{zone['name']}')">View Details</button>
        </div>
        """
        
        folium.Polygon(
            locations=zone['coordinates'],
            color=color,
            weight=2,
            fillColor=color,
            fillOpacity=st.session_state.get('map_opacity', 0.7),
            popup=folium.Popup(popup_content, max_width=250),
            tooltip=f"{zone['name']} - NDVI: {zone['ndvi']:.2f}"
        ).add_to(map_obj)

def get_health_status(ndvi):
    """Get health status text based on NDVI value"""
    if ndvi >= 0.8:
        return "Excellent"
    elif ndvi >= 0.7:
        return "Healthy"
    elif ndvi >= 0.6:
        return "Moderate"
    elif ndvi >= 0.5:
        return "Stressed"
    else:
        return "Critical"

def add_alert_markers(map_obj):
    """Add alert markers to the map"""
    
    # Mock alert locations in Ludhiana area
    alerts = [
        {
            "lat": 31.095,
            "lon": 75.85,
            "type": "Vegetation Stress",
            "severity": "High",
            "zone": "Khanna District Fields",
            "time": "2 hours ago"
        },
        {
            "lat": 31.105,
            "lon": 75.83,
            "type": "Pest Risk",
            "severity": "Medium", 
            "zone": "Pakhowal Road Fields",
            "time": "5 hours ago"
        },
        {
            "lat": 31.115,
            "lon": 75.77,
            "type": "Irrigation Alert",
            "severity": "Medium", 
            "zone": "Raikot Agricultural Zone",
            "time": "1 day ago"
        }
    ]
    
    # Severity colors
    severity_colors = {
        "High": "red",
        "Medium": "orange",
        "Low": "yellow"
    }
    
    # Severity icons
    severity_icons = {
        "High": "exclamation-triangle",
        "Medium": "exclamation-circle",
        "Low": "info-circle"
    }
    
    for alert in alerts:
        popup_content = f"""
        <div style="width: 180px;">
            <h4>🚨 {alert['type']}</h4>
            <p><strong>Severity:</strong> {alert['severity']}</p>
            <p><strong>Zone:</strong> {alert['zone']}</p>
            <p><strong>Time:</strong> {alert['time']}</p>
        </div>
        """
        
        folium.Marker(
            location=[alert['lat'], alert['lon']],
            popup=folium.Popup(popup_content, max_width=200),
            tooltip=f"{alert['type']} - {alert['severity']}",
            icon=folium.Icon(
                color=severity_colors[alert['severity']],
                icon=severity_icons[alert['severity']],
                prefix='fa'
            )
        ).add_to(map_obj)

def add_sensor_markers(map_obj):
    """Add sensor location markers to the map"""
    
    # Mock sensor locations in Ludhiana area
    sensors = [
        {
            "lat": 31.11,
            "lon": 75.81,
            "type": "Weather Station",
            "status": "Online",
            "last_reading": "5 min ago"
        },
        {
            "lat": 31.09,
            "lon": 75.83,
            "type": "Soil Moisture",
            "status": "Online", 
            "last_reading": "2 min ago"
        },
        {
            "lat": 31.12,
            "lon": 75.79,
            "type": "Leaf Wetness",
            "status": "Online",
            "last_reading": "10 min ago"
        },
        {
            "lat": 31.085,
            "lon": 75.81,
            "type": "Temperature Sensor",
            "status": "Offline",
            "last_reading": "2 hours ago"
        }
    ]
    
    # Status colors
    status_colors = {
        "Online": "green",
        "Offline": "red",
        "Warning": "orange"
    }
    
    for sensor in sensors:
        popup_content = f"""
        <div style="width: 160px;">
            <h4>📡 {sensor['type']}</h4>
            <p><strong>Status:</strong> {sensor['status']}</p>
            <p><strong>Last Reading:</strong> {sensor['last_reading']}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[sensor['lat'], sensor['lon']],
            radius=8,
            popup=folium.Popup(popup_content, max_width=180),
            tooltip=f"{sensor['type']} - {sensor['status']}",
            color=status_colors[sensor['status']],
            fillColor=status_colors[sensor['status']],
            fillOpacity=0.8
        ).add_to(map_obj)

def add_map_legend(map_obj):
    """Add legend to the map"""
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>NDVI Health Scale</h4>
    <p><i class="fa fa-square" style="color:#2E8B57"></i> Excellent (0.8+)</p>
    <p><i class="fa fa-square" style="color:#32CD32"></i> Healthy (0.7-0.8)</p>
    <p><i class="fa fa-square" style="color:#FFD700"></i> Moderate (0.6-0.7)</p>
    <p><i class="fa fa-square" style="color:#FF8C00"></i> Stressed (0.5-0.6)</p>
    <p><i class="fa fa-square" style="color:#DC143C"></i> Critical (<0.5)</p>
    </div>
    '''
    
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def handle_map_interactions(map_data):
    """Handle map click interactions"""
    
    if map_data['last_clicked']:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lng = map_data['last_clicked']['lng']
        
        # Display clicked coordinates
        st.sidebar.markdown(f"**Last Clicked:**")
        st.sidebar.markdown(f"Lat: {clicked_lat:.4f}")
        st.sidebar.markdown(f"Lng: {clicked_lng:.4f}")
        
        # Check if click was on a zone (simplified logic)
        # In production, this would use proper spatial queries
        zone_name = get_zone_from_coordinates(clicked_lat, clicked_lng)
        if zone_name:
            st.session_state.selected_zone = zone_name

def get_zone_from_coordinates(lat, lng):
    """Get zone name from coordinates (simplified)"""
    
    # Mock zone detection based on Ludhiana coordinate ranges
    if 31.12 <= lat <= 31.13 and 75.78 <= lng <= 75.80:
        return "Ludhiana North Farm"
    elif 31.10 <= lat <= 31.11 and 75.82 <= lng <= 75.84:
        return "Pakhowal Road Fields"
    elif 31.08 <= lat <= 31.09 and 75.80 <= lng <= 75.82:
        return "Sidhwan Bet Area"
    elif 31.11 <= lat <= 31.12 and 75.76 <= lng <= 75.78:
        return "Raikot Agricultural Zone"
    elif 31.09 <= lat <= 31.10 and 75.84 <= lng <= 75.86:
        return "Khanna District Fields"
    
    return None

def display_zone_details():
    """Display detailed information for selected zone"""
    
    st.subheader(f"📋 Zone Details: {st.session_state.selected_zone}")
    
    # Mock detailed zone data
    zone_details = get_zone_details(st.session_state.selected_zone)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information:**")
        st.markdown(f"- **Area:** {zone_details['area']} hectares")
        st.markdown(f"- **Crop Type:** {zone_details['crop']}")
        st.markdown(f"- **Planting Date:** {zone_details['planting_date']}")
        st.markdown(f"- **Expected Harvest:** {zone_details['harvest_date']}")
        
        st.markdown("**Current Conditions:**")
        st.markdown(f"- **NDVI:** {zone_details['ndvi']:.2f}")
        st.markdown(f"- **SAVI:** {zone_details['savi']:.2f}")
        st.markdown(f"- **Soil Moisture:** {zone_details['soil_moisture']}%")
        st.markdown(f"- **Temperature:** {zone_details['temperature']}°C")
    
    with col2:
        st.markdown("**Health Assessment:**")
        health_status = get_health_status(zone_details['ndvi'])
        status_color = {
            "Excellent": "🟢",
            "Healthy": "🟢", 
            "Moderate": "🟡",
            "Stressed": "🟠",
            "Critical": "🔴"
        }
        st.markdown(f"- **Status:** {status_color.get(health_status, '⚪')} {health_status}")
        
        st.markdown("**Active Alerts:**")
        for alert in zone_details['alerts']:
            st.markdown(f"- 🚨 {alert}")
        
        if not zone_details['alerts']:
            st.markdown("- ✅ No active alerts")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Trends", key=f"trends_{st.session_state.selected_zone}"):
            st.session_state.page_selector = "📈 Temporal Analysis"
            st.rerun()
    
    with col2:
        if st.button("📤 Export Data", key=f"export_{st.session_state.selected_zone}"):
            st.session_state.page_selector = "📤 Data Export"
            st.rerun()
    
    with col3:
        if st.button("❌ Clear Selection", key=f"clear_{st.session_state.selected_zone}"):
            st.session_state.selected_zone = None
            st.rerun()

def get_zone_details(zone_name):
    """Get detailed information for a specific zone"""
    
    # Mock detailed data for Ludhiana agricultural zones
    zone_data = {
        "Ludhiana North Farm": {
            "area": 450,
            "crop": "Wheat",
            "planting_date": "2024-11-15",
            "harvest_date": "2025-04-20",
            "ndvi": 0.78,
            "savi": 0.74,
            "soil_moisture": 68,
            "temperature": 22.3,
            "alerts": []
        },
        "Pakhowal Road Fields": {
            "area": 380,
            "crop": "Rice", 
            "planting_date": "2024-06-20",
            "harvest_date": "2024-11-10",
            "ndvi": 0.72,
            "savi": 0.68,
            "soil_moisture": 75,
            "temperature": 24.1,
            "alerts": ["Pest risk elevated"]
        },
        "Sidhwan Bet Area": {
            "area": 320,
            "crop": "Sugarcane",
            "planting_date": "2024-03-01",
            "harvest_date": "2025-02-15",
            "ndvi": 0.85,
            "savi": 0.81,
            "soil_moisture": 82,
            "temperature": 25.8,
            "alerts": []
        },
        "Raikot Agricultural Zone": {
            "area": 520,
            "crop": "Cotton",
            "planting_date": "2024-05-15",
            "harvest_date": "2024-12-01",
            "ndvi": 0.69,
            "savi": 0.65,
            "soil_moisture": 62,
            "temperature": 26.2,
            "alerts": ["Irrigation needed in sector 2"]
        },
        "Khanna District Fields": {
            "area": 290,
            "crop": "Maize",
            "planting_date": "2024-07-01",
            "harvest_date": "2024-11-30",
            "ndvi": 0.63,
            "savi": 0.59,
            "soil_moisture": 58,
            "temperature": 25.5,
            "alerts": ["Vegetation stress detected", "Low soil moisture"]
        }
    }
    
    return zone_data.get(zone_name, {})