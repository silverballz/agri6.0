"""
UI Components Module - Consistent styling and reusable components for AgriFlux Dashboard
Provides tooltips, help text, color schemes, and visual consistency across all pages
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime

# ============================================================================
# COLOR SCHEMES AND CONSTANTS
# ============================================================================

class ColorScheme:
    """Consistent color scheme for the entire dashboard"""
    
    # Health status colors
    EXCELLENT = "#2E8B57"  # Dark green
    HEALTHY = "#32CD32"    # Lime green
    MODERATE = "#FFD700"   # Gold
    STRESSED = "#FF8C00"   # Dark orange
    CRITICAL = "#DC143C"   # Crimson
    
    # Severity colors
    SEVERITY_CRITICAL = "#d32f2f"
    SEVERITY_HIGH = "#f44336"
    SEVERITY_MEDIUM = "#ff9800"
    SEVERITY_LOW = "#4caf50"
    
    # UI colors
    PRIMARY = "#4caf50"
    SECONDARY = "#2196F3"
    SUCCESS = "#4caf50"
    WARNING = "#ff9800"
    ERROR = "#f44336"
    INFO = "#2196F3"
    
    # Background colors
    BG_DARK = "#0e1117"
    BG_CARD = "#1a202c"
    BG_CARD_LIGHT = "#2d3748"
    BORDER = "#4a5568"
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a0aec0"
    TEXT_MUTED = "#718096"

class Icons:
    """Consistent icon set for the dashboard"""
    
    # Status icons
    EXCELLENT = "🟢"
    HEALTHY = "🟢"
    MODERATE = "🟡"
    STRESSED = "🟠"
    CRITICAL = "🔴"
    
    # Feature icons
    FIELD = "🗺️"
    CHART = "📊"
    ALERT = "🚨"
    EXPORT = "📤"
    HELP = "❓"
    INFO = "ℹ️"
    WARNING = "⚠️"
    SUCCESS = "✅"
    ERROR = "❌"
    
    # Vegetation indices
    NDVI = "🌱"
    SAVI = "🌾"
    EVI = "🌿"
    NDWI = "💧"
    NDSI = "🏜️"

# ============================================================================
# TOOLTIP AND HELP TEXT DEFINITIONS
# ============================================================================

class HelpText:
    """Centralized help text and tooltips for all dashboard elements"""
    
    # Vegetation Indices
    VEGETATION_INDICES = {
        "NDVI": """**NDVI (Normalized Difference Vegetation Index)**
        
The most widely used vegetation index for assessing plant health.

**Range:** -1 to +1
**Interpretation:**
• 0.8-1.0: Dense, healthy vegetation 🟢
• 0.6-0.8: Moderate vegetation 🟡
• 0.4-0.6: Sparse vegetation 🟠
• <0.4: Stressed or no vegetation 🔴

**Best for:** General crop health monitoring, biomass estimation""",
        
        "SAVI": """**SAVI (Soil Adjusted Vegetation Index)**
        
Minimizes soil brightness influences, better for sparse vegetation.

**Range:** -1 to +1
**Interpretation:**
• >0.7: Healthy vegetation 🟢
• 0.5-0.7: Moderate vegetation 🟡
• 0.3-0.5: Sparse vegetation 🟠
• <0.3: Bare soil or stressed 🔴

**Best for:** Early season crops, sparse canopy, exposed soil areas""",
        
        "EVI": """**EVI (Enhanced Vegetation Index)**
        
Enhanced sensitivity in high biomass regions, reduces atmospheric influences.

**Range:** -1 to +1
**Interpretation:**
• >0.6: Dense vegetation 🟢
• 0.4-0.6: Moderate vegetation 🟡
• 0.2-0.4: Sparse vegetation 🟠
• <0.2: Minimal vegetation 🔴

**Best for:** Dense canopy, tropical regions, high biomass crops""",
        
        "NDWI": """**NDWI (Normalized Difference Water Index)**
        
Measures vegetation water content and irrigation monitoring.

**Range:** -1 to +1
**Interpretation:**
• >0.3: High water content 🟢
• 0.1-0.3: Moderate water content 🟡
• -0.1-0.1: Low water content 🟠
• <-0.1: Water stress 🔴

**Best for:** Irrigation management, drought monitoring, water stress detection""",
        
        "NDSI": """**NDSI (Normalized Difference Soil Index)**
        
Detects bare soil and soil moisture conditions.

**Range:** -1 to +1
**Interpretation:**
• >0.5: Dry bare soil 🟠
• 0.2-0.5: Moist soil 🟡
• 0-0.2: Wet soil 🟢
• <0: Water or vegetation 🔵

**Best for:** Soil moisture assessment, tillage monitoring, planting readiness"""
    }
    
    # Alert Severities
    ALERT_SEVERITIES = {
        "critical": """**Critical Alert** 🔴
        
Requires immediate action within 2-4 hours.

**Examples:**
• Severe vegetation stress (NDVI < 0.3)
• Extreme pest infestation detected
• Critical water shortage
• Disease outbreak confirmed

**Recommended Action:** Immediate field inspection and intervention""",
        
        "high": """**High Priority Alert** 🟠
        
Requires action within 24 hours.

**Examples:**
• Moderate vegetation stress (NDVI 0.3-0.5)
• High pest risk conditions
• Irrigation system malfunction
• Nutrient deficiency detected

**Recommended Action:** Schedule field visit and prepare intervention""",
        
        "medium": """**Medium Priority Alert** 🟡
        
Monitor closely, action within 48-72 hours.

**Examples:**
• Mild vegetation stress (NDVI 0.5-0.6)
• Moderate pest risk
• Suboptimal soil moisture
• Weather-related concerns

**Recommended Action:** Increase monitoring frequency, prepare contingency""",
        
        "low": """**Low Priority Alert** 🟢
        
Informational, routine monitoring sufficient.

**Examples:**
• Minor variations in vegetation health
• Seasonal changes
• Preventive maintenance reminders
• Data quality notifications

**Recommended Action:** Continue routine monitoring"""
    }
    
    # System Features
    FEATURES = {
        "demo_mode": """**Demo Mode** 🎬
        
Loads pre-configured sample data for quick demonstrations.

**Includes:**
• 3 field scenarios (healthy, stressed, mixed)
• 5 time points for temporal analysis
• Sample alerts at all severity levels
• AI predictions with confidence scores

**Use Cases:**
• Training new users
• Demonstrating features to stakeholders
• Testing dashboard functionality
• Presentations and demos""",
        
        "ai_predictions": """**AI Crop Health Predictions** 🤖
        
Machine learning-based crop health classification.

**Modes:**
• **AI Mode:** Uses trained CNN model for high accuracy
• **Rule-Based Mode:** Uses NDVI thresholds as fallback

**Classification Categories:**
• Healthy: Thriving vegetation
• Moderate: Monitor closely
• Stressed: Investigate soon
• Critical: Immediate attention

**Confidence Scores:** Indicates prediction reliability (0-100%)""",
        
        "auto_refresh": """**Auto-Refresh** 🔄
        
Automatically updates dashboard data every 30 seconds.

**Benefits:**
• Real-time monitoring
• Immediate alert notifications
• Live sensor data updates

**Note:** May impact performance on slower connections.
Disable for better performance during detailed analysis.""",
        
        "data_quality": """**Data Quality Indicator** 📡
        
Assesses the reliability of satellite imagery.

**Factors:**
• Cloud coverage percentage
• Atmospheric conditions
• Sensor calibration
• Processing completeness

**Quality Levels:**
• Excellent: <10% clouds ☀️
• Good: 10-30% clouds 🌤️
• Fair: 30-50% clouds ⛅
• Poor: >50% clouds ☁️"""
    }
    
    # Export Formats
    EXPORT_FORMATS = {
        "CSV": "Comma-separated values - Compatible with Excel, R, Python",
        "Excel": "Microsoft Excel format - Multiple sheets, formatting preserved",
        "GeoTIFF": "Georeferenced raster - Compatible with GIS software (QGIS, ArcGIS)",
        "GeoJSON": "Geographic JSON - Web-compatible, human-readable",
        "PDF": "Portable Document Format - Professional reports with charts and maps",
        "JSON": "JavaScript Object Notation - API-friendly, structured data",
        "Shapefile": "ESRI Shapefile - Standard GIS vector format",
        "KML": "Keyhole Markup Language - Google Earth compatible"
    }

# ============================================================================
# REUSABLE UI COMPONENTS
# ============================================================================

def metric_card(title: str, value: str, delta: Optional[str] = None, 
                delta_color: str = "normal", icon: str = "", 
                help_text: Optional[str] = None):
    """
    Display a styled metric card with consistent formatting
    
    Args:
        title: Metric title
        value: Metric value
        delta: Change indicator (optional)
        delta_color: Color for delta ("normal", "inverse", "off")
        icon: Emoji icon (optional)
        help_text: Tooltip text (optional)
    """
    
    # Determine delta styling
    delta_class = ""
    if delta and delta_color != "off":
        if delta_color == "inverse":
            delta_class = "metric-delta-negative" if delta.startswith("+") else "metric-delta-positive"
        else:
            delta_class = "metric-delta-positive" if delta.startswith("+") else "metric-delta-negative"
    
    # Build HTML
    html = f"""
    <div class="metric-container">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
    """
    
    if delta:
        html += f'<div class="metric-delta {delta_class}">{delta}</div>'
    
    html += "</div>"
    
    # Display with help text if provided
    if help_text:
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span title="{help_text}">ℹ️</span>', unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def info_box(message: str, box_type: str = "info", icon: Optional[str] = None):
    """
    Display a styled information box
    
    Args:
        message: Message to display
        box_type: Type of box ("info", "success", "warning", "error")
        icon: Custom icon (optional, defaults based on type)
    """
    
    colors = {
        "info": {"bg": "#e3f2fd", "border": "#2196F3", "icon": "ℹ️"},
        "success": {"bg": "#e8f5e9", "border": "#4caf50", "icon": "✅"},
        "warning": {"bg": "#fff3e0", "border": "#ff9800", "icon": "⚠️"},
        "error": {"bg": "#ffebee", "border": "#f44336", "icon": "❌"}
    }
    
    style = colors.get(box_type, colors["info"])
    display_icon = icon or style["icon"]
    
    st.markdown(f"""
    <div style="
        background-color: {style['bg']};
        border-left: 4px solid {style['border']};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    ">
        <strong>{display_icon} {message}</strong>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: Optional[str] = None, icon: str = ""):
    """
    Display a styled section header
    
    Args:
        title: Section title
        subtitle: Optional subtitle
        icon: Emoji icon
    """
    
    st.markdown(f"""
    <div style="
        padding: 1rem 0;
        border-bottom: 2px solid {ColorScheme.PRIMARY};
        margin-bottom: 1.5rem;
    ">
        <h2 style="color: {ColorScheme.PRIMARY}; margin: 0;">
            {icon} {title}
        </h2>
        {f'<p style="color: {ColorScheme.TEXT_SECONDARY}; margin: 0.5rem 0 0 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def tooltip_icon(help_text: str, icon: str = "ℹ️"):
    """
    Display a tooltip icon with hover text
    
    Args:
        help_text: Text to display on hover
        icon: Icon to display
    """
    
    st.markdown(f"""
    <span title="{help_text}" style="cursor: help; font-size: 1.2em;">
        {icon}
    </span>
    """, unsafe_allow_html=True)


def health_status_badge(ndvi_value: float, show_value: bool = True):
    """
    Display a health status badge based on NDVI value
    
    Args:
        ndvi_value: NDVI value (0-1)
        show_value: Whether to show the numeric value
    """
    
    if ndvi_value >= 0.8:
        status = "Excellent"
        color = ColorScheme.EXCELLENT
        icon = Icons.EXCELLENT
    elif ndvi_value >= 0.7:
        status = "Healthy"
        color = ColorScheme.HEALTHY
        icon = Icons.HEALTHY
    elif ndvi_value >= 0.6:
        status = "Moderate"
        color = ColorScheme.MODERATE
        icon = Icons.MODERATE
    elif ndvi_value >= 0.5:
        status = "Stressed"
        color = ColorScheme.STRESSED
        icon = Icons.STRESSED
    else:
        status = "Critical"
        color = ColorScheme.CRITICAL
        icon = Icons.CRITICAL
    
    value_text = f" ({ndvi_value:.2f})" if show_value else ""
    
    st.markdown(f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: bold;
        display: inline-block;
    ">
        {icon} {status}{value_text}
    </span>
    """, unsafe_allow_html=True)


def severity_badge(severity: str):
    """
    Display a severity badge for alerts
    
    Args:
        severity: Severity level ("critical", "high", "medium", "low")
    """
    
    colors = {
        "critical": ColorScheme.SEVERITY_CRITICAL,
        "high": ColorScheme.SEVERITY_HIGH,
        "medium": ColorScheme.SEVERITY_MEDIUM,
        "low": ColorScheme.SEVERITY_LOW
    }
    
    icons = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢"
    }
    
    color = colors.get(severity.lower(), ColorScheme.INFO)
    icon = icons.get(severity.lower(), "⚪")
    
    st.markdown(f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
    ">
        {icon} {severity.upper()}
    </span>
    """, unsafe_allow_html=True)


def progress_bar(value: float, max_value: float = 100, label: str = "", 
                 color: Optional[str] = None):
    """
    Display a styled progress bar
    
    Args:
        value: Current value
        max_value: Maximum value
        label: Label text
        color: Bar color (optional, auto-determined if not provided)
    """
    
    percentage = (value / max_value) * 100
    
    # Auto-determine color based on percentage if not provided
    if color is None:
        if percentage >= 80:
            color = ColorScheme.SUCCESS
        elif percentage >= 50:
            color = ColorScheme.WARNING
        else:
            color = ColorScheme.ERROR
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        {f'<div style="margin-bottom: 0.5rem; color: {ColorScheme.TEXT_PRIMARY};">{label}</div>' if label else ''}
        <div style="
            background-color: {ColorScheme.BG_CARD_LIGHT};
            border-radius: 10px;
            height: 24px;
            overflow: hidden;
        ">
            <div style="
                background-color: {color};
                width: {percentage}%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 12px;
                transition: width 0.3s ease;
            ">
                {percentage:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def data_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None,
               highlight_column: Optional[str] = None):
    """
    Display a styled data table
    
    Args:
        data: List of dictionaries containing row data
        headers: Optional list of column headers
        highlight_column: Column to highlight (optional)
    """
    
    if not data:
        st.info("No data available")
        return
    
    # Use first row keys as headers if not provided
    if headers is None:
        headers = list(data[0].keys())
    
    # Build table HTML
    html = f"""
    <table style="
        width: 100%;
        border-collapse: collapse;
        background-color: {ColorScheme.BG_CARD};
        border-radius: 8px;
        overflow: hidden;
    ">
        <thead>
            <tr style="background-color: {ColorScheme.BG_CARD_LIGHT};">
    """
    
    for header in headers:
        html += f"""
                <th style="
                    padding: 12px;
                    text-align: left;
                    color: {ColorScheme.PRIMARY};
                    font-weight: bold;
                    border-bottom: 2px solid {ColorScheme.BORDER};
                ">{header}</th>
        """
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    for row in data:
        html += "<tr>"
        for header in headers:
            value = row.get(header, "")
            highlight = header == highlight_column
            html += f"""
                <td style="
                    padding: 12px;
                    border-bottom: 1px solid {ColorScheme.BORDER};
                    color: {ColorScheme.TEXT_PRIMARY};
                    {'background-color: ' + ColorScheme.PRIMARY + '20;' if highlight else ''}
                ">{value}</td>
            """
        html += "</tr>"
    
    html += """
        </tbody>
    </table>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def loading_spinner(message: str = "Loading..."):
    """
    Display a loading spinner with message
    
    Args:
        message: Loading message
    """
    
    return st.spinner(f"⏳ {message}")


def empty_state(message: str, icon: str = "📭", action_text: Optional[str] = None,
                action_callback: Optional[callable] = None):
    """
    Display an empty state message
    
    Args:
        message: Empty state message
        icon: Icon to display
        action_text: Optional action button text
        action_callback: Optional callback for action button
    """
    
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 3rem 1rem;
        background-color: {ColorScheme.BG_CARD};
        border-radius: 15px;
        border: 2px dashed {ColorScheme.BORDER};
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: {ColorScheme.TEXT_SECONDARY}; margin: 0;">{message}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if action_text and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_text, key=f"empty_state_action_{message}"):
                action_callback()


# ============================================================================
# HELP DIALOG FUNCTIONS
# ============================================================================

def show_vegetation_index_help(index_name: str):
    """Show help dialog for a vegetation index"""
    
    help_text = HelpText.VEGETATION_INDICES.get(index_name, "No help available")
    
    with st.expander(f"ℹ️ About {index_name}", expanded=False):
        st.markdown(help_text)


def show_alert_severity_help(severity: str):
    """Show help dialog for alert severity"""
    
    help_text = HelpText.ALERT_SEVERITIES.get(severity.lower(), "No help available")
    
    with st.expander(f"ℹ️ About {severity.title()} Alerts", expanded=False):
        st.markdown(help_text)


def show_feature_help(feature_name: str):
    """Show help dialog for a feature"""
    
    help_text = HelpText.FEATURES.get(feature_name, "No help available")
    
    with st.expander(f"ℹ️ About {feature_name.replace('_', ' ').title()}", expanded=False):
        st.markdown(help_text)


# ============================================================================
# RESPONSIVE DESIGN UTILITIES
# ============================================================================

def get_column_config(screen_size: str = "desktop") -> List[int]:
    """
    Get responsive column configuration
    
    Args:
        screen_size: "mobile", "tablet", or "desktop"
    
    Returns:
        List of column ratios
    """
    
    configs = {
        "mobile": [1],
        "tablet": [1, 1],
        "desktop": [1, 1, 1, 1]
    }
    
    return configs.get(screen_size, configs["desktop"])


def is_mobile() -> bool:
    """Check if viewing on mobile device (simplified)"""
    # In production, this would check actual viewport width
    return False


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_number(value: float, decimals: int = 2, unit: str = "") -> str:
    """Format number with consistent styling"""
    
    formatted = f"{value:.{decimals}f}"
    return f"{formatted} {unit}".strip()


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage with consistent styling"""
    
    return f"{value:.{decimals}f}%"


def format_date(date: datetime, format_str: str = "%Y-%m-%d") -> str:
    """Format date with consistent styling"""
    
    return date.strftime(format_str)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"
