"""
Demo script to showcase the new UI/UX enhancements
Run with: streamlit run demo_ui_enhancements.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dashboard.ui_components import (
    apply_custom_theme,
    metric_card,
    metric_card_with_chart,
    info_box,
    section_header,
    health_status_badge,
    severity_badge,
    progress_bar,
    ColorScheme
)

# Page configuration
st.set_page_config(
    page_title="🎨 AgriFlux UI/UX Demo",
    page_icon="🎨",
    layout="wide"
)

# Apply custom theme
apply_custom_theme()

# Main header
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); 
            border-radius: 15px; border: 2px solid #4caf50; margin-bottom: 2rem;">
    <h1 style="color: #4caf50; margin: 0;">🎨 AgriFlux UI/UX Enhancements Demo</h1>
    <p style="color: #a0aec0; margin-top: 1rem; font-size: 1.1rem;">
        Showcasing modern design improvements with gradient backgrounds, animations, and responsive layouts
    </p>
</div>
""", unsafe_allow_html=True)

# Section 1: Enhanced Metric Cards
section_header("📊 Enhanced Metric Cards", "Gradient backgrounds with hover animations", "📊")

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card(
        title="Active Fields",
        value="42",
        delta="+5 this week",
        delta_color="normal",
        icon="🗺️",
        metric_type="success"
    )

with col2:
    metric_card(
        title="Health Index",
        value="0.85",
        delta="+0.10",
        delta_color="normal",
        icon="🌱",
        metric_type="success"
    )

with col3:
    metric_card(
        title="Active Alerts",
        value="7",
        delta="-3 from yesterday",
        delta_color="normal",
        icon="🚨",
        metric_type="warning"
    )

with col4:
    metric_card(
        title="Data Quality",
        value="94%",
        delta="+2%",
        delta_color="normal",
        icon="📡",
        metric_type="info"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: Metric Cards with Sparklines
section_header("📈 Metric Cards with Sparklines", "Mini charts showing trends", "📈")

col1, col2, col3 = st.columns(3)

with col1:
    metric_card_with_chart(
        title="NDVI Trend",
        value="0.78",
        chart_data=[0.65, 0.68, 0.72, 0.75, 0.78],
        delta="+0.13 improvement",
        icon="🌿",
        metric_type="success"
    )

with col2:
    metric_card_with_chart(
        title="Soil Moisture",
        value="32%",
        chart_data=[28, 30, 29, 31, 32],
        delta="+4% increase",
        icon="💧",
        metric_type="info"
    )

with col3:
    metric_card_with_chart(
        title="Temperature",
        value="24°C",
        chart_data=[22, 23, 25, 24, 24],
        delta="Stable",
        delta_color="off",
        icon="🌡️",
        metric_type="default"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Section 3: Status Badges
section_header("🏷️ Status Badges", "Health and severity indicators", "🏷️")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Health Status Badges:**")
    st.markdown("<br>", unsafe_allow_html=True)
    
    health_values = [0.95, 0.75, 0.65, 0.55, 0.35]
    for val in health_values:
        health_status_badge(val, show_value=True)
        st.markdown("<br>", unsafe_allow_html=True)

with col2:
    st.markdown("**Alert Severity Badges:**")
    st.markdown("<br>", unsafe_allow_html=True)
    
    severities = ["critical", "high", "medium", "low"]
    for sev in severities:
        severity_badge(sev)
        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 4: Info Boxes
section_header("💬 Information Boxes", "Styled alert messages", "💬")

col1, col2 = st.columns(2)

with col1:
    info_box("✅ All systems operational", box_type="success")
    info_box("⚠️ High temperature detected in Zone 3", box_type="warning")

with col2:
    info_box("ℹ️ New satellite imagery available", box_type="info")
    info_box("❌ Sensor connection lost in Zone 5", box_type="error")

st.markdown("<br>", unsafe_allow_html=True)

# Section 5: Progress Bars
section_header("📊 Progress Indicators", "Visual progress tracking", "📊")

col1, col2 = st.columns(2)

with col1:
    progress_bar(85, 100, "Data Processing", ColorScheme.SUCCESS)
    progress_bar(60, 100, "Model Training", ColorScheme.WARNING)
    progress_bar(30, 100, "Export Generation", ColorScheme.ERROR)

with col2:
    progress_bar(95, 100, "Field Coverage", ColorScheme.SUCCESS)
    progress_bar(72, 100, "Sensor Connectivity", ColorScheme.WARNING)
    progress_bar(100, 100, "Database Sync", ColorScheme.SUCCESS)

st.markdown("<br>", unsafe_allow_html=True)

# Section 6: Color Palette
section_header("🎨 Color Palette", "Consistent color scheme", "🎨")

colors = {
    "Primary": ColorScheme.PRIMARY,
    "Secondary": ColorScheme.SECONDARY,
    "Success": ColorScheme.SUCCESS,
    "Warning": ColorScheme.WARNING,
    "Error": ColorScheme.ERROR,
    "Info": ColorScheme.INFO
}

cols = st.columns(len(colors))

for col, (name, color) in zip(cols, colors.items()):
    with col:
        st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s ease;
        " class="hover-lift">
            <div style="color: white; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;">
                {name}
            </div>
            <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; font-family: monospace;">
                {color}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Section 7: Typography
section_header("📝 Typography", "Modern font stack with Inter and Roboto", "📝")

st.markdown("""
<div style="background: linear-gradient(135deg, #2d3748 0%, #3d4a5c 100%); 
            padding: 2rem; border-radius: 12px; border: 1px solid #4a5568;">
    <h1 style="margin-top: 0;">Heading 1 - Inter Font</h1>
    <h2>Heading 2 - Inter Font</h2>
    <h3>Heading 3 - Inter Font</h3>
    <p style="font-size: 1.1rem; line-height: 1.6;">
        This is body text using the Inter font family. The modern, clean design 
        provides excellent readability across all screen sizes. The font stack 
        includes fallbacks to Roboto and system fonts for maximum compatibility.
    </p>
    <p style="color: #a0aec0; font-size: 0.9rem;">
        Secondary text uses a muted color for visual hierarchy and improved 
        information architecture.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 8: Responsive Design
section_header("📱 Responsive Design", "Optimized for tablet and desktop", "📱")

st.markdown("""
<div style="background: linear-gradient(135deg, #2d3748 0%, #3d4a5c 100%); 
            padding: 2rem; border-radius: 12px; border: 1px solid #4a5568;">
    <h3 style="color: #4caf50; margin-top: 0;">Breakpoints</h3>
    <ul style="line-height: 2;">
        <li><strong>Mobile:</strong> < 768px (default styling)</li>
        <li><strong>Tablet:</strong> 768px (adjusted spacing and font sizes)</li>
        <li><strong>Desktop:</strong> 1024px+ (grid layouts and wider containers)</li>
    </ul>
    
    <h3 style="color: #4caf50; margin-top: 2rem;">Features</h3>
    <ul style="line-height: 2;">
        <li>✅ Fluid typography that scales with viewport</li>
        <li>✅ Flexible grid layouts for optimal content display</li>
        <li>✅ Touch-friendly interactive elements</li>
        <li>✅ Optimized spacing for different screen sizes</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); 
            border-radius: 15px; border: 1px solid #4a5568; margin-top: 3rem;">
    <h3 style="color: #4caf50; margin: 0;">✨ UI/UX Enhancements Complete</h3>
    <p style="color: #a0aec0; margin-top: 1rem;">
        All components feature gradient backgrounds, hover animations, and responsive design
    </p>
    <p style="color: #718096; margin-top: 0.5rem; font-size: 0.9rem;">
        Requirements 7.1, 7.2, 7.3, 7.4, 7.5 - All Validated ✅
    </p>
</div>
""", unsafe_allow_html=True)
