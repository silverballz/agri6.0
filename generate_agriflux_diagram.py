#!/usr/bin/env python3
"""
AgriFlux Platform Architecture Diagram Generator
Creates a comprehensive visual summary of the agricultural monitoring solution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg

# Set up the figure with high DPI for crisp output
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
fig.patch.set_facecolor('#f8f9fa')

# Define color palette
colors = {
    'primary': '#2E7D32',      # Green
    'secondary': '#1976D2',    # Blue
    'accent': '#FF6F00',       # Orange
    'success': '#388E3C',      # Dark Green
    'warning': '#F57C00',      # Amber
    'info': '#0288D1',         # Light Blue
    'background': '#E8F5E8',   # Light Green
    'text': '#1B5E20',         # Dark Green
    'satellite': '#4A90E2',    # Satellite Blue
    'ai': '#9C27B0',           # Purple for AI
    'sensor': '#FF5722'        # Red-Orange for sensors
}

# Clear axes
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title Section
title_box = FancyBboxPatch((5, 90), 90, 8, 
                          boxstyle="round,pad=0.5", 
                          facecolor=colors['primary'], 
                          edgecolor='white', 
                          linewidth=2)
ax.add_patch(title_box)
ax.text(50, 94, 'AgriFlux: AI-Powered Agricultural Monitoring Platform', 
        fontsize=24, fontweight='bold', color='white', ha='center', va='center')

# Subtitle
ax.text(50, 87, 'Satellite-Based Crop Health Monitoring & Predictive Analytics', 
        fontsize=14, color=colors['text'], ha='center', va='center', style='italic')

# Data Sources Section (Top Left)
data_sources_box = FancyBboxPatch((2, 70), 28, 15, 
                                 boxstyle="round,pad=0.5", 
                                 facecolor=colors['background'], 
                                 edgecolor=colors['primary'], 
                                 linewidth=2)
ax.add_patch(data_sources_box)
ax.text(16, 82, 'DATA SOURCES', fontsize=12, fontweight='bold', 
        color=colors['text'], ha='center')

# Satellite icon representation
satellite_circle = Circle((8, 78), 1.5, facecolor=colors['satellite'], edgecolor='white')
ax.add_patch(satellite_circle)
ax.text(8, 78, '🛰️', fontsize=16, ha='center', va='center')
ax.text(12, 78, 'Sentinel-2A\nSatellite Data', fontsize=9, ha='left', va='center')

# Sensor icon
sensor_circle = Circle((8, 74), 1.5, facecolor=colors['sensor'], edgecolor='white')
ax.add_patch(sensor_circle)
ax.text(8, 74, '📡', fontsize=16, ha='center', va='center')
ax.text(12, 74, 'IoT Sensors\n& Weather Data', fontsize=9, ha='left', va='center')

# Processing Pipeline (Center)
pipeline_box = FancyBboxPatch((35, 45), 30, 35, 
                             boxstyle="round,pad=0.5", 
                             facecolor='white', 
                             edgecolor=colors['secondary'], 
                             linewidth=3)
ax.add_patch(pipeline_box)
ax.text(50, 76, 'AI PROCESSING PIPELINE', fontsize=12, fontweight='bold', 
        color=colors['secondary'], ha='center')

# Processing steps
steps = [
    ('Data Fusion', 72, colors['info']),
    ('Cloud Masking', 68, colors['warning']),
    ('Vegetation Indices', 64, colors['success']),
    ('CNN Analysis', 60, colors['ai']),
    ('LSTM Prediction', 56, colors['ai']),
    ('Risk Assessment', 52, colors['accent'])
]

for i, (step, y_pos, color) in enumerate(steps):
    step_box = FancyBboxPatch((37, y_pos-1), 26, 3, 
                             boxstyle="round,pad=0.2", 
                             facecolor=color, 
                             alpha=0.8)
    ax.add_patch(step_box)
    ax.text(50, y_pos+0.5, step, fontsize=10, fontweight='bold', 
            color='white', ha='center', va='center')
    
    # Add arrow between steps
    if i < len(steps) - 1:
        ax.arrow(50, y_pos-1.5, 0, -1.5, head_width=0.8, head_length=0.3, 
                fc=colors['text'], ec=colors['text'], alpha=0.6)

# Output & Applications (Right Side)
output_box = FancyBboxPatch((70, 45), 28, 35, 
                           boxstyle="round,pad=0.5", 
                           facecolor=colors['background'], 
                           edgecolor=colors['success'], 
                           linewidth=2)
ax.add_patch(output_box)
ax.text(84, 76, 'SMART OUTPUTS', fontsize=12, fontweight='bold', 
        color=colors['text'], ha='center')

# Output features
outputs = [
    ('📊 Real-time Dashboard', 72),
    ('⚠️ Early Warning Alerts', 68),
    ('📈 Predictive Analytics', 64),
    ('🗺️ Crop Health Maps', 60),
    ('📱 Mobile Notifications', 56),
    ('📋 Automated Reports', 52)
]

for output, y_pos in outputs:
    ax.text(72, y_pos, output, fontsize=10, ha='left', va='center', 
            color=colors['text'])

# Key Features Section (Bottom)
features_box = FancyBboxPatch((2, 25), 96, 15, 
                             boxstyle="round,pad=0.5", 
                             facecolor='white', 
                             edgecolor=colors['accent'], 
                             linewidth=2)
ax.add_patch(features_box)
ax.text(50, 37, 'KEY INNOVATIONS & COMPETITIVE ADVANTAGES', 
        fontsize=14, fontweight='bold', color=colors['accent'], ha='center')

# Feature columns
feature_cols = [
    {
        'title': 'MULTI-SOURCE FUSION',
        'items': ['Satellite + IoT Integration', 'Weather Data Correlation', 'Real-time Processing'],
        'x': 15, 'color': colors['info']
    },
    {
        'title': 'AI-POWERED ANALYTICS',
        'items': ['CNN Spatial Analysis', 'LSTM Temporal Prediction', 'Risk Assessment Models'],
        'x': 35, 'color': colors['ai']
    },
    {
        'title': 'SCALABLE ARCHITECTURE',
        'items': ['Cloud-Native Design', 'Edge Computing Ready', 'API-First Approach'],
        'x': 55, 'color': colors['secondary']
    },
    {
        'title': 'COST-EFFECTIVE SOLUTION',
        'items': ['Free Sentinel-2 Data', 'Open Source Foundation', 'Minimal Hardware Needs'],
        'x': 75, 'color': colors['success']
    }
]

for col in feature_cols:
    # Column header
    header_box = FancyBboxPatch((col['x']-8, 33), 16, 2.5, 
                               boxstyle="round,pad=0.2", 
                               facecolor=col['color'], 
                               alpha=0.8)
    ax.add_patch(header_box)
    ax.text(col['x'], 34.2, col['title'], fontsize=9, fontweight='bold', 
            color='white', ha='center', va='center')
    
    # Feature items
    for i, item in enumerate(col['items']):
        ax.text(col['x'], 31-i*1.5, f"• {item}", fontsize=8, 
                ha='center', va='center', color=colors['text'])

# Problem-Solution Flow (Bottom Section)
flow_box = FancyBboxPatch((2, 5), 96, 15, 
                         boxstyle="round,pad=0.5", 
                         facecolor='#FFF3E0', 
                         edgecolor=colors['warning'], 
                         linewidth=2)
ax.add_patch(flow_box)
ax.text(50, 17, 'PROBLEM → SOLUTION TRANSFORMATION', 
        fontsize=14, fontweight='bold', color=colors['warning'], ha='center')

# Problem-Solution pairs
transformations = [
    ('Manual Monitoring', 'Automated 24/7 Surveillance', 12, 15),
    ('Reactive Management', 'Predictive Early Warnings', 37, 15),
    ('Resource Waste', 'Precision Agriculture', 62, 15),
    ('Limited Coverage', 'Satellite-Scale Monitoring', 87, 15)
]

for problem, solution, x, y in transformations:
    # Problem box
    prob_box = FancyBboxPatch((x-10, y-2), 20, 2, 
                             boxstyle="round,pad=0.2", 
                             facecolor='#FFCDD2', 
                             edgecolor='#D32F2F')
    ax.add_patch(prob_box)
    ax.text(x, y-1, problem, fontsize=8, ha='center', va='center', 
            color='#D32F2F', fontweight='bold')
    
    # Arrow
    ax.arrow(x, y-2.5, 0, -2, head_width=1, head_length=0.3, 
            fc=colors['success'], ec=colors['success'])
    
    # Solution box
    sol_box = FancyBboxPatch((x-10, y-7), 20, 2, 
                            boxstyle="round,pad=0.2", 
                            facecolor='#C8E6C9', 
                            edgecolor=colors['success'])
    ax.add_patch(sol_box)
    ax.text(x, y-6, solution, fontsize=8, ha='center', va='center', 
            color=colors['success'], fontweight='bold')

# Add connecting arrows between main sections
# Data Sources to Processing
ax.arrow(30, 77, 4, 0, head_width=1, head_length=1, 
        fc=colors['primary'], ec=colors['primary'], linewidth=2)

# Processing to Outputs
ax.arrow(65, 62, 4, 0, head_width=1, head_length=1, 
        fc=colors['secondary'], ec=colors['secondary'], linewidth=2)

# Add decorative elements
# Corner decorations
for corner in [(5, 95), (95, 95), (5, 5), (95, 5)]:
    circle = Circle(corner, 1, facecolor=colors['accent'], alpha=0.3)
    ax.add_patch(circle)

# Technology badges
tech_badges = [
    ('Python', 85, 85, colors['info']),
    ('TensorFlow', 90, 85, colors['ai']),
    ('PostgreSQL', 85, 82, colors['secondary']),
    ('Streamlit', 90, 82, colors['success'])
]

for tech, x, y, color in tech_badges:
    badge = FancyBboxPatch((x-2, y-0.5), 4, 1, 
                          boxstyle="round,pad=0.1", 
                          facecolor=color, 
                          alpha=0.7)
    ax.add_patch(badge)
    ax.text(x, y, tech, fontsize=7, color='white', 
            ha='center', va='center', fontweight='bold')

# Add subtle grid pattern
for i in range(0, 101, 20):
    ax.axvline(x=i, color='lightgray', alpha=0.1, linewidth=0.5)
    ax.axhline(y=i, color='lightgray', alpha=0.1, linewidth=0.5)

# Final touches
plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save the diagram
plt.savefig('agriflux_platform_diagram.png', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='#f8f9fa', 
           edgecolor='none',
           pad_inches=0.2)

print("✅ AgriFlux platform diagram saved as 'agriflux_platform_diagram.png'")
print("📊 High-resolution diagram showcasing:")
print("   • Multi-source data integration")
print("   • AI-powered processing pipeline") 
print("   • Smart agricultural outputs")
print("   • Key innovations & competitive advantages")
print("   • Problem-solution transformation flow")

plt.show()