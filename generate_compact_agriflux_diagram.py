#!/usr/bin/env python3
"""
Compact AgriFlux Platform Diagram Generator
Creates a concise visual summary of the agricultural monitoring solution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

# Set up compact figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.patch.set_facecolor('white')

# Define color palette
colors = {
    'primary': '#2E7D32',      # Green
    'secondary': '#1976D2',    # Blue
    'accent': '#FF6F00',       # Orange
    'ai': '#9C27B0',           # Purple
    'sensor': '#FF5722',       # Red-Orange
    'bg_light': '#E8F5E8'      # Light Green
}

# Clear axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
title_box = FancyBboxPatch((0.5, 7), 9, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors['primary'], 
                          edgecolor='none')
ax.add_patch(title_box)
ax.text(5, 7.4, 'AgriFlux: AI-Powered Agricultural Monitoring', 
        fontsize=16, fontweight='bold', color='white', ha='center', va='center')

# Data Sources (Left)
data_box = FancyBboxPatch((0.5, 5.5), 2, 1.2, 
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['sensor'], 
                         alpha=0.8)
ax.add_patch(data_box)
ax.text(1.5, 6.3, 'DATA SOURCES', fontsize=10, fontweight='bold', 
        color='white', ha='center')
ax.text(1.5, 5.9, '• Sentinel-2A Satellite', fontsize=8, color='white', ha='center')
ax.text(1.5, 5.7, '• IoT Sensors & Weather', fontsize=8, color='white', ha='center')

# AI Processing (Center)
ai_box = FancyBboxPatch((3.5, 4.5), 3, 2.2, 
                       boxstyle="round,pad=0.1", 
                       facecolor=colors['ai'], 
                       alpha=0.9)
ax.add_patch(ai_box)
ax.text(5, 6.4, 'AI PROCESSING', fontsize=11, fontweight='bold', 
        color='white', ha='center')

# Processing steps - compact
steps = ['Data Fusion', 'Cloud Masking', 'Vegetation Indices', 'CNN + LSTM', 'Risk Prediction']
for i, step in enumerate(steps):
    y_pos = 6.1 - i*0.25
    ax.text(5, y_pos, f'• {step}', fontsize=8, color='white', ha='center')

# Smart Outputs (Right)
output_box = FancyBboxPatch((7.5, 5.5), 2, 1.2, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colors['secondary'], 
                           alpha=0.8)
ax.add_patch(output_box)
ax.text(8.5, 6.3, 'SMART OUTPUTS', fontsize=10, fontweight='bold', 
        color='white', ha='center')
ax.text(8.5, 5.9, '• Real-time Dashboard', fontsize=8, color='white', ha='center')
ax.text(8.5, 5.7, '• Predictive Alerts', fontsize=8, color='white', ha='center')

# Arrows
ax.arrow(2.5, 6.1, 0.8, 0, head_width=0.1, head_length=0.1, 
        fc=colors['primary'], ec=colors['primary'], linewidth=2)
ax.arrow(6.5, 5.6, 0.8, 0, head_width=0.1, head_length=0.1, 
        fc=colors['primary'], ec=colors['primary'], linewidth=2)

# Key Features Section (Bottom)
features_box = FancyBboxPatch((0.5, 2.5), 9, 1.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['bg_light'], 
                             edgecolor=colors['accent'], 
                             linewidth=2)
ax.add_patch(features_box)
ax.text(5, 4.1, 'KEY INNOVATIONS', fontsize=12, fontweight='bold', 
        color=colors['accent'], ha='center')

# Feature grid - 2x2
features = [
    ('Multi-Source Fusion\n• Satellite + IoT Integration\n• Real-time Processing', 2, 3.5),
    ('AI-Powered Analytics\n• CNN Spatial Analysis\n• LSTM Predictions', 8, 3.5),
    ('Scalable Architecture\n• Cloud-Native Design\n• API-First Approach', 2, 2.9),
    ('Cost-Effective Solution\n• Free Sentinel Data\n• Open Source', 8, 2.9)
]

for feature_text, x, y in features:
    lines = feature_text.split('\n')
    ax.text(x, y, lines[0], fontsize=9, fontweight='bold', 
            color=colors['primary'], ha='center', va='top')
    for i, line in enumerate(lines[1:], 1):
        ax.text(x, y-0.15*i, line, fontsize=7, 
                color=colors['primary'], ha='center', va='top')

# Problem-Solution Flow (Bottom)
flow_box = FancyBboxPatch((0.5, 0.3), 9, 1.8, 
                         boxstyle="round,pad=0.1", 
                         facecolor='#FFF3E0', 
                         edgecolor=colors['accent'], 
                         linewidth=1)
ax.add_patch(flow_box)
ax.text(5, 1.9, 'PROBLEM → SOLUTION', fontsize=12, fontweight='bold', 
        color=colors['accent'], ha='center')

# Compact problem-solution pairs
transformations = [
    ('Manual Monitoring', 'Automated 24/7', 2.5),
    ('Reactive Management', 'Predictive Alerts', 5),
    ('Resource Waste', 'Precision Agriculture', 7.5)
]

for problem, solution, x in transformations:
    # Problem
    ax.text(x, 1.5, problem, fontsize=8, ha='center', va='center', 
            color='#D32F2F', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#FFCDD2', alpha=0.7))
    
    # Arrow
    ax.arrow(x, 1.3, 0, -0.2, head_width=0.1, head_length=0.05, 
            fc=colors['primary'], ec=colors['primary'])
    
    # Solution
    ax.text(x, 1.0, solution, fontsize=8, ha='center', va='center', 
            color=colors['primary'], fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#C8E6C9', alpha=0.7))

# Technology stack (small badges)
tech_y = 0.5
techs = ['Python', 'TensorFlow', 'PostgreSQL', 'Streamlit']
for i, tech in enumerate(techs):
    x_pos = 1.5 + i * 2
    ax.text(x_pos, tech_y, tech, fontsize=7, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.1", facecolor=colors['secondary'], alpha=0.7),
            color='white', fontweight='bold')

# Add subtle decorative elements
for corner in [(1, 7.5), (9, 7.5), (1, 0.5), (9, 0.5)]:
    circle = Circle(corner, 0.1, facecolor=colors['accent'], alpha=0.5)
    ax.add_patch(circle)

plt.tight_layout()

# Save the compact diagram
plt.savefig('agriflux_compact_diagram.png', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white', 
           edgecolor='none',
           pad_inches=0.1)

print("✅ Compact AgriFlux diagram saved as 'agriflux_compact_diagram.png'")
print("📊 Compact diagram includes:")
print("   • Data sources & AI processing pipeline")
print("   • Key innovations in 2x2 grid")
print("   • Problem-solution transformation")
print("   • Technology stack")
print("   • File size optimized for presentations")

plt.show()