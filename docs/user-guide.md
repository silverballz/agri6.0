# Agricultural Monitoring Platform - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Field Monitoring](#field-monitoring)
4. [Temporal Analysis](#temporal-analysis)
5. [Alerts and Notifications](#alerts-and-notifications)
6. [Data Export](#data-export)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Accessing the Platform
1. Open your web browser and navigate to the platform URL
2. The dashboard will load automatically with the latest field data
3. Use the sidebar navigation to access different features

### First Time Setup
1. **Define Monitoring Zones**: Start by creating monitoring zones for your fields
2. **Upload Satellite Data**: Import Sentinel-2A imagery for your area of interest
3. **Configure Sensors**: Set up environmental sensor integration if available
4. **Set Alert Thresholds**: Configure alert parameters for your specific crops

## Dashboard Overview

### Main Interface Components

#### Navigation Sidebar
- **Overview**: Main dashboard with key metrics and maps
- **Field Monitoring**: Detailed field analysis and health maps
- **Temporal Analysis**: Time series charts and trend analysis
- **Alerts**: Active alerts and notification management
- **Data Export**: Download reports and raw data

#### Key Metrics Panel
Located at the top of the dashboard, showing:
- **Total Monitored Area**: Sum of all monitoring zones
- **Active Alerts**: Number of current alerts by severity
- **Data Freshness**: Time since last satellite data update
- **System Health**: Overall platform status

### Understanding the Color Coding

#### Vegetation Health Colors
- 🟢 **Green (0.7-1.0 NDVI)**: Healthy, vigorous vegetation
- 🟡 **Yellow (0.4-0.7 NDVI)**: Moderate vegetation health
- 🟠 **Orange (0.2-0.4 NDVI)**: Stressed vegetation
- 🔴 **Red (0.0-0.2 NDVI)**: Severely stressed or bare soil
- ⚫ **Black (<0.0 NDVI)**: Water bodies or non-vegetated areas

#### Alert Severity Colors
- 🔴 **Critical**: Immediate attention required
- 🟠 **High**: Action needed within 24 hours
- 🟡 **Medium**: Monitor closely, action within 48 hours
- 🔵 **Low**: Informational, no immediate action required

## Field Monitoring

### Interactive Map Features

#### Viewing Field Health
1. **Zoom and Pan**: Use mouse wheel to zoom, click and drag to pan
2. **Layer Selection**: Choose between different vegetation indices:
   - NDVI (Normalized Difference Vegetation Index)
   - SAVI (Soil Adjusted Vegetation Index)
   - EVI (Enhanced Vegetation Index)
   - NDWI (Normalized Difference Water Index)

#### Monitoring Zone Management
1. **Create New Zone**:
   - Click "Add Monitoring Zone" button
   - Draw polygon on map by clicking points
   - Double-click to complete the polygon
   - Enter zone details (name, crop type, planting date)

2. **Edit Existing Zone**:
   - Click on any monitoring zone
   - Select "Edit Zone" from popup
   - Modify boundaries or zone information

3. **Zone Information Panel**:
   - **Zone Statistics**: Average NDVI, area coverage, pixel count
   - **Crop Information**: Type, planting date, expected harvest
   - **Recent Trends**: 7-day and 30-day vegetation index changes
   - **Environmental Data**: Latest sensor readings (if available)

### Interpreting Vegetation Indices

#### NDVI (Normalized Difference Vegetation Index)
- **Range**: -1 to +1
- **Interpretation**:
  - 0.8-1.0: Dense, healthy vegetation
  - 0.6-0.8: Moderate vegetation density
  - 0.4-0.6: Sparse vegetation
  - 0.2-0.4: Very sparse vegetation
  - <0.2: Bare soil, water, or stressed vegetation

#### SAVI (Soil Adjusted Vegetation Index)
- **Purpose**: Better for areas with exposed soil
- **Range**: -1 to +1
- **Use Case**: Early growth stages, sparse vegetation

#### EVI (Enhanced Vegetation Index)
- **Purpose**: Improved sensitivity in high biomass areas
- **Range**: -1 to +1
- **Use Case**: Dense vegetation, canopy monitoring

#### NDWI (Normalized Difference Water Index)
- **Purpose**: Water content and irrigation monitoring
- **Range**: -1 to +1
- **Interpretation**:
  - >0.3: High water content
  - 0.1-0.3: Moderate water content
  - <0.1: Low water content

## Temporal Analysis

### Time Series Charts

#### Accessing Temporal Data
1. Navigate to "Temporal Analysis" in the sidebar
2. Select monitoring zone(s) from dropdown
3. Choose date range using the date picker
4. Select vegetation indices to display

#### Chart Features
- **Zoom**: Click and drag to zoom into specific time periods
- **Pan**: Hold Shift and drag to pan across time
- **Hover**: Mouse over data points for exact values
- **Legend**: Click legend items to show/hide data series

#### Trend Interpretation
- **Upward Trend**: Improving vegetation health
- **Downward Trend**: Declining vegetation health
- **Seasonal Patterns**: Normal crop growth cycles
- **Sudden Drops**: Potential stress events or harvest

### Statistical Analysis

#### Available Metrics
- **Mean**: Average vegetation index value
- **Standard Deviation**: Variability within the zone
- **Trend Line**: Linear regression showing overall direction
- **Confidence Intervals**: Statistical uncertainty bounds

#### Comparative Analysis
- **Multi-Zone Comparison**: Compare up to 5 zones simultaneously
- **Year-over-Year**: Compare current season with previous years
- **Benchmark Comparison**: Compare against regional averages

## Alerts and Notifications

### Alert Types

#### Vegetation Stress Alerts
- **Sudden NDVI Drop**: >20% decrease in 7 days
- **Prolonged Low NDVI**: Below threshold for >14 days
- **Irregular Patterns**: Unexpected vegetation behavior

#### Environmental Alerts
- **Extreme Weather**: Temperature, humidity, or precipitation extremes
- **Soil Moisture**: Critical low or high soil moisture levels
- **Pest Risk**: Conditions favorable for pest outbreaks

#### System Alerts
- **Data Quality**: Poor satellite image quality or missing data
- **Sensor Malfunction**: Environmental sensor connectivity issues
- **Model Performance**: AI model accuracy degradation

### Managing Alerts

#### Alert Dashboard
1. Navigate to "Alerts" in the sidebar
2. View active alerts sorted by severity and time
3. Use filters to show specific alert types or zones

#### Alert Actions
- **Acknowledge**: Mark alert as seen (stops notifications)
- **Resolve**: Mark issue as resolved
- **Snooze**: Temporarily suppress notifications (1-24 hours)
- **Add Notes**: Document actions taken or observations

#### Notification Settings
- **Email Notifications**: Configure email alerts for different severities
- **SMS Alerts**: Set up text message notifications for critical alerts
- **Dashboard Notifications**: In-app notification preferences

## Data Export

### Available Export Formats

#### Reports
- **PDF Summary Reports**: Comprehensive field condition reports
- **Executive Dashboards**: High-level overview for management
- **Technical Reports**: Detailed analysis with statistics

#### Raw Data
- **CSV Files**: Vegetation index time series data
- **GeoTIFF Images**: Processed satellite imagery
- **GeoJSON**: Monitoring zone boundaries and metadata
- **Excel Workbooks**: Multi-sheet reports with charts

### Export Process
1. Navigate to "Data Export" in the sidebar
2. Select export type (Report or Raw Data)
3. Choose date range and monitoring zones
4. Select format and customization options
5. Click "Generate Export" and wait for processing
6. Download file when ready

### Scheduled Reports
- **Daily Summaries**: Automated daily condition reports
- **Weekly Trends**: Weekly vegetation trend analysis
- **Monthly Reports**: Comprehensive monthly field reports
- **Custom Schedules**: Configure custom report timing

## Troubleshooting

### Common Issues

#### Map Not Loading
**Symptoms**: Blank map or loading spinner
**Solutions**:
1. Check internet connection
2. Refresh the browser page
3. Clear browser cache and cookies
4. Try a different browser

#### No Satellite Data
**Symptoms**: "No data available" message
**Solutions**:
1. Check if satellite data exists for your date range
2. Verify monitoring zones are properly defined
3. Check for cloud coverage in satellite imagery
4. Contact administrator if data should be available

#### Slow Performance
**Symptoms**: Slow loading, unresponsive interface
**Solutions**:
1. Reduce the number of monitoring zones displayed
2. Limit date range for temporal analysis
3. Close other browser tabs
4. Check system resources (CPU, memory)

#### Alert Not Triggering
**Symptoms**: Expected alerts not appearing
**Solutions**:
1. Verify alert thresholds are properly configured
2. Check if monitoring zone has recent data
3. Ensure notification settings are enabled
4. Review alert history for similar conditions

### Getting Help

#### Documentation
- **User Guide**: This document (comprehensive usage instructions)
- **Technical Documentation**: System administration and API reference
- **FAQ**: Frequently asked questions and solutions
- **Video Tutorials**: Step-by-step video guides

#### Support Channels
- **Help Desk**: Submit support tickets for technical issues
- **User Forum**: Community discussions and tips
- **Training Sessions**: Scheduled group training sessions
- **One-on-One Support**: Individual training and consultation

#### Best Practices
1. **Regular Monitoring**: Check the dashboard daily during growing season
2. **Alert Response**: Respond to alerts promptly, especially critical ones
3. **Data Backup**: Regularly export important data and reports
4. **Zone Maintenance**: Keep monitoring zones updated with current crop information
5. **Feedback**: Provide feedback on system performance and feature requests

---

*For technical support or questions about this guide, please contact the system administrator or submit a support ticket through the help desk.*