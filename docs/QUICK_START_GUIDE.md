# AgriFlux Quick Start Guide 🌱

Welcome to AgriFlux! This guide will help you get started with intelligent agricultural monitoring in just a few minutes.

## Table of Contents
1. [First Time Setup](#first-time-setup)
2. [Understanding the Dashboard](#understanding-the-dashboard)
3. [Monitoring Your Fields](#monitoring-your-fields)
4. [Understanding Vegetation Indices](#understanding-vegetation-indices)
5. [Managing Alerts](#managing-alerts)
6. [Exporting Data](#exporting-data)
7. [Using Demo Mode](#using-demo-mode)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)

---

## First Time Setup

### Prerequisites
- Python 3.9 or higher
- Required dependencies installed (`pip install -r requirements.txt`)
- Processed satellite data (or use Demo Mode)

### Starting the Dashboard
```bash
# From the project root directory
streamlit run src/dashboard/main.py
```

The dashboard will open in your web browser at `http://localhost:8501`

---

## Understanding the Dashboard

### Main Navigation
The dashboard has 5 main pages accessible from the sidebar dropdown:

| Page | Purpose | Icon |
|------|---------|------|
| **Overview** | Summary of all fields | 📊 |
| **Field Monitoring** | Interactive maps | 🗺️ |
| **Temporal Analysis** | Trend analysis | 📈 |
| **Alerts & Notifications** | Alert management | 🚨 |
| **Data Export** | Export and reports | 📤 |

### Sidebar Controls
The sidebar contains global filters that apply across all pages:

- **Date Range**: Select time period for analysis
- **Monitoring Zones**: Choose which fields to display
- **Vegetation Indices**: Select NDVI, SAVI, EVI, NDWI, or NDSI
- **Demo Mode**: Load sample data for testing
- **System Status**: View system health indicators

---

## Monitoring Your Fields

### Step 1: Select Your Fields
1. Look for "Monitoring Zones" in the sidebar
2. Click the dropdown and select your fields
3. Leave empty to show all fields

### Step 2: Choose Date Range
1. Find "Date Range" in the sidebar
2. Click to open date picker
3. Select start and end dates
4. Satellite data is typically available every 5-10 days

### Step 3: Pick Vegetation Indices
1. Find "Vegetation Indices" in the sidebar
2. Select one or more indices:
   - **NDVI**: General crop health (recommended for beginners)
   - **SAVI**: Better for sparse vegetation
   - **EVI**: Enhanced for dense vegetation
   - **NDWI**: Water content monitoring
   - **NDSI**: Soil moisture assessment

### Step 4: Navigate to Overview
1. Select "📊 Overview" from the page dropdown
2. View key metrics at the top
3. Check the health summary chart
4. Review recent trends
5. Check active alerts

---

## Understanding Vegetation Indices

### NDVI (Normalized Difference Vegetation Index)
**Most widely used index for crop health**

| Value Range | Health Status | Color | Action |
|-------------|---------------|-------|--------|
| 0.8 - 1.0 | Excellent | 🟢 | Continue monitoring |
| 0.6 - 0.8 | Healthy | 🟢 | Normal operations |
| 0.4 - 0.6 | Moderate | 🟡 | Monitor closely |
| 0.2 - 0.4 | Stressed | 🟠 | Investigate |
| < 0.2 | Critical | 🔴 | Immediate action |

**Best for**: General crop health, biomass estimation, yield prediction

### SAVI (Soil Adjusted Vegetation Index)
**Better for sparse vegetation and exposed soil**

- Minimizes soil brightness effects
- Use when canopy coverage < 50%
- Ideal for early season crops

**Best for**: Early season monitoring, sparse canopy, exposed soil areas

### EVI (Enhanced Vegetation Index)
**Enhanced sensitivity in high biomass regions**

- Reduces atmospheric influences
- Better for dense vegetation
- Good for tropical regions

**Best for**: Dense canopy, high biomass crops, tropical agriculture

### NDWI (Normalized Difference Water Index)
**Measures vegetation water content**

- Essential for irrigation management
- Detects drought conditions early
- Monitors water stress

**Best for**: Irrigation scheduling, drought monitoring, water stress detection

### NDSI (Normalized Difference Soil Index)
**Detects bare soil and soil moisture**

- Monitors soil moisture conditions
- Useful for tillage monitoring
- Assesses planting readiness

**Best for**: Soil moisture assessment, planting readiness, tillage monitoring

---

## Managing Alerts

### Understanding Alert Severities

#### 🔴 Critical - Immediate Action (2-4 hours)
**Examples:**
- Severe vegetation stress (NDVI < 0.3)
- Extreme pest infestation
- Critical water shortage
- Disease outbreak

**Action**: Immediate field inspection and intervention required

#### 🟠 High Priority - Action Within 24 Hours
**Examples:**
- Moderate vegetation stress (NDVI 0.3-0.5)
- High pest risk conditions
- Irrigation system malfunction
- Nutrient deficiency

**Action**: Schedule field visit and prepare intervention

#### 🟡 Medium Priority - Monitor (48-72 hours)
**Examples:**
- Mild vegetation stress (NDVI 0.5-0.6)
- Moderate pest risk
- Suboptimal soil moisture
- Weather-related concerns

**Action**: Increase monitoring frequency, prepare contingency

#### 🟢 Low Priority - Informational
**Examples:**
- Minor variations in vegetation health
- Seasonal changes
- Preventive maintenance reminders
- Data quality notifications

**Action**: Continue routine monitoring

### Viewing Alerts
1. Navigate to "🚨 Alerts & Notifications" page
2. View active alerts at the top
3. Filter by severity or type
4. Click alert cards for details

### Acknowledging Alerts
1. Find the alert you want to acknowledge
2. Click the "✅ Acknowledge" button
3. Alert moves to history
4. Use "Acknowledge All Visible" for bulk actions

### Setting Up Notifications
1. Go to Alerts page
2. Find "Notification Settings" in sidebar
3. Enable email, SMS, or push notifications
4. Choose notification frequency

---

## Exporting Data

### Quick Export
1. Navigate to "📤 Data Export" page
2. Look for "Quick Exports" section
3. Click "Export" on pre-configured options:
   - Current Week NDVI
   - Alert Summary Report
   - Zone Boundaries
   - Sensor Status Report

### Custom Export
1. Go to "Data Export" page
2. Select "Data Type" (Vegetation Indices, Satellite Images, etc.)
3. Choose "Zones" to include
4. Set "Date Range"
5. Select "Format" (CSV, Excel, GeoTIFF, PDF, etc.)
6. Click "📤 Export Data"
7. Download the generated file

### Available Formats

| Format | Best For | Compatible With |
|--------|----------|-----------------|
| CSV | Time series data | Excel, R, Python |
| Excel | Reports, presentations | Microsoft Office |
| GeoTIFF | Satellite imagery | QGIS, ArcGIS |
| GeoJSON | Field boundaries | Web maps, GIS |
| PDF | Professional reports | Stakeholders |
| Shapefile | Spatial analysis | GIS software |

### Generating Reports
1. Go to "Data Export" page
2. Click "Report Generation" tab
3. Choose template:
   - **Standard**: Comprehensive analysis
   - **Executive**: High-level overview
   - **Technical**: Detailed technical analysis
   - **Field**: Operational summary
4. Select zones and date range
5. Choose options (maps, charts, recommendations)
6. Click "📋 Generate Report"
7. Download PDF

---

## Using Demo Mode

Demo Mode is perfect for:
- Training new users
- Demonstrating features to stakeholders
- Testing dashboard functionality
- Presentations and demos

### Enabling Demo Mode
1. Look for "Demo Mode" section in sidebar
2. Check "Enable Demo Mode" checkbox
3. Wait for data to load
4. Select a scenario:
   - 🌱 **Healthy Field**: Optimal conditions
   - ⚠️ **Stressed Field**: Multiple issues
   - 🔄 **Mixed Field**: Varied conditions

### What's Included
- 3 field scenarios with different health conditions
- 5 time points for temporal analysis
- Sample alerts at all severity levels
- AI predictions with confidence scores
- Complete time series data

### Exiting Demo Mode
1. Find "Demo Mode" section in sidebar
2. Click "🚪 Exit Demo Mode" button
3. Dashboard returns to real data

---

## Common Tasks

### Task 1: Daily Field Check
1. Open dashboard
2. Go to "Overview" page
3. Check key metrics at top
4. Review active alerts
5. Look at health summary chart
6. Check recent trends

**Time**: 2-3 minutes

### Task 2: Investigate Field Issue
1. Note the field with issue from Overview
2. Navigate to "Field Monitoring" page
3. Select the field from dropdown
4. Choose relevant vegetation index
5. Enable AI predictions
6. Click on affected zone for details
7. Check recommendations

**Time**: 5-10 minutes

### Task 3: Analyze Trends
1. Go to "Temporal Analysis" page
2. Select vegetation indices
3. Choose date range (30-90 days recommended)
4. Enable confidence intervals
5. Toggle anomaly highlighting
6. Review trend statistics
7. Export trend analysis if needed

**Time**: 5-10 minutes

### Task 4: Weekly Report
1. Navigate to "Data Export" page
2. Click "Report Generation" tab
3. Select "Executive" template
4. Choose all zones
5. Set date range to last 7 days
6. Enable maps and charts
7. Generate and download PDF
8. Email to stakeholders

**Time**: 3-5 minutes

---

## Troubleshooting

### Problem: Dashboard won't load
**Solutions:**
1. Check if all dependencies are installed: `pip install -r requirements.txt`
2. Verify Python version: `python --version` (should be 3.9+)
3. Try restarting: `streamlit run src/dashboard/main.py`
4. Check logs in `logs/dashboard.log`

### Problem: No data showing
**Solutions:**
1. Check if data has been processed: `python scripts/process_sentinel2_data.py`
2. Verify database exists: `ls data/agriflux.db`
3. Try Demo Mode to test dashboard functionality
4. Check System Status in sidebar

### Problem: Maps not displaying
**Solutions:**
1. Check internet connection (maps require online access)
2. Verify GeoTIFF files exist in `data/processed/`
3. Try refreshing the page
4. Check browser console for errors

### Problem: Alerts not generating
**Solutions:**
1. Verify processed imagery exists in database
2. Check alert thresholds in configuration
3. Run alert generation manually: `python scripts/generate_alerts.py`
4. Review logs for errors

### Problem: Export fails
**Solutions:**
1. Check disk space
2. Verify export directory exists and is writable
3. Try a different export format
4. Check logs for specific error messages

### Problem: Slow performance
**Solutions:**
1. Disable auto-refresh in sidebar
2. Reduce date range for analysis
3. Select fewer zones
4. Clear browser cache
5. Close other browser tabs

---

## Getting Help

### Documentation
- 📖 [User Guide](user-guide.md) - Comprehensive user documentation
- 🔧 [Technical Documentation](technical-documentation.md) - Technical details
- 📚 [FAQ](faq.md) - Frequently asked questions

### In-Dashboard Help
- Click "❓ Help & Documentation" button in sidebar
- Use "ℹ️" icons next to features for tooltips
- Expand "Quick Help" sections in sidebar
- Check page-specific help in expanders

### Support
- 💬 Email: support@agriflux.com
- 📞 Phone: 1-800-AGRIFLUX
- 🌐 Website: www.agriflux.com/support

### Training
- Weekly webinars every Tuesday at 2 PM
- Personalized onboarding sessions available
- Video tutorials on website
- Interactive demos in Demo Mode

---

## Next Steps

Now that you're familiar with the basics:

1. ✅ **Explore Demo Mode** - Practice with sample data
2. ✅ **Process Your Data** - Run data processing pipeline
3. ✅ **Set Up Alerts** - Configure notification preferences
4. ✅ **Create Reports** - Generate your first weekly report
5. ✅ **Analyze Trends** - Review historical vegetation data
6. ✅ **Train Your Team** - Share this guide with colleagues

**Happy Monitoring! 🌱**

---

*Last Updated: December 2024*
*Version: 1.0*
