# 🌱 AgriFlux - Local Development

## 🚀 **Quick Start - Run Locally**

### **Option 1: Simple Run (Recommended)**
```bash
# Clone the repository
git clone https://github.com/your-username/agriflux.git
cd agriflux

# Run the local launcher (installs dependencies automatically)
python run_local.py
```

### **Option 2: Manual Setup**
```bash
# Install dependencies
pip install streamlit pandas numpy plotly

# Run the dashboard
streamlit run src/dashboard/main.py
```

### **Access Your Dashboard**
- Open your browser to `http://localhost:8501`
- Explore the AgriFlux agricultural intelligence platform!

## 🎯 **What You'll See**

### 🌱 **AgriFlux Features:**
- **Dark theme** agricultural dashboard
- **Interactive navigation** with 5 main pages
- **Real-time metrics** showing field health
- **Smart alerts system** with severity levels
- **Agricultural zones** focused on Punjab, India
- **Vegetation health monitoring** with NDVI charts
- **Weather integration** and soil moisture data
- **Mobile responsive** design

### 📊 **Dashboard Pages:**
1. **📊 Overview** - Main dashboard with key metrics and field overview
2. **🗺️ Field Monitoring** - Interactive maps and real-time field health analysis
3. **📈 Temporal Analysis** - Time series charts and vegetation trend analysis
4. **🚨 Alerts & Notifications** - Active alerts, warnings, and notification management
5. **📤 Data Export** - Download reports, data, and generate custom exports

## 🧪 **Demo Script**
```bash
# Run the demonstration script
python demo.py
```

## 📁 **Local Project Structure**

```
agriflux/
├── 📊 src/dashboard/           # Streamlit dashboard application
│   ├── main.py                # Main dashboard entry point
│   └── pages/                 # Individual dashboard pages
├── 🛰️ src/data_processing/    # Satellite data processing modules
├── 🤖 src/ai_models/          # Machine learning models
├── 📡 src/sensors/            # Sensor data integration
├── 🗄️ src/database/           # Database models and operations
├── 📋 src/models/             # Data models and schemas
├── 🧪 tests/                  # Comprehensive test suite
├── 📚 docs/                   # Documentation and guides
├── 🚀 run_local.py            # Local development launcher
├── 🎮 demo.py                 # Demonstration script
└── 📦 requirements.txt        # Python dependencies
```

## 🌱 **Agricultural Sample Data**

The local version includes sample data for:
- **5 Agricultural zones** in Punjab, India (Ludhiana area)
- **NDVI vegetation indices** showing crop health
- **Weather data** with temperature, humidity, precipitation
- **Smart alerts** for vegetation stress and pest risks
- **Soil moisture monitoring** across different zones

## 🛠️ **Local Development**

### **Requirements:**
- Python 3.7+
- Basic dependencies: streamlit, pandas, numpy, plotly

### **No Database Required:**
- Uses mock data for demonstration
- No PostgreSQL setup needed for local development
- All features work with sample data

### **Features Working Locally:**
- ✅ Interactive dashboard with dark theme
- ✅ Multi-page navigation
- ✅ Real-time metrics and charts
- ✅ Agricultural zone monitoring
- ✅ Alert system with sample alerts
- ✅ Weather integration display
- ✅ Vegetation health visualization
- ✅ Mobile responsive interface

---

**🌱 AgriFlux - Local Development Ready!**

*Run `python run_local.py` to start exploring the agricultural intelligence platform.*