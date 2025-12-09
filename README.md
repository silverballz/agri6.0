# AgriFlux 🌱

Smart Agricultural Intelligence Platform - An AI-powered solution for monitoring crop health using satellite imagery and environmental sensors. Transform your farming operations with real-time insights, predictive analytics, and data-driven decision making.

## 🚀 Features

### Core Capabilities
- **Real-time Crop Health Monitoring** using Sentinel-2A satellite data
- **Vegetation Index Calculations** (NDVI, SAVI, EVI, NDWI, NDSI)
- **Environmental Sensor Integration** for comprehensive field monitoring
- **AI-Powered Analytics** for anomaly detection and risk prediction
- **Interactive Dashboard** with temporal analysis and trend visualization
- **Automated Alert System** for crop stress and pest risk detection
- **Data Export & Reporting** with customizable formats and schedules

### Advanced Features
- **Multi-spectral Analysis** with cloud masking and atmospheric correction
- **Temporal Trend Analysis** with statistical modeling
- **Spatial Interpolation** of sensor data
- **Machine Learning Models** for predictive analytics
- **RESTful API** for system integration
- **Role-based Access Control** for team collaboration

### Real Satellite Data Integration
- **Sentinel Hub API Integration** for downloading real Sentinel-2 imagery
- **Automated Data Pipeline** for processing multi-temporal satellite data
- **AI Model Training** on real agricultural data (85%+ accuracy)
- **Data Quality Validation** ensuring production-ready datasets
- **Model Comparison Tools** to quantify improvements over synthetic data
- **Comprehensive Logging** for debugging and data provenance tracking

## 📋 Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd agriflux

# Initialize the system
./scripts/init-system.sh production

# Configure secrets and environment
./scripts/setup-secrets.sh production

# Deploy the platform
./scripts/deploy.sh -e production
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
python -c "from src.database.migrations import run_migrations; run_migrations()"

# Start the dashboard
streamlit run run_dashboard.py
```

### Option 3: Staging Environment

```bash
# Deploy to staging
./scripts/deploy.sh -e staging

# Access at http://localhost:8502
```

### Option 4: Real Satellite Data Pipeline

Train AI models on real Sentinel-2 satellite imagery:

```bash
# Step 1: Set up Sentinel Hub credentials
export SENTINEL_HUB_CLIENT_ID=your_client_id
export SENTINEL_HUB_CLIENT_SECRET=your_client_secret

# Step 2: Download real satellite data (15-20 imagery dates)
python scripts/download_real_satellite_data.py --target-count 20

# Step 3: Validate data quality
python scripts/validate_data_quality.py

# Step 4: Prepare training datasets
python scripts/prepare_real_training_data.py
python scripts/prepare_lstm_training_data.py

# Step 5: Train models on real data
python scripts/train_cnn_on_real_data.py --epochs 50
python scripts/train_lstm_on_real_data.py --epochs 100

# Step 6: Deploy trained models
python scripts/deploy_real_trained_models.py

# Step 7: Enable AI predictions
echo "USE_AI_MODELS=true" >> .env

# See docs/REAL_DATA_PIPELINE_GUIDE.md for complete documentation
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sentinel-2A   │    │  Environmental  │    │   Weather APIs  │
│  Satellite Data │    │     Sensors     │    │   & External    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Image     │  │   Sensor    │  │      Data Fusion &      │  │
│  │ Processing  │  │    Data     │  │    Quality Control      │  │
│  │             │  │ Validation  │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI Analysis Engine                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    CNN      │  │    LSTM     │  │    Risk Prediction      │  │
│  │   Spatial   │  │  Temporal   │  │       Models            │  │
│  │  Analysis   │  │  Analysis   │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Streamlit  │  │   Alert     │  │      Data Export &      │  │
│  │  Dashboard  │  │   System    │  │       Reporting         │  │
│  │             │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
agriflux/
├── 📁 src/                          # Source code
│   ├── 📁 data_processing/          # Satellite data processing
│   │   ├── sentinel2_parser.py      # Sentinel-2A data parsing
│   │   ├── vegetation_indices.py    # NDVI, SAVI, EVI calculations
│   │   ├── cloud_masking.py         # Cloud detection and masking
│   │   └── geospatial_utils.py      # Coordinate transformations
│   ├── 📁 ai_models/                # Machine learning models
│   │   ├── spatial_cnn.py           # CNN for spatial analysis
│   │   ├── temporal_lstm.py         # LSTM for time series
│   │   ├── risk_prediction.py       # Pest/disease risk models
│   │   └── training_pipeline.py     # Model training workflows
│   ├── 📁 dashboard/                # Streamlit web interface
│   │   ├── main.py                  # Main dashboard application
│   │   └── 📁 pages/                # Dashboard pages
│   │       ├── overview.py          # Main overview page
│   │       ├── field_monitoring.py  # Interactive field maps
│   │       ├── temporal_analysis.py # Time series analysis
│   │       ├── alerts.py            # Alert management
│   │       └── data_export.py       # Data export and reports
│   ├── 📁 database/                 # Database layer
│   │   ├── models.py                # Database models
│   │   ├── connection.py            # Database connections
│   │   └── migrations.py            # Schema migrations
│   ├── 📁 models/                   # Data models
│   │   ├── satellite_image.py       # Satellite image model
│   │   ├── monitoring_zone.py       # Field zone model
│   │   └── index_timeseries.py      # Time series model
│   └── 📁 sensors/                  # Sensor integration
│       ├── data_ingestion.py        # Sensor data ingestion
│       ├── data_validation.py       # Data quality control
│       └── data_fusion.py           # Multi-source data fusion
├── 📁 scripts/                      # Deployment and maintenance
│   ├── deploy.sh                    # Deployment script
│   ├── backup.sh                    # Backup automation
│   ├── monitor.sh                   # System monitoring
│   ├── init-system.sh               # System initialization
│   └── setup-secrets.sh             # Secrets management
├── 📁 config/                       # Configuration files
│   ├── development.env              # Development settings
│   ├── staging.env                  # Staging settings
│   └── production.env               # Production settings
├── 📁 docs/                         # Documentation
│   ├── user-guide.md                # User documentation
│   ├── technical-documentation.md   # Technical documentation
│   ├── faq.md                       # Frequently asked questions
│   └── training-materials.md        # Training resources
├── 📁 database/                     # Database initialization
│   └── 📁 init/                     # SQL initialization scripts
├── 📁 nginx/                        # Nginx configuration
├── 📁 monitoring/                   # Monitoring configuration
├── 📁 tests/                        # Test suite
├── docker-compose.yml               # Docker Compose configuration
├── docker-compose.prod.yml          # Production overrides
├── docker-compose.staging.yml       # Staging configuration
├── Dockerfile                       # Container definition
└── requirements.txt                 # Python dependencies
```

## 🛠️ Technology Stack

### Core Technologies
- **Backend**: Python 3.9+, FastAPI, SQLAlchemy
- **Frontend**: Streamlit, Plotly, Folium
- **Database**: PostgreSQL with PostGIS, InfluxDB
- **Cache**: Redis
- **AI/ML**: TensorFlow, scikit-learn, OpenCV
- **Geospatial**: GDAL, rasterio, GeoPandas

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Web Server**: Nginx
- **Monitoring**: Prometheus, Grafana, Loki
- **Deployment**: Automated scripts, CI/CD ready

### Data Sources
- **Satellite**: Sentinel-2A/2B (ESA), Landsat 8/9 (NASA)
- **Weather**: OpenWeatherMap, NOAA APIs
- **Sensors**: IoT devices, weather stations, soil sensors

## 📚 Documentation

### User Documentation
- 📖 **[User Guide](docs/user-guide.md)** - Complete user manual with screenshots
- ❓ **[FAQ](docs/faq.md)** - Frequently asked questions and solutions
- 🎓 **[Training Materials](docs/training-materials.md)** - Step-by-step tutorials

### Technical Documentation
- 🔧 **[Technical Documentation](docs/technical-documentation.md)** - System administration guide
- 🏗️ **[Architecture Overview](docs/architecture.md)** - System design and components
- 🔌 **[API Reference](docs/api-reference.md)** - REST API documentation

### Real Data Pipeline Documentation
- 🛰️ **[Real Data Pipeline Guide](docs/REAL_DATA_PIPELINE_GUIDE.md)** - Complete guide for downloading and processing real satellite data
- ⚡ **[Quick Reference](docs/REAL_DATA_QUICK_REFERENCE.md)** - Quick commands and troubleshooting
- 🔍 **[API Troubleshooting](docs/API_TROUBLESHOOTING_GUIDE.md)** - Detailed solutions for API issues
- 📝 **[Scripts Documentation](scripts/README_REAL_DATA_PIPELINE.md)** - Pipeline scripts reference
- 📊 **[Model Deployment Guide](docs/MODEL_DEPLOYMENT_GUIDE.md)** - Deploying trained models
- 📋 **[Logging System](docs/LOGGING_SYSTEM.md)** - Comprehensive logging documentation

### Deployment Guides
- 🚀 **[Quick Start Guide](#quick-start)** - Get up and running in minutes
- 🐳 **[Docker Deployment](docs/docker-deployment.md)** - Container-based deployment
- ☁️ **[Cloud Deployment](docs/cloud-deployment.md)** - AWS, Azure, GCP guides

## 🔧 Configuration

### Environment Variables

```bash
# Application Configuration
ENVIRONMENT=production
APP_PORT=8501
DEBUG=false

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_DB=agricultural_monitoring
POSTGRES_USER=agri_user
POSTGRES_PASSWORD=<secure_password>

# InfluxDB Configuration
INFLUX_HOST=influxdb
INFLUX_ORG=agricultural_monitoring
INFLUX_BUCKET=sensor_data

# AI Model Configuration
MODEL_PATH=/app/models
MODEL_RETRAIN_INTERVAL=7
MODEL_PERFORMANCE_THRESHOLD=0.85

# External APIs
SENTINEL_HUB_CLIENT_ID=<your_client_id>
WEATHER_API_KEY=<your_api_key>
```

### Secrets Management

```bash
# Generate secure secrets
./scripts/setup-secrets.sh production

# Secrets are stored in secrets/secrets.production.yml
# Never commit secrets to version control
```

## 🚨 Monitoring and Alerts

### System Health Monitoring
```bash
# Run comprehensive system check
./scripts/monitor.sh production

# Generate detailed monitoring report
./scripts/monitor.sh production --report
```

### Automated Backups
```bash
# Create backup
./scripts/backup.sh production

# Backups are stored in ./backups/ with timestamp
# Retention: 30 days (configurable)
```

### Log Management
- **Application logs**: `logs/app/`
- **System logs**: `logs/monitor.log`
- **Nginx logs**: `logs/nginx/`
- **Automatic rotation**: Daily, 30-day retention

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_vegetation_indices.py
python -m pytest tests/test_ai_models.py
python -m pytest tests/test_integration_workflow.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing
- **Data Quality Tests**: Validation of processing accuracy

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd agricultural-monitoring-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Code Standards
- **Python**: PEP 8, Black formatting, type hints
- **Documentation**: Comprehensive docstrings
- **Testing**: Minimum 80% code coverage
- **Git**: Conventional commit messages

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- 📧 **Email**: support@agrimonitor.com
- 📞 **Phone**: 1-800-AGRI-HELP
- 💬 **Chat**: Available in the platform
- 🌐 **Forum**: Community discussions

### Professional Services
- **Training**: On-site and remote training available
- **Consulting**: Custom implementation and optimization
- **Support Plans**: 24/7 support for enterprise customers
- **Custom Development**: Tailored features and integrations

### Reporting Issues
- **Bug Reports**: Use GitHub issues or support email
- **Feature Requests**: Submit via platform or GitHub
- **Security Issues**: security@agrimonitor.com

---

**Experience the future of agriculture with AgriFlux - where intelligence meets cultivation!** 🌱🚀