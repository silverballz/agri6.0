# AgriFlux Local Development Deployment

## 🏠 Local Development Setup

### **Quick Start (5 minutes)**
```bash
# 1. Clone and setup
git clone <repository-url>
cd agriflux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run directly
python run_dashboard.py
```

### **Docker Development Setup**
```bash
# 1. Initialize system
./scripts/init-system.sh development

# 2. Start development environment
./scripts/deploy.sh -e development

# 3. Access at http://localhost:8501
```

## 🛠️ **Development Tools**

### **Hot Reload Development**
```bash
# Run with auto-reload
streamlit run src/dashboard/main.py --server.runOnSave=true

# Or use the development Docker target
docker-compose -f docker-compose.yml --profile development up
```

### **Database Development**
```bash
# Access development database
docker-compose exec postgres psql -U dev_user -d agricultural_monitoring_dev

# Run migrations
python -c "from src.database.migrations import run_migrations; run_migrations()"

# Seed test data
python scripts/seed_test_data.py
```

### **Testing**
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_vegetation_indices.py
```

## 🔧 **Configuration**

### **Development Environment Variables**
```bash
# config/development.env
ENVIRONMENT=development
DEBUG=true
APP_PORT=8501

# Database (local)
POSTGRES_HOST=localhost
POSTGRES_DB=agricultural_monitoring_dev
POSTGRES_USER=dev_user
POSTGRES_PASSWORD=dev_password

# Reduced resource limits
MAX_WORKERS=2
BATCH_SIZE=100
MEMORY_LIMIT_GB=4
```

### **IDE Setup**

#### **VS Code Configuration**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

#### **PyCharm Configuration**
- Set Python interpreter to virtual environment
- Configure run configuration for `run_dashboard.py`
- Enable pytest as test runner
- Set up code formatting with Black

## 🐛 **Debugging**

### **Streamlit Debugging**
```python
# Add debug prints
import streamlit as st
st.write("Debug info:", variable_name)

# Use st.sidebar for debug info
st.sidebar.write("Session state:", st.session_state)

# Enable debug mode
streamlit run src/dashboard/main.py --logger.level=debug
```

### **Database Debugging**
```bash
# Check database logs
docker-compose logs postgres

# Monitor database queries
docker-compose exec postgres tail -f /var/log/postgresql/postgresql.log
```

## 📊 **Development Monitoring**

### **Local Monitoring Stack**
```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access monitoring tools
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
# InfluxDB: http://localhost:8086
```

### **Performance Profiling**
```python
# Add to your code for profiling
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
```

## 🔄 **Development Workflow**

### **Git Workflow**
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
python -m pytest tests/

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature
```

### **Code Quality**
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/

# Security check
bandit -r src/
```

## 🧪 **Testing Strategy**

### **Test Categories**
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# Performance tests
pytest tests/performance/
```

### **Test Data**
```python
# Create test fixtures
@pytest.fixture
def sample_ndvi_data():
    return {
        'zone_id': 'test-zone-1',
        'ndvi_values': [0.7, 0.75, 0.8, 0.72],
        'timestamps': ['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22']
    }
```

## 📱 **Mobile Development Testing**

### **Responsive Testing**
```bash
# Test different screen sizes
streamlit run src/dashboard/main.py --server.port=8501

# Use browser dev tools to simulate mobile devices
# Chrome: F12 -> Toggle device toolbar
# Firefox: F12 -> Responsive Design Mode
```

### **Mobile-Specific Features**
```python
# Detect mobile devices
def is_mobile():
    return st.session_state.get('mobile_device', False)

# Adjust layout for mobile
if is_mobile():
    st.columns(1)  # Single column layout
else:
    st.columns(4)  # Multi-column layout
```

## 🔧 **Troubleshooting**

### **Common Development Issues**

#### **Import Errors**
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or add to your IDE configuration
```

#### **Port Conflicts**
```bash
# Find processes using port 8501
lsof -i :8501

# Kill conflicting processes
pkill -f streamlit
```

#### **Database Connection Issues**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

#### **Memory Issues**
```bash
# Monitor memory usage
docker stats

# Reduce batch sizes in config/development.env
BATCH_SIZE=50
MAX_WORKERS=1
```

### **Performance Optimization**

#### **Streamlit Performance**
```python
# Use caching for expensive operations
@st.cache_data
def load_satellite_data():
    # Expensive data loading
    return data

# Optimize session state usage
if 'expensive_data' not in st.session_state:
    st.session_state.expensive_data = load_data()
```

#### **Database Performance**
```sql
-- Add indexes for development
CREATE INDEX idx_dev_zone_timestamp ON index_timeseries (zone_id, timestamp);

-- Monitor query performance
EXPLAIN ANALYZE SELECT * FROM monitoring_zones WHERE crop_type = 'wheat';
```

## 📚 **Development Resources**

### **Documentation**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Folium Documentation](https://python-visualization.github.io/folium/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Docker Documentation](https://docs.docker.com/)

### **Useful Commands**
```bash
# View all containers
docker-compose ps

# View logs
docker-compose logs -f app

# Execute commands in containers
docker-compose exec app python -c "print('Hello from container')"

# Backup development database
./scripts/backup.sh development
```

---

*Happy coding! 🚀 For questions or issues, check the troubleshooting section or contact the development team.*