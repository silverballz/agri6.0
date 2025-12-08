# AgriFlux Dashboard Deployment Guide

This guide provides step-by-step instructions for deploying the AgriFlux Dashboard in different environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Development Deployment](#development-deployment)
4. [Staging Deployment](#staging-deployment)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended for production)
- 20GB free disk space
- Linux, macOS, or Windows with WSL2

### Required Software

```bash
# Python and pip
python --version  # Should be 3.9+
pip --version

# Git (for cloning repository)
git --version

# Optional: Docker (for containerized deployment)
docker --version
docker-compose --version
```

## Environment Configuration

### 1. Clone the Repository

```bash
git clone <repository-url>
cd agriflux-dashboard
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit; print(streamlit.__version__)"
```

### 4. Configure Environment Variables

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

Edit `.env` file with your specific settings:

```bash
# For development
AGRIFLUX_ENV=development
DATABASE_PATH=data/agriflux.db
LOG_LEVEL=DEBUG
ENABLE_DEMO_MODE=true

# For production
AGRIFLUX_ENV=production
DATABASE_PATH=/var/lib/agriflux/agriflux.db
LOG_LEVEL=WARNING
ENABLE_DEMO_MODE=false
```

## Development Deployment

### Quick Start

1. **Set up development environment:**

```bash
# Activate virtual environment
source venv/bin/activate

# Set environment to development
export AGRIFLUX_ENV=development

# Verify configuration
python config.py
```

2. **Initialize database:**

```bash
# Process Sentinel-2 data and populate database
python scripts/populate_database.py

# Verify database
python scripts/verify_database.py
```

3. **Generate demo data (optional):**

```bash
python scripts/generate_demo_data.py
```

4. **Run the dashboard:**

```bash
# Using streamlit directly
streamlit run src/dashboard/main.py

# Or using the run script
python run_dashboard.py
```

5. **Access the dashboard:**

Open your browser and navigate to: `http://localhost:8501`

### Development Configuration

The development environment uses these defaults:

- **Database**: `data/agriflux.db` (SQLite)
- **Log Level**: `DEBUG`
- **Demo Mode**: Enabled
- **AI Models**: Disabled (uses rule-based fallback)
- **Port**: 8501

### Hot Reload

Streamlit automatically reloads when you save changes to Python files. No need to restart the server during development.

## Staging Deployment

Staging environment mimics production but with additional debugging capabilities.

### 1. Configure Staging Environment

```bash
# Set environment
export AGRIFLUX_ENV=staging

# Update .env file
AGRIFLUX_ENV=staging
DATABASE_PATH=/opt/agriflux/staging/agriflux.db
LOG_LEVEL=INFO
ENABLE_DEMO_MODE=true
USE_AI_MODELS=true
```

### 2. Set Up Staging Server

```bash
# Create staging directory
sudo mkdir -p /opt/agriflux/staging
sudo chown $USER:$USER /opt/agriflux/staging

# Copy application files
cp -r . /opt/agriflux/staging/
cd /opt/agriflux/staging

# Install dependencies
pip install -r requirements.txt

# Initialize database
python scripts/populate_database.py
```

### 3. Run with Systemd (Linux)

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/agriflux-staging.service
```

Add the following content:

```ini
[Unit]
Description=AgriFlux Dashboard (Staging)
After=network.target

[Service]
Type=simple
User=agriflux
WorkingDirectory=/opt/agriflux/staging
Environment="AGRIFLUX_ENV=staging"
ExecStart=/opt/agriflux/staging/venv/bin/streamlit run src/dashboard/main.py --server.port=8502
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable agriflux-staging
sudo systemctl start agriflux-staging
sudo systemctl status agriflux-staging
```

### 4. Configure Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name staging.agriflux.example.com;

    location / {
        proxy_pass http://localhost:8502;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Production Deployment

### 1. Production Configuration

```bash
# Set environment
export AGRIFLUX_ENV=production

# Update .env file
AGRIFLUX_ENV=production
DATABASE_PATH=/var/lib/agriflux/agriflux.db
LOG_LEVEL=WARNING
ENABLE_DEMO_MODE=false
USE_AI_MODELS=true
CACHE_ENABLED=true
```

### 2. Security Hardening

```bash
# Create dedicated user
sudo useradd -r -s /bin/false agriflux

# Set up directories with proper permissions
sudo mkdir -p /var/lib/agriflux
sudo mkdir -p /var/log/agriflux
sudo chown -R agriflux:agriflux /var/lib/agriflux
sudo chown -R agriflux:agriflux /var/log/agriflux
sudo chmod 750 /var/lib/agriflux
sudo chmod 750 /var/log/agriflux

# Copy application
sudo cp -r . /opt/agriflux/production
sudo chown -R agriflux:agriflux /opt/agriflux/production
```

### 3. Database Setup

```bash
# Initialize production database
sudo -u agriflux python scripts/populate_database.py

# Set up automated backups
sudo crontab -e -u agriflux
# Add: 0 2 * * * /opt/agriflux/production/scripts/backup.sh
```

### 4. Systemd Service (Production)

```bash
sudo nano /etc/systemd/system/agriflux.service
```

```ini
[Unit]
Description=AgriFlux Dashboard (Production)
After=network.target

[Service]
Type=simple
User=agriflux
Group=agriflux
WorkingDirectory=/opt/agriflux/production
Environment="AGRIFLUX_ENV=production"
Environment="DATABASE_PATH=/var/lib/agriflux/agriflux.db"
Environment="LOG_PATH=/var/log/agriflux/"
ExecStart=/opt/agriflux/production/venv/bin/streamlit run src/dashboard/main.py --server.port=8501 --server.headless=true
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/agriflux /var/log/agriflux

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable agriflux
sudo systemctl start agriflux
sudo systemctl status agriflux
```

### 5. Nginx Configuration (Production)

```nginx
server {
    listen 80;
    server_name agriflux.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name agriflux.example.com;

    ssl_certificate /etc/letsencrypt/live/agriflux.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/agriflux.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 6. SSL Certificate (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d agriflux.example.com

# Auto-renewal is set up automatically
sudo certbot renew --dry-run
```

### 7. Monitoring and Logging

```bash
# View logs
sudo journalctl -u agriflux -f

# Check application logs
tail -f /var/log/agriflux/dashboard.log

# Monitor system resources
htop
```

## Docker Deployment (Alternative)

### 1. Build Docker Image

```bash
# Build image
docker build -t agriflux-dashboard:latest .

# Or use docker-compose
docker-compose build
```

### 2. Run with Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  dashboard:
    image: agriflux-dashboard:latest
    container_name: agriflux-dashboard
    environment:
      - AGRIFLUX_ENV=production
      - DATABASE_PATH=/data/agriflux.db
    volumes:
      - ./data:/data
      - ./logs:/logs
    ports:
      - "8501:8501"
    restart: unless-stopped
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors

```bash
# Check database file exists
ls -la data/agriflux.db

# Verify permissions
chmod 644 data/agriflux.db

# Reinitialize if corrupted
python scripts/populate_database.py
```

#### 2. Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### 3. Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run src/dashboard/main.py --server.port=8502
```

#### 4. Memory Issues

```bash
# Check memory usage
free -h

# Reduce batch size in config
export BATCH_SIZE=500
export MEMORY_LIMIT_GB=4
```

#### 5. Missing Sentinel-2 Data

```bash
# Verify SAFE directory exists
ls -la S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE

# Process data
python scripts/process_sentinel2_data.py
```

### Health Checks

```bash
# Check configuration
python config.py

# Verify database
python scripts/verify_database.py

# Test dependencies
python src/utils/dependency_checker.py

# Run integration tests
pytest tests/test_integration_workflow.py
```

### Performance Optimization

```bash
# Enable caching
export CACHE_ENABLED=true

# Increase workers
export MAX_WORKERS=8

# Optimize database
sqlite3 data/agriflux.db "VACUUM;"
sqlite3 data/agriflux.db "ANALYZE;"
```

## Maintenance

### Regular Tasks

1. **Daily**: Check logs for errors
2. **Weekly**: Review alert history
3. **Monthly**: Update dependencies
4. **Quarterly**: Review and optimize database

### Backup Strategy

```bash
# Manual backup
cp data/agriflux.db data/backups/agriflux_$(date +%Y%m%d).db

# Automated backup script
./scripts/backup.sh
```

### Updates

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Restart service
sudo systemctl restart agriflux
```

## Support

For issues and questions:

- Check logs: `/var/log/agriflux/dashboard.log`
- Review documentation: `docs/`
- Run diagnostics: `python src/utils/dependency_checker.py`

## Security Considerations

1. **Never commit `.env` file** to version control
2. **Use strong passwords** for production databases
3. **Enable HTTPS** in production
4. **Regular security updates** for all dependencies
5. **Restrict file permissions** appropriately
6. **Monitor access logs** regularly
7. **Use firewall rules** to restrict access

## Performance Benchmarks

Expected performance metrics:

- **Dashboard Load Time**: < 3 seconds
- **Page Navigation**: < 1 second
- **Data Export**: < 5 seconds
- **Alert Generation**: < 2 seconds
- **Memory Usage**: 500MB - 2GB (depending on data size)
- **CPU Usage**: 10-30% (idle), 50-80% (processing)

## License

[Your License Here]

## Version History

- **v1.0.0** (2024-12-08): Initial production release
  - Complete data processing pipeline
  - Real-time dashboard with all features
  - Alert generation system
  - Demo mode support
  - Comprehensive error handling
