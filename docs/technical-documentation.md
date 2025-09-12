# Agricultural Monitoring Platform - Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Installation and Deployment](#installation-and-deployment)
3. [Configuration Management](#configuration-management)
4. [Database Administration](#database-administration)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)
8. [Security](#security)

## System Architecture

### Overview
The Agricultural Monitoring Platform is a containerized microservices architecture built with Docker and Docker Compose. The system processes Sentinel-2A satellite imagery to extract vegetation indices and integrates with environmental sensor networks.

### Core Components

#### Application Stack
- **Frontend**: Streamlit web application (Python)
- **Backend**: Python-based data processing and AI models
- **Reverse Proxy**: Nginx for load balancing and SSL termination
- **Database**: PostgreSQL with PostGIS for spatial data
- **Time Series DB**: InfluxDB for sensor data and metrics
- **Cache**: Redis for session management and caching

#### Supporting Services
- **Monitoring**: Prometheus, Grafana, Loki for observability
- **Backup**: Automated backup scripts for data persistence
- **Security**: SSL/TLS encryption, secrets management

### Data Flow
```
Sentinel-2A Data → Data Processing → AI Analysis → Dashboard
                ↗                              ↘
Environmental Sensors → Data Fusion → Alerts → Notifications
```

## Installation and Deployment

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- Minimum 8GB RAM, 50GB disk space
- Linux/macOS operating system

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd agricultural-monitoring-platform

# Initialize system
./scripts/init-system.sh production

# Configure secrets
./scripts/setup-secrets.sh production

# Deploy
./scripts/deploy.sh -e production
```

### Environment-Specific Deployment

#### Development
```bash
./scripts/deploy.sh -e development
```

#### Staging
```bash
./scripts/deploy.sh -e staging
```

#### Production
```bash
./scripts/deploy.sh -e production --profile monitoring
```

### Manual Deployment Steps

1. **Prepare Environment**
   ```bash
   mkdir -p data/{postgres,influxdb,redis,app} logs models backups
   chmod -R 755 data logs models backups
   ```

2. **Configure Environment Variables**
   ```bash
   cp config/production.env.example config/production.env
   # Edit configuration values
   ```

3. **Generate Secrets**
   ```bash
   ./scripts/setup-secrets.sh production
   ```

4. **Start Services**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

5. **Run Database Migrations**
   ```bash
   docker-compose exec app python -c "from src.database.migrations import run_migrations; run_migrations()"
   ```

## Configuration Management

### Environment Files
Configuration is managed through environment-specific files:
- `config/development.env`: Development settings
- `config/staging.env`: Staging environment settings
- `config/production.env`: Production configuration

### Key Configuration Parameters

#### Database Configuration
```bash
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=agricultural_monitoring
POSTGRES_USER=agri_user
POSTGRES_PASSWORD=<from_secrets>
```

#### Application Configuration
```bash
APP_PORT=8501
DEBUG=false
LOG_LEVEL=INFO
MAX_WORKERS=8
BATCH_SIZE=1000
MEMORY_LIMIT_GB=16
```

#### AI Model Configuration
```bash
MODEL_PATH=/app/models
MODEL_RETRAIN_INTERVAL=7  # days
MODEL_PERFORMANCE_THRESHOLD=0.85
```

### Secrets Management
Sensitive configuration is stored in `secrets/secrets.<environment>.yml`:
```yaml
database:
  postgres_password: "secure_password"
application:
  secret_key: "application_secret"
external_apis:
  sentinel_hub_client_id: "api_client_id"
```

## Database Administration

### PostgreSQL Management

#### Accessing the Database
```bash
# Via Docker Compose
docker-compose exec postgres psql -U agri_user -d agricultural_monitoring

# Direct connection
psql -h localhost -p 5432 -U agri_user -d agricultural_monitoring
```

#### Common Administrative Tasks

**View Database Size**
```sql
SELECT pg_size_pretty(pg_database_size('agricultural_monitoring'));
```

**Monitor Active Connections**
```sql
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
```

**Check Index Usage**
```sql
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

#### Backup and Restore

**Create Backup**
```bash
./scripts/backup.sh production
```

**Manual Backup**
```bash
docker-compose exec postgres pg_dump -U agri_user -d agricultural_monitoring > backup.sql
```

**Restore from Backup**
```bash
docker-compose exec -T postgres psql -U agri_user -d agricultural_monitoring < backup.sql
```

### InfluxDB Management

#### Accessing InfluxDB
```bash
# Web UI
http://localhost:8086

# CLI
docker-compose exec influxdb influx
```

#### Common Operations

**List Buckets**
```bash
influx bucket list --org agricultural_monitoring
```

**Query Data**
```bash
influx query 'from(bucket:"sensor_data") |> range(start:-1h)'
```

**Backup InfluxDB**
```bash
docker-compose exec influxdb influx backup /tmp/backup
```

## Monitoring and Maintenance

### System Monitoring

#### Health Checks
```bash
# Run comprehensive system check
./scripts/monitor.sh production

# Generate monitoring report
./scripts/monitor.sh production --report
```

#### Key Metrics to Monitor
- **CPU Usage**: Should stay below 80%
- **Memory Usage**: Should stay below 85%
- **Disk Usage**: Should stay below 90%
- **Database Connections**: Monitor for connection leaks
- **Response Times**: API and dashboard response times

### Log Management

#### Log Locations
- Application logs: `logs/app/`
- Nginx logs: `logs/nginx/`
- Database logs: Docker container logs
- System logs: `logs/monitor.log`

#### Log Analysis
```bash
# View application logs
docker-compose logs -f app

# Search for errors
grep -r "ERROR" logs/

# Monitor real-time logs
tail -f logs/app/application.log
```

### Automated Maintenance

#### Backup Schedule
```bash
# Daily backups (configured in cron)
0 2 * * * /path/to/scripts/backup.sh production

# Weekly full backup
0 2 * * 0 /path/to/scripts/backup.sh production full
```

#### Log Rotation
Logs are automatically rotated using logrotate:
- Daily rotation
- 30-day retention
- Compression enabled

#### Database Maintenance
```sql
-- Run weekly
VACUUM ANALYZE;

-- Run monthly
REINDEX DATABASE agricultural_monitoring;

-- Update statistics
ANALYZE;
```

## API Reference

### REST API Endpoints

#### Authentication
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "user",
  "password": "password"
}
```

#### Monitoring Zones
```http
# Get all zones
GET /api/zones

# Create new zone
POST /api/zones
Content-Type: application/json

{
  "name": "Field 1",
  "geometry": {...},
  "crop_type": "corn"
}

# Get zone details
GET /api/zones/{zone_id}

# Update zone
PUT /api/zones/{zone_id}

# Delete zone
DELETE /api/zones/{zone_id}
```

#### Vegetation Indices
```http
# Get time series data
GET /api/zones/{zone_id}/indices?start_date=2024-01-01&end_date=2024-12-31&index_type=NDVI

# Get latest values
GET /api/zones/{zone_id}/indices/latest
```

#### Alerts
```http
# Get active alerts
GET /api/alerts?status=active

# Acknowledge alert
POST /api/alerts/{alert_id}/acknowledge

# Resolve alert
POST /api/alerts/{alert_id}/resolve
```

### WebSocket API

#### Real-time Updates
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8501/_stcore/stream');

// Listen for updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    // Handle real-time updates
};
```

## Troubleshooting

### Common Issues

#### Service Won't Start
**Symptoms**: Container exits immediately
**Diagnosis**:
```bash
docker-compose logs <service_name>
docker-compose ps
```
**Solutions**:
- Check environment variables
- Verify file permissions
- Check port conflicts

#### Database Connection Issues
**Symptoms**: "Connection refused" errors
**Diagnosis**:
```bash
docker-compose exec postgres pg_isready
netstat -tlnp | grep 5432
```
**Solutions**:
- Verify PostgreSQL is running
- Check network connectivity
- Validate credentials

#### High Memory Usage
**Symptoms**: System slowdown, OOM errors
**Diagnosis**:
```bash
docker stats
free -h
```
**Solutions**:
- Reduce batch sizes
- Optimize queries
- Add more memory
- Implement data archiving

#### SSL Certificate Issues
**Symptoms**: Browser security warnings
**Diagnosis**:
```bash
openssl x509 -in nginx/ssl/cert.pem -text -noout
```
**Solutions**:
- Renew certificates
- Check certificate chain
- Verify domain names

### Performance Optimization

#### Database Optimization
```sql
-- Analyze slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE tablename = 'index_timeseries';
```

#### Application Optimization
- Enable caching for frequently accessed data
- Optimize image processing pipelines
- Use connection pooling
- Implement data pagination

## Security

### Security Best Practices

#### Network Security
- Use HTTPS/TLS for all communications
- Implement proper firewall rules
- Use VPN for remote access
- Regular security updates

#### Application Security
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF protection

#### Data Security
- Encrypt sensitive data at rest
- Secure backup storage
- Access logging and monitoring
- Regular security audits

### SSL/TLS Configuration

#### Certificate Management
```bash
# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# Check certificate expiration
openssl x509 -enddate -noout -in nginx/ssl/cert.pem
```

#### Nginx SSL Configuration
```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
```

### Access Control

#### User Management
- Implement role-based access control (RBAC)
- Regular access reviews
- Strong password policies
- Multi-factor authentication (MFA)

#### API Security
- API key authentication
- Rate limiting
- Request validation
- Audit logging

---

*This technical documentation is maintained by the development team. For updates or corrections, please submit a pull request or contact the technical lead.*