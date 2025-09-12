#!/bin/bash

# System Initialization Script
# Prepares the system for first-time deployment

set -e

ENVIRONMENT=${1:-production}

echo "🚀 Initializing Agricultural Monitoring Platform for $ENVIRONMENT environment"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check system requirements
check_requirements() {
    log "🔍 Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        echo "⚠️  Warning: Less than 10GB disk space available"
    fi
    
    log "✅ System requirements check passed"
}

# Create directory structure
create_directories() {
    log "📁 Creating directory structure..."
    
    # Main directories
    mkdir -p {data,logs,models,backups,secrets}
    
    # Data subdirectories
    mkdir -p data/{postgres,influxdb,redis,app,prometheus,grafana,loki}
    
    # Backup subdirectories
    mkdir -p backups/{postgres,influxdb,redis,app}
    
    # Log subdirectories
    mkdir -p logs/{app,nginx,monitoring}
    
    # SSL directory
    mkdir -p nginx/ssl
    
    # Monitoring directories
    mkdir -p monitoring/{grafana/{dashboards,datasources},prometheus}
    
    log "✅ Directory structure created"
}

# Set proper permissions
set_permissions() {
    log "🔒 Setting proper permissions..."
    
    # Set directory permissions
    chmod 755 data logs models backups
    chmod -R 755 data/ backups/ logs/
    chmod 700 secrets/
    
    # Set script permissions
    chmod +x scripts/*.sh
    
    # Set SSL directory permissions (if exists)
    if [[ -d "nginx/ssl" ]]; then
        chmod 700 nginx/ssl
    fi
    
    log "✅ Permissions set correctly"
}

# Generate SSL certificates (self-signed for development/staging)
generate_ssl_certificates() {
    log "🔐 Generating SSL certificates..."
    
    if [[ "$ENVIRONMENT" != "production" ]]; then
        # Generate self-signed certificates for non-production
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
            2>/dev/null
        
        chmod 600 nginx/ssl/key.pem
        chmod 644 nginx/ssl/cert.pem
        
        log "✅ Self-signed SSL certificates generated"
    else
        log "⚠️  Production environment detected. Please configure proper SSL certificates manually."
        log "   Place your certificates at:"
        log "   - Certificate: nginx/ssl/cert.pem"
        log "   - Private key: nginx/ssl/key.pem"
    fi
}

# Initialize configuration files
init_configuration() {
    log "⚙️  Initializing configuration files..."
    
    # Copy environment template if it doesn't exist
    if [[ ! -f "config/$ENVIRONMENT.env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example "config/$ENVIRONMENT.env"
            log "✅ Environment configuration created from template"
        else
            log "⚠️  No environment template found"
        fi
    fi
    
    # Initialize secrets
    if [[ ! -f "secrets/secrets.$ENVIRONMENT.yml" ]]; then
        if [[ -f "secrets/secrets.example.yml" ]]; then
            cp secrets/secrets.example.yml "secrets/secrets.$ENVIRONMENT.yml"
            chmod 600 "secrets/secrets.$ENVIRONMENT.yml"
            log "✅ Secrets configuration created from template"
        fi
    fi
    
    log "✅ Configuration files initialized"
}

# Setup monitoring configuration
setup_monitoring() {
    log "📊 Setting up monitoring configuration..."
    
    # Create Grafana datasource configuration
    cat > monitoring/grafana/datasources/datasources.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    
  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    database: ${INFLUX_BUCKET:-sensor_data}
    user: ${INFLUX_USERNAME:-admin}
    secureJsonData:
      password: ${INFLUX_PASSWORD:-}
EOF
    
    # Create basic Grafana dashboard configuration
    cat > monitoring/grafana/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    log "✅ Monitoring configuration setup completed"
}

# Create systemd service (for production Linux systems)
create_systemd_service() {
    if [[ "$ENVIRONMENT" == "production" ]] && [[ -d "/etc/systemd/system" ]]; then
        log "🔧 Creating systemd service..."
        
        sudo tee /etc/systemd/system/agricultural-monitoring.service > /dev/null << EOF
[Unit]
Description=Agricultural Monitoring Platform
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/scripts/deploy.sh -e production
ExecStop=/usr/bin/docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable agricultural-monitoring.service
        
        log "✅ Systemd service created and enabled"
    fi
}

# Setup log rotation
setup_log_rotation() {
    log "📋 Setting up log rotation..."
    
    # Create logrotate configuration
    cat > /tmp/agricultural-monitoring-logrotate << EOF
$(pwd)/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(whoami)
    postrotate
        docker-compose restart app nginx 2>/dev/null || true
    endscript
}
EOF
    
    # Install logrotate configuration (if possible)
    if [[ -d "/etc/logrotate.d" ]] && command -v sudo &> /dev/null; then
        sudo cp /tmp/agricultural-monitoring-logrotate /etc/logrotate.d/agricultural-monitoring
        sudo chmod 644 /etc/logrotate.d/agricultural-monitoring
        log "✅ Log rotation configured"
    else
        log "⚠️  Could not install system log rotation. Manual setup may be required."
    fi
    
    rm -f /tmp/agricultural-monitoring-logrotate
}

# Create cron jobs for maintenance
setup_cron_jobs() {
    log "⏰ Setting up maintenance cron jobs..."
    
    # Create cron script
    cat > scripts/cron-maintenance.sh << 'EOF'
#!/bin/bash
# Automated maintenance tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Run system monitoring
./scripts/monitor.sh production >> logs/cron.log 2>&1

# Run backup (weekly on Sundays at 2 AM)
if [[ $(date +%u) -eq 7 ]] && [[ $(date +%H) -eq 2 ]]; then
    ./scripts/backup.sh production >> logs/cron.log 2>&1
fi

# Clean up old logs (monthly)
if [[ $(date +%d) -eq 1 ]] && [[ $(date +%H) -eq 3 ]]; then
    find logs/ -name "*.log" -mtime +90 -delete
fi
EOF
    
    chmod +x scripts/cron-maintenance.sh
    
    # Add to crontab (if possible)
    if command -v crontab &> /dev/null; then
        (crontab -l 2>/dev/null; echo "0 * * * * $(pwd)/scripts/cron-maintenance.sh") | crontab -
        log "✅ Cron jobs configured"
    else
        log "⚠️  Could not configure cron jobs automatically. Please add manually:"
        log "   0 * * * * $(pwd)/scripts/cron-maintenance.sh"
    fi
}

# Main initialization function
main() {
    log "🚀 Starting system initialization..."
    
    check_requirements
    create_directories
    set_permissions
    generate_ssl_certificates
    init_configuration
    setup_monitoring
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        create_systemd_service
        setup_log_rotation
        setup_cron_jobs
    fi
    
    log "🎉 System initialization completed!"
    log ""
    log "📋 Next steps:"
    log "1. Review and update configuration files:"
    log "   - config/$ENVIRONMENT.env"
    log "   - secrets/secrets.$ENVIRONMENT.yml"
    log "2. Configure external API credentials and SSL certificates"
    log "3. Run deployment: ./scripts/deploy.sh -e $ENVIRONMENT"
    log "4. Setup monitoring: docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile monitoring up -d"
    log ""
    log "📚 Documentation:"
    log "   - View logs: tail -f logs/app/*.log"
    log "   - Monitor system: ./scripts/monitor.sh $ENVIRONMENT"
    log "   - Create backup: ./scripts/backup.sh $ENVIRONMENT"
}

# Handle script interruption
trap 'log "⚠️  Initialization interrupted"; exit 130' INT TERM

# Run main function
main "$@"