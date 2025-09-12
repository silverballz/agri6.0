#!/bin/bash

# Database and System Backup Script
# Performs automated backups of PostgreSQL, InfluxDB, and application data

set -e

# Configuration
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ENVIRONMENT=${1:-production}
RETENTION_DAYS=${2:-30}

# Load environment variables
if [[ -f "config/$ENVIRONMENT.env" ]]; then
    export $(grep -v '^#' "config/$ENVIRONMENT.env" | xargs)
fi

echo "🗄️  Starting backup process for environment: $ENVIRONMENT"
echo "📅 Timestamp: $TIMESTAMP"

# Create backup directory
mkdir -p "$BACKUP_DIR/$ENVIRONMENT/$TIMESTAMP"
BACKUP_PATH="$BACKUP_DIR/$ENVIRONMENT/$TIMESTAMP"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if service is running
check_service() {
    local service_name=$1
    if docker-compose ps | grep -q "$service_name.*Up"; then
        return 0
    else
        return 1
    fi
}

# Backup PostgreSQL database
backup_postgres() {
    log "📊 Backing up PostgreSQL database..."
    
    if check_service "postgres"; then
        docker-compose exec -T postgres pg_dump \
            -U "${POSTGRES_USER:-agri_user}" \
            -d "${POSTGRES_DB:-agricultural_monitoring}" \
            --verbose --clean --if-exists --create \
            > "$BACKUP_PATH/postgres_backup.sql"
        
        # Compress the backup
        gzip "$BACKUP_PATH/postgres_backup.sql"
        log "✅ PostgreSQL backup completed: postgres_backup.sql.gz"
    else
        log "⚠️  PostgreSQL service not running, skipping database backup"
    fi
}

# Backup InfluxDB database
backup_influxdb() {
    log "📈 Backing up InfluxDB database..."
    
    if check_service "influxdb"; then
        # Create InfluxDB backup
        docker-compose exec -T influxdb influx backup \
            --org "${INFLUX_ORG:-agricultural_monitoring}" \
            --bucket "${INFLUX_BUCKET:-sensor_data}" \
            --token "${INFLUX_TOKEN:-}" \
            /tmp/influx_backup 2>/dev/null || true
        
        # Copy backup from container
        docker cp $(docker-compose ps -q influxdb):/tmp/influx_backup "$BACKUP_PATH/influxdb_backup"
        
        # Compress the backup
        tar -czf "$BACKUP_PATH/influxdb_backup.tar.gz" -C "$BACKUP_PATH" influxdb_backup
        rm -rf "$BACKUP_PATH/influxdb_backup"
        
        log "✅ InfluxDB backup completed: influxdb_backup.tar.gz"
    else
        log "⚠️  InfluxDB service not running, skipping time series backup"
    fi
}

# Backup application data
backup_app_data() {
    log "📁 Backing up application data..."
    
    # Backup data directory
    if [[ -d "./data" ]]; then
        tar -czf "$BACKUP_PATH/app_data.tar.gz" -C . data
        log "✅ Application data backup completed: app_data.tar.gz"
    fi
    
    # Backup models directory
    if [[ -d "./models" ]]; then
        tar -czf "$BACKUP_PATH/models.tar.gz" -C . models
        log "✅ Models backup completed: models.tar.gz"
    fi
    
    # Backup logs (last 7 days only)
    if [[ -d "./logs" ]]; then
        find ./logs -name "*.log" -mtime -7 -exec tar -czf "$BACKUP_PATH/recent_logs.tar.gz" {} +
        log "✅ Recent logs backup completed: recent_logs.tar.gz"
    fi
}

# Backup configuration files
backup_config() {
    log "⚙️  Backing up configuration files..."
    
    # Backup configuration directory (excluding secrets)
    tar -czf "$BACKUP_PATH/config.tar.gz" \
        --exclude="*.env" \
        --exclude="secrets" \
        -C . config docker-compose*.yml nginx
    
    log "✅ Configuration backup completed: config.tar.gz"
}

# Create backup manifest
create_manifest() {
    log "📋 Creating backup manifest..."
    
    cat > "$BACKUP_PATH/backup_manifest.txt" << EOF
Agricultural Monitoring Platform Backup
=======================================
Environment: $ENVIRONMENT
Timestamp: $TIMESTAMP
Date: $(date)
Hostname: $(hostname)

Backup Contents:
EOF

    # List backup files with sizes
    ls -lh "$BACKUP_PATH" | grep -v "^total" >> "$BACKUP_PATH/backup_manifest.txt"
    
    # Add system information
    cat >> "$BACKUP_PATH/backup_manifest.txt" << EOF

System Information:
==================
Docker Version: $(docker --version)
Docker Compose Version: $(docker-compose --version)
Disk Usage: $(df -h .)

Service Status:
==============
EOF
    
    docker-compose ps >> "$BACKUP_PATH/backup_manifest.txt" 2>/dev/null || echo "Docker services not running" >> "$BACKUP_PATH/backup_manifest.txt"
    
    log "✅ Backup manifest created: backup_manifest.txt"
}

# Cleanup old backups
cleanup_old_backups() {
    log "🧹 Cleaning up backups older than $RETENTION_DAYS days..."
    
    find "$BACKUP_DIR/$ENVIRONMENT" -type d -name "20*" -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
    
    # Count remaining backups
    BACKUP_COUNT=$(find "$BACKUP_DIR/$ENVIRONMENT" -type d -name "20*" | wc -l)
    log "📊 Retained $BACKUP_COUNT backup(s) for environment: $ENVIRONMENT"
}

# Verify backup integrity
verify_backup() {
    log "🔍 Verifying backup integrity..."
    
    local errors=0
    
    # Check if backup files exist and are not empty
    for file in "$BACKUP_PATH"/*.gz "$BACKUP_PATH"/*.txt; do
        if [[ -f "$file" ]]; then
            if [[ ! -s "$file" ]]; then
                log "❌ Empty backup file: $(basename "$file")"
                ((errors++))
            else
                log "✅ Verified: $(basename "$file") ($(du -h "$file" | cut -f1))"
            fi
        fi
    done
    
    # Test compressed files
    for gz_file in "$BACKUP_PATH"/*.gz; do
        if [[ -f "$gz_file" ]]; then
            if ! gzip -t "$gz_file" 2>/dev/null; then
                log "❌ Corrupted archive: $(basename "$gz_file")"
                ((errors++))
            fi
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        log "✅ All backup files verified successfully"
        return 0
    else
        log "❌ Found $errors error(s) in backup verification"
        return 1
    fi
}

# Send backup notification (if configured)
send_notification() {
    local status=$1
    local webhook_url="${WEBHOOK_URL:-}"
    
    if [[ -n "$webhook_url" ]]; then
        local message
        if [[ "$status" == "success" ]]; then
            message="✅ Backup completed successfully for $ENVIRONMENT environment"
        else
            message="❌ Backup failed for $ENVIRONMENT environment"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$webhook_url" 2>/dev/null || true
    fi
}

# Main backup process
main() {
    local start_time=$(date +%s)
    
    log "🚀 Starting backup process..."
    
    # Perform backups
    backup_postgres
    backup_influxdb
    backup_app_data
    backup_config
    create_manifest
    
    # Verify backup
    if verify_backup; then
        log "✅ Backup verification passed"
        cleanup_old_backups
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local backup_size=$(du -sh "$BACKUP_PATH" | cut -f1)
        
        log "🎉 Backup completed successfully!"
        log "📊 Backup size: $backup_size"
        log "⏱️  Duration: ${duration}s"
        log "📁 Location: $BACKUP_PATH"
        
        send_notification "success"
        exit 0
    else
        log "❌ Backup verification failed"
        send_notification "failure"
        exit 1
    fi
}

# Handle script interruption
trap 'log "⚠️  Backup process interrupted"; exit 130' INT TERM

# Run main function
main "$@"