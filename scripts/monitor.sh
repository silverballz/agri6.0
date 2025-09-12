#!/bin/bash

# System Monitoring Script
# Monitors system health, service status, and performance metrics

set -e

ENVIRONMENT=${1:-production}
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90
LOG_FILE="./logs/monitor.log"

# Load environment variables
if [[ -f "config/$ENVIRONMENT.env" ]]; then
    export $(grep -v '^#' "config/$ENVIRONMENT.env" | xargs)
fi

# Create logs directory
mkdir -p ./logs

# Function to log with timestamp
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message"
    echo "$message" >> "$LOG_FILE"
}

# Function to send alert
send_alert() {
    local severity=$1
    local message=$2
    local webhook_url="${WEBHOOK_URL:-}"
    
    log "🚨 ALERT [$severity]: $message"
    
    if [[ -n "$webhook_url" ]]; then
        local emoji
        case $severity in
            "CRITICAL") emoji="🔴" ;;
            "WARNING") emoji="🟡" ;;
            "INFO") emoji="🔵" ;;
            *) emoji="⚪" ;;
        esac
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji [$severity] Agricultural Monitoring Platform\\n$message\"}" \
            "$webhook_url" 2>/dev/null || true
    fi
}

# Check system resources
check_system_resources() {
    log "📊 Checking system resources..."
    
    # CPU usage
    local cpu_usage=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' | cut -d'.' -f1)
    if [[ $cpu_usage -gt $ALERT_THRESHOLD_CPU ]]; then
        send_alert "WARNING" "High CPU usage: ${cpu_usage}%"
    fi
    
    # Memory usage
    local memory_info=$(vm_stat | grep -E "(free|inactive|active|wired)")
    local pages_free=$(echo "$memory_info" | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    local pages_active=$(echo "$memory_info" | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
    local pages_inactive=$(echo "$memory_info" | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
    local pages_wired=$(echo "$memory_info" | grep "Pages wired down" | awk '{print $4}' | sed 's/\.//')
    
    local total_pages=$((pages_free + pages_active + pages_inactive + pages_wired))
    local used_pages=$((pages_active + pages_inactive + pages_wired))
    local memory_usage=$((used_pages * 100 / total_pages))
    
    if [[ $memory_usage -gt $ALERT_THRESHOLD_MEMORY ]]; then
        send_alert "WARNING" "High memory usage: ${memory_usage}%"
    fi
    
    # Disk usage
    local disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt $ALERT_THRESHOLD_DISK ]]; then
        send_alert "CRITICAL" "High disk usage: ${disk_usage}%"
    fi
    
    log "✅ System resources: CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Disk: ${disk_usage}%"
}

# Check Docker services
check_docker_services() {
    log "🐳 Checking Docker services..."
    
    if ! docker info > /dev/null 2>&1; then
        send_alert "CRITICAL" "Docker daemon is not running"
        return 1
    fi
    
    # Check if docker-compose services are running
    local services_down=()
    
    while IFS= read -r line; do
        if [[ $line == *"Exit"* ]] || [[ $line == *"Down"* ]]; then
            local service_name=$(echo "$line" | awk '{print $1}')
            services_down+=("$service_name")
        fi
    done < <(docker-compose ps 2>/dev/null | tail -n +3)
    
    if [[ ${#services_down[@]} -gt 0 ]]; then
        send_alert "CRITICAL" "Services down: ${services_down[*]}"
    else
        log "✅ All Docker services are running"
    fi
}

# Check application health
check_application_health() {
    log "🏥 Checking application health..."
    
    local app_port="${APP_PORT:-8501}"
    local health_url="http://localhost:${app_port}/_stcore/health"
    
    if curl -f -s "$health_url" > /dev/null; then
        log "✅ Application health check passed"
    else
        send_alert "CRITICAL" "Application health check failed"
    fi
}

# Check database connectivity
check_database_connectivity() {
    log "🗄️  Checking database connectivity..."
    
    # PostgreSQL
    if docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER:-agri_user}" -d "${POSTGRES_DB:-agricultural_monitoring}" > /dev/null 2>&1; then
        log "✅ PostgreSQL is accessible"
    else
        send_alert "CRITICAL" "PostgreSQL connectivity failed"
    fi
    
    # InfluxDB
    local influx_port="${INFLUX_PORT:-8086}"
    if curl -f -s "http://localhost:${influx_port}/health" > /dev/null; then
        log "✅ InfluxDB is accessible"
    else
        send_alert "CRITICAL" "InfluxDB connectivity failed"
    fi
    
    # Redis
    local redis_port="${REDIS_PORT:-6379}"
    if docker-compose exec -T redis redis-cli -p 6379 ping > /dev/null 2>&1; then
        log "✅ Redis is accessible"
    else
        send_alert "CRITICAL" "Redis connectivity failed"
    fi
}

# Check log files for errors
check_log_errors() {
    log "📋 Checking recent log errors..."
    
    local error_count=0
    
    # Check application logs for errors in the last hour
    if [[ -d "./logs" ]]; then
        local recent_errors=$(find ./logs -name "*.log" -mmin -60 -exec grep -i "error\|exception\|critical" {} \; 2>/dev/null | wc -l)
        error_count=$((error_count + recent_errors))
    fi
    
    # Check Docker logs for errors
    local docker_errors=$(docker-compose logs --since=1h 2>/dev/null | grep -i "error\|exception\|critical" | wc -l)
    error_count=$((error_count + docker_errors))
    
    if [[ $error_count -gt 10 ]]; then
        send_alert "WARNING" "High error count in logs: $error_count errors in the last hour"
    elif [[ $error_count -gt 0 ]]; then
        log "⚠️  Found $error_count error(s) in recent logs"
    else
        log "✅ No recent errors found in logs"
    fi
}

# Check SSL certificate expiration
check_ssl_certificates() {
    log "🔒 Checking SSL certificate status..."
    
    local cert_path="${SSL_CERT_PATH:-./nginx/ssl/cert.pem}"
    
    if [[ -f "$cert_path" ]]; then
        local expiry_date=$(openssl x509 -enddate -noout -in "$cert_path" | cut -d= -f2)
        local expiry_timestamp=$(date -d "$expiry_date" +%s 2>/dev/null || date -j -f "%b %d %H:%M:%S %Y %Z" "$expiry_date" +%s 2>/dev/null)
        local current_timestamp=$(date +%s)
        local days_until_expiry=$(( (expiry_timestamp - current_timestamp) / 86400 ))
        
        if [[ $days_until_expiry -lt 30 ]]; then
            send_alert "WARNING" "SSL certificate expires in $days_until_expiry days"
        elif [[ $days_until_expiry -lt 7 ]]; then
            send_alert "CRITICAL" "SSL certificate expires in $days_until_expiry days"
        else
            log "✅ SSL certificate valid for $days_until_expiry days"
        fi
    else
        log "⚠️  SSL certificate not found at $cert_path"
    fi
}

# Check backup status
check_backup_status() {
    log "💾 Checking backup status..."
    
    local backup_dir="./backups/$ENVIRONMENT"
    
    if [[ -d "$backup_dir" ]]; then
        local latest_backup=$(find "$backup_dir" -type d -name "20*" | sort | tail -1)
        
        if [[ -n "$latest_backup" ]]; then
            local backup_age=$(( ($(date +%s) - $(date -r "$latest_backup" +%s)) / 86400 ))
            
            if [[ $backup_age -gt 7 ]]; then
                send_alert "WARNING" "Latest backup is $backup_age days old"
            else
                log "✅ Latest backup is $backup_age day(s) old"
            fi
        else
            send_alert "WARNING" "No backups found"
        fi
    else
        send_alert "WARNING" "Backup directory not found"
    fi
}

# Generate monitoring report
generate_report() {
    log "📊 Generating monitoring report..."
    
    local report_file="./logs/monitoring_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Agricultural Monitoring Platform - System Report
===============================================
Environment: $ENVIRONMENT
Generated: $(date)
Hostname: $(hostname)

System Resources:
================
$(top -l 1 | head -10)

Disk Usage:
===========
$(df -h)

Docker Services:
===============
$(docker-compose ps 2>/dev/null || echo "Docker Compose not available")

Recent Logs (Last 50 lines):
============================
$(tail -50 "$LOG_FILE" 2>/dev/null || echo "No recent logs available")
EOF
    
    log "✅ Monitoring report generated: $report_file"
}

# Main monitoring function
main() {
    log "🚀 Starting system monitoring for environment: $ENVIRONMENT"
    
    check_system_resources
    check_docker_services
    check_application_health
    check_database_connectivity
    check_log_errors
    check_ssl_certificates
    check_backup_status
    
    # Generate report if requested
    if [[ "${2:-}" == "--report" ]]; then
        generate_report
    fi
    
    log "✅ Monitoring check completed"
}

# Handle script interruption
trap 'log "⚠️  Monitoring interrupted"; exit 130' INT TERM

# Run main function
main "$@"