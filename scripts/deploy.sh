#!/bin/bash

# Agricultural Monitoring Platform Deployment Script
# This script handles deployment to different environments

set -e

# Default values
ENVIRONMENT="production"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment    Environment to deploy (development|staging|production)"
            echo "  -f, --file          Docker compose file to use"
            echo "  --env-file          Environment file to use"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "🚀 Deploying Agricultural Monitoring Platform"
echo "Environment: $ENVIRONMENT"

# Set environment-specific configurations
case $ENVIRONMENT in
    development)
        ENV_FILE="${ENV_FILE:-config/development.env}"
        COMPOSE_FILES="-f docker-compose.yml"
        COMPOSE_PROFILES=""
        ;;
    staging)
        ENV_FILE="${ENV_FILE:-config/staging.env}"
        COMPOSE_FILES="-f docker-compose.yml -f docker-compose.staging.yml"
        COMPOSE_PROFILES="--profile staging"
        ;;
    production)
        ENV_FILE="${ENV_FILE:-config/production.env}"
        COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
        COMPOSE_PROFILES="--profile production"
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        echo "Valid environments: development, staging, production"
        exit 1
        ;;
esac

# Check if environment file exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ Environment file not found: $ENV_FILE"
    echo "Please create the environment file or use --env-file to specify a different one"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' "$ENV_FILE" | xargs)

echo "📋 Pre-deployment checks..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if required directories exist
mkdir -p data/{postgres,influxdb,redis,app,prometheus,grafana,loki} logs models backups/{postgres,influxdb,redis,app}

# Set proper permissions
chmod 755 data logs models backups
chmod -R 755 data/
chmod -R 755 backups/

echo "🔧 Building and starting services..."

# Stop existing services
docker-compose $COMPOSE_FILES down

# Pull latest images
docker-compose $COMPOSE_FILES pull

# Build application image
docker-compose $COMPOSE_FILES build --no-cache app

# Start services
docker-compose $COMPOSE_FILES $COMPOSE_PROFILES up -d

echo "⏳ Waiting for services to be ready..."

# Wait for database to be ready
echo "Waiting for PostgreSQL..."
until docker-compose $COMPOSE_FILES exec -T postgres pg_isready -U "${POSTGRES_USER:-agri_user}" -d "${POSTGRES_DB:-agricultural_monitoring}"; do
    sleep 2
done

# Wait for InfluxDB to be ready
echo "Waiting for InfluxDB..."
until curl -f http://localhost:${INFLUX_PORT:-8086}/health > /dev/null 2>&1; do
    sleep 2
done

# Wait for application to be ready
echo "Waiting for application..."
until curl -f http://localhost:${APP_PORT:-8501}/_stcore/health > /dev/null 2>&1; do
    sleep 5
done

echo "🗄️  Running database migrations..."
docker-compose $COMPOSE_FILES exec -T app python -c "
from src.database.migrations import run_migrations
run_migrations()
print('Database migrations completed successfully')
"

echo "🧪 Running health checks..."

# Check application health
if curl -f http://localhost:${APP_PORT:-8501}/_stcore/health > /dev/null 2>&1; then
    echo "✅ Application is healthy"
else
    echo "❌ Application health check failed"
    exit 1
fi

# Check database connectivity
if docker-compose $COMPOSE_FILES exec -T postgres pg_isready -U "${POSTGRES_USER:-agri_user}" -d "${POSTGRES_DB:-agricultural_monitoring}" > /dev/null 2>&1; then
    echo "✅ Database is accessible"
else
    echo "❌ Database connectivity check failed"
    exit 1
fi

echo "📊 Deployment summary:"
echo "  Environment: $ENVIRONMENT"
echo "  Application URL: http://localhost:${APP_PORT:-8501}"
echo "  Database: PostgreSQL on port ${POSTGRES_PORT:-5432}"
echo "  InfluxDB: http://localhost:${INFLUX_PORT:-8086}"
echo "  Redis: localhost:${REDIS_PORT:-6379}"

echo "🎉 Deployment completed successfully!"

# Show running containers
echo "📦 Running containers:"
docker-compose $COMPOSE_FILES ps

echo ""
echo "💡 Useful commands:"
echo "  View logs: docker-compose $COMPOSE_FILES logs -f"
echo "  Stop services: docker-compose $COMPOSE_FILES down"
echo "  Restart app: docker-compose $COMPOSE_FILES restart app"
echo "  Access database: docker-compose $COMPOSE_FILES exec postgres psql -U ${POSTGRES_USER:-agri_user} -d ${POSTGRES_DB:-agricultural_monitoring}"
echo "  Monitor system: ./scripts/monitor.sh $ENVIRONMENT"
echo "  Create backup: ./scripts/backup.sh $ENVIRONMENT"