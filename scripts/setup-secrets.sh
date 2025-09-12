#!/bin/bash

# Setup Secrets Management Script
# This script helps configure secrets for different environments

set -e

ENVIRONMENT=${1:-production}
SECRETS_DIR="./secrets"
CONFIG_DIR="./config"

echo "🔐 Setting up secrets for environment: $ENVIRONMENT"

# Create secrets directory if it doesn't exist
mkdir -p "$SECRETS_DIR"

# Function to generate random password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Function to generate secret key
generate_secret_key() {
    openssl rand -hex 32
}

# Check if secrets file exists
SECRETS_FILE="$SECRETS_DIR/secrets.$ENVIRONMENT.yml"

if [[ -f "$SECRETS_FILE" ]]; then
    echo "⚠️  Secrets file already exists: $SECRETS_FILE"
    read -p "Do you want to regenerate secrets? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing secrets file."
        exit 0
    fi
fi

echo "🔑 Generating new secrets..."

# Generate secrets
POSTGRES_PASSWORD=$(generate_password)
POSTGRES_ROOT_PASSWORD=$(generate_password)
INFLUX_PASSWORD=$(generate_password)
INFLUX_TOKEN=$(generate_secret_key)
REDIS_PASSWORD=$(generate_password)
APP_SECRET_KEY=$(generate_secret_key)
JWT_SECRET=$(generate_secret_key)

# Create secrets file
cat > "$SECRETS_FILE" << EOF
# Generated secrets for $ENVIRONMENT environment
# Generated on: $(date)
# DO NOT commit this file to version control

# Database Secrets
database:
  postgres_password: "$POSTGRES_PASSWORD"
  postgres_root_password: "$POSTGRES_ROOT_PASSWORD"

# InfluxDB Secrets
influxdb:
  admin_password: "$INFLUX_PASSWORD"
  admin_token: "$INFLUX_TOKEN"

# Redis Secrets
redis:
  password: "$REDIS_PASSWORD"

# Application Secrets
application:
  secret_key: "$APP_SECRET_KEY"
  jwt_secret: "$JWT_SECRET"

# Email/SMTP Secrets (to be configured manually)
email:
  smtp_username: ""
  smtp_password: ""

# External API Secrets (to be configured manually)
external_apis:
  sentinel_hub_client_id: ""
  sentinel_hub_client_secret: ""
  weather_api_key: ""

# SSL Certificates (to be configured manually)
ssl:
  certificate_path: "/etc/ssl/certs/cert.pem"
  private_key_path: "/etc/ssl/private/key.pem"

# Monitoring and Alerting (to be configured manually)
monitoring:
  webhook_url: ""
  pager_duty_key: ""
EOF

# Set proper permissions
chmod 600 "$SECRETS_FILE"

echo "✅ Secrets file created: $SECRETS_FILE"
echo "🔒 File permissions set to 600 (owner read/write only)"

# Create environment file with secret references
ENV_FILE="$CONFIG_DIR/$ENVIRONMENT.env"

if [[ -f "$ENV_FILE" ]]; then
    # Update existing environment file with generated passwords
    echo "📝 Updating environment file with generated secrets..."
    
    # Create backup
    cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Update passwords in environment file
    sed -i.tmp "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/" "$ENV_FILE"
    sed -i.tmp "s/INFLUX_PASSWORD=.*/INFLUX_PASSWORD=$INFLUX_PASSWORD/" "$ENV_FILE"
    sed -i.tmp "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$REDIS_PASSWORD/" "$ENV_FILE"
    
    # Clean up temporary file
    rm -f "$ENV_FILE.tmp"
    
    echo "✅ Environment file updated: $ENV_FILE"
fi

echo ""
echo "🚨 IMPORTANT SECURITY NOTES:"
echo "1. The secrets file contains sensitive information"
echo "2. Never commit secrets files to version control"
echo "3. Ensure proper file permissions (600) are maintained"
echo "4. Consider using a proper secrets management system for production"
echo "5. Manually configure external API keys and SSL certificates"
echo ""
echo "📋 Next steps:"
echo "1. Review and update external API credentials in: $SECRETS_FILE"
echo "2. Configure SSL certificates for production"
echo "3. Set up monitoring webhook URLs"
echo "4. Test the deployment with: ./scripts/deploy.sh -e $ENVIRONMENT"