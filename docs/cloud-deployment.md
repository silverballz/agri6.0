# AgriFlux Cloud Deployment Guide

## 🌐 Cloud Platform Options

### **1. 🚀 AWS Deployment**

#### **Option A: AWS ECS (Recommended)**
```bash
# Prerequisites
aws configure
docker login

# Build and push images
docker build -t agriflux:latest .
docker tag agriflux:latest your-account.dkr.ecr.region.amazonaws.com/agriflux:latest
docker push your-account.dkr.ecr.region.amazonaws.com/agriflux:latest

# Deploy using ECS CLI
ecs-cli compose --file docker-compose.prod.yml up --cluster agriflux-cluster
```

#### **Option B: AWS EC2 with Docker**
```bash
# Launch EC2 instance (t3.large or larger recommended)
# Install Docker and Docker Compose
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -a -G docker ec2-user

# Clone and deploy
git clone <your-repo>
cd agriflux
./scripts/init-system.sh production
./scripts/deploy.sh -e production
```

#### **AWS Services Integration:**
- **RDS PostgreSQL**: Managed database
- **ElastiCache Redis**: Managed caching
- **CloudWatch**: Monitoring and logging
- **ALB**: Load balancing
- **Route 53**: DNS management
- **S3**: Data storage and backups

### **2. 🔵 Azure Deployment**

#### **Azure Container Instances**
```bash
# Login to Azure
az login

# Create resource group
az group create --name agriflux-rg --location eastus

# Deploy container group
az container create \
  --resource-group agriflux-rg \
  --name agriflux-app \
  --image agriflux:latest \
  --ports 8501 \
  --environment-variables ENVIRONMENT=production
```

#### **Azure Services:**
- **Azure Database for PostgreSQL**: Managed database
- **Azure Cache for Redis**: Managed caching
- **Azure Monitor**: Monitoring and alerting
- **Azure Load Balancer**: Traffic distribution
- **Azure Blob Storage**: File storage

### **3. 🟡 Google Cloud Platform**

#### **Google Cloud Run**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/agriflux
gcloud run deploy --image gcr.io/PROJECT-ID/agriflux --platform managed
```

#### **GCP Services:**
- **Cloud SQL**: Managed PostgreSQL
- **Memorystore**: Managed Redis
- **Cloud Monitoring**: Observability
- **Cloud Load Balancing**: Traffic management
- **Cloud Storage**: Data storage

### **4. 🟢 DigitalOcean Deployment**

#### **DigitalOcean App Platform**
```yaml
# app.yaml
name: agriflux
services:
- name: web
  source_dir: /
  github:
    repo: your-username/agriflux
    branch: main
  run_command: streamlit run src/dashboard/main.py
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  http_port: 8501
```

#### **DigitalOcean Droplet**
```bash
# Create droplet and install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Deploy AgriFlux
git clone <your-repo>
cd agriflux
./scripts/deploy.sh -e production
```

## 🔧 **Environment Configuration**

### **Production Environment Variables**
```bash
# Database
POSTGRES_HOST=your-db-host
POSTGRES_DB=agriflux_prod
POSTGRES_USER=agriflux_user
POSTGRES_PASSWORD=secure_password

# Application
ENVIRONMENT=production
APP_PORT=8501
DEBUG=false

# External APIs
SENTINEL_HUB_CLIENT_ID=your_client_id
WEATHER_API_KEY=your_api_key

# Security
SECRET_KEY=your_secret_key
SSL_CERT_PATH=/etc/ssl/certs/cert.pem
```

### **Scaling Configuration**
```yaml
# docker-compose.prod.yml scaling
services:
  app:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
      restart_policy:
        condition: on-failure
```

## 📊 **Monitoring & Observability**

### **Health Checks**
```bash
# Application health
curl -f http://your-domain/health

# Database connectivity
./scripts/monitor.sh production

# System metrics
docker stats
```

### **Logging**
```bash
# Application logs
docker-compose logs -f app

# System logs
journalctl -u docker

# Nginx access logs
tail -f logs/nginx/access.log
```

## 🔒 **Security Best Practices**

### **SSL/TLS Configuration**
```bash
# Let's Encrypt SSL (recommended)
certbot --nginx -d your-domain.com

# Or use your own certificates
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem
```

### **Firewall Rules**
```bash
# Allow only necessary ports
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw enable
```

### **Secrets Management**
```bash
# Use cloud-native secret managers
# AWS Secrets Manager, Azure Key Vault, GCP Secret Manager

# Or use Docker secrets
echo "your_secret" | docker secret create db_password -
```

## 🚀 **Deployment Checklist**

### **Pre-Deployment**
- [ ] Configure environment variables
- [ ] Set up SSL certificates
- [ ] Configure database connections
- [ ] Test backup and restore procedures
- [ ] Set up monitoring and alerting

### **Deployment**
- [ ] Run system initialization
- [ ] Deploy application stack
- [ ] Verify all services are running
- [ ] Run health checks
- [ ] Test application functionality

### **Post-Deployment**
- [ ] Configure monitoring dashboards
- [ ] Set up automated backups
- [ ] Configure log rotation
- [ ] Test disaster recovery procedures
- [ ] Document deployment process

## 📞 **Support & Troubleshooting**

### **Common Issues**
1. **Port conflicts**: Check if ports 8501, 5432, 8086 are available
2. **Memory issues**: Ensure sufficient RAM (minimum 8GB recommended)
3. **SSL issues**: Verify certificate paths and permissions
4. **Database connectivity**: Check network settings and credentials

### **Getting Help**
- 📧 Email: support@agriflux.com
- 📖 Documentation: [Technical Docs](technical-documentation.md)
- 🐛 Issues: Submit via GitHub or support portal

---

*Choose the deployment option that best fits your infrastructure requirements and technical expertise.*