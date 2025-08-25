# Production Docker Deployment Guide

This guide explains how to deploy the RAG Comparison App using Docker on your POP server with the lessons learned from successful production deployment.

**✅ Currently Live at:** https://ragindex.zyroi.com  
**🐳 Container Status:** `rag-comparison-app` (Running & Healthy)  
**⚡ Last Deployed:** August 25, 2025

## Prerequisites

- Docker and Docker Compose installed on your server
- Sudo access to the server
- Your API keys and Google Sheets credentials

## 🚀 Deployment Methods

### Option 1: Quick Deploy (Recommended)
For automatic deployment directly from your local machine:
```bash
# Make script executable
chmod +x quick_deploy.sh

# Deploy with single command
./quick_deploy.sh
```

### Option 2: SSH Deployment
For manual control over the deployment process:

1. **SSH into server:**
   ```bash
   ssh -p 8081 chee@75.163.171.40
   ```

2. **Navigate to project:**
   ```bash
   cd RAG-Indexing-Comparison
   ```

3. **Deploy:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

### Option 3: Manual Docker Commands
For complete manual control:
```bash
ssh -p 8081 chee@75.163.171.40
cd RAG-Indexing-Comparison

# Stop existing deployment
docker-compose down --remove-orphans
docker rm -f rag-comparison-app
docker rmi rag-indexing-comparison_rag-app

# Deploy fresh
docker-compose up --build -d
```

## Files Created

- **Dockerfile**: Defines the container image with Python 3.10, dependencies, and Streamlit
- **docker-compose.yml**: Orchestrates the deployment with port mapping and environment variables
- **deploy.sh**: Automated deployment script with health checks

## Configuration

### Port Mapping
- App runs on **port 8502** (mapped from internal 8501) to avoid conflicts
- Access via: `http://your-server-ip:8502`

### Secure Environment Variables
- **API Keys**: Injected via environment variables from `.env` file
- **Google Sheets**: Base64-encoded JSON credentials injected at runtime
- **Sheet ID**: Configurable via `GOOGLE_SHEET_ID` environment variable
- **Timezone**: Container runs in UTC (timestamps are UTC+0)

**🔐 Security Features:**
- No credential files in git repository 
- Runtime credential injection via Docker environment
- Base64 encoding prevents credential exposure
- All secrets managed through environment variables

### Volumes
- `credentials/` folder mounted as read-only
- `logs/` folder mounted for log persistence

## Manual Commands

If you prefer manual deployment:

```bash
# Build and start
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down

# Restart application
docker-compose restart
```

## Troubleshooting

### Check container status:
```bash
docker-compose ps
```

### View logs:
```bash
docker-compose logs rag-app
```

### Access container shell:
```bash
docker-compose exec rag-app bash
```

### Health check:
```bash
curl http://localhost:8502/_stcore/health
```

## Security Notes

- API keys are loaded from `.env` file (not exposed to users)
- Google Sheets credentials are mounted securely
- Container runs with non-root user when possible
- Only necessary ports are exposed

## 🎯 Production Deployment Lessons Learned

### ✅ Issues Resolved
1. **python-magic-bin dependency**: Removed from requirements.txt (was causing build failures on Linux)
2. **Google Sheets credentials**: Now injected via base64-encoded environment variable
3. **Container management**: Automated cleanup of previous deployments
4. **Sheet ID configuration**: Fixed incorrect default sheet ID value

### 🔧 Key Improvements Made
- **Runtime credential injection**: No more manual credential file copying
- **Automated deployment scripts**: One-command deployment from local machine
- **Container health monitoring**: Proper health checks and validation
- **Environment-based configuration**: All settings via environment variables

### ⚠️ Common Issues & Solutions

**Issue: Google Sheets 404 Error**  
**Solution**: Verify `GOOGLE_SHEET_ID` environment variable points to correct sheet

**Issue: Container restart doesn't pick up new environment**  
**Solution**: Use `docker-compose down && docker-compose up -d` instead of `restart`

**Issue: Credential injection fails**  
**Solution**: Verify `GOOGLE_CREDENTIALS_BASE64` is properly base64-encoded JSON

### 📊 Current Production Status
- **Domain**: https://ragindex.zyroi.com (auto-redirects HTTP → HTTPS)
- **Container**: `rag-comparison-app` (healthy, running 5+ days)
- **Port**: 8502 → 8501 (preserved for domain continuity)
- **Logs**: UTC timezone (8 hours behind Singapore time)
- **Google Sheets**: ✅ Connected and logging successfully

## Updates

To update the application:
```bash
# Method 1: Quick update (recommended)
./quick_deploy.sh

# Method 2: Manual update
git pull
docker-compose down
docker-compose up --build -d

# Method 3: SSH update
ssh -p 8081 chee@75.163.171.40
cd RAG-Indexing-Comparison
git pull
docker-compose down
docker-compose up --build -d
```

## 🛡️ Security Best Practices Implemented

1. **Secret Management**: All credentials via environment variables
2. **Container Isolation**: Application runs in isolated Docker environment  
3. **Read-only Mounts**: Sensitive files mounted as read-only where possible
4. **Health Monitoring**: Automatic health checks prevent unhealthy deployments
5. **Clean Deployments**: Previous containers/images removed for fresh starts