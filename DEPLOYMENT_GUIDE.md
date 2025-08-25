# 🚀 RAG Comparison App - Deployment Guide

## Overview

This guide covers deploying the RAG Comparison App to your POP server with the domain `ragindex.zyroi.com`.

## 🔧 Key Solutions Implemented

### ✅ **Fixed Issues from Previous Deployment**

1. **python-magic-bin dependency** - Removed from requirements.txt (was unused and causing build failures)
2. **Google Sheets credentials** - Now injected via base64-encoded environment variable (no more manual file copying)
3. **API key management** - Injected via Docker environment variables from your local .env
4. **Container cleanup** - Automated removal of previous deployments

### 🛡️ **Credential Security**

- Google Sheets JSON credentials are base64-encoded and injected as environment variable
- No credential files are stored in git repository
- Runtime credential file generation inside Docker container
- API keys injected from your local .env file

## 📋 Prerequisites

1. **Local files required:**
   - `.env` file with your API keys:
     ```bash
     OPENAI_API_KEY=sk-your-key-here
     COHERE_API_KEY=your-key-here
     ```
   - `credentials/rag-comparison-app-2ebc99e885e4.json` - Your Google Sheets service account JSON

2. **POP Server access:**
   - SSH access: `ssh -p 8081 chee@75.163.171.40`
   - Docker and docker-compose installed
   - Git repository cloned on server

## 🚀 Deployment Options

### Option 1: Quick Deploy (Recommended)

Single command deployment with automatic cleanup:

```bash
chmod +x quick_deploy.sh
./quick_deploy.sh
```

This script will:
- Encode your Google Sheets credentials
- SSH to your POP server
- Stop and remove existing containers
- Pull latest code from git
- Build and deploy the new version
- Verify the deployment

### Option 2: Local Testing + Manual Deploy

Test locally first, then deploy:

```bash
# Test locally
chmod +x deploy.sh
./deploy.sh

# If local test works, then deploy to POP server
chmod +x deploy_to_pop.sh
./deploy_to_pop.sh
```

### Option 3: Manual SSH Deployment

If you prefer manual control:

```bash
# SSH to POP server
ssh -p 8081 chee@75.163.171.40

# Go to your app directory
cd rag-app

# Stop existing deployment
docker-compose down --remove-orphans
docker rm -f rag-comparison-app
docker rmi rag-indexing-comparison_rag-app

# Pull latest changes
git pull origin main

# Create .env file with your keys
nano .env
# Add:
# OPENAI_API_KEY=your-key
# COHERE_API_KEY=your-key  
# GOOGLE_CREDENTIALS_BASE64=your-base64-encoded-json

# Deploy
docker-compose up --build -d
```

## 🔍 Verification

After deployment, verify everything is working:

1. **Check container status:**
   ```bash
   ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose ps'
   ```

2. **Check application health:**
   ```bash
   curl -f http://ragindex.zyroi.com
   ```

3. **View logs:**
   ```bash
   ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose logs -f'
   ```

## 🛠️ Management Commands

```bash
# View logs
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose logs -f'

# Restart application
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose restart'

# Check status
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose ps'

# Stop application
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose down'

# View container resource usage
ssh -p 8081 chee@75.163.171.40 'docker stats rag-comparison-app'
```

## 🔧 Configuration Details

### Docker Configuration
- **Container name:** `rag-comparison-app`
- **Port mapping:** `8502:8501` (external:internal)
- **Health checks:** Enabled with 30s intervals
- **Restart policy:** `unless-stopped`

### Environment Variables
- `OPENAI_API_KEY` - Your OpenAI API key
- `COHERE_API_KEY` - Your Cohere API key
- `GOOGLE_SHEET_ID` - Target Google Sheet ID (default provided)
- `GOOGLE_CREDENTIALS_BASE64` - Base64-encoded service account JSON

### Network
- **Domain:** ragindex.zyroi.com
- **Port:** 8502 (mapped to container port 8501)
- **Protocol:** HTTP

## 🐛 Troubleshooting

### Container Won't Start
```bash
# Check logs
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose logs'

# Check if port is in use
ssh -p 8081 chee@75.163.171.40 'netstat -tulpn | grep 8502'

# Rebuild from scratch
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose down && docker-compose up --build'
```

### Credentials Issues
```bash
# Verify credentials are injected
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose exec rag-app ls -la /app/credentials/'

# Test credential injection manually
python3 encode_credentials.py
```

### Health Check Failures
```bash
# Check Streamlit health endpoint
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose exec rag-app curl http://localhost:8501/_stcore/health'

# Check application logs
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose logs -f rag-app'
```

## 📊 Post-Deployment Testing

1. **Access the application:** http://ragindex.zyroi.com
2. **Test document upload and processing**
3. **Verify Google Sheets logging works**
4. **Test different embedding providers**
5. **Check retrieval system functionality**

## 🔄 Rolling Back

If you need to rollback:

```bash
ssh -p 8081 chee@75.163.171.40 'cd rag-app && git log --oneline -10'
ssh -p 8081 chee@75.163.171.40 'cd rag-app && git reset --hard COMMIT_HASH'
ssh -p 8081 chee@75.163.171.40 'cd rag-app && docker-compose up --build -d'
```

---

## 🎉 Success!

Your RAG Comparison App should now be running at **http://ragindex.zyroi.com** with:
- ✅ Resolved python-magic-bin dependency issues
- ✅ Automated Google Sheets credential injection
- ✅ Clean container management
- ✅ Production-ready Docker configuration
- ✅ Comprehensive monitoring and logging