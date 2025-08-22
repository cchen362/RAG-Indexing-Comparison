# Docker Deployment Guide

This guide explains how to deploy the RAG Comparison App using Docker on your POP server.

## Prerequisites

- Docker and Docker Compose installed on your server
- Sudo access to the server
- Your API keys and Google Sheets credentials

## Quick Deploy

1. **Upload files to server:**
   ```bash
   scp -P 8081 -r RAG-Indexing-Comparison/ chee@75.163.171.40:~/
   ```

2. **SSH into server:**
   ```bash
   ssh -p 8081 chee@75.163.171.40
   ```

3. **Navigate to project and deploy:**
   ```bash
   cd RAG-Indexing-Comparison/
   chmod +x deploy.sh
   ./deploy.sh
   ```

## Files Created

- **Dockerfile**: Defines the container image with Python 3.10, dependencies, and Streamlit
- **docker-compose.yml**: Orchestrates the deployment with port mapping and environment variables
- **deploy.sh**: Automated deployment script with health checks

## Configuration

### Port Mapping
- App runs on **port 8502** (mapped from internal 8501) to avoid conflicts
- Access via: `http://your-server-ip:8502`

### Environment Variables
- Uses your existing `.env` file with API keys
- Google Sheets credentials mounted from `credentials/` folder
- All secrets are embedded in the container (users don't need their own keys)

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

## Updates

To update the application:
```bash
git pull  # if using git
docker-compose down
docker-compose up --build -d
```