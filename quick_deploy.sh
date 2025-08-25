#!/bin/bash

# Quick SSH Deployment to POP Server
# Usage: ./quick_deploy.sh

set -e

# Configuration
POP_HOST="75.163.171.40"
POP_PORT="8081"
POP_USER="chee"
DEPLOY_DIR="rag-app"

echo "🚀 Quick Deploy to POP Server (ragindex.zyroi.com)"
echo "================================================="

# Check if we have the required files
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    exit 1
fi

if [ ! -f "credentials/rag-comparison-app-2ebc99e885e4.json" ]; then
    echo "❌ Google Sheets credentials not found!"
    exit 1
fi

# Encode credentials
echo "🔐 Preparing deployment files..."
python3 encode_credentials.py

if [ ! -f ".env.credentials" ]; then
    echo "❌ Failed to encode credentials"
    exit 1
fi

# Read environment variables
OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d'=' -f2)
COHERE_API_KEY=$(grep COHERE_API_KEY .env | cut -d'=' -f2)
GOOGLE_CREDENTIALS_BASE64=$(grep GOOGLE_CREDENTIALS_BASE64 .env.credentials | cut -d'=' -f2)

echo "📤 Deploying to POP server..."

# Create and execute remote deployment
ssh -p $POP_PORT $POP_USER@$POP_HOST << DEPLOY_SCRIPT
set -e

echo "🧹 Stopping existing deployment..."
cd $DEPLOY_DIR 2>/dev/null || mkdir -p $DEPLOY_DIR && cd $DEPLOY_DIR

# Stop existing containers
docker-compose down --remove-orphans 2>/dev/null || true
docker rm -f rag-comparison-app 2>/dev/null || true
docker rmi rag-indexing-comparison_rag-app 2>/dev/null || true

echo "📥 Pulling latest code..."
if [ -d ".git" ]; then
    git fetch origin main
    git reset --hard origin/main
else
    echo "Repository not found. Please manually clone your repo first:"
    echo "git clone <your-repo-url> $DEPLOY_DIR"
    exit 1
fi

echo "🔧 Setting up environment..."
cat > .env << ENV_EOF
OPENAI_API_KEY=$OPENAI_API_KEY
COHERE_API_KEY=$COHERE_API_KEY
GOOGLE_SHEET_ID=1BcXm5tgOinYD9MwJ1IzRMGFALJtQ-4bKB8mR8KQ5xGE
GOOGLE_CREDENTIALS_BASE64=$GOOGLE_CREDENTIALS_BASE64
ENV_EOF

echo "🏗️  Building and starting application..."
docker-compose up --build -d

echo "⏳ Waiting for application to start..."
sleep 20

# Check if container is running and healthy
if [ "\$(docker-compose ps -q rag-app)" ]; then
    echo "✅ Container started successfully"
    
    # Wait for health check
    for i in {1..12}; do
        if docker-compose exec -T rag-app curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            echo "🎉 Deployment successful!"
            echo "🌐 App is live at: http://ragindex.zyroi.com"
            exit 0
        fi
        echo "⏳ Health check \$i/12..."
        sleep 10
    done
    
    echo "⚠️  App started but health check failed. Checking logs..."
    docker-compose logs --tail=10 rag-app
else
    echo "❌ Container failed to start. Checking logs..."
    docker-compose logs rag-app
    exit 1
fi
DEPLOY_SCRIPT

deployment_status=$?

# Clean up local temporary files
rm -f .env.credentials

if [ $deployment_status -eq 0 ]; then
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "🌐 Your app is live at: http://ragindex.zyroi.com"
    echo ""
    echo "Management commands:"
    echo "  View logs: ssh -p $POP_PORT $POP_USER@$POP_HOST 'cd $DEPLOY_DIR && docker-compose logs -f'"
    echo "  Restart:   ssh -p $POP_PORT $POP_USER@$POP_HOST 'cd $DEPLOY_DIR && docker-compose restart'"
    echo "  Status:    ssh -p $POP_PORT $POP_USER@$POP_HOST 'cd $DEPLOY_DIR && docker-compose ps'"
else
    echo ""
    echo "❌ Deployment failed. Check the logs above."
    echo "SSH to server: ssh -p $POP_PORT $POP_USER@$POP_HOST"
fi