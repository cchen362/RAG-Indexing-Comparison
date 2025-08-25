#!/bin/bash

# RAG Comparison App - POP Server Deployment Script
# Usage: ./deploy_to_pop.sh

set -e

# Configuration
POP_HOST="75.163.171.40"
POP_PORT="8081"
POP_USER="chee"
APP_NAME="rag-comparison-app"
REPO_URL="https://github.com/yourusername/RAG-Indexing-Comparison.git"  # Update with your repo URL
DEPLOY_DIR="/home/chee/rag-app"

echo "🚀 RAG Comparison App - POP Server Deployment"
echo "==============================================="

# Check prerequisites
echo "🔍 Checking prerequisites..."

if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Please ensure your .env file exists with API keys."
    exit 1
fi

if [ ! -f "credentials/rag-comparison-app-2ebc99e885e4.json" ]; then
    echo "❌ Google Sheets credentials not found!"
    echo "Please ensure credentials/rag-comparison-app-2ebc99e885e4.json exists"
    exit 1
fi

# Encode credentials
echo "🔐 Encoding Google Sheets credentials..."
python3 encode_credentials.py

if [ ! -f ".env.credentials" ]; then
    echo "❌ Failed to encode credentials"
    exit 1
fi

# Read the encoded credentials
GOOGLE_CREDENTIALS_BASE64=$(grep GOOGLE_CREDENTIALS_BASE64 .env.credentials | cut -d'=' -f2)
OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d'=' -f2)
COHERE_API_KEY=$(grep COHERE_API_KEY .env | cut -d'=' -f2)

if [ -z "$GOOGLE_CREDENTIALS_BASE64" ] || [ -z "$OPENAI_API_KEY" ] || [ -z "$COHERE_API_KEY" ]; then
    echo "❌ Missing required environment variables"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create deployment script for remote execution
cat > remote_deploy.sh << 'REMOTE_SCRIPT'
#!/bin/bash
set -e

APP_NAME="rag-comparison-app"
DEPLOY_DIR="/home/chee/rag-app"

echo "🧹 Cleaning up existing deployment..."

# Stop and remove existing container
if [ "$(docker ps -q -f name=$APP_NAME)" ]; then
    echo "Stopping existing container..."
    docker stop $APP_NAME
fi

if [ "$(docker ps -aq -f name=$APP_NAME)" ]; then
    echo "Removing existing container..."
    docker rm $APP_NAME
fi

# Remove existing image
if [ "$(docker images -q rag-indexing-comparison_rag-app)" ]; then
    echo "Removing existing image..."
    docker rmi rag-indexing-comparison_rag-app
fi

echo "🔄 Setting up fresh deployment..."

# Create/update deployment directory
mkdir -p $DEPLOY_DIR
cd $DEPLOY_DIR

# Clone/update repository
if [ -d ".git" ]; then
    echo "Updating existing repository..."
    git fetch origin main
    git reset --hard origin/main
else
    echo "Cloning repository..."
    # Note: You'll need to update this with your actual repo URL
    git clone REPO_URL .
fi

# Create production .env file
echo "📝 Creating production environment file..."
cat > .env << ENV_FILE
OPENAI_API_KEY=API_KEY_PLACEHOLDER
COHERE_API_KEY=COHERE_KEY_PLACEHOLDER
GOOGLE_SHEET_ID=1BcXm5tgOinYD9MwJ1IzRMGFALJtQ-4bKB8mR8KQ5xGE
GOOGLE_CREDENTIALS_BASE64=CREDENTIALS_PLACEHOLDER
ENV_FILE

echo "🏗️  Building and starting application..."

# Build and start with docker-compose
docker-compose up --build -d

echo "⏳ Waiting for application to be healthy..."
sleep 15

# Check if container is running
if [ "$(docker-compose ps -q rag-app)" ]; then
    echo "✅ Container is running"
    
    # Wait for health check
    for i in {1..12}; do
        if docker-compose exec -T rag-app curl -f http://localhost:8501/_stcore/health &> /dev/null; then
            echo "✅ Application is healthy!"
            echo "🌐 App is running at: http://ragindex.zyroi.com"
            echo "📊 View logs: docker-compose logs -f"
            exit 0
        fi
        echo "⏳ Health check $i/12..."
        sleep 10
    done
    
    echo "⚠️  Health check timeout. Check logs:"
    docker-compose logs --tail=20 rag-app
else
    echo "❌ Container failed to start. Check logs:"
    docker-compose logs rag-app
    exit 1
fi
REMOTE_SCRIPT

# Substitute variables in the remote script
sed -i "s|REPO_URL|$REPO_URL|g" remote_deploy.sh
sed -i "s|API_KEY_PLACEHOLDER|$OPENAI_API_KEY|g" remote_deploy.sh
sed -i "s|COHERE_KEY_PLACEHOLDER|$COHERE_API_KEY|g" remote_deploy.sh
sed -i "s|CREDENTIALS_PLACEHOLDER|$GOOGLE_CREDENTIALS_BASE64|g" remote_deploy.sh

echo "📤 Deploying to POP server..."

# Copy and execute deployment script on remote server
scp -P $POP_PORT remote_deploy.sh $POP_USER@$POP_HOST:/tmp/
ssh -p $POP_PORT $POP_USER@$POP_HOST "chmod +x /tmp/remote_deploy.sh && /tmp/remote_deploy.sh"

# Cleanup
rm -f remote_deploy.sh .env.credentials

echo ""
echo "🎉 Deployment completed!"
echo "🌐 Your app should be available at: http://ragindex.zyroi.com"
echo ""
echo "Useful commands:"
echo "  ssh -p $POP_PORT $POP_USER@$POP_HOST 'cd $DEPLOY_DIR && docker-compose logs -f'"
echo "  ssh -p $POP_PORT $POP_USER@$POP_HOST 'cd $DEPLOY_DIR && docker-compose ps'"
echo "  ssh -p $POP_PORT $POP_USER@$POP_HOST 'cd $DEPLOY_DIR && docker-compose restart'"