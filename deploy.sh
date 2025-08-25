#!/bin/bash

# RAG Comparison App Deployment Script
# Usage: ./deploy.sh

set -e

echo "🚀 Starting RAG Comparison App Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Please create .env file with your API keys."
    echo "Required variables:"
    echo "  OPENAI_API_KEY=your_openai_key"
    echo "  COHERE_API_KEY=your_cohere_key"
    exit 1
fi

# Check if credentials exist and encode them
if [ ! -f "credentials/rag-comparison-app-2ebc99e885e4.json" ]; then
    echo "❌ Google Sheets credentials not found!"
    echo "Please ensure credentials/rag-comparison-app-2ebc99e885e4.json exists"
    exit 1
fi

# Encode credentials for Docker injection
echo "🔐 Encoding Google Sheets credentials..."
python3 encode_credentials.py

if [ ! -f ".env.credentials" ]; then
    echo "❌ Failed to encode credentials"
    exit 1
fi

# Append encoded credentials to .env
echo "" >> .env
cat .env.credentials >> .env
echo "✅ Credentials encoded and added to environment"

echo "✅ Prerequisites check passed"

# Clean up existing deployment
echo "🧹 Cleaning up existing deployment..."

# Stop existing containers
docker-compose down --remove-orphans || true

# Remove existing containers and images for fresh build
if [ "$(docker ps -aq -f name=rag-comparison-app)" ]; then
    echo "Removing existing container..."
    docker rm -f rag-comparison-app || true
fi

if [ "$(docker images -q rag-indexing-comparison_rag-app)" ]; then
    echo "Removing existing image for fresh build..."
    docker rmi rag-indexing-comparison_rag-app || true
fi

# Build and start the application
echo "🏗️  Building and starting the application..."
docker-compose up --build -d

# Wait for health check
echo "⏳ Waiting for application to be healthy..."
sleep 10

# Check if container is running
if [ "$(docker-compose ps -q rag-app)" ]; then
    echo "✅ Container is running"
    
    # Check health status
    for i in {1..12}; do  # Wait up to 2 minutes
        if docker-compose exec -T rag-app curl -f http://localhost:8501/_stcore/health &> /dev/null; then
            echo "✅ Application is healthy and ready!"
            echo "🌐 Access your app at: http://$(hostname -I | awk '{print $1}'):8502"
            echo "📊 View logs with: docker-compose logs -f"
            exit 0
        fi
        echo "⏳ Health check attempt $i/12..."
        sleep 10
    done
    
    echo "⚠️  Application started but health check failed. Check logs:"
    docker-compose logs --tail=20 rag-app
else
    echo "❌ Container failed to start. Check logs:"
    docker-compose logs rag-app
    exit 1
fi

# Clean up temporary files
echo "🧹 Cleaning up temporary files..."
rm -f .env.credentials