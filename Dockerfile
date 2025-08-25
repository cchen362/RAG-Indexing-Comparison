# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs credentials

# Create credential injection script
COPY <<EOF /app/inject_credentials.py
#!/usr/bin/env python3
import os
import base64
import json

def inject_credentials():
    """Inject Google Sheets credentials from environment variable"""
    creds_b64 = os.getenv('GOOGLE_CREDENTIALS_BASE64')
    if creds_b64:
        try:
            # Decode base64 credentials
            creds_json = base64.b64decode(creds_b64).decode('utf-8')
            
            # Parse to validate JSON
            creds_dict = json.loads(creds_json)
            
            # Write to credentials file
            with open('/app/credentials/rag-comparison-app-2ebc99e885e4.json', 'w') as f:
                json.dump(creds_dict, f, indent=2)
            
            print("✅ Google Sheets credentials injected successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to inject credentials: {e}")
            return False
    else:
        print("⚠️ No Google Sheets credentials found in environment")
        return False

if __name__ == "__main__":
    inject_credentials()
EOF

# Make injection script executable
RUN chmod +x /app/inject_credentials.py

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Create startup script
COPY <<EOF /app/start.sh
#!/bin/bash
echo "🚀 Starting RAG Comparison App..."

# Inject credentials
python3 /app/inject_credentials.py

# Start Streamlit
echo "🌐 Starting Streamlit server..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none
EOF

RUN chmod +x /app/start.sh

# Run the application
CMD ["/app/start.sh"]