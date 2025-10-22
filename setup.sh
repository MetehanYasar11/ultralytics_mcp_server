#!/bin/bash

echo "🚀 RCT Detector Platform - Complete Setup"
echo "====================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if nvidia-docker is available (optional, for GPU support)
if command_exists nvidia-smi; then
    echo "✅ NVIDIA drivers detected"
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        echo "✅ NVIDIA Docker runtime is working"
    else
        echo "⚠️ NVIDIA Docker runtime might not be properly configured"
        echo "   GPU acceleration may not be available"
    fi
else
    echo "⚠️ NVIDIA drivers not detected. Running in CPU mode."
fi

echo ""
echo "🏗️ Building and starting containers..."
echo ""

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start containers
echo "🔨 Building containers (this may take several minutes)..."
docker-compose build --no-cache

echo "🚀 Starting containers..."
docker-compose up -d

# Wait for containers to be ready
echo ""
echo "⏳ Waiting for services to start..."
sleep 15

# Check container status
echo ""
echo "📋 Container Status:"
docker-compose ps

# Check if Streamlit is accessible
echo ""
echo "🌐 Checking service availability..."

# Function to check if a port is accessible
check_port() {
    local port=$1
    local service=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port" >/dev/null 2>&1; then
            echo "✅ $service is accessible on port $port"
            return 0
        fi
        echo "⏳ Waiting for $service (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service is not accessible on port $port after $max_attempts attempts"
    return 1
}

check_port 8501 "Streamlit"
check_port 6006 "TensorBoard"
check_port 8092 "MCP Server"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📱 Available Services:"
echo "  🌐 Streamlit Web Interface: http://localhost:8501"
echo "  📊 TensorBoard: http://localhost:6006"
echo "  🔌 MCP Server: http://localhost:8092"
echo "  📓 Jupyter (optional): http://localhost:8888"
echo ""
echo "🔧 Management Commands:"
echo "  📊 View logs: docker-compose logs -f ultralytics-container"
echo "  🔄 Restart: docker-compose restart"
echo "  🛑 Stop: docker-compose down"
echo "  💾 Monitor: docker exec ultralytics-container /usr/local/bin/monitor.sh"
echo ""
echo "📚 Features Available:"
echo "  ✅ 10GB ZIP dataset upload limit"
echo "  ✅ Intelligent dataset structure detection"
echo "  ✅ Background processing for large files"
echo "  ✅ Real-time GPU monitoring"
echo "  ✅ Custom model training"
echo "  ✅ MCP integration for N8N automation"
echo ""
echo "🚀 Ready to use! Navigate to http://localhost:8501 to get started."
