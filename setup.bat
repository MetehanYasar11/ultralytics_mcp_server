@echo off
echo 🚀 DENTEX AI Platform - Complete Setup
echo =====================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    echo    Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not available. Please install Docker Desktop with Compose.
    pause
    exit /b 1
)

echo ✅ Docker is available
echo.

REM Check for NVIDIA GPU support (optional)
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️ NVIDIA drivers not detected. Running in CPU mode.
) else (
    echo ✅ NVIDIA drivers detected
)

echo.
echo 🏗️ Building and starting containers...
echo.

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Build and start containers
echo 🔨 Building containers (this may take several minutes)...
docker-compose build --no-cache

echo 🚀 Starting containers...
docker-compose up -d

REM Wait for containers to be ready
echo.
echo ⏳ Waiting for services to start...
timeout /t 15 /nobreak >nul

REM Check container status
echo.
echo 📋 Container Status:
docker-compose ps

echo.
echo ⏳ Checking service availability...
timeout /t 5 /nobreak >nul

echo.
echo 🎉 Setup Complete!
echo ==================
echo.
echo 📱 Available Services:
echo   🌐 Streamlit Web Interface: http://localhost:8501
echo   📊 TensorBoard: http://localhost:6006
echo   🔌 MCP Server: http://localhost:8092
echo   📓 Jupyter (optional): http://localhost:8888
echo.
echo 🔧 Management Commands:
echo   📊 View logs: docker-compose logs -f ultralytics-container
echo   🔄 Restart: docker-compose restart
echo   🛑 Stop: docker-compose down
echo   💾 Monitor: docker exec ultralytics-container /usr/local/bin/monitor.sh
echo.
echo 📚 Features Available:
echo   ✅ 10GB ZIP dataset upload limit
echo   ✅ Intelligent dataset structure detection
echo   ✅ Background processing for large files
echo   ✅ Real-time GPU monitoring
echo   ✅ Custom model training
echo   ✅ MCP integration for N8N automation
echo.
echo 🚀 Ready to use! Navigate to http://localhost:8501 to get started.
echo.
pause
