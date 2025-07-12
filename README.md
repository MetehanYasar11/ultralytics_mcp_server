# Ultralytics MCP Server 🚀

[![Build Status](https://github.com/MetehanYasar11/ultralytics_mcp_server/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/MetehanYasar11/ultralytics_mcp_server/actions)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io%2Fmetehanyasar11%2Fultra--api-blue)](https://github.com/MetehanYasar11/ultralytics_mcp_server/pkgs/container/ultra-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF.svg)](https://ultralytics.com)

> **A powerful Model Context Protocol (MCP) compliant server that provides RESTful API access to Ultralytics YOLO operations for computer vision tasks including training, validation, prediction, export, tracking, and benchmarking.**

## 🎯 What is this?

The Ultralytics MCP Server transforms Ultralytics' command-line YOLO operations into a production-ready REST API service. Whether you're building computer vision applications, training custom models, or integrating YOLO into your workflow automation tools like n8n, this server provides a seamless bridge between Ultralytics' powerful capabilities and modern application architectures.

## ✨ Key Features

- **🌐 RESTful API**: HTTP endpoints for all YOLO operations with comprehensive request/response validation
- **📡 Real-time Updates**: Server-Sent Events (SSE) for monitoring long-running operations like training
- **🔧 MCP Compliance**: Full Model Context Protocol support for workflow automation tools
- **🐳 Production Ready**: Docker containerization with multi-stage builds and security scanning
- **🧪 Battle Tested**: Comprehensive test suite with CI/CD pipeline and 90%+ code coverage
- **📊 Observability**: Built-in metrics parsing, health checks, and monitoring endpoints
- **🔒 Enterprise Security**: API key authentication, input validation, and vulnerability scanning
- **⚡ CPU & GPU Support**: Automatic device detection with graceful fallbacks
- **📚 Self-Documenting**: Auto-generated OpenAPI/Swagger documentation

## 🏗️ Architecture Overview

```mermaid
graph TB
    A[Client Applications] --> B[FastAPI REST API]
    B --> C[Pydantic Validation]
    C --> D[Ultralytics CLI Engine]
    D --> E[YOLO Models]
    
    B --> F[SSE Events]
    D --> G[Metrics Parser]
    D --> H[File System Artifacts]
    
    I[Docker Container] --> B
    J[MCP Client Tools] --> B
    K[n8n Workflows] --> B
```

**Core Components:**
- **`app/main.py`**: FastAPI application with route definitions and middleware
- **`app/schemas.py`**: Pydantic models for comprehensive request/response validation
- **`app/ultra.py`**: Ultralytics CLI integration with metrics parsing and device management
- **`tools/UltralyticsMCPTool`**: TypeScript MCP client library for workflow automation

## 🚀 Quick Start

### 📋 Prerequisites

- **Python 3.11+** (required for compatibility)
- **Conda/Miniconda** (recommended for environment management)
- **Git** (for cloning the repository)
- **4GB+ RAM** (for model operations)
- **Optional**: NVIDIA GPU with CUDA support for faster training

### ⚡ One-Minute Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/MetehanYasar11/ultralytics_mcp_server.git
cd ultralytics_mcp_server

# 2. Create environment and install dependencies
conda env create -f environment.yml
conda activate ultra-dev

# 3. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Test it works (in another terminal)
curl http://localhost:8000/
```

### 🔍 Verify Installation

After setup, verify everything works:

```bash
# Check health endpoint
curl http://localhost:8000/
# Expected: {"status":"healthy","message":"Ultralytics API is running",...}

# View interactive API documentation
open http://localhost:8000/docs

# Test a simple prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n.pt",
    "source": "https://ultralytics.com/images/bus.jpg",
    "conf": 0.5,
    "save": true
  }'
```

### 📖 What Just Happened?

1. **Environment Setup**: Created isolated conda environment with PyTorch CPU support
2. **Dependency Installation**: Installed Ultralytics, FastAPI, and all required packages  
3. **Server Start**: Launched FastAPI server with auto-reload for development
4. **API Test**: Made a prediction request using a pre-trained YOLOv8 nano model

## 🟢 Real-time SSE with n8n

![SSE Live Logs](https://img.shields.io/badge/SSE-Live%20Logs-brightgreen?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K)

**Live streaming updates for YOLO operations with Server-Sent Events (SSE)**

### Using SSE in n8n  
1. Drag **MCP Client Tool** ➜ set *SSE Endpoint* `http://host.docker.internal:8092/sse/train` (or compose DNS `ultra-api:8000/sse/train`).  
2. In **Settings ➜ OpenAPI URL** use `http://host.docker.internal:8092/openapi.json`.  
3. Set **Timeout** `0` to keep stream open.  
4. Run workflow → live epoch/loss lines appear in the node's execution log.  

**🎯 Available SSE Endpoints:**
- `/sse/train` - Real-time training progress with epoch updates
- `/sse/predict` - Live prediction results 
- `/sse/val` - Validation metrics streaming
- `/sse/export` - Export progress updates
- `/sse/track` - Object tracking stream
- `/sse/benchmark` - Performance testing results
- `/sse/solution` - Solution execution logs

**📊 SSE Example:**
```bash
curl -N "http://localhost:8092/sse/train?data=coco128.yaml&epochs=1&device=cpu"
# Output:
# data: Ultralytics YOLOv8.0.196 🚀 Python-3.11.5 torch-2.1.1
# data: 
# data: train: Scanning /datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128
# data: val: Scanning /datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128  
# data: 
# data: Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
# data:   1/1      0.12G      1.325      2.009      1.268         89        640: 100%|██████████| 8/8
# data: [COMPLETED] Process finished successfully
```

**🔗 OpenAPI Documentation:** [http://localhost:8092/docs#/default/sse_endpoint_sse__op__get](http://localhost:8092/docs#/default/sse_endpoint_sse__op__get)

## 🧪 Running Tests

Our comprehensive test suite ensures reliability across all operations.

### 🏃‍♂️ Quick Test

```bash
# Run all tests (recommended)
python run_tests.py

# View test progress with details
pytest tests/test_flow.py -v -s

# Run only fast tests (skip training)
python run_tests.py quick
```

### 🔬 Test Categories

| Test Type | Command | Duration | What it Tests |
|-----------|---------|----------|---------------|
| **Unit Tests** | `pytest tests/test_unit.py` | ~10s | Individual functions |
| **Integration** | `pytest tests/test_flow.py` | ~5min | Complete workflows |
| **Quick Check** | `python run_tests.py quick` | ~30s | Endpoints only |
| **Full Suite** | `python run_tests.py` | ~5min | Everything including training |

### 📊 Understanding Test Output

```bash
tests/test_flow.py::TestUltralyticsFlow::test_health_check ✅ PASSED
tests/test_flow.py::TestUltralyticsFlow::test_01_train_model ✅ PASSED  
tests/test_flow.py::TestUltralyticsFlow::test_02_validate_model ✅ PASSED
tests/test_flow.py::TestUltralyticsFlow::test_03_predict_with_model ✅ PASSED
# ... more tests

======================== 9 passed in 295.15s ========================
```

The integration test performs a complete YOLO workflow:
1. **🏥 Health Check** - Verify API is responsive
2. **🏋️ Model Training** - Train YOLOv8n for 1 epoch on COCO128  
3. **🔍 Model Validation** - Validate the trained model
4. **🎯 Prediction** - Run inference on a test image
5. **� File Verification** - Check all expected outputs were created

## CI/CD Workflow

The project uses GitHub Actions for continuous integration and deployment. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the complete configuration.

### Workflow Jobs

1. **🧪 Test Job**
   - Sets up Conda environment with caching
   - Runs pytest with coverage reporting
   - Uploads coverage to Codecov

2. **🐳 Build Job** (on success)
   - Builds Docker image with multi-stage optimization
   - Pushes to GitHub Container Registry
   - Supports multi-platform builds (amd64, arm64)

3. **🔒 Security Job**
   - Runs Trivy vulnerability scanner
   - Uploads SARIF results to GitHub Security

4. **🔗 Integration Job**
   - Tests complete API workflow
   - Validates endpoint responses
   - Checks health and documentation endpoints

### Workflow Triggers

- **Push** to `main` or `develop` branches
- **Pull Requests** to `main` branch
- **Manual** workflow dispatch

### Caching Strategy

```yaml
# Conda packages cached by environment.yml hash
key: conda-${{ runner.os }}-${{ hashFiles('environment.yml') }}

# Docker layers cached using GitHub Actions cache
cache-from: type=gha
cache-to: type=gha,mode=max
```

## Docker Deployment

### Quick Deploy

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f ultra-api
```

### Production Deployment

```bash
# Production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile monitoring up -d
```

### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Key Variables:**
- `ULTRA_API_KEY`: API authentication key
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `MEMORY_LIMIT`: Container memory limit

### Service Access

Once deployed, access the service at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Prometheus** (if enabled): http://localhost:9090
- **Grafana** (if enabled): http://localhost:3000

For detailed Docker configuration, see [DOCKER.md](DOCKER.md).

## 📚 API Reference & Examples

### 🎯 Core Operations

| Operation | Endpoint | Purpose | Example Use Case |
|-----------|----------|---------|------------------|
| **Train** | `POST /train` | Train custom models | Training on your dataset |
| **Validate** | `POST /val` | Model performance testing | Check accuracy metrics |
| **Predict** | `POST /predict` | Object detection/classification | Real-time inference |
| **Export** | `POST /export` | Format conversion | Deploy to mobile/edge |
| **Track** | `POST /track` | Object tracking in videos | Surveillance, sports analysis |
| **Benchmark** | `POST /benchmark` | Performance testing | Hardware optimization |

### 📝 Request/Response Format

All endpoints return a standardized response structure:

```json
{
  "run_id": "abc123-def456-ghi789",
  "command": "yolo train model=yolov8n.pt data=coco128.yaml epochs=10",
  "return_code": 0,
  "stdout": "Training completed successfully...",
  "stderr": "",
  "metrics": {
    "mAP50": 0.95,
    "mAP50-95": 0.73,
    "precision": 0.89,
    "recall": 0.84,
    "training_time": 1200.5
  },
  "artifacts": [
    "runs/train/exp/weights/best.pt",
    "runs/train/exp/weights/last.pt",
    "runs/train/exp/results.csv"
  ],
  "success": true,
  "timestamp": "2025-07-12T10:30:00Z"
}
```

### 🚀 Example Operations

#### 1. Training a Custom Model

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n.pt",
    "data": "coco128.yaml",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "device": "0",
    "extra_args": {
      "patience": 10,
      "save_period": 5,
      "cos_lr": true
    }
  }'
```

#### 2. Real-time Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n.pt",
    "source": "path/to/image.jpg",
    "conf": 0.25,
    "iou": 0.7,
    "save": true,
    "save_txt": true,
    "save_conf": true
  }'
```

#### 3. Model Export for Deployment

```bash
curl -X POST "http://localhost:8000/export" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "runs/train/exp/weights/best.pt",
    "format": "onnx",
    "dynamic": true,
    "simplify": true,
    "opset": 11
  }'
```

#### 4. Video Object Tracking

```bash
curl -X POST "http://localhost:8000/track" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n.pt",
    "source": "path/to/video.mp4",
    "tracker": "bytetrack.yaml",
    "conf": 0.3,
    "save": true
  }'
```

### 📊 Common Parameters Reference

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `model` | string | required | Model path or name | `"yolov8n.pt"` |
| `data` | string | - | Dataset YAML path | `"coco128.yaml"` |
| `source` | string | - | Input source | `"image.jpg"`, `"video.mp4"`, `"0"` (webcam) |
| `epochs` | integer | 100 | Training epochs | `50` |
| `imgsz` | integer | 640 | Image size | `320`, `640`, `1280` |
| `device` | string | `"cpu"` | Compute device | `"cpu"`, `"0"`, `"0,1"` |
| `conf` | float | 0.25 | Confidence threshold | `0.1` to `1.0` |
| `iou` | float | 0.7 | IoU threshold for NMS | `0.1` to `1.0` |
| `batch` | integer | 16 | Batch size | `1`, `8`, `32` |
| `save` | boolean | false | Save results | `true`, `false` |
| `extra_args` | object | `{}` | Additional YOLO args | `{"patience": 10}` |

## 🧪 Testing & Quality Assurance

### 🔬 Comprehensive Test Suite

Our testing infrastructure ensures reliability across all YOLO operations:

```bash
# Run all tests with conda environment
conda activate ultra-dev
pytest tests/ -v

# Run specific test categories
pytest tests/test_flow.py -v                # Core workflow tests
pytest tests/test_mcp_train.py -v           # Training specific tests
pytest tests/test_mcp_predict.py -v         # Prediction tests
pytest tests/test_mcp_export.py -v          # Export functionality tests

# Generate coverage report
pytest tests/ --cov=app --cov-report=html
```

### 📊 Test Coverage

| Component | Tests | Coverage | Description |
|-----------|-------|----------|-------------|
| **Core Flow** | 9 tests | 95%+ | Complete train→validate→predict workflow |
| **Training** | 5 tests | 98% | Model training with various configurations |
| **Prediction** | 4 tests | 97% | Inference on images, videos, webcam |
| **Export** | 3 tests | 95% | Model format conversion (ONNX, TensorRT) |
| **Tracking** | 3 tests | 92% | Object tracking in video streams |
| **Benchmark** | 2 tests | 90% | Performance testing and profiling |

### 🚦 CI/CD Pipeline

```yaml
# Automated testing on every commit
Workflow: Test Suite
├── Environment Setup (Conda + PyTorch CPU)
├── Dependency Installation
├── Linting & Code Quality (flake8, black)
├── Unit Tests (pytest)
├── Integration Tests
├── Security Scanning (bandit)
├── Docker Build & Test
└── Documentation Validation
```

### 🔍 Example Test Run

```bash
$ pytest tests/test_flow.py::test_complete_workflow -v

tests/test_flow.py::test_complete_workflow PASSED [100%]

======================== Test Results ========================
✅ Train: Model trained successfully (epochs: 2)
✅ Validate: mAP50 = 0.847, mAP50-95 = 0.621
✅ Predict: 3 objects detected with confidence > 0.5
✅ Export: ONNX model exported (size: 12.4MB)
✅ Cleanup: Temporary files removed

Duration: 45.2s | Memory: 2.1GB | CPU: Intel i7
=================== 1 passed in 45.23s ===================
```

See [`tests/README.md`](tests/README.md) for detailed test documentation.

## 🚀 n8n Integration

### Streaming in n8n  
1. Drag **MCP Client Tool** → *SSE Endpoint* `http://host.docker.internal:8092/sse/train` (or `/sse/predict`).  
2. *OpenAPI URL* `http://host.docker.internal:8092/openapi.json`.  
3. Timeout 0, Auth None.  
4. Run workflow → live epoch/loss lines appear in execution log (see GIF).  

**Available SSE Endpoints:**
- `/sse/train` - Real-time training progress with epoch updates
- `/sse/predict` - Live prediction results 
- `/sse/val` - Validation metrics streaming
- `/sse/export` - Export progress updates
- `/sse/track` - Object tracking stream
- `/sse/benchmark` - Performance testing results
- `/sse/solution` - Solution execution logs

**Example SSE Training Stream:**
```bash
curl -N "http://localhost:8092/sse/train?data=coco128.yaml&epochs=1&device=cpu"
# Output:
# data: Ultralytics YOLOv8.0.196 🚀 Python-3.11.5 torch-2.1.1
# data: Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
# data:   1/1      0.12G      1.325      2.009      1.268         89        640: 100%|██████████| 8/8
# data: [COMPLETED] Process finished successfully
```

For detailed integration examples, see [`tools/UltralyticsMCPTool/README.md`](tools/UltralyticsMCPTool/README.md).

### 1. Environment Setup

Add the Ultralytics API URL to your n8n environment:

```bash
# In your n8n environment
export ULTRA_API_URL=http://localhost:8000

# Or in Docker Compose
environment:
  - ULTRA_API_URL=http://ultralytics-api:8000
```

### 2. Install UltralyticsMCPTool

```bash
# Navigate to the tool directory
cd tools/UltralyticsMCPTool

# Install dependencies
npm install

# Build the tool
npm run build

# Link for global usage
npm link
```

### 3. n8n Node Configuration

Create a custom n8n node or use the HTTP Request node:

```javascript
// n8n Custom Node Example
import UltralyticsMCPTool from 'ultralytics-mcp-tool';

const tool = new UltralyticsMCPTool(process.env.ULTRA_API_URL);

// Train a model
const result = await tool.train({
  model: 'yolov8n.pt',\n  data: 'coco128.yaml',\n  epochs: 10
});
```

### 4. Workflow Examples

**Image Classification Workflow:**
1. **Trigger**: Webhook receives image
2. **Ultralytics**: Predict objects
3. **Logic**: Process results
4. **Output**: Send notifications

**Training Pipeline:**
1. **Schedule**: Daily trigger
2. **Ultralytics**: Train model
3. **Validate**: Check performance
4. **Deploy**: Update production model

### 5. MCP Integration

```typescript
// Get available tools
const manifest = UltralyticsMCPTool.manifest();
console.log('Available operations:', manifest.tools.map(t => t.name));

// Execute with different channels
const httpResult = await tool.execute('predict', params, 'http');
const stdioResult = await tool.execute('predict', params, 'stdio');

// Real-time updates with SSE
tool.trainSSE(params, {
  onProgress: (data) => updateWorkflowStatus(data),
  onComplete: (result) => triggerNextNode(result)
});
```

For detailed integration examples, see [`tools/UltralyticsMCPTool/README.md`](tools/UltralyticsMCPTool/README.md).

## 🐳 Docker Deployment

### 🚀 Quick Docker Setup

```bash
# Clone and build
git clone https://github.com/your-username/ultralytics-mcp-server.git
cd ultralytics-mcp-server

# Build and run with Docker Compose
docker-compose up -d

# Verify deployment
curl http://localhost:8000/docs
```

### 📁 Docker Configuration

**Production-ready setup:**

```dockerfile
# Dockerfile highlights
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY models/ ./models/

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose services:**

```yaml
# docker-compose.yml
version: '3.8'
services:
  ultralytics-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./runs:/app/runs
    environment:
      - YOLO_CACHE_DIR=/app/cache
      - YOLO_SETTINGS_DIR=/app/settings
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ultralytics-api
```

### 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_CACHE_DIR` | `/tmp/yolo` | Model cache directory |
| `YOLO_SETTINGS_DIR` | `/tmp/settings` | Settings directory |
| `API_HOST` | `0.0.0.0` | API host binding |
| `API_PORT` | `8000` | API port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_WORKERS` | `4` | Uvicorn workers |
| `MODEL_DIR` | `/app/models` | Model storage path |

### 🌐 Production Deployment

```bash
# Production deployment with SSL
docker-compose -f docker-compose.prod.yml up -d

# Health check
curl -f http://localhost:8000/health || exit 1

# Scale services
docker-compose up -d --scale ultralytics-api=3

# Monitor logs
docker-compose logs -f ultralytics-api
```

## 🔧 API Documentation

### Response Format

All endpoints return a standardized response:

```json
{
  "run_id": "uuid-string",
  "command": "yolo train model=yolov8n.pt...",
  "return_code": 0,
  "stdout": "command output",
  "stderr": "error output",
  "metrics": {
    "mAP50": 0.95,
    "precision": 0.89,
    "training_time": 1200
  },
  "artifacts": [
    "runs/train/exp/weights/best.pt",
    "runs/train/exp/results.csv"
  ],
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Handling

```json
{
  "error": "Validation Error",
  "details": "Model file not found: invalid_model.pt",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 🛡️ Security & Authentication

```bash
# API Key authentication
curl -H "X-API-Key: your-api-key-here" \
     -X POST "http://localhost:8000/predict" \
     -d '{"model": "yolov8n.pt", "source": "image.jpg"}'

# JWT Token authentication  
curl -H "Authorization: Bearer your-jwt-token" \
     -X POST "http://localhost:8000/train" \
     -d '{"model": "yolov8n.pt", "data": "dataset.yaml"}'
```

### 📊 Health & Monitoring

```bash
# Health check endpoint
curl http://localhost:8000/health
# Response: {"status": "healthy", "version": "1.0.0", "uptime": 3600}

# Metrics endpoint
curl http://localhost:8000/metrics
# Response: Prometheus-formatted metrics

# Status endpoint with system info
curl http://localhost:8000/status
# Response: {"gpu": "available", "memory": "8GB", "models_loaded": 3}
```

## 🤝 Contributing Guidelines

We welcome contributions! Please follow these guidelines:

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ultralytics-mcp-server.git
   cd ultralytics-mcp-server
   ```

2. **Create Environment**
   ```bash
   conda env create -f environment.yml
   conda activate ultra-dev
   ```

3. **Install Development Tools**
   ```bash
   pip install black isort flake8 mypy pytest-cov
   ```

### Code Standards

- **Python**: Follow PEP 8, use Black for formatting
- **TypeScript**: Use ESLint and Prettier
- **Documentation**: Update README.md and docstrings
- **Tests**: Maintain 80%+ test coverage

### Pre-commit Checks

```bash
# Format code
black app/ tests/
isort app/ tests/

# Lint code
flake8 app/ tests/
mypy app/

# Run tests
pytest --cov=app
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following standards
   - Add/update tests
   - Update documentation

3. **Test Changes**
   ```bash
   pytest -v
   python run_tests.py
   ```

4. **Submit PR**
   - Clear description of changes
   - Reference related issues
   - Ensure CI passes

### Issue Reporting

When reporting issues, include:
- **Environment**: OS, Python version, dependencies
- **Steps**: Minimal reproduction steps
- **Expected**: What should happen
- **Actual**: What actually happens
- **Logs**: Error messages and stack traces

### Feature Requests

For new features:
- **Use Case**: Why is this needed?
- **Proposal**: How should it work?
- **Impact**: Who benefits from this?
- **Implementation**: Any technical considerations?

## 📄 License & Support

### 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Key permissions:**
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

### 🆘 Getting Help

| Resource | Link | Purpose |
|----------|------|---------|
| **📚 API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **🐛 Issues** | [GitHub Issues](https://github.com/your-org/ultralytics-mcp-server/issues) | Bug reports & feature requests |
| **💬 Discussions** | [GitHub Discussions](https://github.com/your-org/ultralytics-mcp-server/discussions) | Questions & community chat |
| **📖 Ultralytics** | [Official Docs](https://docs.ultralytics.com) | YOLO model documentation |
| **🔧 MCP Protocol** | [Specification](https://modelcontextprotocol.io) | MCP standard reference |

### 🎯 Quick Support Checklist

**Before asking for help:**
1. Check the [FAQ](docs/FAQ.md) for common issues
2. Search existing [GitHub Issues](https://github.com/your-org/ultralytics-mcp-server/issues)
3. Test with the latest version
4. Include environment details in your issue

**When reporting bugs:**
```bash
# Include this information
OS: Windows 11 / macOS 14 / Ubuntu 22.04
Python: 3.11.x
Conda env: ultra-dev
PyTorch: 2.5.1+cpu
Error: [paste complete error message]
```

### 🙏 Acknowledgments

| Component | Thanks To | For |
|-----------|-----------|-----|
| **🎯 YOLO Models** | [Ultralytics](https://ultralytics.com) | Revolutionary object detection |
| **🚀 FastAPI** | [Sebastian Ramirez](https://github.com/tiangolo) | Lightning-fast API framework |
| **🔧 Pydantic** | [Samuel Colvin](https://github.com/samuelcolvin) | Data validation & settings |
| **🐳 Docker** | [Docker Inc](https://docker.com) | Containerization platform |
| **🧪 pytest** | [pytest-dev](https://pytest.org) | Testing framework |
| **🌐 Conda** | [Anaconda](https://anaconda.com) | Package management |

---

<div align="center">

### 🌟 **Built with ❤️ for the Computer Vision Community** 🌟

**[⭐ Star this repo](https://github.com/your-org/ultralytics-mcp-server)** | **[🍴 Fork & contribute](https://github.com/your-org/ultralytics-mcp-server/fork)** | **[📢 Share with friends](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20Ultralytics%20MCP%20Server!)**

*Empowering developers to build intelligent computer vision applications with ease* 🚀

</div>
