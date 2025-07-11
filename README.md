# Ultralytics MCP Server

[![Build Status](https://github.com/your-org/ultralytics-mcp-server/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/your-org/ultralytics-mcp-server/actions)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io%2Fyour--org%2Fultra--api-blue)](https://github.com/your-org/ultralytics-mcp-server/pkgs/container/ultra-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF.svg)](https://ultralytics.com)

A Model Context Protocol (MCP) compliant server that provides RESTful API access to Ultralytics YOLO operations for computer vision tasks including training, validation, prediction, export, tracking, and benchmarking.

## Project Overview

The Ultralytics MCP Server bridges the gap between Ultralytics' powerful YOLO models and modern application architectures by providing:

- **🌐 RESTful API**: HTTP endpoints for all YOLO operations
- **📡 Real-time Updates**: Server-Sent Events (SSE) for long-running operations
- **🔧 MCP Compliance**: Full Model Context Protocol support
- **🐳 Docker Ready**: Production-ready containerization
- **🧪 Test Coverage**: Comprehensive test suite with CI/CD
- **📊 Monitoring**: Built-in metrics and health checks
- **🔒 Security**: API key authentication and input validation

### Features

- **Train** YOLO models on custom datasets
- **Validate** model performance with comprehensive metrics
- **Predict** on images, videos, and streams
- **Export** models to multiple formats (ONNX, TensorRT, CoreML, etc.)
- **Track** objects in videos with various tracking algorithms
- **Benchmark** model performance across different configurations
- **Solutions** access to Ultralytics' specialized solutions

## Architecture

![Architecture Diagram](docs/architecture.png)

The server implements a layered architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Pydantic      │    │   Ultralytics   │
│   REST Endpoints│◄───│   Validation    │◄───│   CLI Engine    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SSE Events    │    │   Metrics       │    │   File System   │
│   Real-time     │    │   Parsing       │    │   Artifacts     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Key Components:**
- **app/main.py**: FastAPI application with route definitions
- **app/schemas.py**: Pydantic models for request/response validation
- **app/ultra.py**: Ultralytics CLI integration and metrics parsing
- **tools/UltralyticsMCPTool**: TypeScript MCP client library

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Conda/Miniconda**
- **Git**

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/ultralytics-mcp-server.git
cd ultralytics-mcp-server

# Create and activate conda environment
conda env create -f environment.yml
conda activate ultra-dev

# Verify installation
python -c "import ultralytics; print('✅ Ultralytics installed successfully')"
```

### 2. Start the Server

```bash
# Start the development server
conda run -n ultra-dev uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the environment directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Verify Installation

Open your browser and navigate to:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### 4. First API Call

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "yolov8n.pt",\n    "source": "https://ultralytics.com/images/bus.jpg",\n    "conf": 0.5\n  }'
```

## Running Tests

The project includes comprehensive tests covering unit, integration, and end-to-end scenarios.

### Quick Test Run

```bash
# Run all tests with minimal output
pytest -q

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_flow.py -v

# Run with coverage
pytest --cov=app --cov-report=html
```

### Test Structure

- **tests/test_flow.py**: Complete workflow integration tests
- **tests/conftest.py**: Pytest configuration and fixtures
- **run_tests.py**: Convenient test runner script

### Test Workflow

The integration tests perform a complete YOLO workflow:

1. **🏥 Health Check** - API availability
2. **🏋️ Training** - 1 epoch on COCO128 dataset
3. **🔍 Validation** - Model performance metrics
4. **🎯 Prediction** - Inference on test images
5. **📊 Results Verification** - Output file validation

```bash
# Run the complete workflow test
python run_tests.py

# Quick tests only (skip training)
python run_tests.py quick
```

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

## Ultralytics CLI Cheat Sheet

| Operation | Endpoint | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| **train** | `POST /train` | Train YOLO model | `model`, `data`, `epochs`, `imgsz`, `device` |
| **val** | `POST /val` | Validate model | `model`, `data`, `batch`, `conf`, `iou` |
| **predict** | `POST /predict` | Run inference | `model`, `source`, `conf`, `save` |
| **export** | `POST /export` | Export model | `model`, `format`, `dynamic`, `simplify` |
| **track** | `POST /track` | Object tracking | `model`, `source`, `tracker`, `conf` |
| **benchmark** | `POST /benchmark` | Performance test | `model`, `data`, `imgsz`, `device` |
| **solution** | `POST /solution` | Specialized solutions | `task`, `model`, `source`, `solution_type` |

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | - | Model path (e.g., `yolov8n.pt`) |
| `data` | string | - | Dataset YAML path |
| `source` | string | - | Image/video source path |
| `epochs` | integer | 100 | Training epochs |
| `imgsz` | integer | 640 | Image size |
| `device` | string | `cpu` | Device (`cpu`, `0`, `1`, etc.) |
| `conf` | float | 0.25 | Confidence threshold |
| `iou` | float | 0.7 | IoU threshold for NMS |
| `batch` | integer | 16 | Batch size |
| `extra_args` | object | {} | Additional arguments |

### Example API Calls

```bash
# Training
curl -X POST "http://localhost:8000/train" \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "yolov8n.pt",\n    "data": "coco128.yaml",\n    "epochs": 10,\n    "device": "cpu"\n  }'

# Prediction
curl -X POST "http://localhost:8000/predict" \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "yolov8n.pt",\n    "source": "path/to/image.jpg",\n    "conf": 0.5\n  }'

# Export
curl -X POST "http://localhost:8000/export" \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "yolov8n.pt",\n    "format": "onnx",\n    "dynamic": true\n  }'
```

## n8n Integration

Integrate Ultralytics operations into your n8n workflows using the MCP tool.

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

## API Documentation

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

### Authentication

```bash
# Set API key in request headers
curl -H "X-API-Key: your-api-key-here" http://localhost:8000/predict
```

## Contributing Guidelines

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: http://localhost:8000/docs
- **Issues**: [GitHub Issues](https://github.com/your-org/ultralytics-mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ultralytics-mcp-server/discussions)
- **Ultralytics**: [Official Documentation](https://docs.ultralytics.com)

## Acknowledgments

- **Ultralytics**: For the amazing YOLO models and CLI
- **FastAPI**: For the excellent web framework
- **Pydantic**: For data validation and settings
- **Contributors**: All the amazing people who contribute to this project

---

**Built with ❤️ for the Computer Vision Community**
