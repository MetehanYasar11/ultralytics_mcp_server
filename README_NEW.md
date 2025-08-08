# 🚀 Ultralytics MCP Server - AI-Powered Computer Vision Platform

> **Unified Development Platform for YOLO Models with N8N Integration**

A comprehensive Model Context Protocol (MCP) server that seamlessly integrates Ultralytics YOLO models with N8N workflows, providing a complete AI-powered computer vision solution in a single command.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4.1-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![N8N](https://img.shields.io/badge/N8N-Integration-orange.svg)](https://n8n.io/)

## ✨ Features

### 🎯 Core Capabilities
- **7 AI-Powered Tools** for comprehensive YOLO operations
- **Real-time Object Detection** with live inference
- **Model Training & Fine-tuning** with custom datasets
- **Performance Analytics** via TensorBoard integration
- **N8N Workflow Integration** for automation

### 🖥️ User Interfaces
- **Streamlit Dashboard** - Interactive web interface for model management
- **Jupyter Lab** - Notebook environment for development
- **TensorBoard** - Real-time training metrics and visualization
- **N8N Integration** - Workflow automation and AI task orchestration

### 🔧 Technical Stack
- **CUDA 12.4.1** - GPU acceleration for training and inference
- **PyTorch** - Deep learning framework with CUDA support
- **Ultralytics YOLO** - State-of-the-art object detection models
- **Docker** - Containerized deployment
- **Node.js MCP Server** - Model Context Protocol implementation

## 🚀 Quick Start

### Prerequisites
- Docker Desktop with GPU support
- NVIDIA drivers compatible with CUDA 12.4.1
- Windows PowerShell or Linux/macOS terminal

### One-Command Deployment

```bash
docker-compose up -d
```

That's it! The entire platform will be available at:

- 🌐 **Streamlit UI**: http://localhost:8501
- 📊 **TensorBoard**: http://localhost:6006  
- 📓 **Jupyter Lab**: http://localhost:8888
- 🔗 **MCP Server**: http://localhost:8092

## 🎮 Available Services

| Service | Port | Description | Status |
|---------|------|-------------|--------|
| Streamlit Dashboard | 8501 | Interactive YOLO model interface | ✅ Ready |
| MCP Server | 8092 | N8N integration endpoint | ✅ Ready |
| TensorBoard | 6006 | Training metrics visualization | ✅ Ready |
| Jupyter Lab | 8888 | Development environment | ✅ Ready |

## 🛠️ MCP Tools Available

Our MCP server provides 7 specialized tools for AI workflows:

1. **`detect_objects`** - Real-time object detection in images
2. **`train_model`** - Custom YOLO model training
3. **`evaluate_model`** - Model performance assessment
4. **`predict_batch`** - Batch processing for multiple images
5. **`export_model`** - Model format conversion (ONNX, TensorRT, etc.)
6. **`benchmark_model`** - Performance benchmarking
7. **`analyze_dataset`** - Dataset statistics and validation

## 🔌 N8N Integration

Connect to N8N using our MCP server:

1. **Server Endpoint**: `http://localhost:8092`
2. **Transport**: Server-Sent Events (SSE)
3. **Health Check**: `http://localhost:8092/health`

### Example N8N Workflow
```json
{
  "mcp_connection": {
    "transport": "sse",
    "endpoint": "http://localhost:8092/sse"
  }
}
```

## 📁 Project Structure

```
ultralytics_mcp_server/
├── 🐳 docker-compose.yml          # Orchestration configuration
├── 🔧 Dockerfile.ultralytics      # CUDA-enabled Ultralytics container
├── 🔧 Dockerfile.mcp-connector    # Node.js MCP server container
├── 📦 src/
│   └── server.js                  # MCP server implementation
├── 🎨 main_dashboard.py           # Streamlit main interface
├── 📄 pages/                      # Streamlit multi-page app
│   ├── train.py                   # Model training interface
│   └── inference.py               # Inference interface
├── ⚡ startup.sh                   # Container initialization script
├── 📋 .dockerignore               # Build optimization
└── 📖 README.md                   # This documentation
```

## 🔧 Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES` - GPU device selection
- `STREAMLIT_PORT` - Streamlit service port (default: 8501)
- `MCP_PORT` - MCP server port (default: 8092)
- `TENSORBOARD_PORT` - TensorBoard port (default: 6006)

### Custom Configuration
Edit `docker-compose.yml` to customize:
- Port mappings
- Volume mounts
- Environment variables
- Resource limits

## 📊 Usage Examples

### Object Detection via Streamlit
1. Navigate to http://localhost:8501
2. Upload an image or video
3. Select YOLO model (YOLOv8, YOLOv11)
4. Run inference and view results

### Training Custom Models
1. Access Jupyter Lab at http://localhost:8888
2. Prepare your dataset in YOLO format
3. Use the training interface in Streamlit
4. Monitor progress in TensorBoard

### N8N Automation
1. Create N8N workflow
2. Add MCP connector node
3. Configure endpoint: `http://localhost:8092`
4. Use available tools for automation

## 🔍 Monitoring & Debugging

### Container Status
```bash
docker ps
docker-compose logs ultralytics-container
docker-compose logs mcp-connector-container
```

### Health Checks
```bash
# MCP Server
curl http://localhost:8092/health

# Streamlit
curl http://localhost:8501/_stcore/health

# TensorBoard
curl http://localhost:6006
```

## 🔄 Restart & Maintenance

### Restart Services
```bash
docker-compose restart
```

### Update & Rebuild
```bash
docker-compose down
docker-compose up --build -d
```

### Clean Reset
```bash
docker-compose down
docker system prune -f
docker-compose up --build -d
```

## 🎯 Performance Optimization

- **GPU Memory**: Automatically managed by CUDA runtime
- **Batch Processing**: Optimized for multiple image inference
- **Model Caching**: Pre-loaded models for faster response
- **Multi-threading**: Concurrent request handling

## 🚨 Troubleshooting

### Common Issues

**Container Restart Loop**
```bash
# Check logs
docker-compose logs ultralytics-container

# Restart with rebuild
docker-compose down
docker-compose up --build -d
```

**Streamlit Not Loading**
```bash
# Verify container status
docker ps

# Check if files are copied correctly
docker exec ultralytics-container ls -la /ultralytics/
```

**GPU Not Detected**
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA in container
docker exec ultralytics-container nvidia-smi
```

## 🔧 Development

### Local Development Setup
1. Clone repository
2. Install dependencies: `npm install` (for MCP server)
3. Set up Python environment for Streamlit
4. Run services individually for debugging

### Adding New MCP Tools
1. Edit `src/server.js`
2. Add tool definition in `tools` array
3. Implement handler in `handleToolCall`
4. Test with N8N integration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the AGPL-3.0 License - see the [Ultralytics License](https://ultralytics.com/license) for details.

## 🙏 Acknowledgments

- **Ultralytics** - For the amazing YOLO implementation
- **N8N** - For the workflow automation platform
- **Streamlit** - For the beautiful web interface framework
- **NVIDIA** - For CUDA support and GPU acceleration

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/MetehanYasar11/ultralytics_mcp_server/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/MetehanYasar11/ultralytics_mcp_server/discussions)
- 📧 **Contact**: Create an issue for support

---

**Made with ❤️ for the AI Community**

> 🚀 **Ready to revolutionize your computer vision workflows? Start with `docker-compose up -d`!**
