# 🦷 DENTEX AI Platform - Ultralytics MCP Server

> **Advanced AI-powered dental X-ray detection platform with intelligent dataset upload, custom model training, and MCP integration for N8N automation.**

A comprehensive Model Context Protocol (MCP) server that seamlessly integrates Ultralytics YOLO models with N8N workflows, providing a complete AI-powered computer vision solution with 10GB dataset upload support and intelligent background processing.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4.1-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![N8N](https://img.shields.io/badge/N8N-Integration-orange.svg)](https://n8n.io/)

## ✨ Key Features

### 🎯 Core Capabilities
- **🔬 Advanced AI Detection**: YOLO-based dental X-ray analysis
- **📦 Smart Dataset Upload**: 10GB limit with intelligent ZIP structure detection  
- **🎯 Custom Model Training**: Train your own models with any YOLO dataset
- **🤖 YOLO11 Model Variants**: Choose from nano/small/medium/large/x-large base models
- **⚡ GPU Acceleration**: NVIDIA CUDA support for fast training/inference
- **🌐 Web Interface**: Beautiful Streamlit dashboard
- **📊 Real-time Monitoring**: Live GPU stats and training progress
- **🔌 MCP Integration**: Connect with N8N for workflow automation
- **🛡️ Background Processing**: Stable upload handling for large files

## � Quick Start

### One-Command Setup

**For Windows users:**
```bash
setup.bat
```

**For Linux/Mac users:**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual setup:**
```bash
docker-compose up --build -d
```

### Access the Platform

- **🌐 Main Interface**: http://localhost:8501
- **📊 TensorBoard**: http://localhost:6006  
- **🔌 MCP Server**: http://localhost:8092
- **📓 Jupyter**: http://localhost:8888

## � Requirements

- **Docker & Docker Compose**
- **NVIDIA Docker Runtime** (for GPU support)
- **8GB+ RAM** recommended
- **50GB+ free disk space**

## 🎯 Dataset Upload

### Supported ZIP Structures

The platform automatically detects and organizes various ZIP structures:

```
✅ Structure 1 (Flat):
dataset.zip
├── data.yaml
├── images/
│   ├── img1.jpg
│   └── img2.jpg
└── labels/
    ├── img1.txt
    └── img2.txt

✅ Structure 2 (Nested):
dataset.zip
└── my_dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

✅ Structure 3 (Split folders):
dataset.zip
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### Upload Process

1. Navigate to **Training** page
2. Click **Upload Custom Dataset**
3. Select your ZIP file (up to 10GB)
4. Enter dataset name
5. Click **Upload Dataset**
6. **Do NOT refresh** during processing
7. Wait for completion message

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
3. Select YOLO model variant and confidence threshold
4. Run inference and view annotated results

### Training Custom Models with YOLO11 Variants
1. Go to **Training** page in Streamlit
2. Upload custom dataset or select DENTEX built-in datasets
3. Choose **YOLO11 Model Variant**:
   - **yolo11n**: Fast training, good for testing (1.9M parameters)
   - **yolo11s**: Balanced performance (9.1M parameters)
   - **yolo11m**: Better accuracy (20.1M parameters) 
   - **yolo11l**: High accuracy training (25.3M parameters)
   - **yolo11x**: Maximum accuracy (43.9M parameters)
4. Configure epochs, batch size, image size
5. Monitor real-time training progress with live GPU stats
6. Models automatically save to workspace

### Training Custom Models
1. Access Training page in Streamlit interface
2. Select **YOLO11 Model Variant** (nano/small/medium/large/x-large)
3. Choose your dataset (DENTEX built-in or custom upload)
4. Configure training parameters (epochs, batch size, image size)
5. Click **Start Training** and monitor progress
6. Models auto-save to workspace for later use

**Model Variant Selection:**
- **yolo11n.pt** - Nano: Fastest, lowest accuracy (1.9M params)
- **yolo11s.pt** - Small: Good balance (9.1M params) 
- **yolo11m.pt** - Medium: Better accuracy (20.1M params)
- **yolo11l.pt** - Large: High accuracy (25.3M params)
- **yolo11x.pt** - X-Large: Highest accuracy (43.9M params)

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
