# 🚀 Unified AI Development Platform

## 📊 Service Overview

Bu platform tek komutla tüm AI geliştirme ihtiyaçlarınızı karşılar:

### 🌐 Web Interfaces

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Streamlit UI** | 8501 | http://localhost:8501 | Ana AI Arayüzü (YOLO, Custom Models) |
| **OpenWebUI** | 8080 | http://localhost:8080 | AI Chat Interface (LLM Integration) |
| **Jupyter Lab** | 8888 | http://localhost:8888 | Development Environment |
| **TensorBoard** | 6006 | http://localhost:6006 | Training Metrics & Visualization |
| **Custom Apps** | 8502 | http://localhost:8502 | Geliştirmeleriniz için |

### 🔗 API Endpoints

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **MCP Server** | 8092 | http://localhost:8092 | N8N Integration (7 AI Tools) |
| **Redis Cache** | 6379 | localhost:6379 | Caching & Sessions |

## 🚀 Quick Start

### Tek Komut - Tüm Servisler
```bash
docker-compose up -d
```

### Logları İzle
```bash
docker-compose logs -f
```

### Servisleri Durdur
```bash
docker-compose down
```

## 📁 Project Structure

```
├── ultralytics/          # Ana AI kodları
├── workspace/            # Model training workspace
├── dev/                  # Geliştirmeleriniz
├── datasets/             # Custom datasets
├── openwebui_data/       # OpenWebUI data
├── openwebui_docs/       # OpenWebUI documents
└── src/                  # MCP Server (N8N entegrasyonu)
```

## 🛠️ Development Workflow

### 1. Arayüz Geliştirme
- **Klasör**: `./dev/`
- **Port**: 8502 (Custom Streamlit apps)
- Real-time editing desteklenir

### 2. OpenWebUI Geliştirme
- **Interface**: http://localhost:8080
- **Data**: `./openwebui_data/`
- **Docs**: `./openwebui_docs/`

### 3. N8N Yetenekleri
- **MCP Endpoint**: http://localhost:8092/sse
- **Tools**: 7 AI tools available
- **Health**: http://localhost:8092/health

## 🎯 Available AI Tools (MCP)

1. `execute_python_code` - Python execution
2. `yolo_operation` - YOLO operations  
3. `list_training_results` - Browse results
4. `analyze_training_results` - Training metrics
5. `view_tensorboard` - TensorBoard access
6. `launch_streamlit_interface` - UI launcher
7. `get_system_info` - System overview

## 💡 Usage Examples

### Development Mode
```bash
# Start all services
docker-compose up -d

# Access your interfaces
# - Streamlit: http://localhost:8501
# - OpenWebUI: http://localhost:8080  
# - Jupyter: http://localhost:8888
```

### N8N Integration
```javascript
// N8N MCP Node Configuration
{
  "endpoint": "http://localhost:8092/sse",
  "tools": ["execute_python_code", "yolo_operation"]
}
```

## 🔧 Customization

- **Custom Streamlit Apps**: Place in `./dev/` folder
- **Datasets**: Add to `./datasets/` folder
- **OpenWebUI Docs**: Add to `./openwebui_docs/` folder

---

**🎯 Tek komutla başlayın: `docker-compose up -d`**
