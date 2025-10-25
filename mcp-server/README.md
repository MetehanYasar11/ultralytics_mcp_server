# Ultralytics Training MCP Server

🤖 **AI Agent-Controlled YOLO Training System**

MCP (Model Context Protocol) server that enables AI agents like GitHub Copilot to automatically manage and control YOLO training operations in the Ultralytics Docker container.

## 🎯 Features

### 🚀 Training Automation
- **start_training** - Start new training jobs with custom parameters
- **get_training_status** - Monitor active training progress
- **stop_training** - Stop running training jobs
- **get_training_logs** - View training logs in real-time

### 📊 Dataset Management
- **list_datasets** - List built-in and custom datasets
- Built-in datasets: YOLO_Disease, YOLO_Tooth, YOLO_Quadrant, YOLO_Full_Hierarchy
- Custom dataset support

### 💾 Model Management
- **list_models** - View trained models in workspace
- **list_training_history** - Browse past training runs with metrics
- **convert_to_tensorboard** - Convert historical trainings to TensorBoard format

### 🖥️ System Monitoring
- **get_gpu_status** - Real-time GPU utilization and memory stats

## 📋 Installation

```bash
cd mcp-server
npm install
```

## 🔧 Configuration

The server is configured in VS Code's `mcp.json`:

```json
{
  "servers": {
    "ultralytics-training": {
      "command": "node",
      "args": ["path/to/mcp-server/index.js"],
      "type": "stdio",
      "description": "Ultralytics YOLO Training Automation"
    }
  }
}
```

## 🤖 Usage with AI Agents

Once configured, you can ask GitHub Copilot to:

### Example Commands:

**Start Training:**
```
"Start a YOLOv11x training with kaggle dataset for 100 epochs"
```

**Check Status:**
```
"What's the current training status?"
```

**List Datasets:**
```
"Show me all available datasets"
```

**Monitor GPU:**
```
"What's the GPU utilization?"
```

**View Models:**
```
"List all trained models in workspace"
```

**Training History:**
```
"Show me the last 5 training runs with their metrics"
```

## 🛠️ Tools Reference

### start_training

**Parameters:**
- `model_variant` (required): "yolo11n" | "yolo11s" | "yolo11m" | "yolo11l" | "yolo11x"
- `dataset_name` (required): Dataset name (e.g., "YOLO_Disease", "kaggle")
- `epochs` (required): Number of epochs (1-1000)
- `batch_size` (optional): 4, 8, 16, 32, 64, 128, 256 (default: 16)
- `img_size` (optional): 320-1280 (default: 640)
- `device` (optional): "0" (GPU) or "cpu" (default: "0")
- `model_name` (optional): Custom model name
- `transfer_learning` (optional): Use transfer learning (default: false)
- `base_model_path` (optional): Path to base model for transfer learning

**Returns:**
```json
{
  "success": true,
  "message": "Training started successfully",
  "config": {
    "model": "yolo11x",
    "dataset": "kaggle",
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "device": "0",
    "model_name": "model_kaggle"
  }
}
```

### get_training_status

**Returns:**
```json
{
  "is_training": true,
  "latest_training": "training_111703",
  "epochs_completed": 45,
  "current_metrics": {
    "epoch": "45",
    "box_loss": "0.823",
    "cls_loss": "0.654",
    "mAP50": "0.876",
    "mAP50_95": "0.642"
  }
}
```

### list_datasets

**Parameters:**
- `type` (optional): "all" | "builtin" | "custom" (default: "all")

**Returns:**
```json
{
  "builtin": [
    {
      "name": "YOLO_Disease",
      "path": "/ultralytics/YOLO_MultiLevel_Datasets/YOLO_Disease/data.yaml",
      "has_yaml": true
    }
  ],
  "custom": [
    {
      "name": "kaggle",
      "path": "/ultralytics/custom_datasets/kaggle/data.yaml",
      "has_yaml": true
    }
  ]
}
```

### list_models

**Returns:**
```json
{
  "models": [
    {
      "name": "model_kaggle_best.pt",
      "path": "/workspace/trained_models/model_kaggle_best.pt",
      "size_mb": "109.33",
      "modified": "2025-10-23T14:17:02.000Z"
    }
  ],
  "count": 1
}
```

### get_gpu_status

**Returns:**
```json
{
  "available": true,
  "name": "NVIDIA GeForce RTX 4070",
  "temperature": 65,
  "gpu_utilization": 95,
  "memory_utilization": 87,
  "memory_used_mb": 10240,
  "memory_total_mb": 12288,
  "power_draw_w": 185.5,
  "power_limit_w": 200.0
}
```

## 🔍 How It Works

1. **AI Agent Request** → GitHub Copilot receives user request
2. **MCP Tool Call** → Agent calls appropriate tool via MCP
3. **Docker Execution** → MCP server executes commands in Docker container
4. **Response** → Results returned to agent in structured format
5. **AI Response** → Agent formats response for user

## 🐳 Docker Integration

The server communicates with the `ultralytics-container` Docker container:

```bash
docker exec ultralytics-container <command>
```

All training operations run inside the container with GPU access.

## 📊 TensorBoard Integration

Convert historical trainings to TensorBoard format:

```
"Convert all trainings to TensorBoard format"
```

Then start TensorBoard:

```bash
docker exec -it ultralytics-container bash
tensorboard --logdir=/ultralytics/runs/detect/tensorboard_logs
```

Access at: http://localhost:6006

## 🎯 Use Cases

### Automated Training Pipeline
```
"Train YOLOv11m on YOLO_Disease dataset for 50 epochs, 
then train YOLOv11l on YOLO_Tooth for 100 epochs"
```

### Hyperparameter Optimization
```
"Train YOLOv11s on kaggle dataset with different batch sizes 
(8, 16, 32) for 20 epochs each and compare results"
```

### Transfer Learning Experiments
```
"Use the best model from workspace and train it on 
YOLO_Quadrant dataset for 30 more epochs"
```

### Performance Monitoring
```
"Check GPU status every 5 minutes and alert if temperature > 80°C"
```

## 📝 Logs

Training logs are saved to:
- Container: `/tmp/training_log.txt`
- Access via: `get_training_logs` tool

## 🔒 Security

- Server runs locally only (stdio transport)
- No network exposure
- Docker container isolation
- Direct container access required

## 🤝 Contributing

This MCP server is part of the RCT Detector Platform project.

## 📄 License

MIT

---

**Author:** Metehan Yasar  
**Version:** 1.0.0  
**Last Updated:** October 25, 2025
