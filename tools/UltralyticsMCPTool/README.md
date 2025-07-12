# 🎯 Ultralytics MCP Tool

> **A Model Context Protocol (MCP) compliant TypeScript tool for seamless Ultralytics YOLO operations**

[![npm version](https://badge.fury.io/js/ultralytics-mcp-tool.svg)](https://www.npmjs.com/package/ultralytics-mcp-tool)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg)](https://www.typescriptlang.org/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

🚀 **Multi-Channel Communication**
- **HTTP**: RESTful API calls with axios
- **SSE**: Server-Sent Events for real-time progress tracking  
- **Stdio**: Direct process communication for CLI integration

🎯 **Complete YOLO Operations**
- **Train**: Custom model training with real-time progress
- **Predict**: Object detection on images/videos/streams
- **Validate**: Model performance evaluation
- **Export**: Model format conversion (ONNX, TensorRT, etc.)
- **Track**: Object tracking in video sequences
- **Benchmark**: Performance testing and optimization

🔧 **Developer Experience**
- **TypeScript**: Full type safety and IntelliSense support
- **MCP Compliant**: Follows Model Context Protocol specification
- **Minimal Dependencies**: Only axios and eventsource required
- **Comprehensive Examples**: Ready-to-use code snippets

## 📦 Installation

```bash
# Install from npm
npm install ultralytics-mcp-tool

# Or with yarn
yarn add ultralytics-mcp-tool

# Development dependencies (optional)
npm install --save-dev @types/node @types/eventsource typescript
```

## 🚀 Quick Start

### 1. Basic Setup

```typescript
import UltralyticsMCPTool from 'ultralytics-mcp-tool';

// Initialize with your Ultralytics API endpoint
const tool = new UltralyticsMCPTool('http://localhost:8000');

// Train a model with minimal configuration
const result = await tool.train({
  model: 'yolov8n.pt',
  data: 'coco128.yaml',
  epochs: 10
});

console.log(`Training ${result.success ? 'completed' : 'failed'}`);
```

### 2. Real-time Training Progress

```typescript
// Subscribe to live training updates
const eventSource = tool.trainSSE({
  model: 'yolov8n.pt',
  data: 'coco128.yaml',
  epochs: 100,
  imgsz: 640
}, {
  onProgress: (data) => {
    console.log(`Epoch ${data.epoch}/${data.total_epochs}`);
    console.log(`Loss: ${data.loss}, mAP: ${data.map50}`);
  },
  onComplete: (result) => {
    console.log('🎉 Training completed!');
    console.log(`Best mAP: ${result.metrics.mAP50}`);
  },
  onError: (error) => console.error('❌ Training failed:', error)
});
```

### 3. Object Detection Pipeline

```typescript
// Complete detection workflow
async function detectObjects() {
  // 1. Run inference
  const prediction = await tool.predict({
    model: 'yolov8n.pt',
    source: 'path/to/images/',
    conf: 0.5,
    save: true,
    save_txt: true
  });
  
  // 2. Export optimized model
  if (prediction.success) {
    const exported = await tool.export({
      model: prediction.artifacts[0], // Use trained model
      format: 'onnx',
      dynamic: true,
      simplify: true
    });
    
    console.log(`Model exported: ${exported.artifacts[0]}`);
  }
}
```
## 📋 MCP Manifest & Discovery

### 🔍 Get Available Tools

```typescript
// Get complete MCP manifest
const manifest = UltralyticsMCPTool.manifest();

console.log('📦 Available operations:', manifest.tools.map(t => t.name));
// Output: ['train', 'val', 'predict', 'export', 'track', 'benchmark']

// Inspect specific tool
const trainTool = manifest.tools.find(t => t.name === 'train');
console.log('🎯 Training parameters:', trainTool.inputSchema.properties);
```

### 🌐 Communication Channels

```typescript
// 1. HTTP (Default) - Standard RESTful API
const httpResult = await tool.execute('predict', params, 'http');

// 2. SSE - Real-time streaming updates
const sseSource = tool.predictSSE(params, callbacks);

// 3. Stdio - Direct CLI communication
const stdioResult = await tool.executeStdio('train', params);

// 4. Dynamic channel selection
const result = await tool.execute('predict', params, 'auto'); // Choose best channel
```

## 🎯 Complete API Reference

### 🚂 Training Operations

```typescript
// Basic training
const training = await tool.train({
  model: 'yolov8n.pt',
  data: 'coco128.yaml',
  epochs: 50,
  imgsz: 640,
  batch: 16,
  device: 'cpu'
});

// Advanced training with custom parameters
const advancedTraining = await tool.train({
  model: 'yolov8n.pt',
  data: 'custom-dataset.yaml',
  epochs: 100,
  imgsz: 1280,
  batch: 8,
  device: '0',
  extra_args: {
    patience: 10,
    save_period: 5,
    cos_lr: true,
    augment: true,
    mixup: 0.1,
    copy_paste: 0.1
  }
});

// Real-time training with progress tracking
const trainingSSE = tool.trainSSE({
  model: 'yolov8s.pt',
  data: 'dataset.yaml',
  epochs: 200
}, {
  onProgress: (data) => updateProgressBar(data.epoch, data.total_epochs),
  onMetrics: (data) => updateMetricsDisplay(data.metrics),
  onComplete: (result) => showTrainingComplete(result),
  onError: (error) => handleTrainingError(error)
});
```

### 🔍 Validation & Testing

```typescript
// Model validation
const validation = await tool.val({
  model: 'runs/train/exp/weights/best.pt',
  data: 'coco128.yaml',
  split: 'val',
  imgsz: 640,
  conf: 0.001,
  iou: 0.6,
  device: 'cpu'
});

console.log(`Validation mAP50: ${validation.metrics.mAP50}`);
console.log(`Precision: ${validation.metrics.precision}`);
console.log(`Recall: ${validation.metrics.recall}`);
```

### 🎯 Prediction & Inference

```typescript
// Image prediction
const imagePrediction = await tool.predict({
  model: 'yolov8n.pt',
  source: 'path/to/images/',
  conf: 0.25,
  iou: 0.7,
  save: true,
  save_txt: true,
  save_conf: true,
  classes: [0, 1, 2], // Only detect persons, bicycles, cars
  agnostic_nms: true
});

// Video prediction with tracking
const videoPrediction = await tool.predict({
  model: 'yolov8n.pt',
  source: 'video.mp4',
  conf: 0.3,
  save: true,
  show_labels: false,
  show_conf: false,
  line_width: 2
});

// Real-time webcam prediction
const webcamPrediction = await tool.predict({
  model: 'yolov8n.pt',
  source: '0', // Webcam
  show: true,
  conf: 0.5
});
```

### 📤 Model Export & Deployment

```typescript
// Export to ONNX for cross-platform deployment
const onnxExport = await tool.export({
  model: 'runs/train/exp/weights/best.pt',
  format: 'onnx',
  dynamic: true,
  simplify: true,
  opset: 11,
  workspace: 4096
});

// Export to TensorRT for NVIDIA GPUs
const tensorRTExport = await tool.export({
  model: 'yolov8n.pt',
  format: 'engine',
  workspace: 4,
  int8: true,
  fp16: true,
  dynamic: true
});

// Export to CoreML for iOS/macOS
const coreMLExport = await tool.export({
  model: 'yolov8n.pt',
  format: 'coreml',
  nms: true,
  int8: true
});
```

### 📊 Object Tracking

```typescript
// Multi-object tracking in video
const tracking = await tool.track({
  model: 'yolov8n.pt',
  source: 'traffic.mp4',
  tracker: 'bytetrack.yaml',
  conf: 0.3,
  iou: 0.5,
  save: true,
  show_trajectories: true,
  save_txt: true
});

// Custom tracker configuration
const customTracking = await tool.track({
  model: 'yolov8n.pt',
  source: 'surveillance.mp4',
  tracker: 'botsort.yaml',
  conf: 0.25,
  persist: true,
  save_crop: true,
  line_width: 3
});
```

### ⚡ Performance Benchmarking

```typescript
// Comprehensive model benchmarking
const benchmark = await tool.benchmark({
  model: 'yolov8n.pt',
  data: 'coco128.yaml',
  imgsz: 640,
  half: false,
  device: 'cpu',
  verbose: true
});

// GPU benchmarking with different precisions
const gpuBenchmark = await tool.benchmark({
  model: 'yolov8s.pt',
  data: 'coco128.yaml',
  imgsz: [640, 1280],
  half: true,
  device: '0',
  batch: [1, 8, 16]
});
```

## 🔧 Response Format & Error Handling
  });
} catch (error) {
  console.error('Training failed:', error.message);
}
```

## Connection Testing

```typescript
const connectionStatus = await tool.testConnection();
console.log('HTTP available:', connectionStatus.http);
console.log('SSE available:', connectionStatus.sse);
console.log('Stdio available:', connectionStatus.stdio);
```

## TypeScript Support

The tool is written in TypeScript and provides full type definitions:

```typescript
import UltralyticsMCPTool, { 
  UltralyticsRequest, 
  UltralyticsResponse, 
  MCPManifest 
} from 'ultralytics-mcp-tool';
```

## Examples

### Complete Training Workflow

```typescript
async function trainAndValidate() {
  const tool = new UltralyticsMCPTool();
  
  // 1. Train model
  const trainResult = await tool.train({
    model: 'yolov8n.pt',
    data: 'coco128.yaml',
    epochs: 10,
    project: 'runs/train',
    name: 'experiment1'
  });
  
  if (!trainResult.success) {
    throw new Error('Training failed');
  }
  
  // 2. Validate trained model
  const valResult = await tool.val({
    model: 'runs/train/experiment1/weights/best.pt',
    data: 'coco128.yaml'
  });
  
  // 3. Export model
  const exportResult = await tool.export({
    model: 'runs/train/experiment1/weights/best.pt',
    format: 'onnx'
  });
  
  return {
    training: trainResult,
    validation: valResult,
    export: exportResult
  };
}
```

### Real-time Training with Progress Updates

```typescript
// Complete training pipeline with real-time monitoring
function trainWithProgress() {
  return new Promise((resolve, reject) => {
    const eventSource = tool.trainSSE({
      model: 'yolov8n.pt',
      data: 'coco128.yaml',
      epochs: 100
    }, {
      onProgress: (data) => {
        console.log(`📊 Epoch ${data.epoch}/${data.total_epochs} - Loss: ${data.loss}`);
        updateProgressBar(data.epoch / data.total_epochs);
      },
      onMetrics: (data) => {
        console.log(`🎯 mAP: ${data.mAP50}, Precision: ${data.precision}`);
        updateMetricsChart(data);
      },
      onComplete: (result) => {
        console.log('🎉 Training completed successfully!');
        console.log(`📈 Final mAP50: ${result.metrics.mAP50}`);
        resolve(result);
      },
      onError: (error) => {
        console.error('❌ Training failed:', error);
        reject(error);
      }
    });
  });
}
```

## 🔧 Development & Building

### 📦 Package Development

```bash
# Clone and setup
git clone https://github.com/your-org/ultralytics-mcp-tool.git
cd ultralytics-mcp-tool

# Install dependencies
npm install

# Build TypeScript
npm run build

# Run development server
npm run dev

# Watch mode for development
npm run watch
```

### 🧪 Testing Suite

```bash
# Run all tests
npm test

# Unit tests only
npm run test:unit

# Integration tests (requires API server)
npm run test:integration

# E2E tests with real workflows
npm run test:e2e

# Test coverage report
npm run test:coverage
```

### 📋 Code Quality

```bash
# Lint TypeScript code
npm run lint

# Format code with Prettier
npm run format

# Type checking
npm run type-check

# Pre-commit hooks
npm run pre-commit
```

## 📄 License & Support

### 📝 License
**MIT License** - Full commercial and personal use permitted.

### 🆘 Getting Help

| Resource | Link | Purpose |
|----------|------|---------|
| **📖 API Documentation** | [Ultralytics Docs](https://docs.ultralytics.com) | Official YOLO documentation |
| **🐛 Report Issues** | [GitHub Issues](https://github.com/your-org/ultralytics-mcp-tool/issues) | Bug reports & feature requests |
| **💬 Community** | [GitHub Discussions](https://github.com/your-org/ultralytics-mcp-tool/discussions) | Questions & community support |
| **🔧 MCP Specification** | [Model Context Protocol](https://modelcontextprotocol.io) | Protocol documentation |

### 🤝 Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

**Contribution Guidelines:**
- Follow TypeScript best practices
- Add tests for new features
- Update documentation
- Ensure CI/CD passes

---

<div align="center">

### 🚀 **Built for Seamless YOLO Integration**

*Empowering developers to build intelligent computer vision applications with ease*

**[⭐ Star on GitHub](https://github.com/your-org/ultralytics-mcp-tool)** | **[📦 NPM Package](https://www.npmjs.com/package/ultralytics-mcp-tool)** | **[📖 Documentation](https://docs.ultralytics.com)**

</div>
