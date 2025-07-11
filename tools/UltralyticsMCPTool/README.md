# Ultralytics MCP Tool

A Model Context Protocol (MCP) compliant tool for interacting with Ultralytics YOLO operations through multiple communication channels.

## Features

- **MCP Compliant**: Fully compatible with Model Context Protocol specification
- **Multi-Channel Communication**:
  - **HTTP**: RESTful API calls via axios
  - **SSE**: Server-Sent Events for real-time progress updates
  - **Stdio**: Standard input/output for direct process communication
- **Complete YOLO Operations**: Train, validate, predict, export, track, benchmark
- **TypeScript**: Full type safety and IntelliSense support
- **Minimal Dependencies**: Only axios and eventsource as external dependencies

## Installation

```bash
npm install ultralytics-mcp-tool
```

Or install dependencies manually:

```bash
npm install axios eventsource
npm install --save-dev @types/node @types/eventsource typescript
```

## Usage

### Basic Usage

```typescript
import UltralyticsMCPTool from 'ultralytics-mcp-tool';

const tool = new UltralyticsMCPTool('http://localhost:8000');

// Train a model
const trainResult = await tool.train({
  model: 'yolov8n.pt',
  data: 'coco128.yaml',
  epochs: 10,
  device: 'cpu'
});

console.log('Training completed:', trainResult.success);
```

### Get MCP Manifest

```typescript
// Static method to get the full MCP manifest
const manifest = UltralyticsMCPTool.manifest();
console.log('Available tools:', manifest.tools.map(t => t.name));
```

### HTTP Channel (Default)

```typescript
// All operations use HTTP by default
const result = await tool.predict({
  model: 'yolov8n.pt',
  source: 'path/to/images',
  conf: 0.5
});
```

### SSE Channel (Real-time Updates)

```typescript
// Subscribe to real-time training updates
const eventSource = tool.trainSSE({
  model: 'yolov8n.pt',
  data: 'coco128.yaml',
  epochs: 100
}, {
  onProgress: (data) => console.log('Training progress:', data),
  onMetrics: (data) => console.log('Metrics update:', data),
  onComplete: (result) => console.log('Training complete:', result),
  onError: (error) => console.error('Training error:', error)
});

// Stop listening
// eventSource.close();
```

### Stdio Channel

```typescript
// Use stdio for direct process communication
const result = await tool.executeStdio('train', {
  model: 'yolov8n.pt',
  data: 'coco128.yaml',
  epochs: 5
});
```

### Generic Execute Method

```typescript
// Choose channel dynamically
const httpResult = await tool.execute('predict', params, 'http');
const stdioResult = await tool.execute('predict', params, 'stdio');
```

## Available Operations

### Core YOLO Operations

- **`train(params)`**: Train a YOLO model
- **`val(params)`**: Validate a model
- **`predict(params)`**: Run predictions
- **`export(params)`**: Export model to different formats
- **`track(params)`**: Object tracking
- **`benchmark(params)`**: Benchmark model performance

### Utility Operations

- **`copyCfg(params)`**: Copy configuration files
- **`settings(params)`**: View/update Ultralytics settings
- **`health()`**: Check API health

## MCP Manifest

The tool provides a complete MCP manifest via the static `manifest()` method:

```json
{
  "schema_version": "1.0",
  "name": "ultralytics_mcp",
  "description": "YOLO tasks via Ultralytics CLI",
  "version": "0.1.0",
  "interfaces": {
    "http": {
      "openapi_url": "http://localhost:8000/openapi.json"
    },
    "sse": {
      "base_url": "http://localhost:8000/sse"
    },
    "stdio": {
      "command": "python -m ultralytics_stdio"
    }
  },
  "tools": [...]
}
```

## Parameters

### Common Parameters

- **`model`**: Model path or name (e.g., 'yolov8n.pt')
- **`data`**: Dataset YAML file path
- **`source`**: Source images/video path
- **`epochs`**: Number of training epochs
- **`imgsz`**: Image size (default: 640)
- **`device`**: Device to use ('cpu', '0', '1', etc.)
- **`extra_args`**: Additional arguments as key-value pairs

### Operation-Specific Parameters

Each operation has specific parameters documented in the MCP manifest. See the `inputSchema` for each tool.

## Response Format

All operations return a standardized response:

```typescript
interface UltralyticsResponse {
  run_id: string;           // Unique operation ID
  command: string;          // Executed command
  return_code: number;      // Process return code
  stdout: string;           // Standard output
  stderr: string;           // Standard error
  metrics: object;          // Parsed metrics
  artifacts: string[];      // Generated files
  success: boolean;         // Success status
  timestamp: string;        // ISO timestamp
}
```

## Error Handling

```typescript
try {
  const result = await tool.train({
    model: 'yolov8n.pt',
    data: 'invalid-dataset.yaml'
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
function trainWithProgress() {
  return new Promise((resolve, reject) => {
    const eventSource = tool.trainSSE({
      model: 'yolov8n.pt',
      data: 'coco128.yaml',
      epochs: 100
    }, {
      onProgress: (data) => {
        console.log(`Epoch ${data.epoch}/${data.total_epochs} - Loss: ${data.loss}`);
      },
      onMetrics: (data) => {
        console.log(`mAP: ${data.mAP50}, Precision: ${data.precision}`);
      },
      onComplete: (result) => {
        console.log('Training completed successfully!');
        resolve(result);
      },
      onError: (error) => {
        console.error('Training failed:', error);
        reject(error);
      }
    });
  });
}
```

## Building

```bash
npm run build
```

## Testing

```bash
npm test
```

## License

MIT

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests to the main repository.
