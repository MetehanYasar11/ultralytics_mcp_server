import axios, { AxiosInstance } from 'axios';
import EventSource from 'eventsource';
import { spawn, ChildProcess } from 'child_process';

interface MCPManifest {
  schema_version: string;
  name: string;
  description: string;
  version: string;
  interfaces: {
    http: {
      openapi_url: string;
    };
    sse: {
      base_url: string;
    };
    stdio: {
      command: string;
    };
  };
  tools: MCPTool[];
}

interface MCPTool {
  name: string;
  description: string;
  inputSchema: {
    type: string;
    properties: Record<string, any>;
    required: string[];
  };
}

interface UltralyticsRequest {
  model?: string;
  data?: string;
  source?: string;
  epochs?: number;
  imgsz?: number;
  device?: string;
  extra_args?: Record<string, any>;
  [key: string]: any;
}

interface UltralyticsResponse {
  run_id: string;
  command: string;
  return_code: number;
  stdout: string;
  stderr: string;
  metrics: Record<string, any>;
  artifacts: string[];
  success: boolean;
  timestamp: string;
}

interface SSEMessage {
  type: 'progress' | 'metrics' | 'complete' | 'error';
  data: any;
}

interface StdioMessage {
  method: string;
  params: any;
  id?: string;
}

export default class UltralyticsMCPTool {
  private httpClient: AxiosInstance;
  private baseUrl: string;
  private sseUrl: string;
  private stdioCommand: string;

  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.sseUrl = `${baseUrl}/sse`;
    this.stdioCommand = 'python -m ultralytics_stdio';
    
    this.httpClient = axios.create({
      baseURL: baseUrl,
      timeout: 3600000, // 1 hour timeout
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Static method to return the MCP manifest
   */
  static manifest(): MCPManifest {
    return {
      schema_version: '1.0',
      name: 'ultralytics_mcp',
      description: 'YOLO tasks via Ultralytics CLI',
      version: '0.1.0',
      interfaces: {
        http: {
          openapi_url: 'http://localhost:8000/openapi.json'
        },
        sse: {
          base_url: 'http://localhost:8000/sse'
        },
        stdio: {
          command: 'python -m ultralytics_stdio'
        }
      },
      tools: [
        {
          name: 'train',
          description: 'Train a YOLO model on a dataset',
          inputSchema: {
            type: 'object',
            properties: {
              model: { type: 'string', description: 'Model to train (e.g., yolov8n.pt)' },
              data: { type: 'string', description: 'Dataset YAML file path' },
              epochs: { type: 'number', description: 'Number of training epochs', default: 100 },
              imgsz: { type: 'number', description: 'Image size', default: 640 },
              batch: { type: 'number', description: 'Batch size', default: 16 },
              device: { type: 'string', description: 'Device to use', default: 'cpu' },
              project: { type: 'string', description: 'Project directory', default: 'runs/train' },
              name: { type: 'string', description: 'Experiment name' },
              extra_args: { type: 'object', description: 'Additional arguments' }
            },
            required: ['model', 'data']
          }
        },
        {
          name: 'val',
          description: 'Validate a YOLO model',
          inputSchema: {
            type: 'object',
            properties: {
              model: { type: 'string', description: 'Model to validate' },
              data: { type: 'string', description: 'Dataset YAML file path' },
              imgsz: { type: 'number', description: 'Image size', default: 640 },
              batch: { type: 'number', description: 'Batch size', default: 32 },
              device: { type: 'string', description: 'Device to use', default: 'cpu' },
              project: { type: 'string', description: 'Project directory', default: 'runs/val' },
              extra_args: { type: 'object', description: 'Additional arguments' }
            },
            required: ['model', 'data']
          }
        },
        {
          name: 'predict',
          description: 'Run prediction with a YOLO model',
          inputSchema: {
            type: 'object',
            properties: {
              model: { type: 'string', description: 'Model to use for prediction' },
              source: { type: 'string', description: 'Source images/video path' },
              imgsz: { type: 'number', description: 'Image size', default: 640 },
              conf: { type: 'number', description: 'Confidence threshold', default: 0.25 },
              device: { type: 'string', description: 'Device to use', default: 'cpu' },
              project: { type: 'string', description: 'Project directory', default: 'runs/predict' },
              save: { type: 'boolean', description: 'Save results', default: true },
              extra_args: { type: 'object', description: 'Additional arguments' }
            },
            required: ['model', 'source']
          }
        },
        {
          name: 'export',
          description: 'Export a YOLO model to different formats',
          inputSchema: {
            type: 'object',
            properties: {
              model: { type: 'string', description: 'Model to export' },
              format: { type: 'string', description: 'Export format', default: 'onnx' },
              imgsz: { type: 'number', description: 'Image size', default: 640 },
              device: { type: 'string', description: 'Device to use', default: 'cpu' },
              dynamic: { type: 'boolean', description: 'Dynamic axes', default: false },
              simplify: { type: 'boolean', description: 'Simplify ONNX model', default: false },
              extra_args: { type: 'object', description: 'Additional arguments' }
            },
            required: ['model']
          }
        },
        {
          name: 'track',
          description: 'Run object tracking with a YOLO model',
          inputSchema: {
            type: 'object',
            properties: {
              model: { type: 'string', description: 'Model to use for tracking' },
              source: { type: 'string', description: 'Source video path' },
              imgsz: { type: 'number', description: 'Image size', default: 640 },
              conf: { type: 'number', description: 'Confidence threshold', default: 0.25 },
              device: { type: 'string', description: 'Device to use', default: 'cpu' },
              tracker: { type: 'string', description: 'Tracker config', default: 'bytetrack.yaml' },
              project: { type: 'string', description: 'Project directory', default: 'runs/track' },
              extra_args: { type: 'object', description: 'Additional arguments' }
            },
            required: ['model', 'source']
          }
        },
        {
          name: 'benchmark',
          description: 'Benchmark a YOLO model',
          inputSchema: {
            type: 'object',
            properties: {
              model: { type: 'string', description: 'Model to benchmark' },
              imgsz: { type: 'number', description: 'Image size', default: 640 },
              device: { type: 'string', description: 'Device to use', default: 'cpu' },
              data: { type: 'string', description: 'Dataset for benchmarking' },
              extra_args: { type: 'object', description: 'Additional arguments' }
            },
            required: ['model']
          }
        },
        {
          name: 'copyCfg',
          description: 'Copy default configuration files',
          inputSchema: {
            type: 'object',
            properties: {
              type: { type: 'string', description: 'Config type to copy (tracker, cfg, etc.)' },
              destination: { type: 'string', description: 'Destination path' }
            },
            required: ['type']
          }
        },
        {
          name: 'settings',
          description: 'View or update Ultralytics settings',
          inputSchema: {
            type: 'object',
            properties: {
              key: { type: 'string', description: 'Setting key to view/update' },
              value: { type: 'string', description: 'New value for the setting' },
              reset: { type: 'boolean', description: 'Reset settings to default', default: false }
            },
            required: []
          }
        }
      ]
    };
  }

  // HTTP Channel Methods
  async executeHTTP(operation: string, params: UltralyticsRequest): Promise<UltralyticsResponse> {
    try {
      const response = await this.httpClient.post(`/${operation}`, params);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`HTTP request failed: ${error.message}${error.response ? ` - ${JSON.stringify(error.response.data)}` : ''}`);
      }
      throw error;
    }
  }

  async train(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeHTTP('train', params);
  }

  async val(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeHTTP('val', params);
  }

  async predict(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeHTTP('predict', params);
  }

  async export(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeHTTP('export', params);
  }

  async track(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeHTTP('track', params);
  }

  async benchmark(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeHTTP('benchmark', params);
  }

  async solution(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeHTTP('solution', params);
  }

  async health(): Promise<any> {
    return this.executeHTTP('', {});
  }

  // SSE Channel Methods
  subscribeSSE(operation: string, params: UltralyticsRequest, callbacks: {
    onProgress?: (data: any) => void;
    onMetrics?: (data: any) => void;
    onComplete?: (data: UltralyticsResponse) => void;
    onError?: (error: any) => void;
  }): EventSource {
    const url = new URL(`${this.sseUrl}/${operation}`);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.append(key, String(value));
      }
    });

    const eventSource = new EventSource(url.toString());

    eventSource.onmessage = (event) => {
      try {
        const message: SSEMessage = JSON.parse(event.data);
        
        switch (message.type) {
          case 'progress':
            callbacks.onProgress?.(message.data);
            break;
          case 'metrics':
            callbacks.onMetrics?.(message.data);
            break;
          case 'complete':
            callbacks.onComplete?.(message.data);
            eventSource.close();
            break;
          case 'error':
            callbacks.onError?.(message.data);
            eventSource.close();
            break;
        }
      } catch (error) {
        callbacks.onError?.(error);
      }
    };

    eventSource.onerror = (error) => {
      callbacks.onError?.(error);
    };

    return eventSource;
  }

  trainSSE(params: UltralyticsRequest, callbacks: Parameters<typeof this.subscribeSSE>[2]): EventSource {
    return this.subscribeSSE('train', params, callbacks);
  }

  valSSE(params: UltralyticsRequest, callbacks: Parameters<typeof this.subscribeSSE>[2]): EventSource {
    return this.subscribeSSE('val', params, callbacks);
  }

  predictSSE(params: UltralyticsRequest, callbacks: Parameters<typeof this.subscribeSSE>[2]): EventSource {
    return this.subscribeSSE('predict', params, callbacks);
  }

  // Stdio Channel Methods
  executeStdio(operation: string, params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return new Promise((resolve, reject) => {
      const child: ChildProcess = spawn('python', ['-m', 'ultralytics_stdio'], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdoutData = '';
      let stderrData = '';

      const message: StdioMessage = {
        method: operation,
        params: params,
        id: Date.now().toString()
      };

      child.stdin?.write(JSON.stringify(message) + '\n');
      child.stdin?.end();

      child.stdout?.on('data', (data) => {
        stdoutData += data.toString();
      });

      child.stderr?.on('data', (data) => {
        stderrData += data.toString();
      });

      child.on('close', (code) => {
        try {
          if (code === 0 && stdoutData.trim()) {
            const response = JSON.parse(stdoutData.trim());
            resolve(response);
          } else {
            reject(new Error(`Stdio process failed with code ${code}: ${stderrData}`));
          }
        } catch (error) {
          reject(new Error(`Failed to parse stdio response: ${error} - Output: ${stdoutData}`));
        }
      });

      child.on('error', (error) => {
        reject(new Error(`Stdio process error: ${error.message}`));
      });

      // Timeout after 1 hour
      setTimeout(() => {
        child.kill();
        reject(new Error('Stdio operation timed out'));
      }, 3600000);
    });
  }

  trainStdio(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeStdio('train', params);
  }

  valStdio(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeStdio('val', params);
  }

  predictStdio(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeStdio('predict', params);
  }

  exportStdio(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeStdio('export', params);
  }

  trackStdio(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeStdio('track', params);
  }

  benchmarkStdio(params: UltralyticsRequest): Promise<UltralyticsResponse> {
    return this.executeStdio('benchmark', params);
  }

  // Utility Methods
  async copyCfg(params: { type: string; destination?: string }): Promise<UltralyticsResponse> {
    const cfgParams = {
      task: 'copy-cfg',
      extra_args: params
    };
    return this.executeHTTP('solution', cfgParams);
  }

  async settings(params: { key?: string; value?: string; reset?: boolean } = {}): Promise<UltralyticsResponse> {
    const settingsParams = {
      task: 'settings',
      extra_args: params
    };
    return this.executeHTTP('solution', settingsParams);
  }

  // Generic execute method that can use any channel
  async execute(
    operation: string, 
    params: UltralyticsRequest, 
    channel: 'http' | 'stdio' = 'http'
  ): Promise<UltralyticsResponse> {
    switch (channel) {
      case 'http':
        return this.executeHTTP(operation, params);
      case 'stdio':
        return this.executeStdio(operation, params);
      default:
        throw new Error(`Unsupported channel: ${channel}`);
    }
  }

  // Connection testing
  async testConnection(): Promise<{ http: boolean; sse: boolean; stdio: boolean }> {
    const results = {
      http: false,
      sse: false,
      stdio: false
    };

    // Test HTTP
    try {
      await this.health();
      results.http = true;
    } catch (error) {
      // HTTP connection failed
    }

    // Test SSE
    try {
      const testSSE = new EventSource(`${this.sseUrl}/health`);
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          testSSE.close();
          reject(new Error('SSE timeout'));
        }, 5000);

        testSSE.onopen = () => {
          clearTimeout(timeout);
          testSSE.close();
          results.sse = true;
          resolve(true);
        };

        testSSE.onerror = () => {
          clearTimeout(timeout);
          testSSE.close();
          reject(new Error('SSE error'));
        };
      });
    } catch (error) {
      // SSE connection failed
    }

    // Test Stdio
    try {
      await this.executeStdio('health', {});
      results.stdio = true;
    } catch (error) {
      // Stdio connection failed
    }

    return results;
  }

  // Cleanup method
  cleanup(): void {
    // Any cleanup logic for open connections, event sources, etc.
    this.httpClient.defaults.timeout = 1000; // Short timeout for cleanup
  }
}
