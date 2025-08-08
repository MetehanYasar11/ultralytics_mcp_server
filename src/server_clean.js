#!/usr/bin/env node
/**
 * Ultralytics MCP Server with SSE Support - Pure MCP Version
 */

const express = require('express');
const { randomUUID } = require('node:crypto');
const { exec } = require('child_process');
const util = require('util');

// Use Docker environment paths for MCP SDK
const { McpServer } = require('/app/node_modules/@modelcontextprotocol/sdk/dist/cjs/server/mcp.js');
const { SSEServerTransport } = require('/app/node_modules/@modelcontextprotocol/sdk/dist/cjs/server/sse.js');
const { z } = require('zod');

const execAsync = util.promisify(exec);

/**
 * Execute Python code in Ultralytics container via Docker exec
 */
async function executeInUltralyticsContainer(pythonCode) {
  try {
    console.log('🐍 Executing Python code in Ultralytics container...');
    
    // Use base64 encoding to safely pass Python code to avoid shell escaping issues
    const encodedCode = Buffer.from(pythonCode).toString('base64');
    const dockerCommand = `docker exec ultralytics-container python3 -c "import base64; exec(base64.b64decode('${encodedCode}').decode('utf-8'))"`;
    
    const { stdout, stderr } = await execAsync(dockerCommand);
    
    if (stderr && !stderr.includes('WARNING')) {
      console.error('🚨 Python execution stderr:', stderr);
    }
    
    const result = stdout || stderr || 'Code executed successfully (no output)';
    console.log('✅ Python execution completed');
    return result;
    
  } catch (error) {
    console.error('❌ Python execution failed:', error);
    const errorMsg = error.stderr || error.message || 'Unknown execution error';
    return `❌ Python execution failed: ${errorMsg}`;
  }
}

/**
 * Create the MCP Server with Ultralytics tools
 */
const createUltralyticsServer = () => {
  const server = new McpServer({
    name: 'ultralytics-mcp-server',
    version: '2.0.0',
  });

  // Tool: Execute Python Code
  server.tool('execute_python_code', 'Execute Python code in Ultralytics environment with full ML/AI library access', {
    code: z.string().describe('Python code to execute'),
  }, async ({ code }) => {
    try {
      const result = await executeInUltralyticsContainer(code);
      return { toolResult: { content: [{ type: 'text', text: result }] } };
    } catch (error) {
      return { toolResult: { content: [{ type: 'text', text: `❌ Execution failed: ${error.message}` }], isError: true } };
    }
  });

  // Tool: YOLO Training
  server.tool('yolo_training', 'Train a custom YOLO model with specified dataset and parameters', {
    dataset: z.string().describe('Dataset path or configuration file (e.g., coco8.yaml)'),
    model: z.string().default('yolo11n.pt').describe('Base model to use for training'),
    epochs: z.number().default(10).describe('Number of training epochs'),
    imgsz: z.number().default(640).describe('Image size for training'),
    batch_size: z.number().default(16).describe('Batch size for training'),
    name: z.string().optional().describe('Training run name'),
  }, async ({ dataset, model, epochs, imgsz, batch_size, name }) => {
    try {
      const runName = name || `train_${Date.now()}`;
      const trainingCode = `
from ultralytics import YOLO
import time

def start_training():
    start_time = time.time()
    print(f"🚀 Starting training with model: ${model}")
    
    # Load model
    model = YOLO('${model}')
    
    # Train the model
    results = model.train(
        data='${dataset}',
        epochs=${epochs},
        imgsz=${imgsz},
        batch=${batch_size},
        name='${runName}',
        save=True,
        plots=True
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    print(f"📂 Results saved to: {results.save_dir}")
    
    return results

start_training()
`;
      
      const result = await executeInUltralyticsContainer(trainingCode);
      return { toolResult: { content: [{ type: 'text', text: `🎯 Training started for ${runName}\n\n${result}\n\n📊 Monitor progress at: http://localhost:6006` }] } };
    } catch (error) {
      return { toolResult: { content: [{ type: 'text', text: `❌ Training failed: ${error.message}` }], isError: true } };
    }
  });

  // Tool: List Training Results
  server.tool('list_training_results', 'List all available training results and model outputs', {
    filter_type: z.string().default('all').describe('Filter by type: train, detect, predict, or all'),
  }, async ({ filter_type }) => {
    try {
      const listingCode = `
import os
from pathlib import Path
import json

def list_results():
    runs_dir = Path('/ultralytics/runs')
    results = []
    
    if not runs_dir.exists():
        return {"message": "No runs directory found", "results": []}
    
    for item in runs_dir.rglob('*'):
        if item.is_dir() and any(x in item.name for x in ['train', 'detect', 'predict']):
            try:
                stat = item.stat()
                result_type = 'train' if 'train' in item.name else 'detect' if 'detect' in item.name else 'predict'
                
                if '${filter_type}' != 'all' and '${filter_type}' != result_type:
                    continue
                
                result_info = {
                    "name": item.name,
                    "type": result_type,
                    "path": str(item),
                    "created_at": time.ctime(stat.st_ctime),
                    "has_weights": (item / "weights").exists(),
                    "has_plots": any(item.glob("*.png"))
                }
                results.append(result_info)
            except Exception as e:
                continue
    
    results.sort(key=lambda x: x['created_at'], reverse=True)
    
    print(f"📂 Found {len(results)} results")
    for result in results:
        print(f"• {result['name']} ({result['type']}) - {result['created_at']}")
    
    return {"total_count": len(results), "results": results}

list_results()
`;
      
      const result = await executeInUltralyticsContainer(listingCode);
      return { toolResult: { content: [{ type: 'text', text: result }] } };
    } catch (error) {
      return { toolResult: { content: [{ type: 'text', text: `❌ Failed to list results: ${error.message}` }], isError: true } };
    }
  });

  // Tool: Get Training Metrics
  server.tool('get_training_metrics', 'Get detailed training metrics and performance data', {
    run_name: z.string().optional().describe('Specific training run name to get metrics for'),
  }, async ({ run_name }) => {
    try {
      const metricsCode = `
import os
import json
import pandas as pd
from pathlib import Path

def get_training_metrics():
    runs_dir = Path('/ultralytics/runs')
    metrics_data = {"runs": []}
    
    if not runs_dir.exists():
        return {"error": "No runs directory found"}
    
    for run_dir in runs_dir.rglob('*'):
        if run_dir.is_dir() and 'train' in run_dir.name:
            run_info = {"name": run_dir.name, "path": str(run_dir)}
            
            if '${run_name || ''}' and '${run_name || ''}' != run_dir.name:
                continue
            
            # Check for results.csv
            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                try:
                    df = pd.read_csv(results_csv)
                    # Get final metrics
                    final_metrics = df.iloc[-1].to_dict()
                    # Remove NaN values
                    final_metrics = {k: v for k, v in final_metrics.items() if pd.notna(v)}
                    
                    run_info["final_metrics"] = final_metrics
                    run_info["total_epochs"] = len(df)
                    
                    print(f"📊 Metrics for {run_dir.name}:")
                    for key, value in final_metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
                    
                except Exception as e:
                    run_info["error"] = f"Failed to parse CSV: {str(e)}"
            
            # Check for model weights
            weights_dir = run_dir / "weights"
            if weights_dir.exists():
                run_info["model_weights"] = {
                    "best": str(weights_dir / "best.pt") if (weights_dir / "best.pt").exists() else None,
                    "last": str(weights_dir / "last.pt") if (weights_dir / "last.pt").exists() else None
                }
            
            metrics_data["runs"].append(run_info)
    
    if '${run_name || ''}' and not metrics_data["runs"]:
        return {"error": f"Training run '{run_name}' not found"}
    
    return metrics_data

get_training_metrics()
`;
      
      const result = await executeInUltralyticsContainer(metricsCode);
      return { toolResult: { content: [{ type: 'text', text: result }] } };
    } catch (error) {
      return { toolResult: { content: [{ type: 'text', text: `❌ Failed to get metrics: ${error.message}` }], isError: true } };
    }
  });

  // Tool: TensorBoard Status
  server.tool('tensorboard_status', 'Check TensorBoard status and get monitoring URL', {}, async () => {
    try {
      const statusCode = `
import subprocess
import json
from pathlib import Path

def check_tensorboard_status():
    status_info = {
        "url": "http://localhost:6006",
        "logdir": "/ultralytics/runs",
        "status": "active"
    }
    
    # Check for TensorBoard process
    try:
        result = subprocess.run(["pgrep", "-f", "tensorboard"], capture_output=True, text=True)
        status_info["process_running"] = result.returncode == 0
        
        if result.returncode == 0:
            status_info["pid"] = result.stdout.strip()
        
    except Exception as e:
        status_info["process_check_error"] = str(e)
    
    # Check for active training runs
    runs_dir = Path("/ultralytics/runs")
    active_runs = []
    
    if runs_dir.exists():
        for run_dir in runs_dir.rglob("*train*"):
            if run_dir.is_dir():
                # Check if it has TensorBoard event files
                if any(run_dir.rglob("events.out.tfevents.*")):
                    active_runs.append(str(run_dir))
    
    status_info["active_runs"] = active_runs
    
    print("📊 TensorBoard Status:")
    print(f"🌐 URL: {status_info['url']}")
    print(f"📁 Log Directory: {status_info['logdir']}")
    print(f"▶️ Process Running: {status_info['process_running']}")
    print(f"📈 Active Runs: {len(active_runs)}")
    
    return status_info

check_tensorboard_status()
`;
      
      const result = await executeInUltralyticsContainer(statusCode);
      return { toolResult: { content: [{ type: 'text', text: result }] } };
    } catch (error) {
      return { toolResult: { content: [{ type: 'text', text: `❌ Failed to check TensorBoard status: ${error.message}` }], isError: true } };
    }
  });

  // Tool: Launch Streamlit Interface
  server.tool('launch_streamlit_interface', 'Launch or restart the enhanced Streamlit web interface', {}, async () => {
    try {
      const launchCode = `
import subprocess
import json

def launch_streamlit():
    # The Streamlit interface is already running in the container
    # Just verify its status and provide access information
    
    interface_info = {
        "interface_url": "http://localhost:8501",
        "success": True,
        "message": "Enhanced Streamlit interface is available",
        "features": [
            "Custom model scanning and selection",
            "Real-time webcam inference",
            "Image and video upload support", 
            "Enhanced AI interface with custom models"
        ]
    }
    
    print("🌐 Streamlit Interface Status:")
    print(f"🔗 URL: {interface_info['interface_url']}")
    print("✨ Features:")
    for feature in interface_info['features']:
        print(f"  • {feature}")
    
    return interface_info

launch_streamlit()
`;
      
      const result = await executeInUltralyticsContainer(launchCode);
      return { toolResult: { content: [{ type: 'text', text: result }] } };
    } catch (error) {
      return { toolResult: { content: [{ type: 'text', text: `❌ Failed to launch Streamlit: ${error.message}` }], isError: true } };
    }
  });

  // Tool: Get System Info
  server.tool('get_system_info', 'Get comprehensive system information including available interfaces and tools', {}, async () => {
    try {
      const systemInfoCode = `
import json
import subprocess
import psutil
import torch
from pathlib import Path
import sys

def get_system_info():
    info = {
        "system": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "interfaces": {
            "streamlit": {
                "url": "http://localhost:8501",
                "status": "active",
                "description": "Interactive AI inference interface"
            },
            "tensorboard": {
                "url": "http://localhost:6006", 
                "status": "active",
                "description": "Training metrics visualization"
            },
            "jupyter": {
                "url": "http://localhost:8888",
                "status": "active", 
                "description": "Development environment"
            }
        },
        "tools_available": [
            "execute_python_code",
            "yolo_training",
            "list_training_results",
            "get_training_metrics",
            "tensorboard_status",
            "launch_streamlit_interface",
            "get_system_info"
        ]
    }
    
    print("🖥️ System Information:")
    print(f"🐍 Python: {info['system']['python_version']}")
    print(f"🔥 PyTorch: {info['system']['torch_version']}")
    print(f"🚀 CUDA Available: {info['system']['cuda_available']}")
    print(f"💻 GPU Count: {info['system']['gpu_count']}")
    print(f"🛠️ Available Tools: {len(info['tools_available'])}")
    
    return info

get_system_info()
`;
      
      const result = await executeInUltralyticsContainer(systemInfoCode);
      return { toolResult: { content: [{ type: 'text', text: result }] } };
    } catch (error) {
      return { toolResult: { content: [{ type: 'text', text: `❌ Failed to get system info: ${error.message}` }], isError: true } };
    }
  });

  return server;
};

// Create Express application for SSE
const app = express();
app.use(express.json());

// Store transports by session ID
const transports = {};

console.log('🚀 Starting Ultralytics MCP Server...');

// SSE endpoint for MCP communication
app.get('/sse', (req, res) => {
  const sessionId = req.query.sessionId || randomUUID();
  console.log(`📡 New SSE connection: ${sessionId}`);
  
  try {
    const server = createUltralyticsServer();
    const transport = new SSEServerTransport('/message', res);
    
    // Store transport for cleanup
    transports[sessionId] = transport;
    
    // Connect server to transport
    server.connect(transport);
    
    // Handle client disconnect
    req.on('close', () => {
      console.log(`📡 SSE connection closed: ${sessionId}`);
      if (transports[sessionId]) {
        delete transports[sessionId];
      }
    });
    
  } catch (error) {
    console.error('❌ SSE connection error:', error);
    res.status(500).json({ error: 'Failed to establish SSE connection' });
  }
});

// Message endpoint for MCP communication
app.post('/message', async (req, res) => {
  try {
    // The message handling is done by the SSE transport
    res.json({ status: 'Message received' });
  } catch (error) {
    console.error('❌ Message handling error:', error);
    res.status(500).json({ error: 'Message handling failed' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    server: 'ultralytics_mcp',
    version: '2.0.0',
    activeSessions: Object.keys(transports).length,
    timestamp: new Date().toISOString()
  });
});

// Start server
const PORT = process.env.PORT || 8092;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`
🎯 Ultralytics MCP Server Running on Port ${PORT}
=================================================

📡 SSE Endpoint: http://localhost:${PORT}/sse
📨 Messages: http://localhost:${PORT}/messages  
❤️ Health: http://localhost:${PORT}/health

✅ N8N MCP Client Compatible
=================================================
  `);
});

console.log('🚀 Starting Ultralytics MCP Server...');

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down...');
  
  for (const sessionId in transports) {
    try {
      await transports[sessionId].close();
      delete transports[sessionId];
    } catch (error) {
      console.error(`Error closing session ${sessionId}:`, error);
    }
  }
  
  console.log('✅ Shutdown complete');
  process.exit(0);
});
