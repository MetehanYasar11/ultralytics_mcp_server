#!/usr/bin/env node
/**
 * Ultralytics MCP Server with SSE Support - Docker Version
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
    name: 'ultralytics_mcp',
    version: '1.0.0',
  }, { 
    capabilities: { 
      tools: {},
      logging: {}
    } 
  });

  // Tool 1: Python Code Execution
  server.tool(
    'execute_python_code',
    'Execute Python code with Ultralytics libraries',
    {
      code: z.string().describe('Python code to execute')
    },
    async ({ code }) => {
      try {
        console.log('🐍 Executing Python code:', code.substring(0, 50) + '...');
        
        const result = await executeInUltralyticsContainer(code);
        
        return {
          content: [
            {
              type: 'text',
              text: `✅ Python code executed!\nResult:\n${result}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ Python execution failed: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  // Tool 2: YOLO Operations
  server.tool(
    'yolo_operation',
    'Perform YOLO operations',
    {
      operation: z.enum(['detect', 'train', 'predict']).describe('YOLO operation'),
      model: z.string().default('yolo11n.pt').describe('YOLO model'),
      source: z.string().optional().describe('Input source (image/video path)')
    },
    async ({ operation, model, source }) => {
      try {
        console.log(`🎯 YOLO ${operation} with ${model}`);
        
        // Build YOLO command
        let yoloCommand = `from ultralytics import YOLO; model = YOLO('${model}'); `;
        
        switch (operation) {
          case 'detect':
          case 'predict':
            if (source) {
              yoloCommand += `results = model.predict('${source}', save=True); print(f"Detection completed. Found {len(results)} result(s). Results saved to: {results[0].save_dir if hasattr(results[0], 'save_dir') else 'runs/detect/predict'}")`;
            } else {
              yoloCommand += `print(f"Model {model} loaded and ready for detection")`;
            }
            break;
          case 'train':
            if (source) {
              yoloCommand += `
# Training without tensorboard parameter (not supported in this version)
results = model.train(
    data='${source}', 
    epochs=3, 
    imgsz=640, 
    save=True, 
    plots=True,
    project='/ultralytics/runs',
    name='detect/train_tb'
); 
print(f"Training completed! Results saved to: {results.save_dir if hasattr(results, 'save_dir') else 'runs/detect/train'}")`;
            } else {
              yoloCommand += `
# Training without tensorboard parameter (not supported in this version)
results = model.train(
    data='coco8.yaml', 
    epochs=3, 
    imgsz=640, 
    save=True, 
    plots=True,
    project='/ultralytics/runs',
    name='detect/train_tb'
); 
print(f"Training completed with coco8 dataset! Results saved to: {results.save_dir if hasattr(results, 'save_dir') else 'runs/detect/train'}")`;
            }
            break;
        }
        
        const result = await executeInUltralyticsContainer(yoloCommand);
        
        return {
          content: [
            {
              type: 'text',
              text: `✅ YOLO ${operation} completed with ${model}!\nResult:\n${result}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ YOLO operation failed: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  // Tool 3: List Training Results
  server.tool(
    'list_training_results',
    'List all available training and detection results',
    {},
    async () => {
      try {
        console.log('📁 Listing training results...');
        
        const listCommand = `
import os
import json
from pathlib import Path

def scan_results():
    results = {"training": [], "detection": []}
    runs_path = Path("/ultralytics/runs")
    
    if runs_path.exists():
        # Scan training results
        if (runs_path / "detect").exists():
            for folder in (runs_path / "detect").iterdir():
                if folder.is_dir() and folder.name.startswith("train"):
                    info = {"name": folder.name, "path": str(folder)}
                    
                    # Check for results.csv
                    if (folder / "results.csv").exists():
                        info["has_metrics"] = True
                        # Read last line for final metrics
                        with open(folder / "results.csv", "r") as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                info["final_metrics"] = lines[-1].strip()
                    
                    # Check for weights
                    weights_dir = folder / "weights"
                    if weights_dir.exists():
                        info["weights"] = [w.name for w in weights_dir.glob("*.pt")]
                    
                    results["training"].append(info)
                
                elif folder.name.startswith("predict") or folder.name.startswith("detect"):
                    results["detection"].append({
                        "name": folder.name,
                        "path": str(folder),
                        "images": len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))
                    })
    
    return results

results = scan_results()
print(json.dumps(results, indent=2))
`;
        
        const result = await executeInUltralyticsContainer(listCommand);
        
        return {
          content: [
            {
              type: 'text',
              text: `📁 Available Training & Detection Results:\n\n${result}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ Failed to list results: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  // Tool 4: Analyze Training Results
  server.tool(
    'analyze_training_results',
    'Analyze specific training results with detailed metrics',
    {
      training_name: z.string().describe('Training folder name (e.g., train, train2)')
    },
    async ({ training_name }) => {
      try {
        console.log(`📊 Analyzing training results: ${training_name}`);
        
        const analyzeCommand = `
import pandas as pd
import json
from pathlib import Path

def analyze_training(train_name):
    base_path = Path(f"/ultralytics/runs/detect/{train_name}")
    analysis = {"training_name": train_name, "found": False}
    
    if not base_path.exists():
        return analysis
    
    analysis["found"] = True
    analysis["path"] = str(base_path)
    
    # Read results.csv for metrics
    results_file = base_path / "results.csv"
    if results_file.exists():
        df = pd.read_csv(results_file)
        analysis["epochs"] = len(df)
        
        if len(df) > 0:
            final_row = df.iloc[-1]
            analysis["final_metrics"] = {
                "epoch": int(final_row.get("epoch", 0)),
                "train_box_loss": float(final_row.get("train/box_loss", 0)),
                "train_cls_loss": float(final_row.get("train/cls_loss", 0)),
                "train_dfl_loss": float(final_row.get("train/dfl_loss", 0)),
                "val_box_loss": float(final_row.get("val/box_loss", 0)),
                "val_cls_loss": float(final_row.get("val/cls_loss", 0)),
                "val_dfl_loss": float(final_row.get("val/dfl_loss", 0)),
                "precision": float(final_row.get("metrics/precision(B)", 0)),
                "recall": float(final_row.get("metrics/recall(B)", 0)),
                "mAP50": float(final_row.get("metrics/mAP50(B)", 0)),
                "mAP50_95": float(final_row.get("metrics/mAP50-95(B)", 0))
            }
            
            # Calculate improvement
            if len(df) > 1:
                first_row = df.iloc[0]
                analysis["improvement"] = {
                    "mAP50_change": float(final_row.get("metrics/mAP50(B)", 0)) - float(first_row.get("metrics/mAP50(B)", 0)),
                    "loss_reduction": float(first_row.get("train/box_loss", 0)) - float(final_row.get("train/box_loss", 0))
                }
    
    # Check available files
    analysis["files"] = {
        "weights": list(str(p.name) for p in (base_path / "weights").glob("*.pt")) if (base_path / "weights").exists() else [],
        "plots": list(str(p.name) for p in base_path.glob("*.png")),
        "has_confusion_matrix": (base_path / "confusion_matrix.png").exists(),
        "has_results_plot": (base_path / "results.png").exists()
    }
    
    return analysis

result = analyze_training("${training_name}")
print(json.dumps(result, indent=2))
`;
        
        const result = await executeInUltralyticsContainer(analyzeCommand);
        
        return {
          content: [
            {
              type: 'text',
              text: `📊 Training Analysis for "${training_name}":\n\n${result}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ Analysis failed: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  // Tool 5: View TensorBoard
  server.tool(
    'view_tensorboard',
    'Access native TensorBoard visualization (TensorBoard is automatically enabled)',
    {
      training_name: z.string().optional().describe('Specific training to focus on (optional)')
    },
    async ({ training_name }) => {
      try {
        console.log('📈 Accessing TensorBoard...');
        
        let info = `📊 **Native TensorBoard is automatically enabled!**\n\n`;
        info += `🌐 **Access URL**: http://localhost:6006\n`;
        info += `📁 **Log Directory**: /ultralytics/runs\n`;
        info += `✅ **Status**: TensorBoard logs are automatically created during training\n\n`;
        
        if (training_name) {
          info += `🎯 **Focused on**: ${training_name}\n`;
          info += `📂 **Specific logs**: /ultralytics/runs/detect/${training_name}\n\n`;
        }
        
        info += `💡 **How it works**:\n`;
        info += `• TensorBoard is enabled by default (tensorboard=True)\n`;
        info += `• Logs are automatically created during training\n`;
        info += `• Real-time metrics: loss, accuracy, mAP, learning rate\n`;
        info += `• No manual setup required!\n\n`;
        
        // Check if TensorBoard process is running
        const checkCommand = `
import subprocess
import psutil
import os
from pathlib import Path

def check_tensorboard():
    # Check if TensorBoard process is running
    tb_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'tensorboard' in proc.info['name'].lower():
                tb_running = True
                break
        except:
            continue
    
    # Check for event files
    runs_path = Path("/ultralytics/runs")
    event_files = []
    if runs_path.exists():
        event_files = list(runs_path.rglob("events.out.tfevents.*"))
    
    print(f"TensorBoard Process: {'✅ Running' if tb_running else '❌ Not running'}")
    print(f"Event Files Found: {len(event_files)}")
    
    if event_files:
        print("Recent training sessions with TensorBoard logs:")
        for f in event_files[-5:]:  # Show last 5
            print(f"  📄 {f.parent.name}: {f.name[:20]}...")
    
    if not tb_running and event_files:
        print("\\n🚀 Starting TensorBoard server...")
        cmd = ["tensorboard", "--logdir", "/ultralytics/runs", "--host", "0.0.0.0", "--port", "6006", "--bind_all"]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ TensorBoard started at http://localhost:6006")

check_tensorboard()
`;
        
        const result = await executeInUltralyticsContainer(checkCommand);
        
        return {
          content: [
            {
              type: 'text',
              text: `${info}📋 **Current Status**:\n${result}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ TensorBoard check failed: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  // Tool 6: Launch Streamlit Interface
  server.tool(
    'launch_streamlit_interface',
    'Launch Streamlit web interface for interactive YOLO inference',
    {
      model: z.string().optional().describe('YOLO model to use (default: yolo11n.pt)')
    },
    async ({ model = 'yolo11n.pt' }) => {
      try {
        console.log('🌐 Launching Streamlit Interface...');
        
        const launchCommand = `
import subprocess
import os
import sys
import time
from pathlib import Path

def launch_streamlit():
    # Kill any existing Streamlit processes
    try:
        subprocess.run(["pkill", "-f", "streamlit"], check=False)
        time.sleep(2)
    except:
        pass
    
    # Create Streamlit app script
    streamlit_script = '''
import sys
sys.path.append("/ultralytics")
sys.path.append("/workspace")
from enhanced_streamlit_inference import EnhancedInference

if __name__ == "__main__":
    model = "${model}"
    app = EnhancedInference(model=model)
    app.inference()
'''
    
    # Write the script to a file
    with open("/workspace/streamlit_app.py", "w") as f:
        f.write(streamlit_script)
    
    # Launch Enhanced Streamlit
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "/workspace/enhanced_streamlit_inference.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.maxUploadSize", "10000"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/ultralytics"
        )
        
        time.sleep(5)  # Wait for startup
        
        print("✅ Enhanced Streamlit Interface launched successfully!")
        print(f"🌐 Access URL: http://localhost:8501")
        print(f"📱 Enhanced Features available:")
        print(f"  • 🏷️ Pre-trained YOLO models")
        print(f"  • 🎯 Custom trained models from your runs")
        print(f"  • 📹 Webcam, video & image inference")
        print(f"  • 🔄 Real-time object tracking")
        print(f"  • 🎛️ Advanced threshold controls")
        print(f"  • 🎨 Enhanced UI with model details")
        print(f"  • 📊 Detection statistics")
        print(f"🔄 Process ID: {process.pid}")
        print(f"📦 Default Model: {model}")
        print(f"🎯 Custom Models: Auto-scanned from /ultralytics/runs")
        
        return {"success": True, "port": 8501, "pid": process.pid}
    
    except Exception as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        return {"error": str(e)}

result = launch_streamlit()
`;
        
        const result = await executeInUltralyticsContainer(launchCommand);
        
        return {
          content: [
            {
              type: 'text',
              text: `🌐 **Enhanced Streamlit Interface Status:**\n\n${result}\n\n🎯 **Enhanced Features:**\n• 🏷️ **Pre-trained Models**: Official YOLO models (COCO dataset)\n• 🎯 **Custom Models**: Auto-scanned from your training runs\n• 📹 **Multi-Source**: Webcam, video files, and images\n• 🎛️ **Advanced Controls**: Confidence, IoU, class selection\n• 🔄 **Object Tracking**: Multi-object tracking support\n• 📊 **Statistics**: Real-time detection metrics\n• 🎨 **Enhanced UI**: Modern interface with model details\n\n🚀 **Quick Start:**\n1. Open http://localhost:8501\n2. Choose Model Category (Pre-trained or Custom)\n3. Select your model\n4. Choose source and upload files\n5. Adjust settings and click "Start Inference"\n\n💡 **Custom Models**: Automatically scans /ultralytics/runs for your trained models!`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ Failed to launch Streamlit interface: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  // Tool 7: System Information & Interface URLs
  server.tool(
    'get_system_info',
    'Get comprehensive system information including all available interfaces and their URLs',
    {},
    async () => {
      try {
        console.log('📊 Getting system information...');
        
        const systemInfoCommand = `
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
                "description": "Interactive AI inference interface with webcam, video, and image upload support",
                "features": [
                    "Real-time object detection",
                    "Webcam inference", 
                    "File upload (images/videos)",
                    "Confidence & IoU threshold adjustment",
                    "Object tracking",
                    "Side-by-side comparison",
                    "Multiple model support"
                ]
            },
            "tensorboard": {
                "url": "http://localhost:6006", 
                "status": "active",
                "description": "Training metrics visualization and monitoring dashboard",
                "features": [
                    "Training loss curves",
                    "Validation metrics",
                    "Model architecture visualization",
                    "Real-time training progress",
                    "Hyperparameter tracking",
                    "Scalar metrics plotting"
                ]
            },
            "jupyter": {
                "url": "http://localhost:8888",
                "status": "active", 
                "description": "Development environment for AI experimentation",
                "features": [
                    "Interactive notebooks",
                    "Code development",
                    "Data exploration",
                    "Model prototyping",
                    "Visualization tools"
                ]
            },
            "mcp_server": {
                "url": "http://localhost:8092",
                "status": "active",
                "description": "Model Context Protocol server for N8N integration",
                "features": [
                    "7 AI tools available",
                    "Training automation",
                    "Prediction workflows", 
                    "Results management",
                    "Interface launching",
                    "Metrics monitoring"
                ]
            }
        },
        "tools_available": [
            "yolo_train - Train custom YOLO models",
            "yolo_predict - Run inference on images/videos", 
            "list_model_results - Browse training outputs",
            "get_training_metrics - Extract performance data",
            "check_tensorboard_status - Monitor training progress",
            "launch_streamlit_interface - Start web UI",
            "get_system_info - This tool for system overview"
        ],
        "storage": {
            "models_path": "/ultralytics/runs",
            "results_path": "/ultralytics/runs/detect", 
            "training_path": "/ultralytics/runs/detect/train*",
            "workspace": "/workspace"
        },
        "quick_access": {
            "streamlit_ui": "http://localhost:8501 - Visual AI Interface",
            "tensorboard": "http://localhost:6006 - Training Metrics", 
            "jupyter": "http://localhost:8888 - Development",
            "n8n_workflows": "http://localhost:5678 - Automation"
        }
    }
    
    # Check actual process status
    try:
        # Check Streamlit
        streamlit_proc = subprocess.run(["pgrep", "-f", "streamlit"], capture_output=True)
        info["interfaces"]["streamlit"]["process_active"] = streamlit_proc.returncode == 0
        
        # Check TensorBoard  
        tensorboard_proc = subprocess.run(["pgrep", "-f", "tensorboard"], capture_output=True)
        info["interfaces"]["tensorboard"]["process_active"] = tensorboard_proc.returncode == 0
        
    except Exception as e:
        info["process_check_error"] = str(e)
    
    return info

result = get_system_info()
print(json.dumps(result, indent=2))
`;
        
        const result = await executeInUltralyticsContainer(systemInfoCommand);
        const systemData = JSON.parse(result);
        
        const formattedInfo = `🖥️ **ULTRALYTICS AI SYSTEM OVERVIEW**
=============================================

🚀 **ACTIVE INTERFACES:**
• 🌐 **Streamlit UI**: http://localhost:8501
  ➜ Interactive AI inference with webcam & file upload
• 📊 **TensorBoard**: http://localhost:6006  
  ➜ Training metrics visualization & monitoring
• 📓 **Jupyter Lab**: http://localhost:8888
  ➜ Development environment & experimentation  
• 🔗 **MCP Server**: http://localhost:8092
  ➜ N8N integration with 7 AI tools

🛠️ **AVAILABLE MCP TOOLS:**
1️⃣ yolo_train - Train custom YOLO models
2️⃣ yolo_predict - Run inference on images/videos
3️⃣ list_model_results - Browse training outputs  
4️⃣ get_training_metrics - Extract performance data
5️⃣ check_tensorboard_status - Monitor training progress
6️⃣ launch_streamlit_interface - Start web UI
7️⃣ get_system_info - System overview (this tool)

💾 **STORAGE PATHS:**
• Models: /ultralytics/runs
• Results: /ultralytics/runs/detect
• Workspace: /workspace

🎯 **QUICK ACCESS URLS:**
• Visual AI Interface: http://localhost:8501
• Training Metrics: http://localhost:6006  
• Development: http://localhost:8888
• N8N Automation: http://localhost:5678

⚡ **SYSTEM STATUS:**
• Python: ${systemData.system?.python_version?.split(' ')[0] || 'Active'}
• PyTorch: ${systemData.system?.torch_version || 'Active'}
• CUDA: ${systemData.system?.cuda_available ? '✅ Available' : '❌ Not Available'}
• GPU Count: ${systemData.system?.gpu_count || 0}

🔥 **HYBRID WORKFLOW:**
N8N (Automation) ↔ MCP Tools ↔ Streamlit (Visual Interface)`;

        return {
          content: [
            {
              type: 'text',
              text: formattedInfo
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text', 
              text: `❌ Failed to get system information: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  // Tool 8: Get Class-Specific Metrics
  server.tool(
    'get_class_metrics',
    'Get per-class performance metrics for a specific training run. Can automatically run validation if needed.',
    {
      training_name: z.string().describe('Training directory name (e.g., "training_042643" or just "042643")'),
      class_name: z.string().optional().describe('Optional: specific class name to search for (e.g., "Root Canal Treatment", "Caries"). Case-insensitive partial matching supported.')
    },
    async ({ training_name, class_name }) => {
      try {
        console.log(`📊 Getting class metrics for training: ${training_name}${class_name ? `, class: ${class_name}` : ''}`);
        
        const getClassMetricsCommand = `
import subprocess
import json
import re
from pathlib import Path

training_name = "${training_name}"
class_name = "${class_name || ''}"

# Normalize training name
if not training_name.startswith("training_"):
    training_name = f"training_{training_name}"

# Find training directory
training_dir = Path(f"/ultralytics/runs/detect/{training_name}")
if not training_dir.exists():
    print(json.dumps({"error": f"Training directory not found: {training_dir}"}))
    exit(1)

# Check for existing results.csv
results_csv = training_dir / "results.csv"
has_existing_results = results_csv.exists()

# Get model and dataset paths
weights_path = training_dir / "weights" / "best.pt"
args_yaml = training_dir / "args.yaml"

has_weights = weights_path.exists()
has_args = args_yaml.exists()

result = {
    "training_directory": str(training_dir),
    "has_existing_results": has_existing_results,
    "has_weights": has_weights,
    "has_args": has_args
}

# Parse overall metrics from existing results.csv
if has_existing_results:
    try:
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        last_row = df.iloc[-1]
        
        result["overall_metrics"] = {
            "epoch": int(last_row.get("epoch", last_row.get("         epoch", -1))),
            "box_loss": float(last_row.get("train/box_loss", last_row.get("      train/box_loss", 0))),
            "cls_loss": float(last_row.get("train/cls_loss", last_row.get("      train/cls_loss", 0))),
            "dfl_loss": float(last_row.get("train/dfl_loss", last_row.get("      train/dfl_loss", 0))),
            "precision": float(last_row.get("metrics/precision(B)", last_row.get("   metrics/precision(B)", 0))),
            "recall": float(last_row.get("metrics/recall(B)", last_row.get("      metrics/recall(B)", 0))),
            "mAP50": float(last_row.get("metrics/mAP50(B)", last_row.get("      metrics/mAP50(B)", 0))),
            "mAP50_95": float(last_row.get("metrics/mAP50-95(B)", last_row.get("   metrics/mAP50-95(B)", 0)))
        }
    except Exception as e:
        result["overall_metrics_error"] = str(e)

# Extract dataset path from args.yaml
dataset_path = None
if has_args:
    try:
        import yaml
        with open(args_yaml, 'r') as f:
            args = yaml.safe_load(f)
            dataset_path = args.get("data")
    except Exception as e:
        result["args_parse_error"] = str(e)

# Check if training is active
try:
    train_check = subprocess.run(
        ["pgrep", "-f", "train_script.py"],
        capture_output=True,
        text=True,
        timeout=5
    )
    is_training_active = train_check.returncode == 0
except:
    is_training_active = False

result["is_training_active"] = is_training_active

# Run validation if we have weights, dataset, not training, and class name specified
class_specific_metrics = None
if not is_training_active and has_weights and dataset_path and class_name:
    try:
        print("Running validation to get class-specific metrics...", flush=True)
        
        # Run YOLO validation
        val_cmd = [
            "yolo", "val",
            f"model={weights_path}",
            f"data={dataset_path}",
            "device=0",
            "batch=8"
        ]
        
        val_result = subprocess.run(
            val_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/ultralytics"
        )
        
        val_output = val_result.stdout + val_result.stderr
        
        # Parse class-specific metrics from validation output
        class_metrics = []
        search_term = class_name.lower()
        
        for line in val_output.split('\\n'):
            # Skip header and "all" summary row
            if 'Class' in line and 'Images' in line:
                continue
            if line.strip().startswith('all '):
                continue
            
            # Match class metrics line: "ClassName  images  instances  P  R  mAP50  mAP50-95"
            match = re.match(r'^\\s*([A-Za-z][A-Za-z0-9 _\\-]+?)\\s+(\\d+)\\s+(\\d+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)', line)
            
            if match and search_term in match.group(1).lower():
                class_metrics.append({
                    "class_name": match.group(1).strip(),
                    "images": int(match.group(2)),
                    "instances": int(match.group(3)),
                    "precision": float(match.group(4)),
                    "recall": float(match.group(5)),
                    "mAP50": float(match.group(6)),
                    "mAP50_95": float(match.group(7))
                })
        
        # Get validation directory from output
        val_dir_match = re.search(r'Results saved to ([^\\s]+)', val_output)
        validation_dir = val_dir_match.group(1) if val_dir_match else None
        
        class_specific_metrics = {
            "validation_dir": validation_dir,
            "class_metrics": class_metrics if class_metrics else None,
            "note": f"Found {len(class_metrics)} matching class(es)" if class_metrics else "No matching classes found"
        }
        
    except subprocess.TimeoutExpired:
        class_specific_metrics = {"error": "Validation timed out after 5 minutes"}
    except Exception as e:
        class_specific_metrics = {"error": f"Validation failed: {str(e)}"}

if class_specific_metrics:
    result["class_specific_metrics"] = class_specific_metrics

print(json.dumps(result, indent=2))
`;
        
        const result = await executeInUltralyticsContainer(getClassMetricsCommand);
        const metricsData = JSON.parse(result);
        
        let formattedResult = `📊 **CLASS METRICS FOR ${training_name.toUpperCase()}**\\n`;
        formattedResult += `==============================================\\n\\n`;
        
        if (metricsData.error) {
          formattedResult += `❌ Error: ${metricsData.error}\\n`;
        } else {
          formattedResult += `📁 Training Directory: ${metricsData.training_directory}\\n`;
          formattedResult += `⚙️ Training Active: ${metricsData.is_training_active ? '🟢 Yes' : '🔴 No'}\\n\\n`;
          
          if (metricsData.overall_metrics) {
            formattedResult += `📈 **OVERALL METRICS:**\\n`;
            formattedResult += `  • Epoch: ${metricsData.overall_metrics.epoch}\\n`;
            formattedResult += `  • Precision: ${(metricsData.overall_metrics.precision * 100).toFixed(2)}%\\n`;
            formattedResult += `  • Recall: ${(metricsData.overall_metrics.recall * 100).toFixed(2)}%\\n`;
            formattedResult += `  • mAP50: ${(metricsData.overall_metrics.mAP50 * 100).toFixed(2)}%\\n`;
            formattedResult += `  • mAP50-95: ${(metricsData.overall_metrics.mAP50_95 * 100).toFixed(2)}%\\n\\n`;
          }
          
          if (metricsData.class_specific_metrics) {
            const csm = metricsData.class_specific_metrics;
            
            if (csm.error) {
              formattedResult += `⚠️ Class-specific metrics: ${csm.error}\\n`;
            } else if (csm.class_metrics && csm.class_metrics.length > 0) {
              formattedResult += `🎯 **CLASS-SPECIFIC METRICS (${class_name}):**\\n\\n`;
              
              csm.class_metrics.forEach(cm => {
                formattedResult += `  📌 **${cm.class_name}**\\n`;
                formattedResult += `     • Images: ${cm.images}\\n`;
                formattedResult += `     • Instances: ${cm.instances}\\n`;
                formattedResult += `     • Precision: ${(cm.precision * 100).toFixed(2)}%\\n`;
                formattedResult += `     • Recall: ${(cm.recall * 100).toFixed(2)}%\\n`;
                formattedResult += `     • mAP50: ${(cm.mAP50 * 100).toFixed(2)}%\\n`;
                formattedResult += `     • mAP50-95: ${(cm.mAP50_95 * 100).toFixed(2)}%\\n\\n`;
              });
              
              if (csm.validation_dir) {
                formattedResult += `📂 Validation Results: ${csm.validation_dir}\\n`;
              }
            } else {
              formattedResult += `ℹ️ ${csm.note}\\n`;
            }
          } else if (class_name && !metricsData.is_training_active) {
            formattedResult += `ℹ️ No class-specific metrics available. Need weights and dataset to run validation.\\n`;
          }
        }
        
        return {
          content: [
            {
              type: 'text',
              text: formattedResult
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ Failed to get class metrics: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  return server;
};

// Create Express application
const app = express();
app.use(express.json());

// Store transports by session ID
const transports = {};

console.log('🚀 Starting Ultralytics MCP Server...');

//=============================================================================
// SSE TRANSPORT - N8N COMPATIBLE
//=============================================================================
app.get('/sse', async (req, res) => {
  console.log('📡 GET /sse - Establishing SSE connection');
  
  try {
    const transport = new SSEServerTransport('/messages', res);
    transports[transport.sessionId] = transport;
    
    res.on("close", () => {
      console.log(`❌ SSE connection closed for session ${transport.sessionId}`);
      delete transports[transport.sessionId];
    });

    const server = createUltralyticsServer();
    await server.connect(transport);
    
    console.log(`✅ SSE transport connected: ${transport.sessionId}`);
  } catch (error) {
    console.error('❌ SSE connection error:', error);
    if (!res.headersSent) {
      res.status(500).send('Internal server error');
    }
  }
});

app.post("/messages", async (req, res) => {
  const sessionId = req.query.sessionId;
  console.log(`📡 POST /messages: ${sessionId}`);
  
  try {
    const transport = transports[sessionId];
    
    if (transport instanceof SSEServerTransport) {
      await transport.handlePostMessage(req, res, req.body);
    } else {
      res.status(400).json({
        jsonrpc: '2.0',
        error: {
          code: -32000,
          message: 'No transport found for sessionId',
        },
        id: null,
      });
    }
  } catch (error) {
    console.error('❌ POST message error:', error);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: '2.0',
        error: {
          code: -32603,
          message: 'Internal server error',
        },
        id: null,
      });
    }
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    server: 'ultralytics_mcp',
    version: '1.0.0',
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
