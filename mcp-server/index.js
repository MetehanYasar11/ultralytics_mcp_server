#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";
import path from "path";

const execAsync = promisify(exec);

// Docker container name
const CONTAINER_NAME = "ultralytics-container";

// Helper: Execute command in Docker container
async function execInContainer(command) {
  try {
    const { stdout, stderr } = await execAsync(
      `docker exec ${CONTAINER_NAME} bash -c "${command.replace(/"/g, '\\"')}"`
    );
    return { success: true, stdout, stderr };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      stdout: error.stdout || "",
      stderr: error.stderr || "",
    };
  }
}

// Helper: Read file from container
async function readFileFromContainer(filePath) {
  const result = await execInContainer(`cat ${filePath} 2>/dev/null || echo ""`);
  return result.stdout;
}

// Helper: List directories in container
async function listDirsInContainer(dirPath) {
  const result = await execInContainer(
    `find ${dirPath} -maxdepth 1 -type d 2>/dev/null | tail -n +2 | sort`
  );
  if (!result.success) return [];
  return result.stdout
    .trim()
    .split("\n")
    .filter((line) => line.length > 0);
}

// Helper: List files in container
async function listFilesInContainer(dirPath, pattern = "*") {
  const result = await execInContainer(
    `find ${dirPath} -maxdepth 1 -type f -name "${pattern}" 2>/dev/null | sort`
  );
  if (!result.success) return [];
  return result.stdout
    .trim()
    .split("\n")
    .filter((line) => line.length > 0);
}

// Create MCP server
const server = new Server(
  {
    name: "ultralytics-training-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "start_training",
        description:
          "Start a new YOLO training job with specified parameters. Returns training job ID and status.",
        inputSchema: {
          type: "object",
          properties: {
            model_variant: {
              type: "string",
              enum: ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
              description: "YOLO model variant (n=nano, s=small, m=medium, l=large, x=xlarge)",
            },
            dataset_name: {
              type: "string",
              description:
                "Dataset name (e.g., 'YOLO_Disease', 'YOLO_Tooth', 'deneme', 'kaggle')",
            },
            epochs: {
              type: "number",
              description: "Number of training epochs (1-1000)",
              minimum: 1,
              maximum: 1000,
            },
            batch_size: {
              type: "number",
              enum: [4, 8, 16, 32, 64, 128, 256],
              description: "Batch size for training",
              default: 16,
            },
            img_size: {
              type: "number",
              enum: [320, 416, 480, 512, 640, 736, 832, 896, 960, 1024, 1152, 1280],
              description: "Input image size",
              default: 640,
            },
            device: {
              type: "string",
              enum: ["0", "cpu"],
              description: "Training device (0=GPU, cpu=CPU)",
              default: "0",
            },
            model_name: {
              type: "string",
              description: "Custom name for trained model (optional)",
            },
            transfer_learning: {
              type: "boolean",
              description: "Use transfer learning from workspace model",
              default: false,
            },
            base_model_path: {
              type: "string",
              description: "Path to base model for transfer learning (if enabled)",
            },
          },
          required: ["model_variant", "dataset_name", "epochs"],
        },
      },
      {
        name: "get_training_status",
        description:
          "Get current training status including active jobs, progress, and latest metrics",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "list_datasets",
        description: "List all available datasets (built-in and custom)",
        inputSchema: {
          type: "object",
          properties: {
            type: {
              type: "string",
              enum: ["all", "builtin", "custom"],
              description: "Filter datasets by type",
              default: "all",
            },
          },
        },
      },
      {
        name: "list_models",
        description: "List trained models in workspace with metadata",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "get_training_logs",
        description: "Get training logs from last training run",
        inputSchema: {
          type: "object",
          properties: {
            lines: {
              type: "number",
              description: "Number of lines to show from end of log",
              default: 50,
              minimum: 10,
              maximum: 500,
            },
          },
        },
      },
      {
        name: "list_training_history",
        description: "List all training runs with their details and results",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Maximum number of training runs to return",
              default: 10,
              minimum: 1,
              maximum: 100,
            },
          },
        },
      },
      {
        name: "convert_to_tensorboard",
        description:
          "Convert historical training results to TensorBoard format for visualization",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "stop_training",
        description: "Stop currently running training job",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "get_gpu_status",
        description: "Get current GPU status and utilization",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "start_training": {
        const {
          model_variant,
          dataset_name,
          epochs,
          batch_size = 16,
          img_size = 640,
          device = "0",
          model_name,
          transfer_learning = false,
          base_model_path,
        } = args;

        // Determine dataset path
        let dataset_path;
        if (dataset_name.startsWith("YOLO_")) {
          dataset_path = `/ultralytics/YOLO_MultiLevel_Datasets/${dataset_name}/data.yaml`;
        } else {
          dataset_path = `/ultralytics/custom_datasets/${dataset_name}/data.yaml`;
        }

        // Determine model path
        let selected_model_path;
        if (transfer_learning && base_model_path) {
          selected_model_path = base_model_path;
        } else {
          selected_model_path = `${model_variant}.pt`;
        }

        // Generate model name
        const final_model_name =
          model_name || `model_${dataset_name.toLowerCase().replace(/[^a-z0-9]/g, "_")}`;

        // Create training script
        const training_script = `
import os
from ultralytics import YOLO
from datetime import datetime
import shutil
from pathlib import Path

print("🚀 Starting AI Agent Training...")
print(f"Dataset: ${dataset_path}")
print(f"Base Model: ${selected_model_path}")
print(f"Epochs: ${epochs}")
print(f"Batch: ${batch_size}")
print(f"Image size: ${img_size}")
print(f"Device: ${device}")
print(f"Model name: ${final_model_name}")

# Load model
model = YOLO('${selected_model_path}')

time_str = datetime.now().strftime('%H%M%S')
training_name = f'training_{time_str}'

# Start training
results = model.train(
    data='${dataset_path}',
    epochs=${epochs},
    imgsz=${img_size},
    batch=${batch_size},
    device='${device}',
    project='/ultralytics/runs/detect',
    name=training_name,
    exist_ok=True,
    verbose=True,
    save=True,
    plots=True
)

print("✅ Training completed successfully!")
print(f"Results saved to: {results.save_dir}")

# Auto-save to workspace
try:
    workspace_models = Path("/workspace/trained_models")
    workspace_models.mkdir(exist_ok=True)
    
    weights_dir = Path(results.save_dir) / "weights"
    if weights_dir.exists():
        best_model = weights_dir / "best.pt"
        last_model = weights_dir / "last.pt"
        
        if best_model.exists():
            shutil.copy2(best_model, workspace_models / f"${final_model_name}_best.pt")
            print(f"✅ Best model saved: ${final_model_name}_best.pt")
        if last_model.exists():
            shutil.copy2(last_model, workspace_models / f"${final_model_name}_last.pt")
            print(f"✅ Last model saved: ${final_model_name}_last.pt")
except Exception as e:
    print(f"❌ Error saving to workspace: {e}")
`;

        // Write training script to container
        await execInContainer(
          `cat > /tmp/agent_train_script.py << 'EOFPYTHON'\n${training_script}\nEOFPYTHON`
        );

        // Start training in background
        await execInContainer(
          `nohup python /tmp/agent_train_script.py > /tmp/training_log.txt 2>&1 &`
        );

        // Wait a bit and check if started
        await new Promise((resolve) => setTimeout(resolve, 2000));

        const log = await readFileFromContainer("/tmp/training_log.txt");

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  success: true,
                  message: "Training started successfully",
                  config: {
                    model: model_variant,
                    dataset: dataset_name,
                    epochs,
                    batch_size,
                    img_size,
                    device,
                    model_name: final_model_name,
                  },
                  initial_log: log.split("\n").slice(0, 10).join("\n"),
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "get_training_status": {
        // Check if training process is running
        const psResult = await execInContainer(
          `ps aux | grep "agent_train_script.py" | grep -v grep || echo ""`
        );
        const isTraining = psResult.stdout.trim().length > 0;

        // Get latest training directory
        const trainDirs = await listDirsInContainer("/ultralytics/runs/detect");
        const latestTraining = trainDirs[trainDirs.length - 1];

        let status = {
          is_training: isTraining,
          latest_training: latestTraining ? path.basename(latestTraining) : null,
        };

        if (latestTraining) {
          // Try to read results.csv
          const resultsCSV = await readFileFromContainer(
            `${latestTraining}/results.csv`
          );
          if (resultsCSV) {
            const lines = resultsCSV.trim().split("\n");
            status.epochs_completed = lines.length - 1; // -1 for header
            if (lines.length > 1) {
              const lastLine = lines[lines.length - 1];
              const values = lastLine.split(",").map((v) => v.trim());
              status.current_metrics = {
                epoch: values[0],
                box_loss: values[7],
                cls_loss: values[8],
                dfl_loss: values[9],
                precision: values[10],
                recall: values[11],
                mAP50: values[12],
                mAP50_95: values[13],
              };
            }
          }

          // Read args.yaml
          const argsYAML = await readFileFromContainer(`${latestTraining}/args.yaml`);
          if (argsYAML) {
            status.config = argsYAML;
          }
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(status, null, 2),
            },
          ],
        };
      }

      case "list_datasets": {
        const { type = "all" } = args;
        const datasets = { builtin: [], custom: [] };

        if (type === "all" || type === "builtin") {
          const builtinDirs = await listDirsInContainer(
            "/ultralytics/YOLO_MultiLevel_Datasets"
          );
          for (const dir of builtinDirs) {
            const name = path.basename(dir);
            if (name.startsWith("YOLO_")) {
              const yamlPath = `${dir}/data.yaml`;
              const yamlContent = await readFileFromContainer(yamlPath);
              datasets.builtin.push({
                name,
                path: yamlPath,
                has_yaml: yamlContent.length > 0,
              });
            }
          }
        }

        if (type === "all" || type === "custom") {
          const customDirs = await listDirsInContainer("/ultralytics/custom_datasets");
          for (const dir of customDirs) {
            const name = path.basename(dir);
            const yamlPath = `${dir}/data.yaml`;
            const yamlContent = await readFileFromContainer(yamlPath);
            datasets.custom.push({
              name,
              path: yamlPath,
              has_yaml: yamlContent.length > 0,
            });
          }
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(datasets, null, 2),
            },
          ],
        };
      }

      case "list_models": {
        const modelFiles = await listFilesInContainer("/workspace/trained_models", "*.pt");
        const models = [];

        for (const modelPath of modelFiles) {
          const name = path.basename(modelPath);
          const statResult = await execInContainer(
            `stat -c "%s %Y" ${modelPath} 2>/dev/null || echo ""`
          );
          if (statResult.stdout.trim()) {
            const [size, mtime] = statResult.stdout.trim().split(" ");
            models.push({
              name,
              path: modelPath,
              size_mb: (parseInt(size) / (1024 * 1024)).toFixed(2),
              modified: new Date(parseInt(mtime) * 1000).toISOString(),
            });
          }
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ models, count: models.length }, null, 2),
            },
          ],
        };
      }

      case "get_training_logs": {
        const { lines = 50 } = args;
        const log = await readFileFromContainer("/tmp/training_log.txt");
        const logLines = log.split("\n");
        const lastLines = logLines.slice(-lines).join("\n");

        return {
          content: [
            {
              type: "text",
              text: lastLines || "No training logs found",
            },
          ],
        };
      }

      case "list_training_history": {
        const { limit = 10 } = args;
        const trainDirs = await listDirsInContainer("/ultralytics/runs/detect");
        const history = [];

        for (const dir of trainDirs.slice(-limit)) {
          const name = path.basename(dir);
          if (!name.startsWith("training_") && !name.startsWith("custom_")) continue;

          const argsYAML = await readFileFromContainer(`${dir}/args.yaml`);
          const resultsCSV = await readFileFromContainer(`${dir}/results.csv`);

          const item = { name, path: dir };

          if (argsYAML) {
            // Parse YAML for key info
            const modelMatch = argsYAML.match(/model:\s*(.+)/);
            const dataMatch = argsYAML.match(/data:\s*(.+)/);
            const epochsMatch = argsYAML.match(/epochs:\s*(\d+)/);

            if (modelMatch) item.model = modelMatch[1].trim();
            if (dataMatch) item.dataset = dataMatch[1].trim();
            if (epochsMatch) item.epochs = parseInt(epochsMatch[1]);
          }

          if (resultsCSV) {
            const lines = resultsCSV.trim().split("\n");
            item.epochs_completed = lines.length - 1;
            if (lines.length > 1) {
              const lastLine = lines[lines.length - 1];
              const values = lastLine.split(",").map((v) => v.trim());
              item.final_metrics = {
                mAP50: parseFloat(values[12]),
                mAP50_95: parseFloat(values[13]),
                precision: parseFloat(values[10]),
                recall: parseFloat(values[11]),
              };
            }
          }

          history.push(item);
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ history, count: history.length }, null, 2),
            },
          ],
        };
      }

      case "convert_to_tensorboard": {
        const result = await execInContainer(
          `cd /ultralytics && python /ultralytics/convert_to_tensorboard.py 2>&1`
        );

        return {
          content: [
            {
              type: "text",
              text: result.stdout || result.stderr || "Conversion completed",
            },
          ],
        };
      }

      case "stop_training": {
        await execInContainer(`pkill -f agent_train_script.py`);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                message: "Training stopped",
              }),
            },
          ],
        };
      }

      case "get_gpu_status": {
        const result = await execInContainer(
          `nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null || echo "No GPU"`
        );

        if (result.stdout.includes("No GPU")) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ available: false, message: "No GPU available" }),
              },
            ],
          };
        }

        const [name, temp, gpu_util, mem_util, mem_used, mem_total, power_draw, power_limit] =
          result.stdout.trim().split(", ");

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  available: true,
                  name,
                  temperature: parseInt(temp),
                  gpu_utilization: parseInt(gpu_util),
                  memory_utilization: parseInt(mem_util),
                  memory_used_mb: parseInt(mem_used),
                  memory_total_mb: parseInt(mem_total),
                  power_draw_w: parseFloat(power_draw),
                  power_limit_w: parseFloat(power_limit),
                },
                null,
                2
              ),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            {
              error: error.message,
              tool: name,
              arguments: args,
            },
            null,
            2
          ),
        },
      ],
      isError: true,
    };
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Ultralytics Training MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
