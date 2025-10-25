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
  const result = await execInContainer(`cat "${filePath}" 2>/dev/null || echo ""`);
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
    .filter((d) => d.length > 0);
}

// Helper: Generate all hyperparameter combinations
function generateCombinations(modelVariants, hyperparameters, datasetName) {
  const combinations = [];
  const {
    learning_rates = [0.01],
    batch_sizes = [16],
    epochs_list = [100],
    optimizers = ["auto"],
    img_sizes = [640],
  } = hyperparameters;

  for (const model of modelVariants) {
    for (const lr of learning_rates) {
      for (const batch of batch_sizes) {
        for (const epochs of epochs_list) {
          for (const optimizer of optimizers) {
            for (const imgSize of img_sizes) {
              combinations.push({
                model_variant: model,
                dataset_name: datasetName,
                learning_rate: lr,
                batch_size: batch,
                epochs: epochs,
                optimizer: optimizer,
                img_size: imgSize,
              });
            }
          }
        }
      }
    }
  }
  return combinations;
}

// Helper: Read grid search queue
async function readGridSearchQueue(searchId) {
  try {
    const queueFile = `/tmp/grid_search_${searchId}.json`;
    const content = await readFileFromContainer(queueFile);
    if (!content) return null;
    return JSON.parse(content);
  } catch (error) {
    return null;
  }
}

// Helper: Save grid search queue
async function saveGridSearchQueue(searchId, queueData) {
  const queueFile = `/tmp/grid_search_${searchId}.json`;
  const jsonStr = JSON.stringify(queueData, null, 2).replace(/"/g, '\\"');
  await execInContainer(`echo "${jsonStr}" > ${queueFile}`);
}

// Helper: Estimate training time
function estimateTrainingTime(combination) {
  // Rough estimates in hours based on model size and epochs
  const modelTimes = {
    yolo11n: 0.5,
    yolo11s: 1.0,
    yolo11m: 2.0,
    yolo11l: 3.5,
    yolo11x: 5.0,
  };
  const baseTime = modelTimes[combination.model_variant] || 2.0;
  return (baseTime * combination.epochs) / 100; // Normalize to per 100 epochs
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
      {
        name: "start_grid_search",
        description:
          "Start an automated grid search to find optimal hyperparameters. Tests all combinations of specified parameters and returns best performing model.",
        inputSchema: {
          type: "object",
          properties: {
            model_variants: {
              type: "array",
              items: {
                type: "string",
                enum: ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
              },
              description: "YOLO model variants to test",
              default: ["yolo11n", "yolo11s"],
            },
            dataset_name: {
              type: "string",
              description: "Dataset to use for all experiments",
            },
            hyperparameters: {
              type: "object",
              properties: {
                learning_rates: {
                  type: "array",
                  items: { type: "number" },
                  description: "Learning rates to test",
                  default: [0.001, 0.01],
                },
                batch_sizes: {
                  type: "array",
                  items: { type: "number" },
                  description: "Batch sizes to test",
                  default: [8, 16],
                },
                epochs_list: {
                  type: "array",
                  items: { type: "number" },
                  description: "Epoch counts to test",
                  default: [50],
                },
                optimizers: {
                  type: "array",
                  items: { type: "string" },
                  description: "Optimizers to test",
                  default: ["auto"],
                },
                img_sizes: {
                  type: "array",
                  items: { type: "number" },
                  description: "Image sizes to test",
                  default: [640],
                },
              },
            },
            max_parallel_jobs: {
              type: "number",
              description: "Maximum number of parallel training jobs (limited by GPU)",
              default: 1,
              minimum: 1,
              maximum: 4,
            },
          },
          required: ["model_variants", "dataset_name"],
        },
      },
      {
        name: "get_grid_search_status",
        description:
          "Get status and results of an ongoing or completed grid search",
        inputSchema: {
          type: "object",
          properties: {
            search_id: {
              type: "string",
              description: "Grid search ID returned by start_grid_search",
            },
          },
          required: ["search_id"],
        },
      },
      {
        name: "stop_grid_search",
        description: "Stop a running grid search and return best results so far",
        inputSchema: {
          type: "object",
          properties: {
            search_id: {
              type: "string",
              description: "Grid search ID to stop",
            },
          },
          required: ["search_id"],
        },
      },
      {
        name: "get_class_metrics",
        description:
          "Get per-class performance metrics for a specific training run. Useful for analyzing individual class performance like 'Root Canal Treatment', 'Caries', etc.",
        inputSchema: {
          type: "object",
          properties: {
            training_name: {
              type: "string",
              description: "Training directory name (e.g., 'training_042643')",
            },
            class_name: {
              type: "string",
              description: "Class name to search for (e.g., 'Root Canal Treatment', 'Caries'). Case-insensitive partial matching supported.",
            },
          },
          required: ["training_name"],
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
        // Get latest training directory by modification time (not alphabetically)
        const lsResult = await execInContainer(
          `find /ultralytics/runs/detect -maxdepth 1 -type d \\( -name "training_*" -o -name "custom_*" \\) -printf "%T@ %p\\n" | sort -rn | head -1 | cut -d' ' -f2 | xargs basename`
        );
        const latestTrainingName = lsResult.stdout.trim();
        const latestTraining = latestTrainingName ? `/ultralytics/runs/detect/${latestTrainingName}` : null;

        let status = {
          is_training: false,
          latest_training: latestTrainingName || null,
        };

        if (latestTraining) {
          // Check training log to see if training is active
          const logContent = await readFileFromContainer("/tmp/training_log.txt");
          
          // Get last 500 lines for better detection
          const logLines = logContent.trim().split("\n");
          const recentLines = logLines.slice(-500).join("\n");
          
          // Check if training is active based on log patterns
          const hasRecentEpochInfo = /\d+\/\d+\s+[\d.]+G/.test(recentLines); // "8/50  9.07G" pattern
          const hasProgressBars = /\|\s*\d+\/\d+/.test(recentLines); // Progress bar pattern
          const hasIterationSpeed = /it\/s/.test(recentLines); // "4.50it/s" pattern
          const hasStartedRecently = recentLines.includes("Starting training") && 
                                     !recentLines.includes("Training completed");
          
          const isActiveLog = hasRecentEpochInfo || hasProgressBars || 
                             hasIterationSpeed || hasStartedRecently;

          // Also check for Python training processes
          const psResult = await execInContainer(
            `ps aux | grep -E "(train_script|ultralytics|YOLO)" | grep -v grep || echo ""`
          );
          const hasTrainingProcess = psResult.stdout.trim().length > 0;

          status.is_training = isActiveLog || hasTrainingProcess;

          // Try to read results.csv
          const resultsCSV = await readFileFromContainer(
            `${latestTraining}/results.csv`
          );
          
          if (resultsCSV) {
            const lines = resultsCSV.trim().split("\n");
            
            // If training is active, find the last COMPLETED epoch from logs
            let lastCompletedEpoch = lines.length - 1; // Default to last line
            
            if (status.is_training) {
              // Look for last completed validation in logs (lines with "all" class summary)
              const validationMatches = recentLines.match(/all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/g);
              if (validationMatches && validationMatches.length > 0) {
                // Get the last validation result
                const lastValidation = validationMatches[validationMatches.length - 1];
                const valMatch = lastValidation.match(/all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/);
                if (valMatch) {
                  const [_, precision, recall, mAP50, mAP50_95] = valMatch;
                  
                  // Find which epoch has these exact metrics in CSV
                  for (let i = 1; i < lines.length; i++) {
                    const csvValues = lines[i].split(",").map(v => v.trim());
                    const csvMAP50 = parseFloat(csvValues[12]);
                    const csvMAP50_95 = parseFloat(csvValues[13]);
                    
                    if (Math.abs(csvMAP50 - parseFloat(mAP50)) < 0.001 && 
                        Math.abs(csvMAP50_95 - parseFloat(mAP50_95)) < 0.001) {
                      lastCompletedEpoch = i;
                      break;
                    }
                  }
                }
              }
            }
            
            status.epochs_completed = lastCompletedEpoch;
            
            if (lines.length > lastCompletedEpoch) {
              const targetLine = lines[lastCompletedEpoch];
              const values = targetLine.split(",").map((v) => v.trim());
              status.current_metrics = {
                epoch: parseInt(values[0]),
                box_loss: parseFloat(values[2]), // train/box_loss
                cls_loss: parseFloat(values[3]), // train/cls_loss
                dfl_loss: parseFloat(values[4]), // train/dfl_loss
                precision: parseFloat(values[5]), // metrics/precision(B)
                recall: parseFloat(values[6]), // metrics/recall(B)
                mAP50: parseFloat(values[7]), // metrics/mAP50(B)
                mAP50_95: parseFloat(values[8]), // metrics/mAP50-95(B)
              };
            }
          }

          // Read args.yaml for config
          const argsYAML = await readFileFromContainer(`${latestTraining}/args.yaml`);
          if (argsYAML) {
            // Parse key config values
            const modelMatch = argsYAML.match(/model:\s*(.+)/);
            const dataMatch = argsYAML.match(/data:\s*(.+)/);
            const epochsMatch = argsYAML.match(/epochs:\s*(\d+)/);
            const batchMatch = argsYAML.match(/batch:\s*(\d+)/);
            const deviceMatch = argsYAML.match(/device:\s*'?(\d+|cpu)'?/);

            status.training_config = {
              model: modelMatch ? modelMatch[1].trim() : null,
              dataset: dataMatch ? dataMatch[1].trim() : null,
              total_epochs: epochsMatch ? parseInt(epochsMatch[1]) : null,
              batch_size: batchMatch ? parseInt(batchMatch[1]) : null,
              device: deviceMatch ? deviceMatch[1].trim() : null,
            };

            // Calculate progress percentage
            if (status.epochs_completed && status.training_config.total_epochs) {
              status.progress_percent = 
                ((status.epochs_completed / status.training_config.total_epochs) * 100).toFixed(1);
            }
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

      case "start_grid_search": {
        const {
          model_variants = ["yolo11n", "yolo11s"],
          dataset_name,
          hyperparameters = {},
          max_parallel_jobs = 1,
        } = args;

        // Generate all combinations
        const combinations = generateCombinations(
          model_variants,
          hyperparameters,
          dataset_name
        );

        // Create search ID
        const searchId = `gs_${Date.now()}`;

        // Calculate total estimated time
        const totalTimeHours = combinations.reduce(
          (sum, combo) => sum + estimateTrainingTime(combo),
          0
        );

        // Initialize queue
        const queue = {
          id: searchId,
          status: "running",
          total_combinations: combinations.length,
          completed: 0,
          running: [],
          pending: combinations,
          results: [],
          best_model: null,
          started_at: new Date().toISOString(),
          max_parallel: max_parallel_jobs,
        };

        // Save queue
        await saveGridSearchQueue(searchId, queue);

        // Create and start worker script
        const workerScript = `#!/usr/bin/env python3
import json
import time
import subprocess
import sys
from pathlib import Path

QUEUE_FILE = "/tmp/grid_search_${searchId}.json"
MAX_PARALLEL = ${max_parallel_jobs}

def load_queue():
    try:
        with open(QUEUE_FILE) as f:
            return json.load(f)
    except:
        return None

def save_queue(data):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def get_training_status():
    try:
        result = subprocess.run([
            "bash", "-c",
            "ps aux | grep -E '(train_script|ultralytics)' | grep -v grep"
        ], capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def start_training(combo):
    from datetime import datetime
    from ultralytics import YOLO
    
    print(f"\\n🚀 Starting experiment: {combo['model_variant']} lr={combo['learning_rate']} batch={combo['batch_size']}")
    
    # Determine dataset path
    if combo['dataset_name'].startswith('YOLO_'):
        dataset_path = f"/ultralytics/YOLO_MultiLevel_Datasets/{combo['dataset_name']}/data.yaml"
    else:
        dataset_path = f"/ultralytics/custom_datasets/{combo['dataset_name']}/data.yaml"
    
    model = YOLO(f"{combo['model_variant']}.pt")
    
    time_str = datetime.now().strftime('%H%M%S')
    training_name = f'grid_search_{time_str}'
    
    try:
        results = model.train(
            data=dataset_path,
            epochs=combo['epochs'],
            imgsz=combo['img_size'],
            batch=combo['batch_size'],
            device='0',
            lr0=combo['learning_rate'],
            optimizer=combo['optimizer'],
            project='/ultralytics/runs/detect',
            name=training_name,
            exist_ok=True,
            verbose=True
        )
        
        # Get metrics from results
        metrics = results.results_dict
        return {
            'training_name': training_name,
            'success': True,
            'mAP50': float(metrics.get('metrics/mAP50(B)', 0)),
            'mAP50_95': float(metrics.get('metrics/mAP50-95(B)', 0)),
            'precision': float(metrics.get('metrics/precision(B)', 0)),
            'recall': float(metrics.get('metrics/recall(B)', 0)),
        }
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return {
            'training_name': training_name,
            'success': False,
            'error': str(e),
            'mAP50': 0,
            'mAP50_95': 0,
            'precision': 0,
            'recall': 0,
        }

print("🔍 Grid Search Worker Started")
print(f"Search ID: ${searchId}")

while True:
    queue = load_queue()
    if not queue or queue['status'] == 'stopped':
        print("⏹️  Grid search stopped")
        break
    
    # Check if we can start new training
    if len(queue['pending']) > 0 and not get_training_status():
        # Start next experiment
        combo = queue['pending'].pop(0)
        
        print(f"\\n📊 Progress: {queue['completed']}/{queue['total_combinations']}")
        print(f"⏳ Remaining: {len(queue['pending'])}")
        
        # Run training
        result = start_training(combo)
        
        # Add to results
        result_entry = {**combo, **result}
        queue['results'].append(result_entry)
        queue['completed'] += 1
        
        # Update best model
        if result['success']:
            if not queue['best_model'] or result['mAP50'] > queue['best_model']['mAP50']:
                queue['best_model'] = result_entry
                print(f"\\n🏆 New best model! mAP50: {result['mAP50']:.4f}")
        
        save_queue(queue)
    
    # Check if done
    if len(queue['pending']) == 0:
        queue['status'] = 'completed'
        save_queue(queue)
        print(f"\\n✅ Grid search completed!")
        print(f"Best model: {queue['best_model']['model_variant']} with mAP50: {queue['best_model']['mAP50']:.4f}")
        break
    
    time.sleep(10)  # Check every 10 seconds
`;

        const scriptPath = `/tmp/grid_search_worker_${searchId}.py`;
        await execInContainer(`cat > ${scriptPath} << 'EOF'
${workerScript}
EOF`);
        await execInContainer(`chmod +x ${scriptPath}`);

        // Start worker in background
        execInContainer(`nohup python3 ${scriptPath} > /tmp/grid_search_${searchId}.log 2>&1 &`);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  search_id: searchId,
                  total_experiments: combinations.length,
                  estimated_time_hours: totalTimeHours.toFixed(1),
                  combinations: combinations.slice(0, 5), // Show first 5
                  status: "Grid search started. Use get_grid_search_status to monitor.",
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "get_grid_search_status": {
        const { search_id } = args;

        const queue = await readGridSearchQueue(search_id);
        if (!queue) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ error: "Grid search not found" }),
              },
            ],
          };
        }

        // Sort results by mAP50
        const topN = queue.results
          .filter((r) => r.success)
          .sort((a, b) => b.mAP50 - a.mAP50)
          .slice(0, 5);

        const progressPercent =
          queue.total_combinations > 0
            ? ((queue.completed / queue.total_combinations) * 100).toFixed(1)
            : 0;

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  search_id: queue.id,
                  status: queue.status,
                  progress: `${queue.completed}/${queue.total_combinations}`,
                  progress_percent: progressPercent,
                  pending_experiments: queue.pending.length,
                  completed_experiments: queue.completed,
                  top_5_models: topN.map((r, idx) => ({
                    rank: idx + 1,
                    model: r.model_variant,
                    learning_rate: r.learning_rate,
                    batch_size: r.batch_size,
                    epochs: r.epochs,
                    mAP50: r.mAP50,
                    mAP50_95: r.mAP50_95,
                    precision: r.precision,
                    recall: r.recall,
                    training_name: r.training_name,
                  })),
                  best_model: queue.best_model
                    ? {
                        model: queue.best_model.model_variant,
                        learning_rate: queue.best_model.learning_rate,
                        batch_size: queue.best_model.batch_size,
                        epochs: queue.best_model.epochs,
                        mAP50: queue.best_model.mAP50,
                        training_name: queue.best_model.training_name,
                      }
                    : null,
                  started_at: queue.started_at,
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "stop_grid_search": {
        const { search_id } = args;

        const queue = await readGridSearchQueue(search_id);
        if (!queue) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ error: "Grid search not found" }),
              },
            ],
          };
        }

        // Update status to stopped
        queue.status = "stopped";
        await saveGridSearchQueue(search_id, queue);

        // Kill worker process
        await execInContainer(`pkill -f grid_search_worker_${search_id}.py`);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  message: "Grid search stopped",
                  completed_experiments: queue.completed,
                  total_experiments: queue.total_combinations,
                  best_result: queue.best_model
                    ? {
                        model: queue.best_model.model_variant,
                        mAP50: queue.best_model.mAP50,
                        learning_rate: queue.best_model.learning_rate,
                        batch_size: queue.best_model.batch_size,
                      }
                    : null,
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "get_class_metrics": {
        const { training_name, class_name } = args;

        const trainingPath = `/ultralytics/runs/detect/${training_name}`;

        // Check if there's an active training running
        const statusCheckResult = await execInContainer(
          `ps aux | grep "train_script.py" | grep -v grep | wc -l`
        );
        const isTrainingActive = parseInt(statusCheckResult.stdout.trim()) > 0;

        // Read args.yaml to get dataset path and model info
        const argsYAML = await readFileFromContainer(`${trainingPath}/args.yaml`);
        if (!argsYAML) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ error: "Training not found or args.yaml missing" }),
              },
            ],
          };
        }

        // Extract dataset path and model path
        const dataMatch = argsYAML.match(/data:\s*(.+)/);
        const modelMatch = argsYAML.match(/model:\s*(.+)/);
        
        if (!dataMatch) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ error: "Could not find dataset path in args.yaml" }),
              },
            ],
          };
        }

        const datasetPath = dataMatch[1].trim();
        const modelName = modelMatch ? modelMatch[1].trim() : null;
        
        // Get the trained model weights
        const weightsPath = `${trainingPath}/weights/best.pt`;
        const weightsExist = await execInContainer(`test -f "${weightsPath}" && echo "yes" || echo "no"`);
        const hasWeights = weightsExist.stdout.trim() === "yes";

        // Read dataset yaml to get class names
        const datasetYAML = await readFileFromContainer(datasetPath);
        if (!datasetYAML) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ error: `Dataset YAML not found: ${datasetPath}` }),
              },
            ],
          };
        }

        // Parse class names
        const classLines = datasetYAML.match(/names:\s*\n([\s\S]+?)(?=\ntest:|train:|val:|nc:|path:|$)/);
        if (!classLines) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ 
                  error: "Could not parse class names from dataset YAML",
                  yaml_preview: datasetYAML.substring(0, 500)
                }),
              },
            ],
          };
        }

        const classes = {};
        // Support both formats: "0: ClassName" and "  - ClassName"
        const yamlContent = classLines[1];
        
        // Try format: "  0: ClassName"
        let classMatches = [...yamlContent.matchAll(/^\s*(\d+):\s*['"]?([^'"#\n]+)['"]?\s*$/gm)];
        
        if (classMatches.length === 0) {
          // Try list format: "  - ClassName"
          classMatches = [...yamlContent.matchAll(/^\s*-\s*['"]?([^'"#\n]+)['"]?\s*$/gm)];
          classMatches.forEach((match, index) => {
            classes[index] = match[1].trim();
          });
        } else {
          // Use numbered format
          classMatches.forEach((match) => {
            const [_, index, name] = match;
            classes[parseInt(index)] = name.trim();
          });
        }

        // If class_name provided, search for it
        let targetClasses = [];
        if (class_name) {
          const searchTerm = class_name.toLowerCase();
          for (const [index, name] of Object.entries(classes)) {
            if (name.toLowerCase().includes(searchTerm)) {
              targetClasses.push({ index: parseInt(index), name });
            }
          }

          if (targetClasses.length === 0) {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    error: `No class found matching '${class_name}'`,
                    available_classes: classes,
                  }),
                },
              ],
            };
          }
        } else {
          // Return all classes
          targetClasses = Object.entries(classes).map(([index, name]) => ({
            index: parseInt(index),
            name,
          }));
        }

        // Read results.csv for per-class metrics if available
        // YOLO typically saves overall metrics, not per-class in CSV
        // Best approach: Read from confusion matrix or validation logs

        // Try to get class metrics from results.csv columns
        // YOLOv8+ may have class-specific columns in newer versions
        const resultsCSV = await readFileFromContainer(`${trainingPath}/results.csv`);
        
        // Read latest metrics from CSV
        let overallMetrics = {};
        if (resultsCSV) {
          const lines = resultsCSV.trim().split("\n");
          if (lines.length > 1) {
            const header = lines[0].split(",").map((v) => v.trim());
            const lastLine = lines[lines.length - 1];
            const values = lastLine.split(",").map((v) => v.trim());
            
            // Find correct column indices
            const precisionIdx = header.findIndex(h => h.includes("precision(B)"));
            const recallIdx = header.findIndex(h => h.includes("recall(B)"));
            const mAP50Idx = header.findIndex(h => h.includes("mAP50(B)") && !h.includes("mAP50-95"));
            const mAP5095Idx = header.findIndex(h => h.includes("mAP50-95(B)"));
            
            overallMetrics = {
              overall_precision: precisionIdx >= 0 ? parseFloat(values[precisionIdx]) : null,
              overall_recall: recallIdx >= 0 ? parseFloat(values[recallIdx]) : null,
              overall_mAP50: mAP50Idx >= 0 ? parseFloat(values[mAP50Idx]) : null,
              overall_mAP50_95: mAP5095Idx >= 0 ? parseFloat(values[mAP5095Idx]) : null,
              epoch: parseInt(values[0]),
            };
          }
        }

        // If no active training and we have weights, run validation for specific class metrics
        let classSpecificMetrics = null;
        if (!isTrainingActive && hasWeights && class_name) {
          // Use YOLO CLI to run validation - much simpler!
          const valResult = await execInContainer(
            `cd /ultralytics && yolo val model="${weightsPath}" data="${datasetPath}" save_json=True 2>&1`
          );

          // YOLO saves results to runs/detect/val/predictions.json
          const latestValDir = await execInContainer(
            `ls -td /ultralytics/runs/detect/val* 2>/dev/null | head -1`
          );
          
          if (latestValDir.success && latestValDir.stdout.trim()) {
            const valDir = latestValDir.stdout.trim();
            
            // Try to read predictions.json or results.json
            let resultsJSON = await readFileFromContainer(`${valDir}/predictions.json`);
            if (!resultsJSON) {
              resultsJSON = await readFileFromContainer(`${valDir}/results.json`);
            }
            
            if (resultsJSON) {
              try {
                const results = JSON.parse(resultsJSON);
                
                // Extract class-specific metrics
                const searchTerm = class_name.toLowerCase();
                const classMetrics = [];
                
                // YOLO results typically have per-class metrics
                if (results.metrics && results.metrics.per_class) {
                  for (const [className, metrics] of Object.entries(results.metrics.per_class)) {
                    if (className.toLowerCase().includes(searchTerm)) {
                      classMetrics.push({
                        class_name: className,
                        ...metrics
                      });
                    }
                  }
                }
                
                classSpecificMetrics = {
                  validation_dir: valDir,
                  class_metrics: classMetrics.length > 0 ? classMetrics : null,
                  note: classMetrics.length > 0 
                    ? "Class-specific metrics from validation run"
                    : "Validation completed but class-specific format not available. Check confusion matrix."
                };
              } catch (e) {
                classSpecificMetrics = {
                  error: "Failed to parse validation results",
                  validation_output: valResult.stdout.substring(0, 1000)
                };
              }
            } else {
              // Even if JSON not available, we ran validation
              classSpecificMetrics = {
                validation_completed: true,
                validation_dir: valDir,
                note: "Validation completed. Check confusion_matrix.png for class-specific details."
              };
            }
          }
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  training_name,
                  dataset: datasetPath,
                  total_classes: Object.keys(classes).length,
                  searched_classes: targetClasses,
                  overall_metrics: overallMetrics,
                  class_specific_metrics: classSpecificMetrics,
                  note: classSpecificMetrics 
                    ? "Class-specific metrics obtained via validation run"
                    : isTrainingActive 
                      ? "Training is active - skipping validation to avoid interference. Overall metrics shown."
                      : !hasWeights
                        ? "No trained weights found. Overall metrics shown."
                        : "YOLO stores per-class metrics in validation logs and confusion matrix. Overall metrics shown above.",
                  confusion_matrix_path: `${trainingPath}/confusion_matrix.png`,
                  all_classes: classes,
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
