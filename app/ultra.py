import subprocess
import json
import yaml
import re
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def run_ultralytics(args: List[str]) -> Dict[str, Any]:
    """
    Execute YOLO command via subprocess and parse results.
    
    Args:
        args: List of command line arguments for YOLO
        
    Returns:
        Dictionary containing execution results and parsed metrics
    """
    # Build the full command
    full_command = ["yolo"] + args
    
    result_dict = {
        "command": " ".join(full_command),
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "metrics": {},
        "artifacts": [],
        "success": False
    }
    
    try:
        # Execute the command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=os.getcwd()
        )
        
        result_dict["return_code"] = result.returncode
        result_dict["stdout"] = result.stdout
        result_dict["stderr"] = result.stderr
        result_dict["success"] = result.returncode == 0
        
        # Parse metrics from output
        metrics = _parse_metrics_from_output(result.stdout, result.stderr)
        result_dict["metrics"] = metrics
        
        # Find artifacts
        artifacts = _find_artifacts()
        result_dict["artifacts"] = artifacts
        
    except subprocess.TimeoutExpired:
        result_dict["return_code"] = -1
        result_dict["stderr"] = "Command timed out after 1 hour"
        result_dict["success"] = False
        
    except subprocess.CalledProcessError as e:
        result_dict["return_code"] = e.returncode
        result_dict["stdout"] = e.stdout if e.stdout else ""
        result_dict["stderr"] = e.stderr if e.stderr else ""
        result_dict["success"] = False
        
    except Exception as e:
        result_dict["return_code"] = -1
        result_dict["stderr"] = f"Unexpected error: {str(e)}"
        result_dict["success"] = False
    
    return result_dict


def _parse_metrics_from_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """Parse metrics from YOLO command output."""
    metrics = {}
    
    # Combine stdout and stderr for parsing
    full_output = stdout + "\n" + stderr
    
    # Parse training metrics
    metrics.update(_parse_training_metrics(full_output))
    
    # Parse validation metrics
    metrics.update(_parse_validation_metrics(full_output))
    
    # Parse prediction metrics
    metrics.update(_parse_prediction_metrics(full_output))
    
    # Parse export metrics
    metrics.update(_parse_export_metrics(full_output))
    
    # Look for results files and parse them
    results_files = _find_results_files()
    for file_path in results_files:
        file_metrics = _parse_results_file(file_path)
        metrics.update(file_metrics)
    
    return metrics


def _parse_training_metrics(output: str) -> Dict[str, Any]:
    """Parse training-specific metrics from output."""
    metrics = {}
    
    # Parse epoch information
    epoch_pattern = r"Epoch\s+(\d+)/(\d+)"
    epoch_matches = re.findall(epoch_pattern, output)
    if epoch_matches:
        last_epoch = epoch_matches[-1]
        metrics["current_epoch"] = int(last_epoch[0])
        metrics["total_epochs"] = int(last_epoch[1])
    
    # Parse loss values
    loss_patterns = {
        "box_loss": r"box_loss:\s*([\d.]+)",
        "obj_loss": r"obj_loss:\s*([\d.]+)",
        "cls_loss": r"cls_loss:\s*([\d.]+)",
        "total_loss": r"total_loss:\s*([\d.]+)"
    }
    
    for loss_name, pattern in loss_patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            metrics[loss_name] = float(matches[-1])
    
    # Parse mAP values
    map_patterns = {
        "mAP50": r"mAP50:\s*([\d.]+)",
        "mAP50-95": r"mAP50-95:\s*([\d.]+)"
    }
    
    for map_name, pattern in map_patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            metrics[map_name] = float(matches[-1])
    
    return metrics


def _parse_validation_metrics(output: str) -> Dict[str, Any]:
    """Parse validation-specific metrics from output."""
    metrics = {}
    
    # Parse precision, recall, mAP
    precision_pattern = r"Precision:\s*([\d.]+)"
    recall_pattern = r"Recall:\s*([\d.]+)"
    
    precision_matches = re.findall(precision_pattern, output)
    recall_matches = re.findall(recall_pattern, output)
    
    if precision_matches:
        metrics["precision"] = float(precision_matches[-1])
    if recall_matches:
        metrics["recall"] = float(recall_matches[-1])
    
    return metrics


def _parse_prediction_metrics(output: str) -> Dict[str, Any]:
    """Parse prediction-specific metrics from output."""
    metrics = {}
    
    # Parse inference time
    inference_pattern = r"inference:\s*([\d.]+)ms"
    matches = re.findall(inference_pattern, output)
    if matches:
        metrics["inference_time_ms"] = float(matches[-1])
    
    # Parse detection counts
    detection_pattern = r"(\d+)\s+detections"
    matches = re.findall(detection_pattern, output)
    if matches:
        metrics["total_detections"] = int(matches[-1])
    
    return metrics


def _parse_export_metrics(output: str) -> Dict[str, Any]:
    """Parse export-specific metrics from output."""
    metrics = {}
    
    # Parse export format and file size
    export_pattern = r"Export complete \(([\d.]+)s\)"
    matches = re.findall(export_pattern, output)
    if matches:
        metrics["export_time_s"] = float(matches[-1])
    
    # Parse exported file info
    file_pattern = r"Results saved to (.+)"
    matches = re.findall(file_pattern, output)
    if matches:
        metrics["exported_file"] = matches[-1].strip()
    
    return metrics


def _find_results_files() -> List[str]:
    """Find results files (YAML/JSON) in common output directories."""
    results_files = []
    
    # Common paths where YOLO saves results
    search_paths = [
        "runs/train",
        "runs/val", 
        "runs/predict",
        "runs/export",
        "runs/track",
        "runs/benchmark"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(('.yaml', '.yml', '.json')):
                        full_path = os.path.join(root, file)
                        results_files.append(full_path)
    
    return results_files


def _parse_results_file(file_path: str) -> Dict[str, Any]:
    """Parse metrics from a results file (YAML or JSON)."""
    metrics = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                metrics[f"file_{Path(file_path).stem}"] = data
            elif file_path.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(f)
                metrics[f"file_{Path(file_path).stem}"] = data
    except Exception as e:
        metrics[f"file_{Path(file_path).stem}_error"] = str(e)
    
    return metrics


def _find_artifacts() -> List[str]:
    """Find artifact files generated by YOLO operations."""
    artifacts = []
    
    # Common output directories
    output_dirs = [
        "runs",
        "weights", 
        "results",
        "exports"
    ]
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    # Get relative path for cleaner output
                    rel_path = os.path.relpath(full_path)
                    artifacts.append(rel_path)
    
    return sorted(artifacts)


def parse_yolo_args(args_dict: Dict[str, Any]) -> List[str]:
    """Convert dictionary of arguments to YOLO CLI format."""
    args = []
    
    # Ensure device defaults to CPU when CUDA is not available
    if 'device' not in args_dict or args_dict['device'] is None:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            args_dict['device'] = 'cuda'
        else:
            args_dict['device'] = 'cpu'
    
    for key, value in args_dict.items():
        if value is not None:
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])
    
    return args
