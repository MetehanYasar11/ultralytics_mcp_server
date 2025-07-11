from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union


class BaseRequest(BaseModel):
    """Base request model with common YOLO parameters"""
    task: Optional[str] = Field(None, description="YOLO task type")
    model: str = Field(..., description="Model path or name")
    data: Optional[str] = Field(None, description="Dataset YAML file path")
    source: Optional[str] = Field(None, description="Source path for images/videos")
    epochs: Optional[int] = Field(None, description="Number of training epochs")
    imgsz: Optional[int] = Field(640, description="Image size for inference/training")
    device: Optional[str] = Field("cpu", description="Device to use (cpu, 0, 1, etc.)")
    extra_args: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional arguments")


class TrainRequest(BaseRequest):
    """Training request with training-specific parameters"""
    task: str = Field("train", description="Training task")
    data: str = Field(..., description="Dataset YAML file path (required for training)")
    epochs: Optional[int] = Field(100, description="Number of training epochs")
    batch: Optional[int] = Field(16, description="Batch size")
    lr0: Optional[float] = Field(0.01, description="Initial learning rate")
    lrf: Optional[float] = Field(0.01, description="Final learning rate")
    momentum: Optional[float] = Field(0.937, description="SGD momentum")
    weight_decay: Optional[float] = Field(0.0005, description="Optimizer weight decay")
    warmup_epochs: Optional[float] = Field(3.0, description="Warmup epochs")
    warmup_momentum: Optional[float] = Field(0.8, description="Warmup initial momentum")
    warmup_bias_lr: Optional[float] = Field(0.1, description="Warmup initial bias lr")
    box: Optional[float] = Field(7.5, description="Box loss gain")
    cls: Optional[float] = Field(0.5, description="Cls loss gain")
    dfl: Optional[float] = Field(1.5, description="DFL loss gain")
    pose: Optional[float] = Field(12.0, description="Pose loss gain")
    kobj: Optional[float] = Field(2.0, description="Keypoint obj loss gain")
    label_smoothing: Optional[float] = Field(0.0, description="Label smoothing")
    nbs: Optional[int] = Field(64, description="Nominal batch size")
    overlap_mask: Optional[bool] = Field(True, description="Masks should overlap during training")
    mask_ratio: Optional[int] = Field(4, description="Mask downsample ratio")
    dropout: Optional[float] = Field(0.0, description="Use dropout regularization")
    val: Optional[bool] = Field(True, description="Validate/test during training")
    project: Optional[str] = Field("runs/train", description="Project directory")
    name: Optional[str] = Field(None, description="Experiment name")
    exist_ok: Optional[bool] = Field(False, description="Whether to overwrite existing experiment")
    pretrained: Optional[Union[bool, str]] = Field(True, description="Use pretrained model")
    optimizer: Optional[str] = Field("auto", description="Optimizer to use")
    verbose: Optional[bool] = Field(False, description="Whether to print verbose output")
    seed: Optional[int] = Field(0, description="Global training seed")
    deterministic: Optional[bool] = Field(True, description="Whether to enable deterministic mode")
    single_cls: Optional[bool] = Field(False, description="Train multi-class data as single-class")
    rect: Optional[bool] = Field(False, description="Rectangular training")
    cos_lr: Optional[bool] = Field(False, description="Use cosine learning rate scheduler")
    close_mosaic: Optional[int] = Field(10, description="Disable mosaic augmentation for final epochs")
    resume: Optional[bool] = Field(False, description="Resume training from last checkpoint")
    amp: Optional[bool] = Field(True, description="Automatic Mixed Precision training")
    fraction: Optional[float] = Field(1.0, description="Dataset fraction to train on")
    profile: Optional[bool] = Field(False, description="Profile ONNX and TensorRT speeds during training")
    freeze: Optional[Union[int, list]] = Field(None, description="Freeze layers during training")


class ValRequest(BaseRequest):
    """Validation request with validation-specific parameters"""
    task: str = Field("val", description="Validation task")
    data: str = Field(..., description="Dataset YAML file path (required for validation)")
    batch: Optional[int] = Field(32, description="Batch size for validation")
    save_json: Optional[bool] = Field(False, description="Save results to JSON file")
    save_hybrid: Optional[bool] = Field(False, description="Save hybrid labels")
    conf: Optional[float] = Field(0.001, description="Object confidence threshold")
    iou: Optional[float] = Field(0.6, description="Intersection over Union threshold for NMS")
    max_det: Optional[int] = Field(300, description="Maximum number of detections per image")
    half: Optional[bool] = Field(True, description="Use half precision")
    dnn: Optional[bool] = Field(False, description="Use OpenCV DNN for ONNX inference")
    plots: Optional[bool] = Field(False, description="Show plots during training")
    project: Optional[str] = Field("runs/val", description="Project directory")
    name: Optional[str] = Field(None, description="Experiment name")
    exist_ok: Optional[bool] = Field(False, description="Whether to overwrite existing experiment")
    verbose: Optional[bool] = Field(True, description="Whether to print verbose output")
    split: Optional[str] = Field("val", description="Dataset split to use")
    save_txt: Optional[bool] = Field(False, description="Save results as .txt file")
    save_conf: Optional[bool] = Field(False, description="Save confidences in labels")
    save_crop: Optional[bool] = Field(False, description="Save cropped prediction boxes")
    show_labels: Optional[bool] = Field(True, description="Show labels")
    show_conf: Optional[bool] = Field(True, description="Show confidences")
    visualize: Optional[bool] = Field(False, description="Visualize features")
    augment: Optional[bool] = Field(False, description="Apply image augmentation to prediction sources")
    agnostic_nms: Optional[bool] = Field(False, description="Class-agnostic NMS")
    retina_masks: Optional[bool] = Field(False, description="Use high-resolution segmentation masks")
    format: Optional[str] = Field("torchscript", description="Format to export to")
    keras: Optional[bool] = Field(False, description="Use Keras")
    optimize: Optional[bool] = Field(False, description="TorchScript: optimize for mobile")
    int8: Optional[bool] = Field(False, description="CoreML/TF INT8 quantization")
    dynamic: Optional[bool] = Field(False, description="ONNX/TF/TensorRT: dynamic axes")
    simplify: Optional[bool] = Field(False, description="ONNX: simplify model")
    opset: Optional[int] = Field(None, description="ONNX: opset version")
    workspace: Optional[int] = Field(4, description="TensorRT: workspace size (GB)")
    nms: Optional[bool] = Field(False, description="CoreML: add NMS")


class PredictRequest(BaseRequest):
    """Prediction request with prediction-specific parameters"""
    task: str = Field("predict", description="Prediction task")
    source: str = Field(..., description="Source path for images/videos (required for prediction)")
    conf: Optional[float] = Field(0.25, description="Object confidence threshold")
    iou: Optional[float] = Field(0.7, description="Intersection over Union threshold for NMS")
    half: Optional[bool] = Field(False, description="Use half precision")
    show: Optional[bool] = Field(False, description="Show results")
    save: Optional[bool] = Field(True, description="Save images/videos with results")
    save_frames: Optional[bool] = Field(False, description="Save predicted individual video frames")
    save_txt: Optional[bool] = Field(False, description="Save results as .txt file")
    save_conf: Optional[bool] = Field(False, description="Save confidences in labels")
    save_crop: Optional[bool] = Field(False, description="Save cropped prediction boxes")
    show_labels: Optional[bool] = Field(True, description="Show labels")
    show_conf: Optional[bool] = Field(True, description="Show confidences")
    vid_stride: Optional[int] = Field(1, description="Video frame-rate stride")
    stream_buffer: Optional[bool] = Field(False, description="Buffer all streaming frames")
    line_width: Optional[int] = Field(None, description="Bounding box thickness")
    visualize: Optional[bool] = Field(False, description="Visualize features")
    augment: Optional[bool] = Field(False, description="Apply image augmentation to prediction sources")
    agnostic_nms: Optional[bool] = Field(False, description="Class-agnostic NMS")
    retina_masks: Optional[bool] = Field(False, description="Use high-resolution segmentation masks")
    classes: Optional[list] = Field(None, description="Filter by class")
    boxes: Optional[bool] = Field(True, description="Show boxes in segmentation predictions")
    project: Optional[str] = Field("runs/predict", description="Project directory")
    name: Optional[str] = Field(None, description="Experiment name")
    exist_ok: Optional[bool] = Field(False, description="Whether to overwrite existing experiment")
    verbose: Optional[bool] = Field(True, description="Whether to print verbose output")
    max_det: Optional[int] = Field(300, description="Maximum number of detections per image")


class ExportRequest(BaseRequest):
    """Export request with export-specific parameters"""
    task: str = Field("export", description="Export task")
    format: Optional[str] = Field("onnx", description="Export format")
    keras: Optional[bool] = Field(False, description="Use Keras")
    optimize: Optional[bool] = Field(False, description="TorchScript: optimize for mobile")
    half: Optional[bool] = Field(False, description="FP16 quantization")
    int8: Optional[bool] = Field(False, description="INT8 quantization")
    dynamic: Optional[bool] = Field(False, description="ONNX/TensorRT: dynamic axes")
    simplify: Optional[bool] = Field(False, description="ONNX: simplify model")
    opset: Optional[int] = Field(None, description="ONNX: opset version")
    workspace: Optional[int] = Field(4, description="TensorRT: workspace size (GB)")
    nms: Optional[bool] = Field(False, description="CoreML: add NMS")
    batch: Optional[int] = Field(1, description="Batch size")
    verbose: Optional[bool] = Field(False, description="Whether to print verbose output")


class TrackRequest(BaseRequest):
    """Tracking request with tracking-specific parameters"""
    task: str = Field("track", description="Tracking task")
    source: str = Field(..., description="Source video path (required for tracking)")
    tracker: Optional[str] = Field("bytetrack.yaml", description="Tracker configuration file")
    conf: Optional[float] = Field(0.3, description="Detection confidence threshold")
    iou: Optional[float] = Field(0.5, description="Intersection over Union threshold")
    show: Optional[bool] = Field(False, description="Show tracking results")
    save: Optional[bool] = Field(True, description="Save tracking results")
    save_frames: Optional[bool] = Field(False, description="Save individual frames")
    save_txt: Optional[bool] = Field(False, description="Save tracking results as .txt")
    save_id_crops: Optional[bool] = Field(False, description="Save tracking ID crops")
    show_labels: Optional[bool] = Field(True, description="Show labels")
    show_conf: Optional[bool] = Field(True, description="Show confidences")
    show_trajectories: Optional[bool] = Field(False, description="Show trajectories")
    line_width: Optional[int] = Field(None, description="Bounding box thickness")
    per_class: Optional[bool] = Field(False, description="Apply tracking per class")
    verbose: Optional[bool] = Field(True, description="Whether to print verbose output")
    vid_stride: Optional[int] = Field(1, description="Video frame-rate stride")
    project: Optional[str] = Field("runs/track", description="Project directory")
    name: Optional[str] = Field(None, description="Experiment name")
    exist_ok: Optional[bool] = Field(False, description="Whether to overwrite existing experiment")


class BenchmarkRequest(BaseRequest):
    """Benchmark request with benchmark-specific parameters"""
    task: str = Field("benchmark", description="Benchmark task")
    data: Optional[str] = Field(None, description="Dataset for benchmarking")
    verbose: Optional[bool] = Field(False, description="Whether to print verbose output")
    half: Optional[bool] = Field(False, description="Use half precision")
    int8: Optional[bool] = Field(False, description="Use INT8 quantization")
    batch: Optional[int] = Field(1, description="Batch size")
    project: Optional[str] = Field("runs/benchmark", description="Project directory")
    name: Optional[str] = Field(None, description="Experiment name")
    exist_ok: Optional[bool] = Field(False, description="Whether to overwrite existing experiment")


class SolutionRequest(BaseRequest):
    """Solution request with flexible parameters for various solutions"""
    task: str = Field("solution", description="Solution task")
    solution_type: str = Field(..., description="Type of solution to run")
    source: str = Field(..., description="Source path (required for solutions)")
    region_type: Optional[str] = Field("polygon", description="Region type for solutions")
    classes: Optional[list] = Field(None, description="Filter by class")
    conf: Optional[float] = Field(0.25, description="Confidence threshold")
    iou: Optional[float] = Field(0.7, description="IoU threshold")
    show: Optional[bool] = Field(False, description="Show results")
    save: Optional[bool] = Field(True, description="Save results")
    line_width: Optional[int] = Field(2, description="Line width for annotations")
    verbose: Optional[bool] = Field(True, description="Whether to print verbose output")
    project: Optional[str] = Field("runs/solution", description="Project directory")
    name: Optional[str] = Field(None, description="Experiment name")
    exist_ok: Optional[bool] = Field(False, description="Whether to overwrite existing experiment")


# Response models
class OperationResponse(BaseModel):
    """Unified response model for all operations"""
    run_id: str = Field(..., description="Unique identifier for this operation")
    command: str = Field(..., description="Full command that was executed")
    return_code: int = Field(..., description="Process return code")
    stdout: str = Field(..., description="Standard output from the command")
    stderr: str = Field(..., description="Standard error from the command")
    metrics: Dict[str, Any] = Field(..., description="Parsed metrics from the operation")
    artifacts: list[str] = Field(..., description="List of generated files/artifacts")
    success: bool = Field(..., description="Whether the operation was successful")
    timestamp: str = Field(..., description="ISO timestamp of when the operation completed")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="ISO timestamp of when the error occurred")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
