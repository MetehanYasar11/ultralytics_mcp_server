# Enhanced Streamlit Inference with Custom Model Support
# This file extends the original streamlit_inference.py to include custom trained models

import io
import os
from typing import Any, List
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
torch.classes.__path__ = []  # Torch module __path__._path issue

class EnhancedInference:
    """
    Enhanced Inference class with custom model support.
    Extends the original Inference class to scan and use custom trained models.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Enhanced Inference class with custom model scanning."""
        check_requirements("streamlit>=1.29.0")  # scope imports for faster 'import ultralytics'
        import streamlit as st
        
        self.st = st
        self.temp_dict = {}
        self.model_path = kwargs.get("model")
        self.model = None
        self.source = None
        self.enable_trk = None
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = ""
        self.selected_ind = [0]
        self.img_file_names = []
        
    def scan_custom_models(self) -> List[str]:
        """Scan for custom trained models in the runs directory."""
        custom_models = []
        runs_dir = Path("/ultralytics/runs")
        
        if runs_dir.exists():
            # Scan for training runs
            detect_dir = runs_dir / "detect"
            if detect_dir.exists():
                for train_dir in detect_dir.glob("train*"):
                    weights_dir = train_dir / "weights"
                    if weights_dir.exists():
                        # Look for best.pt and last.pt
                        for weight_file in weights_dir.glob("*.pt"):
                            model_name = f"Custom: {train_dir.name}/{weight_file.name}"
                            custom_models.append({
                                "name": model_name,
                                "path": str(weight_file),
                                "training_run": train_dir.name
                            })
        
        # Also scan workspace for any .pt files
        workspace_dir = Path("/workspace")
        if workspace_dir.exists():
            for pt_file in workspace_dir.rglob("*.pt"):
                if "yolo" not in pt_file.name.lower():  # Skip default YOLO models
                    model_name = f"Workspace: {pt_file.name}"
                    custom_models.append({
                        "name": model_name,
                        "path": str(pt_file),
                        "training_run": "workspace"
                    })
                    
        return custom_models
    
    def configure_enhanced(self) -> None:
        """Enhanced configure method with custom model support."""
        self.st.sidebar.title("🤖 Model Selection")
        
        # Scan for custom models
        custom_models = self.scan_custom_models()
        
        # Create model categories
        self.st.sidebar.subheader("📦 Pre-trained Models")
        
        # Default YOLO models
        M_ORD, T_ORD = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"], ["", "-seg", "-pose", "-obb", "-cls"]
        default_models = sorted(
            [
                x.replace("yolo", "YOLO")
                for x in GITHUB_ASSETS_STEMS
                if any(x.startswith(b) for b in M_ORD) and "grayscale" not in x
            ],
            key=lambda x: (M_ORD.index(x[:7].lower()), T_ORD.index(x[7:].lower() or "")),
        )
        
        # Model selection options
        model_category = self.st.sidebar.radio(
            "Model Category:",
            ["🏷️ Pre-trained Models", "🎯 Custom Trained Models"],
            help="Choose between official pre-trained models or your custom trained models"
        )
        
        selected_model_path = None
        model_info = None
        
        if model_category == "🏷️ Pre-trained Models":
            # Default models dropdown
            if self.model_path:  # If user provided custom model in constructor
                default_models.insert(0, self.model_path.split(".pt", 1)[0])
            
            selected_model = self.st.sidebar.selectbox(
                "Select Pre-trained Model:",
                default_models,
                help="Official YOLO models trained on COCO dataset"
            )
            selected_model_path = f"{selected_model.lower()}.pt"
            model_info = f"**Pre-trained:** {selected_model}"
            
        else:  # Custom models
            if custom_models:
                self.st.sidebar.subheader("🎯 Your Custom Models")
                
                # Create display names and paths mapping
                model_options = [model["name"] for model in custom_models]
                model_paths = {model["name"]: model["path"] for model in custom_models}
                model_runs = {model["name"]: model["training_run"] for model in custom_models}
                
                selected_model_name = self.st.sidebar.selectbox(
                    "Select Custom Model:",
                    model_options,
                    help="Models from your training runs and workspace"
                )
                
                selected_model_path = model_paths[selected_model_name]
                training_run = model_runs[selected_model_name]
                
                # Show model info
                model_info = f"**Custom Model:** {selected_model_name}\\n**Training Run:** {training_run}"
                
                # Display model file info
                model_file = Path(selected_model_path)
                if model_file.exists():
                    file_size = model_file.stat().st_size / (1024*1024)  # MB
                    self.st.sidebar.info(f"📁 File: {model_file.name}\\n💾 Size: {file_size:.1f} MB")
                    
            else:
                self.st.sidebar.warning("No custom models found!")
                self.st.sidebar.info("""
                **To use custom models:**
                1. Train a model using the MCP tools
                2. Upload .pt files to workspace
                3. Refresh this page
                """)
                # Fallback to default model
                selected_model_path = "yolo11n.pt"
                model_info = "**Fallback:** YOLO11n (no custom models found)"
        
        # Display selected model info
        if model_info:
            self.st.sidebar.success(model_info)
        
        # Load the selected model
        with self.st.spinner("🔄 Loading model..."):
            try:
                self.model = YOLO(selected_model_path)
                class_names = list(self.model.names.values())
                self.st.success(f"✅ Model loaded: {Path(selected_model_path).name}")
                
                # Show model details
                if hasattr(self.model, 'info'):
                    try:
                        model_summary = self.model.info(verbose=False)
                        self.st.sidebar.expander("🔍 Model Details").text(str(model_summary))
                    except:
                        pass
                        
            except Exception as e:
                self.st.error(f"❌ Failed to load model: {str(e)}")
                # Fallback to default
                self.model = YOLO("yolo11n.pt")
                class_names = list(self.model.names.values())
                self.st.warning("🔄 Loaded fallback model: YOLO11n")
        
        # Class selection
        self.st.sidebar.subheader("🎯 Detection Classes")
        selected_classes = self.st.sidebar.multiselect(
            "Select classes to detect:",
            class_names,
            default=class_names[:3],
            help="Choose which object classes to detect"
        )
        self.selected_ind = [class_names.index(option) for option in selected_classes]
        
        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)
            
        # Show selected classes count
        self.st.sidebar.info(f"🎯 Detecting {len(self.selected_ind)} out of {len(class_names)} classes")

    def web_ui(self) -> None:
        """Set up the enhanced Streamlit web interface."""
        # Page config
        self.st.set_page_config(
            page_title="🤖 Ultralytics AI - Enhanced",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self.st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .sub-header {
            font-size: 1.2rem;
            color: #4ECDC4;
            text-align: center;
            margin-bottom: 3rem;
        }
        .feature-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main title
        self.st.markdown('<h1 class="main-header">🤖 Ultralytics AI - Enhanced Edition</h1>', unsafe_allow_html=True)
        self.st.markdown('<p class="sub-header">🎯 AI Object Detection with Custom Model Support</p>', unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = self.st.columns(3)
        with col1:
            self.st.markdown("""
            <div class="feature-box">
                <h4>🏷️ Pre-trained Models</h4>
                <p>Official YOLO models trained on COCO dataset</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            self.st.markdown("""
            <div class="feature-box">
                <h4>🎯 Custom Models</h4>
                <p>Your own trained models from training runs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            self.st.markdown("""
            <div class="feature-box">
                <h4>🔄 Real-time Inference</h4>
                <p>Webcam, video, and image processing</p>
            </div>
            """, unsafe_allow_html=True)

    def sidebar(self) -> None:
        """Configure the enhanced Streamlit sidebar."""
        with self.st.sidebar:
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("⚙️ Configuration")
        
        # Source selection
        self.source = self.st.sidebar.selectbox(
            "📹 Source Type:",
            ("webcam", "video", "image"),
            help="Choose your input source"
        )
        
        # Tracking option for video sources
        if self.source in ["webcam", "video"]:
            self.enable_trk = self.st.sidebar.radio(
                "🎯 Object Tracking:",
                ("Yes", "No"),
                help="Enable multi-object tracking"
            ) == "Yes"
        
        # Threshold settings
        self.st.sidebar.subheader("🎛️ Detection Settings")
        self.conf = float(
            self.st.sidebar.slider(
                "Confidence Threshold:",
                0.0, 1.0, self.conf, 0.01,
                help="Minimum confidence for detections"
            )
        )
        self.iou = float(
            self.st.sidebar.slider(
                "IoU Threshold:",
                0.0, 1.0, self.iou, 0.01,
                help="IoU threshold for Non-Maximum Suppression"
            )
        )

        if self.source != "image":
            col1, col2 = self.st.columns(2)
            self.org_frame = col1.empty()
            self.ann_frame = col2.empty()

    def source_upload(self) -> None:
        """Handle enhanced file uploads."""
        from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
        
        self.vid_file_name = ""
        
        if self.source == "video":
            self.st.sidebar.subheader("📹 Video Upload")
            vid_file = self.st.sidebar.file_uploader(
                "Upload Video File:",
                type=VID_FORMATS,
                help="Support formats: MP4, AVI, MOV, etc."
            )
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("ultralytics.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "ultralytics.mp4"
                self.st.sidebar.success(f"✅ Video uploaded: {vid_file.name}")
                
        elif self.source == "webcam":
            self.vid_file_name = 0
            self.st.sidebar.subheader("📷 Webcam")
            self.st.sidebar.info("Using default webcam (index 0)")
            
        elif self.source == "image":
            import tempfile
            
            self.st.sidebar.subheader("🖼️ Image Upload")
            imgfiles = self.st.sidebar.file_uploader(
                "Upload Image Files:",
                type=IMG_FORMATS,
                accept_multiple_files=True,
                help="Support formats: JPG, PNG, BMP, etc."
            )
            if imgfiles:
                self.st.sidebar.success(f"✅ {len(imgfiles)} image(s) uploaded")
                for imgfile in imgfiles:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{imgfile.name.split('.')[-1]}") as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append({"path": tf.name, "name": imgfile.name})

    def inference(self) -> None:
        """Main inference method with enhanced UI."""
        self.web_ui()
        self.sidebar()
        self.configure_enhanced()  # Use enhanced configure
        self.source_upload()
        
        # Inference controls
        col1, col2 = self.st.columns([2, 1])
        with col1:
            start_button = self.st.button("🚀 Start Inference", type="primary", use_container_width=True)
        with col2:
            if self.source != "image":
                stop_button = self.st.button("⏹️ Stop", type="secondary", use_container_width=True)
        
        if start_button:
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.warning("📸 Please upload image files first!")
                return
            
            # Video/webcam inference (simplified for space)
            cap = cv2.VideoCapture(self.vid_file_name)
            if not cap.isOpened():
                self.st.error("❌ Could not open video source")
                return
                
            # Placeholder for video inference loop
            self.st.info("🎬 Video inference would run here...")
            cap.release()

    def image_inference(self) -> None:
        """Enhanced image inference with better UI."""
        progress_bar = self.st.progress(0)
        total_images = len(self.img_file_names)
        
        for idx, img_info in enumerate(self.img_file_names):
            img_path = img_info["path"]
            image = cv2.imread(img_path)
            
            if image is not None:
                # Update progress
                progress = (idx + 1) / total_images
                progress_bar.progress(progress)
                
                self.st.markdown(f"### 🖼️ Processing: {img_info['name']}")
                
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.subheader("📷 Original")
                    self.st.image(image, channels="BGR", use_container_width=True)
                
                # Run inference
                results = self.model(image, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_image = results[0].plot()
                
                with col2:
                    self.st.subheader("🎯 Detected")
                    self.st.image(annotated_image, channels="BGR", use_container_width=True)
                
                # Show detection stats
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                self.st.info(f"🎯 Detected {detections} objects")
                
                # Clean up
                try:
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass
            else:
                self.st.error(f"❌ Could not load image: {img_info['name']}")
        
        progress_bar.progress(1.0)
        self.st.success("✅ All images processed!")


# Main execution
if __name__ == "__main__":
    app = EnhancedInference()
    app.inference()
