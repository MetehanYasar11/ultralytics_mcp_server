import streamlit as st
import subprocess
import os
from pathlib import Path
import torch
from PIL import Image
import tempfile
import shutil

# Page configuration
st.set_page_config(
    page_title="RCT Detector Inference",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

def get_gpu_info():
    """Get GPU information"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            torch.cuda.empty_cache()
            gpu_memory_allocated = torch.cuda.memory_allocated(0)
            gpu_memory_available = gpu_memory - gpu_memory_allocated
            gpu_memory_available_gb = gpu_memory_available / (1024**3)
            
            return {
                'name': gpu_name,
                'total_memory': gpu_memory_gb,
                'available_memory': gpu_memory_available_gb,
                'available': True
            }
        else:
            return {'name': 'No GPU Available', 'available': False}
    except Exception as e:
        return {'name': f'GPU Error: {str(e)}', 'available': False}

def get_available_models():
    """Get available trained models"""
    models = []
    
    # Get workspace models
    workspace_models = Path("/workspace/trained_models")
    if workspace_models.exists():
        for model_file in workspace_models.glob("*.pt"):
            models.append({
                'name': model_file.name,
                'path': str(model_file),
                'type': 'Workspace Model',
                'size': f"{model_file.stat().st_size / (1024*1024):.1f} MB"
            })
    
    # Get YOLO pre-trained models
    yolo_models = [
        # YOLO11 Series
        {'name': 'yolo11n.pt', 'path': 'yolo11n.pt', 'type': 'YOLO11 Nano', 'size': '5.8 MB'},
        {'name': 'yolo11s.pt', 'path': 'yolo11s.pt', 'type': 'YOLO11 Small', 'size': '19.8 MB'},
        {'name': 'yolo11m.pt', 'path': 'yolo11m.pt', 'type': 'YOLO11 Medium', 'size': '40.8 MB'},
        {'name': 'yolo11l.pt', 'path': 'yolo11l.pt', 'type': 'YOLO11 Large', 'size': '52.8 MB'},
        {'name': 'yolo11x.pt', 'path': 'yolo11x.pt', 'type': 'YOLO11 Extra Large', 'size': '68.2 MB'},
        
        # YOLOv8 Series
        {'name': 'yolov8n.pt', 'path': 'yolov8n.pt', 'type': 'YOLOv8 Nano', 'size': '6.2 MB'},
        {'name': 'yolov8s.pt', 'path': 'yolov8s.pt', 'type': 'YOLOv8 Small', 'size': '21.5 MB'},
        {'name': 'yolov8m.pt', 'path': 'yolov8m.pt', 'type': 'YOLOv8 Medium', 'size': '49.7 MB'},
        {'name': 'yolov8l.pt', 'path': 'yolov8l.pt', 'type': 'YOLOv8 Large', 'size': '83.7 MB'},
        {'name': 'yolov8x.pt', 'path': 'yolov8x.pt', 'type': 'YOLOv8 Extra Large', 'size': '136.7 MB'},
        
        # YOLOv9 Series
        {'name': 'yolov9t.pt', 'path': 'yolov9t.pt', 'type': 'YOLOv9 Tiny', 'size': '4.8 MB'},
        {'name': 'yolov9s.pt', 'path': 'yolov9s.pt', 'type': 'YOLOv9 Small', 'size': '15.8 MB'},
        {'name': 'yolov9m.pt', 'path': 'yolov9m.pt', 'type': 'YOLOv9 Medium', 'size': '43.0 MB'},
        {'name': 'yolov9c.pt', 'path': 'yolov9c.pt', 'type': 'YOLOv9 Compact', 'size': '51.0 MB'},
        {'name': 'yolov9e.pt', 'path': 'yolov9e.pt', 'type': 'YOLOv9 Extended', 'size': '115.0 MB'}
    ]
    
    models.extend(yolo_models)
    return models

def get_model_classes(model_path):
    """Get class names from a YOLO model"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        if hasattr(model, 'names') and model.names:
            return model.names
        return {}
    except Exception as e:
        return {}

def run_inference(model_path, image_path, confidence, iou_threshold, device, selected_classes=None):
    """Run YOLO inference on uploaded image with optional class filtering"""
    try:
        # Prepare class filter parameter
        classes_param = ""
        if selected_classes and len(selected_classes) > 0:
            # Convert class names to indices
            from ultralytics import YOLO
            temp_model = YOLO(model_path)
            class_indices = []
            for cls_name in selected_classes:
                for idx, name in temp_model.names.items():
                    if name == cls_name:
                        class_indices.append(idx)
                        break
            if class_indices:
                classes_param = f"classes={class_indices},"
        
        # Create inference script
        script_content = f"""
import sys
from ultralytics import YOLO
import os

print("Loading model: {model_path}")
model = YOLO('{model_path}')

print("Running inference...")
results = model.predict(
    source='{image_path}',
    conf={confidence},
    iou={iou_threshold},
    device='{device}',
    {classes_param}
    save=True,
    project='/tmp/inference_results',
    name='latest',
    exist_ok=True,
    show_labels=True,
    show_conf=True,
    line_width=2
)

print("Inference completed!")
print(f"Results saved to: /tmp/inference_results/latest/")

# Print detection results
if results:
    result = results[0]
    if result.boxes is not None:
        print(f"Detected {{len(result.boxes)}} objects:")
        for i, box in enumerate(result.boxes):
            conf = float(box.conf)
            cls_id = int(box.cls)
            class_name = model.names[cls_id] if hasattr(model, 'names') and cls_id in model.names else f"Class {{cls_id}}"
            print(f"  - {{class_name}}: {{conf:.3f}}")
    else:
        print("No objects detected")
else:
    print("No results returned")
"""
        
        # Write and execute inference script
        with open('/tmp/inference_script.py', 'w') as f:
            f.write(script_content)
        
        result = subprocess.run(
            ["python", "/tmp/inference_script.py"],
            cwd="/ultralytics",
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Check for result image
            result_image_path = Path("/tmp/inference_results/latest") / Path(image_path).name
            if result_image_path.exists():
                return True, str(result_image_path), result.stdout
            else:
                return False, "Result image not found", result.stdout
        else:
            return False, "Inference failed", result.stderr
            
    except Exception as e:
        return False, f"Error: {str(e)}", ""

# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #28a745 0%, #1e7e34 100%); color: white; 
            padding: 1rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
    <h1>🔍 RCT Detector Inference</h1>
    <p>Professional Object Detection & Analysis</p>
</div>
""", unsafe_allow_html=True)

# Navigation
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🏠 Back to Dashboard"):
        st.switch_page("main_dashboard.py")

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: #28a745; color: white; padding: 0.5rem; 
                border-radius: 5px; text-align: center; margin-bottom: 1rem;">
        <h3>Inference Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # GPU Information
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        st.success(f"🖥️ **GPU:** {gpu_info['name']}")
        st.info(f"💾 **Available VRAM:** {gpu_info['available_memory']:.1f} GB / {gpu_info['total_memory']:.1f} GB")
        device_option = "0"
        device_display = f"GPU: {gpu_info['name']}"
    else:
        st.warning(f"⚠️ **Device:** {gpu_info['name']}")
        device_option = "cpu"
        device_display = "CPU"
    
    # Model selection
    models = get_available_models()
    if not models:
        st.error("No models available!")
        st.stop()
    
    model_options = {f"{model['type']} - {model['name']} ({model['size']})": model['path'] for model in models}
    selected_model_name = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        help="Choose a model for inference"
    )
    selected_model_path = model_options[selected_model_name]
    
    # Get and display model classes
    model_classes = get_model_classes(selected_model_path)
    selected_classes = []
    
    if model_classes:
        st.markdown("---")
        st.markdown("### 🎯 Class Filter")
        st.caption("Select specific classes to detect (leave empty for all)")
        
        # Create multi-select with all classes
        class_names = list(model_classes.values())
        selected_classes = st.multiselect(
            "Filter by classes",
            options=class_names,
            default=[],
            help="Select one or more classes to detect. If nothing is selected, all classes will be detected."
        )
        
        if selected_classes:
            st.success(f"✅ Filtering {len(selected_classes)} class(es)")
        else:
            st.info(f"📊 Detecting all {len(class_names)} classes")
    
    st.markdown("---")
    
    # Inference parameters
    confidence = st.slider("Confidence Threshold", 
                          min_value=0.0, max_value=1.0, value=0.25, step=0.05,
                          help="Minimum confidence for detections")
    
    iou_threshold = st.slider("IoU Threshold", 
                             min_value=0.0, max_value=1.0, value=0.45, step=0.05,
                             help="IoU threshold for Non-Maximum Suppression")
    
    st.markdown("---")
    st.info(f"**Device:** {device_display}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a dental X-ray image for detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_image_path = tmp_file.name
        
        # Run inference button
        if st.button("🔍 Run Detection", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                success, result_path, output = run_inference(
                    selected_model_path, temp_image_path, confidence, iou_threshold, device_option, selected_classes
                )
                
                if success:
                    st.success("✅ Detection completed!")
                    
                    # Store results in session state
                    st.session_state['inference_result'] = result_path
                    st.session_state['inference_output'] = output
                    st.rerun()
                else:
                    st.error(f"❌ Detection failed: {result_path}")
                    if output:
                        with st.expander("Error Details"):
                            st.code(output)

with col2:
    st.subheader("📊 Detection Results")
    
    if 'inference_result' in st.session_state and st.session_state['inference_result']:
        result_path = st.session_state['inference_result']
        
        if Path(result_path).exists():
            # Display result image
            result_image = Image.open(result_path)
            st.image(result_image, caption="Detection Results", use_column_width=True)
            
            # Parse detection output
            if 'inference_output' in st.session_state:
                output = st.session_state['inference_output']
                
                # Extract detection information
                if "Detected" in output:
                    lines = output.split('\n')
                    detection_lines = [line for line in lines if line.strip().startswith('- ')]
                    
                    if detection_lines:
                        st.subheader("🎯 Detected Objects")
                        for line in detection_lines:
                            clean_line = line.strip().replace('- ', '')
                            if ':' in clean_line:
                                obj_name, confidence_str = clean_line.split(':', 1)
                                st.metric(obj_name.strip(), confidence_str.strip())
                    else:
                        st.info("No objects detected in the image")
                
                # Show raw output
                with st.expander("📋 Raw Output"):
                    st.code(output)
            
            # Download button for result
            with open(result_path, "rb") as file:
                st.download_button(
                    label="💾 Download Result",
                    data=file.read(),
                    file_name=f"detection_result_{Path(result_path).name}",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.warning("Result image not found")
    else:
        st.info("👆 Upload an image and run detection to see results here")
        
        # Show example
        st.markdown("""
        ### 💡 How to use:
        1. **Select a Model**: Choose from workspace models or pre-trained YOLO models
        2. **Upload Image**: Select a dental X-ray image
        3. **Adjust Settings**: Fine-tune confidence and IoU thresholds
        4. **Run Detection**: Click the detection button
        5. **View Results**: See detected objects with confidence scores
        """)

# Clear results
if st.button("🗑️ Clear Results"):
    if 'inference_result' in st.session_state:
        del st.session_state['inference_result']
    if 'inference_output' in st.session_state:
        del st.session_state['inference_output']
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔍 RCT Detector Inference - Professional Detection with Custom Models</p>
</div>
""", unsafe_allow_html=True)
