import streamlit as st
import subprocess
import os
import time
from pathlib import Path
import pandas as pd
import shutil
import torch
import yaml
import zipfile
import tempfile
import re

# Page configuration
st.set_page_config(
    page_title="DENTEX AI Training Center",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

def get_nvidia_smi_info():
    """Get live GPU information from nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(', ')
            return {
                'name': gpu_data[0],
                'temperature': int(gpu_data[1]),
                'gpu_utilization': int(gpu_data[2]),
                'memory_utilization': int(gpu_data[3]),
                'memory_used': int(gpu_data[4]),
                'memory_total': int(gpu_data[5]),
                'power_draw': float(gpu_data[6]),
                'power_limit': float(gpu_data[7]),
                'available': True
            }
    except Exception as e:
        st.error(f"nvidia-smi error: {e}")
    
    # Fallback to torch
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
                'memory_total_gb': gpu_memory_gb,
                'memory_available_gb': gpu_memory_available_gb,
                'available': True,
                'torch_fallback': True
            }
        else:
            return {'name': 'No GPU Available', 'available': False}
    except Exception as e:
        return {'name': f'GPU Error: {str(e)}', 'available': False}

def get_dentex_datasets():
    """Get available DENTEX datasets"""
    dentex_path = Path("/DENTEX/YOLO_MultiLevel_Datasets")
    if not dentex_path.exists():
        return []
    
    datasets = []
    for folder in dentex_path.iterdir():
        if folder.is_dir() and folder.name.startswith("YOLO_"):
            yaml_file = folder / "data.yaml"
            if yaml_file.exists():
                datasets.append({
                    'name': folder.name,
                    'path': str(yaml_file),
                    'description': folder.name.replace("YOLO_", "").replace("_", " ")
                })
    return datasets

def get_custom_datasets():
    """Get uploaded custom datasets"""
    custom_path = Path("/ultralytics/custom_datasets")
    if not custom_path.exists():
        return []
    
    datasets = []
    for folder in custom_path.iterdir():
        if folder.is_dir():
            yaml_file = folder / "data.yaml"
            if yaml_file.exists():
                datasets.append({
                    'name': f"Custom: {folder.name}",
                    'path': str(yaml_file),
                    'description': f"Custom Dataset: {folder.name}"
                })
    return datasets

def validate_yolo_dataset(zip_path):
    """Validate uploaded YOLO dataset"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Check for required structure
            has_images = any('images/' in f for f in file_list)
            has_labels = any('labels/' in f for f in file_list)
            has_data_yaml = any(f.endswith('data.yaml') or f.endswith('data.yml') for f in file_list)
            
            if not (has_images and has_labels and has_data_yaml):
                return False, "Dataset must contain images/, labels/ folders and data.yaml file"
            
            return True, "Valid YOLO dataset structure"
    except Exception as e:
        return False, f"Error validating dataset: {str(e)}"

def extract_custom_dataset(uploaded_file, dataset_name):
    """Extract and setup custom dataset"""
    try:
        # Create custom datasets directory
        custom_datasets_dir = Path("/ultralytics/custom_datasets")
        custom_datasets_dir.mkdir(exist_ok=True)
        
        dataset_dir = custom_datasets_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir()
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Validate dataset
        is_valid, message = validate_yolo_dataset(tmp_path)
        if not is_valid:
            os.unlink(tmp_path)
            return False, message
        
        # Extract dataset
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Update data.yaml paths
        data_yaml_files = list(dataset_dir.rglob("data.yaml")) + list(dataset_dir.rglob("data.yml"))
        if data_yaml_files:
            data_yaml = data_yaml_files[0]
            with open(data_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update paths to be absolute
            base_path = str(dataset_dir)
            if 'train' in config:
                config['train'] = os.path.join(base_path, config['train'])
            if 'val' in config:
                config['val'] = os.path.join(base_path, config['val'])
            if 'test' in config:
                config['test'] = os.path.join(base_path, config['test'])
            
            with open(data_yaml, 'w') as f:
                yaml.dump(config, f)
        
        os.unlink(tmp_path)
        return True, f"Dataset '{dataset_name}' uploaded successfully"
        
    except Exception as e:
        return False, f"Error extracting dataset: {str(e)}"

def check_training_status():
    """Check if training is currently running"""
    try:
        result = subprocess.run(
            ["bash", "-c", "ps aux | grep -E 'train_script|model.train' | grep -v grep | wc -l"],
            cwd="/ultralytics",
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip()) > 0
    except:
        return False

def get_latest_training_results():
    """Get latest training results"""
    try:
        runs_dir = Path("/ultralytics/runs/detect")
        if not runs_dir.exists():
            return None
            
        # Find latest dentex training
        dentex_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d.name.startswith("dentex_") or d.name.startswith("custom_"))]
        if not dentex_dirs:
            return None
            
        latest_dir = max(dentex_dirs, key=lambda x: x.stat().st_mtime)
        results_file = latest_dir / "results.csv"
        
        if results_file.exists():
            df = pd.read_csv(results_file)
            return {
                'training_dir': str(latest_dir.name),
                'training_path': str(latest_dir),
                'epochs_completed': len(df),
                'current_metrics': df.iloc[-1].to_dict() if len(df) > 0 else {},
                'df': df
            }
    except Exception as e:
        st.error(f"Error reading results: {e}")
        return None

def save_model_to_workspace(training_path, model_name):
    """Save trained model to workspace"""
    try:
        weights_dir = Path(training_path) / "weights"
        if weights_dir.exists():
            best_model = weights_dir / "best.pt"
            last_model = weights_dir / "last.pt"
            
            # Create workspace models directory
            workspace_models = Path("/workspace/trained_models")
            workspace_models.mkdir(exist_ok=True)
            
            # Copy models
            if best_model.exists():
                shutil.copy2(best_model, workspace_models / f"{model_name}_best.pt")
            if last_model.exists():
                shutil.copy2(last_model, workspace_models / f"{model_name}_last.pt")
            
            return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False

# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #0066CC 0%, #004499 100%); color: white; 
            padding: 1rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
    <h1>🦷 DENTEX AI Training Center</h1>
    <p>Professional Dental X-Ray Detection Model Training with Custom Dataset Support</p>
</div>
""", unsafe_allow_html=True)

# Navigation
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🏠 Back to Dashboard"):
        st.switch_page("main_dashboard.py")

# Check training status
training_active = check_training_status()
if training_active:
    st.warning("🔄 **Training is currently active!** Monitor progress below.")

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: #0066CC; color: white; padding: 0.5rem; 
                border-radius: 5px; text-align: center; margin-bottom: 1rem;">
        <h3>Training Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # GPU Information
    gpu_info = get_nvidia_smi_info()
    if gpu_info['available']:
        if 'torch_fallback' in gpu_info:
            st.success(f"🖥️ **GPU:** {gpu_info['name']}")
            st.info(f"💾 **Available VRAM:** {gpu_info['memory_available_gb']:.1f} GB / {gpu_info['memory_total_gb']:.1f} GB")
        else:
            st.success(f"🖥️ **GPU:** {gpu_info['name']}")
            st.info(f"💾 **VRAM:** {gpu_info['memory_used']} MB / {gpu_info['memory_total']} MB")
            st.info(f"🌡️ **Temp:** {gpu_info['temperature']}°C | ⚡ **Power:** {gpu_info['power_draw']:.1f}W")
        device_option = "0"
        device_display = f"GPU: {gpu_info['name']}"
    else:
        st.warning(f"⚠️ **Device:** {gpu_info['name']}")
        device_option = "cpu"
        device_display = "CPU"
    
    # Dataset selection
    st.markdown("### Dataset Selection")
    
    # Custom dataset upload
    with st.expander("📁 Upload Custom Dataset"):
        uploaded_file = st.file_uploader(
            "Choose a YOLO dataset ZIP file",
            type=['zip'],
            help="Upload a ZIP file containing YOLO format dataset (images/, labels/, data.yaml)"
        )
        
        if uploaded_file is not None:
            dataset_name = st.text_input("Dataset Name", value="my_custom_dataset")
            
            if st.button("🚀 Upload Dataset"):
                with st.spinner("Uploading and extracting dataset..."):
                    success, message = extract_custom_dataset(uploaded_file, dataset_name)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    # Get all available datasets
    dentex_datasets = get_dentex_datasets()
    custom_datasets = get_custom_datasets()
    all_datasets = dentex_datasets + custom_datasets
    
    if not all_datasets:
        st.error("No datasets found! Upload a custom dataset or check DENTEX datasets.")
        st.stop()
    
    dataset_options = {ds['description']: ds['path'] for ds in all_datasets}
    selected_dataset_name = st.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        help="Choose a dataset for training"
    )
    selected_dataset_path = dataset_options[selected_dataset_name]
    
    st.markdown("---")
    
    # Training parameters
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=5, step=1, 
                            help="Number of training epochs")
    
    batch_size = st.select_slider("Batch Size", 
                                 options=[4, 8, 16, 32, 64, 128, 256], 
                                 value=8,
                                 help="Batch size for training")
    
    img_size = st.select_slider("Image Size", 
                               options=[320, 416, 480, 512, 640, 736, 832, 896, 960, 1024, 1152, 1280], 
                               value=640,
                               help="Input image size")
    
    # Model name input
    st.markdown("---")
    model_name = st.text_input("Model Name", 
                              value=f"model_{selected_dataset_name.lower().replace(' ', '_').replace(':', '_')}", 
                              help="Name for the trained model")
    
    st.markdown("---")
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", help="Refresh training status"):
            st.rerun()
    
    with col2:
        if st.button("🛑 Stop Training", disabled=not training_active):
            if training_active:
                subprocess.run(["pkill", "-f", "train_script"], cwd="/ultralytics")
                st.success("Training stopped!")
                time.sleep(1)
                st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if training_active:
        st.subheader("🔄 Live Training Progress")
        
        # Get latest results
        results = get_latest_training_results()
        if results:
            st.success(f"**Training Active:** {results['training_dir']}")
            
            # Progress metrics
            current_epoch = results['epochs_completed']
            metrics = results['current_metrics']
            
            # Progress bar
            progress = min(current_epoch / epochs, 1.0)
            st.progress(progress)
            st.text(f"Epoch: {current_epoch}/{epochs}")
            
            # Current metrics
            if metrics:
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Box Loss", f"{metrics.get('train/box_loss', 0):.3f}")
                with metric_cols[1]:
                    st.metric("Class Loss", f"{metrics.get('train/cls_loss', 0):.3f}")
                with metric_cols[2]:
                    st.metric("mAP50", f"{metrics.get('metrics/mAP50(B)', 0):.3f}")
                with metric_cols[3]:
                    st.metric("Precision", f"{metrics.get('metrics/precision(B)', 0):.3f}")
            
            # Training plot
            if len(results['df']) > 1:
                st.subheader("Training Metrics")
                
                # Plot metrics
                chart_cols = st.columns(2)
                with chart_cols[0]:
                    st.line_chart(results['df'][['train/box_loss', 'val/box_loss']].set_index(results['df'].index))
                    st.caption("Box Loss (Train vs Val)")
                
                with chart_cols[1]:
                    st.line_chart(results['df'][['metrics/mAP50(B)']].set_index(results['df'].index))
                    st.caption("mAP50 Progress")
            
            # Check if training completed
            if current_epoch >= epochs and not training_active:
                st.success("🎉 **Training Completed!**")
                
                # Save model to workspace
                if st.button("💾 Save Model to Workspace"):
                    if save_model_to_workspace(results['training_path'], model_name):
                        st.success(f"✅ Model '{model_name}' saved to workspace!")
                    else:
                        st.error("❌ Failed to save model to workspace")
        
        # Auto refresh every 5 seconds during training
        time.sleep(5)
        st.rerun()
        
    else:
        st.subheader("Start New Training")
        
        # Configuration summary
        st.info(f"""
        **Selected Dataset:** {selected_dataset_name}  
        **Epochs:** {epochs}  
        **Batch Size:** {batch_size}  
        **Image Size:** {img_size}px  
        **Device:** {device_display}  
        **Model Name:** {model_name}
        """)
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Starting YOLO training..."):
                try:
                    # Create training script
                    training_prefix = "custom" if "Custom:" in selected_dataset_name else "dentex"
                    script_content = f"""
import os
from ultralytics import YOLO
from datetime import datetime
import shutil
from pathlib import Path

print("Starting training...")
print(f"Dataset: {selected_dataset_path}")
print(f"Epochs: {epochs}")
print(f"Batch: {batch_size}")
print(f"Image size: {img_size}")
print(f"Device: {device_option}")
print(f"Model name: {model_name}")

model = YOLO('yolo11n.pt')

time_str = datetime.now().strftime('%H%M%S')
training_name = f'{training_prefix}_{{time_str}}'

results = model.train(
    data='{selected_dataset_path}',
    epochs={epochs},
    imgsz={img_size},
    batch={batch_size},
    device='{device_option}',
    project='/ultralytics/runs/detect',
    name=training_name,
    exist_ok=True,
    verbose=True,
    save=True
)

print("Training completed successfully!")
print(f"Results saved to: {{results.save_dir}}")

# Auto-save to workspace
try:
    workspace_models = Path("/workspace/trained_models")
    workspace_models.mkdir(exist_ok=True)
    
    weights_dir = Path(results.save_dir) / "weights"
    if weights_dir.exists():
        best_model = weights_dir / "best.pt"
        last_model = weights_dir / "last.pt"
        
        if best_model.exists():
            shutil.copy2(best_model, workspace_models / f"{model_name}_best.pt")
            print(f"Best model saved to workspace: {model_name}_best.pt")
        if last_model.exists():
            shutil.copy2(last_model, workspace_models / f"{model_name}_last.pt")
            print(f"Last model saved to workspace: {model_name}_last.pt")
    
    print("Models automatically saved to workspace!")
except Exception as e:
    print(f"Error saving to workspace: {{e}}")
"""
                    
                    # Write training script
                    with open('/tmp/train_script.py', 'w') as f:
                        f.write(script_content)
                    
                    # Start training in background
                    subprocess.Popen(
                        ["python", "/tmp/train_script.py"],
                        cwd="/ultralytics",
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    
                    st.success("✅ Training started! Refreshing to show progress...")
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error starting training: {str(e)}")

with col2:
    st.subheader("🖥️ System Information")
    
    # Live GPU monitoring with auto-refresh
    if 'system_refresh' not in st.session_state:
        st.session_state.system_refresh = 0
    
    # GPU Info Card with live data
    gpu_info = get_nvidia_smi_info()
    if gpu_info['available']:
        if 'torch_fallback' in gpu_info:
            st.markdown(f"""
            <div style="background: #f0f8ff; border: 2px solid #0066CC; color: #000000; 
                        padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="color: #0066CC; margin-top: 0;">🖥️ GPU Status (PyTorch)</h4>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Device:</strong> {gpu_info['name']}</p>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Total VRAM:</strong> {gpu_info['memory_total_gb']:.1f} GB</p>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Available:</strong> {gpu_info['memory_available_gb']:.1f} GB</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            memory_usage_percent = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100
            power_usage_percent = (gpu_info['power_draw'] / gpu_info['power_limit']) * 100
            
            # Temperature color coding
            temp_color = "#28a745" if gpu_info['temperature'] < 70 else "#ffc107" if gpu_info['temperature'] < 80 else "#dc3545"
            
            st.markdown(f"""
            <div style="background: #f0f8ff; border: 2px solid #28a745; color: #000000; 
                        padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="color: #28a745; margin-top: 0;">🖥️ GPU Status (Live)</h4>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Device:</strong> {gpu_info['name']}</p>
                <p style="color: {temp_color}; margin: 0.3rem 0;"><strong>Temperature:</strong> {gpu_info['temperature']}°C</p>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>GPU Load:</strong> {gpu_info['gpu_utilization']}%</p>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Memory:</strong> {gpu_info['memory_used']} MB / {gpu_info['memory_total']} MB ({memory_usage_percent:.1f}%)</p>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Power:</strong> {gpu_info['power_draw']:.1f}W / {gpu_info['power_limit']:.1f}W ({power_usage_percent:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #fff3cd; border: 2px solid #ffc107; color: #000000; 
                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="color: #856404; margin-top: 0;">⚠️ Device Status</h4>
            <p style="color: #000000; margin: 0.3rem 0;">{gpu_info['name']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-refresh system info every 10 seconds
    if st.session_state.system_refresh % 10 == 0:
        time.sleep(1)
        st.session_state.system_refresh += 1
        st.rerun()
    else:
        st.session_state.system_refresh += 1
    
    st.subheader("📊 Training History")
    
    # Show recent training results
    results = get_latest_training_results()
    if results:
        with st.expander(f"📊 Latest: {results['training_dir']}"):
            st.text(f"Epochs completed: {results['epochs_completed']}")
            if results['current_metrics']:
                metrics = results['current_metrics']
                st.text(f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f}")
                st.text(f"Box Loss: {metrics.get('train/box_loss', 0):.3f}")
    
    st.markdown("---")
    st.subheader("💾 Workspace Models")
    
    # Show saved models
    workspace_models = Path("/workspace/trained_models")
    if workspace_models.exists():
        model_files = list(workspace_models.glob("*.pt"))
        if model_files:
            for model_file in sorted(model_files):
                file_size = model_file.stat().st_size / (1024*1024)  # MB
                st.text(f"📦 {model_file.name} ({file_size:.1f} MB)")
        else:
            st.text("No models saved yet")
    else:
        st.text("No workspace models directory")
    
    st.markdown("---")
    st.subheader("📁 Dataset Information")
    
    # Show dataset info
    col_dentex, col_custom = st.columns(2)
    
    with col_dentex:
        st.write("**DENTEX Datasets:**")
        if dentex_datasets:
            for dataset in dentex_datasets:
                st.text(f"• {dataset['description']}")
        else:
            st.text("No DENTEX datasets")
    
    with col_custom:
        st.write("**Custom Datasets:**")
        if custom_datasets:
            for dataset in custom_datasets:
                st.text(f"• {dataset['description'].replace('Custom Dataset: ', '')}")
        else:
            st.text("No custom datasets")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🦷 DENTEX AI Training Center - Professional Training with Custom Dataset Support & Live GPU Monitoring</p>
</div>
""", unsafe_allow_html=True)
