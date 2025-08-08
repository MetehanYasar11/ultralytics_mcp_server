import streamlit as st
import subprocess
import torch
import os
from pathlib import Path
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="DENTEX AI Platform",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        pass
    
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

def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # CPU Usage
        cpu_result = subprocess.run(
            ["bash", "-c", "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | sed 's/%us,//'"],
            capture_output=True,
            text=True
        )
        cpu_usage = cpu_result.stdout.strip() if cpu_result.returncode == 0 else "N/A"
        
        # Memory Usage
        mem_result = subprocess.run(
            ["bash", "-c", "free -h | grep '^Mem:' | awk '{print $3 \"/\" $2}'"],
            capture_output=True,
            text=True
        )
        memory_usage = mem_result.stdout.strip() if mem_result.returncode == 0 else "N/A"
        
        # Disk Usage
        disk_result = subprocess.run(
            ["bash", "-c", "df -h / | tail -1 | awk '{print $3 \"/\" $2 \" (\" $5 \")\"}'"],
            capture_output=True,
            text=True
        )
        disk_usage = disk_result.stdout.strip() if disk_result.returncode == 0 else "N/A"
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage
        }
    except:
        return {
            'cpu_usage': "N/A",
            'memory_usage': "N/A",
            'disk_usage': "N/A"
        }

def get_latest_training_results():
    """Get latest training results"""
    try:
        runs_dir = Path("/ultralytics/runs/detect")
        if not runs_dir.exists():
            return None
            
        # Find latest training
        training_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d.name.startswith("dentex_") or d.name.startswith("custom_"))]
        if not training_dirs:
            return None
            
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        results_file = latest_dir / "results.csv"
        
        if results_file.exists():
            df = pd.read_csv(results_file)
            return {
                'training_dir': str(latest_dir.name),
                'epochs_completed': len(df),
                'current_metrics': df.iloc[-1].to_dict() if len(df) > 0 else {},
            }
    except Exception as e:
        return None

def get_workspace_models():
    """Get list of workspace models"""
    try:
        workspace_models = Path("/workspace/trained_models")
        if workspace_models.exists():
            model_files = list(workspace_models.glob("*.pt"))
            return len(model_files)
        return 0
    except:
        return 0

def get_available_pretrained_models():
    """Get count of available pre-trained YOLO models"""
    # Count of all available YOLO variants
    return 17  # 5 YOLO11 + 5 YOLOv8 + 5 YOLOv9 + 2 special variants

# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #0066CC 0%, #004499 100%); color: white; 
            padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
    <h1>🦷 DENTEX AI Platform</h1>
    <p style="font-size: 1.2em;">Professional Dental X-Ray Detection & Training Platform</p>
</div>
""", unsafe_allow_html=True)

# Navigation
st.markdown("### 🚀 Quick Navigation")

col1, col2 = st.columns(2)

with col1:
    if st.button("🎯 **Train Models**", use_container_width=True, type="primary"):
        st.switch_page("pages/train.py")
    st.markdown("""
    <div style="background: #f0f8ff; border: 1px solid #0066CC; padding: 1rem; 
                border-radius: 8px; margin-top: 0.5rem;">
        <p style="margin: 0; color: #000;"><strong>Model Training</strong></p>
        <p style="margin: 0; color: #666; font-size: 0.9em;">Train YOLO models on DENTEX datasets or custom data</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("🔍 **Inference & Detection**", use_container_width=True, type="secondary"):
        st.switch_page("pages/inference.py")
    st.markdown("""
    <div style="background: #f0fff0; border: 1px solid #28a745; padding: 1rem; 
                border-radius: 8px; margin-top: 0.5rem;">
        <p style="margin: 0; color: #000;"><strong>Model Inference</strong></p>
        <p style="margin: 0; color: #666; font-size: 0.9em;">Run detection on images using trained models</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# System Status Dashboard
st.markdown("### 📊 System Status Dashboard")

# Get system information
gpu_info = get_nvidia_smi_info()
system_stats = get_system_stats()
training_active = check_training_status()
latest_training = get_latest_training_results()
model_count = get_workspace_models()
pretrained_models = get_available_pretrained_models()

# Status Overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    if training_active:
        st.metric("🔄 Training Status", "ACTIVE", delta="Running")
    else:
        st.metric("⏸️ Training Status", "IDLE", delta="Ready")

with col2:
    if gpu_info['available']:
        if 'torch_fallback' in gpu_info:
            st.metric("💾 GPU Memory", f"{gpu_info['memory_available_gb']:.1f} GB", delta="Available")
        else:
            memory_usage_percent = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100
            st.metric("💾 GPU Memory", f"{memory_usage_percent:.1f}%", delta="Used")
    else:
        st.metric("💾 GPU Memory", "N/A", delta="No GPU")

with col3:
    total_models = model_count + pretrained_models
    st.metric("🤖 Available Models", str(total_models), delta=f"{model_count} Custom + {pretrained_models} Pre-trained")

with col4:
    if latest_training:
        st.metric("📈 Last Training", f"{latest_training['epochs_completed']} epochs", delta="Completed")
    else:
        st.metric("📈 Last Training", "None", delta="No history")

st.markdown("---")

# Detailed System Information
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🖥️ Hardware Status")
    
    # GPU Information
    if gpu_info['available']:
        if 'torch_fallback' in gpu_info:
            st.markdown(f"""
            <div style="background: #f0f8ff; border: 2px solid #0066CC; color: #000000; 
                        padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="color: #0066CC; margin-top: 0;">🖥️ GPU Status</h4>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Device:</strong> {gpu_info['name']}</p>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Total VRAM:</strong> {gpu_info['memory_total_gb']:.1f} GB</p>
                <p style="color: #000000; margin: 0.3rem 0;"><strong>Available:</strong> {gpu_info['memory_available_gb']:.1f} GB</p>
                <p style="color: #28a745; margin: 0.3rem 0;"><strong>Status:</strong> ✅ Ready for Training</p>
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
            <h4 style="color: #856404; margin-top: 0;">⚠️ GPU Status</h4>
            <p style="color: #000000; margin: 0.3rem 0;">{gpu_info['name']}</p>
            <p style="color: #dc3545; margin: 0.3rem 0;"><strong>Status:</strong> ❌ GPU Not Available</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System Resources
    st.markdown(f"""
    <div style="background: #f8f9fa; border: 2px solid #6c757d; color: #000000; 
                padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #495057; margin-top: 0;">💻 System Resources</h4>
        <p style="color: #000000; margin: 0.3rem 0;"><strong>CPU Usage:</strong> {system_stats['cpu_usage']}</p>
        <p style="color: #000000; margin: 0.3rem 0;"><strong>Memory:</strong> {system_stats['memory_usage']}</p>
        <p style="color: #000000; margin: 0.3rem 0;"><strong>Disk:</strong> {system_stats['disk_usage']}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("📈 Quick Stats")
    
    # Training status
    if training_active:
        st.success("🔄 **Training Active**")
        if latest_training:
            st.info(f"📊 **Current:** {latest_training['training_dir']}")
            st.info(f"🔢 **Epochs:** {latest_training['epochs_completed']}")
            if latest_training['current_metrics']:
                metrics = latest_training['current_metrics']
                st.info(f"📈 **mAP50:** {metrics.get('metrics/mAP50(B)', 0):.3f}")
    else:
        st.info("⏸️ **No Active Training**")
    
    st.markdown("---")
    
    # Available datasets
    st.subheader("📊 Available Data")
    
    # DENTEX datasets
    dentex_path = Path("/DENTEX/YOLO_MultiLevel_Datasets")
    if dentex_path.exists():
        dentex_count = len([d for d in dentex_path.iterdir() if d.is_dir() and d.name.startswith("YOLO_")])
        st.text(f"🦷 DENTEX Datasets: {dentex_count}")
    else:
        st.text("🦷 DENTEX Datasets: 0")
    
    # Custom datasets
    custom_path = Path("/ultralytics/custom_datasets")
    if custom_path.exists():
        custom_count = len([d for d in custom_path.iterdir() if d.is_dir()])
        st.text(f"📁 Custom Datasets: {custom_count}")
    else:
        st.text("📁 Custom Datasets: 0")
    
    st.text(f"🤖 Workspace Models: {model_count}")

# Auto-refresh every 30 seconds
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0

st.session_state.refresh_counter += 1
if st.session_state.refresh_counter % 30 == 0:
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🦷 DENTEX AI Platform - Professional Dental Detection with Real-Time Monitoring</p>
    <p style="font-size: 0.9em;">Live GPU monitoring • Custom dataset support • Workspace model management</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()
