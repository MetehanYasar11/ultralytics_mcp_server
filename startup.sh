#!/bin/bash
set -e

echo "🚀 Starting DENTEX AI Platform..."

# Create necessary directories
mkdir -p /ultralytics/runs
mkdir -p /ultralytics/custom_datasets
mkdir -p /tmp/streamlit_uploads
mkdir -p /workspace/trained_models

# Set proper permissions
chmod 755 /ultralytics/upload_processor.py
chmod 755 /usr/local/bin/monitor.sh

# Check if DENTEX datasets are mounted
if [ -d "/DENTEX/YOLO_MultiLevel_Datasets" ]; then
    echo "✅ DENTEX datasets found at /DENTEX/YOLO_MultiLevel_Datasets"
    ls -la /DENTEX/YOLO_MultiLevel_Datasets/ | head -5
else
    echo "⚠️ DENTEX datasets not found at /DENTEX/YOLO_MultiLevel_Datasets"
fi

# Check workspace
if [ -d "/workspace" ]; then
    echo "✅ Workspace mounted at /workspace"
    ls -la /workspace/ | head -5
else
    echo "⚠️ Workspace not found"
fi

# Start TensorBoard in background
echo "📊 Starting TensorBoard..."
cd /ultralytics
nohup tensorboard --logdir=runs --host=0.0.0.0 --port=6006 > /tmp/tensorboard.log 2>&1 &
TENSORBOARD_PID=$!

# Wait for TensorBoard to start
sleep 3

# Check if TensorBoard started successfully
if ps -p $TENSORBOARD_PID > /dev/null; then
    echo "✅ TensorBoard started successfully (PID: $TENSORBOARD_PID)"
else
    echo "⚠️ TensorBoard failed to start, checking logs..."
    cat /tmp/tensorboard.log | tail -10
fi

# Start Jupyter in background (optional)
echo "📓 Starting Jupyter Lab..."
cd /ultralytics
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' > /tmp/jupyter.log 2>&1 &
JUPYTER_PID=$!

sleep 2

if ps -p $JUPYTER_PID > /dev/null; then
    echo "✅ Jupyter Lab started successfully (PID: $JUPYTER_PID)"
else
    echo "⚠️ Jupyter Lab failed to start"
fi

echo ""
echo "🌐 Services Status:"
echo "  📊 TensorBoard: http://localhost:6006"
echo "  🌐 Streamlit UI: http://localhost:8501"
echo "  📓 Jupyter Lab: http://localhost:8888"
echo "  🔌 MCP Server: http://localhost:8092"
echo ""

# Start Streamlit interface in foreground (keeps container alive)
echo "🌐 Starting Streamlit Interface..."
cd /ultralytics

exec streamlit run main_dashboard.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.maxUploadSize 10000 \
    --browser.gatherUsageStats false \
    --global.developmentMode false
