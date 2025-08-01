#!/bin/bash

echo "🚀 Starting Ultralytics AI System..."

# Start TensorBoard in background
echo "📊 Starting TensorBoard..."
cd /ultralytics && tensorboard --logdir=runs --host=0.0.0.0 --port=6006 &

# Wait a bit for TensorBoard to start
sleep 2

# Start Streamlit in background  
echo "🌐 Starting Enhanced Streamlit Interface..."
cd /workspace && streamlit run enhanced_streamlit_inference.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.maxUploadSize 200 &

# Wait a bit for Streamlit to start
sleep 3

echo "✅ All services started:"
echo "  🌐 Streamlit UI: http://localhost:8501"
echo "  📊 TensorBoard: http://localhost:6006"
echo "  📓 Jupyter Lab: http://localhost:8888"

# Start Jupyter Lab (keeps container running)
exec jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''
