#!/bin/bash

echo "🚀 Starting Ultralytics AI System..."

# Start TensorBoard in background
echo "📊 Starting TensorBoard..."
cd /ultralytics && tensorboard --logdir=runs --host=0.0.0.0 --port=6006 &

# Wait a bit for TensorBoard to start
sleep 2

echo "✅ Services starting:"
echo "  📊 TensorBoard: http://localhost:6006"
echo "  🌐 Streamlit UI: http://localhost:8501"
echo "  📓 Jupyter Lab: http://localhost:8888"

# Start Streamlit interface in foreground (keeps container alive)
echo "🌐 Starting Streamlit Interface..."
cd /ultralytics && streamlit run main_dashboard.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.maxUploadSize 200
