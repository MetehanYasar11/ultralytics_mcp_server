#!/bin/bash
# Simple system monitor for container
echo "=== System Status Check ==="
echo "Memory Usage:"
free -h | grep Mem:
echo ""
echo "Streamlit Process:"
ps aux | grep streamlit | grep -v grep | head -1
echo ""
echo "Disk Space:"
df -h /ultralytics | tail -1
echo ""
echo "Custom Datasets:"
ls -la /ultralytics/custom_datasets/ 2>/dev/null || echo "No custom datasets directory"
echo ""
