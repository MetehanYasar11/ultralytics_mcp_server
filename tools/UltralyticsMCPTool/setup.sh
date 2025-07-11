#!/bin/bash

# Ultralytics MCP Tool Setup Script
echo "🚀 Setting up Ultralytics MCP Tool..."

# Change to the tool directory
cd "$(dirname "$0")"

# Install Node.js dependencies
echo "📦 Installing dependencies..."
npm install

# Build the TypeScript project
echo "🔨 Building TypeScript project..."
npm run build

# Create symlink for global usage (optional)
echo "🔗 Creating global symlink..."
npm link

echo "✅ Setup complete!"
echo ""
echo "Usage:"
echo "  import UltralyticsMCPTool from 'ultralytics-mcp-tool';"
echo "  const tool = new UltralyticsMCPTool('http://localhost:8000');"
echo ""
echo "Run examples:"
echo "  npm run build && node dist/examples.js"
