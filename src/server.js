#!/usr/bin/env node
/**
 * Ultralytics MCP Server with SSE Support - Docker Version
 */

const express = require('express');
const { randomUUID } = require('node:crypto');
const { exec } = require('child_process');
const util = require('util');

// Use Docker environment paths for MCP SDK
const { McpServer } = require('/app/node_modules/@modelcontextprotocol/sdk/dist/cjs/server/mcp.js');
const { SSEServerTransport } = require('/app/node_modules/@modelcontextprotocol/sdk/dist/cjs/server/sse.js');
const { z } = require('zod');

const execAsync = util.promisify(exec);

/**
 * Execute Python code in Ultralytics container via Docker exec
 */
async function executeInUltralyticsContainer(pythonCode) {
  try {
    console.log('🐍 Executing Python code in Ultralytics container...');
    
    // Use base64 encoding to safely pass Python code to avoid shell escaping issues
    const encodedCode = Buffer.from(pythonCode).toString('base64');
    const dockerCommand = `docker exec ultralytics-container python3 -c "import base64; exec(base64.b64decode('${encodedCode}').decode('utf-8'))"`;
    
    const { stdout, stderr } = await execAsync(dockerCommand);
    
    if (stderr && !stderr.includes('WARNING')) {
      console.error('🚨 Python execution stderr:', stderr);
    }
    
    const result = stdout || stderr || 'Code executed successfully (no output)';
    console.log('✅ Python execution completed');
    return result;
    
  } catch (error) {
    console.error('❌ Python execution failed:', error);
    const errorMsg = error.stderr || error.message || 'Unknown execution error';
    return `❌ Python execution failed: ${errorMsg}`;
  }
}

/**
 * Create the MCP Server with Ultralytics tools
 */
const createUltralyticsServer = () => {
  const server = new McpServer({
    name: 'ultralytics_mcp',
    version: '1.0.0',
  }, { 
    capabilities: { 
      tools: {},
      logging: {}
    } 
  });

  // Tool 1: Python Code Execution
  server.tool(
    'execute_python_code',
    'Execute Python code with Ultralytics libraries',
    {
      code: z.string().describe('Python code to execute')
    },
    async ({ code }) => {
      try {
        console.log('🐍 Executing Python code:', code.substring(0, 50) + '...');
        
        const result = await executeInUltralyticsContainer(code);
        
        return {
          content: [
            {
              type: 'text',
              text: `✅ Python code executed!\nResult:\n${result}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ Python execution failed: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  // Tool 2: YOLO Operations
  server.tool(
    'yolo_operation',
    'Perform YOLO operations',
    {
      operation: z.enum(['detect', 'train', 'predict']).describe('YOLO operation'),
      model: z.string().default('yolo11n.pt').describe('YOLO model'),
      source: z.string().optional().describe('Input source (image/video path)')
    },
    async ({ operation, model, source }) => {
      try {
        console.log(`🎯 YOLO ${operation} with ${model}`);
        
        // Build YOLO command
        let yoloCommand = `from ultralytics import YOLO; model = YOLO('${model}'); `;
        
        switch (operation) {
          case 'detect':
          case 'predict':
            if (source) {
              yoloCommand += `results = model.predict('${source}', save=True); print(f"Detection completed. Found {len(results)} result(s). Results saved to: {results[0].save_dir if hasattr(results[0], 'save_dir') else 'runs/detect/predict'}")`;
            } else {
              yoloCommand += `print(f"Model {model} loaded and ready for detection")`;
            }
            break;
          case 'train':
            if (source) {
              yoloCommand += `results = model.train(data='${source}', epochs=3, imgsz=640, save=True); print(f"Training completed! Results saved to: {results.save_dir if hasattr(results, 'save_dir') else 'runs/detect/train'}")`;
            } else {
              yoloCommand += `results = model.train(data='coco8.yaml', epochs=3, imgsz=640, save=True); print(f"Training completed with coco8 dataset! Results saved to: {results.save_dir if hasattr(results, 'save_dir') else 'runs/detect/train'}")`;
            }
            break;
        }
        
        const result = await executeInUltralyticsContainer(yoloCommand);
        
        return {
          content: [
            {
              type: 'text',
              text: `✅ YOLO ${operation} completed with ${model}!\nResult:\n${result}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `❌ YOLO operation failed: ${error.message}`
            }
          ],
          isError: true
        };
      }
    }
  );

  return server;
};

// Create Express application
const app = express();
app.use(express.json());

// Store transports by session ID
const transports = {};

console.log('🚀 Starting Ultralytics MCP Server...');

//=============================================================================
// SSE TRANSPORT - N8N COMPATIBLE
//=============================================================================
app.get('/sse', async (req, res) => {
  console.log('📡 GET /sse - Establishing SSE connection');
  
  try {
    const transport = new SSEServerTransport('/messages', res);
    transports[transport.sessionId] = transport;
    
    res.on("close", () => {
      console.log(`❌ SSE connection closed for session ${transport.sessionId}`);
      delete transports[transport.sessionId];
    });

    const server = createUltralyticsServer();
    await server.connect(transport);
    
    console.log(`✅ SSE transport connected: ${transport.sessionId}`);
  } catch (error) {
    console.error('❌ SSE connection error:', error);
    if (!res.headersSent) {
      res.status(500).send('Internal server error');
    }
  }
});

app.post("/messages", async (req, res) => {
  const sessionId = req.query.sessionId;
  console.log(`📡 POST /messages: ${sessionId}`);
  
  try {
    const transport = transports[sessionId];
    
    if (transport instanceof SSEServerTransport) {
      await transport.handlePostMessage(req, res, req.body);
    } else {
      res.status(400).json({
        jsonrpc: '2.0',
        error: {
          code: -32000,
          message: 'No transport found for sessionId',
        },
        id: null,
      });
    }
  } catch (error) {
    console.error('❌ POST message error:', error);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: '2.0',
        error: {
          code: -32603,
          message: 'Internal server error',
        },
        id: null,
      });
    }
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    server: 'ultralytics_mcp',
    version: '1.0.0',
    activeSessions: Object.keys(transports).length,
    timestamp: new Date().toISOString()
  });
});

// Start server
const PORT = process.env.PORT || 8092;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`
🎯 Ultralytics MCP Server Running on Port ${PORT}
=================================================

📡 SSE Endpoint: http://localhost:${PORT}/sse
📨 Messages: http://localhost:${PORT}/messages  
❤️ Health: http://localhost:${PORT}/health

✅ N8N MCP Client Compatible
=================================================
  `);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down...');
  
  for (const sessionId in transports) {
    try {
      await transports[sessionId].close();
      delete transports[sessionId];
    } catch (error) {
      console.error(`Error closing session ${sessionId}:`, error);
    }
  }
  
  console.log('✅ Shutdown complete');
  process.exit(0);
});
