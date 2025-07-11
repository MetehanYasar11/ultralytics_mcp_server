/**
 * Example usage of the Ultralytics MCP Tool
 * 
 * This file demonstrates how to use the tool for various YOLO operations
 */

import UltralyticsMCPTool from './index';

async function exampleUsage() {
  // Initialize the tool
  const tool = new UltralyticsMCPTool('http://localhost:8000');
  
  console.log('🚀 Ultralytics MCP Tool Example');
  console.log('================================');
  
  try {
    // 1. Get MCP Manifest
    console.log('\n📋 MCP Manifest:');
    const manifest = UltralyticsMCPTool.manifest();
    console.log(`Name: ${manifest.name}`);
    console.log(`Version: ${manifest.version}`);
    console.log(`Available tools: ${manifest.tools.map(t => t.name).join(', ')}`);
    
    // 2. Test connections
    console.log('\n🔗 Testing connections...');
    const connections = await tool.testConnection();
    console.log(`HTTP: ${connections.http ? '✅' : '❌'}`);
    console.log(`SSE: ${connections.sse ? '✅' : '❌'}`);
    console.log(`Stdio: ${connections.stdio ? '✅' : '❌'}`);
    
    // 3. Health check
    console.log('\n🏥 Health check...');
    const health = await tool.health();
    console.log(`Status: ${health.status}`);
    
    // 4. Quick prediction example
    console.log('\n🎯 Running prediction...');
    const predictResult = await tool.predict({
      model: 'yolov8n.pt',
      source: 'https://ultralytics.com/images/bus.jpg',
      conf: 0.5,
      save: true
    });
    
    console.log(`Prediction success: ${predictResult.success}`);
    console.log(`Artifacts created: ${predictResult.artifacts.length}`);
    
    // 5. Training example with SSE
    console.log('\n🏋️ Training with real-time updates...');
    await new Promise((resolve, reject) => {
      const eventSource = tool.trainSSE({
        model: 'yolov8n.pt',
        data: 'coco128.yaml',
        epochs: 2,  // Just 2 epochs for demo
        imgsz: 640,
        batch: 4,
        device: 'cpu'
      }, {
        onProgress: (data) => {
          console.log(`📊 Progress: ${JSON.stringify(data)}`);
        },
        onMetrics: (data) => {
          console.log(`📈 Metrics: ${JSON.stringify(data)}`);
        },
        onComplete: (result) => {
          console.log(`✅ Training completed! Success: ${result.success}`);
          console.log(`📁 Artifacts: ${result.artifacts.length} files created`);
          resolve(result);
        },
        onError: (error) => {
          console.error(`❌ Training error: ${error}`);
          reject(error);
        }
      });
      
      // Set a timeout for the demo
      setTimeout(() => {
        eventSource.close();
        console.log('⏰ Demo timeout - closing SSE connection');
        resolve(null);
      }, 60000); // 1 minute timeout
    });
    
    // 6. Export example
    console.log('\n📤 Exporting model...');
    const exportResult = await tool.export({
      model: 'yolov8n.pt',
      format: 'onnx',
      imgsz: 640
    });
    
    console.log(`Export success: ${exportResult.success}`);
    
    // 7. Stdio example
    console.log('\n💻 Stdio channel example...');
    const stdioResult = await tool.executeStdio('predict', {
      model: 'yolov8n.pt',
      source: 'https://ultralytics.com/images/bus.jpg'
    });
    
    console.log(`Stdio prediction success: ${stdioResult.success}`);
    
  } catch (error) {
    console.error('❌ Example failed:', error);
  } finally {
    // Cleanup
    tool.cleanup();
    console.log('\n🧹 Cleanup completed');
  }
}

// Advanced example: Complete workflow
async function completeWorkflowExample() {
  const tool = new UltralyticsMCPTool();
  
  console.log('\n🔄 Complete Workflow Example');
  console.log('=============================');
  
  try {
    // 1. Train a model
    const trainResult = await tool.train({
      model: 'yolov8n.pt',
      data: 'coco128.yaml',
      epochs: 1,
      project: 'runs/train',
      name: 'demo',
      exist_ok: true
    });
    
    if (!trainResult.success) {
      throw new Error('Training failed');
    }
    
    console.log('✅ Training completed');
    
    // 2. Validate the trained model
    const valResult = await tool.val({
      model: 'runs/train/demo/weights/best.pt',
      data: 'coco128.yaml'
    });
    
    console.log(`✅ Validation completed: ${valResult.success}`);
    
    // 3. Run predictions
    const predictResult = await tool.predict({
      model: 'runs/train/demo/weights/best.pt',
      source: 'https://ultralytics.com/images/bus.jpg'
    });
    
    console.log(`✅ Prediction completed: ${predictResult.success}`);
    
    // 4. Export the model
    const exportResult = await tool.export({
      model: 'runs/train/demo/weights/best.pt',
      format: 'onnx'
    });
    
    console.log(`✅ Export completed: ${exportResult.success}`);
    
    return {
      train: trainResult,
      val: valResult,
      predict: predictResult,
      export: exportResult
    };
    
  } catch (error) {
    console.error('❌ Workflow failed:', error);
    throw error;
  }
}

// Channel comparison example
async function channelComparisonExample() {
  const tool = new UltralyticsMCPTool();
  
  console.log('\n🔀 Channel Comparison Example');
  console.log('==============================');
  
  const params = {
    model: 'yolov8n.pt',
    source: 'https://ultralytics.com/images/bus.jpg'
  };
  
  try {
    // HTTP channel
    console.time('HTTP');
    const httpResult = await tool.execute('predict', params, 'http');
    console.timeEnd('HTTP');
    console.log(`HTTP success: ${httpResult.success}`);
    
    // Stdio channel
    console.time('Stdio');
    const stdioResult = await tool.execute('predict', params, 'stdio');
    console.timeEnd('Stdio');
    console.log(`Stdio success: ${stdioResult.success}`);
    
  } catch (error) {
    console.error('❌ Channel comparison failed:', error);
  }
}

// Run examples if this file is executed directly
if (require.main === module) {
  exampleUsage()
    .then(() => completeWorkflowExample())
    .then(() => channelComparisonExample())
    .then(() => {
      console.log('\n🎉 All examples completed successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n💥 Examples failed:', error);
      process.exit(1);
    });
}

export {
  exampleUsage,
  completeWorkflowExample,
  channelComparisonExample
};
