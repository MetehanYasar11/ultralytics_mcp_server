import pytest
import httpx
import asyncio
from app.main import app


@pytest.mark.asyncio
async def test_sse_predict():
    """Test SSE endpoint for prediction with real-time streaming"""
    print("\n📡 Testing SSE prediction endpoint...")
    
    # SSE endpoint URL with prediction parameters
    sse_url = "/sse/predict"
    params = {
        "model": "yolo11n.pt",
        "source": "https://ultralytics.com/images/bus.jpg",
        "device": "cpu",
        "save": "true",
        "project": "runs/detect",
        "name": "sse_test_predict",
        "exist_ok": "true"
    }
    
    # Use httpx AsyncClient for SSE streaming
    async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
        async with client.stream("GET", sse_url, params=params) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            
            # Read first few chunks to verify SSE format
            chunks_received = 0
            async for chunk in response.aiter_text():
                if chunk.strip():
                    print(f"SSE chunk: {chunk[:100]}...")  # Print first 100 chars
                    
                    # Verify SSE format: should start with "data: "
                    assert chunk.startswith("data: "), f"Invalid SSE format: {chunk[:50]}"
                    
                    chunks_received += 1
                    # Only test first few chunks to avoid long test duration
                    if chunks_received >= 3:
                        break
            
            # Ensure we received at least some SSE data
            assert chunks_received > 0, "No SSE chunks received"
            print(f"   ✅ Received {chunks_received} SSE chunks in correct format")


@pytest.mark.asyncio
async def test_sse_train():
    """Test SSE endpoint for training with minimal parameters"""
    print("\n📡 Testing SSE training endpoint...")
    
    # SSE endpoint URL with minimal training parameters
    sse_url = "/sse/train"
    params = {
        "data": "coco128.yaml",
        "epochs": "1",
        "device": "cpu",
        "imgsz": "640",
        "batch": "2",
        "project": "runs/detect",
        "name": "sse_test_train",
        "exist_ok": "true",
        "verbose": "false",
        "cache": "false"
    }
    
    # Use httpx AsyncClient for SSE streaming
    async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
        async with client.stream("GET", sse_url, params=params) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            
            # Read first few chunks to verify SSE format
            chunks_received = 0
            async for chunk in response.aiter_text():
                if chunk.strip():
                    print(f"SSE chunk: {chunk[:100]}...")  # Print first 100 chars
                    
                    # Verify SSE format: should start with "data: "
                    assert chunk.startswith("data: "), f"Invalid SSE format: {chunk[:50]}"
                    
                    chunks_received += 1
                    # Only test first few chunks to avoid long test duration
                    if chunks_received >= 5:
                        break
            
            # Ensure we received at least some SSE data
            assert chunks_received > 0, "No SSE chunks received"
            print(f"   ✅ Received {chunks_received} SSE chunks in correct format")


if __name__ == "__main__":
    asyncio.run(test_sse_predict())
    asyncio.run(test_sse_train())
