from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import asyncio
import json
from datetime import datetime
from .ultra import run_ultralytics, parse_yolo_args, run_ultra_stream
from .manifest import get_manifest
from .schemas import (
    TrainRequest, ValRequest, PredictRequest, ExportRequest,
    TrackRequest, BenchmarkRequest, SolutionRequest, 
    OperationResponse, HealthResponse
)

app = FastAPI(title="Ultralytics API", description="API for Ultralytics YOLO operations")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def execute_ultralytics_command(command: str, args: List[str]) -> OperationResponse:
    """Execute ultralytics command and return structured response"""
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Build full args list with command
    full_args = [command] + args
    
    # Execute using the ultra module
    result = run_ultralytics(full_args)
    
    return OperationResponse(
        run_id=run_id,
        command=result["command"],
        return_code=result["return_code"],
        stdout=result["stdout"],
        stderr=result["stderr"],
        metrics=result["metrics"],
        artifacts=result["artifacts"],
        success=result["success"],
        timestamp=timestamp
    )


def process_request_args(request_data: dict) -> List[str]:
    """Process request data and merge extra_args with main args"""
    # Extract extra_args if present
    extra_args = request_data.pop('extra_args', {})
    
    # Remove 'task' from request_data since it's handled as the command mode
    request_data.pop('task', None)
    
    # Convert main request to args
    args = parse_yolo_args(request_data)
    
    # Add extra arguments in YOLO format (key=value)
    if extra_args:
        for key, value in extra_args.items():
            if value is not None:
                if isinstance(value, bool):
                    args.append(f"{key}={str(value).lower()}")
                else:
                    args.append(f"{key}={value}")
    
    return args

def build_cli_from_query(op: str, query_params) -> List[str]:
    """Build CLI arguments from query parameters for SSE endpoints"""
    # Convert query params to dict
    params = dict(query_params)
    
    # Build CLI arguments using existing parser
    args = parse_yolo_args(params)
    
    # Add the operation as the first argument
    return [op] + args

@app.get("/mcp/manifest.json", tags=["MCP"])
async def manifest():
    """MCP manifest endpoint providing server capabilities and tool definitions"""
    return get_manifest()

@app.get("/sse", tags=["MCP"], summary="MCP keep-alive stream")
async def sse_root():
    """MCP handshake endpoint providing tools and keep-alive functionality"""
    async def stream():
        # First event → tells MCP Client which tools exist
        manifest = {
            'tools': ['train','val','predict','export','track','benchmark'],
            'info' : 'Ultralytics MCP ready'
        }
        yield f'data: {json.dumps(manifest)}\n\n'
        # Every 15 s send a comment to keep the HTTP/1.1 connection open
        while True:
            yield b': ping\n\n'
            await asyncio.sleep(15)
    return StreamingResponse(stream(), media_type='text/event-stream')

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Ultralytics API is running",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/train", response_model=OperationResponse)
async def train(request: TrainRequest):
    """Train a YOLO model"""
    args = process_request_args(request.dict())
    return execute_ultralytics_command("train", args)

@app.post("/val", response_model=OperationResponse)
async def validate(request: ValRequest):
    """Validate a YOLO model"""
    args = process_request_args(request.dict())
    return execute_ultralytics_command("val", args)

@app.post("/predict", response_model=OperationResponse)
async def predict(request: PredictRequest):
    """Run prediction with a YOLO model"""
    args = process_request_args(request.dict())
    return execute_ultralytics_command("predict", args)

@app.post("/export", response_model=OperationResponse)
async def export(request: ExportRequest):
    """Export a YOLO model to different formats"""
    args = process_request_args(request.dict())
    return execute_ultralytics_command("export", args)

@app.post("/track", response_model=OperationResponse)
async def track(request: TrackRequest):
    """Run object tracking with a YOLO model"""
    args = process_request_args(request.dict())
    return execute_ultralytics_command("track", args)

@app.post("/benchmark", response_model=OperationResponse)
async def benchmark(request: BenchmarkRequest):
    """Benchmark a YOLO model"""
    args = process_request_args(request.dict())
    return execute_ultralytics_command("benchmark", args)

@app.post("/solution", response_model=OperationResponse)
async def solution(request: SolutionRequest):
    """Run Ultralytics solutions"""
    args = process_request_args(request.dict())
    return execute_ultralytics_command("solution", args)

@app.get("/sse/{op}", tags=["SSE"])
async def sse_op(op: str, request: Request):
    """
    Server-Sent Events endpoint for real-time YOLO operation streaming.
    
    Accepts query parameters identical to REST operations and streams
    live stdout from the YOLO command execution.
    
    Supported operations: train, val, predict, export, track, benchmark, solution
    """
    # Build CLI arguments from query parameters
    cli = build_cli_from_query(op, request.query_params)
    
    # Return streaming response
    return StreamingResponse(
        run_ultra_stream(cli),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
