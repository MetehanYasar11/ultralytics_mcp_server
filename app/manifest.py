"""
MCP (Model Context Protocol) manifest and configuration.
"""

def get_manifest() -> dict:
    """
    Return MCP manifest with server capabilities and tool definitions.
    
    Returns:
        Dictionary containing MCP schema-compliant manifest
    """
    return {
        "schema_version": "1.0",
        "name": "ultralytics_mcp",
        "description": "YOLO train/val/predict/export/track/benchmark via SSE",
        "version": "0.1.0",
        "interfaces": {
            "http": { "openapi_url": "/openapi.json" },
            "sse":  { "base_url": "/sse" }
        },
        "tools": [
            { "name": "train",     "path": "/train" },
            { "name": "val",       "path": "/val" },
            { "name": "predict",   "path": "/predict" },
            { "name": "export",    "path": "/export" },
            { "name": "track",     "path": "/track" },
            { "name": "benchmark", "path": "/benchmark" }
        ]
    }
