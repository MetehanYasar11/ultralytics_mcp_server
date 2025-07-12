import pytest
import json
from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)


def test_mcp_manifest():
    """Test MCP manifest endpoint returns valid manifest"""
    print("\n🔧 Testing MCP manifest endpoint...")
    
    response = client.get("/mcp/manifest.json")
    
    # Assert successful response
    assert response.status_code == 200, f"Manifest request failed with status {response.status_code}"
    
    # Parse JSON response
    manifest = response.json()
    
    # Validate manifest structure
    assert "schema_version" in manifest, "Missing schema_version in manifest"
    assert "name" in manifest, "Missing name in manifest"
    assert "description" in manifest, "Missing description in manifest"
    assert "version" in manifest, "Missing version in manifest"
    assert "interfaces" in manifest, "Missing interfaces in manifest"
    assert "tools" in manifest, "Missing tools in manifest"
    
    # Validate specific values
    assert manifest["schema_version"] == "1.0", f"Expected schema_version 1.0, got {manifest['schema_version']}"
    assert manifest["name"] == "ultralytics_mcp", f"Expected name ultralytics_mcp, got {manifest['name']}"
    assert manifest["version"] == "0.1.0", f"Expected version 0.1.0, got {manifest['version']}"
    
    # Validate interfaces
    interfaces = manifest["interfaces"]
    assert "http" in interfaces, "Missing http interface"
    assert "sse" in interfaces, "Missing sse interface"
    assert interfaces["http"]["openapi_url"] == "/openapi.json", "Invalid OpenAPI URL"
    assert interfaces["sse"]["base_url"] == "/sse", "Invalid SSE base URL"
    
    # Validate tools
    tools = manifest["tools"]
    assert len(tools) == 6, f"Expected 6 tools, got {len(tools)}"
    
    expected_tools = ["train", "val", "predict", "export", "track", "benchmark"]
    tool_names = [tool["name"] for tool in tools]
    
    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
    
    # Validate tool structure
    for tool in tools:
        assert "name" in tool, f"Tool missing name: {tool}"
        assert "method" in tool, f"Tool missing method: {tool}"
        assert "path" in tool, f"Tool missing path: {tool}"
        assert tool["method"] == "POST", f"Expected POST method for tool {tool['name']}, got {tool['method']}"
        assert tool["path"].startswith("/"), f"Tool path should start with /: {tool['path']}"
    
    print(f"   ✅ Manifest validation passed")
    print(f"   📋 Name: {manifest['name']}")
    print(f"   📋 Version: {manifest['version']}")
    print(f"   📋 Tools: {', '.join(tool_names)}")


def test_sse_base_endpoint():
    """Test base SSE endpoint for keep-alive functionality"""
    print("\n📡 Testing base SSE endpoint...")
    
    response = client.get("/sse")
    
    # Assert successful response
    assert response.status_code == 200, f"SSE base request failed with status {response.status_code}"
    
    # Check content type
    assert "text/event-stream" in response.headers.get("content-type", ""), "Invalid content type for SSE"
    
    print(f"   ✅ SSE base endpoint accessible")
    print(f"   📡 Content-Type: {response.headers.get('content-type')}")


def test_manifest_content_type():
    """Test manifest endpoint returns JSON content type"""
    print("\n🔧 Testing manifest content type...")
    
    response = client.get("/mcp/manifest.json")
    
    # Check content type
    content_type = response.headers.get("content-type", "")
    assert "application/json" in content_type, f"Expected JSON content type, got {content_type}"
    
    print(f"   ✅ Content-Type: {content_type}")


if __name__ == "__main__":
    test_mcp_manifest()
    test_sse_base_endpoint()
    test_manifest_content_type()
