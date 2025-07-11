#!/usr/bin/env python3
"""
Test runner script for the Ultralytics MCP Server
"""
import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run the test suite"""
    project_root = Path(__file__).parent
    
    # Change to project directory
    import os
    os.chdir(project_root)
    
    print("🧪 Running Ultralytics MCP Server Tests")
    print("=" * 50)
    
    # Run pytest with verbose output using ultra-dev environment
    cmd = [
        "conda", "run", "-n", "ultra-dev", "pytest",
        "tests/test_flow.py",
        "-v", "-s", 
        "--tb=short",
        "--color=yes"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 1

def run_quick_tests():
    """Run only quick tests (skip slow integration tests)"""
    project_root = Path(__file__).parent
    
    import os
    os.chdir(project_root)
    
    print("🏃 Running Quick Tests (excluding slow tests)")
    print("=" * 50)
    
    cmd = [
        "conda", "run", "-n", "ultra-dev", "pytest",
        "tests/test_flow.py::TestUltralyticsFlow::test_health_check",
        "tests/test_flow.py::TestIndividualEndpoints",
        "-v", "-s", 
        "--tb=short",
        "--color=yes"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Quick tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Quick tests failed with return code: {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        exit_code = run_quick_tests()
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)
