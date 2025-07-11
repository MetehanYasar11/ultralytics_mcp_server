import pytest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several minutes)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory"""
    return Path(__file__).parent / "data"

@pytest.fixture
def clean_runs_dir():
    """Fixture to clean runs directory before and after tests"""
    runs_dir = Path("runs")
    
    # Clean before test
    if runs_dir.exists():
        import shutil
        shutil.rmtree(runs_dir)
    
    yield runs_dir
    
    # Optionally clean after test (commented out to preserve results for inspection)
    # if runs_dir.exists():
    #     shutil.rmtree(runs_dir)

# Pytest command line options
def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
