# Ultralytics MCP Server Tests

This directory contains comprehensive tests for the Ultralytics MCP Server API.

## Test Structure

### `test_flow.py`
Complete integration test that runs a full workflow:

1. **Health Check** - Verifies API is running
2. **Training** - Trains YOLOv8n on COCO128 for 1 epoch
3. **Validation** - Validates the trained model
4. **Prediction** - Makes predictions using the trained model
5. **Results Verification** - Checks for expected output files

## Running Tests

### Prerequisites
Make sure you have the conda environment activated:
```bash
conda activate ultra-dev
```

### Full Test Suite
Run the complete workflow test:
```bash
# From project root
python run_tests.py

# Or using pytest directly
python -m pytest tests/test_flow.py -v -s
```

### Quick Tests Only
Run only validation and endpoint tests (skips training):
```bash
python run_tests.py quick
```

### Individual Test Classes
```bash
# Test only the workflow
python -m pytest tests/test_flow.py::TestUltralyticsFlow -v -s

# Test only individual endpoints
python -m pytest tests/test_flow.py::TestIndividualEndpoints -v -s

# Test specific method
python -m pytest tests/test_flow.py::TestUltralyticsFlow::test_health_check -v -s
```

## Test Configuration

### Expected Behavior
- **Training**: Creates `runs/detect/train/` with model weights and results
- **Validation**: Creates `runs/detect/val/` with validation metrics
- **Prediction**: Creates `runs/detect/predict/` with prediction images
- **Artifacts**: All operations should generate artifacts in their respective directories

### Test Data
- **Model**: YOLOv8n (yolov8n.pt) - downloaded automatically
- **Dataset**: COCO128 (coco128.yaml) - downloaded automatically  
- **Test Image**: Bus image from Ultralytics examples

### Performance
- **Training**: ~1-3 minutes (1 epoch, CPU)
- **Validation**: ~30 seconds
- **Prediction**: ~10 seconds
- **Total**: ~2-4 minutes for complete workflow

## Test Output

### Success Indicators
✅ All endpoints return HTTP 200  
✅ All operations have `success: true` in response  
✅ Expected directories and files are created  
✅ Model weights are generated  
✅ Results CSV contains training metrics  

### Common Files Created
```
runs/
├── detect/
│   ├── train/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   ├── results.csv
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── predict/
│       └── *.jpg (prediction images)
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError**: Make sure you're in the correct conda environment
```bash
conda activate ultra-dev
```

**Download Failures**: Ensure internet connection for model/dataset downloads

**Disk Space**: Training generates ~100-500MB of files

**Timeout**: Tests have 1-hour timeout for individual operations

### Debugging
Add `-s` flag to see print statements:
```bash
python -m pytest tests/test_flow.py -v -s
```

View detailed output:
```bash
python -m pytest tests/test_flow.py -v -s --tb=long
```

## CI/CD Integration

For automated testing, use:
```bash
# Run without interactive output
python -m pytest tests/test_flow.py --tb=short --disable-warnings

# With coverage (if coverage.py is installed)
python -m pytest tests/test_flow.py --cov=app --cov-report=html
```

## Notes

- Tests use CPU-only training for compatibility
- Small batch sizes (2) and single epoch for speed
- Results are preserved after tests for inspection
- Each test is designed to be independent but workflow tests run in sequence
