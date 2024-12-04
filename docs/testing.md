# Testing Documentation

## Overview

Farq includes a comprehensive test suite to ensure reliability and performance. The tests cover core functionality, spectral indices, visualization components, and performance benchmarks.

## Running Tests

To run the test suite:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_core.py

# Run tests with coverage report
python -m pytest --cov=farq tests/
```

## Test Structure

### Core Tests (`test_core.py`)
- Data loading and validation
- Array operations
- Statistical functions
- Resampling operations
- NDWI calculations
- Error handling and input validation

### Spectral Index Tests (`test_indices.py`)
- NDWI calculation and validation
- NDVI calculation and validation
- EVI calculation and validation
- SAVI calculation and validation
- Edge cases and error handling
- Input validation

### Visualization Tests (`test_visualization.py`)
- Plot function validation
- Compare function validation
- RGB visualization
- Histogram plotting
- Distribution comparison
- Colormap handling
- Figure management
- Colorbar handling

### Performance Tests (`test_performance.py`)
- Small array operations
- Medium array operations
- Large array operations
- Memory usage monitoring
- Processing speed benchmarks

## Test Data

Test data includes various sizes of Landsat imagery:
- Small (100x100 pixels)
- Medium (1000x1000 pixels)
- Large (5000x5000 pixels)

## Performance Benchmarks

Latest benchmark results for common operations:

### Small Dataset (100x100)
- Load time: < 0.1s
- NDWI calculation: < 0.5s
- Visualization: < 0.5s
- Memory usage: < 100MB

### Medium Dataset (1000x1000)
- Load time: < 0.5s
- NDWI calculation: < 2.0s
- Visualization: < 2.0s
- Memory usage: < 500MB

### Large Dataset (5000x5000)
- Load time: < 5.0s
- NDWI calculation: < 10.0s
- Visualization: < 10.0s
- Memory usage: < 2GB

## Memory Usage

Memory usage is monitored for:
- Data loading
- Index calculations
- Statistical operations
- Visualization functions

Memory limits are enforced to ensure efficient operation:
- Small operations: < 500MB
- Medium operations: < 1GB
- Large operations: < 2GB

## Test Coverage

Current test coverage includes:
- Core functions: 95%
- Spectral indices: 100%
- Visualization: 90%
- Statistical operations: 95%
- Error handling: 100%
- Input validation: 100%

## Continuous Integration

The test suite runs automatically on:
- Pull requests
- Main branch commits
- Release tags

Tests are run on multiple platforms:
- Linux
- Windows
- macOS

And multiple Python versions:
- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10

## Contributing Tests

When adding new features:
1. Add corresponding test cases
2. Ensure test coverage
3. Include performance benchmarks
4. Document test cases
5. Verify error handling
6. Add input validation tests

## Test Categories

### Unit Tests
- Individual function testing
- Input validation
- Error handling
- Edge cases

### Integration Tests
- Multi-function workflows
- File I/O operations
- Cross-module functionality

### Performance Tests
- Processing speed
- Memory usage
- Resource efficiency
- Scalability

### Visualization Tests
- Plot accuracy
- Figure properties
- Colormap handling
- Interactive features

## Error Handling Tests

All functions are tested for proper error handling:
- Invalid inputs
- Missing data
- Type mismatches
- Shape mismatches
- Out of memory conditions
- File I/O errors

## Input Validation Tests

Comprehensive validation testing for:
- Data types
- Array shapes
- Value ranges
- NaN handling
- Missing data
- Parameter validation

## Benchmark Tests

Performance benchmarks include:
- Execution time
- Memory usage
- CPU utilization
- I/O operations
- Visualization rendering

## Test Configuration

Test settings are configured in `pytest.ini`:
- Test discovery patterns
- Test markers
- Performance thresholds
- Coverage settings