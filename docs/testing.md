# Farq Testing Documentation

## Overview
This document describes the testing strategy and test suite for the Farq library.

## Test Categories

### 1. Unit Tests
Located in `farq/tests/test_core.py`, these tests verify the core functionality:
- Spectral indices (NDVI, NDWI, MNDWI, etc.)
- Input validation
- Edge cases (zeros, NaN values)
- Expected output ranges

### 2. Analysis Tests
Located in `farq/tests/test_analysis.py`, these tests cover:
- Water statistics calculations
- Change detection
- Water body identification
- Area calculations
- Input validation

### 3. Visualization Tests
Located in `farq/tests/test_visualization.py`, these tests verify:
- Single raster plotting
- Raster comparison plotting
- Change visualization
- Histogram generation
- Figure and axes management

### 4. Performance Tests
Located in `farq/tests/test_performance.py`, these tests ensure:
- Execution time limits for different input sizes
- Memory usage monitoring
- Scalability with large datasets
- Resource efficiency

## Running Tests

### Running All Tests
```bash
pytest
```

### Running Specific Test Categories
```bash
# Run only unit tests
pytest farq/tests/test_core.py

# Run only performance tests
pytest -m performance

# Run with verbose output
pytest -v

# Run with test coverage
pytest --cov=farq
```

## Performance Benchmarks
The library should meet these performance criteria:
- NDWI calculation: < 1s for 1000x1000 arrays
- Water statistics: < 1s for 2000x2000 masks
- Water body identification: < 2s for 1000x1000 masks
- Memory usage: < 1GB for 5000x5000 arrays

## Test Data
- Test data includes synthetic arrays of various sizes
- Edge cases are tested with special arrays (zeros, ones, NaN)
- Random data generation for performance testing

## Adding New Tests
When adding new tests:
1. Choose appropriate test category
2. Follow existing naming conventions
3. Include docstrings explaining test purpose
4. Add performance tests for computationally intensive functions
5. Update this documentation

## Continuous Integration
Tests are run automatically on:
- Every pull request
- Main branch commits
- Release tags

## Test Dependencies
Required packages for testing:
- pytest
- numpy
- matplotlib
- psutil (for memory tests)

Install test dependencies:
```bash
pip install -r requirements-test.txt
``` 