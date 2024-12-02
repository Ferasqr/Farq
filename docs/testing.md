# Testing Documentation

## Overview

Farq includes a comprehensive test suite to ensure reliability and performance. The tests cover core functionality, spectral indices, and visualization components.

## Running Tests

To run the test suite:

```bash
python -m pytest tests/
```

## Test Structure

### Core Tests
- Data loading and validation
- Array operations
- Statistical functions
- Resampling operations

### Spectral Index Tests
- NDWI calculation and validation
- NDVI calculation and validation
- EVI calculation and validation
- SAVI calculation and validation

### Visualization Tests
- Plot function validation
- Compare function validation
- Colormap handling
- Figure management

## Performance Tests

### Test Data
Test data includes various sizes of Landsat imagery:
- Small (100x100 pixels)
- Medium (1000x1000 pixels)
- Large (10000x10000 pixels)

### Memory Usage
Memory usage is monitored for:
- Data loading
- Index calculations
- Statistical operations
- Visualization functions

### Processing Speed
Performance benchmarks for:
- Raster loading
- Index calculations
- Statistical operations
- Visualization rendering

## Example Test Cases

### Testing NDWI Calculation
```python
def test_ndwi_calculation():
    # Create test data
    green = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.2, 0.3], [0.4, 0.5]])
    
    # Calculate NDWI
    ndwi = farq.ndwi(green, nir)
    
    # Validate results
    assert ndwi.shape == green.shape
    assert np.all(ndwi >= -1) and np.all(ndwi <= 1)
```

### Testing Visualization
```python
def test_plot_function():
    # Create test data
    data = np.random.rand(100, 100)
    
    # Test basic plotting
    farq.plot(data, title="Test Plot")
    
    # Validate figure properties
    assert plt.gcf() is not None
    plt.close()
```

## Continuous Integration

The test suite runs automatically on:
- Pull requests
- Main branch commits
- Release tags

## Test Coverage

Current test coverage includes:
- Core functions: 95%
- Spectral indices: 100%
- Visualization: 90%
- Statistical operations: 95%

## Contributing Tests

When adding new features:
1. Add corresponding test cases
2. Ensure test coverage
3. Include performance benchmarks
4. Document test cases

## Performance Benchmarks

Latest benchmark results for common operations:

### Small Dataset (100x100)
- Load time: < 0.1s
- NDWI calculation: < 0.01s
- Visualization: < 0.1s

### Medium Dataset (1000x1000)
- Load time: < 0.5s
- NDWI calculation: < 0.1s
- Visualization: < 0.5s

### Large Dataset (10000x10000)
- Load time: < 5s
- NDWI calculation: < 1s
- Visualization: < 2s