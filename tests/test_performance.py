"""
Performance tests for the Farq library.
"""
import numpy as np
import pytest
import time
import psutil
import os
import farq

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

@pytest.mark.performance
def test_small_array_performance():
    """Test performance with small arrays (100x100)."""
    size = (100, 100)
    data1 = np.random.rand(*size)
    data2 = np.random.rand(*size)
    
    # Test NDWI calculation
    start_time = time.time()
    ndwi = farq.ndwi(data1, data2)
    calc_time = time.time() - start_time
    
    assert calc_time < 0.1  # Should be very fast for small arrays
    
    # Test visualization
    start_time = time.time()
    farq.plot(ndwi)
    plot_time = time.time() - start_time
    
    assert plot_time < 0.1

@pytest.mark.performance
def test_medium_array_performance():
    """Test performance with medium arrays (1000x1000)."""
    size = (1000, 1000)
    data1 = np.random.rand(*size)
    data2 = np.random.rand(*size)
    
    # Test NDWI calculation
    start_time = time.time()
    ndwi = farq.ndwi(data1, data2)
    calc_time = time.time() - start_time
    
    assert calc_time < 0.5  # Should be reasonably fast
    
    # Test visualization
    start_time = time.time()
    farq.plot(ndwi)
    plot_time = time.time() - start_time
    
    assert plot_time < 0.5

@pytest.mark.performance
def test_large_array_performance():
    """Test performance with large arrays (10000x10000)."""
    size = (10000, 10000)
    data1 = np.random.rand(*size)
    data2 = np.random.rand(*size)
    
    # Measure memory before
    mem_before = get_memory_usage()
    
    # Test NDWI calculation
    start_time = time.time()
    ndwi = farq.ndwi(data1, data2)
    calc_time = time.time() - start_time
    
    # Measure memory after
    mem_after = get_memory_usage()
    mem_increase = mem_after - mem_before
    
    assert calc_time < 5.0  # Should complete within reasonable time
    assert mem_increase < 2000  # Memory increase should be reasonable

@pytest.mark.performance
def test_memory_efficiency():
    """Test memory efficiency with various operations."""
    size = (5000, 5000)
    data1 = np.random.rand(*size)
    data2 = np.random.rand(*size)
    
    # Test memory usage during calculations
    mem_before = get_memory_usage()
    
    # Perform multiple operations
    ndwi = farq.ndwi(data1, data2)
    ndvi = farq.ndvi(data1, data2)
    
    mem_after = get_memory_usage()
    mem_increase = mem_after - mem_before
    
    assert mem_increase < 1000  # Memory increase should be reasonable

@pytest.mark.performance
def test_resample_performance():
    """Test resampling performance."""
    original_size = (5000, 5000)
    target_size = (1000, 1000)
    data = np.random.rand(*original_size)
    
    start_time = time.time()
    resampled = farq.resample(data, target_size)
    resample_time = time.time() - start_time
    
    assert resample_time < 2.0  # Resampling should be reasonably fast

@pytest.mark.performance
def test_statistical_operations():
    """Test performance of statistical operations."""
    size = (10000, 10000)
    data = np.random.rand(*size)
    
    # Test mean calculation
    start_time = time.time()
    mean = farq.mean(data)
    mean_time = time.time() - start_time
    
    assert mean_time < 0.5
    
    # Test standard deviation calculation
    start_time = time.time()
    std = farq.std(data)
    std_time = time.time() - start_time
    
    assert std_time < 0.5

@pytest.mark.performance
def test_visualization_memory():
    """Test memory usage during visualization."""
    size = (5000, 5000)
    data = np.random.rand(*size)
    
    mem_before = get_memory_usage()
    
    # Create visualization
    farq.plot(data)
    
    mem_after = get_memory_usage()
    mem_increase = mem_after - mem_before
    
    assert mem_increase < 500  # Visualization should be memory efficient 