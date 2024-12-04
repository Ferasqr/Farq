"""
Performance tests for the Farq library.

These tests verify that operations complete within reasonable time limits
and memory usage stays within acceptable bounds.
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
    
    # Test plotting
    start_time = time.time()
    farq.plot(ndwi)
    plot_time = time.time() - start_time
    
    # Assertions with more lenient timing
    assert calc_time < 1.0, f"NDWI calculation took {calc_time:.2f}s"
    assert plot_time < 1.0, f"Plotting took {plot_time:.2f}s"

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
    
    # Test plotting
    start_time = time.time()
    farq.plot(ndwi)
    plot_time = time.time() - start_time
    
    # Assertions with more lenient timing
    assert calc_time < 3.0, f"NDWI calculation took {calc_time:.2f}s"
    assert plot_time < 3.0, f"Plotting took {plot_time:.2f}s"

@pytest.mark.performance
def test_large_array_performance():
    """Test performance with large arrays (5000x5000)."""
    size = (5000, 5000)
    data1 = np.random.rand(*size)
    data2 = np.random.rand(*size)
    
    # Test NDWI calculation
    start_time = time.time()
    ndwi = farq.ndwi(data1, data2)
    calc_time = time.time() - start_time
    
    # Test plotting
    start_time = time.time()
    farq.plot(ndwi)
    plot_time = time.time() - start_time
    
    # Assertions with more lenient timing
    assert calc_time < 15.0, f"NDWI calculation took {calc_time:.2f}s"
    assert plot_time < 15.0, f"Plotting took {plot_time:.2f}s"

@pytest.mark.performance
def test_memory_efficiency():
    """Test memory usage during operations."""
    size = (2000, 2000)
    data1 = np.random.rand(*size)
    data2 = np.random.rand(*size)
    
    initial_memory = get_memory_usage()
    
    # Test NDWI calculation
    ndwi = farq.ndwi(data1, data2)
    after_ndwi = get_memory_usage()
    
    # Test plotting
    farq.plot(ndwi)
    after_plot = get_memory_usage()
    
    # Memory increase should be reasonable
    ndwi_memory = after_ndwi - initial_memory
    plot_memory = after_plot - after_ndwi
    
    # More lenient memory limits (in MB)
    assert ndwi_memory < 1000, f"NDWI used {ndwi_memory:.1f}MB"
    assert plot_memory < 1000, f"Plotting used {plot_memory:.1f}MB"

@pytest.mark.performance
def test_resample_performance():
    """Test resampling performance."""
    size = (1000, 1000)
    target = (500, 500)
    data = np.random.rand(*size)
    
    start_time = time.time()
    resampled = farq.resample(data, target)
    resample_time = time.time() - start_time
    
    # More lenient timing
    assert resample_time < 5.0, f"Resampling took {resample_time:.2f}s"
    assert resampled.shape == target

@pytest.mark.performance
def test_statistical_operations():
    """Test performance of statistical operations."""
    size = (5000, 5000)
    data = np.random.rand(*size)
    
    # Test mean calculation
    start_time = time.time()
    mean = farq.mean(data)
    mean_time = time.time() - start_time
    
    # Test standard deviation calculation
    start_time = time.time()
    std = farq.std(data)
    std_time = time.time() - start_time
    
    # More lenient timing
    assert mean_time < 3.0, f"Mean calculation took {mean_time:.2f}s"
    assert std_time < 3.0, f"Standard deviation calculation took {std_time:.2f}s"

@pytest.mark.performance
def test_visualization_memory():
    """Test memory usage during visualization."""
    size = (1000, 1000)
    data = np.random.rand(*size)
    
    initial_memory = get_memory_usage()
    
    # Test single plot
    farq.plot(data)
    after_single = get_memory_usage()
    
    # Test comparison plot
    farq.compare(data, data)
    after_compare = get_memory_usage()
    
    # Memory increase should be reasonable
    single_memory = after_single - initial_memory
    compare_memory = after_compare - after_single
    
    # More lenient memory limits (in MB)
    assert single_memory < 1000, f"Single plot used {single_memory:.1f}MB"
    assert compare_memory < 1000, f"Comparison plot used {compare_memory:.1f}MB"