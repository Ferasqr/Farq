import numpy as np
import pytest
import time
from farq import ndwi, water_stats, water_change, get_water_bodies

def measure_time(func, *args, **kwargs):
    """Helper function to measure execution time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

@pytest.mark.performance
def test_ndwi_performance():
    # Test with different sizes
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    times = []
    
    for size in sizes:
        green = np.random.random(size)
        nir = np.random.random(size)
        
        _, execution_time = measure_time(ndwi, green, nir)
        times.append(execution_time)
        
        # Performance assertions
        assert execution_time < 1.0, f"NDWI calculation for size {size} took too long: {execution_time:.2f}s"

@pytest.mark.performance
def test_water_stats_performance():
    # Test with different sizes
    sizes = [(1000, 1000), (2000, 2000)]
    times = []
    
    for size in sizes:
        mask = np.random.choice([True, False], size=size)
        _, execution_time = measure_time(water_stats, mask, pixel_size=30.0)
        times.append(execution_time)
        
        # Performance assertions
        assert execution_time < 1.0, f"Water stats calculation for size {size} took too long: {execution_time:.2f}s"

@pytest.mark.performance
def test_water_change_performance():
    size = (1000, 1000)
    mask1 = np.random.choice([True, False], size=size)
    mask2 = np.random.choice([True, False], size=size)
    
    _, execution_time = measure_time(water_change, mask1, mask2)
    
    # Performance assertions
    assert execution_time < 1.0, "Water change calculation took too long"

@pytest.mark.performance
def test_get_water_bodies_performance():
    # Test with different numbers of water bodies
    size = (1000, 1000)
    
    # Create mask with multiple separate water bodies
    mask = np.zeros(size, dtype=bool)
    for i in range(0, size[0], 50):  # Create water bodies of size 20x20 every 50 pixels
        for j in range(0, size[1], 50):
            if np.random.random() > 0.5:  # 50% chance of water body
                mask[i:i+20, j:j+20] = True
    
    _, execution_time = measure_time(get_water_bodies, mask, pixel_size=30.0)
    
    # Performance assertions
    assert execution_time < 2.0, "Water bodies identification took too long"

@pytest.mark.performance
def test_memory_usage():
    """Test memory usage with large arrays"""
    import psutil
    import os
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    # Test with a large array
    size = (5000, 5000)
    initial_memory = get_memory_usage()
    
    # Create and process large arrays
    green = np.random.random(size)
    nir = np.random.random(size)
    result = ndwi(green, nir)
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    # Memory usage should not exceed reasonable limits
    assert memory_increase < 1000, f"Memory usage increased by {memory_increase:.1f}MB" 