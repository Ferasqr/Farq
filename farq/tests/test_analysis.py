import numpy as np
import pytest
from farq import water_stats, water_change, get_water_bodies

def test_water_stats():
    # Create a simple water mask
    mask = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0]
    ], dtype=bool)
    
    stats = water_stats(mask, pixel_size=30.0)
    
    # Test area calculation (4 pixels * (30m * 30m) = 3600m² = 0.0036km²)
    assert np.isclose(stats['total_area'], 0.0036, rtol=1e-5)
    assert np.isclose(stats['coverage_percent'], (4/12)*100, rtol=1e-5)

def test_water_change():
    # Create two water masks
    mask1 = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0]
    ], dtype=bool)
    
    mask2 = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0]
    ], dtype=bool)
    
    changes = water_change(mask1, mask2, pixel_size=30.0)
    
    # Both masks have same number of water pixels but in different locations
    assert np.isclose(changes['gained_area'], 0.0036, rtol=1e-5)
    assert np.isclose(changes['lost_area'], 0.0036, rtol=1e-5)
    assert np.isclose(changes['net_change'], 0.0, rtol=1e-5)
    assert np.isclose(changes['change_percent'], 0.0, rtol=1e-5)

def test_get_water_bodies():
    # Create a mask with two separate water bodies
    mask = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ], dtype=bool)
    
    labeled, areas = get_water_bodies(mask, pixel_size=30.0)
    
    # Should identify 2 water bodies
    assert len(areas) == 2
    # Each body should have 4 pixels
    assert np.all(np.array(list(areas.values())) == 0.0036)
    # Labels should be 1 and 2
    assert set(np.unique(labeled)) == {0, 1, 2}

def test_edge_cases():
    # Empty mask
    empty = np.zeros((3, 3), dtype=bool)
    stats = water_stats(empty)
    assert stats['total_area'] == 0
    assert stats['coverage_percent'] == 0
    
    # Full mask
    full = np.ones((3, 3), dtype=bool)
    stats = water_stats(full)
    assert stats['coverage_percent'] == 100
    
    # Single pixel
    single = np.array([[1]], dtype=bool)
    stats = water_stats(single, pixel_size=30.0)
    assert np.isclose(stats['total_area'], 0.0009, rtol=1e-5)  # 30m * 30m = 900m² = 0.0009km²

def test_input_validation():
    # Test invalid pixel size
    with pytest.raises(ValueError):
        water_stats(np.zeros((2, 2)), pixel_size=0)
    
    # Test different shaped masks for change detection
    with pytest.raises(ValueError):
        water_change(np.zeros((2, 2)), np.zeros((3, 3)))