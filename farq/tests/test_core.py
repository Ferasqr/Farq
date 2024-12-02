import numpy as np
import pytest
from farq import ndvi, ndwi, mndwi, ndbi, savi, evi

def test_ndwi():
    # Test with perfect water (high green, low NIR)
    green = np.array([[0.8, 0.7], [0.9, 0.8]])
    nir = np.array([[0.1, 0.2], [0.1, 0.15]])
    result = ndwi(green, nir)
    assert np.all(result > 0)  # Should indicate water
    assert result.shape == (2, 2)
    
    # Test with perfect non-water (low green, high NIR)
    green = np.array([[0.1, 0.2], [0.15, 0.1]])
    nir = np.array([[0.8, 0.7], [0.9, 0.8]])
    result = ndwi(green, nir)
    assert np.all(result < 0)  # Should indicate non-water

def test_ndvi():
    # Test with healthy vegetation (high NIR, low red)
    nir = np.array([[0.8, 0.7], [0.9, 0.8]])
    red = np.array([[0.1, 0.2], [0.1, 0.15]])
    result = ndvi(nir, red)
    assert np.all(result > 0)  # Should indicate vegetation
    
    # Test with non-vegetation (low NIR, high red)
    nir = np.array([[0.1, 0.2], [0.15, 0.1]])
    red = np.array([[0.8, 0.7], [0.9, 0.8]])
    result = ndvi(nir, red)
    assert np.all(result < 0)  # Should indicate non-vegetation

def test_savi():
    nir = np.array([[0.8, 0.7], [0.9, 0.8]])
    red = np.array([[0.1, 0.2], [0.1, 0.15]])
    result = savi(nir, red, L=0.5)
    assert result.shape == (2, 2)
    assert np.all(result >= -1.5) and np.all(result <= 1.5)

def test_evi():
    nir = np.array([[0.8, 0.7], [0.9, 0.8]])
    red = np.array([[0.1, 0.2], [0.1, 0.15]])
    blue = np.array([[0.05, 0.1], [0.08, 0.07]])
    result = evi(nir, red, blue)
    assert result.shape == (2, 2)
    assert np.all(result >= -1) and np.all(result <= 1)

def test_edge_cases():
    # Test with zeros
    zeros = np.zeros((2, 2))
    result = ndwi(zeros, zeros)
    assert np.allclose(result, 0)  # Use allclose for floating point comparison
    
    # Test with ones
    ones = np.ones((2, 2))
    result = ndwi(ones, ones)
    assert np.allclose(result, 0)  # Use allclose for floating point comparison
    
    # Test with NaN handling
    data = np.array([[1.0, np.nan], [1.0, 1.0]])
    result = ndwi(data, data)
    assert not np.any(np.isnan(result))  # NaNs should be handled

def test_input_validation():
    # Test different shapes
    with pytest.raises(ValueError):
        ndwi(np.zeros((2, 2)), np.zeros((3, 3)))