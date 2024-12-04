"""
Tests for core functionality of the Farq library.
"""
import numpy as np
import pytest
import farq
from farq.core import ndwi

def test_min():
    """Test minimum value calculation."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    assert farq.min(data) == 1

def test_max():
    """Test maximum value calculation."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    assert farq.max(data) == 6

def test_mean():
    """Test mean value calculation."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    assert farq.mean(data) == 3.5

def test_std():
    """Test standard deviation calculation."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    assert abs(farq.std(data) - np.std(data)) < 1e-10

def test_sum():
    """Test sum calculation."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    assert farq.sum(data) == 21

def test_resample():
    """Test raster resampling."""
    data = np.array([[1, 2], [3, 4]])
    target_shape = (3, 3)
    resampled = farq.resample(data, target_shape)
    assert resampled.shape == target_shape

def test_read_invalid_file():
    """Test reading an invalid file."""
    with pytest.raises(FileNotFoundError):
        farq.read("nonexistent.tif")

def test_invalid_array_shape():
    """Test handling of invalid array shapes."""
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[1, 2, 3], [4, 5, 6]])
    
    with pytest.raises(ValueError):
        farq.ndwi(data1, data2)

def test_nan_handling():
    """Test handling of NaN values."""
    data = np.array([[1, np.nan, 3], [4, 5, 6]])
    assert not np.isnan(farq.mean(data))
    assert not np.isnan(farq.std(data))

def test_empty_array():
    """Test handling of empty arrays."""
    data = np.array([])
    with pytest.raises(ValueError):
        farq.mean(data)

def test_negative_values():
    """Test handling of negative values."""
    data = np.array([[-1, -2], [-3, -4]])
    assert farq.min(data) == -4
    assert farq.max(data) == -1

def test_zero_array():
    """Test handling of zero arrays."""
    data = np.zeros((3, 3))
    assert farq.mean(data) == 0
    assert farq.std(data) == 0

def test_single_value():
    """Test handling of single value arrays."""
    data = np.array([[5]])
    assert farq.mean(data) == 5
    assert farq.std(data) == 0

def test_large_values():
    """Test handling of large values."""
    data = np.array([[1e6, 2e6], [3e6, 4e6]])
    assert farq.mean(data) == 2.5e6

def test_ndwi_range():
    """Test that NDWI returns values in [-1, 1] range."""
    # Create test data
    green = np.array([100, 1000, 500, 2000], dtype=float)
    nir = np.array([1000, 100, 500, 2000], dtype=float)
    
    result = ndwi(green, nir)
    
    # Check range
    assert np.all(result >= -1.0), "NDWI values below -1.0 found"
    assert np.all(result <= 1.0), "NDWI values above 1.0 found"
    
    # Check specific cases
    assert np.isclose(result[0], -0.81818182), "Unexpected value for nir >> green"
    assert np.isclose(result[1], 0.81818182), "Unexpected value for green >> nir"
    assert np.isclose(result[2], 0.0), "Unexpected value for green == nir"
    assert np.isclose(result[3], 0.0), "Unexpected value for equal values" 