"""
Tests for spectral indices calculations.
"""
import numpy as np
import pytest
import farq

def test_ndwi_calculation():
    """Test NDWI calculation."""
    green = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.2, 0.3], [0.4, 0.5]])
    
    ndwi = farq.ndwi(green, nir)
    
    assert ndwi.shape == green.shape
    assert np.all(ndwi >= -1) and np.all(ndwi <= 1)

def test_ndvi_calculation():
    """Test NDVI calculation."""
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.2, 0.3], [0.4, 0.5]])
    
    ndvi = farq.ndvi(red, nir)
    
    assert ndvi.shape == red.shape
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1)

def test_evi_calculation():
    """Test EVI calculation."""
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.2, 0.3], [0.4, 0.5]])
    blue = np.array([[0.05, 0.15], [0.25, 0.35]])
    
    evi = farq.evi(red, nir, blue)
    
    assert evi.shape == red.shape
    assert not np.any(np.isnan(evi))

def test_savi_calculation():
    """Test SAVI calculation."""
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.2, 0.3], [0.4, 0.5]])
    
    savi = farq.savi(nir, red, L=0.5)
    
    assert savi.shape == red.shape
    assert not np.any(np.isnan(savi))

def test_ndwi_water_detection():
    """Test NDWI water detection capability."""
    # Create synthetic data where water should be detected
    green = np.array([[0.3, 0.1], [0.4, 0.2]])
    nir = np.array([[0.1, 0.3], [0.2, 0.4]])
    
    ndwi = farq.ndwi(green, nir)
    water_mask = ndwi > 0
    
    # First pixel should be water (NDWI > 0)
    assert water_mask[0, 0] == True

def test_ndvi_vegetation_detection():
    """Test NDVI vegetation detection capability."""
    # Create synthetic data where vegetation should be detected
    red = np.array([[0.1, 0.3], [0.2, 0.4]])
    nir = np.array([[0.3, 0.1], [0.4, 0.2]])
    
    ndvi = farq.ndvi(red, nir)
    veg_mask = ndvi > 0.2
    
    # First pixel should be vegetation (NDVI > 0.2)
    assert veg_mask[0, 0] == True

def test_indices_zero_division():
    """Test handling of zero division cases."""
    red = np.array([[0, 0], [0.3, 0.4]])
    nir = np.array([[0, 0], [0.4, 0.5]])
    
    ndvi = farq.ndvi(red, nir)
    assert not np.any(np.isnan(ndvi))
    assert not np.any(np.isinf(ndvi))

def test_indices_negative_values():
    """Test handling of negative input values."""
    red = np.array([[-0.1, 0.2], [0.3, -0.4]])
    nir = np.array([[0.2, -0.3], [0.4, 0.5]])
    
    ndvi = farq.ndvi(red, nir)
    assert not np.any(np.isnan(ndvi))
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1)

def test_indices_invalid_shapes():
    """Test handling of mismatched array shapes."""
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.2, 0.3, 0.4], [0.4, 0.5, 0.6]])
    
    with pytest.raises(ValueError):
        farq.ndvi(red, nir)

def test_indices_large_values():
    """Test handling of large input values."""
    red = np.array([[100, 200], [300, 400]])
    nir = np.array([[200, 300], [400, 500]])
    
    ndvi = farq.ndvi(red, nir)
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1)

def test_evi_parameters():
    """Test EVI calculation with different parameters."""
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.2, 0.3], [0.4, 0.5]])
    blue = np.array([[0.05, 0.15], [0.25, 0.35]])
    
    evi = farq.evi(red, nir, blue)
    assert not np.any(np.isnan(evi))

def test_savi_parameters():
    """Test SAVI calculation with different L values."""
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.2, 0.3], [0.4, 0.5]])
    
    savi_05 = farq.savi(nir, red, L=0.5)
    savi_1 = farq.savi(nir, red, L=1.0)
    
    assert not np.array_equal(savi_05, savi_1) 