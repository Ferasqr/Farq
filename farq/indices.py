"""
Spectral indices module for the Farq library.

This module provides functions for calculating various spectral indices:
- NDWI (Normalized Difference Water Index)
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil Adjusted Vegetation Index)
"""

import numpy as np
from typing import Optional

def validate_bands(*bands: np.ndarray) -> None:
    """
    Validate band arrays for spectral index calculations.
    
    Args:
        *bands: Variable number of band arrays to validate
        
    Raises:
        TypeError: If any band is not a numpy array
        ValueError: If bands have different shapes or are empty
    """
    if not bands:
        raise ValueError("No bands provided")
    
    # Check each band
    for i, band in enumerate(bands):
        if not isinstance(band, np.ndarray):
            raise TypeError(f"Band {i} must be a numpy array")
        if band.size == 0:
            raise ValueError(f"Band {i} cannot be empty")
    
    # Check shapes match
    shape = bands[0].shape
    for i, band in enumerate(bands[1:], 1):
        if band.shape != shape:
            raise ValueError(f"Band shapes do not match: {shape} != {band.shape}")

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    
    The NDVI is calculated as (NIR - RED) / (NIR + RED) and ranges from -1 to 1.
    Higher values indicate denser vegetation, while lower values indicate bare soil
    or water.
    
    Args:
        nir: Near-infrared band array (typically band 4 in Landsat/Sentinel)
        red: Red band array (typically band 3 in Landsat/Sentinel)
        
    Returns:
        NDVI array with values in range [-1, 1]
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If arrays have different shapes or contain invalid values
    """
    validate_bands(nir, red)
    
    # Calculate NDVI with proper error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        
    # Handle division by zero and invalid values
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure output is in [-1, 1] range
    np.clip(ndvi, -1.0, 1.0, out=ndvi)
    
    return ndvi

def evi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray,
        G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index (EVI).
    
    EVI = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)
    
    This index minimizes canopy background variations and maintains sensitivity over
    dense vegetation conditions.
    
    Args:
        red: Red band array
        nir: Near-infrared band array
        blue: Blue band array
        G: Gain factor (default: 2.5)
        C1: Coefficient 1 for atmospheric resistance (default: 6.0)
        C2: Coefficient 2 for atmospheric resistance (default: 7.5)
        L: Canopy background adjustment (default: 1.0)
        
    Returns:
        EVI array with values typically in range [-1, 1]
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If arrays have different shapes or contain invalid values
    """
    validate_bands(red, nir, blue)
    
    # Parameter validation
    if not all(isinstance(x, (int, float)) for x in [G, C1, C2, L]):
        raise TypeError("All coefficients must be numeric")
    if L < 0:
        raise ValueError("L must be non-negative")
    if G <= 0:
        raise ValueError("G must be positive")
    
    # Calculate EVI with proper error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = nir + C1 * red - C2 * blue + L
        evi = G * (nir - red) / denominator
        
    # Handle division by zero and invalid values
    evi = np.nan_to_num(evi, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip to reasonable range
    np.clip(evi, -1.0, 1.0, out=evi)
    
    return evi

def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    Calculate Soil Adjusted Vegetation Index (SAVI).
    
    SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
    
    This index minimizes soil brightness influences using a soil brightness
    correction factor (L).
    
    Args:
        nir: Near-infrared band array
        red: Red band array
        L: Soil brightness correction factor (default: 0.5)
            - 0.0 for very high vegetation cover
            - 0.5 for intermediate vegetation cover
            - 1.0 for low vegetation cover
        
    Returns:
        SAVI array with values in range [-1, 1]
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If arrays have different shapes or L is invalid
    """
    validate_bands(nir, red)
    
    # Parameter validation
    if not isinstance(L, (int, float)):
        raise TypeError("L must be numeric")
    if not 0 <= L <= 1:
        raise ValueError("L must be between 0 and 1")
    
    # Calculate SAVI with proper error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        savi = ((nir - red) / (nir + red + L)) * (1 + L)
        
    # Handle division by zero and invalid values
    savi = np.nan_to_num(savi, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure output is in [-1, 1] range
    np.clip(savi, -1.0, 1.0, out=savi)
    
    return savi

def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index (NDWI).
    
    The NDWI is calculated as (GREEN - NIR) / (GREEN + NIR) and ranges from -1 to 1.
    Positive values generally indicate water features, while negative values
    indicate vegetation or soil features.
    
    Args:
        green: Green band array (typically band 3 in Landsat/Sentinel)
        nir: Near-infrared band array (typically band 4 in Landsat/Sentinel)
        
    Returns:
        NDWI array with values in range [-1, 1]
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If arrays have different shapes or contain invalid values
    """
    validate_bands(green, nir)
    
    # Calculate NDWI with proper error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir)
        
    # Handle division by zero and invalid values
    ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure output is in [-1, 1] range
    np.clip(ndwi, -1.0, 1.0, out=ndwi)
    
    return ndwi 