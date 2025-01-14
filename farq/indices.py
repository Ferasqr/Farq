"""
Spectral indices module for the Farq library.

This module provides functions for calculating various spectral indices:
- NDWI (Normalized Difference Water Index)
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil Adjusted Vegetation Index)
- NDBI (Normalized Difference Built-up Index)
- NBR (Normalized Burn Ratio)
- NDMI (Normalized Difference Moisture Index)
"""

import numpy as np
from typing import Optional, Dict, Union, List, Tuple

def validate_bands(*bands: np.ndarray, reflectance_scale: Optional[float] = None) -> List[np.ndarray]:
    """
    Validate band arrays for spectral index calculations.
    
    Args:
        *bands: Variable number of band arrays to validate
        reflectance_scale: Optional scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
        
    Returns:
        List of validated and optionally scaled band arrays
        
    Raises:
        TypeError: If any band is not a numpy array
        ValueError: If bands have different shapes or are empty
    """
    if not bands:
        raise ValueError("No bands provided")
    
    validated_bands = []
    
    # Check each band
    for i, band in enumerate(bands):
        if not isinstance(band, np.ndarray):
            raise TypeError(f"Band {i} must be a numpy array")
        if band.size == 0:
            raise ValueError(f"Band {i} cannot be empty")
            
        # Make a copy and apply scaling if needed
        band_data = band.copy()
        if reflectance_scale is not None:
            band_data = band_data / reflectance_scale
            
        validated_bands.append(band_data)
    
    # Check shapes match
    shape = validated_bands[0].shape
    for i, band in enumerate(validated_bands[1:], 1):
        if band.shape != shape:
            raise ValueError(f"Band shapes do not match: {shape} != {band.shape}")
            
    return validated_bands

def calculate_normalized_difference(band1: np.ndarray, 
                                 band2: np.ndarray, 
                                 clip: bool = True) -> np.ndarray:
    """
    Calculate normalized difference between two bands.
    
    Args:
        band1: First band array
        band2: Second band array
        clip: Whether to clip values to [-1, 1] range
        
    Returns:
        Normalized difference array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        nd = (band1 - band2) / (band1 + band2)
        
    # Handle division by zero and invalid values
    nd = np.nan_to_num(nd, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip values if requested
    if clip:
        np.clip(nd, -1.0, 1.0, out=nd)
        
    return nd

def ndvi(nir: np.ndarray, 
         red: np.ndarray, 
         reflectance_scale: Optional[float] = None) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI) for Landsat 8.
    
    NDVI = (NIR - RED) / (NIR + RED)
    
    Args:
        nir: Near-infrared band (B5)
        red: Red band (B4)
        reflectance_scale: Scale factor for reflectance data (10000 for Landsat 8 SR)
        
    Returns:
        NDVI array with values in range [-1, 1]
        Higher values (>0.2) indicate vegetation
    """
    nir, red = validate_bands(nir, red, reflectance_scale=reflectance_scale)
    return calculate_normalized_difference(nir, red)

def evi(red: np.ndarray, 
        nir: np.ndarray, 
        blue: np.ndarray,
        reflectance_scale: Optional[float] = None,
        G: float = 2.5, 
        C1: float = 6.0, 
        C2: float = 7.5, 
        L: float = 1.0) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index (EVI) for Landsat 8.
    
    EVI = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)
    
    Args:
        red: Red band (B4)
        nir: Near-infrared band (B5)
        blue: Blue band (B2)
        reflectance_scale: Scale factor for reflectance data (10000 for Landsat 8 SR)
        G: Gain factor (default: 2.5)
        C1: Coefficient 1 for atmospheric resistance (default: 6.0)
        C2: Coefficient 2 for atmospheric resistance (default: 7.5)
        L: Canopy background adjustment (default: 1.0)
    """
    red, nir, blue = validate_bands(red, nir, blue, reflectance_scale=reflectance_scale)
    
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

def savi(nir: np.ndarray, 
         red: np.ndarray, 
         reflectance_scale: Optional[float] = None,
         L: float = 0.5) -> np.ndarray:
    """
    Calculate Soil Adjusted Vegetation Index (SAVI) for Landsat 8.
    
    SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
    
    Args:
        nir: Near-infrared band (B5)
        red: Red band (B4)
        reflectance_scale: Scale factor for reflectance data (10000 for Landsat 8 SR)
        L: Soil brightness correction factor (default: 0.5)
    """
    nir, red = validate_bands(nir, red, reflectance_scale=reflectance_scale)
    
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

def ndwi(green: np.ndarray, 
         nir: np.ndarray, 
         reflectance_scale: Optional[float] = None) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index (NDWI).
    
    For Landsat 8:
    NDWI = (NIR - GREEN) / (NIR + GREEN)
    Water typically has values < 0
    
    Args:
        green: Green band array (B3 in Landsat 8)
        nir: Near-infrared band array (B5 in Landsat 8)
        reflectance_scale: Scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
        
    Returns:
        NDWI array with values in range [-1, 1]
    """
    green, nir = validate_bands(green, nir, reflectance_scale=reflectance_scale)
    return calculate_normalized_difference(nir, green)  # Flipped order for Landsat 8

def ndbi(swir1: np.ndarray, 
         nir: np.ndarray, 
         reflectance_scale: Optional[float] = None) -> np.ndarray:
    """
    Calculate Normalized Difference Built-up Index (NDBI) for Landsat 8.
    
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    
    Args:
        swir1: Short-wave infrared band 1 (B6)
        nir: Near-infrared band (B5)
        reflectance_scale: Scale factor for reflectance data (10000 for Landsat 8 SR)
        
    Returns:
        NDBI array with values in range [-1, 1]
        Higher values indicate built-up areas
    """
    swir1, nir = validate_bands(swir1, nir, reflectance_scale=reflectance_scale)
    return calculate_normalized_difference(swir1, nir)

def nbr(nir: np.ndarray, 
        swir2: np.ndarray, 
        reflectance_scale: Optional[float] = None) -> np.ndarray:
    """
    Calculate Normalized Burn Ratio (NBR) for Landsat 8.
    
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    
    Args:
        nir: Near-infrared band (B5)
        swir2: Short-wave infrared band 2 (B7)
        reflectance_scale: Scale factor for reflectance data (10000 for Landsat 8 SR)
        
    Returns:
        NBR array with values in range [-1, 1]
        Lower values indicate burned areas
    """
    nir, swir2 = validate_bands(nir, swir2, reflectance_scale=reflectance_scale)
    return calculate_normalized_difference(nir, swir2)

def ndmi(nir: np.ndarray, 
         swir1: np.ndarray, 
         reflectance_scale: Optional[float] = None) -> np.ndarray:
    """
    Calculate Normalized Difference Moisture Index (NDMI) for Landsat 8.
    
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    
    Args:
        nir: Near-infrared band (B5)
        swir1: Short-wave infrared band 1 (B6)
        reflectance_scale: Scale factor for reflectance data (10000 for Landsat 8 SR)
        
    Returns:
        NDMI array with values in range [-1, 1]
        Higher values indicate higher moisture content
    """
    nir, swir1 = validate_bands(nir, swir1, reflectance_scale=reflectance_scale)
    return calculate_normalized_difference(nir, swir1)

def calculate_indices(bands: Dict[str, np.ndarray], 
                     indices: List[str],
                     reflectance_scale: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Calculate multiple spectral indices at once.
    
    Args:
        bands: Dictionary of band arrays with keys like 'red', 'nir', 'swir1', etc.
        indices: List of index names to calculate ('ndvi', 'ndwi', etc.)
        reflectance_scale: Scale factor for reflectance data
        
    Returns:
        Dictionary of calculated indices
        
    Example:
        >>> bands = {'red': red_array, 'nir': nir_array, 'green': green_array}
        >>> indices = calculate_indices(bands, ['ndvi', 'ndwi'], reflectance_scale=10000)
        >>> ndvi_array = indices['ndvi']
        >>> ndwi_array = indices['ndwi']
    """
    available_indices = {
        'ndvi': (('nir', 'red'), ndvi),
        'ndwi': (('green', 'nir'), ndwi),
        'evi': (('red', 'nir', 'blue'), evi),
        'savi': (('nir', 'red'), savi),
        'ndbi': (('swir1', 'nir'), ndbi),
        'nbr': (('nir', 'swir2'), nbr),
        'ndmi': (('nir', 'swir1'), ndmi)
    }
    
    result = {}
    
    for index_name in indices:
        if index_name not in available_indices:
            raise ValueError(f"Unknown index: {index_name}")
            
        required_bands, func = available_indices[index_name]
        
        # Check if all required bands are available
        missing_bands = [band for band in required_bands if band not in bands]
        if missing_bands:
            raise ValueError(f"Missing required bands for {index_name}: {missing_bands}")
        
        # Calculate the index
        band_arrays = [bands[band] for band in required_bands]
        result[index_name] = func(*band_arrays, reflectance_scale=reflectance_scale)
    
    return result 