"""
Core functionality for the Farq library.
Provides functions for loading raster data, calculating spectral indices,
and performing basic raster operations.

The module includes:
- Raster I/O functions
- Spectral index calculations (NDVI, NDWI, etc.)
- Basic raster operations (resampling, differencing)
"""
import numpy as np
import rasterio
from scipy import stats
from typing import Tuple, Dict, Union, Optional
import pandas as pd
from rasterio.enums import Resampling
from .utils import mean, std, min, max, percentile

class FarqError(Exception):
    """Base exception class for Farq-specific errors."""
    pass

class InvalidBandError(FarqError):
    """Exception raised for invalid band data."""
    pass

class ShapeMismatchError(FarqError):
    """Exception raised when array shapes don't match."""
    pass

def read(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load a raster file and return its data and metadata.
    
    Args:
        filepath: Path to the raster file
        
    Returns:
        Tuple containing:
        - np.ndarray: Raster data array
        - Dict: Metadata dictionary with raster properties
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        rasterio.errors.RasterioError: If there's an error reading the file
        
    Example:
        >>> import farq
        >>> data, meta = farq.read("landsat_band.tif")
    """
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)  # Read the first band
            metadata = src.meta
        return data, metadata
    except rasterio.errors.RasterioError as e:
        raise FarqError(f"Error reading raster file: {str(e)}")

def _validate_bands(*bands: np.ndarray) -> None:
    """
    Validate band arrays for spectral index calculations.
    
    Args:
        *bands: Variable number of band arrays to validate
        
    Raises:
        InvalidBandError: If any band is invalid
        ShapeMismatchError: If bands have different shapes
    """
    if not bands:
        raise InvalidBandError("No bands provided")
    
    # Check for valid arrays
    for i, band in enumerate(bands):
        if not isinstance(band, np.ndarray):
            raise InvalidBandError(f"Band {i} is not a numpy array")
        if band.size == 0:
            raise InvalidBandError(f"Band {i} is empty")
        if not np.issubdtype(band.dtype, np.number):
            raise InvalidBandError(f"Band {i} contains non-numeric data")
    
    # Check shapes match
    shape = bands[0].shape
    for i, band in enumerate(bands[1:], 1):
        if band.shape != shape:
            raise ShapeMismatchError(
                f"Band shape mismatch: Band 0 {shape} != Band {i} {band.shape}"
            )

def _normalize(band1: np.ndarray, band2: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Calculate normalized difference between two bands.
    
    Args:
        band1: First band array
        band2: Second band array
        epsilon: Small value to avoid division by zero
        
    Returns:
        np.ndarray: Normalized difference values
        
    Raises:
        InvalidBandError: If input bands are invalid
        ShapeMismatchError: If band shapes don't match
    """
    # Validate inputs
    _validate_bands(band1, band2)
    
    # Convert to float and handle NaN/Inf values
    band1 = band1.astype(float)
    band2 = band2.astype(float)
    
    # Create mask for valid values
    valid_mask = ~(np.isnan(band1) | np.isnan(band2) | 
                  np.isinf(band1) | np.isinf(band2))
    
    # Initialize result array
    result = np.zeros_like(band1, dtype=float)
    result[~valid_mask] = np.nan
    
    # Calculate normalized difference only for valid values
    valid_data = valid_mask.copy()
    
    # Add mask for zero sum to avoid division by zero
    sum_data = band1 + band2
    valid_data &= (np.abs(sum_data) > epsilon)
    
    # Calculate normalized difference
    if np.any(valid_data):
        result[valid_data] = ((band1[valid_data] - band2[valid_data]) / 
                            (band1[valid_data] + band2[valid_data]))
    
    return result

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        nir: NIR band data
        red: Red band data
        
    Returns:
        np.ndarray: NDVI values [-1, 1]
        
    Raises:
        InvalidBandError: If input bands are invalid
        ShapeMismatchError: If band shapes don't match
        
    Example:
        >>> import farq
        >>> ndvi = farq.ndvi(nir_band, red_band)
    """
    return np.clip(_normalize(nir, red), -1, 1)

def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index.
    NDWI = (Green - NIR) / (Green + NIR)
    
    The NDWI is used to delineate open water features. Values > 0 typically
    indicate water, while values ≤ 0 indicate non-water features.
    
    Args:
        green: Green band data (typically Landsat band 3)
        nir: NIR band data (typically Landsat band 5)
        
    Returns:
        np.ndarray: NDWI values [-1, 1]
        
    Raises:
        InvalidBandError: If input bands are invalid
        ShapeMismatchError: If band shapes don't match
        
    Note:
        - Input bands should be surface reflectance values
        - Values are clipped to [-1, 1] range
        - NaN values in input will result in NaN output
        - Division by zero is handled gracefully
    """
    # Validate inputs
    if np.all(green <= 0) or np.all(nir <= 0):
        raise InvalidBandError(
            "Input bands appear to be invalid. "
            "Surface reflectance values should be positive."
        )
    
    # Calculate NDWI
    ndwi_values = _normalize(green, nir)
    
    # Clip values to valid range
    return np.clip(ndwi_values, -1, 1)

def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Calculate Modified Normalized Difference Water Index.
    MNDWI = (Green - SWIR) / (Green + SWIR)
    
    Args:
        green: Green band data
        swir: SWIR band data
        
    Returns:
        np.ndarray: MNDWI values [-1, 1]
        
    Raises:
        InvalidBandError: If input bands are invalid
        ShapeMismatchError: If band shapes don't match
        
    Example:
        >>> import farq
        >>> mndwi = farq.mndwi(green_band, swir_band)
    """
    return np.clip(_normalize(green, swir), -1, 1)

def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Built-up Index.
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    
    Args:
        swir: SWIR band data
        nir: NIR band data
        
    Returns:
        np.ndarray: NDBI values [-1, 1]
        
    Raises:
        InvalidBandError: If input bands are invalid
        ShapeMismatchError: If band shapes don't match
        
    Example:
        >>> import farq
        >>> ndbi = farq.ndbi(swir_band, nir_band)
    """
    return np.clip(_normalize(swir, nir), -1, 1)

def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    Calculate Soil Adjusted Vegetation Index.
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    
    Args:
        nir: NIR band data
        red: Red band data
        L: Soil brightness correction factor (default: 0.5)
        
    Returns:
        np.ndarray: SAVI values [-1, 1]
        
    Raises:
        InvalidBandError: If input bands are invalid
        ShapeMismatchError: If band shapes don't match
        ValueError: If L is not in range [0, 1]
        
    Example:
        >>> import farq
        >>> savi = farq.savi(nir_band, red_band, L=0.5)
    """
    if not 0 <= L <= 1:
        raise ValueError("L must be between 0 and 1")
    
    _validate_bands(nir, red)
    
    nir = nir.astype(float)
    red = red.astype(float)
    return np.clip(((nir - red) / (nir + red + L)) * (1 + L), -1, 1)

def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, 
        G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index.
    EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
    
    Args:
        nir: NIR band data
        red: Red band data
        blue: Blue band data
        G: Gain factor (default: 2.5)
        C1: First coefficient (default: 6.0)
        C2: Second coefficient (default: 7.5)
        L: Canopy background adjustment (default: 1.0)
        
    Returns:
        np.ndarray: EVI values typically [-1, 1]
        
    Raises:
        InvalidBandError: If input bands are invalid
        ShapeMismatchError: If band shapes don't match
        ValueError: If coefficients are invalid
        
    Example:
        >>> import farq
        >>> evi = farq.evi(nir_band, red_band, blue_band)
    """
    if G <= 0:
        raise ValueError("G must be positive")
    if C1 < 0 or C2 < 0:
        raise ValueError("C1 and C2 must be non-negative")
    if L < 0:
        raise ValueError("L must be non-negative")
    
    _validate_bands(nir, red, blue)
    
    nir = nir.astype(float)
    red = red.astype(float)
    blue = blue.astype(float)
    
    evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L)
    return np.clip(evi, -1, 1)

def resample(raster: np.ndarray, shape: Tuple[int, int], 
            method: Resampling = Resampling.bilinear) -> np.ndarray:
    """
    Resample a raster to a new shape.
    
    Args:
        raster: Input raster array
        shape: Target shape (height, width)
        method: Resampling method (default: bilinear)
        
    Returns:
        np.ndarray: Resampled raster array
        
    Raises:
        InvalidBandError: If input raster is invalid
        ValueError: If target shape is invalid
        
    Example:
        >>> import farq
        >>> resampled = farq.resample(data, (100, 100))
    """
    if not isinstance(raster, np.ndarray):
        raise InvalidBandError("Input must be a numpy array")
    if len(shape) != 2 or not all(s > 0 for s in shape):
        raise ValueError("Shape must be a tuple of two positive integers")
    
    profile = {
        'driver': 'GTiff',
        'height': raster.shape[0],
        'width': raster.shape[1],
        'count': 1,
        'dtype': raster.dtype
    }
    
    try:
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(raster, 1)
                data = dataset.read(1, out_shape=shape, resampling=method)
        return data
    except Exception as e:
        raise FarqError(f"Error resampling raster: {str(e)}")

def diff(raster1: np.ndarray, raster2: np.ndarray, 
        method: str = "simple",
        resample_method: Resampling = Resampling.bilinear) -> np.ndarray:
    """
    Detect changes between two rasters.
    
    Args:
        raster1: First raster
        raster2: Second raster
        method: Method ('simple', 'ratio', or 'norm')
        resample_method: Resampling method if sizes differ
        
    Returns:
        np.ndarray: Change detection results
        
    Raises:
        InvalidBandError: If input rasters are invalid
        ValueError: If method is invalid
        
    Example:
        >>> import farq
        >>> changes = farq.diff(raster2, raster1, method="simple")
    """
    _validate_bands(raster1, raster2)
    
    valid_methods = {"simple", "ratio", "norm"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method. Must be one of: {valid_methods}")
    
    if raster1.shape != raster2.shape:
        print(f"Warning: Resampling raster2 to shape {raster1.shape}")
        try:
            raster2 = resample(raster2, raster1.shape, resample_method)
        except Exception as e:
            raise FarqError(f"Error resampling for difference calculation: {str(e)}")
    
    try:
        if method == "simple":
            changes = raster2 - raster1
        elif method == "ratio":
            changes = raster2 / (raster1 + 1e-10)  # Avoid division by zero
        else:  # method == "norm"
            changes = (raster2 - raster1) / (raster2 + raster1 + 1e-10)
        return changes
    except Exception as e:
        raise FarqError(f"Error calculating difference: {str(e)}")

def stats(data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate basic statistics of data.
    
    Args:
        data: Input data array
        
    Returns:
        Dict containing:
        - mean: Mean value
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - p25_50_75: 25th, 50th, and 75th percentiles
        - hist: Histogram data
        
    Raises:
        InvalidBandError: If input data is invalid
        
    Example:
        >>> import farq
        >>> stats = farq.stats(data)
        >>> print(f"Mean: {stats['mean']:.2f}")
    """
    if not isinstance(data, np.ndarray):
        raise InvalidBandError("Input must be a numpy array")
    if data.size == 0:
        raise InvalidBandError("Input array is empty")
    
    try:
        return {
            "mean": mean(data),
            "std": std(data),
            "min": min(data),
            "max": max(data),
            "p25_50_75": percentile(data, [25, 50, 75]),
            "hist": np.histogram(data, bins=50)
        }
    except Exception as e:
        raise FarqError(f"Error calculating statistics: {str(e)}")