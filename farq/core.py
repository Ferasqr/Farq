import numpy as np
import rasterio
from scipy import stats
from typing import Tuple, Dict, Union
import pandas as pd
from rasterio.enums import Resampling

def read(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load a raster file.
    
    Args:
        filepath (str): Path to the raster file
        
    Returns:
        Tuple[np.ndarray, Dict]: Raster data array and metadata dictionary
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)  # Read the first band
        metadata = src.meta
    return data, metadata

def _normalize(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    """
    Helper function to calculate normalized difference indices.
    """
    band1 = band1.astype(float)
    band2 = band2.astype(float)
    return (band1 - band2) / (band1 + band2 + 1e-10)

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI (Normalized Difference Vegetation Index).
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        nir (np.ndarray): NIR band data
        red (np.ndarray): Red band data
        
    Returns:
        np.ndarray: NDVI values [-1, 1]
    """
    return np.clip(_normalize(nir, red), -1, 1)

def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate NDWI (Normalized Difference Water Index).
    NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        green (np.ndarray): Green band data
        nir (np.ndarray): NIR band data
        
    Returns:
        np.ndarray: NDWI values [-1, 1]
    """
    return np.clip(_normalize(green, nir), -1, 1)

def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Calculate MNDWI (Modified Normalized Difference Water Index).
    MNDWI = (Green - SWIR) / (Green + SWIR)
    
    Args:
        green (np.ndarray): Green band data
        swir (np.ndarray): SWIR band data
        
    Returns:
        np.ndarray: MNDWI values [-1, 1]
    """
    return np.clip(_normalize(green, swir), -1, 1)

def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate NDBI (Normalized Difference Built-up Index).
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    
    Args:
        swir (np.ndarray): SWIR band data
        nir (np.ndarray): NIR band data
        
    Returns:
        np.ndarray: NDBI values [-1, 1]
    """
    return np.clip(_normalize(swir, nir), -1, 1)

def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    Calculate SAVI (Soil Adjusted Vegetation Index).
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    
    Args:
        nir (np.ndarray): NIR band data
        red (np.ndarray): Red band data
        L (float): Soil brightness correction factor (default 0.5)
        
    Returns:
        np.ndarray: SAVI values [-1, 1]
    """
    nir = nir.astype(float)
    red = red.astype(float)
    return np.clip(((nir - red) / (nir + red + L)) * (1 + L), -1, 1)

def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, 
        G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
    """
    Calculate EVI (Enhanced Vegetation Index).
    EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
    
    Args:
        nir (np.ndarray): NIR band data
        red (np.ndarray): Red band data
        blue (np.ndarray): Blue band data
        G (float): Gain factor
        C1, C2 (float): Aerosol resistance coefficients
        L (float): Canopy background adjustment
        
    Returns:
        np.ndarray: EVI values typically [-1, 1]
    """
    nir = nir.astype(float)
    red = red.astype(float)
    blue = blue.astype(float)
    
    evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L)
    return np.clip(evi, -1, 1)

def resample(raster: np.ndarray, shape: Tuple[int, int], 
            method: Resampling = Resampling.bilinear) -> np.ndarray:
    """
    Resample a raster to new shape.
    
    Args:
        raster (np.ndarray): Input raster array
        shape (Tuple[int, int]): Target shape (height, width)
        method (Resampling): Resampling method
        
    Returns:
        np.ndarray: Resampled raster array
    """
    profile = {
        'driver': 'GTiff',
        'height': raster.shape[0],
        'width': raster.shape[1],
        'count': 1,
        'dtype': raster.dtype
    }
    
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(raster, 1)
            data = dataset.read(1, out_shape=shape, resampling=method)
    
    return data

def diff(raster1: np.ndarray, raster2: np.ndarray, 
        method: str = "simple",
        resample_method: Resampling = Resampling.bilinear) -> np.ndarray:
    """
    Detect changes between rasters.
    
    Args:
        raster1 (np.ndarray): First raster
        raster2 (np.ndarray): Second raster
        method (str): Method ('simple', 'ratio', or 'norm')
        resample_method (Resampling): Resampling method if sizes differ
        
    Returns:
        np.ndarray: Change detection results
    """
    if raster1.shape != raster2.shape:
        print(f"Warning: Resampling to shape {raster1.shape}")
        raster2 = resample(raster2, raster1.shape, resample_method)
    
    if method == "simple":
        changes = raster2 - raster1
    elif method == "ratio":
        changes = raster2 / (raster1 + 1e-10)
    elif method == "norm":
        changes = (raster2 - raster1) / (raster2 + raster1 + 1e-10)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return changes

def stats(data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Get basic statistics of data.
    
    Args:
        data (np.ndarray): Input data array
        
    Returns:
        Dict: Statistics dictionary
    """
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "p25_50_75": np.percentile(data, [25, 50, 75]),
        "hist": np.histogram(data, bins=50)
    }