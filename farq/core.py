"""
Core functionality for the Farq library.

This module provides fundamental operations for raster data processing including:
- Raster resampling
- File I/O operations
- Band validation

All functions include input validation and detailed error messages.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from typing import Tuple, Union, Optional
import os
from .utils import validate_array

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
        validate_array(band, name=f"Band {i}")
    
    # Check shapes match
    shape = bands[0].shape
    for i, band in enumerate(bands[1:], 1):
        if band.shape != shape:
            raise ValueError(f"Band shapes do not match: {shape} != {band.shape}")

def resample(array: np.ndarray, target_shape: Tuple[int, int], 
            method: Resampling = Resampling.bilinear) -> np.ndarray:
    """
    Resample array to target shape using specified resampling method.
    
    Args:
        array: Input numpy array
        target_shape: Desired output shape as (height, width)
        method: Resampling method from rasterio.enums.Resampling
        
    Returns:
        Resampled array with target shape
        
    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If array is empty or target shape is invalid
    """
    # Validate inputs
    validate_array(array)
    if not isinstance(target_shape, tuple) or len(target_shape) != 2:
        raise TypeError("target_shape must be a tuple of (height, width)")
    if not all(isinstance(x, int) and x > 0 for x in target_shape):
        raise ValueError("target_shape dimensions must be positive integers")
    
    # Create temporary rasterio dataset for resampling
    profile = {
        'driver': 'MEM',
        'height': array.shape[0],
        'width': array.shape[1],
        'count': 1,
        'dtype': array.dtype
    }
    
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            # Write array to dataset
            dataset.write(array, 1)
            
            # Perform resampling
            resampled = dataset.read(
                1,
                out_shape=target_shape,
                resampling=method
            )
    
    return resampled

def read(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Read raster data from file.
    
    Args:
        filepath: Path to raster file
        
    Returns:
        Tuple containing:
            - Numpy array containing raster data
            - Dictionary of metadata from the raster file
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be read as a raster
        RuntimeError: If there are issues reading the file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            metadata = src.meta.copy()
            return data, metadata
    except rasterio.errors.RasterioIOError as e:
        raise ValueError(f"Unable to read raster file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {filepath}: {e}")

def write(filepath: str, data: np.ndarray, metadata: dict) -> None:
    """
    Write raster data to file.
    
    Args:
        filepath: Path to output file
        data: Numpy array containing raster data
        metadata: Dictionary of raster metadata
        
    Raises:
        ValueError: If data or metadata are invalid
        RuntimeError: If there are issues writing the file
    """
    validate_array(data)
    
    try:
        with rasterio.open(filepath, 'w', **metadata) as dst:
            dst.write(data, 1)
    except Exception as e:
        raise RuntimeError(f"Error writing file {filepath}: {e}")