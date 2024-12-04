"""
Utility functions for array operations.

This module provides basic array operations with enhanced error handling and input validation.
All functions handle NaN values gracefully and provide clear error messages.
"""
import numpy as np
from typing import Union, Tuple, Optional

def validate_array(array: np.ndarray, name: str = "array") -> None:
    """
    Validate numpy array inputs for basic operations.
    
    Args:
        array: Input numpy array to validate
        name: Name of the array for error messages
        
    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or contains all NaN values
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array)}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.all(np.isnan(array)):
        raise ValueError(f"{name} cannot contain all NaN values")

def sum(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calculate the sum of array elements, ignoring NaN values.
    
    Args:
        data: Input array
        axis: Axis along which to calculate sum (None for entire array)
        
    Returns:
        Sum of array elements
        
    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or contains all NaN values
    """
    validate_array(data)
    return np.nansum(data, axis=axis)

def mean(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calculate the mean of array elements, ignoring NaN values.
    
    Args:
        data: Input array
        axis: Axis along which to calculate mean (None for entire array)
        
    Returns:
        Mean of array elements
        
    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or contains all NaN values
    """
    validate_array(data)
    result = np.nanmean(data, axis=axis)
    if np.isnan(result).any():
        raise ValueError("No valid values found in array (all NaN)")
    return result

def std(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calculate the standard deviation of array elements, ignoring NaN values.
    
    Args:
        data: Input array
        axis: Axis along which to calculate std (None for entire array)
        
    Returns:
        Standard deviation of array elements
        
    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or contains all NaN values
    """
    validate_array(data)
    result = np.nanstd(data, axis=axis)
    if np.isnan(result).any():
        raise ValueError("No valid values found in array (all NaN)")
    return result

def min(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calculate the minimum of array elements, ignoring NaN values.
    
    Args:
        data: Input array
        axis: Axis along which to calculate minimum (None for entire array)
        
    Returns:
        Minimum of array elements
        
    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or contains all NaN values
    """
    validate_array(data)
    result = np.nanmin(data, axis=axis)
    if np.isnan(result).any():
        raise ValueError("No valid values found in array (all NaN)")
    return result

def max(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calculate the maximum of array elements, ignoring NaN values.
    
    Args:
        data: Input array
        axis: Axis along which to calculate maximum (None for entire array)
        
    Returns:
        Maximum of array elements
        
    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or contains all NaN values
    """
    validate_array(data)
    result = np.nanmax(data, axis=axis)
    if np.isnan(result).any():
        raise ValueError("No valid values found in array (all NaN)")
    return result

def median(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculate the median of array elements."""
    return np.median(data, axis=axis)

def percentile(data: np.ndarray, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the qth percentile of the data."""
    return np.percentile(data, q)

def count_nonzero(data: np.ndarray, axis: Optional[int] = None) -> Union[int, np.ndarray]:
    """Count non-zero values in the array."""
    return np.count_nonzero(data, axis=axis)

def unique(data: np.ndarray, return_counts: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Find unique elements in array."""
    return np.unique(data, return_counts=return_counts) 