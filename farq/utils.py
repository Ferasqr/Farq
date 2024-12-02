"""
Utility functions for array operations.
"""
import numpy as np
from typing import Union, Tuple, Optional

def sum(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculate the sum of array elements."""
    return np.sum(data, axis=axis)

def mean(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculate the mean of array elements."""
    return np.mean(data, axis=axis)

def std(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculate the standard deviation of array elements."""
    return np.std(data, axis=axis)

def min(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Find the minimum value in the array."""
    return np.min(data, axis=axis)

def max(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Find the maximum value in the array."""
    return np.max(data, axis=axis)

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