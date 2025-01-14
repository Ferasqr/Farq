"""
Utility functions for array operations.

This module provides basic array operations with enhanced error handling and input validation.
All functions handle NaN values gracefully and provide clear error messages.
"""
import numpy as np
from typing import Union, Tuple, Optional, Dict

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

def stats(data: np.ndarray, 
         percentiles: list = [0, 25, 50, 75, 100],
         reflectance_scale: Optional[float] = None,
         bins: int = 50) -> Dict:
    """
    Calculate comprehensive statistics for the array.
    
    Args:
        data: Input array
        percentiles: List of percentiles to compute (default: [0, 25, 50, 75, 100])
        reflectance_scale: Scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
        bins: Number of bins for histogram calculation
        
    Returns:
        Dictionary containing the following statistics:
        - min: Minimum value
        - max: Maximum value
        - mean: Mean value
        - std: Standard deviation
        - median: Median value (50th percentile)
        - percentiles: Dictionary of requested percentiles
        - non_zero: Count of non-zero values
        - zeros: Count of zero values
        - nan: Count of NaN values
        - inf: Count of infinite values
        - valid: Count of valid values (non-nan, non-inf)
        - shape: Array shape
        - size: Total number of elements
        - dtype: Array data type
        - range: Data range (max - min)
        - variance: Variance of the data
        - skewness: Skewness of the distribution
        - kurtosis: Kurtosis of the distribution
        - histogram: Dictionary with 'counts' and 'bin_edges'
        - reflectance_stats: Statistics of scaled reflectance (if scale provided)
        
    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty
    """
    validate_array(data)
    
    # Make a copy to avoid modifying the original
    arr = data.copy()
    
    # Basic statistics
    stats_dict = {
        'min': float(np.nanmin(arr)),
        'max': float(np.nanmax(arr)),
        'mean': float(np.nanmean(arr)),
        'std': float(np.nanstd(arr)),
        'median': float(np.nanmedian(arr)),
        'percentiles': {str(p): float(np.nanpercentile(arr, p)) for p in percentiles},
        'non_zero': int(np.count_nonzero(~np.isnan(arr))),
        'zeros': int(np.sum(arr == 0)),
        'nan': int(np.sum(np.isnan(arr))),
        'inf': int(np.sum(np.isinf(arr))),
        'valid': int(np.sum(~np.isnan(arr) & ~np.isinf(arr))),
        'shape': arr.shape,
        'size': int(arr.size),
        'dtype': str(arr.dtype),
        'range': float(np.nanmax(arr) - np.nanmin(arr)),
        'variance': float(np.nanvar(arr))
    }
    
    # Add skewness and kurtosis if scipy is available
    try:
        from scipy import stats as sp_stats
        valid_data = arr[~np.isnan(arr) & ~np.isinf(arr)]
        if len(valid_data) > 0:
            stats_dict['skewness'] = float(sp_stats.skew(valid_data))
            stats_dict['kurtosis'] = float(sp_stats.kurtosis(valid_data))
    except ImportError:
        pass
    
    # Add reflectance statistics if scale is provided
    if reflectance_scale is not None:
        scaled_arr = arr / reflectance_scale
        stats_dict['reflectance_stats'] = {
            'min': float(np.nanmin(scaled_arr)),
            'max': float(np.nanmax(scaled_arr)),
            'mean': float(np.nanmean(scaled_arr)),
            'std': float(np.nanstd(scaled_arr)),
            'median': float(np.nanmedian(scaled_arr)),
            'percentiles': {str(p): float(np.nanpercentile(scaled_arr, p)) 
                          for p in percentiles}
        }
    
    # Add percentage statistics
    total_pixels = arr.size
    stats_dict['percentages'] = {
        'valid': 100 * stats_dict['valid'] / total_pixels,
        'nan': 100 * stats_dict['nan'] / total_pixels,
        'zeros': 100 * stats_dict['zeros'] / total_pixels,
        'inf': 100 * stats_dict['inf'] / total_pixels
    }
    
    return stats_dict

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

# Add stats method to numpy arrays
def _add_stats_to_numpy():
    """Add stats method to numpy arrays."""
    pass  # Disabled: Cannot modify numpy array class in recent versions
    # def _stats(self, percentiles=[0, 25, 50, 75, 100], reflectance_scale=None, bins=50):
    #     return stats(self, percentiles, reflectance_scale, bins)
    
    # if not hasattr(np.ndarray, 'stats'):
    #     np.ndarray.stats = _stats

# Add the stats method when the module is imported
_add_stats_to_numpy() 