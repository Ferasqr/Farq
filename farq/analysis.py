"""
Analysis functions for water detection and change analysis.

This module provides functions for:
- Water body statistics calculation
- Change detection between water masks
- Individual water body identification and analysis

All functions include input validation and detailed error messages.
"""
import numpy as np
from typing import Dict, Union, Tuple, Optional
from scipy import ndimage
from .core import sum

def water_stats(water_mask: np.ndarray, 
               pixel_size: Union[float, Tuple[float, float]] = 30.0) -> Dict[str, float]:
    """
    Calculate basic water surface statistics from a water mask.
    
    Args:
        water_mask: Binary water mask (True/1 for water, False/0 for non-water)
        pixel_size: Pixel size in meters. Default 30.0 (Landsat resolution)
            Can be a single float for square pixels or a tuple (width, height)
        
    Returns:
        Dict with statistics:
            - total_area: Total water surface area in square kilometers
            - coverage_percent: Percentage of area covered by water
            
    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If water_mask is empty or pixel_size is invalid
    """
    # Input validation
    if not isinstance(water_mask, np.ndarray):
        raise TypeError("water_mask must be a numpy array")
    if water_mask.size == 0:
        raise ValueError("water_mask cannot be empty")
    
    if isinstance(pixel_size, (int, float)):
        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive")
        pixel_area = (pixel_size * pixel_size) / 1_000_000  # Convert to km²
    elif isinstance(pixel_size, tuple):
        if len(pixel_size) != 2:
            raise ValueError("pixel_size tuple must have exactly 2 elements")
        if any(p <= 0 for p in pixel_size):
            raise ValueError("Pixel sizes must be positive")
        pixel_area = (pixel_size[0] * pixel_size[1]) / 1_000_000  # Convert to km²
    else:
        raise TypeError("pixel_size must be a number or tuple of two numbers")
    
    # Convert to binary mask
    mask = water_mask.astype(bool)
    
    # Calculate total water area and coverage
    total_pixels = sum(mask)
    total_area = total_pixels * pixel_area
    coverage = (total_pixels / mask.size) * 100
    
    return {
        "total_area": total_area,  # km²
        "coverage_percent": coverage  # %
    }

def water_change(mask1: np.ndarray, 
                mask2: np.ndarray,
                pixel_size: Union[float, Tuple[float, float]] = 30.0) -> Dict[str, float]:
    """
    Analyze changes between two water masks.
    
    Args:
        mask1: First water mask (True/1 for water)
        mask2: Second water mask (True/1 for water)
        pixel_size: Pixel size in meters
            Can be a single float for square pixels or a tuple (width, height)
        
    Returns:
        Dict with change statistics:
            - gained_area: New water area in square kilometers
            - lost_area: Lost water area in square kilometers
            - net_change: Net change in water area in square kilometers
            - change_percent: Percentage change relative to original area
            
    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If masks have different shapes or pixel_size is invalid
    """
    # Input validation
    if not isinstance(mask1, np.ndarray) or not isinstance(mask2, np.ndarray):
        raise TypeError("Masks must be numpy arrays")
    if mask1.shape != mask2.shape:
        raise ValueError("Input masks must have the same shape")
    if mask1.size == 0 or mask2.size == 0:
        raise ValueError("Input masks cannot be empty")
    
    if isinstance(pixel_size, (int, float)):
        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive")
        pixel_area = (pixel_size * pixel_size) / 1_000_000  # Convert to km²
    elif isinstance(pixel_size, tuple):
        if len(pixel_size) != 2:
            raise ValueError("pixel_size tuple must have exactly 2 elements")
        if any(p <= 0 for p in pixel_size):
            raise ValueError("Pixel sizes must be positive")
        pixel_area = (pixel_size[0] * pixel_size[1]) / 1_000_000  # Convert to km²
    else:
        raise TypeError("pixel_size must be a number or tuple of two numbers")
    
    # Convert to binary masks
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate changes
    gained = np.logical_and(~mask1, mask2)
    lost = np.logical_and(mask1, ~mask2)
    
    # Calculate areas
    gained_area = sum(gained) * pixel_area
    lost_area = sum(lost) * pixel_area
    net_change = gained_area - lost_area
    
    # Calculate percentage change
    original_area = sum(mask1) * pixel_area
    if original_area > 0:
        change_percent = (net_change / original_area) * 100
    else:
        change_percent = float('inf') if gained_area > 0 else 0
    
    return {
        "gained_area": gained_area,  # km²
        "lost_area": lost_area,  # km²
        "net_change": net_change,  # km²
        "change_percent": change_percent  # %
    }

def get_water_bodies(water_mask: np.ndarray,
                    pixel_size: Union[float, Tuple[float, float]] = 30.0,
                    min_area: Optional[float] = None) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Label individual water bodies and calculate their areas.
    
    Args:
        water_mask: Binary water mask
        pixel_size: Pixel size in meters
            Can be a single float for square pixels or a tuple (width, height)
        min_area: Minimum water body area in square meters (optional)
            Water bodies smaller than this will be filtered out
        
    Returns:
        Tuple containing:
            - Labeled array where each water body has a unique integer ID
            - Dictionary mapping water body IDs to their areas in square kilometers
            
    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If water_mask is empty or pixel_size is invalid
    """
    # Input validation
    if not isinstance(water_mask, np.ndarray):
        raise TypeError("water_mask must be a numpy array")
    if water_mask.size == 0:
        raise ValueError("water_mask cannot be empty")
    
    if isinstance(pixel_size, (int, float)):
        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive")
        pixel_area = (pixel_size * pixel_size) / 1_000_000  # Convert to km²
    elif isinstance(pixel_size, tuple):
        if len(pixel_size) != 2:
            raise ValueError("pixel_size tuple must have exactly 2 elements")
        if any(p <= 0 for p in pixel_size):
            raise ValueError("Pixel sizes must be positive")
        pixel_area = (pixel_size[0] * pixel_size[1]) / 1_000_000  # Convert to km²
    else:
        raise TypeError("pixel_size must be a number or tuple of two numbers")
    
    # Convert to binary mask
    mask = water_mask.astype(bool)
    
    # Label water bodies
    labeled_mask, num_features = ndimage.label(mask)
    
    if min_area is not None:
        min_pixels = min_area / (pixel_area * 1_000_000)  # Convert min_area to pixels
        # Calculate areas in one go
        areas = np.bincount(labeled_mask.ravel())[1:]  # Skip background (0)
        valid_labels = np.where(areas >= min_pixels)[0] + 1  # +1 because labels start at 1
        
        # Create mapping array
        label_map = np.zeros(num_features + 1, dtype=int)
        label_map[valid_labels] = np.arange(1, len(valid_labels) + 1)
        
        # Relabel the mask
        labeled_mask = label_map[labeled_mask]
        areas_dict = {i: areas[i-1] * pixel_area for i in range(1, len(valid_labels) + 1)}
    else:
        areas = np.bincount(labeled_mask.ravel())[1:]  # Skip background (0)
        areas_dict = {i: areas[i-1] * pixel_area for i in range(1, num_features + 1)}
    
    return labeled_mask, areas_dict