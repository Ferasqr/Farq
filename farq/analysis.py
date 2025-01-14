"""
Analysis functions for water detection and change analysis.

This module provides functions for:
- Water body statistics calculation
- Change detection between water masks
- Individual water body identification and analysis
- Water body shape and morphology analysis

All functions include input validation and detailed error messages.
"""
import numpy as np
from typing import Dict, Union, Tuple, Optional, List
from scipy import ndimage
from .utils import sum

def calculate_shape_metrics(water_body_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate shape metrics for a single water body.
    
    Args:
        water_body_mask: Binary mask of a single water body
        
    Returns:
        Dictionary containing shape metrics:
        - perimeter: Length of the boundary in pixels
        - compactness: Ratio of area to perimeter squared
        - elongation: Ratio of major to minor axis length
        - orientation: Angle of major axis in degrees
    """
    # Calculate perimeter using gradient
    gradient_x = np.gradient(water_body_mask.astype(float), axis=0)
    gradient_y = np.gradient(water_body_mask.astype(float), axis=1)
    perimeter = np.sum(np.sqrt(gradient_x**2 + gradient_y**2))
    
    # Calculate area
    area = np.sum(water_body_mask)
    
    # Calculate compactness (normalized to [0,1])
    compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
    
    # Calculate moments for elongation and orientation
    y, x = np.nonzero(water_body_mask)
    if len(x) > 0 and len(y) > 0:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        u20 = np.sum((x - x_mean)**2) / area
        u02 = np.sum((y - y_mean)**2) / area
        u11 = np.sum((x - x_mean) * (y - y_mean)) / area
        
        # Calculate elongation using eigenvalues
        tmp = np.sqrt((u20 - u02)**2 + 4*u11**2)
        lambda1 = (u20 + u02 + tmp) / 2
        lambda2 = (u20 + u02 - tmp) / 2
        elongation = np.sqrt(lambda1 / lambda2) if lambda2 > 0 else 1
        
        # Calculate orientation in degrees
        orientation = np.degrees(0.5 * np.arctan2(2*u11, u20 - u02))
    else:
        elongation = 1
        orientation = 0
    
    return {
        'perimeter': float(perimeter),
        'compactness': float(compactness),
        'elongation': float(elongation),
        'orientation': float(orientation)
    }

def water_stats(water_mask: np.ndarray, 
               pixel_size: Union[float, Tuple[float, float]] = 30.0,
               calculate_shapes: bool = False) -> Dict[str, Union[float, Dict]]:
    """
    Calculate comprehensive water surface statistics from a water mask.
    
    Args:
        water_mask: Binary water mask (True/1 for water, False/0 for non-water)
        pixel_size: Pixel size in meters. Default 30.0 (Landsat resolution)
            Can be a single float for square pixels or a tuple (width, height)
        calculate_shapes: Whether to calculate shape metrics for water bodies
        
    Returns:
        Dict with statistics:
            - total_area: Total water surface area in square kilometers
            - coverage_percent: Percentage of area covered by water
            - num_water_bodies: Number of distinct water bodies
            - mean_body_size: Average water body size in square kilometers
            - largest_body: Size of largest water body in square kilometers
            - shape_metrics: Dictionary of shape metrics (if calculate_shapes=True)
            
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
    
    # Calculate basic statistics
    total_pixels = np.sum(mask)
    total_area = total_pixels * pixel_area
    coverage = (total_pixels / mask.size) * 100
    
    # Calculate water body sizes
    if num_features > 0:
        body_sizes = np.bincount(labeled_mask[labeled_mask > 0]) * pixel_area
        mean_body_size = np.mean(body_sizes)
        largest_body = np.max(body_sizes)
    else:
        mean_body_size = 0
        largest_body = 0
    
    stats_dict = {
        "total_area": total_area,  # km²
        "coverage_percent": coverage,  # %
        "num_water_bodies": num_features,
        "mean_body_size": mean_body_size,  # km²
        "largest_body": largest_body  # km²
    }
    
    # Calculate shape metrics if requested
    if calculate_shapes and num_features > 0:
        shape_metrics = []
        for i in range(1, num_features + 1):
            body_mask = labeled_mask == i
            metrics = calculate_shape_metrics(body_mask)
            shape_metrics.append(metrics)
        
        # Add summary of shape metrics
        stats_dict["shape_metrics"] = {
            "mean_compactness": np.mean([m["compactness"] for m in shape_metrics]),
            "mean_elongation": np.mean([m["elongation"] for m in shape_metrics]),
            "body_metrics": shape_metrics
        }
    
    return stats_dict

def water_change(mask1: np.ndarray, 
                mask2: np.ndarray,
                pixel_size: Union[float, Tuple[float, float]] = 30.0,
                min_change_area: Optional[float] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Analyze changes between two water masks.
    
    Args:
        mask1: First water mask (True/1 for water)
        mask2: Second water mask (True/1 for water)
        pixel_size: Pixel size in meters
            Can be a single float for square pixels or a tuple (width, height)
        min_change_area: Minimum area (in square meters) to consider as change
            Smaller changes will be filtered out
        
    Returns:
        Dict with change statistics:
            - gained_area: New water area in square kilometers
            - lost_area: Lost water area in square kilometers
            - net_change: Net change in water area in square kilometers
            - change_percent: Percentage change relative to original area
            - change_mask: Array showing changes (1: gained, -1: lost, 0: no change)
            - stable_water: Array showing stable water bodies
            
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
    stable = np.logical_and(mask1, mask2)
    
    # Filter small changes if requested
    if min_change_area is not None:
        min_pixels = min_change_area / (pixel_area * 1_000_000)
        gained = ndimage.binary_opening(gained, structure=np.ones((3,3)), iterations=int(min_pixels**0.5))
        lost = ndimage.binary_opening(lost, structure=np.ones((3,3)), iterations=int(min_pixels**0.5))
    
    # Calculate areas
    gained_area = np.sum(gained) * pixel_area
    lost_area = np.sum(lost) * pixel_area
    net_change = gained_area - lost_area
    
    # Calculate percentage change
    original_area = np.sum(mask1) * pixel_area
    if original_area > 0:
        change_percent = (net_change / original_area) * 100
    else:
        change_percent = float('inf') if gained_area > 0 else 0
    
    # Create change mask (-1: lost, 0: no change, 1: gained)
    change_mask = gained.astype(int) - lost.astype(int)
    
    return {
        "gained_area": gained_area,  # km²
        "lost_area": lost_area,  # km²
        "net_change": net_change,  # km²
        "change_percent": change_percent,  # %
        "change_mask": change_mask,
        "stable_water": stable
    }

def get_water_bodies(water_mask: np.ndarray,
                    pixel_size: Union[float, Tuple[float, float]] = 30.0,
                    min_area: Optional[float] = None,
                    calculate_shapes: bool = False) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """
    Label individual water bodies and calculate their characteristics.
    
    Args:
        water_mask: Binary water mask
        pixel_size: Pixel size in meters
            Can be a single float for square pixels or a tuple (width, height)
        min_area: Minimum water body area in square meters (optional)
            Water bodies smaller than this will be filtered out
        calculate_shapes: Whether to calculate shape metrics for each water body
        
    Returns:
        Tuple containing:
            - Labeled array where each water body has a unique integer ID
            - Dictionary mapping water body IDs to their characteristics:
                - area: Area in square kilometers
                - perimeter: Perimeter length (if calculate_shapes=True)
                - compactness: Shape compactness (if calculate_shapes=True)
                - elongation: Shape elongation (if calculate_shapes=True)
                - orientation: Major axis orientation (if calculate_shapes=True)
            
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
        num_features = len(valid_labels)
    
    # Calculate characteristics for each water body
    characteristics = {}
    for i in range(1, num_features + 1):
        body_mask = labeled_mask == i
        area = np.sum(body_mask) * pixel_area
        
        body_stats = {"area": area}
        
        if calculate_shapes:
            shape_metrics = calculate_shape_metrics(body_mask)
            body_stats.update(shape_metrics)
        
        characteristics[i] = body_stats
    
    return labeled_mask, characteristics