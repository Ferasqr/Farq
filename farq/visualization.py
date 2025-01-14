"""
Visualization functions for raster data analysis.

This module provides functions for:
- Single raster visualization
- Side-by-side comparisons
- Change detection visualization
- Distribution analysis
- RGB composite visualization

All functions include input validation and detailed error messages.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Union, List
from .utils import min, max

def plot(data: np.ndarray, 
         title: str = None,
         cmap: str = "viridis",
         figsize: Tuple[int, int] = (10, 8),
         vmin: Optional[float] = None,
         vmax: Optional[float] = None,
         colorbar_label: str = None,
         reflectance_scale: Optional[float] = None) -> plt.Figure:
    """
    Plot a single raster or array.
    """
    # Close any existing figures to prevent multiple plots
    plt.close('all')
    
    # Input validation
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Input must be a numpy array, got {type(data)}")
    
    if data.size == 0:
        raise ValueError("Input array is empty")
    if data.ndim != 2:
        raise ValueError(f"Input must be a 2D array, got shape {data.shape}")
    
    # Apply reflectance scaling if provided
    plot_data = data.copy()
    if reflectance_scale is not None:
        plot_data = plot_data / reflectance_scale
    
    # Create new figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    if colorbar_label:
        fig.colorbar(im, ax=ax, label=colorbar_label)
    else:
        fig.colorbar(im, ax=ax)
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    ax.axis('off')
    fig.tight_layout()
    
    return fig

def compare(data1: np.ndarray, 
            data2: np.ndarray,
            title1: str = None,
            title2: str = None,
            cmap: str = "viridis",
            figsize: Tuple[int, int] = (15, 6),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            colorbar_label: str = None,
            reflectance_scale: Optional[float] = None) -> plt.Figure:
    """
    Compare two rasters or arrays side by side.
    
    Args:
        data1: First 2D array
        data2: Second 2D array
        title1: Title for first plot (optional)
        title2: Title for second plot (optional)
        cmap: Colormap name (default: "viridis")
        figsize: Figure size as (width, height)
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        colorbar_label: Label for both colorbars (optional)
        reflectance_scale: Scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Convert inputs to numpy arrays
    if not isinstance(data1, np.ndarray):
        raise TypeError(f"First input must be a numpy array, got {type(data1)}")
    if not isinstance(data2, np.ndarray):
        raise TypeError(f"Second input must be a numpy array, got {type(data2)}")
    
    # Validate inputs
    if data1.size == 0 or data2.size == 0:
        raise ValueError("Input arrays are empty")
    if data1.ndim != 2 or data2.ndim != 2:
        raise ValueError(f"Inputs must be 2D arrays, got shapes {data1.shape} and {data2.shape}")
    if data1.shape != data2.shape:
        raise ValueError(f"Input arrays must have the same shape: {data1.shape} != {data2.shape}")
    
    # Apply reflectance scaling if provided
    plot_data1 = data1.copy()
    plot_data2 = data2.copy()
    if reflectance_scale is not None:
        plot_data1 = plot_data1 / reflectance_scale
        plot_data2 = plot_data2 / reflectance_scale
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate vmin/vmax if not provided
    if vmin is None or vmax is None:
        vmin = min(plot_data1) if vmin is None else vmin
        vmax = max(plot_data1) if vmax is None else vmax
    
    # Plot first array
    im1 = ax1.imshow(plot_data1, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar_label:
        fig.colorbar(im1, ax=ax1, label=colorbar_label)
    else:
        fig.colorbar(im1, ax=ax1)
    if title1:
        ax1.set_title(title1)
    ax1.axis('off')
    
    # Plot second array
    im2 = ax2.imshow(plot_data2, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar_label:
        fig.colorbar(im2, ax=ax2, label=colorbar_label)
    else:
        fig.colorbar(im2, ax=ax2)
    if title2:
        ax2.set_title(title2)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def changes(data: np.ndarray, 
           title: str = None,
           cmap: str = "RdYlBu",
           figsize: Tuple[int, int] = (10, 8),
           vmin: Optional[float] = None,
           vmax: Optional[float] = None,
           symmetric: bool = True,
           colorbar_label: str = "Change",
           reflectance_scale: Optional[float] = None) -> plt.Figure:
    """
    Plot change detection results with optional symmetric scaling.
    
    Args:
        data: Change detection array
        title: Plot title (optional)
        cmap: Colormap name (default: "RdYlBu")
        figsize: Figure size as (width, height)
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        symmetric: If True, use symmetric scaling around zero
        colorbar_label: Label for the colorbar
        reflectance_scale: Scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
    
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or not 2D
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Input must be a numpy array, got {type(data)}")
    
    if data.size == 0:
        raise ValueError("Input array is empty")
    if data.ndim != 2:
        raise ValueError(f"Input must be a 2D array, got shape {data.shape}")
    
    # Apply reflectance scaling if provided
    plot_data = data.copy()
    if reflectance_scale is not None:
        plot_data = plot_data / reflectance_scale
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate symmetric scaling if needed
    if symmetric and (vmin is None or vmax is None):
        abs_max = max(np.abs(plot_data))
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    
    # Plot data
    im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)
    
    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def hist(data: Union[np.ndarray, List],
         bins: int = 50,
         title: str = None,
         figsize: Tuple[int, int] = (10, 6),
         density: bool = True,
         xlabel: str = "Value",
         ylabel: str = None,
         alpha: float = 0.6,
         reflectance_scale: Optional[float] = None) -> plt.Figure:
    """
    Plot histogram of values with customizable labels.
    
    Args:
        data: Input data (numpy array or list)
        bins: Number of histogram bins
        title: Plot title (optional)
        figsize: Figure size as (width, height)
        density: If True, plot density instead of counts
        xlabel: Label for x-axis
        ylabel: Label for y-axis (defaults to "Density" or "Count")
        alpha: Transparency of the histogram bars
        reflectance_scale: Scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
    
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Raises:
        ValueError: If input array is empty
    """
    # Convert input to numpy array
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if data.size == 0:
        raise ValueError("Input array is empty")
    
    # Apply reflectance scaling if provided
    plot_data = data.copy()
    if reflectance_scale is not None:
        plot_data = plot_data / reflectance_scale
    
    # Flatten array if multidimensional
    plot_data = plot_data.ravel()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(plot_data, bins=bins, density=density, alpha=alpha)
    
    # Add labels and title
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else ('Density' if density else 'Count'))
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def distribution_comparison(data1: Union[np.ndarray, List],
                          data2: Union[np.ndarray, List],
                          title1: str = None,
                          title2: str = None,
                          bins: int = 50,
                          figsize: Tuple[int, int] = (12, 6),
                          density: bool = True,
                          xlabel: str = "Value",
                          ylabel: str = None,
                          alpha: float = 0.6,
                          reflectance_scale: Optional[float] = None) -> plt.Figure:
    """
    Compare distributions of two datasets side by side.
    
    Args:
        data1: First dataset (numpy array or list)
        data2: Second dataset (numpy array or list)
        title1: Title for first plot (optional)
        title2: Title for second plot (optional)
        bins: Number of histogram bins
        figsize: Figure size as (width, height)
        density: If True, plot density instead of counts
        xlabel: Label for x-axis
        ylabel: Label for y-axis (defaults to "Density" or "Count")
        alpha: Transparency of the histogram bars
        reflectance_scale: Scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
    
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Raises:
        ValueError: If input arrays are empty
    """
    # Convert inputs to numpy arrays
    if not isinstance(data1, np.ndarray):
        data1 = np.asarray(data1)
    if not isinstance(data2, np.ndarray):
        data2 = np.asarray(data2)
    
    if data1.size == 0 or data2.size == 0:
        raise ValueError("Input arrays are empty")
    
    # Apply reflectance scaling if provided
    plot_data1 = data1.copy()
    plot_data2 = data2.copy()
    if reflectance_scale is not None:
        plot_data1 = plot_data1 / reflectance_scale
        plot_data2 = plot_data2 / reflectance_scale
    
    # Flatten arrays if multidimensional
    plot_data1 = plot_data1.ravel()
    plot_data2 = plot_data2.ravel()
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot first histogram
    ax1.hist(plot_data1, bins=bins, density=density, alpha=alpha)
    if title1:
        ax1.set_title(title1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel if ylabel else ('Density' if density else 'Count'))
    ax1.grid(True, alpha=0.3)
    
    # Plot second histogram
    ax2.hist(plot_data2, bins=bins, density=density, alpha=alpha)
    if title2:
        ax2.set_title(title2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel if ylabel else ('Density' if density else 'Count'))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_rgb(red: np.ndarray,
            green: np.ndarray,
            blue: np.ndarray,
            title: str = None,
            figsize: Tuple[int, int] = (10, 8),
            scale_factor: float = 1.0,
            gamma: float = 1.0,
            percentile: float = 98.0,
            reflectance_scale: Optional[float] = None) -> plt.Figure:
    """
    Plot RGB composite from three bands with enhancement options.
    
    Args:
        red: Red band array
        green: Green band array
        blue: Blue band array
        title: Plot title (optional)
        figsize: Figure size as (width, height)
        scale_factor: Scale factor for brightness adjustment
        gamma: Gamma correction value
        percentile: Percentile for contrast enhancement
        reflectance_scale: Scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
        
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If arrays have different shapes or are empty
    """
    # Input validation
    if not all(isinstance(x, np.ndarray) for x in [red, green, blue]):
        raise TypeError("All inputs must be numpy arrays")
    if not (red.shape == green.shape == blue.shape):
        raise ValueError("All bands must have the same shape")
    if red.size == 0:
        raise ValueError("Input arrays are empty")
    
    # Stack bands and convert to float
    rgb = np.dstack((red, green, blue)).astype(float)
    
    # Apply reflectance scaling if provided
    if reflectance_scale is not None:
        rgb = rgb / reflectance_scale
    
    # Apply scale factor
    rgb *= scale_factor
    
    # Enhance contrast using percentile
    for i in range(3):
        p = np.percentile(rgb[..., i], percentile)
        rgb[..., i] = np.clip(rgb[..., i] / p, 0, 1)
    
    # Apply gamma correction
    if gamma != 1.0:
        rgb = np.power(rgb, 1/gamma)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot RGB composite
    ax.imshow(rgb)
    
    if title:
        ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    
    return fig

def compare_rgb(rgb1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                rgb2: Tuple[np.ndarray, np.ndarray, np.ndarray],
                title1: str = None,
                title2: str = None,
                figsize: Tuple[int, int] = (15, 6),
                scale_factor: float = 1.0,
                gamma: float = 1.0,
                percentile: float = 98.0,
                reflectance_scale: Optional[float] = None) -> plt.Figure:
    """
    Compare two RGB composites side by side.
    
    Args:
        rgb1: Tuple of (red, green, blue) arrays for first image
        rgb2: Tuple of (red, green, blue) arrays for second image
        title1: Title for first plot (optional)
        title2: Title for second plot (optional)
        figsize: Figure size as (width, height)
        scale_factor: Scale factor for brightness adjustment
        gamma: Gamma correction value
        percentile: Percentile for contrast enhancement
        reflectance_scale: Scale factor for reflectance data (e.g., 10000 for Landsat 8 SR)
        
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If arrays have different shapes or are empty
    """
    # Input validation
    for bands in [rgb1, rgb2]:
        if not all(isinstance(x, np.ndarray) for x in bands):
            raise TypeError("All inputs must be numpy arrays")
        if not (bands[0].shape == bands[1].shape == bands[2].shape):
            raise ValueError("All bands must have the same shape")
        if bands[0].size == 0:
            raise ValueError("Input arrays are empty")
    if rgb1[0].shape != rgb2[0].shape:
        raise ValueError("Both images must have the same shape")
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Process and plot first image
    rgb_stack1 = np.dstack(rgb1).astype(float)
    
    # Apply reflectance scaling if provided
    if reflectance_scale is not None:
        rgb_stack1 = rgb_stack1 / reflectance_scale
    
    # Apply scale factor and enhancements
    rgb_stack1 *= scale_factor
    for i in range(3):
        p = np.percentile(rgb_stack1[..., i], percentile)
        rgb_stack1[..., i] = np.clip(rgb_stack1[..., i] / p, 0, 1)
    if gamma != 1.0:
        rgb_stack1 = np.power(rgb_stack1, 1/gamma)
    ax1.imshow(rgb_stack1)
    if title1:
        ax1.set_title(title1)
    ax1.axis('off')
    
    # Process and plot second image
    rgb_stack2 = np.dstack(rgb2).astype(float)
    
    # Apply reflectance scaling if provided
    if reflectance_scale is not None:
        rgb_stack2 = rgb_stack2 / reflectance_scale
    
    # Apply scale factor and enhancements
    rgb_stack2 *= scale_factor
    for i in range(3):
        p = np.percentile(rgb_stack2[..., i], percentile)
        rgb_stack2[..., i] = np.clip(rgb_stack2[..., i] / p, 0, 1)
    if gamma != 1.0:
        rgb_stack2 = np.power(rgb_stack2, 1/gamma)
    ax2.imshow(rgb_stack2)
    if title2:
        ax2.set_title(title2)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig