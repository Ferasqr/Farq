"""
Visualization functions for raster data analysis.
Provides easy-to-use functions for plotting rasters, comparisons, and distributions.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Union, List
from . import min, max

def plot(data: np.ndarray, 
         title: str = None,
         cmap: str = "viridis",
         figsize: Tuple[int, int] = (10, 8),
         vmin: Optional[float] = None,
         vmax: Optional[float] = None,
         colorbar_label: str = None) -> plt.Figure:
    """
    Plot a single raster or array.
    
    Args:
        data: 2D array to plot
        title: Plot title (optional)
        cmap: Colormap name (default: "viridis")
        figsize: Figure size as (width, height)
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        colorbar_label: Label for the colorbar (optional)
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if data.size == 0:
        raise ValueError("Input array is empty")
    if data.ndim != 2:
        raise ValueError(f"Input must be a 2D array, got shape {data.shape}")
    
    # Create new figure
    fig = plt.figure(figsize=figsize)
    
    # Plot data
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    if colorbar_label:
        plt.colorbar(im, label=colorbar_label)
    else:
        plt.colorbar(im)
    
    # Add title if provided
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def compare(data1: np.ndarray, 
            data2: np.ndarray,
            title1: str = None,
            title2: str = None,
            cmap: str = "viridis",
            figsize: Tuple[int, int] = (15, 6),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            colorbar_label: str = None) -> plt.Figure:
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
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Convert inputs to numpy arrays
    if not isinstance(data1, np.ndarray):
        data1 = np.asarray(data1)
    if not isinstance(data2, np.ndarray):
        data2 = np.asarray(data2)
    
    # Validate inputs
    if data1.size == 0 or data2.size == 0:
        raise ValueError("Input arrays are empty")
    if data1.ndim != 2 or data2.ndim != 2:
        raise ValueError(f"Inputs must be 2D arrays, got shapes {data1.shape} and {data2.shape}")
    if data1.shape != data2.shape:
        raise ValueError(f"Input arrays must have the same shape: {data1.shape} != {data2.shape}")
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate vmin/vmax if not provided
    if vmin is None or vmax is None:
        vmin = min([min(data1), min(data2)]) if vmin is None else vmin
        vmax = max([max(data1), max(data2)]) if vmax is None else vmax
    
    # Plot first array
    im1 = ax1.imshow(data1, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar_label:
        plt.colorbar(im1, ax=ax1, label=colorbar_label)
    else:
        plt.colorbar(im1, ax=ax1)
    if title1:
        ax1.set_title(title1)
    ax1.axis('off')
    
    # Plot second array
    im2 = ax2.imshow(data2, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar_label:
        plt.colorbar(im2, ax=ax2, label=colorbar_label)
    else:
        plt.colorbar(im2, ax=ax2)
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
           colorbar_label: str = "Change") -> plt.Figure:
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
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if data.size == 0:
        raise ValueError("Input array is empty")
    if data.ndim != 2:
        raise ValueError(f"Input must be a 2D array, got shape {data.shape}")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate symmetric scaling if needed
    if symmetric and (vmin is None or vmax is None):
        abs_max = max(np.abs(data))
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    
    # Plot data
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def hist(data: Union[np.ndarray, List],
         bins: int = 50,
         title: str = None,
         figsize: Tuple[int, int] = (10, 6),
         density: bool = True,
         xlabel: str = "Value",
         ylabel: str = None,
         alpha: float = 0.6) -> plt.Figure:
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
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Convert input to numpy array
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if data.size == 0:
        raise ValueError("Input array is empty")
    
    # Flatten array if multidimensional
    data = data.ravel()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot histogram
    plt.hist(data, bins=bins, density=density, alpha=alpha)
    
    # Add labels and title
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else ('Density' if density else 'Count'))
    
    plt.grid(True, alpha=0.3)
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
                          alpha: float = 0.6) -> plt.Figure:
    """
    Compare distributions of two datasets side by side.
    
    Args:
        data1: First dataset (numpy array or list)
        data2: Second dataset (numpy array or list)
        title1: Title for first histogram (optional)
        title2: Title for second histogram (optional)
        bins: Number of histogram bins
        figsize: Figure size as (width, height)
        density: If True, plot density instead of counts
        xlabel: Label for x-axis
        ylabel: Label for y-axis (defaults to "Density" or "Count")
        alpha: Transparency of the histogram bars
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Convert inputs to numpy arrays
    if not isinstance(data1, np.ndarray):
        data1 = np.asarray(data1)
    if not isinstance(data2, np.ndarray):
        data2 = np.asarray(data2)
    
    if data1.size == 0 or data2.size == 0:
        raise ValueError("Input arrays are empty")
    
    # Flatten arrays if multidimensional
    data1 = data1.ravel()
    data2 = data2.ravel()
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot first distribution
    ax1.hist(data1, bins=bins, density=density, alpha=alpha)
    if title1:
        ax1.set_title(title1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel if ylabel else ('Density' if density else 'Count'))
    ax1.grid(True, alpha=0.3)
    
    # Plot second distribution
    ax2.hist(data2, bins=bins, density=density, alpha=alpha)
    if title2:
        ax2.set_title(title2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel if ylabel else ('Density' if density else 'Count'))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig