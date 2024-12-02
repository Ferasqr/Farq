import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

def plot(raster: np.ndarray, 
        title: str = "Raster Data",
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 8),
        vmin: Optional[float] = None,
        vmax: Optional[float] = None) -> None:
    """
    Plot a single raster.
    """
    if raster.size == 0 or raster.ndim != 2:
        raise ValueError("Input must be a non-empty 2D array")
        
    plt.figure(figsize=figsize)
    im = plt.imshow(raster, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Value')
    plt.title(title)
    plt.axis('off')
    return plt.gcf()

def compare(raster1: np.ndarray, 
           raster2: np.ndarray,
           title1: str = "First",
           title2: str = "Second",
           cmap: str = "viridis",
           figsize: Tuple[int, int] = (15, 6),
           vmin: Optional[float] = None,
           vmax: Optional[float] = None) -> None:
    """
    Compare two rasters side by side.
    """
    if raster1.shape != raster2.shape:
        raise ValueError("Input rasters must have the same shape")
    if raster1.size == 0 or raster1.ndim != 2:
        raise ValueError("Inputs must be non-empty 2D arrays")
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # If vmin/vmax not provided, calculate from data
    if vmin is None or vmax is None:
        vmin = min(np.nanmin(raster1), np.nanmin(raster2)) if vmin is None else vmin
        vmax = max(np.nanmax(raster1), np.nanmax(raster2)) if vmax is None else vmax
    
    # Plot first raster
    im1 = ax1.imshow(raster1, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=ax1, label='Value')
    ax1.set_title(title1)
    ax1.axis('off')
    
    # Plot second raster
    im2 = ax2.imshow(raster2, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=ax2, label='Value')
    ax2.set_title(title2)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def changes(data: np.ndarray, 
           title: str = "Changes",
           cmap: str = "RdYlBu",
           figsize: Tuple[int, int] = (10, 8),
           vmin: Optional[float] = None,
           vmax: Optional[float] = None) -> None:
    """
    Plot change detection results.
    """
    plt.figure(figsize=figsize)
    
    # If vmin/vmax not provided, use symmetric scaling around zero
    if vmin is None or vmax is None:
        abs_max = np.max(np.abs(data))
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Change')
    plt.title(title)
    plt.axis('off')
    
    return plt.gcf()

def hist(data: np.ndarray,
         bins: int = 50,
         title: str = "Distribution",
         figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot histogram of values.
    """
    plt.figure(figsize=figsize)
    plt.hist(data.ravel(), bins=bins, density=True)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    return plt.gcf() 