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
           figsize: Tuple[int, int] = (15, 6)) -> None:
    """
    Compare two rasters side by side.
    """
    # Calculate common min and max for consistent scaling
    vmin = min(np.min(raster1), np.min(raster2))
    vmax = max(np.max(raster1), np.max(raster2))
    
    plt.figure(figsize=figsize)
    
    # Plot first raster
    plt.subplot(121)
    im1 = plt.imshow(raster1, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im1, label='Value')
    plt.title(title1)
    plt.axis('off')
    
    # Plot second raster
    plt.subplot(122)
    im2 = plt.imshow(raster2, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im2, label='Value')
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

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
        vmin = -abs_max
        vmax = abs_max
    
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