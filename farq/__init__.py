"""
Farq - A Python library for raster change detection and analysis.
"""
import numpy as _np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import os
from typing import Tuple, Dict, Union, Optional

# Import core functionality
from .core import (
    read,
    write,
    resample,
    validate_bands
)

# Import utility functions
from .utils import (
    stats,
    min,
    max,
    mean,
    std,
    sum,
    median,
    percentile,
    count_nonzero,
    unique,
    validate_array
)

# Import spectral indices
from .indices import (
    ndwi,
    ndvi,
    evi,
    savi,
    ndbi,
    nbr,
    ndmi,
    calculate_indices,
    calculate_normalized_difference
)

# Import visualization functions
from .visualization import (
    plot,
    compare,
    changes,
    hist,
    distribution_comparison,
    plot_rgb,
    compare_rgb
)

# Import analysis functions
from .analysis import (
    water_stats,
    water_change,
    get_water_bodies,
    calculate_shape_metrics
)

# Make commonly used functions and modules available at package level
__version__ = "0.1.3"

# Export all necessary functions and objects
__all__ = [
    # Core functions
    'read',
    'write',
    'resample',
    'validate_bands',
    
    # Utility functions
    'stats',
    'min',
    'max',
    'mean',
    'std',
    'sum',
    'median',
    'percentile',
    'count_nonzero',
    'unique',
    'validate_array',
    
    # Spectral indices
    'ndwi',
    'ndvi',
    'evi',
    'savi',
    'ndbi',
    'nbr',
    'ndmi',
    'calculate_indices',
    'calculate_normalized_difference',
    
    # Visualization functions
    'plot',
    'compare',
    'changes',
    'hist',
    'distribution_comparison',
    'plot_rgb',
    'compare_rgb',
    
    # Analysis functions
    'water_stats',
    'water_change',
    'get_water_bodies',
    'calculate_shape_metrics',
    
    # Common libraries
    'plt',
    'os',
    'Resampling'
] 