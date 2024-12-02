"""
Farq - A Python library for raster change detection and analysis.
"""
import numpy as _np
import matplotlib.pyplot as plt
import rasterio
from scipy import stats
import os
from typing import Tuple, Dict, Union, Optional
from rasterio.enums import Resampling

# Import utility functions
from .utils import (
    mean,
    std,
    min,
    max,
    sum,
    percentile,
    median,
    count_nonzero,
    unique
)

# Import core functionality
from .core import (
    read,
    diff,
    stats,
    # Spectral indices
    ndvi,
    ndwi,
    mndwi,
    ndbi,
    savi,
    evi,
    resample
)

# Import visualization functions
from .visualization import (
    plot,
    compare,
    changes,
    hist,
    distribution_comparison
)

# Import analysis functions
from .analysis import (
    water_stats,
    water_change,
    get_water_bodies
)

# Make commonly used functions and modules available at package level
__version__ = "0.1.0"

# Export all necessary functions and objects
__all__ = [
    # Core functions
    'read',
    'diff',
    'stats',
    'resample',
    
    # Spectral indices
    'ndvi',
    'ndwi',
    'mndwi',
    'ndbi',
    'savi',
    'evi',
    
    # Visualization functions
    'plot',
    'compare',
    'changes',
    'hist',
    'distribution_comparison',
    
    # Analysis functions
    'water_stats',
    'water_change',
    'get_water_bodies',
    
    # Utility functions
    'mean',
    'std',
    'min',
    'max',
    'sum',
    'percentile',
    'median',
    'count_nonzero',
    'unique',
    
    # Common libraries
    'plt',
    'os',
    'Resampling'
] 