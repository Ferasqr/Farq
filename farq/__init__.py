import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy import stats
import os
from typing import Tuple, Dict, Union, Optional
from rasterio.enums import Resampling

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
from .visualization import (
    plot,
    compare,
    changes,
    hist
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
    
    # Common libraries
    'np',
    'plt',
    'os',
    'Resampling'
] 