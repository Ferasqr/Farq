# Farq Documentation

## Overview
Farq (فَرْق) is a Python library for raster change detection and analysis, specializing in water body detection and monitoring using satellite imagery.

## Installation

```bash
pip install farq
```

## Quick Start

```python
import farq

# Load raster bands
green, _ = farq.read("landsat_green.tif")
nir, _ = farq.read("landsat_nir.tif")

# Calculate NDWI
ndwi = farq.ndwi(green, nir)

# Create water mask
water_mask = ndwi > 0

# Get water statistics
stats = farq.water_stats(water_mask, pixel_size=30.0)
print(f"Total water area: {stats['total_area']:.2f} km²")
```

## Core Features

### Spectral Indices
- `ndwi(green, nir)`: Normalized Difference Water Index
- `ndvi(nir, red)`: Normalized Difference Vegetation Index
- `mndwi(green, swir)`: Modified NDWI
- `ndbi(swir, nir)`: Normalized Difference Built-up Index
- `savi(nir, red, L=0.5)`: Soil Adjusted Vegetation Index
- `evi(nir, red, blue)`: Enhanced Vegetation Index

### Water Analysis
- `water_stats()`: Calculate water surface statistics
- `water_change()`: Analyze changes between two time periods
- `get_water_bodies()`: Identify and analyze individual water bodies

### Visualization
- `plot()`: Single raster visualization
- `compare()`: Side-by-side comparison
- `changes()`: Change detection visualization
- `hist()`: Distribution histogram

## Usage Examples

### Water Body Detection
```python
import farq

# Load Landsat 8 bands
green, _ = farq.read("LC08_B3.tif")  # Green band
nir, _ = farq.read("LC08_B5.tif")    # NIR band

# Calculate NDWI
ndwi = farq.ndwi(green, nir)

# Create water mask
water_mask = ndwi > 0

# Analyze water bodies
stats = farq.water_stats(water_mask, pixel_size=30.0)
print(f"Water coverage: {stats['coverage_percent']:.1f}%")

# Visualize results
farq.plot(ndwi, cmap="RdYlBu", title="NDWI")
farq.plt.show()
```

### Change Detection
```python
# Load data from two time periods
mask1 = ndwi1 > 0
mask2 = ndwi2 > 0

# Analyze changes
changes = farq.water_change(mask1, mask2, pixel_size=30.0)
print(f"Net change: {changes['net_change']:.2f} km²")

# Visualize changes
farq.compare(mask1, mask2, 
            title1="Before", 
            title2="After",
            cmap="Blues")
farq.plt.show()
```

## Performance Considerations
- Optimized for large raster datasets
- Memory-efficient operations
- Parallel processing capabilities
- Benchmark results available in testing documentation

## API Reference

### Core Functions
- `read(filepath)`: Load raster data and metadata
- `resample(raster, shape)`: Resample raster to new dimensions

### Spectral Indices
Detailed parameters and return values for each spectral index function.

### Analysis Functions
Complete description of water analysis functionality.

### Visualization Functions
All available plotting and visualization options.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Testing
See [testing.md](testing.md) for testing documentation.

## License
MIT License - see LICENSE file for details. 