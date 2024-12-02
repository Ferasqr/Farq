# Getting Started with Farq

## Overview

Farq is a Python library designed for raster change detection and analysis, with a focus on water body monitoring using satellite imagery. The name Farq (Arabic: فَرْق) means "difference", reflecting its primary purpose of analyzing changes in raster data over time.

## Installation

```bash
pip install farq
```

## Basic Usage

Here's a simple example of water detection using NDWI:

```python
import farq

# Load bands
green, meta = farq.read("landsat_green.tif")
nir, _ = farq.read("landsat_nir.tif")

# Calculate NDWI
ndwi = farq.ndwi(green, nir)

# Calculate water coverage
water_mask = ndwi > 0
water_percentage = (farq.sum(water_mask) / water_mask.size) * 100

print(f"Water coverage: {water_percentage:.1f}%")

# Visualize results
farq.plot(ndwi, title="NDWI Analysis", cmap="RdYlBu", vmin=-1, vmax=1)
farq.plt.show()
```

## Core Features

### Data Loading and Preprocessing
- Read raster data from various formats
- Resample rasters to match dimensions
- Basic statistical operations

### Spectral Indices
- NDWI (Normalized Difference Water Index)
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil Adjusted Vegetation Index)

### Visualization
- Single raster visualization
- Side-by-side comparison
- Customizable colormaps and scaling

## Example Applications

### Water Change Detection
```python
# Load data from two periods
green_1, _ = farq.read("green_2020.tif")
nir_1, _ = farq.read("nir_2020.tif")
green_2, _ = farq.read("green_2024.tif")
nir_2, _ = farq.read("nir_2024.tif")

# Calculate NDWI
ndwi_1 = farq.ndwi(green_1, nir_1)
ndwi_2 = farq.ndwi(green_2, nir_2)

# Compare results
farq.compare(ndwi_1, ndwi_2,
    title1="NDWI 2020",
    title2="NDWI 2024",
    cmap="RdYlBu",
    vmin=-1, vmax=1)
farq.plt.show()
```

### Vegetation Analysis
```python
# Calculate vegetation indices
ndvi = farq.ndvi(red, nir)
evi = farq.evi(red, nir, blue)

# Analyze vegetation coverage
veg_mask = ndvi > 0.2
veg_percentage = (farq.sum(veg_mask) / veg_mask.size) * 100

print(f"Vegetation coverage: {veg_percentage:.1f}%")
```

## Performance Considerations

Farq is optimized for:
- Memory-efficient operations
- Vectorized computations
- Large raster datasets
- Parallel processing capabilities

## Next Steps

- Check out the [API Reference](api.md) for detailed function documentation
- See [Examples](examples.md) for more use cases
- Review [Testing](testing.md) for performance information

## Support

For issues and feature requests, please visit the project's GitHub repository. 