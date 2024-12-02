# Farq - فَرْق

A Python library for raster change detection and analysis, specializing in water body detection and monitoring using satellite imagery. Farq (Arabic: فَرْق, meaning "difference") simplifies the process of identifying and analyzing changes between raster datasets over time, with a focus on remote sensing applications.

## Features

### Core Functions
- Efficient raster data loading and handling
- Change detection using multiple methods
- Statistical analysis tools
- Raster resampling and preprocessing

### Water Analysis
- Water body detection and delineation
- Surface area calculations
- Temporal change analysis
- Individual water body statistics
- Performance-optimized for large datasets

### Spectral Indices
- **Water Indices:**
  - NDWI (Normalized Difference Water Index)
  - MNDWI (Modified Normalized Difference Water Index)
- **Vegetation Indices:**
  - NDVI (Normalized Difference Vegetation Index)
  - SAVI (Soil Adjusted Vegetation Index)
  - EVI (Enhanced Vegetation Index)
- **Urban Indices:**
  - NDBI (Normalized Difference Built-up Index)

### Visualization Tools
- Single raster visualization
- Side-by-side raster comparison
- Change detection visualization
- Distribution histograms
- Customizable colormaps and scaling

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
print(f"Water coverage: {stats['coverage_percent']:.1f}%")

# Visualize results
farq.plot(ndwi, title="NDWI", cmap="RdYlBu")
farq.plt.show()
```

## Common Operations

### Water Analysis
```python
# Detect water bodies
water_mask = ndwi > 0
stats = farq.water_stats(water_mask, pixel_size=30.0)

# Analyze changes between two periods
changes = farq.water_change(mask1, mask2, pixel_size=30.0)
print(f"Net change: {changes['net_change']:.2f} km²")

# Identify individual water bodies
labeled, areas = farq.get_water_bodies(water_mask, min_area=9000)
```

### Spectral Indices
```python
# Water indices
ndwi = farq.ndwi(green, nir)
mndwi = farq.mndwi(green, swir)

# Vegetation indices
ndvi = farq.ndvi(nir, red)
savi = farq.savi(nir, red, L=0.5)
evi = farq.evi(nir, red, blue)

# Built-up index
ndbi = farq.ndbi(swir, nir)
```

### Visualization
```python
# Single raster
farq.plot(ndwi, cmap="RdYlBu")

# Compare two time periods
farq.compare(mask1, mask2, title1="Before", title2="After")

# Plot changes
farq.changes(change_data, cmap="RdYlBu")

# Show distribution
farq.hist(ndwi, bins=50)
```

## Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

- [Getting Started](docs/index.md) - Overview and installation
- [API Reference](docs/api.md) - Detailed function documentation
- [Examples](docs/examples.md) - Code examples and use cases
- [Testing](docs/testing.md) - Testing documentation and benchmarks

## Performance

Farq is optimized for large raster datasets with:
- Memory-efficient operations
- Parallel processing capabilities
- Vectorized computations
- Benchmark results available in testing documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.