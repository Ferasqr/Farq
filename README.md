# Farq - فَرْق

A Python library for raster change detection and analysis. Farq simplifies the process of identifying and analyzing changes between raster datasets over time, with a focus on remote sensing applications.

## Features

### Core Functions
- Simple and efficient raster data loading
- Change detection using multiple methods
- Statistical analysis tools
- Raster resampling capabilities

### Spectral Indices
- **Vegetation Indices:**
  - NDVI (Normalized Difference Vegetation Index)
  - SAVI (Soil Adjusted Vegetation Index)
  - EVI (Enhanced Vegetation Index)
- **Water Indices:**
  - NDWI (Normalized Difference Water Index)
  - MNDWI (Modified Normalized Difference Water Index)
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

# Load raster data
data, meta = farq.read("image.tif")

# Calculate indices
ndvi = farq.ndvi(nir_band, red_band)
ndwi = farq.ndwi(green_band, nir_band)
savi = farq.savi(nir_band, red_band, L=0.5)

# Visualize results
farq.plot(ndvi, title="NDVI", cmap="RdYlGn")
farq.plt.show()

# Get statistics
stats = farq.stats(ndvi)
print(f"Mean NDVI: {stats['mean']:.4f}")
```

## Common Operations

### Loading Data
```python
# Load a single band
data, meta = farq.read("band.tif")
```

### Spectral Indices
```python
# Vegetation indices
ndvi = farq.ndvi(nir, red)
savi = farq.savi(nir, red, L=0.5)
evi = farq.evi(nir, red, blue)

# Water indices
ndwi = farq.ndwi(green, nir)
mndwi = farq.mndwi(green, swir)

# Built-up index
ndbi = farq.ndbi(swir, nir)
```

### Change Detection
```python
# Load two time periods
t1, _ = farq.read("time1.tif")
t2, _ = farq.read("time2.tif")

# Detect changes
changes = farq.diff(t1, t2, method="simple")  # or "ratio", "norm"
```

### Visualization
```python
# Single raster
farq.plot(data, cmap="viridis")

# Compare two rasters
farq.compare(raster1, raster2, title1="Before", title2="After")

# Plot changes
farq.changes(diff_data, cmap="RdYlBu")

# Show histogram
farq.hist(data, bins=50)
```

## Documentation

For detailed documentation and examples, visit [docs/](docs/).

## License

This project is licensed under the MIT License - see the LICENSE file for details.