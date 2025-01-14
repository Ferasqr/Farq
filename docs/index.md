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
- Feature extraction for ML

### Machine Learning
- Supervised classification for water detection
- Unsupervised clustering (K-means, DBSCAN)
- ML-based change detection
- Model training and persistence
- Parameter optimization
- Feature extraction and analysis

### Spectral Indices
- NDWI (Normalized Difference Water Index)
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil Adjusted Vegetation Index)
- NBR (Normalized Burn Ratio)
- NDMI (Normalized Difference Moisture Index)

### Visualization
- Single raster visualization
- Side-by-side comparison
- Change detection visualization
- Distribution analysis
- RGB composite visualization
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

### Machine Learning Water Detection
```python
import farq

# Load and preprocess data
nir, _ = farq.read("landsat_nir.tif")
green, _ = farq.read("landsat_green.tif")

# Calculate water index for reference
ndwi = farq.ndwi(green, nir)

# Detect water bodies using clustering
labels, metadata = farq.cluster_water_bodies(
    nir,
    method='kmeans',
    n_clusters=2,
    water_index=ndwi
)

# Analyze results
stats = farq.analyze_water_clusters(labels, metadata['water_cluster'])
print(f"Number of water bodies: {stats['num_water_bodies']}")
print(f"Total water area: {stats['total_water_area']:.2f} km²")

# Visualize results
farq.plot(labels == metadata['water_cluster'], 
         title="Detected Water Bodies",
         cmap="Blues")
farq.plt.show()
```

### ML-based Change Detection
```python
# Load data from two periods
nir_2020, _ = farq.read("nir_2020.tif")
nir_2024, _ = farq.read("nir_2024.tif")

# Detect changes using ML
changes = farq.detect_changes_ml(nir_2020, nir_2024, threshold=0.5)

# Visualize changes
farq.changes(changes, 
    title="ML-detected Changes",
    cmap="RdYlBu",
    symmetric=True)
farq.plt.show()
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