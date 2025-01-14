# Farq - فَرْق

A Python library for raster change detection and analysis, specializing in water body detection and monitoring using satellite imagery. Farq (Arabic: فَرْق, meaning "difference") simplifies the process of identifying and analyzing changes between raster datasets over time, with a focus on remote sensing applications.

## Features

### Core Functions
- Efficient raster data loading and handling
- Change detection using multiple methods
- Statistical analysis tools
- Raster resampling and preprocessing
- Memory-efficient operations
- Robust error handling

### Machine Learning
- Image classification and feature extraction
- Unsupervised water body detection using clustering
- Change detection using ML models
- Model training, saving, and loading
- Automated parameter optimization
- Support for multiple clustering algorithms (K-means, DBSCAN)

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
  - NBR (Normalized Burn Ratio)
  - NDMI (Normalized Difference Moisture Index)

### Visualization Tools
- Single raster visualization
- Side-by-side raster comparison
- Change detection visualization
- Distribution analysis
- RGB composite visualization
- Customizable colormaps and scaling
- Interactive plotting capabilities

## Installation

```bash
pip install farq
```

## Quick Start

```python
import farq

# Load raster bands
green, meta = farq.read("landsat_green.tif")
nir, _ = farq.read("landsat_nir.tif")

# Calculate NDWI
ndwi = farq.ndwi(green, nir)

# Create water mask and calculate statistics
water_mask = ndwi > 0
water_pixels = farq.sum(water_mask)
water_percentage = (water_pixels / water_mask.size) * 100

print(f"Water coverage: {water_percentage:.1f}%")

# Visualize results
farq.plot(ndwi, title="NDWI Analysis", cmap="RdYlBu", vmin=-1, vmax=1)
farq.plt.show()
```

## Common Operations

### Water Analysis
```python
# Load and preprocess data
green_1, meta = farq.read("landsat_green_2020.tif")
nir_1, _ = farq.read("landsat_nir_2020.tif")
green_2, _ = farq.read("landsat_green_2024.tif")
nir_2, _ = farq.read("landsat_nir_2024.tif")

# Calculate NDWI for both periods
ndwi_1 = farq.ndwi(green_1, nir_1)
ndwi_2 = farq.ndwi(green_2, nir_2)

# Compare water coverage
farq.compare(ndwi_1, ndwi_2, 
    title1="NDWI 2020", 
    title2="NDWI 2024",
    cmap="RdYlBu",
    vmin=-1, vmax=1)
farq.plt.show()
```

### Machine Learning Analysis
```python
import farq

# Load and preprocess data
bands = {
    'blue': farq.read("landsat_blue.tif")[0],
    'green': farq.read("landsat_green.tif")[0],
    'nir': farq.read("landsat_nir.tif")[0]
}

# Extract features
features = farq.extract_features(bands['nir'], window_size=3)

# Train a classifier
model, metrics = farq.train_classifier(features, labels, model_type='rf')
print(f"Model accuracy: {metrics['accuracy']:.2f}")

# Save the model
farq.save_model(model, "water_classifier.joblib")

# Detect water bodies using clustering
labels, metadata = farq.cluster_water_bodies(
    bands['nir'],
    method='kmeans',
    n_clusters=2,
    water_index=farq.ndwi(bands['green'], bands['nir'])
)

# Analyze water clusters
stats = farq.analyze_water_clusters(labels, metadata['water_cluster'])
print(f"Number of water bodies: {stats['num_water_bodies']}")
print(f"Total water area: {stats['total_water_area']:.2f} km²")
```

### Change Detection with ML
```python
import farq

# Load data from two time periods
nir_2020 = farq.read("landsat_nir_2020.tif")[0]
nir_2024 = farq.read("landsat_nir_2024.tif")[0]

# Detect changes using ML
changes = farq.detect_changes_ml(nir_2020, nir_2024, threshold=0.5)

# Visualize changes
farq.changes(changes, title="Water Body Changes (2020-2024)",
            cmap="RdYlBu", symmetric=True)
farq.plt.show()
```

### Vegetation Analysis
```python
# Load bands
bands = {
    'blue': farq.read("landsat_blue.tif")[0],
    'green': farq.read("landsat_green.tif")[0],
    'red': farq.read("landsat_red.tif")[0],
    'nir': farq.read("landsat_nir.tif")[0]
}

# Calculate indices
ndvi = farq.ndvi(bands['nir'], bands['red'])
evi = farq.evi(bands['red'], bands['nir'], bands['blue'])
savi = farq.savi(bands['nir'], bands['red'])

# Analyze vegetation coverage
veg_mask = ndvi > 0.2
veg_percentage = (farq.sum(veg_mask) / veg_mask.size) * 100
print(f"Vegetation coverage: {veg_percentage:.1f}%")

# Visualize indices
farq.plot(ndvi, title="NDVI Analysis", cmap="RdYlGn", vmin=-1, vmax=1)
farq.plt.show()
```

### RGB Visualization
```python
# Load RGB bands
red = farq.read("landsat_red.tif")[0]
green = farq.read("landsat_green.tif")[0]
blue = farq.read("landsat_blue.tif")[0]

# Create RGB composite
farq.plot_rgb(red, green, blue, title="RGB Composite")
farq.plt.show()
```

## Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

- [Getting Started](docs/index.md)
  - Installation and setup
  - Basic concepts and terminology
  - Quick start tutorials
  - Common workflows

- [API Reference](docs/api.md)
  - Core functions and utilities
  - Water analysis functions
  - Spectral indices
  - Machine learning APIs
  - Visualization tools
  - Error handling and best practices

- [Examples](docs/examples.md)
  - Basic usage examples
  - Water body detection and analysis
  - Change detection workflows
  - Machine learning tutorials
  - Clustering and classification
  - Advanced visualization techniques
  - Performance optimization tips

- [Machine Learning Guide](docs/ml.md)
  - Feature extraction and preprocessing
  - Model training and evaluation
  - Clustering algorithms
  - Change detection with ML
  - Model persistence
  - Parameter optimization
  - Best practices and tips

- [Testing](docs/testing.md)
  - Unit tests and coverage
  - Integration tests
  - Performance benchmarks
  - Memory usage analysis
  - Test data and fixtures

- [Contributing](docs/contributing.md)
  - Development setup
  - Code style guide
  - Pull request workflow
  - Testing guidelines
  - Documentation standards

## Performance

Farq is optimized for large raster datasets with:
- Memory-efficient operations
- Parallel processing capabilities
- Vectorized computations
- Optimized array operations
- Robust error handling
- Comprehensive input validation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.