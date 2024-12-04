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

- [Getting Started](docs/index.md) - Overview and installation
- [API Reference](docs/api.md) - Detailed function documentation
- [Examples](docs/examples.md) - Code examples and use cases
- [Testing](docs/testing.md) - Testing documentation and benchmarks

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