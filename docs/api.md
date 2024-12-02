# API Reference

## Core Functions

### read(filepath: str) -> Tuple[ndarray, Dict]
Reads a raster file and returns the data array and metadata.

```python
data, meta = farq.read("landsat_band.tif")
```

### resample(data: ndarray, target_shape: Tuple[int, int]) -> ndarray
Resamples a raster array to match the target shape.

```python
resampled = farq.resample(data, (1000, 1000))
```

## Statistical Functions

### min(data: ndarray) -> float
Returns the minimum value in the array.

### max(data: ndarray) -> float
Returns the maximum value in the array.

### mean(data: ndarray) -> float
Returns the mean value of the array.

### std(data: ndarray) -> float
Returns the standard deviation of the array.

### sum(data: ndarray) -> float
Returns the sum of all values in the array.

## Spectral Indices

### ndwi(green: ndarray, nir: ndarray) -> ndarray
Calculates the Normalized Difference Water Index.

```python
ndwi = farq.ndwi(green, nir)
```

### ndvi(red: ndarray, nir: ndarray) -> ndarray
Calculates the Normalized Difference Vegetation Index.

```python
ndvi = farq.ndvi(red, nir)
```

### evi(red: ndarray, nir: ndarray, blue: ndarray) -> ndarray
Calculates the Enhanced Vegetation Index.

```python
evi = farq.evi(red, nir, blue)
```

### savi(nir: ndarray, red: ndarray, L: float = 0.5) -> ndarray
Calculates the Soil Adjusted Vegetation Index.

```python
savi = farq.savi(nir, red, L=0.5)
```

## Visualization Functions

### plot(data: ndarray, **kwargs) -> None
Creates a single plot visualization.

Parameters:
- data: Array to visualize
- title: Plot title (optional)
- cmap: Colormap name (optional)
- vmin: Minimum value for colormap (optional)
- vmax: Maximum value for colormap (optional)

```python
farq.plot(ndwi, title="NDWI Analysis", cmap="RdYlBu", vmin=-1, vmax=1)
farq.plt.show()
```

### compare(data1: ndarray, data2: ndarray, **kwargs) -> None
Creates a side-by-side comparison plot.

Parameters:
- data1: First array to visualize
- data2: Second array to visualize
- title1: Title for first plot (optional)
- title2: Title for second plot (optional)
- cmap: Colormap name (optional)
- vmin: Minimum value for colormap (optional)
- vmax: Maximum value for colormap (optional)

```python
farq.compare(ndwi_1, ndwi_2, 
    title1="NDWI 2020", 
    title2="NDWI 2024",
    cmap="RdYlBu",
    vmin=-1, vmax=1)
farq.plt.show()
```

## Utility Functions

### plt
Access to plotting utilities. Always call `plt.show()` after creating visualizations.

```python
farq.plt.show()
``` 