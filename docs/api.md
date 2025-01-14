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
Returns the minimum value in the array, ignoring NaN values.

### max(data: ndarray) -> float
Returns the maximum value in the array, ignoring NaN values.

### mean(data: ndarray) -> float
Returns the mean value of the array, ignoring NaN values.

### std(data: ndarray) -> float
Returns the standard deviation of the array, ignoring NaN values.

### sum(data: ndarray) -> float
Returns the sum of all values in the array, ignoring NaN values.

## Spectral Indices

### ndwi(green: ndarray, nir: ndarray) -> ndarray
Calculates the Normalized Difference Water Index.

```python
ndwi = farq.ndwi(green, nir)
```

### ndvi(nir: ndarray, red: ndarray) -> ndarray
Calculates the Normalized Difference Vegetation Index.

```python
ndvi = farq.ndvi(nir, red)
```

### evi(red: ndarray, nir: ndarray, blue: ndarray, G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> ndarray
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

### plot(data: ndarray, **kwargs) -> Figure
Creates a single plot visualization.

Parameters:
- data: Array to visualize
- title: Plot title (optional)
- cmap: Colormap name (optional)
- vmin: Minimum value for colormap (optional)
- vmax: Maximum value for colormap (optional)
- colorbar_label: Label for colorbar (optional)

```python
farq.plot(ndwi, title="NDWI Analysis", cmap="RdYlBu", vmin=-1, vmax=1)
farq.plt.show()
```

### compare(data1: ndarray, data2: ndarray, **kwargs) -> Figure
Creates a side-by-side comparison plot.

Parameters:
- data1: First array to visualize
- data2: Second array to visualize
- title1: Title for first plot (optional)
- title2: Title for second plot (optional)
- cmap: Colormap name (optional)
- vmin: Minimum value for colormap (optional)
- vmax: Maximum value for colormap (optional)
- colorbar_label: Label for colorbars (optional)

```python
farq.compare(ndwi_1, ndwi_2, 
    title1="NDWI 2020", 
    title2="NDWI 2024",
    cmap="RdYlBu",
    vmin=-1, vmax=1)
farq.plt.show()
```

### plot_rgb(red: ndarray, green: ndarray, blue: ndarray, **kwargs) -> Figure
Creates an RGB composite visualization.

Parameters:
- red: Red band array
- green: Green band array
- blue: Blue band array
- title: Plot title (optional)
- stretch: Contrast stretch method ('linear', 'hist') (optional)
- percentiles: Tuple of (min, max) percentiles for stretch (optional)

```python
farq.plot_rgb(red, green, blue, title="RGB Composite")
farq.plt.show()
```

### compare_rgb(rgb1: Tuple[ndarray, ndarray, ndarray], rgb2: Tuple[ndarray, ndarray, ndarray], **kwargs) -> Figure
Creates a side-by-side comparison of RGB composites.

Parameters:
- rgb1: Tuple of (red, green, blue) arrays for first image
- rgb2: Tuple of (red, green, blue) arrays for second image
- title1: Title for first plot (optional)
- title2: Title for second plot (optional)
- stretch: Contrast stretch method (optional)
- percentiles: Tuple of (min, max) percentiles for stretch (optional)

```python
farq.compare_rgb(
    (red1, green1, blue1),
    (red2, green2, blue2),
    title1="2020",
    title2="2024"
)
farq.plt.show()
```

### hist(data: ndarray, **kwargs) -> Figure
Creates a histogram visualization.

Parameters:
- data: Array to visualize
- title: Plot title (optional)
- bins: Number of bins (optional)
- range: Tuple of (min, max) for bin range (optional)
- density: Whether to normalize the histogram (optional)

```python
farq.hist(ndwi, title="NDWI Distribution", bins=50)
farq.plt.show()
```

### distribution_comparison(data1: ndarray, data2: ndarray, **kwargs) -> Figure
Creates a side-by-side comparison of distributions.

Parameters:
- data1: First array to visualize
- data2: Second array to visualize
- title1: Title for first plot (optional)
- title2: Title for second plot (optional)
- bins: Number of bins (optional)
- range: Tuple of (min, max) for bin range (optional)
- density: Whether to normalize the histograms (optional)

```python
farq.distribution_comparison(
    ndwi_1, ndwi_2,
    title1="NDWI 2020",
    title2="NDWI 2024",
    bins=50
)
farq.plt.show()
```

## Analysis Functions

### water_stats(ndwi: ndarray) -> Dict
Calculates water statistics from NDWI array.

```python
stats = farq.water_stats(ndwi)
```

### water_change(ndwi1: ndarray, ndwi2: ndarray) -> Dict
Analyzes water body changes between two time periods.

```python
changes = farq.water_change(ndwi_1, ndwi_2)
```

### get_water_bodies(ndwi: ndarray, threshold: float = 0.0) -> ndarray
Identifies individual water bodies from NDWI array.

```python
water_bodies = farq.get_water_bodies(ndwi)
```

## Utility Functions

### validate_array(array: ndarray, name: str = "array") -> None
Validates numpy array inputs for basic operations.

### plt
Access to plotting utilities. Always call `plt.show()` after creating visualizations.

```python
farq.plt.show()
```

## Machine Learning Functions

### extract_features(raster_data: ndarray, indices: Optional[List[str]] = None, window_size: int = 3) -> ndarray
Extracts features from raster data for ML analysis.

```python
features = farq.extract_features(nir_band, window_size=3)
```

### train_classifier(features: ndarray, labels: ndarray, model_type: str = 'rf', test_size: float = 0.2, random_state: int = 42, **model_params) -> Tuple[object, Dict]
Trains a classifier for water detection.

```python
model, metrics = farq.train_classifier(features, labels, model_type='rf')
print(f"Accuracy: {metrics['accuracy']}")
```

### detect_changes_ml(raster1: ndarray, raster2: ndarray, model: Optional[object] = None, threshold: float = 0.5) -> ndarray
Detects changes between two rasters using ML.

```python
changes = farq.detect_changes_ml(nir_2020, nir_2024, threshold=0.5)
```

### cluster_water_bodies(raster_data: ndarray, method: str = 'kmeans', n_clusters: int = 2, water_index: Optional[ndarray] = None, **kwargs) -> Tuple[ndarray, Dict]
Performs unsupervised clustering for water body detection.

```python
labels, metadata = farq.cluster_water_bodies(
    nir_band,
    method='kmeans',
    n_clusters=2,
    water_index=ndwi
)
```

### analyze_water_clusters(cluster_labels: ndarray, water_cluster: int, pixel_size: Union[float, Tuple[float, float]] = 30.0) -> Dict
Analyzes water bodies identified through clustering.

```python
stats = farq.analyze_water_clusters(labels, metadata['water_cluster'])
```

### optimize_clustering(raster_data: ndarray, water_index: Optional[ndarray] = None, method: str = 'kmeans', param_grid: Optional[Dict] = None) -> Tuple[Dict, Dict]
Optimizes clustering parameters for water body detection.

```python
best_params, results = farq.optimize_clustering(
    nir_band,
    water_index=ndwi,
    method='kmeans'
)
```

### save_model(model: object, filepath: Union[str, Path], metadata: Optional[Dict] = None) -> None
Saves a trained model to file.

```python
farq.save_model(model, "water_classifier.joblib", metadata={'date': '2024-03-20'})
```

### load_model(filepath: Union[str, Path]) -> Tuple[object, Dict]
Loads a trained model from file.

```python
model, metadata = farq.load_model("water_classifier.joblib")
``` 