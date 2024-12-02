# Farq API Reference

## Core Functions

### read(filepath: str) -> Tuple[np.ndarray, Dict]
Load a raster file and return its data and metadata.

**Parameters:**
- `filepath`: Path to the raster file

**Returns:**
- Tuple containing:
  - `np.ndarray`: Raster data array
  - `Dict`: Metadata dictionary

### resample(raster: np.ndarray, shape: Tuple[int, int], method: Resampling = Resampling.bilinear) -> np.ndarray
Resample a raster array to a target shape.

**Parameters:**
- `raster`: Input raster array
- `shape`: Target shape (height, width)
- `method`: Resampling method from rasterio.enums.Resampling

**Returns:**
- `np.ndarray`: Resampled raster array

## Spectral Indices

### ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray
Calculate Normalized Difference Water Index.

**Parameters:**
- `green`: Green band data
- `nir`: NIR band data

**Returns:**
- `np.ndarray`: NDWI values [-1, 1]

### ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray
Calculate Normalized Difference Vegetation Index.

**Parameters:**
- `nir`: NIR band data
- `red`: Red band data

**Returns:**
- `np.ndarray`: NDVI values [-1, 1]

### mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray
Calculate Modified Normalized Difference Water Index.

**Parameters:**
- `green`: Green band data
- `swir`: SWIR band data

**Returns:**
- `np.ndarray`: MNDWI values [-1, 1]

### savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray
Calculate Soil Adjusted Vegetation Index.

**Parameters:**
- `nir`: NIR band data
- `red`: Red band data
- `L`: Soil brightness correction factor

**Returns:**
- `np.ndarray`: SAVI values [-1.5, 1.5]

## Analysis Functions

### water_stats(water_mask: np.ndarray, pixel_size: Union[float, Tuple[float, float]] = 30.0) -> Dict[str, float]
Calculate water surface statistics.

**Parameters:**
- `water_mask`: Binary water mask (True/1 for water)
- `pixel_size`: Pixel size in meters

**Returns:**
- Dictionary containing:
  - `total_area`: Total water surface area (km²)
  - `coverage_percent`: Percentage of area covered by water

### water_change(mask1: np.ndarray, mask2: np.ndarray, pixel_size: Union[float, Tuple[float, float]] = 30.0) -> Dict[str, float]
Analyze changes between two water masks.

**Parameters:**
- `mask1`: First water mask
- `mask2`: Second water mask
- `pixel_size`: Pixel size in meters

**Returns:**
- Dictionary containing:
  - `gained_area`: New water area (km²)
  - `lost_area`: Lost water area (km²)
  - `net_change`: Net change in water area (km²)
  - `change_percent`: Percentage change

## Visualization Functions

### plot(raster: np.ndarray, **kwargs) -> plt.Figure
Plot a single raster.

**Parameters:**
- `raster`: Input raster data
- `title`: Plot title (default: "Raster Data")
- `cmap`: Colormap (default: "viridis")
- `vmin`, `vmax`: Value range for colormap

### compare(raster1: np.ndarray, raster2: np.ndarray, **kwargs) -> plt.Figure
Compare two rasters side by side.

**Parameters:**
- `raster1`, `raster2`: Input rasters
- `title1`, `title2`: Titles for each plot
- `cmap`: Colormap
- `vmin`, `vmax`: Value range for both plots

### changes(data: np.ndarray, **kwargs) -> plt.Figure
Plot change detection results.

**Parameters:**
- `data`: Change detection results
- `title`: Plot title (default: "Changes")
- `cmap`: Colormap (default: "RdYlBu")

### hist(data: np.ndarray, **kwargs) -> plt.Figure
Plot histogram of values.

**Parameters:**
- `data`: Input data
- `bins`: Number of histogram bins
- `title`: Plot title 