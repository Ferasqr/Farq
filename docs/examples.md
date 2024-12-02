# Farq Examples

## Basic Usage

### Loading and Visualizing Data
```python
import farq

# Load raster data
data, meta = farq.read("landsat_band.tif")

# Visualize the data
farq.plot(data, title="Landsat Band", cmap="viridis")
farq.plt.show()
```

### Calculating Water Indices

#### NDWI Calculation
```python
# Load required bands
green, _ = farq.read("LC08_B3.tif")  # Landsat 8 Green band
nir, _ = farq.read("LC08_B5.tif")    # Landsat 8 NIR band

# Calculate NDWI
ndwi = farq.ndwi(green, nir)

# Create water mask
water_mask = ndwi > 0

# Visualize results
farq.plot(ndwi, title="NDWI", cmap="RdYlBu")
farq.plt.show()
```

#### MNDWI Calculation
```python
# Load required bands
green, _ = farq.read("LC08_B3.tif")   # Green band
swir, _ = farq.read("LC08_B6.tif")    # SWIR band

# Calculate MNDWI
mndwi = farq.mndwi(green, swir)

# Visualize results
farq.plot(mndwi, title="MNDWI", cmap="RdYlBu")
farq.plt.show()
```

## Water Analysis

### Basic Water Statistics
```python
# Create water mask from NDWI
water_mask = ndwi > 0

# Calculate statistics
stats = farq.water_stats(water_mask, pixel_size=30.0)

print(f"Total water area: {stats['total_area']:.2f} km²")
print(f"Water coverage: {stats['coverage_percent']:.1f}%")
```

### Water Body Identification
```python
# Get individual water bodies
labeled, areas = farq.get_water_bodies(water_mask, pixel_size=30.0, min_area=9000)

# Print information about water bodies
print(f"Number of water bodies: {len(areas)}")
for id, area in areas.items():
    print(f"Water body {id}: {area:.2f} km²")

# Visualize water bodies
farq.plot(labeled, title="Water Bodies", cmap="tab20")
farq.plt.show()
```

## Change Detection

### Basic Change Detection
```python
# Load data from two time periods
ndwi_1985 = farq.ndwi(green_1985, nir_1985)
ndwi_2024 = farq.ndwi(green_2024, nir_2024)

# Create water masks
mask_1985 = ndwi_1985 > 0
mask_2024 = ndwi_2024 > 0

# Analyze changes
changes = farq.water_change(mask_1985, mask_2024, pixel_size=30.0)

print(f"New water area: {changes['gained_area']:.2f} km²")
print(f"Lost water area: {changes['lost_area']:.2f} km²")
print(f"Net change: {changes['net_change']:.2f} km²")
```

### Visualizing Changes
```python
# Compare water extent
farq.compare(
    mask_1985, mask_2024,
    title1="Water Extent 1985",
    title2="Water Extent 2024",
    cmap="Blues"
)
farq.plt.show()

# Plot change distribution
farq.hist(changes, bins=50, title="Change Distribution")
farq.plt.show()
```

## Advanced Usage

### Custom Visualization
```python
# Create custom visualization with multiple indices
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

farq.plot(ndvi, title="NDVI", cmap="RdYlGn", ax=axes[0,0])
farq.plot(ndwi, title="NDWI", cmap="RdYlBu", ax=axes[0,1])
farq.plot(mndwi, title="MNDWI", cmap="RdYlBu", ax=axes[1,0])
farq.plot(savi, title="SAVI", cmap="RdYlGn", ax=axes[1,1])

plt.tight_layout()
plt.show()
```

### Batch Processing
```python
import glob

# Process multiple files
files = glob.glob("landsat/*.tif")
results = {}

for file in files:
    data, meta = farq.read(file)
    stats = farq.water_stats(data > 0)
    results[file] = stats
``` 