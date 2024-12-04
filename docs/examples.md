# Examples

## Basic Water Detection

```python
import farq

# Load bands
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

## Temporal Change Analysis

```python
import farq
import os

# Load data for two time periods
base_dir = "examples/chad_lake"

# 1985 data
green_85, _ = farq.read(os.path.join(base_dir, "1985", "LT05_L2SP_185051_19850501_20200918_02_T1_SR_B2.TIF"))
nir_85, _ = farq.read(os.path.join(base_dir, "1985", "LT05_L2SP_185051_19850501_20200918_02_T1_SR_B4.TIF"))

# 2024 data
green_24, _ = farq.read(os.path.join(base_dir, "2024", "LC08_L2SP_185051_20230323_20230404_02_T1_SR_B3.TIF"))
nir_24, _ = farq.read(os.path.join(base_dir, "2024", "LC08_L2SP_185051_20230323_20230404_02_T1_SR_B5.TIF"))

# Resample 2024 data to match 1985 dimensions
green_24 = farq.resample(green_24, green_85.shape)
nir_24 = farq.resample(nir_24, nir_85.shape)

# Calculate NDWI for both periods
ndwi_85 = farq.ndwi(green_85, nir_85)
ndwi_24 = farq.ndwi(green_24, nir_24)

# Calculate water coverage for each period
water_mask_85 = ndwi_85 > 0
water_percentage_85 = (farq.sum(water_mask_85) / water_mask_85.size) * 100

water_mask_24 = ndwi_24 > 0
water_percentage_24 = (farq.sum(water_mask_24) / water_mask_24.size) * 100

# Print results
print(f"1985 Water coverage: {water_percentage_85:.2f}%")
print(f"2024 Water coverage: {water_percentage_24:.2f}%")
print(f"Change in coverage: {water_percentage_24 - water_percentage_85:.2f}%")

# Visualize comparison
farq.compare(ndwi_85, ndwi_24,
    title1="NDWI 1985",
    title2="NDWI 2024",
    cmap="RdYlBu",
    vmin=-1, vmax=1)
farq.plt.show()

# Analyze distribution changes
farq.distribution_comparison(
    ndwi_85, ndwi_24,
    title1="NDWI Distribution 1985",
    title2="NDWI Distribution 2024",
    bins=50
)
farq.plt.show()
```

## Multi-Index Analysis

```python
import farq

# Load all required bands
bands = {
    'blue': farq.read("landsat_blue.tif")[0],
    'green': farq.read("landsat_green.tif")[0],
    'red': farq.read("landsat_red.tif")[0],
    'nir': farq.read("landsat_nir.tif")[0]
}

# Calculate multiple indices
ndvi = farq.ndvi(bands['nir'], bands['red'])
ndwi = farq.ndwi(bands['green'], bands['nir'])
evi = farq.evi(bands['red'], bands['nir'], bands['blue'])
savi = farq.savi(bands['nir'], bands['red'])

# Calculate coverage statistics
veg_mask = ndvi > 0.2
veg_percentage = (farq.sum(veg_mask) / veg_mask.size) * 100

water_mask = ndwi > 0
water_percentage = (farq.sum(water_mask) / water_mask.size) * 100

# Print results
print(f"Vegetation coverage: {veg_percentage:.1f}%")
print(f"Water coverage: {water_percentage:.1f}%")

# Visualize indices
farq.plot(ndvi, title="NDVI Analysis", cmap="RdYlGn", vmin=-1, vmax=1)
farq.plt.show()

farq.plot(ndwi, title="NDWI Analysis", cmap="RdYlBu", vmin=-1, vmax=1)
farq.plt.show()

farq.plot(evi, title="EVI Analysis", cmap="RdYlGn", vmin=-1, vmax=1)
farq.plt.show()

farq.plot(savi, title="SAVI Analysis", cmap="RdYlGn", vmin=-1, vmax=1)
farq.plt.show()
```

## RGB Visualization

```python
import farq

# Load RGB bands
red = farq.read("landsat_red.tif")[0]
green = farq.read("landsat_green.tif")[0]
blue = farq.read("landsat_blue.tif")[0]

# Create RGB composite
farq.plot_rgb(red, green, blue, title="RGB Composite")
farq.plt.show()

# Compare RGB composites from different dates
red_2020 = farq.read("landsat_red_2020.tif")[0]
green_2020 = farq.read("landsat_green_2020.tif")[0]
blue_2020 = farq.read("landsat_blue_2020.tif")[0]

red_2024 = farq.read("landsat_red_2024.tif")[0]
green_2024 = farq.read("landsat_green_2024.tif")[0]
blue_2024 = farq.read("landsat_blue_2024.tif")[0]

farq.compare_rgb(
    (red_2020, green_2020, blue_2020),
    (red_2024, green_2024, blue_2024),
    title1="RGB 2020",
    title2="RGB 2024"
)
farq.plt.show()
```

## Statistical Analysis

```python
import farq

# Load data
data, meta = farq.read("landsat_band.tif")

# Calculate basic statistics
print(f"Min: {farq.min(data):.2f}")
print(f"Max: {farq.max(data):.2f}")
print(f"Mean: {farq.mean(data):.2f}")
print(f"Standard deviation: {farq.std(data):.2f}")

# Create histogram
farq.hist(data, title="Band Distribution", bins=50)
farq.plt.show()

# Analyze water bodies
ndwi = farq.ndwi(green, nir)
water_bodies = farq.get_water_bodies(ndwi)
stats = farq.water_stats(ndwi)

print("\nWater Body Statistics:")
print(f"Number of water bodies: {stats['num_bodies']}")
print(f"Total water area: {stats['total_area']:.2f} pixels")
print(f"Average water body size: {stats['mean_size']:.2f} pixels")
``` 