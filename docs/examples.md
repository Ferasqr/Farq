# Examples

Practical recipes for the parts of Farq that the [change detection](change_detection.md)
and [drone](drone.md) guides do not cover. The recipes on this page run in order and
share variables. They use single-band Landsat-style GeoTIFFs (`blue_2024.tif`,
`green_2024.tif`, `red_2024.tif`, `nir_2024.tif`, `swir1_2024.tif`, `swir2_2024.tif`, and
the same for 2020) with surface reflectance scaled by 10000 and nodata = 0.

## Reading and writing

```python
import numpy as np
import farq

# Single band -> 2-D array; masked=True turns nodata into NaN (uint16 -> float32).
green, meta = farq.read("green_2024.tif", masked=True)
print(green.shape, green.dtype, meta["crs"], meta["transform"].a, meta["nodata"])

raw, _ = farq.read("green_2024.tif")               # raw digital numbers, 0 = nodata
print(raw.dtype, np.count_nonzero(raw == 0), np.isnan(green).sum())

# Several bands -> (bands, rows, cols). band=None reads every band of a multi-band file.
names = ("blue", "green", "red", "nir", "swir1", "swir2")
bands = {n: farq.read(f"{n}_2024.tif", masked=True)[0] for n in names}
stack = np.stack([bands[n] for n in names])
farq.write("stack_2024.tif", stack, meta, compress="deflate")   # count/dtype updated
again, again_meta = farq.read("stack_2024.tif", band=None)
print(again.shape, again_meta["count"], again_meta["dtype"])

# Decimated read (preview) and plain array resampling.
preview, preview_meta = farq.read("nir_2024.tif", masked=True, out_shape=(100, 100))
print(preview.shape, preview_meta["transform"].a)    # pixel size grew to match
half = farq.resample(bands["nir"], (195, 198), method="average")
```

`write` copies the metadata and never modifies it. It adapts `count`, `dtype`, `width`
and `height` to the data. A nodata value that the output dtype cannot hold raises
`ValueError`, so pass `nodata=` (or `nodata=None`) when you write an integer product with
float metadata.

## Spectral indices

```python
ndwi = farq.ndwi(bands["green"], bands["nir"])        # water > 0
mndwi = farq.mndwi(bands["green"], bands["swir1"])    # water > 0, better near cities
ndvi = farq.ndvi(bands["nir"], bands["red"])
ndbi = farq.ndbi(bands["swir1"], bands["nir"])
nbr = farq.nbr(bands["nir"], bands["swir2"])
ndmi = farq.ndmi(bands["nir"], bands["swir1"])

# EVI and SAVI depend on absolute reflectance: give the scale of the stored integers.
evi = farq.evi(bands["red"], bands["nir"], bands["blue"], reflectance_scale=10000)
savi = farq.savi(bands["nir"], bands["red"], reflectance_scale=10000, L=0.5)

# Several at once from a dict of named bands.
idx = farq.calculate_indices(bands, ["ndwi", "ndvi", "evi"], reflectance_scale=10000)
print(sorted(idx), np.nanmax(idx["ndwi"]))

# Integer input is safe: no uint16 overflow, 0/0 -> NaN instead of 0.
dn = np.array([[0, 1000]], dtype=np.uint16)
print(farq.ndwi(dn, dn * 3))                          # [[nan -0.5]]
```

The RGB-only indices (`vari`, `exg`, `exr`, `exgr`, `gli`, `ngrdi`, `tgi`) take
`(red, green, blue)` and are covered in the [drone guide](drone.md#6-rgb-vegetation-indices).

## Statistics

```python
s = farq.stats(ndwi)
print(f"mean {s['mean']:.3f}, median {s['median']:.3f}, valid {s['valid']}, NaN {s['nan']}")
print(s["percentiles"]["25"], s["percentages"]["nan"])

# NaN-aware reductions (NaN is ignored, never propagated).
print(farq.mean(ndwi), farq.median(ndwi), farq.percentile(ndwi, [5, 95]))
print(farq.std(bands["nir"], ddof=1), farq.max(bands["nir"], axis=0).shape)

# Reflectance statistics from scaled integers.
print(farq.stats(bands["nir"], reflectance_scale=10000)["reflectance_stats"]["mean"])
```

## Water analysis

`farq.analysis` works on 2-D water masks with `pixel_size` in metres and reports areas in
km² and perimeters in km. To keep nodata out of the statistics, pass a float mask with
NaN for unknown pixels:

```python
water = np.where(np.isfinite(ndwi), ndwi > 0, np.nan)   # 1 = water, 0 = land, NaN = unknown
px = farq.pixel_size(meta)                                # (30.0, 30.0) metres

stats = farq.water_stats(water, pixel_size=px, calculate_shapes=True)
print(f"{stats['total_area']:.2f} km² of water in {stats['num_water_bodies']} bodies, "
      f"{stats['coverage_percent']:.1f}% of valid pixels")
for body in stats["shape_metrics"]["body_metrics"]:
    print(f"  {body['area']:.3f} km², perimeter {body['perimeter']:.2f} km, "
          f"compactness {body['compactness']:.2f}")

# Label individual bodies, dropping those under 1 ha (10,000 m²).
labeled, bodies = farq.get_water_bodies(water, pixel_size=px, min_area=10_000,
                                        calculate_shapes=True)
largest = max(bodies, key=lambda k: bodies[k]["area"])
print(f"{len(bodies)} bodies; largest #{largest}: {bodies[largest]['area']:.2f} km²")

# Metrics for one body (pixel_size defaults to 1 -> lengths in pixels).
print(farq.calculate_shape_metrics(labeled == largest, pixel_size=px))
```

### Water change between two dates

Both masks must be on the same grid. See
[change detection](change_detection.md#prepare-two-dates) for `align_pair`.

```python
g20, m20 = farq.read("green_2020.tif", masked=True)
n20, _ = farq.read("nir_2020.tif", masked=True)
b, a, grid = farq.align_pair(np.stack([g20, n20]), m20,
                             np.stack([bands["green"], bands["nir"]]), meta)
ndwi_20, ndwi_24 = farq.ndwi(*b), farq.ndwi(*a)
w20 = np.where(np.isfinite(ndwi_20), ndwi_20 > 0, np.nan)
w24 = np.where(np.isfinite(ndwi_24), ndwi_24 > 0, np.nan)

change = farq.water_change(w20, w24, pixel_size=farq.pixel_size(grid),
                           min_change_area=5 * 900)  # ignore patches < 5 pixels (m²)
print(f"gained {change['gained_area']:.2f} km², lost {change['lost_area']:.2f} km², "
      f"net {change['net_change']:+.2f} km² ({change['change_percent']:+.1f}%)")
cmap = change["change_mask"]   # int8: 1 gained, -1 lost, 0 no change
```

## Machine learning

`farq.ml` uses the **`(rows, cols, bands)`** layout. Convert a stack with
`np.moveaxis(stack, 0, -1)`.

### Supervised classification

```python
pixels = np.moveaxis(stack[:4], 0, -1)                 # blue, green, red, nir -> (r, c, 4)
features = farq.extract_features(pixels, indices={"ndwi": ndwi, "ndvi": ndvi},
                                 window_size=3)
print(features.shape)    # (rows, cols, (4 bands + 2 indices) * 3)

# Training labels: 1 = water, 0 = land, -1 = unlabelled (e.g. digitized polygons).
labels = np.full(ndwi.shape, -1)
labels[ndwi > 0.3] = 1
labels[ndwi < -0.3] = 0

model, metrics = farq.train_classifier(features, labels, ignore_label=-1,
                                       test_size=0.2, n_estimators=50, n_jobs=-1)
print(f"accuracy {metrics['accuracy']:.3f}, train {metrics['n_train']}, "
      f"test {metrics['n_test']}, dropped {metrics['n_dropped']}")
print(metrics["confusion_matrix"])

water_pred = farq.predict_raster(model, features, batch_size=50_000)  # -1 where NaN
print(np.unique(water_pred))
```

A random pixel split is spatially autocorrelated, so the hold-out accuracy is optimistic.
Validate on a separate area or scene.

### Augmenting tabular training data

```python
X = features[labels >= 0]                   # (n_samples, n_features)
y = labels[labels >= 0]
X_aug, y_aug = farq.augment_training_data(X, y, augmentation_factor=3, noise_level=0.05)
print(X.shape, "->", X_aug.shape)           # originals first, then 2 jittered copies
```

Augment only the training split, never before splitting.

### Saving and loading models

```python
digest = farq.save_model(model, "models/water_rf.joblib",
                         metadata={"features": "4 bands + ndwi + ndvi, window 3"})
model2, info = farq.load_model("models/water_rf.joblib", expected_sha256=digest)
print(info["features"], info["_farq"]["sklearn_version"])

# A modified file is rejected before unpickling.
with open("models/water_rf.joblib", "ab") as fh:
    fh.write(b"tampered")
try:
    farq.load_model("models/water_rf.joblib")
except farq.ModelIntegrityError as err:
    print("refused:", type(err).__name__)
```

`save_model` writes `water_rf.joblib` and `water_rf.joblib.sha256`. Loading a pickle can
execute code, so **never load model files from untrusted sources**. Pin
`expected_sha256` with a hash that you received through a trusted channel.

### Unsupervised water detection

```python
labels_k, info_k = farq.cluster_water_bodies(pixels, method="kmeans", n_clusters=3,
                                             water_index=ndwi, sample_size=20_000)
print(info_k["water_rule"], info_k["water_cluster"], np.round(info_k["cluster_means"], 2))

clusters = farq.analyze_water_clusters(labels_k, info_k["water_cluster"],
                                       pixel_size=farq.pixel_size(meta))
print(f"{clusters['num_water_bodies']} bodies, "
      f"{clusters['total_water_area'] / 1e6:.2f} km² (reported in m²)")

best, search = farq.optimize_clustering(pixels, water_index=ndwi, method="kmeans",
                                        param_grid={"n_clusters": [2, 3, 4]},
                                        sample_size=2_000)
print("best:", best, f"silhouette {search['best_score']:.3f}")
```

`analyze_water_clusters` reports areas in m² and perimeters in m, unlike
`farq.analysis`, which uses km² and km.

### Simple change mask without a model

```python
before_px = np.moveaxis(b, 0, -1)          # (rows, cols, 2): green, nir in 2020
after_px = np.moveaxis(a, 0, -1)
changed = farq.detect_changes_ml(before_px, after_px, threshold=1000)  # |Δ| norm > 1000
print(changed.shape, changed.dtype, changed.sum())    # 2-D boolean mask
```

## Visualization

Every plotting function returns a `Figure`. Save it, show it, or draw into your own axes.

```python
import matplotlib.pyplot as plt

fig = farq.plot(ndwi, title="NDWI", cmap="RdYlBu", vmin=-1, vmax=1, colorbar_label="NDWI")
fig.savefig("ndwi_2024.png", dpi=150)

fig = farq.compare(ndwi_20, ndwi_24, title1="2020", title2="2024", cmap="RdYlBu")
fig = farq.changes(ndwi_24 - ndwi_20, title="ΔNDWI")         # symmetric around 0
fig = farq.hist(ndwi, bins=100, title="NDWI distribution", xlabel="NDWI")
fig = farq.distribution_comparison(ndwi_20, ndwi_24, title1="2020", title2="2024")
fig = farq.plot_rgb(bands["red"], bands["green"], bands["blue"], title="True colour",
                    percentile=98, gamma=1.2)
fig = farq.plot_rgb(bands["swir1"], bands["nir"], bands["red"], title="SWIR/NIR/Red")

# Your own layout: pass ax= (single panel) or axes= (two panels).
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
farq.plot(ndwi, ax=axs[0, 0], title="NDWI", cmap="RdYlBu", vmin=-1, vmax=1)
farq.plot(ndvi, ax=axs[0, 1], title="NDVI", cmap="RdYlGn", vmin=-1, vmax=1)
farq.compare_rgb((bands["red"], bands["green"], bands["blue"]),
                 (bands["swir1"], bands["nir"], bands["red"]), axes=axs[1])
fig.savefig("panel.png")
plt.close("all")
```

In a script, call `farq.plt.show()` (or `matplotlib.pyplot.show()`) to open the windows.
In a notebook, a returned figure displays automatically.
