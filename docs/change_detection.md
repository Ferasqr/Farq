# Change detection guide

Change detection in Farq is a short pipeline:

1. **Align**: put both dates on one pixel grid ([`align_pair`](api.md#align_pair)).
2. **Measure**: compute a per-pixel change magnitude (difference, log-ratio, normalized
   difference, CVA or PCA).
3. **Threshold**: turn the magnitude into a boolean mask (Otsu, mean + k·std,
   percentile, or a fixed value).
4. **Clean**: remove speckle and fill holes ([`clean_mask`](api.md#clean_mask)).
5. **Summarize**: count pixels and report areas
   ([`change_summary`](api.md#change_summary)).

[`detect_changes`](api.md#detect_changes) runs steps 2-4 in one call. This guide shows
both the one-call version and the individual steps. Change is always **after relative to
before**, and NaN pixels are never reported as change.

The recipes on this page run in order and share variables.

## Prepare two dates

```python
import numpy as np
import farq

names = ("blue", "green", "red", "nir")
stack_20 = np.stack([farq.read(f"{b}_2020.tif", masked=True)[0] for b in names])
stack_24 = np.stack([farq.read(f"{b}_2024.tif", masked=True)[0] for b in names])
_, meta_20 = farq.read("green_2020.tif")
_, meta_24 = farq.read("green_2024.tif")

# Common grid, cropped to the overlap; (4, rows, cols) each.
before, after, meta = farq.align_pair(stack_20, meta_20, stack_24, meta_24)
blue_b, green_b, red_b, nir_b = before
blue_a, green_a, red_a, nir_a = after

ndwi_b = farq.ndwi(green_b, nir_b)  # water > 0
ndwi_a = farq.ndwi(green_a, nir_a)
print(before.shape, farq.pixel_size(meta))
```

## One call: `detect_changes`

```python
result = farq.detect_changes(
    ndwi_b, ndwi_a,
    method="difference",   # |after - before|
    threshold="otsu",      # or "std", "percentile", or a number
    min_size=5,            # drop patches smaller than 5 pixels
    fill_holes=10,         # fill holes of up to 10 pixels
)
print(f"threshold = {result.threshold:.3f}, changed pixels = {result.mask.sum()}")

summary = result.summary(pixel_size=meta)
print(f"{summary['changed_area_km2']:.2f} km² changed "
      f"({summary['changed_percent']:.1f}% of {summary['valid_pixels']} valid pixels, "
      f"{summary['nodata_pixels']} nodata)")
```

`result` is a `ChangeResult` named tuple: `magnitude` (NaN where invalid), `mask` (bool),
`threshold` and `method`.

## Step by step

### Signed change and direction

`detect_changes` thresholds the *absolute* magnitude. To keep the direction, for example
to separate water gain from loss, work with the signed measure:

```python
diff = farq.difference(ndwi_b, ndwi_a)                  # after - before, NaN if invalid
t = farq.compute_threshold(np.abs(diff), "otsu")
increase = farq.clean_mask(diff > t, min_size=5)        # NDWI went up: wetter
decrease = farq.clean_mask(diff < -t, min_size=5)       # NDWI went down: drier
print(f"wetter: {increase.sum()} px, drier: {decrease.sum()} px")

# The same mask in one call; absolute=True counts increases and decreases.
any_change = farq.threshold_change(diff, "otsu", absolute=True)
```

### Choosing a threshold

```python
magnitude = np.abs(diff)
for method in ("otsu", "std", "percentile"):
    t = farq.compute_threshold(magnitude, method, k=2.0, percentile=95)
    print(f"{method:>10}: {t:.3f} -> {np.count_nonzero(magnitude > t)} px")
```

- **`"otsu"`** (default) splits a bimodal histogram. It works well when real change is
  a distinct population.
- **`"std"`** (`mean + k * std`) flags statistical outliers. Use it when change is rare.
- **`"percentile"`** always flags a fixed share of pixels. It is useful for ranking, not
  for estimating area.
- **A number** is best when you know a physically meaningful threshold, such as a 0.2
  change in NDWI.

### Cleaning the mask

```python
raw = magnitude > farq.otsu_threshold(magnitude)
clean = farq.clean_mask(raw, min_size=10, connectivity=8, fill_holes=20)
print(f"raw {raw.sum()} px -> clean {clean.sum()} px")
```

`min_size` removes connected regions with fewer pixels. `fill_holes=20` fills background
holes of up to 20 pixels that are enclosed by change. `fill_holes=True` fills every
enclosed hole, whatever its size. Be careful with `fill_holes=True` when a ring of change
surrounds unchanged land: a shrinking lake leaves a ring of "lost" pixels around the
stable lake, and `True` would mark the whole lake as changed.

## Other change measures

### Log-ratio

The log-ratio `ln(after / before)` treats increases and decreases symmetrically and turns
multiplicative noise into additive noise. It is the standard choice for SAR intensity
and also works for reflectance:

```python
log_ratio = farq.ratio(nir_b, nir_a)                     # NaN where before == 0
db = log_ratio * 10 / np.log(10)                         # in dB
nir_change = farq.detect_changes(nir_b, nir_a, method="ratio", threshold="std", k=2.5)
print(f"NIR log-ratio change: {nir_change.mask.sum()} px")
```

`method="normalized_difference"` uses `(after - before) / (after + before)`, which is
bounded in [-1, 1].

### Multi-band: change vector analysis

CVA uses all bands at once. The change-vector length measures how much a pixel changed,
and its direction tells how:

```python
cva = farq.change_vector_analysis(before, after)          # (bands, rows, cols) stacks
print(cva.magnitude.shape, np.nanmax(cva.magnitude))

# sector: bit i is set when band i increased (-1 where invalid).
nir_up = (cva.sector >= 0) & ((cva.sector & 0b1000) > 0)  # band 3 = NIR increased

cva_result = farq.detect_changes(before, after, method="cva", min_size=5)
```

Bands with very different ranges dominate the CVA magnitude. Scale them first, or use
PCA with `standardize=True`.

### Multi-band: PCA

```python
pca = farq.pca_change(before, after, n_components=2, standardize=True)
print("explained variance:", np.round(pca.explained_variance_ratio, 3))
pc1 = pca.components[0]                                   # dominant change signal

pca_result = farq.detect_changes(before, after, method="pca", threshold="otsu")
```

## Categorical change

### Gained, lost and stable

For binary maps (water / not water), `classify_change` labels every pixel. Pass `valid`
so that nodata pixels are not counted as "no change":

```python
water_b = ndwi_b > 0
water_a = ndwi_a > 0
valid = np.isfinite(ndwi_b) & np.isfinite(ndwi_a)

classes = farq.classify_change(water_b, water_a, valid=valid)  # uint8 codes
summary = farq.change_summary(
    classes, pixel_size=meta, labels=farq.CHANGE_LABELS, nodata=farq.CHANGE_NODATA
)
for name, entry in summary["classes"].items():
    print(f"{name:>10}: {entry['pixels']:7d} px  {entry['area_km2']:7.3f} km²  "
          f"{entry['percent']:5.1f}%")

farq.write("water_classes.tif", classes, meta, nodata=farq.CHANGE_NODATA)
```

The codes are `farq.NO_CHANGE` (0), `farq.GAINED` (1), `farq.LOST` (2), `farq.STABLE`
(3) and `farq.CHANGE_NODATA` (255).

### From-to transitions between classified maps

For multi-class maps (post-classification comparison), use `transition_matrix`:

```python
def classify(ndwi, ndvi):
    """0 = other, 1 = water, 2 = vegetation, 255 = nodata."""
    cls = np.zeros(ndwi.shape, dtype=np.uint8)
    cls[ndvi > 0.4] = 2
    cls[ndwi > 0] = 1
    cls[~(np.isfinite(ndwi) & np.isfinite(ndvi))] = 255
    return cls

map_b = classify(ndwi_b, farq.ndvi(nir_b, red_b))
map_a = classify(ndwi_a, farq.ndvi(nir_a, red_a))

tm = farq.transition_matrix(map_b, map_a, classes=[0, 1, 2], nodata=255)
print(tm.classes)                       # rows = before, columns = after
print(tm.counts)
print(f"{tm.changed} of {tm.total} pixels changed class")
print(np.round(tm.normalized(by="before"), 3))   # where did each class go?
print(tm.areas(meta) / 1e6)                       # km² (meta gives m² per pixel)
```

## Plotting and saving

```python
fig = farq.changes(diff, title="NDWI change", colorbar_label="ΔNDWI")
fig.savefig("ndwi_diff.png", dpi=150)

fig = farq.compare(ndwi_b, ndwi_a, title1="Before", title2="After", cmap="RdYlBu",
                   vmin=-1, vmax=1, colorbar_label="NDWI")
fig.savefig("ndwi_before_after.png")

farq.write("change_magnitude.tif", result.magnitude, meta)             # float32, NaN nodata
farq.write("change_mask.tif", result.mask, meta, dtype="uint8", nodata=None)
```

## Tips

- **Always align first.** Pixel-wise methods require identical grids. `detect_changes`
  raises a `ValueError` on a shape mismatch, but two arrays of the same shape on
  different grids give meaningless results.
- **Use `masked=True` when reading** so that nodata, cloud masks and image borders become
  NaN and are excluded everywhere.
- **Normalize radiometry.** Differences in illumination or season between dates show up
  as change. Ratio indices (NDWI, NDVI) and the log-ratio are more robust than raw band
  differences.
- **Remove residual misregistration** with `coregister`/`apply_shift` for very high
  resolution data. See the [drone guide](drone.md).
