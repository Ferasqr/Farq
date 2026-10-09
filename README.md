# Farq - فَرْق

<p align="center">
  <img src="https://github.com/user-attachments/assets/51b7fd5d-2167-4f68-9c74-dae944c4a8f5" alt="Farq logo" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/farq/"><img src="https://img.shields.io/pypi/v/farq.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/farq/"><img src="https://img.shields.io/pypi/pyversions/farq.svg" alt="Python versions"></a>
  <a href="https://github.com/ferasqr/farq/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/farq.svg" alt="License: MIT"></a>
</p>

**Farq** (Arabic فَرْق, "difference") is a Python library for raster change detection with
satellite and drone imagery. It takes two images of the same place from different dates,
and handles the steps from reading them to reporting the change in km²: reading rasters
with nodata as NaN, putting both dates on one pixel grid (including GCP-only drone
orthomosaics and residual sub-pixel shifts), computing spectral or RGB indices, measuring
and thresholding change, cleaning the change mask, and summarizing it as areas, water-body
statistics or plots. It also masks clouds from the products' quality bands, normalizes
radiometry between dates, measures elevation change and volumes from DEMs, processes
rasters larger than memory, and exports change polygons to GIS formats. Farq is built on
NumPy, SciPy, rasterio/GDAL, scikit-learn and matplotlib.

## Features

**Change detection** (`farq.change`)
- One-call `detect_changes` with six change measures: difference, log-ratio (suits SAR),
  normalized difference, change vector analysis (CVA), PCA and calibrated IR-MAD
- Automatic thresholds (Otsu, mean + k·std, percentile) or a fixed value; IR-MAD is
  thresholded at a chosen false-alarm rate (`alpha`)
- Optional radiometric normalization of the later date (`normalize="pif"`)
- Mask cleanup: remove small patches and fill holes (`clean_mask`)
- Categorical change: gained / lost / stable maps (`classify_change`) and from-to
  transition matrices for classified maps (`transition_matrix`)
- `change_summary`: pixel counts, percentages and areas in m² and km², returned as
  JSON-serializable dicts

**Cloud and quality masking** (`farq.masking`)
- Masks from Landsat Collection 2 `QA_PIXEL` and `QA_RADSAT`, Sentinel-2 SCL and cloud
  probability, and HLS Fmask, with buffering, union and clear-overlap checks
- Digital numbers to reflectance, including the Sentinel-2 baseline 04.00 offset

**Radiometric normalization** (`farq.radiometry`)
- Histogram matching, and linear or automatic pseudo-invariant-feature (PIF) normalization
- IR-MAD change detection that ignores gain and offset differences between dates, with a
  calibrated false-alarm rate

**Elevation and volumes** (`farq.elevation`)
- DEMs of difference, Nuth & Kääb co-registration and level of detection
- Cut/fill volumes with (spatially correlated) uncertainty, and stockpile volumes from a
  single drone survey; slope, aspect and hillshade

**Large rasters** (`farq.tiling`)
- Block-wise, file-to-file change detection, indices, statistics and custom functions
  for rasters larger than memory, in parallel; global thresholds and exact `min_size`
  across block borders

**GIS export** (`farq.vector`)
- Change masks and class maps to polygons with area, perimeter and centroid
- GeoJSON with no extra dependency; GeoPackage, Shapefile and FlatGeobuf with
  `farq[vector]`

**Spectral and RGB indices** (`farq.indices`)
- Multispectral: NDWI, MNDWI, NDVI, EVI, SAVI, NDBI, NBR, NDMI
- RGB-only, for drone cameras without a NIR band: VARI, ExG, ExR, ExGR, GLI, NGRDI, TGI
- Integer bands are converted to float, so `uint16` data cannot overflow. A zero
  denominator gives NaN.

**Drone and GCP georeferencing and alignment** (`farq.georef`)
- Read, build and check ground control points (`read_gcps`, `make_gcps`, `gcp_residuals`
  with RMSE and leave-one-out errors to find bad GCPs)
- Rectify GCP-referenced images with a polynomial or thin plate spline transform
  (`georeference`, `rectify`)
- Put two rasters with different CRS, resolution, extent or georeferencing on one common
  grid cropped to their overlap (`align_pair`, `align`)
- Sub-pixel co-registration by phase correlation (`coregister`, `apply_shift`)
- Downsample huge orthomosaics while reading (`farq.read(..., out_shape=...)`)

**Water analysis** (`farq.analysis`)
- Water area, coverage, number and size of water bodies
- Gained, lost and stable water between two dates, with a minimum patch area
- Per-body shape metrics: area, perimeter, compactness, elongation and orientation

**Machine learning** (`farq.ml`)
- Per-pixel features (bands, index layers, moving-window mean and variance)
- Random-forest training, tiled prediction and training-data augmentation
- Unsupervised water detection with k-means or DBSCAN, and silhouette-based parameter
  search
- Model files saved with a SHA-256 integrity check

**Visualization** (`farq.visualization`)
- Single rasters, side-by-side comparisons, diverging change maps, histograms and RGB
  composites
- Every function returns a matplotlib `Figure`. Nothing is shown or closed for you, and
  you can draw into your own axes with `ax=` or `axes=`.

**Performance and reliability**
- Vectorized NumPy/SciPy code with no per-pixel or per-object Python loops
- `import farq` is fast: submodules and heavy dependencies load on first use
- NaN is the nodata value everywhere. Invalid pixels are never counted as change or as
  water.
- Inputs are validated with specific error messages, and inputs are never modified in place
- Type hints (`py.typed`), Python 3.9-3.13, tested on Linux, macOS and Windows

## Installation

```bash
pip install farq
```

Farq requires Python 3.9 or newer. rasterio wheels include GDAL, so no separate GDAL
installation is needed on most platforms.

To export polygons to GeoPackage, Shapefile or FlatGeobuf, install the optional `vector`
extra (pyogrio and shapely). GeoJSON export works without it.

```bash
pip install "farq[vector]"
```

## Quick start

This example detects change between two dates of the same area:

```python
import farq

before, before_meta = farq.read("nir_2020.tif", masked=True)  # nodata -> NaN
after, after_meta = farq.read("nir_2024.tif", masked=True)
before, after, meta = farq.align_pair(before, before_meta, after, after_meta)  # common grid

result = farq.detect_changes(before, after, method="difference", threshold="otsu", min_size=5)
summary = result.summary(pixel_size=meta)
print(f"{summary['changed_percent']:.1f}% changed = {summary['changed_area_km2']:.2f} km²")
fig = farq.plot(result.mask, title="Change mask", cmap="Reds")
fig.savefig("change_mask.png")
```

## Satellite workflow: water change from Landsat

This workflow reads two dates with nodata as NaN, puts them on one grid, computes NDWI
(water > 0), and then detects and summarizes the change:

```python
import numpy as np
import farq

# 1. Read green (B3) and NIR (B5) for both dates. masked=True turns nodata into NaN.
green_20, meta_20 = farq.read("green_2020.tif", masked=True)
nir_20, _ = farq.read("nir_2020.tif", masked=True)
green_24, meta_24 = farq.read("green_2024.tif", masked=True)
nir_24, _ = farq.read("nir_2024.tif", masked=True)

# 2. Put both dates on one pixel grid, cropped to their overlap. Bands of one date are
#    stacked as (bands, rows, cols) so they are aligned together.
before, after, meta = farq.align_pair(
    np.stack([green_20, nir_20]), meta_20,
    np.stack([green_24, nir_24]), meta_24,
)

# 3. NDWI = (green - nir) / (green + nir). Open water is > 0.
ndwi_20 = farq.ndwi(before[0], before[1])
ndwi_24 = farq.ndwi(after[0], after[1])

# 4. Change magnitude |after - before|, Otsu threshold, drop patches below 5 pixels.
result = farq.detect_changes(ndwi_20, ndwi_24, threshold="otsu", min_size=5)
summary = result.summary(pixel_size=meta)  # areas from the grid's transform
print(f"Changed: {summary['changed_area_km2']:.2f} km² ({summary['changed_percent']:.1f}%)")

# 5. Gained / lost / stable water in km².
valid = np.isfinite(ndwi_20) & np.isfinite(ndwi_24)
classes = farq.classify_change(ndwi_20 > 0, ndwi_24 > 0, valid=valid)
water = farq.change_summary(
    classes, pixel_size=meta, labels=farq.CHANGE_LABELS, nodata=farq.CHANGE_NODATA
)
for name in ("gained", "lost", "stable"):
    print(f"{name:>7}: {water['classes'][name]['area_km2']:.2f} km²")

# 6. Plot (functions return a Figure; nothing is shown automatically).
fig = farq.changes(ndwi_24 - ndwi_20, title="NDWI change 2020 to 2024", colorbar_label="ΔNDWI")
fig.savefig("ndwi_change.png", dpi=150)

# 7. Write GeoTIFFs on the common grid.
farq.write("ndwi_change.tif", ndwi_24 - ndwi_20, meta)
farq.write("water_change_classes.tif", classes, meta, nodata=farq.CHANGE_NODATA)
```

## Drone workflow: vegetation change between two flights

Drone orthomosaics from two flights rarely share a grid. They may also be georeferenced
only by ground control points (GCPs). `align_pair` rectifies GCP-referenced inputs and
resamples both flights onto one grid in a single step. `coregister` then removes the
small remaining translation, which would otherwise appear as false change along every
edge.

```python
import numpy as np
import farq

# 1. Read RGB (bands 1-3; skip an alpha band if present). Both files carry only GCPs.
rgb_23, meta_23 = farq.read("flight_2023.tif", band=[1, 2, 3])
rgb_24, meta_24 = farq.read("flight_2024.tif", band=[1, 2, 3])
print(farq.has_gcps(meta_23), farq.has_gcps(meta_24))  # True True

# 2. Check GCP quality before trusting the georeferencing.
for meta in (meta_23, meta_24):
    report = farq.gcp_residuals(meta["gcps"], order=1)
    print(f"RMSE {report.rmse:.3f} m ({report.rmse_pixels:.2f} px), suspicious GCPs: "
          f"{[report.ids[i] for i in report.outliers()]}")

# 3. Rectify both flights onto one grid at the coarser resolution, cropped to the overlap.
#    uint8 input without a nodata value becomes float32 with NaN outside each footprint.
before, after, meta = farq.align_pair(rgb_23, meta_23, rgb_24, meta_24, target="coarsest")

# 4. Remove the residual sub-pixel shift (translation only).
shift = farq.coregister(before, after)
print(f"Residual shift (rows, cols): {shift} px")
after = farq.apply_shift(after, shift)

# 5. RGB vegetation index (no NIR band needed). farq.vari is an alternative.
exg_23 = farq.exg(*before)
exg_24 = farq.exg(*after)

# 6. Detect change, then clean the mask: drop specks, fill small holes.
result = farq.detect_changes(exg_23, exg_24, threshold="otsu")
mask = farq.clean_mask(result.mask, min_size=50, fill_holes=200)

# 7. Summarize in m² (drone areas are small).
summary = farq.change_summary(mask, pixel_size=meta, valid=np.isfinite(result.magnitude))
print(f"Vegetation change: {summary['changed_area_m2']:.1f} m² "
      f"({summary['changed_percent']:.2f}% of the overlap)")
```

## More recipes

Each recipe below is self-contained. The guides linked from each one explain the methods
and options in detail.

### Large rasters

`farq.tiling` runs the change-detection pipeline file to file, block by block, so memory
depends on the block size and not on the raster size. The inputs must be on one grid.
[Guide](https://github.com/ferasqr/farq/blob/main/docs/tiling.md).

```python
import farq

summary = farq.detect_changes_file(
    "ndwi_2020.tif", "ndwi_2024.tif", "change.tif",  # co-registered rasters
    threshold="otsu", min_size=10, n_jobs=4,
)
print(f"{summary['changed_area_km2']:.2f} km² changed (threshold {summary['threshold']:.3f})")
```

### Export to GIS

`changes_to_vector` turns a change mask or a gained/lost class map into polygons with
area, perimeter and centroid attributes. The format follows the file extension:
`.geojson`, or `.gpkg`, `.shp` and `.fgb` with `farq[vector]`.
[Guide](https://github.com/ferasqr/farq/blob/main/docs/vector.md).

```python
import farq

classes, meta = farq.read("water_change_classes.tif", masked=True)  # from classify_change
polygons = farq.changes_to_vector(classes, meta, "water_change.gpkg", min_area=900)
farq.to_geojson(polygons, "water_change.geojson", crs=meta)          # WGS 84 lon/lat
print(len(polygons), polygons[0]["properties"]["label"], polygons[0]["properties"]["area_m2"])
```

### Cloud masking

Clouds and shadows in one date are the most common false change. `farq.masking` decodes
the quality bands of Landsat Collection 2, Sentinel-2 L2A and HLS, and `apply_mask` sets
the flagged pixels to NaN so they are never counted as change.
[Guide](https://github.com/ferasqr/farq/blob/main/docs/masking.md).

```python
import farq

scene = "LC09_L2SP_044034_20240715_20240716_02_T1"
qa, meta = farq.read(f"{scene}_QA_PIXEL.TIF")
nir, _ = farq.read(f"{scene}_SR_B5.TIF")

clouds = farq.landsat_qa_mask(qa)  # fill, cloud, cloud shadow, cirrus
clouds = farq.buffer_mask(clouds, distance=90, pixel_size=meta)  # grow by 90 m
nir = farq.apply_mask(farq.landsat_c2_scale(nir), clouds)  # reflectance, NaN where masked
print(f"{farq.clear_fraction(clouds):.0%} of the scene is clear")
```

### Radiometric normalization

Sun angle, haze, exposure and sensor calibration differ between acquisitions and show up
as change. `normalize="pif"` fits a per-band gain and offset on automatically selected
unchanged pixels first. `method="irmad"` ignores such differences altogether and is
thresholded at a false-alarm rate `alpha`.
[Guide](https://github.com/ferasqr/farq/blob/main/docs/radiometry.md).

```python
import numpy as np
import farq

bands = ("green", "red", "nir")
stack_20 = np.stack([farq.read(f"{b}_2020.tif", masked=True)[0] for b in bands])
stack_24 = np.stack([farq.read(f"{b}_2024.tif", masked=True)[0] for b in bands])
_, meta_20 = farq.read("nir_2020.tif")
_, meta_24 = farq.read("nir_2024.tif")
before, after, meta = farq.align_pair(stack_20, meta_20, stack_24, meta_24)

cva = farq.detect_changes(before, after, method="cva", normalize="pif", min_size=5)
irmad = farq.detect_changes(before, after, method="irmad", alpha=0.01, min_size=5)
print(f"CVA: {cva.summary(pixel_size=meta)['changed_area_km2']:.2f} km², "
      f"IR-MAD: {irmad.summary(pixel_size=meta)['changed_area_km2']:.2f} km²")
```

### Drone volumes

Cut and fill between two drone surveys (DSMs in metres): co-register the surfaces,
ignore changes below the level of detection, and report volumes with an uncertainty.
[Guide](https://github.com/ferasqr/farq/blob/main/docs/elevation.md).

```python
import farq

dsm_23, meta_23 = farq.read("dsm_2023.tif", masked=True)
dsm_24, meta_24 = farq.read("dsm_2024.tif", masked=True)
before, after, meta = farq.align_pair(dsm_23, meta_23, dsm_24, meta_24)

reg = farq.coregister_dem(before, after, meta)  # better: stable_mask=<unchanged ground>
dod = farq.elevation_change(before, reg.dem)    # after - before, in metres
lod = farq.level_of_detection(reg.nmad_after, 0)
vol = farq.volume_change(dod, meta, lod=lod, sigma=reg.nmad_after, correlation_length=5.0)
print(f"cut {vol.cut_m3:.1f} m³, fill {vol.fill_m3:.1f} m³, "
      f"net {vol.net_m3:+.1f} ± {vol.uncertainty_m3:.1f} m³")
```

## Saving and loading models

`save_model` writes the model file and a `<file>.sha256` sidecar, and it returns the
SHA-256 digest. `load_model` checks the hash before it unpickles the file, and raises
`farq.ModelIntegrityError` if the hash does not match.

```python
import numpy as np
import farq

bands = ("blue", "green", "red", "nir")
stack = np.stack([farq.read(f"{b}_2024.tif", masked=True)[0] for b in bands], axis=-1)
ndwi = farq.ndwi(stack[..., 1], stack[..., 3])
features = farq.extract_features(stack, indices={"ndwi": ndwi}, window_size=3)

# Weak labels from NDWI: 1 = water, 0 = land, -1 = unlabelled (ignored).
labels = np.where(ndwi > 0.2, 1, np.where(ndwi < -0.2, 0, -1))
model, metrics = farq.train_classifier(features, labels, ignore_label=-1, n_estimators=50)
print(f"Hold-out accuracy: {metrics['accuracy']:.3f}")

digest = farq.save_model(model, "models/water_rf.joblib", metadata={"bands": bands})
model, info = farq.load_model("models/water_rf.joblib", expected_sha256=digest)
water = farq.predict_raster(model, features)  # -1 where any feature is NaN
```

> **Security warning:** model files are Python pickles (joblib), and loading a pickle
> can execute arbitrary code. **Never load model files from untrusted sources.** The sidecar hash only detects
> accidental corruption or a swapped file. Someone who can replace the model can also
> replace the sidecar. To guard against a malicious file, pin a hash that you received
> through a trusted channel with `load_model(path, expected_sha256=...)`. Files without
> any integrity data still load, with a warning.

## Documentation

- [Getting started](https://github.com/ferasqr/farq/blob/main/docs/index.md): concepts
  and conventions (array layouts, NaN as nodata, units)
- [API reference](https://github.com/ferasqr/farq/blob/main/docs/api.md): every public
  function, with signatures and return values
- [Change detection guide](https://github.com/ferasqr/farq/blob/main/docs/change_detection.md):
  change measures, thresholds, mask cleanup, transition matrices
- [Drone imagery guide](https://github.com/ferasqr/farq/blob/main/docs/drone.md): GCPs,
  rectification, alignment and co-registration
- [Cloud masking guide](https://github.com/ferasqr/farq/blob/main/docs/masking.md):
  Landsat, Sentinel-2 and HLS quality bands
- [Radiometric normalization guide](https://github.com/ferasqr/farq/blob/main/docs/radiometry.md):
  histogram matching, PIF normalization and calibrated IR-MAD
- [Elevation guide](https://github.com/ferasqr/farq/blob/main/docs/elevation.md): DEMs of
  difference, cut/fill and stockpile volumes
- [Large rasters guide](https://github.com/ferasqr/farq/blob/main/docs/tiling.md):
  out-of-core, block-wise processing
- [Vector export guide](https://github.com/ferasqr/farq/blob/main/docs/vector.md):
  polygons, GeoJSON, GeoPackage, Shapefile and FlatGeobuf
- [Examples](https://github.com/ferasqr/farq/blob/main/docs/examples.md): recipes for
  indices, water analysis, ML and plotting
- [Testing](https://github.com/ferasqr/farq/blob/main/docs/testing.md): running the test
  suite, linting and building
- [Changelog](https://github.com/ferasqr/farq/blob/main/CHANGELOG.md): what is new in
  0.3, and the 0.1 to 0.2 migration notes

## Contributing

Contributions are welcome. To set up a development environment:

```bash
git clone https://github.com/ferasqr/farq.git
cd farq
pip install -e ".[dev,vector]"

python -m pytest                    # test suite (performance tests are skipped)
ruff check farq tests               # lint
ruff format --check farq tests      # formatting
```

Please add tests for new behaviour and update the docs and `CHANGELOG.md`, then open a
pull request. See [docs/testing.md](https://github.com/ferasqr/farq/blob/main/docs/testing.md)
for details.

## License

Farq is released under the [MIT License](https://github.com/ferasqr/farq/blob/main/LICENSE).
