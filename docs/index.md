# Getting started with Farq

Farq (Arabic فَرْق, "difference") detects and measures change between rasters of the
same area acquired at different times. It works with multispectral satellite imagery
(Landsat, Sentinel-2 and similar), with RGB drone orthomosaics, including those
georeferenced only by ground control points, and with elevation models (DEMs and DSMs).

## Installation

```bash
pip install farq
```

Python 3.9 or newer is required. The dependencies are NumPy, SciPy, rasterio (which
bundles GDAL in its wheels), scikit-learn, joblib and matplotlib.

Exporting polygons to GeoPackage, Shapefile or FlatGeobuf needs the optional `vector`
extra (pyogrio and shapely). GeoJSON export works without it.

```bash
pip install "farq[vector]"
```

For development (tests, linting, building):

```bash
pip install -e ".[dev]"
```

## A first example

```python
import numpy as np
import farq

green, meta = farq.read("green_2024.tif", masked=True)  # nodata -> NaN
nir, _ = farq.read("nir_2024.tif", masked=True)

ndwi = farq.ndwi(green, nir)  # (green - nir) / (green + nir); water > 0
water = ndwi > 0
print(f"Water: {farq.mean(np.where(np.isfinite(ndwi), water, np.nan)) * 100:.1f}% of valid pixels")

fig = farq.plot(ndwi, title="NDWI", cmap="RdYlBu", vmin=-1, vmax=1, colorbar_label="NDWI")
fig.savefig("ndwi.png")
```

For a full change-detection workflow see the
[README](https://github.com/ferasqr/farq#readme), the
[change detection guide](change_detection.md) and the [drone guide](drone.md).

## Guides

| Guide | Covers |
| --- | --- |
| [Change detection](change_detection.md) | Change measures, thresholds, mask cleanup, categorical change, transition matrices |
| [Drone imagery](drone.md) | GCPs, rectification, alignment and co-registration of orthomosaics |
| [Cloud and quality masking](masking.md) | Landsat `QA_PIXEL`/`QA_RADSAT`, Sentinel-2 SCL and cloud probability, HLS Fmask, buffering |
| [Radiometric normalization](radiometry.md) | Histogram matching, PIF and linear normalization, calibrated IR-MAD |
| [Elevation change and volumes](elevation.md) | DEMs of difference, DEM co-registration, level of detection, cut/fill and stockpile volumes |
| [Large rasters](tiling.md) | Block-wise, out-of-core change detection, indices, statistics and custom functions |
| [Vector export](vector.md) | Polygonizing change masks and class maps, GeoJSON, GeoPackage, Shapefile, FlatGeobuf |
| [Examples](examples.md) | Short recipes for I/O, indices, statistics, water analysis, ML and plotting |

## Package layout

Every public name is available directly as `farq.<name>`. Submodules load lazily on
first use, so `import farq` is fast. matplotlib, scikit-learn and rasterio load only when
you first need them.

| Module | Purpose |
| --- | --- |
| `farq.core` | `read`, `write`, `resample`, `validate_bands` |
| `farq.indices` | NDWI, MNDWI, NDVI, EVI, SAVI, NDBI, NBR, NDMI and the RGB indices VARI, ExG, ExR, ExGR, GLI, NGRDI, TGI |
| `farq.change` | `detect_changes`, change measures, thresholds, `clean_mask`, `classify_change`, `transition_matrix`, `change_summary` |
| `farq.georef` | GCPs, `georeference`/`rectify`, `align`/`align_pair`, `coregister`/`apply_shift`, `pixel_size`/`pixel_area` |
| `farq.masking` | Cloud/shadow/snow masks: `landsat_qa_mask`, `sentinel2_scl_mask`, `hls_fmask_mask`, `buffer_mask`, `combine_masks`, `apply_mask`, `valid_overlap`, DN scaling |
| `farq.radiometry` | `histogram_match`, `linear_normalize`, `pif_normalize`, `irmad`, `irmad_change` |
| `farq.elevation` | `slope`/`aspect`/`hillshade`, `vertical_offset`, `coregister_dem`, `elevation_change`, `level_of_detection`, `volume_change`, `stockpile_volume` |
| `farq.tiling` | Out-of-core `detect_changes_file`, `index_file`, `map_blocks`, `summarize_file`, `iter_windows` |
| `farq.vector` | `polygonize`, `to_geojson`, `write_vector`, `changes_to_vector` |
| `farq.analysis` | `water_stats`, `water_change`, `get_water_bodies`, `calculate_shape_metrics` |
| `farq.ml` | features, random-forest classifier, model persistence, clustering |
| `farq.visualization` | `plot`, `compare`, `changes`, `hist`, `distribution_comparison`, `plot_rgb`, `compare_rgb` |
| `farq.utils` | NaN-aware statistics: `stats`, `mean`, `median`, `percentile`, ... |

The complete list with signatures is in the [API reference](api.md).

## Conventions

These conventions apply across the library. The [API reference](api.md) lists the
exceptions.

### Arrays and metadata

- **Raster stacks are `(bands, rows, cols)`**, the rasterio layout, in `core`, `change`,
  `georef`, `masking`, `radiometry`, `tiling` and `visualization`. DEMs in `elevation`
  are 2-D. `farq.read(path, band=None)` returns this layout.
- **`farq.ml` uses `(rows, cols, bands)`** (scikit-learn style, one row per pixel).
  Convert with `np.moveaxis(stack, 0, -1)`.
- **Metadata** is a plain dict in rasterio's profile format (`crs`, `transform`, `width`,
  `height`, `count`, `dtype`, `nodata`, `driver`). A raster georeferenced only by ground
  control points also has `gcps` and `gcps_crs`. `farq.read` returns this dict,
  `farq.write` accepts it, and the georef functions return it for their outputs.

### Nodata

- **NaN is nodata.** `farq.read(..., masked=True)` converts nodata pixels (nodata value,
  alpha band or internal mask) to NaN. Indices, change measures, statistics and plots
  propagate or ignore NaN.
- Invalid pixels are never counted as change (`ChangeResult.mask` is `False` there) and
  never counted as water. Float water masks may hold NaN to mark unknown pixels.
- Undefined arithmetic gives NaN, never 0. For example, an index whose denominator is
  zero is NaN, and the log-ratio of zero is NaN.
- `farq.write` stores NaN as the file's nodata value. A float product written with the
  metadata of integer imagery gets NaN as nodata, so valid zeros stay valid.
- **Quality masks are `True` where a pixel is masked** (cloud, shadow, fill).
  `farq.apply_mask(data, mask)` sets those pixels to NaN, so they are never counted as
  change. Change masks, in contrast, are `True` where a pixel changed.

### Change direction

Change is always measured as **after relative to before**: `difference = after - before`,
`ratio = ln(after / before)`, and `classify_change` codes *gained* = absent before and
present after. A DEM of difference is `after - before` too (positive = fill). Radiometric
normalization brings `after` onto the scale of `before`
(`detect_changes(..., normalize="pif")`).

### Indices

- **NDWI uses McFeeters (1996): `(green - nir) / (green + nir)`, so open water is
  > 0.** MNDWI (green, SWIR1) also gives water > 0.
- Index inputs are converted to float (`float32` when lossless, e.g. for `uint8` or
  `uint16`), so integer bands never overflow. Normalized indices are clipped to `[-1, 1]`
  by default (`clip=True`).

### Units

- `change_summary`, `ChangeResult.summary` and `TransitionMatrix.areas` report areas in
  m² and km² for projected CRSs such as UTM.
- **Geographic CRSs (degrees) are refused** with a `ValueError` wherever areas, volumes,
  slopes or ground distances are computed: `change_summary` and `ChangeResult.summary`
  (whether the CRS is a `CRS` object, `"EPSG:4326"` or `4326`), `pixel_area`,
  `detect_changes_file`, `polygonize`/`changes_to_vector`, `farq.elevation` and
  `buffer_mask(distance=...)`. Reproject first (e.g.
  `farq.align_pair(..., dst_crs="EPSG:326xx")`) or pass the pixel size in metres.
- `farq.elevation` needs elevations in **metres** and reports volumes in **m³** and
  areas in **m²**. `farq.vector` polygon attributes `area_m2` and `perimeter_m` are in
  CRS units (metres for UTM).
- `farq.analysis` takes `pixel_size` in metres and returns **areas in km²** and per-body
  **perimeters in km**.
- `farq.ml.analyze_water_clusters` returns **areas in m² and perimeters in m**. These
  units are kept for backward compatibility.
- `farq.pixel_size(meta)` returns the pixel size in CRS units, and `farq.pixel_area(meta)`
  returns the pixel area. `pixel_area` refuses geographic CRSs.

### Georeferencing

- GCP pixel coordinates follow GDAL: `(row, col) = (0, 0)` is the **top-left corner** of
  the top-left pixel, so the centre of that pixel is `(0.5, 0.5)`.
- `coregister` estimates **translation only**. It does not model rotation, scale or local
  distortion.

### Plotting

Every plotting function **returns a matplotlib `Figure`**. Nothing is shown with
`plt.show()` and no other figures are closed. Save the figure with `fig.savefig(...)`,
display it in a notebook, or call `farq.plt.show()` (`farq.plt` is `matplotlib.pyplot`).
To draw into your own layout, pass `ax=` (single-panel functions) or `axes=` (two-panel
functions).

## Next steps

- [API reference](api.md)
- [Change detection guide](change_detection.md)
- [Drone imagery guide](drone.md)
- [Cloud and quality masking](masking.md)
- [Radiometric normalization](radiometry.md)
- [Elevation change and volumes](elevation.md)
- [Processing rasters larger than memory](tiling.md)
- [Vector export](vector.md)
- [Examples](examples.md)
- [Testing and development](testing.md)
- [Changelog and migration notes](https://github.com/ferasqr/farq/blob/main/CHANGELOG.md)

Report issues and feature requests at <https://github.com/ferasqr/farq/issues>.
