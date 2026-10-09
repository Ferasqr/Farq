# API reference

Every function and class below is available as `farq.<name>` and also from its submodule
(for example, `farq.detect_changes` is `farq.change.detect_changes`). Signatures are
copied from the code. Parameters after `*` are keyword-only.

- [Key conventions](#key-conventions)
- [farq.core](#farqcore): read, write, resample
- [farq.indices](#farqindices): spectral and RGB indices
- [farq.change](#farqchange): change detection
- [farq.georef](#farqgeoref): GCPs, rectification, alignment, co-registration
- [farq.masking](#farqmasking): cloud, shadow and quality masks
- [farq.radiometry](#farqradiometry): radiometric normalization and IR-MAD
- [farq.elevation](#farqelevation): DEM differencing, co-registration and volumes
- [farq.tiling](#farqtiling): out-of-core processing of large rasters
- [farq.vector](#farqvector): polygons and GIS export
- [farq.analysis](#farqanalysis): water statistics and shape metrics
- [farq.ml](#farqml): machine learning
- [farq.visualization](#farqvisualization): plotting
- [farq.utils](#farqutils): NaN-aware statistics
- [Compatibility names](#compatibility-names)

## Key conventions

| Topic | Convention |
| --- | --- |
| Nodata | **NaN is nodata.** Use `read(..., masked=True)` to convert a file's nodata to NaN. Invalid pixels are never flagged as change or counted as water. |
| NDWI sign | `ndwi = (green - nir) / (green + nir)` (McFeeters 1996): **open water is > 0**. |
| Division by zero | Index pixels with a zero denominator are **NaN** (not 0), and no `RuntimeWarning` is emitted. |
| Change direction | **after relative to before**: `after - before`, `ln(after / before)`. *Gained* means absent before and present after. |
| Array layout | `(bands, rows, cols)` everywhere except `farq.ml`, which uses `(rows, cols, bands)`. |
| GCP pixels | GDAL convention: `(row, col) = (0, 0)` is the top-left **corner** of the image, and the centre of the first pixel is `(0.5, 0.5)`. |
| Co-registration | `coregister` estimates a **translation only** (no rotation, scale or local distortion). |
| Units | `change_summary`, `detect_changes_file` and `farq.vector` report areas in m² and km², converting from the CRS's linear unit (e.g. US survey feet) when areas come from raster metadata; a plain `pixel_size` number is taken as metres. `farq.analysis` takes `pixel_size` in metres and returns areas in **km²** and per-body perimeters in **km**. `ml.analyze_water_clusters` returns areas in **m²** and perimeters in **m**. `farq.elevation` takes elevations in metres and returns m, m² and m³. `farq.vector` polygon attributes `area_m2` and `perimeter_m` are in metres; coordinates and `simplify` stay in CRS units. |
| Geographic CRS | Areas, volumes, slopes, polygon metrics and ground distances need a projected CRS such as UTM. Metadata in a geographic CRS (degrees), in any spelling (a `CRS`, `"EPSG:4326"`, `4326`), raises `ValueError` in `change_summary` and `ChangeResult.summary`, `pixel_area`, `detect_changes_file` (unless `pixel_size=` is given), `polygonize` and `changes_to_vector`, every `farq.elevation` function that takes `meta`, and `buffer_mask(distance=...)`. Reproject first (`align_pair(..., dst_crs=...)`) or pass the pixel size in metres where accepted. |
| Masks | Quality masks from `farq.masking` are `True` where a pixel is **masked** (unusable). Change masks are `True` where a pixel changed. |
| Large rasters | `farq.tiling` functions work file to file, block by block; their inputs must share one grid. |
| Figures | Plot functions return a `matplotlib.figure.Figure`, never call `plt.show()`, and never close other figures. |
| Inputs | Input arrays are never modified in place. |

---

## farq.core

### `read`

```text
read(filepath, band=1, *, masked=False, out_shape=None, resampling=Resampling.average)
    -> (data, metadata)
```

Reads a raster from a path (`str` or `os.PathLike`) or a GDAL-readable URL.

- `band`: a 1-based index returns a 2-D array. A sequence of indices, or `None` for all
  bands, returns a 3-D `(bands, rows, cols)` array.
- `masked=True`: sets nodata pixels (nodata value, alpha band or internal mask) to NaN.
  Integer data becomes `float32` (8/16-bit) or `float64`.
- `out_shape=(height, width)`: decimated read, for previews of huge orthomosaics.
  `metadata["height"/"width"/"transform"]` and the GCP pixel positions are updated to
  match. `resampling` is a `rasterio.enums.Resampling` member or its name. See also
  [Output size limit](#output-size-limit).
- Returns `metadata`, a copy of the dataset profile (`driver`, `dtype`, `nodata`,
  `width`, `height`, `count`, `crs`, `transform`). GCP-referenced files also get `gcps`
  (list of `rasterio.control.GroundControlPoint`) and `gcps_crs`.
- Raises `FileNotFoundError` (missing local file), `IndexError` (band out of range),
  `ValueError` (not a raster) or `RuntimeError`.

### `write`

```text
write(filepath, data, metadata=None, *, dtype=None, nodata=<from metadata>, **profile) -> None
```

Writes a `(rows, cols)` or `(bands, rows, cols)` array. `metadata` (usually from `read`
or a georef function) is copied and never modified. `height`, `width`, `count` and
`dtype` are updated from `data`, and `driver` defaults to `"GTiff"`.

- `dtype`: cast the output. `bool` is written as `uint8` and `float16` as `float32`.
- `nodata`: overrides `metadata["nodata"]`. `nodata=None` writes no nodata value. A nodata
  value that the output dtype cannot represent raises `ValueError`. For example, to write
  an integer array with metadata whose nodata is NaN, pass `nodata=` explicitly.
- If `nodata` is not given and a **float** result is written with the metadata of
  **integer** data (e.g. an index computed from `uint8` imagery with nodata 0), NaN is
  used as nodata, so valid zeros are not turned into nodata.
- NaN values are stored as the nodata value when that value is finite. Writing NaN or inf
  to an integer dtype requires a nodata value, otherwise `ValueError` is raised.
- Masked arrays are filled with the nodata value (NaN for float data without one).
- GeoTIFF (and COG) files are written **atomically**: a temporary file in the destination
  directory replaces `filepath` only when it is complete, so a failed write never leaves
  a truncated file or destroys an existing one.
- `gcps`/`gcps_crs` in `metadata` are written, unless `metadata["transform"]` is a real
  geotransform, which takes precedence.
- `**profile`: creation options such as `compress="deflate"`.
- Raises `TypeError`, `ValueError` or `RuntimeError`.

### `resample`

```text
resample(array, target_shape, method=Resampling.bilinear) -> ndarray
```

Resamples a 2-D or `(bands, rows, cols)` array to `target_shape = (height, width)` with
GDAL. NaN is treated as nodata. The output keeps the input dtype (`bool` and `float16`
are processed internally as `uint8` and `float32`). `method` is a `Resampling` member or
a name such as `"nearest"`, `"bilinear"` or `"average"`. `resample` only changes the
array shape, and is subject to the [output size limit](#output-size-limit). To match two georeferenced rasters, use [`align`](#align) or
[`align_pair`](#align_pair).

### Output size limit

`read(out_shape=...)`, `resample` and the georef warping functions (`georeference`,
`rectify`, `align`, `align_pair`) refuse to allocate an output larger than
2³² pixels (bands × rows × cols) and raise `ValueError`. This catches mistakes such as a
resolution given in metres for a CRS in degrees. To change the limit, set the
`FARQ_MAX_OUTPUT_PIXELS` environment variable to a positive integer.

### `validate_bands`

```text
validate_bands(*bands, reflectance_scale=None) -> list[ndarray]
```

Checks that the bands are non-empty numeric arrays of the same shape, and returns them as
float arrays: `float32` when lossless (e.g. `uint8`, `uint16`, `int16`, `float32`),
otherwise `float64`. Masked values become NaN. If `reflectance_scale` is given (e.g.
`10000`), the bands are divided by it. Raises `TypeError` or `ValueError`.

---

## farq.indices

Common behaviour of all index functions:

- Bands are converted with `validate_bands`, so `uint8` and `uint16` input never
  overflows. Inputs are not modified.
- A zero denominator gives NaN. NaN input propagates. No `RuntimeWarning` is emitted.
- `clip=True` (keyword-only, where available) clips finite results to `[-1, 1]`.
- `reflectance_scale` is validated, but it changes the result only for EVI, SAVI and
  TGI. The other indices are ratios and do not depend on scale.

### Multispectral indices

Landsat 8/9 OLI band numbers are given for reference.

| Function | Formula | Notes |
| --- | --- | --- |
| `ndwi(green, nir, reflectance_scale=None, *, clip=True)` | (G − NIR) / (G + NIR) | McFeeters 1996. **Water > 0.** Landsat B3, B5. |
| `mndwi(green, swir1, reflectance_scale=None, *, clip=True)` | (G − SWIR1) / (G + SWIR1) | Xu 2006. Water > 0, separates water from built-up areas better than NDWI. |
| `ndvi(nir, red, reflectance_scale=None, *, clip=True)` | (NIR − R) / (NIR + R) | Vegetation is roughly > 0.2. |
| `evi(red, nir, blue, reflectance_scale=None, G=2.5, C1=6.0, C2=7.5, L=1.0, *, clip=True)` | G·(NIR − R) / (NIR + C1·R − C2·B + L) | Needs reflectance in [0, 1]. Pass `reflectance_scale` for scaled integers. `G > 0`, `L ≥ 0`. |
| `savi(nir, red, reflectance_scale=None, L=0.5, *, clip=True)` | (1 + L)·(NIR − R) / (NIR + R + L) | Needs reflectance. `L` in [0, 1]. `L=0` is NDVI. |
| `ndbi(swir1, nir, reflectance_scale=None, *, clip=True)` | (SWIR1 − NIR) / (SWIR1 + NIR) | Built-up areas are high. |
| `nbr(nir, swir2, reflectance_scale=None, *, clip=True)` | (NIR − SWIR2) / (NIR + SWIR2) | Burned areas are low. |
| `ndmi(nir, swir1, reflectance_scale=None, *, clip=True)` | (NIR − SWIR1) / (NIR + SWIR1) | Moisture. |

### RGB-only indices (drone cameras)

Lower-case r, g, b are chromatic coordinates, for example `r = R / (R + G + B)`.

| Function | Formula | Range and notes |
| --- | --- | --- |
| `vari(red, green, blue, reflectance_scale=None, *, clip=True)` | (G − R) / (G + R − B) | Gitelson 2002. The denominator can approach 0, so values are clipped to [-1, 1] by default. |
| `exg(red, green, blue, reflectance_scale=None)` | 2g − r − b | [-1, 2]. Vegetation is roughly > 0.1. |
| `exr(red, green, blue, reflectance_scale=None)` | 1.4r − g | [-1, 1.4]. Soil and residue are high. |
| `exgr(red, green, blue, reflectance_scale=None)` | ExG − ExR = 3g − 2.4r − b | [-2.4, 3]. Vegetation > 0. |
| `gli(red, green, blue, reflectance_scale=None, *, clip=True)` | (2G − R − B) / (2G + R + B) | [-1, 1]. Vegetation > 0. |
| `ngrdi(red, green, reflectance_scale=None, *, clip=True)` | (G − R) / (G + R) | [-1, 1]. Vegetation > 0. |
| `tgi(red, green, blue, reflectance_scale=None, *, wavelengths=(670.0, 550.0, 480.0))` | −0.5·[(λr − λb)(R − G) − (λr − λg)(R − B)] | Not normalized: it scales with the input. Use `reflectance_scale=255` for 8-bit images. `wavelengths` are the band centres in nm. |

### `calculate_normalized_difference`

```text
calculate_normalized_difference(band1, band2, clip=True) -> ndarray
```

Returns `(band1 - band2) / (band1 + band2)`, with NaN where `band1 + band2 == 0`.

### `calculate_indices`

```text
calculate_indices(bands, indices, reflectance_scale=None) -> dict[str, ndarray]
```

`bands` maps band names (`"blue"`, `"green"`, `"red"`, `"nir"`, `"swir1"`, `"swir2"`) to
arrays. `indices` is one name or a list of names (case-insensitive): `ndvi`, `ndwi`,
`mndwi`, `evi`, `savi`, `ndbi`, `nbr`, `ndmi`, `vari`, `exg`, `exr`, `exgr`, `gli`,
`ngrdi`, `tgi`. Returns a dict of lower-case name to array. Raises `ValueError` for an
unknown index or a missing band.

---

## farq.change

Change measures accept masked arrays and a `nodata=` sentinel. NaN and ±inf are always
invalid, and invalid pixels are NaN in magnitudes and `False` in masks. Integer inputs
are processed as `float32` (`float64` for 32/64-bit integers).

### `detect_changes`

```text
detect_changes(before, after, method="difference", threshold="otsu", *, k=2.0,
               percentile=95.0, min_size=0, connectivity=8, fill_holes=False,
               nodata=None, alpha=0.01, normalize=None) -> ChangeResult
```

Runs the whole change-detection pipeline: magnitude, then threshold, then cleanup.

- `method`: the magnitude is the absolute value of the signed measure.
  - `"difference"`: `|after - before|`
  - `"ratio"`: `|ln(after / before)|`, recommended for SAR
  - `"normalized_difference"`: `|(after - before) / (after + before)|`
  - `"cva"`: change-vector length
  - `"pca"`: `|PC1|` of the difference image
  - `"irmad"`: the calibrated IR-MAD chi-square statistic ([`irmad`](#ir-mad)), which
    ignores per-band gain and offset differences between the dates

  `"cva"`, `"pca"` and `"irmad"` accept `(rows, cols)` or `(bands, rows, cols)` input.
  The other methods need 2-D input (a leading band axis of size 1 is accepted).
- `threshold`: `"otsu"`, `"std"` (mean + `k`·std), `"percentile"` (the `percentile`-th
  value) or a number. See [`compute_threshold`](#compute_threshold). Not used with
  `"irmad"`: passing a `threshold` with `method="irmad"` raises `ValueError`.
- `alpha` (default 0.01, in (0, 1)): for `"irmad"` only, the false-alarm rate. The
  statistic is thresholded at the `1 - alpha` quantile of the chi-square distribution
  with one degree of freedom per band, so about a fraction `alpha` of the unchanged
  pixels are flagged (under Gaussian no-change noise). `ChangeResult.threshold` holds
  that quantile.
- `normalize`: `None` (default), `"pif"` or `"histogram"`. Radiometrically normalizes
  `after` to `before` before the magnitude is computed, so illumination, exposure or
  sensor differences are not detected as change. `"pif"` runs
  [`pif_normalize`](#normalization) with its defaults and is recommended, because it is
  fitted on unchanged pixels only. `"histogram"` runs `histogram_match`, which also
  reshapes the distribution and can attenuate real change that covers a noticeable part
  of the scene.
- `min_size`, `connectivity` (4 or 8) and `fill_holes` (bool or maximum hole size in
  pixels) are passed to [`clean_mask`](#clean_mask).

### `ChangeResult`

A `NamedTuple` with these fields:

- `magnitude`: non-negative, NaN where invalid
- `mask`: bool, `False` where invalid
- `threshold`: float. Pixels with `magnitude > threshold` are change, before cleanup.
- `method`: str

`ChangeResult.summary(pixel_size=None)` returns `change_summary(mask, pixel_size=...,
valid=isfinite(magnitude))`, so invalid pixels count as nodata.

### Change measures

```text
difference(before, after, *, nodata=None, absolute=False) -> ndarray
ratio(before, after, *, log=True, nodata=None) -> ndarray
normalized_difference_change(before, after, *, nodata=None) -> ndarray
change_vector_analysis(before_stack, after_stack, *, nodata=None) -> CVAResult
pca_change(before_stack, after_stack, *, n_components=None, standardize=False,
           nodata=None) -> PCAChangeResult
```

- `difference`: `after - before` (or its absolute value), for any matching shape.
- `ratio`: `ln(after / before)`, or the plain ratio with `log=False`. NaN where
  `before == 0` or where the ratio is ≤ 0 with `log=True`. Multiply by `10 / ln(10)` to
  convert to dB.
- `normalized_difference_change`: `(after - before) / (after + before)`, in [-1, 1] for
  non-negative inputs.
- `change_vector_analysis` takes `(bands, rows, cols)` stacks (2-D is one band, at most
  31 bands) and returns `CVAResult(magnitude, direction, sector)`:
  - `direction`: degrees in [0, 360) in the plane of bands 0 and 1
  - `sector`: `int32`, bit *i* set when band *i* increased, `-1` where invalid
- `pca_change` computes the PCA of the multi-band difference image, with statistics
  accumulated in float64 over chunks. It returns `PCAChangeResult(components,
  explained_variance_ratio, loadings, mean)`. `components` has shape
  `(n_components, rows, cols)`, ordered by decreasing variance. `standardize=True` gives
  correlation PCA.

### Thresholds

```text
otsu_threshold(values, bins=256) -> float
compute_threshold(values, method="otsu", *, k=2.0, percentile=95.0, bins=256) -> float
threshold_change(magnitude, method="otsu", *, k=2.0, percentile=95.0, absolute=False,
                 bins=256) -> ndarray[bool]
```

<a id="compute_threshold"></a>

- `otsu_threshold` maximizes the between-class variance, ignoring NaN and ±inf. If all
  values are equal, it returns that value.
- `compute_threshold` methods: `"otsu"`, `"std"` (mean + k·std), `"percentile"`, or a
  finite number that is returned as is.
- `threshold_change` returns `magnitude > threshold`, and NaN is never change. Use
  `absolute=True` for signed measures (difference, log-ratio) so that both increases and
  decreases count.

### `clean_mask`

<a id="clean_mask"></a>

```text
clean_mask(mask, *, min_size=0, connectivity=8, fill_holes=False) -> ndarray[bool]
```

Removes connected regions smaller than `min_size` pixels from a 2-D mask and optionally
fills holes enclosed by change. `fill_holes=True` fills every enclosed hole, and an
integer fills only holes of at most that many pixels. `connectivity` is 4 or 8. Holes use
the complementary connectivity. NaN entries count as `False`. Returns a new array.

### Categorical change

```text
classify_change(before_mask, after_mask, *, valid=None) -> ndarray[uint8]
transition_matrix(class_before, class_after, classes=None, *, nodata=None) -> TransitionMatrix
```

- `classify_change` compares two binary maps and returns these codes:

  | Constant | Code | Meaning |
  | --- | --- | --- |
  | `farq.NO_CHANGE` | 0 | absent in both |
  | `farq.GAINED` | 1 | absent before, present after |
  | `farq.LOST` | 2 | present before, absent after |
  | `farq.STABLE` | 3 | present in both |
  | `farq.CHANGE_NODATA` | 255 | invalid in either date |

  NaN or masked inputs, and pixels where `valid` is `False`, are `CHANGE_NODATA`.
  `farq.CHANGE_LABELS` maps codes 0-3 to names (`"no_change"`, `"gained"`, `"lost"`,
  `"stable"`).
- `transition_matrix` does post-classification comparison and returns
  `TransitionMatrix(counts, classes)`, where `counts[i, j]` is the number of pixels that
  were `classes[i]` before and `classes[j]` after. NaN labels and `nodata` are ignored.
  Properties and methods:
  - `.total` and `.changed`: off-diagonal sum
  - `.normalized(by="all" | "before" | "after")`: fractions
  - `.areas(pixel_size)`: m² (converted from the CRS unit when `pixel_size` is metadata)

### `change_summary`

```text
change_summary(data, *, pixel_size=None, labels=None, nodata=None, valid=None) -> dict
```

Pixel counts, percentages and areas for a boolean mask or a class map.

- `pixel_size` can be a number (square pixels), an `(x, y)` pair, an `affine.Affine`, a
  rasterio meta/profile dict with a real geotransform (such as the metadata returned by
  `align_pair`), or an open rasterio dataset. Metadata with a geographic CRS (in any
  spelling: a `CRS`, `"EPSG:4326"` or `4326`) raises `ValueError`, since areas would be in
  squared degrees. Metadata without a geotransform (GCP-only, or no CRS and an
  identity transform) raises `ValueError` instead of silently reporting 1 unit² per
  pixel. Rectify or align the raster first, or pass the pixel size explicitly.
- Returns `total_pixels`, `valid_pixels`, `nodata_pixels`, `pixel_area_m2` (`None` if no
  `pixel_size`) and `classes`, a dict keyed by label. Each class has `value`, `pixels`,
  `percent` (of valid pixels), `area_m2` and `area_km2`. Boolean masks also get
  `changed_pixels`, `changed_percent`, `changed_area_m2` and `changed_area_km2`. All
  values are plain Python numbers (JSON-serializable).

---

## farq.georef

Metadata dicts use the rasterio profile format, optionally with `gcps` and `gcps_crs`.
Output pixels outside a source footprint are NaN for float outputs, or `nodata` for
integer outputs. Integer input without any nodata value is promoted to `float32`, so
those pixels can be NaN. Pass `nodata=` (e.g. `0` for RGB with a black border) to keep
the integer dtype. A `nodata` value in the metadata of float arrays is not applied: NaN is
the marker.

### GCP handling

```text
has_gcps(source) -> bool
read_gcps(path) -> (list[GroundControlPoint], CRS | None)
make_gcps(pixels, coords, *, ids=None) -> list[GroundControlPoint]
gcp_residuals(gcps, order=1) -> GCPResiduals
```

- `has_gcps` accepts a path, an open rasterio dataset or a metadata dict.
- `read_gcps` raises `ValueError` if the file has no GCPs.
- `make_gcps` builds GCPs from `pixels` as `(row, col)` pairs (GDAL corner convention,
  fractions allowed) and `coords` as `(x, y)` or `(x, y, z)` map coordinates. Default IDs
  are `"1"`, `"2"`, and so on.
- `gcp_residuals` fits a least-squares pixel-to-map polynomial (order 1, 2 or 3 needs at
  least 3, 6 or 10 GCPs) and raises `ValueError` for collinear or degenerate GCPs. It
  warns when the fit is exactly determined.

`GCPResiduals` is a frozen dataclass. Distances are in GCP map units:

- `order` and `ids`
- `residuals` `(N, 2)`, `errors` `(N,)` and `loo_errors` `(N,)`: leave-one-out errors,
  which expose a bad GCP that a least-squares fit hides. NaN where they cannot be
  computed.
- `rmse`, `max_error` and `dof` (number of GCPs minus number of coefficients)
- `pixel_size`: ground size of one pixel, and `rmse_pixels`: the RMSE in pixels
- `studentized` `(N,)`: externally studentized residuals (residual divided by the noise
  level estimated without that GCP, corrected for leverage). NaN when `dof < 2`.
- `.outliers(threshold=None, *, alpha=0.05)`: indices of suspicious GCPs, worst first.
  By default each GCP's studentized residual is tested against an F distribution with
  a Bonferroni correction, so on clean GCPs the chance of any false alarm is about
  `alpha`. Needs `dof >= 2` (returns `[]` otherwise). An explicit `threshold` flags
  leave-one-out errors above that distance in map units instead.

### Rectification

```text
georeference(array, gcps, crs, *, dst_crs=None, resolution=None, method="polynomial",
             order=None, resampling="bilinear", nodata=None) -> (ndarray, dict)
rectify(path, out_path=None, *, bands=None, gcps=None, gcps_crs=None, dst_crs=None,
        resolution=None, method="polynomial", order=None, resampling="bilinear",
        nodata=None) -> (ndarray, dict)
```

Both functions warp a GCP-referenced image onto a regular north-up grid.

- `method="polynomial"` with `order` 1 (default, affine), 2 or 3, or `method="tps"` (thin
  plate spline, which passes exactly through every GCP).
- `resolution`: output pixel size in `dst_crs` units. By default it is estimated from
  the GCPs. Use a coarser value with `resampling="average"` to downsample.
- `georeference` takes an in-memory array and `crs` (the CRS of the GCP map coordinates).
- `rectify` reads a file and uses its GCPs, or `gcps`/`gcps_crs` if given. It streams
  from disk when it can. `bands` is a 1-based index (which returns 2-D output) or a
  list. `out_path` also writes a compressed, tiled GeoTIFF.
- Both return `(array, metadata)`. The metadata contains `crs`, `transform`, `width`,
  `height`, `count`, `dtype`, `nodata` and `driver`.

### Alignment

<a id="align"></a>

```text
align(array, meta, reference_meta, *, resampling="bilinear", nodata=None,
      method="polynomial", order=None) -> (ndarray, dict)
```

Reprojects `array` onto the grid defined by `reference_meta` (`crs`, `transform`,
`width`, `height`). `meta` may be affine (`crs` + `transform`) or GCP-referenced. A
GCP-referenced input is warped in a single resampling step.

<a id="align_pair"></a>

```text
align_pair(before, before_meta, after, after_meta, *, target="before", dst_crs=None,
           resolution=None, resampling="bilinear", nodata=None, method="polynomial",
           order=None) -> (before_aligned, after_aligned, metadata)
```

Puts two rasters, 2-D or `(bands, rows, cols)` of any size, onto one common grid cropped
to their overlap. This is the step to run before any pixel-wise comparison.

- `target`:
  - `"before"` or `"after"`: keep that raster's grid. It is only cropped, never
    resampled.
  - `"coarsest"`: the most robust choice across sensors or flights.
  - `"finest"`: picks the pixel size per axis and snaps to the grid of the raster that
    provides it.
- `dst_crs`: common CRS (default: the CRS of the target raster). `resolution` overrides
  the pixel size.
- GCP-referenced inputs are rectified directly (`method`, `order` as in `georeference`).
- If the two outputs would have different dtypes or different nodata values, both are
  promoted to float with NaN as nodata.
- `metadata` is the common grid profile. Its `count` is that of `before`.
- Raises `ValueError` if the rasters do not overlap by at least one pixel.

### Co-registration

```text
coregister(reference, moving, *, upsample=10, window=True, whitening=0.5, refine=1)
    -> (dy, dx)
apply_shift(array, shift, *, order=1) -> ndarray
```

- `coregister` estimates the sub-pixel **translation** that aligns `moving` with
  `reference`. It uses FFT phase correlation with upsampled-DFT refinement
  (Guizar-Sicairos et al. 2008).
  - Inputs must already be on the same grid (run `align_pair` first). Multi-band inputs
    are averaged into one band, and NaN is filled with the mean. At least 8×8 pixels and
    at least 10 % valid pixels are needed.
  - `upsample=10` gives 0.1 px precision. `whitening` ranges from 0 (cross-correlation)
    to 1 (pure phase correlation).
  - Returns `(dy, dx)` in pixels, ready for `apply_shift`. If `moving` is `reference`
    displaced by `(sy, sx)`, the result is `(-sy, -sx)`.
  - Rotation, scale and local distortion are not modelled. For huge mosaics, estimate
    the shift on a downsampled copy or a crop.
- `apply_shift` translates a 2-D or `(bands, rows, cols)` array with spline interpolation
  (`order` 0-5). It returns float (`float32` for integer input), with NaN for pixels
  shifted in from outside or interpolated from NaN.

### Pixel geometry

```text
pixel_size(meta) -> (xres, yres)
pixel_area(meta) -> float
```

- `pixel_size` returns the positive pixel width and height in CRS units, accounting for
  rotation. For GCP-only metadata, it estimates them from an affine fit through the GCPs.
- `pixel_area` returns the area of one pixel in squared CRS units, and raises
  `ValueError` for a missing or geographic CRS.

---

## farq.masking

Cloud, shadow, snow and quality masks from the quality bands of Landsat Collection 2,
Sentinel-2 L2A and HLS v2.0. Guide: [Cloud and quality masking](masking.md).

- Masks are boolean arrays where **`True` means masked** (unusable).
- Quality bands are integer arrays. Float quality bands (e.g. read with `masked=True`)
  are accepted when every finite value is a whole number. NaN and masked entries are
  nodata and always come out masked.
- For a change pair, mask the **union** of both dates (`combine_masks`).
- Inputs are never modified, and no `RuntimeWarning` is emitted.

### Building masks

```text
landsat_qa_mask(qa_pixel, *, cloud=True, shadow=True, cirrus=True, snow=False,
                dilated=True, fill=True, water=False, min_confidence=None) -> ndarray[bool]
landsat_radsat_mask(qa_radsat, *, sensor="oli", bands=None, terrain_occlusion=True,
                    dropped_pixel=True) -> ndarray[bool]
sentinel2_scl_mask(scl, *, classes=DEFAULT_S2_BAD_CLASSES, target_shape=None)
    -> ndarray[bool]
sentinel2_cloud_probability_mask(probability, threshold=50, *, target_shape=None)
    -> ndarray[bool]
hls_fmask_mask(fmask, *, cloud=True, adjacent=True, shadow=True, snow=False, water=False,
               cirrus=False, aerosol="high", fill=True) -> ndarray[bool]
decode_bits(qa, bit, width=1) -> ndarray
decode_landsat_qa(qa_pixel) -> dict[str, ndarray]
```

- `landsat_qa_mask` decodes `QA_PIXEL` (Landsat 4-9). The single-bit flags are
  high-confidence detections. `min_confidence` (`"low"`, `"medium"`, `"high"` or 1-3)
  also masks pixels whose cloud confidence is at least that level. `"low"` masks almost
  every clear pixel, because CFMask gives low confidence to most of them.
- `landsat_radsat_mask` decodes `QA_RADSAT`. `sensor` is `"oli"` (Landsat 8/9), `"tm"`
  (4/5) or `"etm"` (7). `bands` limits saturation to some band numbers (e.g. `[3, 5]`);
  `None` uses every band.
- `sentinel2_scl_mask` masks Scene Classification classes. `classes` takes codes,
  `SCLClass` members or names from `SCL_NAMES` (`"cloud_high"`, `"snow_ice"`, ...).
  The default `DEFAULT_S2_BAD_CLASSES` is no data, saturated/defective, cloud shadow,
  cloud medium and high probability and thin cirrus (0, 1, 3, 8, 9, 10). Values outside
  0-11 raise `ValueError`.
- `sentinel2_cloud_probability_mask` masks `MSK_CLDPRB` pixels with
  `probability >= threshold` (percent).
- `target_shape=(rows, cols)` repeats a 20 m or 60 m mask onto the 10 m grid of the same
  tile (exact integer factor; see `upsample_mask`).
- `hls_fmask_mask` decodes the HLS v2.0 `Fmask`. `aerosol` masks pixels whose aerosol
  level is at least `"low"`, `"moderate"` or `"high"` (or 1-3); `None` ignores it.
  `fill=True` masks the fill value 255.
- `decode_bits` returns `(qa >> bit) & (2**width - 1)` in the smallest unsigned dtype
  that holds it. `decode_landsat_qa` returns every `QA_PIXEL` field: the booleans
  `fill`, `dilated_cloud`, `cirrus`, `cloud`, `cloud_shadow`, `snow`, `clear`, `water`
  and the 0-3 confidences `cloud_confidence`, `cloud_shadow_confidence`,
  `snow_ice_confidence`, `cirrus_confidence`.

### Combining and applying masks

```text
buffer_mask(mask, pixels=None, *, distance=None, pixel_size=None) -> ndarray[bool]
combine_masks(*masks) -> ndarray[bool]
upsample_mask(mask, target_shape) -> ndarray
apply_mask(data, mask, *, fill=nan) -> ndarray
clear_fraction(mask, valid=None) -> float
valid_overlap(mask_before, mask_after, *, footprint=None) -> MaskOverlap
```

- `buffer_mask` grows a 2-D mask by a radius in `pixels`, or by a ground `distance`
  with `pixel_size` (a number, an `(x, y)` pair, an `affine.Affine` or rasterio
  metadata). Give exactly one of the two. Every pixel whose centre is within the radius
  of a masked pixel centre is masked (an exact Euclidean distance transform, so the cost
  does not grow with the radius). A `distance` with metadata in a geographic CRS raises
  `ValueError`.
- `combine_masks` is the union of masks of the same shape; `None` entries are skipped.
- `upsample_mask` repeats each pixel by an exact integer factor. A `target_shape` that is
  not an integer multiple raises `ValueError`.
- `apply_mask` returns a float copy of `(rows, cols)` or `(bands, rows, cols)` data with
  masked pixels set to `fill` (NaN). The mask is `(rows, cols)` (applied to every band)
  or has the shape of `data`. Output is `float32` for integers of up to 16 bits and for
  `float32`, otherwise `float64`.
- `clear_fraction` returns the fraction of (`valid`) pixels that are not masked, or NaN
  if there are none.
- `valid_overlap` returns `MaskOverlap(valid, n_valid, n_total, fraction, before_clear,
  after_clear)`, a `NamedTuple`. `valid` is `True` where both dates are usable; pass it
  as `valid=` to `change_summary` or `classify_change`.

### Scaling digital numbers

```text
landsat_c2_scale(dn, kind="sr", *, fill=0) -> ndarray
sentinel2_l2a_scale(dn, offset=-1000, quantification=10000, *, nodata=0) -> ndarray
```

- `landsat_c2_scale`: Collection 2 Level-2 surface reflectance (`kind="sr"`,
  `DN * 0.0000275 - 0.2`) or surface temperature in kelvin (`kind="st"`,
  `DN * 0.00341802 + 149.0`). Not for Collection 1.
- `sentinel2_l2a_scale`: `(DN + offset) / quantification`. Products from processing
  baseline 04.00 (25 January 2022) onwards have `BOA_ADD_OFFSET = -1000`; pass
  `offset=0` for older products.
- Both return `float32` (`float64` for 32/64-bit or `float64` input) with the fill or
  nodata digital number (`None` disables it) and NaN inputs as NaN. Values are not
  clipped.

### Constants

`LandsatQA` (`QA_PIXEL` bit positions), `Confidence` (`NONE`, `LOW`, `MEDIUM`, `HIGH` =
0-3), `SCLClass` (SCL classes 0-11) and `SCL_NAMES` (class to name), `HLSFmask` (`Fmask`
bit positions) are `IntEnum`s or dicts. `DEFAULT_S2_BAD_CLASSES` is a `frozenset` of
`SCLClass` members.

---

## farq.radiometry

Relative radiometric normalization between dates or flights, and calibrated IR-MAD
change detection. Guide: [Radiometric normalization](radiometry.md).

- Stacks are `(bands, rows, cols)`; a 2-D array is one band.
- NaN, ±inf, `nodata` and masked entries are invalid: they are excluded from every fit
  and come out as NaN.
- Outputs are `float32` for 8/16-bit integer and `float32` inputs, otherwise `float64`.
  Statistics are accumulated in `float64`. Results are deterministic.
- Normalize the later image to the earlier one (`source=after`, `reference=before`), so
  that change magnitudes stay in "before" units.

### Normalization

```text
histogram_match(source, reference, *, valid=None, n_quantiles=None, nodata=None)
    -> ndarray
linear_normalize(source, reference, *, mask=None, method="ols", nodata=None)
    -> NormalizationResult
pif_normalize(source, reference, *, method="irmad", regression="orthogonal", valid=None,
              min_prob=0.9, n_sigma=2.0, percentile=25.0, min_pixels=50, max_iter=50,
              tol=1e-06, nodata=None) -> NormalizationResult
```

- `histogram_match` maps each band through the empirical CDFs (quantile mapping), so
  the output takes the reference's histogram. The images need not be co-registered or
  the same size. `valid` (same-shape images only) selects the pixels that build both
  distributions. `n_quantiles` (at least 2) gives a smoother piecewise-linear mapping.
  It forces the distributions to agree, so it can attenuate real change that covers a
  large part of the scene.
- `linear_normalize` fits a per-band gain and offset on the `mask` pixels (default: all
  valid pixels). `method`: `"ols"`, `"orthogonal"` (total least squares, noise in both
  images), `"theil_sen"` (robust to about 29 % outliers) or `"mean_std"`.
- `pif_normalize` selects pseudo-invariant pixels automatically, then fits them with
  `regression`. `method`:
  - `"irmad"` (default): IR-MAD no-change probability above `min_prob`. Best with three
    or more bands.
  - `"pca"`: within `n_sigma` robust standard deviations of each band's major axis,
    refitted iteratively. Suits one or two bands.
  - `"percentile"`: the `percentile` % of pixels with the smallest robust Theil–Sen
    residuals. Breaks down when more than about 29 % of the scene changed.

  `valid` limits the pixels allowed as PIFs. Fewer than `min_pixels` PIFs raises
  `ValueError`.

`NormalizationResult` is a `NamedTuple`. The model is
`normalized[i] = gains[i] * source[i] + offsets[i]`:

- `normalized`: `source` on the reference's scale, NaN where `source` is invalid
- `gains`, `offsets`: `float64` arrays of shape `(bands,)`
- `invariant_mask`: bool `(rows, cols)`, the pixels the fit used
- `r2`, `rmse`: per band, on the fit pixels (`rmse` in reference units). A low `r2`
  means no linear relation exists; use `histogram_match` instead.
- `n_invariant`: number of fit pixels
- `.apply(image, *, nodata=None)`: applies the same gains and offsets to another image with
  the same bands, for example the full-resolution raster after a fit on a decimated read

### IR-MAD

```text
irmad(before_stack, after_stack, *, max_iter=50, tol=1e-06, valid=None, nodata=None)
    -> IRMADResult
irmad_change(before_stack, after_stack, *, alpha=0.01, max_iter=50, tol=1e-06,
             valid=None, nodata=None) -> ndarray[bool]
```

- `irmad` is Nielsen's (2007) iteratively reweighted Multivariate Alteration Detection.
  Its statistic does not change under any per-band gain or offset, so exposure and
  calibration differences are not detected as change. Inputs are co-registered
  `(bands, rows, cols)` stacks (or 2-D images); a pixel is invalid if any band in either
  stack is. `valid` limits the pixels used to estimate the statistics; outputs are still
  computed for every valid pixel. `max_iter=1` gives plain MAD.
- `IRMADResult` is a `NamedTuple`:
  - `mad_variates`: `(bands, rows, cols)`, ordered by ascending canonical correlation,
    so `mad_variates[0]` carries the most change
  - `chi2`: `(rows, cols)` change statistic, chi-square distributed with `bands` degrees
    of freedom on unchanged pixels
  - `no_change_prob`: its p-value
  - `canonical_correlations` (ascending), `n_iter`, `converged`
- `irmad_change` flags pixels with `chi2 > chi2.ppf(1 - alpha, bands)`, so about a
  fraction `alpha` of the unchanged pixels are flagged. It returns a bool `(rows, cols)`
  mask, `False` where invalid.
- **Calibration.** The published scheme uses the *weighted* MAD variances, which
  under-estimate the no-change variances, so the false-alarm rate far exceeds `alpha`
  (over 50 % at `alpha=0.01` in simulations). Farq multiplies the statistic by the exact
  consistency factor for this weighting, so `alpha` is the false-alarm rate under
  Gaussian no-change noise. Misregistration and other heavy-tailed differences still
  raise it.
- Raises `ValueError` with too few valid pixels (at least `max(10, 2 * bands + 2)`), a
  singular band covariance (constant or duplicated bands), or when `after` is an exact
  linear function of `before`.

---

## farq.elevation

DEM differencing, DEM co-registration, cut/fill volumes and stockpile volumes. Guide:
[Elevation change and volumes](elevation.md).

- DEMs are 2-D arrays (a `(1, rows, cols)` stack is accepted). NaN, ±inf, masked
  entries and `nodata` are invalid: NaN in every output and never counted in volumes.
- **Elevations are in metres**; volumes are in m³ and areas in m².
- Change is `after - before`: positive is fill (deposition), negative is cut (erosion).
- `meta` is a rasterio metadata dict (with `crs` and `transform`), an open dataset, an
  `affine.Affine` (assumed metres), a pixel size in metres or an `(xres, yres)` pair.
  Horizontal units of projected CRSs are converted to metres (e.g. US survey feet).
  **Geographic CRSs, metadata without a CRS and GCP-only metadata raise `ValueError`.**
- Inputs are never modified, and no `RuntimeWarning` is emitted.

### Terrain derivatives

```text
slope(dem, meta, *, units="degrees", nodata=None) -> ndarray
aspect(dem, meta, *, nodata=None) -> ndarray
hillshade(dem, meta, *, azimuth=315.0, altitude=45.0, z_factor=1.0, nodata=None)
    -> ndarray
```

Central differences with the true pixel spacing (including rotated or non-square
pixels), one-sided at edges and next to nodata. `slope` units are `"degrees"`,
`"radians"` or `"percent"`. `aspect` is in degrees clockwise from grid north in
`[0, 360)` (NaN on flat pixels). `hillshade` returns illumination in `[0, 1]`.

### Co-registration

```text
vertical_offset(before_dem, after_dem, *, stable_mask=None, method="median", nodata=None,
                min_pixels=100) -> VerticalOffset
coregister_dem(reference_dem, dem, meta, *, stable_mask=None, nodata=None,
               max_iterations=10, tolerance=0.01, min_slope=2.0, max_slope=70.0)
    -> DEMCoregistration
shift_dem(dem, meta, dx, dy, dz=0.0, *, nodata=None) -> ndarray
```

- `vertical_offset` returns `VerticalOffset(offset, nmad)`, a `NamedTuple`: the vertical
  bias of `after_dem` over stable ground (subtract it from `after_dem`) and the NMAD of
  the differences. `method` is `"median"` or `"nmad_trimmed"` (drop differences beyond
  3 NMAD, then average). Fewer than `min_pixels` stable pixels raise `ValueError`.
- `coregister_dem` implements Nuth & Kääb (2011): it fits the horizontal shift from
  `dh / tan(slope)` against aspect on stable terrain between `min_slope` and
  `max_slope` degrees, iterates until the update is below `tolerance` pixels, then takes
  the median vertical bias. It models a **translation only**, and raises `ValueError`
  when the terrain is flat or planar. It warns if it did not converge.
- `DEMCoregistration` (frozen dataclass): `dx`, `dy`, `dz` (metres east, north, up of
  `dem` relative to the reference), `dem` (the co-registered DEM on the reference grid),
  `iterations`, `converged`, `nmad_before`, `nmad_after`, `n_pixels`.
- `shift_dem` removes a known offset (bilinear resampling), for example one estimated on
  a crop.

### DEM of difference and level of detection

```text
elevation_change(before_dem, after_dem, *, nodata=None) -> ndarray
level_of_detection(sigma_before, sigma_after=None, *, confidence=0.95) -> float | ndarray
significant_change(dod, lod) -> ndarray
```

- `elevation_change` returns `after - before` (`float32` unless an input is `float64` or
  a wide integer), NaN where either DEM is invalid.
- `level_of_detection` is `t · sqrt(σ_before² + σ_after²)`, with `t` the two-sided
  normal quantile (1.96 at 95 %). `sigma_after` defaults to `sigma_before`; pass
  `sigma_after=0` when `sigma_before` is already the error of the difference. Sigmas can
  be per-pixel arrays.
- `significant_change` returns a float copy of `dod` with `|dh| <= lod` set to 0, NaN
  where `dod` or `lod` is NaN.

### Volumes

```text
volume_change(dod, meta, *, lod=None, mask=None, sigma=None, correlation_length=None)
    -> VolumeResult
stockpile_volume(dem, mask, meta, *, base="plane", ring_width=1, nodata=None, sigma=None,
                 correlation_length=None) -> StockpileResult
```

- `volume_change` sums `fill = A · Σ dh` over `dh > lod` and `cut = A · Σ |dh|` over
  `dh < -lod` (every non-zero change without `lod`), inside the optional `mask`. A
  number for `meta` is the pixel side in metres.
- `sigma` (one-sigma DoD error, scalar or per pixel) adds uncertainties: uncorrelated
  `σ · A · sqrt(n)`, or with `correlation_length` the spatially correlated estimate of
  Rolstad et al. (2009). The larger of the two is reported.
- `VolumeResult` (frozen dataclass, plain Python numbers): `cut_m3`, `fill_m3` (both
  positive), `net_m3` (`fill - cut`), `cut_area_m2`, `fill_area_m2`,
  `unchanged_area_m2`, `valid_area_m2`, `nodata_area_m2` (check it: gaps are left out
  of the volumes), `pixel_area_m2`, and `cut_uncertainty_m3`, `fill_uncertainty_m3`,
  `uncertainty_m3` (`None` without `sigma`). `.to_dict()` gives a standard-JSON dict
  (NaN and ±inf become `None`).
- `stockpile_volume` measures a pile from one survey. The base surface is fitted to the
  toe ring (valid pixels within `ring_width` pixels outside the footprint `mask`).
  `base` is `"plane"` (least-squares plane, default), `"lowest"`, `"mean"`, a fixed
  elevation, or an array on the same grid (e.g. a survey of the empty pad).
- `StockpileResult` (frozen dataclass): `volume_m3`, `below_base_m3`, `net_m3`,
  `area_m2`, `max_height_m`, `base` (`"plane"`, `"lowest"`, `"mean"`, `"surface"` for
  an array, or the number), `base_elevation_m`, `base_slope_deg`, `base_rmse_m` (misfit
  of the toe ring; large means the base is uncertain), `missing_area_m2`,
  `uncertainty_m3`, and `.to_dict()` (NaN, e.g. `base_rmse_m` of a fixed base, becomes
  `None`).

---

## farq.tiling

Out-of-core processing: inputs are read block by block from disk and results are
written block by block, so memory depends on the block size, not on the raster size.
Guide: [Processing rasters larger than memory](tiling.md).

- **All inputs of one call must be on one grid** (CRS, transform and size), otherwise
  `ValueError`. Align them first (`align_pair`, `align` or `gdalwarp`).
- Inputs are read like `read(path, masked=True)`: nodata, alpha bands and internal masks
  become NaN, and integers become `float32` (`float64` for 32/64-bit).
- Outputs are tiled (256 × 256), deflate-compressed GeoTIFFs (BigTIFF when needed) on
  the input grid, written **atomically**. Extra keyword arguments or `profile=` override
  the creation options.
- `block_size` (int or `(rows, cols)`, default 1024) is rounded down to a multiple of the
  input's internal tile size. `n_jobs` threads (`-1` for all CPUs) process blocks in
  parallel; results do not depend on `n_jobs`. `progress(done, total)` is called after
  each block.

### `detect_changes_file`

```text
detect_changes_file(before_path, after_path, out_path, *, method="difference",
                    threshold="otsu", k=2.0, percentile=95.0, min_size=0, connectivity=8,
                    fill_holes=False, bands=1, sample_size=1000000, seed=0,
                    magnitude_path=None, pixel_size=None, block_size=1024, n_jobs=1,
                    progress=None, profile=None) -> dict
```

The out-of-core `detect_changes`. `method` is `"difference"`, `"ratio"`,
`"normalized_difference"` or `"cva"` (use `bands=[...]` or `bands=None`); `"pca"` and
`"irmad"` need whole-image statistics and raise `ValueError`.

- A threshold rule is applied globally, estimated from a reproducible random sample of
  `sample_size` valid pixels (`seed`; `None` uses every valid pixel). With a numeric
  threshold, or when the sample covers every valid pixel, the mask equals
  `detect_changes` on the full arrays, pixel for pixel.
- `min_size` and `fill_holes` are exact across block borders.
- `out_path` is a `uint8` mask: 1 = change, 0 = no change, 255 (`CHANGE_NODATA`, the
  nodata value) where either date is invalid. `magnitude_path` also writes the float
  magnitude.
- Returns a JSON-serializable dict: `method`, `threshold`, `threshold_method`,
  `threshold_exact`, `sampled_pixels`, `total_pixels`, `valid_pixels`,
  `nodata_pixels`, `changed_pixels`, `changed_percent`, `pixel_area_m2`,
  `changed_area_m2`, `changed_area_km2`. Areas come from the geotransform or
  `pixel_size` (as in `change_summary`) and are `None` without one. A geographic CRS
  raises `ValueError` before any processing unless `pixel_size` is given in metres.

### `index_file`

```text
index_file(index_name, band_paths, out_path, *, block_size=1024, n_jobs=1, progress=None,
           dtype=None, profile=None, **kwargs) -> dict
```

Computes any farq index (`"ndvi"`, `"ndwi"`, `"mndwi"`, `"evi"`, `"savi"`, `"ndbi"`,
`"nbr"`, `"ndmi"`, `"vari"`, `"exg"`, `"exr"`, `"exgr"`, `"gli"`, `"ngrdi"`, `"tgi"`)
file to file. `band_paths` maps band names (`"blue"`, `"green"`, `"red"`, `"nir"`,
`"swir1"`, `"swir2"`) to a path (band 1) or a `(path, band)` pair. `**kwargs` go to the
index function (e.g. `clip=False`, `reflectance_scale=10000`). The output is float with
NaN nodata and equals the in-memory index. Returns the output metadata.

### `map_blocks`

```text
map_blocks(func, inputs, out_path, *, bands=1, block_size=1024, overlap=0, dtype=None,
           nodata=<NaN for float output>, masked=True, n_jobs=1, progress=None,
           **profile) -> dict
```

- Calls `func(*arrays)` with the same window of each input and writes the result, which
  must be `(rows, cols)` or `(bands, rows, cols)` with the spatial shape of the inputs
  and the same band count for every block. `bool` results are written as `uint8`.
- `inputs` is a sequence of paths or `(path, bands)` pairs (a single path is accepted).
  `bands` is an int (2-D arrays), a sequence, or `None` for all bands (3-D arrays).
- `overlap` adds a halo of that many pixels around each block. With a halo at least the
  radius of a neighbourhood operation, the result equals the full-raster one.
- `dtype` defaults to the dtype of the first result. `nodata` defaults to NaN for float
  output and none for integer output; writing NaN to an integer dtype requires a
  `nodata` value. `masked=False` passes raw values.
- `func` must be thread-safe when `n_jobs > 1`.
- Returns the output metadata (`driver`, `dtype`, `nodata`, `width`, `height`, `count`,
  `crs`, `transform`), usable with `write`. Raises `FileNotFoundError`, `IndexError`,
  `ValueError` or `TypeError`.

### `summarize_file`

```text
summarize_file(path, band=1, *, bins=50, masked=True, block_size=1024, n_jobs=1,
               progress=None) -> dict | list[dict]
```

Streaming statistics: a dict for an int `band`, otherwise a list with one dict per band
(`None` = all bands). Keys: `band`, `shape`, `size`, `valid`, `nan`, `inf`, `min`, `max`,
`range`, `mean`, `std` (population), `variance`, `sum`, `percentages` and `histogram`
(`counts`, `bin_edges`; omitted with `bins=None`, which also skips the second pass). Mean
and variance agree with `stats` to rounding; counts, extremes and the histogram are
exact. Percentiles and the median are not computed.

### `iter_windows` and `Block`

```text
iter_windows(width, height, block_size=1024, overlap=0, *, align_to=None)
    -> Iterator[Block]
```

Splits a raster into blocks in row-major order. `align_to` is the file's internal block
shape (e.g. `dataset.block_shapes[0]`). Each `Block` is a `NamedTuple`:

- `number`: position in row-major order (0, 1, 2, ...)
- `read_window`: the block plus up to `overlap` pixels on each side, clipped to the raster
- `write_window`: the part this block is responsible for (the write windows tile the
  raster exactly)
- `inner`: `(row_slice, col_slice)` selecting the `write_window` part of an array read
  with `read_window`

---

## farq.vector

Polygonize change masks and class maps, and export them to GIS formats. Guide:
[Vector export](vector.md).

- GeoJSON needs no extra dependency. GeoPackage, Shapefile and FlatGeobuf need pyogrio
  (preferred) or fiona: `pip install "farq[vector]"` installs pyogrio and shapely.
- NaN, ±inf, `nodata` and masked entries never become polygons. Boolean masks give
  polygons for `True` only.
- Polygons follow pixel edges. Areas and perimeters are in CRS units (metres for UTM).
  **Rasters in a geographic CRS, and rasters without a geotransform (identity transform
  without a CRS, or GCP-only), raise `ValueError`.**

### `polygonize`

```text
polygonize(data, meta, *, valid=None, connectivity=4, values=None, min_area=None,
           simplify=None, preserve_topology=False, labels=None, nodata=None) -> list[dict]
```

- `data` is a 2-D (or `(1, rows, cols)`) boolean mask or numeric class map. `meta` is a
  rasterio metadata dict, an open dataset or an `affine.Affine` (units assumed metres).
- `values` keeps only some values. `labels` names them (e.g. `CHANGE_LABELS`); boolean
  masks are labelled `"changed"` and other unlabelled values `str(value)`.
- `connectivity=4` (default) gives OGC-valid polygons. With `8`, diagonal neighbours
  join one polygon whose ring touches itself (invalid by OGC rules).
- `min_area` drops polygons smaller than that area (m²). `simplify` is a
  Douglas–Peucker tolerance in CRS units, applied per ring (neighbouring polygons can
  develop gaps or overlaps). `preserve_topology=True` simplifies all polygons together
  with `shapely.coverage_simplify` (needs `shapely>=2.1` and `connectivity=4`).
- Returns GeoJSON-like features (`type`, `id`, `geometry`, `properties`) in the raster
  CRS. Properties: `value`, `label`, `pixel_count`, `area_m2`
  (`pixel_count` × pixel area), `perimeter_m` (pixel-edge length, holes included),
  `centroid_x`, `centroid_y`. They describe the pixel region before simplification.

### `to_geojson` and `write_vector`

```text
to_geojson(features, path=None, *, crs=None, to_wgs84=True, precision=None) -> dict
write_vector(features, path, *, crs, driver=None, layer=None, to_wgs84=None, engine=None)
    -> None
```

- `to_geojson` returns a `FeatureCollection` and, with `path`, writes it atomically as
  UTF-8. Coordinates are reprojected to WGS 84 lon/lat (RFC 7946) by default, so `crs`
  is required (a `CRS`, EPSG code, string, WKT or rasterio metadata).
  `to_wgs84=False` keeps the input coordinates and records the CRS in the legacy `"crs"`
  member. `precision` rounds coordinates. Rings follow the right-hand rule, and NaN
  property values become `null`.
- `write_vector` writes GeoJSON, GeoPackage, Shapefile or FlatGeobuf; `driver` is
  inferred from the extension (`.geojson`/`.json`, `.gpkg`, `.shp`, `.fgb`). `to_wgs84`
  defaults to `True` for GeoJSON only. An existing file is replaced as a whole, through
  a temporary file. Shapefile field names are shortened (`perim_m`, `pixels`,
  `mean_mag`, `max_mag`). `engine` is `"pyogrio"` or `"fiona"` (default: pyogrio if
  installed). Raises `ImportError` for a non-GeoJSON format without either library.

### `changes_to_vector`

```text
changes_to_vector(change, meta, path=None, *, values=None, labels=None, nodata=None,
                  valid=None, min_area=None, connectivity=4, simplify=None,
                  preserve_topology=False, driver=None, layer=None, to_wgs84=None,
                  engine=None) -> list[dict]
```

Polygonizes change-detection output and, with `path`, writes it (format from the
extension). `change` is:

- a `ChangeResult`: polygons of its `mask`, with `mean_magnitude` and `max_magnitude`
- a boolean mask: polygons labelled `"changed"`
- a class map from `classify_change` (also read back from a file): by default the
  `GAINED` and `LOST` regions, labelled with `CHANGE_LABELS`, with `CHANGE_NODATA`
  excluded

Returns the features in the raster CRS.

---

## farq.analysis

These functions take 2-D water masks (`True` or non-zero is water). Float masks may hold
NaN for unknown pixels (clouds, nodata), and masked entries of a `numpy.ma.MaskedArray`
are unknown too. Unknown pixels are never counted as water and are excluded from
coverage and change. `pixel_size` is in **metres**, as a number or as a
`(width, height)` pair. `connectivity` is `1` (4-neighbour, default) or `2`
(8-neighbour). This differs from `farq.change`, which uses `4`/`8`.

Shape metric definitions:

- `perimeter`: exact pixel-edge length, so a 3×3 block of 1 m pixels has a perimeter of
  12 m.
- `compactness`: 4π·area / perimeter². A square scores π/4.
- `elongation`: √(λ1/λ2) of the second-moment matrix.
- `orientation`: angle of the major axis in degrees in [-90, 90], counter-clockwise from
  the x (column) axis.

### `water_stats`

```text
water_stats(water_mask, pixel_size=30.0, calculate_shapes=False, *, connectivity=1) -> dict
```

Returns a dict with these keys:

- `total_area` (km²)
- `coverage_percent`: water pixels as a percentage of **valid** pixels
- `num_water_bodies`
- `mean_body_size` and `largest_body` (km²)

With `calculate_shapes=True`, it also returns `shape_metrics`, which holds
`mean_compactness`, `mean_elongation` and `body_metrics`. `body_metrics` is a list of
per-body dicts with `area` (km²), `pixel_count`, `perimeter` (**km**), `compactness`,
`elongation` and `orientation`.

### `water_change`

```text
water_change(mask1, mask2, pixel_size=30.0, min_change_area=None, *, connectivity=1) -> dict
```

Compares an earlier mask (`mask1`) with a later one (`mask2`). Returns:

- `gained_area`, `lost_area` and `net_change` (km²)
- `change_percent`: net change relative to the earlier water area, `inf` if there was no
  water before but some was gained
- `change_mask`: `int8`, 1 = gained, -1 = lost, 0 = no change
- `stable_water`: bool

`min_change_area` (m²) removes connected gain or loss patches smaller than that area.
NaN pixels in either mask are never counted as change.

### `get_water_bodies`

```text
get_water_bodies(water_mask, pixel_size=30.0, min_area=None, calculate_shapes=False, *,
                 connectivity=1) -> (labeled, characteristics)
```

Returns an `int32` label image (0 = background) and a dict `{label: {...}}` with `area`
(km²), `pixel_count` and, if `calculate_shapes=True`, `perimeter` (km), `compactness`,
`elongation` and `orientation`. `min_area` (m²) removes small bodies and relabels the
rest `1..n`.

### `calculate_shape_metrics`

```text
calculate_shape_metrics(water_body_mask, pixel_size=1.0) -> dict
```

Treats every non-zero pixel as one object. Returns `area` (squared `pixel_size` units),
`perimeter` (`pixel_size` units), `compactness`, `elongation` and `orientation`. The
default `pixel_size=1.0` gives lengths in pixels. An empty mask returns zeros, with an
elongation of 1.

---

## farq.ml

Rasters here are **`(rows, cols, bands)`**. Convert a `farq.read(..., band=None)` stack
with `np.moveaxis(stack, 0, -1)`. Non-finite values mark invalid pixels: they are
excluded from training and clustering and receive `fill_value` in predictions. Stochastic
functions take `random_state`.

### Features and classification

```text
extract_features(raster_data, indices=None, window_size=3) -> ndarray
train_classifier(features, labels, model_type="rf", test_size=0.2, random_state=42, *,
                 ignore_label=None, stratify=True, **model_params) -> (model, metrics)
predict_raster(model, features, batch_size=None, *, fill_value=-1) -> ndarray
```

- `extract_features` returns a `(rows, cols, n_features)` array with:
  1. the bands
  2. the `indices` layers: a dict or list of precomputed 2-D arrays, such as
     `{"ndwi": farq.ndwi(g, n)}`
  3. if `window_size > 1`, a NaN-aware moving-window mean and variance for each layer
     above

  `window_size=1` disables the window features.
- `train_classifier` trains a random forest (`model_type="rf"` is the only type). It
  accepts `(n_samples, n_features)` features with 1-D labels, or raster-shaped
  `(rows, cols, n_features)` features with `(rows, cols)` labels.
  - It drops samples with non-finite features, NaN labels or `ignore_label`.
  - The split is stratified when every class has at least two samples. `test_size=0`
    trains on all samples and leaves the metrics `None`.
  - `metrics` has `accuracy`, `confusion_matrix` (list of lists), `classification_report`,
    `classes`, `n_train`, `n_test` and `n_dropped`.
  - A random pixel split is spatially autocorrelated, so the accuracy is optimistic.
    Validate on a separate area.
- `predict_raster` returns `(rows, cols)` for raster features, or `(n_samples,)`.
  `batch_size` bounds memory, and pixels with any non-finite feature get `fill_value`.

### Model persistence

```text
save_model(model, filepath, metadata=None, *, compress=0) -> str
load_model(filepath, *, expected_sha256=None) -> (model, metadata)
class ModelIntegrityError(ValueError)
```

- `save_model` atomically writes a joblib file containing `{"model", "metadata",
  "farq_info"}` and a `<filepath>.sha256` sidecar in `sha256sum` format. It returns the
  SHA-256 hex digest. `farq_info` records the format version, the farq, scikit-learn,
  NumPy and Python versions, the model class and the save time.
- `load_model` reads the file into memory once, then hashes and unpickles **the same
  bytes**, so the file cannot change between the check and the load. It verifies those
  bytes against `expected_sha256` (if given) and the sidecar (if present), using a
  constant-time comparison. A mismatch, or a malformed or oversized sidecar, raises
  `ModelIntegrityError`. A file
  with no integrity data loads with a `UserWarning`. A scikit-learn version mismatch also
  warns. The returned `metadata` is your metadata plus a `"_farq"` entry. Plain joblib
  dumps of an estimator are accepted.

> **Security:** model files are pickles, and loading one can execute arbitrary code.
> Never load model files from untrusted sources. The sidecar cannot protect against an
> attacker who can replace both files. For that, pin `expected_sha256` with a hash
> obtained through a trusted channel.

### Change detection and augmentation

```text
detect_changes_ml(raster1, raster2, model=None, threshold=0.5, *, window_size=3,
                  batch_size=None) -> ndarray[bool]
augment_training_data(features, labels, augmentation_factor=2, random_state=42, *,
                      noise_level=0.1) -> (features, labels)
```

- `detect_changes_ml` returns a **2-D `(rows, cols)` boolean mask** for 2-D or
  `(rows, cols, bands)` input. It computes `|raster2 - raster1|` in float, so unsigned
  inputs cannot wrap around.
  - Without a model, a pixel changed if the magnitude exceeds `threshold`. For several
    bands, the magnitude is the Euclidean norm across bands.
  - With a model, features are extracted from the difference image
    (`extract_features(diff, window_size=window_size)`). For binary classifiers with
    `predict_proba`, the probability of `classes_[1]` is compared with `threshold`.
  - Non-finite pixels are unchanged.
- `augment_training_data` returns the **original samples followed by
  `augmentation_factor - 1` jittered copies**: `n_samples * augmentation_factor` rows in
  total. Each feature gets Gaussian noise with a standard deviation of `noise_level`
  times that feature's standard deviation. A local generator is used, and the global
  NumPy random state is not touched. Augment the training split only.

### Unsupervised water detection

```text
cluster_water_bodies(raster_data, method="kmeans", n_clusters=2, water_index=None, *,
                     random_state=42, water_high=None, sample_size=None, **kwargs)
    -> (labels, metadata)
analyze_water_clusters(cluster_labels, water_cluster, pixel_size=30.0, *, connectivity=1)
    -> dict
optimize_clustering(raster_data, water_index=None, method="kmeans", param_grid=None, *,
                    random_state=42, sample_size=10000) -> (best_params, info)
```

- `cluster_water_bodies` clusters standardized pixels with k-means or DBSCAN (`**kwargs`
  go to scikit-learn). It returns `int32` labels, with -1 for invalid pixels and DBSCAN
  noise.
  - The water cluster is the one with the highest mean `water_index` (water is high for
    NDWI and MNDWI). Without an index, it is the highest band mean for 2-D input or the
    lowest for multi-band reflectance. Override with `water_high`.
  - `sample_size` fits k-means on a subset of pixels.
  - `metadata` has `water_cluster` (`None` if DBSCAN found nothing), `cluster_means`,
    `n_clusters`, `n_invalid` and `water_rule`. k-means also adds `cluster_centers` and
    `inertia`, and DBSCAN adds `noise_points`.
- `analyze_water_clusters` computes statistics of the connected bodies of
  `water_cluster`. **Areas are in m² and perimeters in m** (unlike `farq.analysis`). It
  returns `num_water_bodies`, `total_water_area`, `mean_water_body_area`,
  `max_water_body_area`, `min_water_body_area`, `mean_perimeter`, `mean_compactness`,
  and the lists `water_body_sizes`, `water_body_perimeters` and
  `water_body_compactness`.
- `optimize_clustering` grid-searches parameters by silhouette score on a fixed subsample.
  For DBSCAN it also subtracts the noise fraction. The default grids are k-means
  `{"n_clusters": [2, 3, 4, 5]}` and DBSCAN `{"eps": [0.1, 0.2, 0.3, 0.4],
  "min_samples": [5, 10, 15, 20]}`. It returns `best_params` (`None` if no combination
  gives two or more clusters) and `info` with `results` and `best_score`.

---

## farq.visualization

Every function returns a `matplotlib.figure.Figure`. None of them call `plt.show()` or
close other figures. Single-panel functions take `ax=` and two-panel functions take
`axes=` (two axes) to draw into an existing figure, in which case `figsize` is ignored.
NaN and ±inf are blank in images, skipped in histograms and transparent in RGB
composites. Histograms and percentile stretches use a reproducible sample of at most
`max_samples` pixels (default 1,000,000).

```text
plot(data, title=None, cmap="viridis", figsize=(10, 8), vmin=None, vmax=None,
     colorbar_label=None, reflectance_scale=None, *, ax=None, colorbar=True) -> Figure
compare(data1, data2, title1=None, title2=None, cmap="viridis", figsize=(15, 6),
        vmin=None, vmax=None, colorbar_label=None, reflectance_scale=None, *,
        axes=None, colorbar=True) -> Figure
changes(data, title=None, cmap="RdYlBu", figsize=(10, 8), vmin=None, vmax=None,
        symmetric=True, colorbar_label="Change", reflectance_scale=None, *, ax=None,
        colorbar=True) -> Figure
hist(data, bins=50, title=None, figsize=(10, 6), density=True, xlabel="Value",
     ylabel=None, alpha=0.6, reflectance_scale=None, *, ax=None,
     max_samples=1000000) -> Figure
distribution_comparison(data1, data2, title1=None, title2=None, bins=50,
                        figsize=(12, 6), density=True, xlabel="Value", ylabel=None,
                        alpha=0.6, reflectance_scale=None, *, axes=None,
                        max_samples=1000000) -> Figure
plot_rgb(red, green, blue, title=None, figsize=(10, 8), scale_factor=1.0, gamma=1.0,
         percentile=98.0, reflectance_scale=None, *, ax=None,
         max_samples=1000000) -> Figure
compare_rgb(rgb1, rgb2, title1=None, title2=None, figsize=(15, 6), scale_factor=1.0,
            gamma=1.0, percentile=98.0, reflectance_scale=None, *, axes=None,
            max_samples=1000000) -> Figure
```

- `plot`: a single 2-D raster (bool masks are fine).
- `compare`: two rasters of the same shape side by side on a **shared** colour scale.
- `changes`: a change map. With `symmetric=True`, missing limits are centred on 0.
- `hist`: histogram of the finite values of an array of any shape. With
  `density=False`, sampled counts are rescaled to the full array.
- `distribution_comparison`: two histograms with identical bin edges.
- `plot_rgb`: RGB composite from three 2-D bands or one `(3, rows, cols)` stack
  (`farq.plot_rgb(stack)`). Each band is divided by its `percentile`-th percentile
  (`None` disables the stretch), multiplied by `scale_factor`, clipped to [0, 1] and
  gamma-corrected (`gamma > 1` brightens dark tones). Pixels with any NaN band are
  transparent.
- `compare_rgb`: two composites. `rgb1` and `rgb2` are each a `(3, rows, cols)` stack or a
  sequence of three 2-D arrays `(red, green, blue)`.
- `reflectance_scale` divides the data before plotting. It must be non-zero.

---

## farq.utils

These functions are NaN-aware statistics with input validation. Some are named like
Python builtins (`farq.min`, `farq.max`, `farq.sum`). Reductions ignore NaN. With
`axis`, all-NaN slices give NaN without warnings. An empty array raises `ValueError`, and
so does an all-NaN array, except in `count_nonzero` and `unique`.

```text
stats(data, percentiles=(0, 25, 50, 75, 100), reflectance_scale=None, bins=50) -> dict
sum(data, axis=None)      mean(data, axis=None)      std(data, axis=None, ddof=0)
min(data, axis=None)      max(data, axis=None)       median(data, axis=None)
percentile(data, q, axis=None)
count_nonzero(data, axis=None)
unique(data, return_counts=False)
validate_array(array, name="array", allow_all_nan=False) -> None
```

- `stats` computes values over finite values only. It returns:
  - `min`, `max`, `mean`, `std`, `median`, `range`, `variance`
  - `skewness` and `kurtosis` (Fisher)
  - `percentiles`, a dict keyed by `str(p)`
  - the counts `non_zero`, `zeros`, `nan`, `inf` and `valid`
  - `shape`, `size` and `dtype`
  - `histogram` (`counts`, `bin_edges`) and `percentages`
  - `reflectance_stats`, if `reflectance_scale` is given
- `count_nonzero` follows NumPy and counts NaN as non-zero. `unique` sorts NaN last.
- `validate_array` raises `TypeError` for non-arrays and `ValueError` for empty or
  all-NaN arrays.

---

## Compatibility names

For compatibility with farq 0.1, these names are also available:

- `farq.plt`: `matplotlib.pyplot`
- `farq.Resampling`: `rasterio.enums.Resampling`
- `farq.os`: the `os` module

`farq.__version__` is `"0.3.0"`.
