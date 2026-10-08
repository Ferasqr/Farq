# API reference

Every function and class below is available as `farq.<name>` and also from its submodule
(for example, `farq.detect_changes` is `farq.change.detect_changes`). Signatures are
copied from the code. Parameters after `*` are keyword-only.

- [Key conventions](#key-conventions)
- [farq.core](#farqcore): read, write, resample
- [farq.indices](#farqindices): spectral and RGB indices
- [farq.change](#farqchange): change detection
- [farq.georef](#farqgeoref): GCPs, rectification, alignment, co-registration
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
| Units | `change_summary` reports areas in m² (squared CRS units) and km². `farq.analysis` takes `pixel_size` in metres and returns areas in **km²** and per-body perimeters in **km**. `ml.analyze_water_clusters` returns areas in **m²** and perimeters in **m**. |
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
               nodata=None) -> ChangeResult
```

Runs the whole change-detection pipeline: magnitude, then threshold, then cleanup.

- `method`: the magnitude is the absolute value of the signed measure.
  - `"difference"`: `|after - before|`
  - `"ratio"`: `|ln(after / before)|`, recommended for SAR
  - `"normalized_difference"`: `|(after - before) / (after + before)|`
  - `"cva"`: change-vector length
  - `"pca"`: `|PC1|` of the difference image

  `"cva"` and `"pca"` accept `(rows, cols)` or `(bands, rows, cols)` input. The other
  methods need 2-D input (a leading band axis of size 1 is accepted).
- `threshold`: `"otsu"`, `"std"` (mean + `k`·std), `"percentile"` (the `percentile`-th
  value) or a number. See [`compute_threshold`](#compute_threshold).
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
  - `.areas(pixel_size)`: squared CRS units

### `change_summary`

```text
change_summary(data, *, pixel_size=None, labels=None, nodata=None, valid=None) -> dict
```

Pixel counts, percentages and areas for a boolean mask or a class map.

- `pixel_size` can be a number (square pixels), an `(x, y)` pair, an `affine.Affine`, a
  rasterio meta/profile dict with a real geotransform (such as the metadata returned by
  `align_pair`), or an open rasterio dataset. A warning is raised when the metadata
  reports a geographic CRS. Metadata without a geotransform (GCP-only, or no CRS and an
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

`farq.__version__` is `"0.2.0"`.
