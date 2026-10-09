# Changelog

All notable changes to Farq are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.3.0]

Farq 0.3.0 adds five modules that cover the steps around change detection: cloud
masking, radiometric normalization, elevation change and volumes, out-of-core processing
of large rasters, and export of change polygons to GIS formats. Every new public name is
available at the top level (`farq.<name>`) as well as from its submodule. One change can
break existing code: `change_summary` now refuses geographic CRSs (see *Changed*).

### Added

- **`farq.masking`**, cloud, shadow and quality masks from satellite quality bands
  ([guide](docs/masking.md)):
  - `landsat_qa_mask` and `decode_landsat_qa` (Landsat 4-9 Collection 2 `QA_PIXEL`,
    with `min_confidence=`), `landsat_radsat_mask` (`QA_RADSAT`, OLI, TM and ETM+).
  - `sentinel2_scl_mask` (L2A Scene Classification, classes by code, `SCLClass` member
    or name), `sentinel2_cloud_probability_mask` (`MSK_CLDPRB`), and `target_shape=` /
    `upsample_mask` to apply 20 m masks to 10 m bands exactly.
  - `hls_fmask_mask` (HLS v2.0 `Fmask`, including the aerosol level).
  - `buffer_mask` (by pixels or by ground distance, exact Euclidean),
    `combine_masks`, `apply_mask`, `clear_fraction`, `valid_overlap` (`MaskOverlap`)
    and `decode_bits`.
  - `landsat_c2_scale` and `sentinel2_l2a_scale` (with the processing baseline 04.00
    `BOA_ADD_OFFSET`).
  - Constants `LandsatQA`, `Confidence`, `SCLClass`, `SCL_NAMES`,
    `DEFAULT_S2_BAD_CLASSES` and `HLSFmask`.
- **`farq.radiometry`**, relative radiometric normalization between dates or flights
  ([guide](docs/radiometry.md)):
  - `histogram_match` (exact quantile mapping, or `n_quantiles=`).
  - `linear_normalize` (OLS, orthogonal, Theil–Sen or mean/std regression on a known
    no-change mask) and `pif_normalize` (pseudo-invariant pixels selected by IR-MAD,
    PCA or a percentile rule), both returning `NormalizationResult` with gains, offsets,
    `r2`, `rmse`, the fit pixels and `.apply()` to reuse a fit on another image.
  - `irmad` (`IRMADResult`: MAD variates, chi-square statistic, no-change probability,
    canonical correlations) and `irmad_change` (mask at a false-alarm rate `alpha`).
- **`farq.elevation`**, elevation change and volumes from DEMs and DSMs
  ([guide](docs/elevation.md)):
  - `slope`, `aspect` and `hillshade` using the pixel spacing of the geotransform.
  - `vertical_offset` (`VerticalOffset`), `coregister_dem` (Nuth & Kääb 2011,
    `DEMCoregistration`) and `shift_dem`.
  - `elevation_change`, `level_of_detection` and `significant_change`.
  - `volume_change` (`VolumeResult`: cut, fill and net volumes and areas, with
    uncorrelated or spatially correlated uncertainty after Rolstad et al. 2009) and
    `stockpile_volume` (`StockpileResult`: volume above a base fitted to the toe ring,
    a fixed elevation or an earlier surface).
- **`farq.tiling`**, out-of-core processing of rasters larger than memory
  ([guide](docs/tiling.md)):
  - `detect_changes_file`: `detect_changes` file to file, with a global threshold from
    a reproducible pixel sample and `min_size` / `fill_holes` exact across block borders.
  - `index_file` (any farq index), `map_blocks` (any NumPy function, with an overlap
    halo for neighbourhood operations), `summarize_file` (streaming statistics and an
    exact histogram), and `iter_windows` / `Block`.
  - Tiled, compressed outputs written atomically, `n_jobs=` threads with results that
    do not depend on `n_jobs` or (except where documented) on `block_size`, and
    `progress=` callbacks.
- **`farq.vector`**, change polygons for GIS ([guide](docs/vector.md)):
  - `polygonize`: masks and class maps to GeoJSON-like features with `pixel_count`,
    `area_m2`, `perimeter_m` and centroid, optional `min_area`, simplification and
    topology-preserving simplification.
  - `to_geojson` (RFC 7946, WGS 84 by default, no extra dependency) and
    `write_vector` (GeoJSON, GeoPackage, Shapefile, FlatGeobuf).
  - `changes_to_vector`: one call for a `ChangeResult`, a boolean mask or a
    `classify_change` class map.
- **`detect_changes(method="irmad")`** and **`detect_changes(normalize=...)`**; see
  *Changed*.
- **Optional extra `vector`**: `pip install "farq[vector]"` installs pyogrio and shapely
  for GeoPackage, Shapefile and FlatGeobuf output and topology-preserving
  simplification. GeoJSON needs nothing extra.
- **Documentation**: guides for [masking](docs/masking.md),
  [radiometry](docs/radiometry.md), [elevation](docs/elevation.md),
  [tiling](docs/tiling.md) and [vector export](docs/vector.md); the API reference,
  README and getting-started page cover the new modules.

### Changed

- **Areas are true m² for CRSs in feet.** `change_summary`, `ChangeResult.summary`,
  `TransitionMatrix.areas`, `detect_changes_file` and `farq.vector` (`area_m2`,
  `perimeter_m`, `min_area`) convert from the CRS's linear unit (e.g. EPSG:2263, US
  survey feet) when areas come from raster metadata, matching `farq.elevation`. Results
  for metre-based CRSs and plain pixel sizes are unchanged.
- **`farq.tiling.Block.index` is named `Block.number`** so it no longer shadows
  `tuple.index`.

- **`detect_changes` has two new keyword arguments.** Existing calls are unaffected.
  - `method="irmad"` uses the calibrated IR-MAD chi-square statistic as the magnitude.
    It is insensitive to per-band gain and offset differences between the dates, and it
    is thresholded at the false-alarm rate `alpha` (new, default 0.01) instead of
    `threshold`. Passing `threshold` together with `method="irmad"` raises `ValueError`.
    `ChangeResult.threshold` holds the chi-square quantile that was applied.
  - `normalize="pif"` or `normalize="histogram"` normalizes `after` to `before` with
    `pif_normalize` or `histogram_match` before the magnitude is computed. `"pif"` is
    recommended: it is fitted on unchanged pixels only. Histogram matching also reshapes
    the distribution, so it can attenuate real change that covers a noticeable part of
    the scene.
- **Breaking: `change_summary` raises `ValueError` for a geographic CRS in any
  spelling.** Metadata whose CRS is geographic (degrees) is refused whether the CRS is
  a `CRS` object, a string such as `"EPSG:4326"` or an EPSG code such as `4326`, and
  so is an invalid CRS. Previously a `CRS` object only gave a warning and a string or
  code skipped the check, so areas in squared degrees were reported as m². This also
  applies to `ChangeResult.summary`. *Migration:* reproject to a projected CRS (e.g.
  `farq.align_pair(..., dst_crs="EPSG:326xx")`) or pass the pixel size in metres.
- **CI** installs the `vector` extra (`pip install -e ".[test,vector]"`), so the
  GeoPackage, Shapefile and FlatGeobuf tests run on every platform.
- The version is `0.3.0`.

### Fixed

- The strict-warnings integration tests failed on Python 3.9-3.11 with rasterio older
  than 1.4.4, whose `from_origin` triggers a third-party `PendingDeprecationWarning`.
  Warnings raised by farq itself still fail the tests.

### Notes on the IR-MAD calibration

- In Nielsen's published IR-MAD scheme (2007), the chi-square statistic is computed from
  the *weighted* MAD variances. Weighting each pixel by its no-change probability
  shrinks these variances below the true no-change variances, so the statistic is too
  large: in simulations, the false-alarm rate at `alpha = 0.01` is above 50 %.
- Farq multiplies the statistic by the exact consistency factor for this weighting
  (`c_p = 2 I_{1/2}((p + 2) / 2, p / 2)`, the weighted-to-true covariance ratio for
  Gaussian no-change data with `p` bands), both for the weights and for the output.
  The `chi2` of unchanged pixels then follows a chi-square distribution with `p`
  degrees of freedom, and **`alpha` is the false-alarm rate** of `irmad_change` and
  `detect_changes(method="irmad")` under Gaussian no-change noise.
- Heavy-tailed no-change differences, such as residual misregistration or moved
  shadows, still raise the false-alarm rate. Align and co-register the images first.

## [0.2.0]

Farq 0.2.0 turns the library into a full raster change-detection toolkit for satellite
and drone imagery. It also fixes many correctness bugs in 0.1.x. Several of those fixes
change results, so **read the breaking changes and migration notes below before
upgrading.**

### Added

- **`farq.change`**, a new change-detection module:
  - `detect_changes` runs the whole pipeline (magnitude, threshold, cleanup) in one call
    and returns a `ChangeResult` with a `.summary()` method.
  - Change measures: `difference`, `ratio` (log-ratio), `normalized_difference_change`,
    `change_vector_analysis` (`CVAResult`), `pca_change` (`PCAChangeResult`).
  - Thresholds: `otsu_threshold`, `compute_threshold` (`"otsu"`, `"std"`,
    `"percentile"` or a number), `threshold_change`.
  - `clean_mask` (minimum region size and hole filling), `classify_change` (with the
    `NO_CHANGE`, `GAINED`, `LOST`, `STABLE` and `CHANGE_NODATA` codes and
    `CHANGE_LABELS`), `transition_matrix` (`TransitionMatrix` with `normalized()`,
    `areas()`, `changed`, `total`), and `change_summary` (counts, percentages, m² and km²,
    JSON-serializable).
- **`farq.georef`**, for georeferencing, alignment and co-registration, especially for
  drone imagery:
  - GCPs: `has_gcps`, `read_gcps`, `make_gcps`, `gcp_residuals` (`GCPResiduals` with
    RMSE, leave-one-out errors, externally studentized residuals and `outliers()`,
    a statistical test with a controlled false-alarm rate).
  - Rectification: `georeference` (array) and `rectify` (file), with polynomial (order
    1-3) or thin plate spline transforms.
  - Alignment: `align` (onto a reference grid) and `align_pair` (two rasters onto one
    common grid cropped to their overlap). Both accept GCP-referenced inputs directly.
  - Co-registration: `coregister` (sub-pixel translation by phase correlation) and
    `apply_shift`.
  - Pixel geometry: `pixel_size`, `pixel_area`.
- **RGB-only indices** for cameras without NIR: `vari`, `exg`, `exr`, `exgr`, `gli`,
  `ngrdi`, `tgi`. Also **`mndwi`** (Xu 2006). All of them are available through
  `calculate_indices`.
- **`farq.read`**:
  - `band=` selects a band, a list of bands, or all bands (`None`).
  - `masked=True` converts nodata, alpha bands and internal masks to NaN.
  - `out_shape=` and `resampling=` give decimated reads.
  - GCPs are returned in the metadata (`gcps`, `gcps_crs`).
- **`farq.write`**:
  - Writes multi-band arrays.
  - `dtype=` and `nodata=` overrides, and creation options such as `compress="deflate"`.
  - Masked-array support and GCP round-tripping.
- **`farq.resample`** supports `(bands, rows, cols)` arrays and NaN-aware resampling, and
  accepts method names such as `"average"`.
- **Index functions** take a `clip=` keyword (default `True`).
- **`farq.ml`**:
  - `ModelIntegrityError`.
  - `save_model` returns a SHA-256 digest and writes a `.sha256` sidecar.
    `load_model(expected_sha256=...)` pins the expected hash.
  - `train_classifier(ignore_label=, stratify=)`, raster-shaped training input,
    `metrics` entries `classes`, `n_train`, `n_test` and `n_dropped`, and `test_size=0`.
  - `predict_raster(fill_value=)`.
  - `detect_changes_ml(window_size=, batch_size=)`.
  - `augment_training_data(noise_level=)`.
  - `cluster_water_bodies(random_state=, water_high=, sample_size=)`.
  - `optimize_clustering(random_state=, sample_size=)`.
  - `connectivity=` for `analyze_water_clusters`.
- **`farq.analysis`**: `connectivity=` (4- or 8-neighbour bodies) and per-body
  `pixel_count`. `calculate_shape_metrics` takes `pixel_size=`. `get_water_bodies` with
  `calculate_shapes=True` returns all shape metrics.
- **`farq.visualization`**:
  - `ax=` and `axes=` to draw into existing axes.
  - `colorbar=` to toggle the colorbar.
  - `max_samples=` for fast histograms and contrast stretches on huge rasters.
- **`farq.utils`**: `percentile(axis=)`, `std(ddof=)`, `validate_array(allow_all_nan=)`.
  `stats` reports `skewness`, `kurtosis`, `histogram`, `percentages` and
  `reflectance_stats`.
- **Packaging**:
  - Type hints with a `py.typed` marker, and SPDX license metadata.
  - Optional extras `test` and `dev`.
  - Ruff, pytest and mypy configuration in `pyproject.toml`.
  - GitHub Actions CI (lint, tests on Python 3.9-3.13 on Linux, macOS and Windows,
    build) and a PyPI publish workflow.
- **Documentation**: a rewritten README and API reference, plus new
  [change detection](docs/change_detection.md) and [drone](docs/drone.md) guides.

### Changed

#### Breaking changes and migration notes

- **NDWI sign.** `ndwi` now uses the standard McFeeters (1996) definition
  `(green - nir) / (green + nir)`, so **open water is > 0**. 0.1.x computed
  `(nir - green) / (nir + green)`, which gave water < 0.
  *Migration:* replace `water = ndwi < 0` with `water = ndwi > 0` and negate any stored
  thresholds (e.g. `< -0.2` becomes `> 0.2`). `ndwi(green, nir)` keeps its argument
  order.
- **Undefined index values are NaN.** In every index, a zero denominator (e.g. `0 / 0`)
  gives NaN instead of 0, and NaN inputs propagate. Integer bands are converted to float
  first, so `uint16` input no longer wraps around.
  *Migration:* use the NaN-aware `farq.mean`, `farq.sum` and so on, `np.nan*`
  functions, or `np.isfinite(...)` masks. Use `np.nan_to_num(index)` only if you really
  want zeros.
- **Index clipping.** The other normalized indices are clipped to [-1, 1] by default as
  before. EVI and SAVI are now also controlled by the `clip=` keyword.
- **Perimeters in km.** `farq.analysis` per-body `perimeter` (in `water_stats(...,
  calculate_shapes=True)["shape_metrics"]["body_metrics"]` and in `get_water_bodies`) is
  now in **km**, consistent with areas in km². It was previously a gradient-based pixel
  count.
  - Perimeter is now the exact pixel-edge length. Compactness (4π·area/perimeter²) and
    elongation are computed in ground units, so values differ from 0.1.x.
  - `calculate_shape_metrics(mask, pixel_size=1.0)` reports `area` and `perimeter` in
    pixel units by default.
  - `ml.analyze_water_clusters` still reports m² and m.

  *Migration:* multiply by 1000 for metres, and recompute any stored metrics.
- **Visualization returns figures and has no side effects.** Every plotting function
  returns a `matplotlib.figure.Figure`, never calls `plt.show()`, and **no longer closes
  other open figures** (0.1.x `plot` called `plt.close("all")`).
  *Migration:* keep calling `farq.plt.show()` in scripts, or use `fig.savefig(...)`. In
  loops that create many figures, close them yourself with `plt.close(fig)`.
- **`detect_changes_ml` always returns a 2-D `(rows, cols)` boolean mask.** For
  multi-band `(rows, cols, bands)` input, the magnitude is the Euclidean norm across
  bands (0.1.x returned a per-band 3-D mask). The difference is computed in float, so
  unsigned inputs no longer wrap around.
  - With a model, features are built with `extract_features(diff,
    window_size=window_size)`.
  - Binary probabilistic classifiers are thresholded on the probability of
    `classes_[1]`.

  *Migration:* drop any `.any(axis=-1)` reduction you applied to the result.
- **`augment_training_data` returns the originals followed by
  `augmentation_factor - 1` jittered copies** (`n_samples * augmentation_factor` rows).
  - The noise is scaled per feature (`noise_level` times the feature's standard
    deviation) instead of an absolute 0.1.
  - The `np.rot90` "rotation" of tabular samples, which misaligned features and labels,
    was removed.
  - A local generator is used, so the global NumPy random state is no longer reseeded.
- **`load_model` checks integrity before unpickling.** It verifies the
  `<file>.sha256` sidecar written by `save_model` (and `expected_sha256` if given), and
  raises `ModelIntegrityError` (a `ValueError`) on a mismatch. The returned metadata
  gains a `"_farq"` entry.
  *Migration:* models saved by 0.1.x have no sidecar. They still load, with a
  `UserWarning`. Re-save them with `farq.save_model` to add the hash.
- **`cluster_water_bodies` picks the water cluster with the highest mean `water_index`**
  (water > 0 for NDWI and MNDWI). Previously it picked the lowest, which matched the old
  inverted NDWI.
  - Without an index, it picks the brightest cluster for 2-D input and the darkest for
    multi-band reflectance.
  - `random_state` is honoured, and invalid (NaN) pixels get label -1.
  - `metadata` always contains `water_cluster`, `cluster_means`, `n_clusters`,
    `n_invalid` and `water_rule`.
- **`optimize_clustering` scores by silhouette coefficient** instead of k-means inertia,
  which always favoured the largest `n_clusters`. It passes `water_index` to each trial.
  The default k-means grid no longer contains `random_state`.
- **`extract_features(indices=...)`** takes precomputed 2-D arrays (a dict or list) and
  now actually uses them. Strings raise `TypeError`.
- **`water_stats`** computes `coverage_percent` over **valid** pixels, so NaN pixels and
  masked entries of masked arrays are excluded (this applies to all `farq.analysis`
  functions). `water_change` returns `change_mask` as `int8`. Its `min_change_area` now
  removes connected gain or loss patches smaller than the area, instead of applying a
  morphological opening that also eroded large patches.
- **`validate_bands`** returns float arrays (`float32` or `float64`) and raises
  `TypeError` for non-numeric input.
- **`farq.utils`**:
  - `median` and `percentile` now ignore NaN.
  - `mean`, `std`, `min` and `max` return NaN for all-NaN slices along an axis instead of
    raising.
  - `stats()["non_zero"]` counts finite non-zero values. It previously counted non-NaN
    values.
- **`write`** sets `count`, `dtype`, `width` and `height` from the data instead of
  trusting the metadata. A float result written with the metadata of a `uint16` band is
  no longer silently cast to integers.
  - A nodata value that the output dtype cannot represent raises `ValueError`, and so
    does writing NaN or inf to an integer dtype without a nodata value.
  - A float result written with integer metadata gets NaN as nodata (unless `nodata=` is
    passed), so that valid zeros are not marked as nodata.
- **`change_summary` and `ChangeResult.summary`** refuse to compute areas from metadata
  without a geotransform (GCP-only or unreferenced rasters) instead of assuming 1 unit²
  per pixel.
- **`align_pair`** promotes both outputs to float with NaN as nodata when the two inputs
  have different nodata values (as it already did for different dtypes).
- **Python >= 3.9** is required (previously 3.7).
- **Packaging moved to `pyproject.toml`.** `setup.py`, `requirements*.txt` and
  `pytest.ini` were removed. Install development dependencies with
  `pip install -e ".[dev]"`.
- **Lazy imports.** `import farq` is fast. Submodules and matplotlib, scikit-learn and
  rasterio load on first use. All public names, and the 0.1 conveniences `farq.plt`,
  `farq.Resampling` and `farq.os`, remain available as attributes.

### Fixed

- **Indices:** inverted NDWI. Integer overflow and wrap-around for `uint16` and other
  integer bands. Silent zeros for undefined pixels.
- **I/O:**
  - `read` returned only band 1 and ignored nodata.
  - `write` could only write 2-D data and did not update `dtype` or `count`.
  - `resample` handled only 2-D arrays and let NaN bleed into neighbouring pixels.
- **`farq.analysis`:**
  - `mean_body_size` was biased low by an empty background bin.
  - NaN pixels in float masks were counted as water.
  - Shape metrics used a gradient approximation and ignored `pixel_size`.
  - Per-body metrics are now vectorized: one pass instead of a Python loop with a full
    image copy per water body.
- **`farq.ml`:**
  - `extract_features` crashed for any `window_size > 1` (including the default),
    because it passed `size=` to `scipy.ndimage.variance`, and it ignored `indices`.
    Window statistics are now NaN-aware.
  - `augment_training_data` crashed for non-square 2-D feature arrays.
  - `train_classifier` and `predict_raster` passed NaN and nodata pixels to the model
    (which errors with older scikit-learn). Invalid samples are now dropped for training
    and get `fill_value` in predictions, and splits are stratified.
  - `cluster_water_bodies` failed on NaN, ignored `random_state` and chose an arbitrary
    cluster (0) without a water index.
  - `analyze_water_clusters` used a gradient perimeter. It now uses the exact pixel-edge
    perimeter and is vectorized.
- **`farq.visualization`:**
  - `compare` crashed unless both `vmin` and `vmax` were given, and only used the first
    array for the colour limits. Both panels now share limits computed from both arrays.
  - `plot_rgb` produced a blank image when a band contained NaN, because the
    percentile stretch became NaN. NaN pixels are now transparent and ignored by the
    stretch.
  - `plot` closed all other figures.
- **`farq.utils`:** `median` and `percentile` were not NaN-aware, and the reductions
  emitted `RuntimeWarning`s on all-NaN slices.
- **Package:** the version strings disagreed (`farq.__version__` was `0.1.3`, `setup.py`
  had `0.1.5.1`). There is now one version, `0.2.0`.

- `plot_rgb` and `compare_rgb` accept a `(3, rows, cols)` stack, the layout
  `farq.read(path, band=[1, 2, 3])` returns.
- `get_water_bodies` validates `min_area` even when the mask has no water bodies.

### Security

- `save_model` writes the model atomically, together with a SHA-256 sidecar
  (`<file>.sha256`), and returns the digest.
- `load_model` verifies the file's SHA-256 **before** unpickling, against the sidecar and
  an optional caller-pinned `expected_sha256`. A mismatch raises `ModelIntegrityError`
  and the file is not loaded.
  - The file is read once, and the same in-memory bytes are hashed and unpickled, so
    there is no gap between the check and the load in which the file could change.
  - Hashes are compared in constant time, and sidecars that are oversized or not text
    are rejected.
  - Files without any integrity data load with a warning.
  - Non-regular files are refused, and a warning is raised when the scikit-learn version
    differs from the one used to save the model.
- `write` (and `rectify(out_path=...)`) writes GeoTIFFs atomically through a temporary
  file, so a failed write never truncates or destroys an existing file.
- Output-size guard: `read(out_shape=...)`, `resample` and the georef warping functions
  refuse outputs above 2³² pixels by default (override with the `FARQ_MAX_OUTPUT_PIXELS`
  environment variable). This prevents runaway memory use from, for example, a
  resolution in metres applied to a CRS in degrees.
- Model files are pickles, and loading one can execute arbitrary code. The docs now say
  this clearly: **never load model files from untrusted sources**. The sidecar only
  detects corruption or an accidentally swapped file. Pin `expected_sha256` with a hash
  obtained through a trusted channel to defend against tampering.

## [0.1.5.1] and earlier

Initial releases: raster I/O, NDWI/NDVI/EVI/SAVI/NDBI/NBR/NDMI, water statistics,
basic ML helpers and plotting.

[0.3.0]: https://github.com/ferasqr/farq/releases/tag/v0.3.0
[0.2.0]: https://github.com/ferasqr/farq/releases/tag/v0.2.0
[0.1.5.1]: https://pypi.org/project/farq/0.1.5.1/
