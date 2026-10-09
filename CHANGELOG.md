# Changelog

All notable changes to Farq are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

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

[0.2.0]: https://github.com/ferasqr/farq/releases/tag/v0.2.0
[0.1.5.1]: https://pypi.org/project/farq/0.1.5.1/
