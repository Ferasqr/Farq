# Radiometric normalization guide

Two images of an unchanged scene almost never have the same pixel values. Sun angle,
exposure and white balance, haze, sensor calibration and processing all differ between
acquisitions. A plain `after - before` reports all of these as change. This is the
largest source of false positives in drone change detection, because every flight has
its own exposure.

`farq.radiometry` removes these differences before you measure change (relative
normalization), or measures change in a way that ignores them (IR-MAD):

| Function | What it does | Use it when |
|---|---|---|
| `histogram_match` | Maps each band's distribution onto the reference's (non-linear) | Images are not co-registered, change covers a small part of the scene, or the radiometric difference is non-linear (different cameras, tone curves) |
| `linear_normalize` | Per-band gain/offset fitted on pixels **you** say are unchanged | You have a no-change mask: calibration targets, roofs and roads, a digitized area |
| `pif_normalize` | Selects pseudo-invariant pixels automatically, then fits gain/offset | The default for co-registered pairs. Robust when part of the scene really changed |
| `irmad` / `irmad_change` | Iteratively reweighted MAD: a change statistic that is insensitive to any per-band linear radiometric difference, with a calibrated false-alarm rate | Multi-band data when you want a change mask directly, with a statistical threshold instead of Otsu |

Every function is available as `farq.<name>` (for example `farq.pif_normalize`) and from
`farq.radiometry`. No extra dependency is needed beyond `pip install farq`.

Conventions are the same as in the rest of Farq. Stacks are `(bands, height, width)`
and 2-D arrays are one band. NaN, ±inf, `nodata` values and masked pixels are excluded
from every fit and come out as NaN. Inputs of any real dtype give floating-point
output, and everything is deterministic.

The examples on this page run in order and share variables. They use a synthetic
two-flight scene so that you can check the results against the truth.

```python
import numpy as np
import farq

rng = np.random.default_rng(0)
scene = rng.uniform(0.05, 0.45, size=(3, 200, 200))           # true reflectance, RGB
flight_1 = scene + 0.003 * rng.standard_normal(scene.shape)

# Flight 2 is brighter and has a different white balance ...
gain = np.array([1.30, 1.15, 0.90])[:, None, None]
offset = np.array([0.04, 0.02, -0.01])[:, None, None]
flight_2 = gain * scene + offset + 0.003 * rng.standard_normal(scene.shape)
# ... and one 40 x 40 area really changed (e.g. a new roof).
flight_2[:, 50:90, 120:160] += np.array([0.25, -0.20, 0.30])[:, None, None]
truth = np.zeros((200, 200), bool)
truth[50:90, 120:160] = True

# Flag changes larger than 0.1 reflectance. Without normalization, most of the
# scene looks changed:
raw = farq.detect_changes(flight_1, flight_2, method="cva", threshold=0.1)
print(f"raw: {raw.mask[~truth].mean():.0%} of unchanged pixels flagged")  # 86%
```

## Automatic normalization with pseudo-invariant features

`pif_normalize(source, reference)` returns `source` on the radiometric scale of
`reference`. To measure change, normalize the later image to the earlier one, so that
magnitudes stay in "before" units.

```python
res = farq.pif_normalize(flight_2, flight_1)         # method="irmad" by default
print(res.gains.round(3), res.offsets.round(3))      # ~ 1/gain and -offset/gain
print(res.n_invariant, "PIFs; any inside the changed area?",
      bool(res.invariant_mask[truth].any()))
print("r2", res.r2.round(4), "rmse", res.rmse.round(4))

normalized = farq.detect_changes(flight_1, res.normalized, method="cva", threshold=0.1)
print(f"normalized: {normalized.mask[~truth].mean():.1%} of unchanged pixels flagged, "
      f"{normalized.mask[truth].mean():.0%} of the change found")  # 0.0%, 100%
```

The fitted model is `normalized[i] = gains[i] * source[i] + offsets[i]`. Check `r2` and
`rmse` per band, which are computed on the invariant pixels: a low `r2` means that no
linear relation exists, for example with different cameras or strong non-linear tone
mapping. In that case use `histogram_match` instead.

Choosing the PIF selection (`method`):

- `"irmad"` (default) keeps pixels whose IR-MAD no-change probability exceeds
  `min_prob` (default 0.9, which keeps the most stable 10% of unchanged pixels). The
  test is multivariate and unaffected by the radiometric differences you are trying to
  remove, so it is the most reliable rule. It needs at least two bands to be really
  discriminative, so prefer three or more.
- `"pca"` keeps pixels within `n_sigma` robust standard deviations of the major axis
  of each band's scatter plot, refitted iteratively. It is a good alternative for one
  or two bands.
- `"percentile"` fits a robust Theil–Sen line per band and keeps the `percentile`%
  of pixels with the smallest residuals. It is the fastest rule, but it breaks down
  when more than about 29% of the scene changed.

The `regression` used on the PIFs defaults to `"orthogonal"` (total least squares).
This is the standard choice for PIF normalization because both images contain noise:
ordinary least squares would underestimate the gain. Use `valid=` to forbid surfaces
that are known to vary (water, vegetation, shadows, clouds) as PIFs.

```python
for method in ("pca", "percentile"):
    r = farq.pif_normalize(flight_2, flight_1, method=method)
    print(method, r.gains.round(3), bool(r.invariant_mask[truth].any()))
```

### Shortcut: `detect_changes(normalize=...)`

`farq.detect_changes` can normalize `after` to `before` itself before it measures change.
`normalize="pif"` runs `pif_normalize` with its defaults (recommended), and
`normalize="histogram"` runs `histogram_match`:

```python
quick = farq.detect_changes(flight_1, flight_2, method="cva", threshold=0.1,
                            normalize="pif")
print(f"normalize='pif': {quick.mask[~truth].mean():.1%} false alarms, "
      f"{quick.mask[truth].mean():.0%} of the change found")  # 0.0%, 100%
```

Use the functions directly when you want to check `r2` and `rmse`, restrict the PIFs with
`valid=`, or reuse the fit on another image. `normalize="histogram"` reshapes the whole
distribution, so it can attenuate real change that covers a noticeable part of the scene.

## Normalization with a known no-change mask

When you know which pixels did not change, fit on them directly:

```python
no_change = np.zeros((200, 200), bool)
no_change[150:, :] = True                       # e.g. a road and car park, digitized
lin = farq.linear_normalize(flight_2, flight_1, mask=no_change, method="orthogonal")
print(lin.gains.round(3), lin.offsets.round(3))
```

Regressions: `"ols"` assumes a noise-free source. `"orthogonal"` allows noise in both
images. `"theil_sen"` (median of pairwise slopes) tolerates a mask that accidentally
contains some changed pixels. `"mean_std"` only matches means and standard deviations
and should be used only on clean masks.

### Fit on a preview, apply to the full raster

Gains and offsets are per band, so a fit on a decimated read applies to the
full-resolution image:

```python
preview = farq.pif_normalize(flight_2[:, ::4, ::4], flight_1[:, ::4, ::4])
full = preview.apply(flight_2)                  # same model on the full array
print(full.shape, np.abs(full - res.normalized)[:, ~truth].mean().round(4))
```

With files, read the preview with `farq.read(path, band=None, masked=True,
out_shape=(h, w))`.

## Histogram matching

```python
matched = farq.histogram_match(flight_2, flight_1)
print(np.quantile(matched[0], [0.1, 0.5, 0.9]).round(3),
      np.quantile(flight_1[0], [0.1, 0.5, 0.9]).round(3))
```

Each band is mapped through the empirical cumulative distribution functions, so the
output has (up to interpolation) exactly the reference's histogram. The images do not
need the same size or grid, which makes this useful for normalizing a mosaic to a
different-footprint reference. Pass `valid=` (same-shape images only) to build both
histograms from the overlap or from a no-change mask. Use `n_quantiles=256` for a
smoother mapping of sparse 8-bit histograms.

Keep in mind that histogram matching *forces* the distributions to agree. If a large
part of the scene really changed, for example after a flood, matching partly cancels
that change. For change detection on co-registered pairs, prefer `pif_normalize`.

## IR-MAD: change detection that ignores radiometry

Multivariate Alteration Detection (MAD) finds, by canonical correlation analysis, the
linear band combinations of the two dates that are maximally correlated, and takes
their differences (the MAD variates). Canonical correlations do not change under any
per-band gain or offset, so exposure and calibration differences cancel out. IR-MAD
(Nielsen 2007) repeats the analysis and down-weights pixels that look changed, so the
no-change relationship is learned from unchanged pixels.

```python
mad = farq.irmad(flight_1, flight_2)
print(mad.converged, mad.n_iter, mad.canonical_correlations.round(4))
print(mad.mad_variates.shape, mad.chi2.shape)

mask = farq.irmad_change(flight_1, flight_2, alpha=0.01)
print(f"IR-MAD: false-alarm rate {mask[~truth].mean():.4f} (alpha 0.01), "
      f"detected {mask[truth].mean():.0%}")  # about 0.01, 100%
```

`chi2` is a change magnitude, a chi-square statistic with `bands` degrees of freedom,
and `no_change_prob` is its p-value. `irmad_change` flags pixels with
`chi2 > chi2.ppf(1 - alpha, bands)`, so about a fraction `alpha` of the unchanged
pixels are flagged. Combine it with `farq.clean_mask` to remove isolated false
alarms.

`farq.detect_changes(before, after, method="irmad", alpha=0.01)` does the same inside
the usual pipeline: the magnitude is `chi2`, the threshold is the chi-square quantile
for `alpha` (`threshold=` cannot be combined with `"irmad"` and raises `ValueError`), and
`min_size` / `fill_holes` clean the mask:

```python
result = farq.detect_changes(flight_1, flight_2, method="irmad", alpha=0.01, min_size=5)
print(f"threshold chi2 > {result.threshold:.2f}; "
      f"false alarms {result.mask[~truth].mean():.4f}, detected {result.mask[truth].mean():.0%}")
```

**Calibration note.** In the published IR-MAD scheme, the chi-square statistic uses the
*weighted* MAD variances. Weighting by the no-change probability shrinks these
variances, and the false-alarm rate then far exceeds `alpha` (over 50% at
`alpha = 0.01` in simulations). Farq applies the exact consistency factor for this
weighting, so the statistic is chi-square distributed on unchanged pixels, assuming
Gaussian no-change noise. Misregistration, shadows that moved and other heavy-tailed
"no-change" differences still raise the false-alarm rate. Align and co-register first
(see the [drone guide](drone.md)).

## Limitations

- Linear methods assume one gain and offset per band for the whole image. Vignetting,
  bidirectional reflectance effects (BRDF) across a drone mosaic and changing shadows
  are not removed. For those, normalize tiles separately or use `histogram_match` on
  tiles.
- `irmad` raises a `ValueError` if a band is constant or duplicates other bands
  (singular covariance), or if `after` is an exact linear function of `before`, because
  then there is no noise to test change against.
- Saturated pixels (for example 255 in 8-bit images) are not linear. Mark them as
  `nodata` or exclude them with `valid=` / `mask=`.
