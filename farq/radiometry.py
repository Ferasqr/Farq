"""
Relative radiometric normalization between dates or flights.

Two images of an unchanged scene rarely have the same pixel values. Sun angle,
exposure, white balance, atmosphere, sensor gain and processing differ between
acquisitions, and simple change measures report these differences as change. This is
one of the largest sources of false positives, especially for drone imagery, where
every flight has its own exposure. This module brings a ``source`` image onto the
radiometry of a ``reference`` image so that only real change remains.

Methods
-------
* :func:`histogram_match` maps each band's distribution onto the reference's
  (non-linear, needs no co-registration). Simple, but it also equalizes real change
  when change covers a large part of the scene.
* :func:`linear_normalize` fits a per-band gain and offset on pixels you know are
  unchanged.
* :func:`pif_normalize` selects pseudo-invariant features (PIFs) automatically,
  with IR-MAD, a PCA major-axis rule or a percentile rule, and then fits per-band
  gains and offsets on them.
* :func:`irmad` is Nielsen's (2007) Iteratively Reweighted Multivariate Alteration
  Detection. Its chi-square image is a change statistic that is insensitive to
  linear radiometric differences; :func:`irmad_change` thresholds it at a chosen
  false-alarm rate.

Conventions
-----------
* Stacks are shaped ``(bands, height, width)``; a 2-D array is one band.
* Invalid pixels are NaN or ±inf, equal to ``nodata`` or masked in a
  :class:`numpy.ma.MaskedArray`. They are excluded from every fit and come out as
  NaN.
* Outputs are floating point: ``float32`` for 8/16-bit integer and ``float32``
  inputs, ``float64`` otherwise. Statistics are accumulated in ``float64``.
* Results are deterministic, and no global random state is used.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from scipy import linalg, special, stats

from .change import _check_same_shape, _prepare
from .utils import _check_output_size

__all__ = [
    "IRMADResult",
    "NormalizationResult",
    "histogram_match",
    "irmad",
    "irmad_change",
    "linear_normalize",
    "pif_normalize",
]

_FIT_METHODS = ("ols", "orthogonal", "theil_sen", "mean_std")
_PIF_METHODS = ("irmad", "pca", "percentile")
_MIN_FIT_PIXELS = 10  # minimum pixels for a gain/offset fit
_CHUNK = 1 << 18  # pixels per chunk for IR-MAD statistics
_THEIL_SEN_PAIRS = 200_000  # pairs sampled by the Theil-Sen estimator
_MAD_SCALE = 1.482602218505602  # MAD -> standard deviation for normal data


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
class NormalizationResult(NamedTuple):
    """Output of :func:`linear_normalize` and :func:`pif_normalize`.

    The model is ``normalized[i] = gains[i] * source[i] + offsets[i]`` for each band.

    Attributes
    ----------
    normalized : numpy.ndarray
        ``source`` on the radiometric scale of ``reference``, same shape as
        ``source``; NaN where ``source`` is invalid.
    gains, offsets : numpy.ndarray
        ``float64`` arrays of shape ``(bands,)``.
    invariant_mask : numpy.ndarray
        Boolean ``(H, W)`` mask of the pixels the gains and offsets were fitted on.
    r2 : numpy.ndarray
        Per-band coefficient of determination of ``reference`` predicted by the fit
        on the fit pixels (NaN if the reference band is constant there).
    rmse : numpy.ndarray
        Per-band root-mean-square difference between ``reference`` and
        ``normalized`` on the fit pixels, in reference units.
    """

    normalized: np.ndarray
    gains: np.ndarray
    offsets: np.ndarray
    invariant_mask: np.ndarray
    r2: np.ndarray
    rmse: np.ndarray

    @property
    def n_invariant(self) -> int:
        """Number of pixels used for the fit."""
        return int(np.count_nonzero(self.invariant_mask))

    def apply(self, image: np.ndarray, *, nodata: float | None = None) -> np.ndarray:
        """Apply the fitted gains and offsets to another image of the same sensor.

        Use this, for example, to fit on a decimated preview and then normalize the
        full-resolution raster.

        Parameters
        ----------
        image : numpy.ndarray
            ``(bands, H, W)`` stack with the same number of bands as the fit, or a
            2-D image for a one-band fit.
        nodata : float, optional
            Sentinel value marking invalid pixels; they come out as NaN.

        Returns
        -------
        numpy.ndarray
            Normalized floating-point image with the shape of ``image``.
        """
        data, bad, squeeze = _stack(image, "image", nodata)
        if data.shape[0] != self.gains.shape[0]:
            raise ValueError(
                f"image has {data.shape[0]} band(s) but the normalization was fitted on "
                f"{self.gains.shape[0]}"
            )
        out = _apply_linear(data, bad, self.gains, self.offsets, data.dtype)
        return out[0] if squeeze else out


class IRMADResult(NamedTuple):
    """Output of :func:`irmad`.

    Attributes
    ----------
    mad_variates : numpy.ndarray
        MAD variates ``U_i - V_i``, shape ``(bands, H, W)``; NaN where invalid.
        ``mad_variates[0]`` belongs to the smallest canonical correlation and
        carries the most change.
    chi2 : numpy.ndarray
        Change statistic ``(H, W)``: the sum of the squared, standardized MAD
        variates. For unchanged pixels it follows a chi-square distribution with
        ``bands`` degrees of freedom, so it can be thresholded at a chosen
        false-alarm rate (see :func:`irmad_change`). NaN where invalid.
    no_change_prob : numpy.ndarray
        ``P(chi2_bands > chi2)``, the probability of no change (a p-value);
        NaN where invalid.
    canonical_correlations : numpy.ndarray
        Weighted canonical correlations of the final iteration, in ascending order
        (matching ``mad_variates``).
    n_iter : int
        Number of canonical correlation analyses performed.
    converged : bool
        Whether the canonical correlations changed by less than ``tol`` between
        the last two iterations.
    """

    mad_variates: np.ndarray
    chi2: np.ndarray
    no_change_prob: np.ndarray
    canonical_correlations: np.ndarray
    n_iter: int
    converged: bool


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def _stack(array: Any, name: str, nodata: float | None) -> tuple[np.ndarray, Any, bool]:
    """Return ``(float (bands, H, W) data, per-element invalid mask or None, was_2d)``."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    data, bad = _prepare(array, name, nodata)
    if data.ndim == 2:
        return data[np.newaxis], (None if bad is None else bad[np.newaxis]), True
    if data.ndim != 3:
        raise ValueError(
            f"{name} must be a 2-D image or a 3-D stack shaped (bands, height, width), "
            f"got shape {data.shape}"
        )
    return data, bad, False


def _pixel_mask(mask: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    """Turn a user mask into a boolean ``(H, W)`` array (NaN / masked entries = False)."""
    if isinstance(mask, np.ma.MaskedArray):
        invalid = np.ma.getmaskarray(mask)
        mask = np.asarray(mask.data)
    else:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(mask).__name__}")
        invalid = None
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape} (height, width), got {mask.shape}")
    if mask.dtype.kind not in "biuf":
        raise TypeError(f"{name} must have a boolean, integer or float dtype, got {mask.dtype}")
    if mask.dtype == bool:
        out = mask.copy()
    else:
        out = mask != 0
        if mask.dtype.kind == "f":
            out &= ~np.isnan(mask)
    if invalid is not None:
        out &= ~invalid
    return out


def _joint_valid(*bads: Any, shape: tuple[int, int]) -> np.ndarray:
    """Pixels valid in every band of every stack."""
    valid = np.ones(shape, dtype=bool)
    for bad in bads:
        if bad is not None:
            valid &= ~bad.any(axis=0)
    return valid


def _apply_linear(
    data: np.ndarray, bad: Any, gains: np.ndarray, offsets: np.ndarray, dtype: np.dtype
) -> np.ndarray:
    out = np.empty(data.shape, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(data.shape[0]):
            np.multiply(data[i], gains[i], out=out[i], casting="unsafe")
            out[i] += dtype.type(offsets[i])
    if bad is not None:
        out[bad] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Histogram matching
# --------------------------------------------------------------------------- #
def _source_cdf(values: np.ndarray, n_quantiles: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Strictly increasing CDF nodes ``(x, p)`` of ``values`` (mid-rank plotting positions)."""
    if n_quantiles is None:
        x, first, counts = np.unique(np.sort(values), return_index=True, return_counts=True)
        # ``first`` indexes the sorted values, so it is the rank of each value's first copy.
        p = (first + 0.5 * counts) / values.size
        return x, p
    probs = np.linspace(0.0, 1.0, n_quantiles)
    q = np.quantile(values, probs)
    x, inverse = np.unique(q, return_inverse=True)
    p = np.bincount(inverse.ravel(), weights=probs) / np.bincount(inverse.ravel())
    return x, p


def _reference_quantiles(
    values: np.ndarray, n_quantiles: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-function nodes ``(p, x)`` of ``values``."""
    if n_quantiles is None:
        x = np.sort(values)
        p = (np.arange(x.size) + 0.5) / x.size
        return p, x
    p = np.linspace(0.0, 1.0, n_quantiles)
    return p, np.quantile(values, p)


def histogram_match(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    n_quantiles: int | None = None,
    nodata: float | None = None,
) -> np.ndarray:
    """Match the histogram of each ``source`` band to the matching ``reference`` band.

    Every source pixel gets the reference value at the same cumulative-distribution
    position (quantile mapping). By default the mapping is exact: it uses all
    sorted values, and ties in the source get their mid-rank position, so equal
    inputs give equal outputs. Values between fitted values are interpolated
    linearly and values outside the fitted range are clamped to the reference's
    minimum and maximum.

    Histogram matching does not need co-registered images, but it is non-linear and
    forces the two histograms to agree. If a large part of the scene really
    changed, that change is partly equalized away; prefer :func:`pif_normalize`
    for change detection when change may be extensive.

    Parameters
    ----------
    source : numpy.ndarray
        Image to adjust, ``(H, W)`` or ``(bands, H, W)``, any real dtype.
    reference : numpy.ndarray
        Image with the target radiometry and the same number of bands. Its height
        and width may differ from the source's.
    valid : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of the pixels used to build both distributions,
        for example the overlap of two flights or a no-change mask. It requires
        ``source`` and ``reference`` to have the same shape. All valid source
        pixels are transformed, including those outside ``valid``.
    n_quantiles : int, optional
        Use this many evenly spaced quantiles (at least 2) instead of every value.
        This gives a smoother, piecewise-linear mapping, which helps with sparse
        histograms (for example 8-bit data).
    nodata : float, optional
        Sentinel value marking invalid pixels in either image.

    Returns
    -------
    numpy.ndarray
        Floating-point array with the shape of ``source``. Each band is matched
        independently, and NaN marks pixels that are invalid in that band.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.radiometry import histogram_match
    >>> source = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> reference = np.array([[10.0, 40.0], [30.0, 20.0]])
    >>> histogram_match(source, reference)
    array([[10., 20.],
           [30., 40.]])
    """
    src, bad_s, squeeze = _stack(source, "source", nodata)
    ref, bad_r, _ = _stack(reference, "reference", nodata)
    if src.shape[0] != ref.shape[0]:
        raise ValueError(
            f"source and reference must have the same number of bands, got {src.shape[0]} "
            f"and {ref.shape[0]}"
        )
    if n_quantiles is not None and (
        isinstance(n_quantiles, bool)
        or not isinstance(n_quantiles, (int, np.integer))
        or n_quantiles < 2
    ):
        raise ValueError(f"n_quantiles must be an integer >= 2 or None, got {n_quantiles!r}")
    if n_quantiles is not None:
        _check_output_size((n_quantiles,), "n_quantiles")
    fit = None
    if valid is not None:
        if src.shape[1:] != ref.shape[1:]:
            raise ValueError(
                "valid requires source and reference to have the same height and width, "
                f"got {src.shape[1:]} and {ref.shape[1:]}"
            )
        fit = _pixel_mask(valid, src.shape[1:], "valid")

    dt = np.result_type(src.dtype, ref.dtype)
    out = np.full(src.shape, np.nan, dtype=dt)
    for i in range(src.shape[0]):
        ok_s = np.ones(src.shape[1:], bool) if bad_s is None else ~bad_s[i]
        ok_r = np.ones(ref.shape[1:], bool) if bad_r is None else ~bad_r[i]
        fit_s, fit_r = ok_s, ok_r
        if fit is not None:
            fit_s = fit_s & fit & ok_r
            fit_r = fit_r & fit & ok_s
        s_vals = src[i][fit_s].astype(np.float64)
        r_vals = ref[i][fit_r].astype(np.float64)
        if s_vals.size == 0 or r_vals.size == 0:
            raise ValueError(
                f"band {i} has no valid pixels to build the "
                f"{'source' if s_vals.size == 0 else 'reference'} histogram from"
            )
        x_nodes, p_nodes = _source_cdf(s_vals, n_quantiles)
        p_ref, x_ref = _reference_quantiles(r_vals, n_quantiles)
        del s_vals, r_vals
        values = src[i][ok_s].astype(np.float64)
        # np.interp is much faster on sorted queries (cache-friendly searches).
        order = np.argsort(values, kind="stable")
        mapped = np.empty_like(values)
        mapped[order] = np.interp(np.interp(values[order], x_nodes, p_nodes), p_ref, x_ref)
        out[i][ok_s] = mapped
    return out[0] if squeeze else out


# --------------------------------------------------------------------------- #
# Linear (gain/offset) fits
# --------------------------------------------------------------------------- #
def _fit_band(
    x: np.ndarray, y: np.ndarray, method: str, band: int
) -> tuple[float, float, float, float]:
    """Fit ``y ≈ gain * x + offset``; return ``(gain, offset, r2, rmse)``."""
    n = x.size
    mx, my = float(x.mean()), float(y.mean())
    xc, yc = x - mx, y - my
    sxx, syy, sxy = float(xc @ xc), float(yc @ yc), float(xc @ yc)
    if np.sqrt(sxx / n) <= 1e-12 * max(abs(mx), 1e-300):
        raise ValueError(
            f"source band {band} is constant over the fit pixels; a gain cannot be estimated"
        )
    if method == "ols":
        gain = sxy / sxx
    elif method == "mean_std":
        gain = float(np.sqrt(syy / sxx))
    elif method == "orthogonal":
        if sxy == 0:
            raise ValueError(
                f"band {band}: source and reference are uncorrelated over the fit pixels; "
                "orthogonal regression is undefined"
            )
        d = syy - sxx
        gain = (d + float(np.hypot(d, 2.0 * sxy))) / (2.0 * sxy)
    else:  # theil_sen
        gain = _theil_sen_slope(x, y, band)
    offset = float(np.median(y - gain * x)) if method == "theil_sen" else my - gain * mx
    resid = y - (gain * x + offset)
    ss_res = float(resid @ resid)
    rmse = float(np.sqrt(ss_res / n))
    r2 = 1.0 - ss_res / syy if syy > 0 else float("nan")
    return gain, offset, r2, rmse


def _theil_sen_slope(x: np.ndarray, y: np.ndarray, band: int) -> float:
    n = x.size
    if n * (n - 1) // 2 <= _THEIL_SEN_PAIRS:
        i, j = np.triu_indices(n, k=1)
    else:  # median of slopes of a fixed random sample of pairs (deterministic seed)
        rng = np.random.default_rng(0)
        i = rng.integers(0, n, _THEIL_SEN_PAIRS)
        j = rng.integers(0, n, _THEIL_SEN_PAIRS)
    dx = x[j] - x[i]
    keep = dx != 0
    if not keep.any():
        raise ValueError(f"source band {band} is constant over the fit pixels")
    return float(np.median((y[j][keep] - y[i][keep]) / dx[keep]))


def _fit_all(
    src: np.ndarray, ref: np.ndarray, fit: np.ndarray, method: str
) -> tuple[np.ndarray, ...]:
    n_bands = src.shape[0]
    idx = np.flatnonzero(fit)
    gains, offsets, r2, rmse = (np.empty(n_bands) for _ in range(4))
    for i in range(n_bands):
        x = src[i].ravel()[idx].astype(np.float64)
        y = ref[i].ravel()[idx].astype(np.float64)
        gains[i], offsets[i], r2[i], rmse[i] = _fit_band(x, y, method, i)
    return gains, offsets, r2, rmse


def _check_pair(
    source: Any, reference: Any, nodata: float | None
) -> tuple[np.ndarray, Any, np.ndarray, Any, bool]:
    src, bad_s, squeeze = _stack(source, "source", nodata)
    ref, bad_r, _ = _stack(reference, "reference", nodata)
    _check_same_shape(src, ref, ("source", "reference"))
    return src, bad_s, ref, bad_r, squeeze


def _check_fit_method(method: str, name: str = "method") -> None:
    if method not in _FIT_METHODS:
        raise ValueError(f"Unknown {name} {method!r}; use one of {_FIT_METHODS}")


def _result(
    src: np.ndarray,
    bad_s: Any,
    ref: np.ndarray,
    fit: np.ndarray,
    params: tuple[np.ndarray, ...],
    squeeze: bool,
) -> NormalizationResult:
    gains, offsets, r2, rmse = params
    dt = np.result_type(src.dtype, ref.dtype)
    normalized = _apply_linear(src, bad_s, gains, offsets, dt)
    return NormalizationResult(
        normalized[0] if squeeze else normalized, gains, offsets, fit, r2, rmse
    )


def linear_normalize(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    method: str = "ols",
    nodata: float | None = None,
) -> NormalizationResult:
    """Fit and apply a per-band gain and offset that map ``source`` onto ``reference``.

    Use this when you know which pixels did not change (a digitized no-change mask,
    calibration targets, roofs or roads). Without such knowledge, use
    :func:`pif_normalize`.

    Parameters
    ----------
    source, reference : numpy.ndarray
        Co-registered ``(H, W)`` or ``(bands, H, W)`` images of the same shape.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of no-change pixels to fit on (default: all pixels).
        Pixels invalid in any band of either image are always excluded.
    method : {"ols", "orthogonal", "theil_sen", "mean_std"}, default "ols"
        Regression of reference on source:

        * ``"ols"``: ordinary least squares. It is optimal when the source is
          nearly noise-free, but it underestimates the gain (attenuation) when the
          source is noisy.
        * ``"orthogonal"``: total least squares (major axis), which allows noise
          in both images. This is the usual choice for PIF normalization.
        * ``"theil_sen"``: median of pairwise slopes, robust to up to about 29% of
          outliers (for example a mask that includes some changed pixels). Large
          inputs use a fixed sample of 200 000 pairs, so the result is
          deterministic.
        * ``"mean_std"``: matches each band's mean and standard deviation
          (gain = std ratio). It ignores the correlation between the images, so
          changed pixels in the mask bias it.
    nodata : float, optional
        Sentinel value marking invalid pixels in either image.

    Returns
    -------
    NormalizationResult
        ``(normalized, gains, offsets, invariant_mask, r2, rmse)``;
        ``invariant_mask`` holds the pixels actually used for the fit.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.radiometry import linear_normalize
    >>> rng = np.random.default_rng(0)
    >>> before = rng.uniform(0.0, 0.5, size=(2, 40, 40))
    >>> after = 1.25 * before + np.array([0.02, -0.01])[:, None, None]
    >>> res = linear_normalize(after, before)
    >>> res.gains.round(3), res.offsets.round(3)
    (array([0.8, 0.8]), array([-0.016,  0.008]))
    """
    _check_fit_method(method)
    src, bad_s, ref, bad_r, squeeze = _check_pair(source, reference, nodata)
    fit = _joint_valid(bad_s, bad_r, shape=src.shape[1:])
    if mask is not None:
        fit &= _pixel_mask(mask, src.shape[1:], "mask")
    n_fit = int(np.count_nonzero(fit))
    if n_fit < _MIN_FIT_PIXELS:
        raise ValueError(
            f"only {n_fit} valid pixel(s) are available for the fit; at least "
            f"{_MIN_FIT_PIXELS} are required (check mask and nodata)"
        )
    return _result(src, bad_s, ref, fit, _fit_all(src, ref, fit, method), squeeze)


# --------------------------------------------------------------------------- #
# Pseudo-invariant features
# --------------------------------------------------------------------------- #
def _robust_sigma(values: np.ndarray) -> float:
    return _MAD_SCALE * float(np.median(np.abs(values - np.median(values))))


def _pif_pca(
    src: np.ndarray, ref: np.ndarray, candidates: np.ndarray, n_sigma: float, max_iter: int = 10
) -> np.ndarray:
    """Pixels close to the major axis of the source/reference scatter in every band."""
    idx = np.flatnonzero(candidates)
    xs = [src[i].ravel()[idx].astype(np.float64) for i in range(src.shape[0])]
    ys = [ref[i].ravel()[idx].astype(np.float64) for i in range(src.shape[0])]
    keep = np.ones(idx.size, dtype=bool)
    for _ in range(max_iter):
        new = np.ones(idx.size, dtype=bool)
        for x, y in zip(xs, ys):
            if np.count_nonzero(keep) < 3:
                break
            pts = np.stack([x[keep], y[keep]])
            mean = pts.mean(axis=1)
            _, vecs = np.linalg.eigh(np.cov(pts))
            minor = vecs[:, 0]  # eigh sorts ascending: smallest variance first
            dist = minor[0] * (x - mean[0]) + minor[1] * (y - mean[1])
            sigma = _robust_sigma(dist[keep])
            if sigma == 0:
                new &= dist == 0
            else:
                new &= np.abs(dist) <= n_sigma * sigma
        if np.array_equal(new, keep):
            break
        keep = new
    out = np.zeros(candidates.size, dtype=bool)
    out[idx[keep]] = True
    return out.reshape(candidates.shape)


def _pif_percentile(
    src: np.ndarray, ref: np.ndarray, candidates: np.ndarray, percentile: float
) -> np.ndarray:
    """Pixels whose multi-band robust residual is in the lowest ``percentile`` percent.

    Each band gets a Theil-Sen line (robust to up to ~29% changed pixels); the score
    is the sum over bands of the squared residuals in robust standard deviations.
    """
    idx = np.flatnonzero(candidates)
    score = np.zeros(idx.size)
    for i in range(src.shape[0]):
        x = src[i].ravel()[idx].astype(np.float64)
        y = ref[i].ravel()[idx].astype(np.float64)
        gain = _theil_sen_slope(x, y, i)
        resid = y - gain * x
        resid -= np.median(resid)
        sigma = _robust_sigma(resid)
        if sigma == 0:
            sigma = float(resid.std())
        if sigma > 0:
            score += (resid / sigma) ** 2
    cut = np.percentile(score, percentile)
    out = np.zeros(candidates.size, dtype=bool)
    out[idx[score <= cut]] = True
    return out.reshape(candidates.shape)


def pif_normalize(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    method: str = "irmad",
    regression: str = "orthogonal",
    valid: np.ndarray | None = None,
    min_prob: float = 0.9,
    n_sigma: float = 2.0,
    percentile: float = 25.0,
    min_pixels: int = 50,
    max_iter: int = 50,
    tol: float = 1e-6,
    nodata: float | None = None,
) -> NormalizationResult:
    """Normalize ``source`` to ``reference`` using automatically selected invariant pixels.

    Pseudo-invariant features (PIFs) are pixels whose reflectance did not change
    between the dates. They are selected without user input, and per-band gains
    and offsets are then fitted on them (see :func:`linear_normalize`).

    Parameters
    ----------
    source, reference : numpy.ndarray
        Co-registered ``(H, W)`` or ``(bands, H, W)`` images of the same shape.
    method : {"irmad", "pca", "percentile"}, default "irmad"
        How PIFs are selected:

        * ``"irmad"`` (Canty & Nielsen 2008): pixels whose :func:`irmad` no-change
          probability exceeds ``min_prob``. The test is multivariate and invariant
          to linear radiometric differences, which makes it the most reliable
          choice. It works best with 3 or more bands.
        * ``"pca"``: pixels within ``n_sigma`` robust standard deviations of the
          major axis of the source/reference scatter plot, in every band. The axis
          is refitted iteratively on the retained pixels.
        * ``"percentile"``: a robust Theil-Sen line is fitted per band, and the
          pixels whose multi-band residual (in robust standard deviations) is in
          the lowest ``percentile`` percent are kept. This is simple and
          non-iterative, but it breaks down when more than about 29% of the
          valid pixels changed.
    regression : {"orthogonal", "ols", "theil_sen", "mean_std"}, default "orthogonal"
        Fit applied to the PIFs. Orthogonal (total least squares) regression is
        the standard choice, because both images contain noise.
    valid : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of pixels allowed as PIFs, for example to exclude
        water, vegetation or clouds that are known to vary.
    min_prob : float, default 0.9
        ``"irmad"``: minimum no-change probability. For unchanged pixels the
        probability is uniform on [0, 1], so 0.9 keeps the most stable 10%.
    n_sigma : float, default 2.0
        ``"pca"``: maximum distance from the major axis, in robust standard
        deviations.
    percentile : float, default 25.0
        ``"percentile"``: percentage of valid pixels kept, in (0, 100].
    min_pixels : int, default 50
        Raise ``ValueError`` if fewer PIFs are selected.
    max_iter, tol
        Passed to :func:`irmad` (``"irmad"`` only).
    nodata : float, optional
        Sentinel value marking invalid pixels in either image.

    Returns
    -------
    NormalizationResult
        ``invariant_mask`` holds the selected PIFs.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.radiometry import pif_normalize
    >>> rng = np.random.default_rng(1)
    >>> before = rng.uniform(0.05, 0.4, size=(3, 60, 60))
    >>> after = 1.2 * before + 0.03 + 0.002 * rng.standard_normal((3, 60, 60))
    >>> after[:, :20, :20] = 0.6          # real change
    >>> res = pif_normalize(after, before)
    >>> bool(res.invariant_mask[:20, :20].any()), res.gains.round(2)
    (False, array([0.83, 0.83, 0.83]))
    """
    if method not in _PIF_METHODS:
        raise ValueError(f"Unknown method {method!r}; use one of {_PIF_METHODS}")
    _check_fit_method(regression, "regression")
    if not 0.0 <= min_prob < 1.0:
        raise ValueError(f"min_prob must be in [0, 1), got {min_prob}")
    if not n_sigma > 0:
        raise ValueError(f"n_sigma must be positive, got {n_sigma}")
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    if (
        isinstance(min_pixels, bool)
        or not isinstance(min_pixels, (int, np.integer))
        or min_pixels < _MIN_FIT_PIXELS
    ):
        raise ValueError(f"min_pixels must be an integer >= {_MIN_FIT_PIXELS}, got {min_pixels!r}")

    src, bad_s, ref, bad_r, squeeze = _check_pair(source, reference, nodata)
    candidates = _joint_valid(bad_s, bad_r, shape=src.shape[1:])
    if valid is not None:
        candidates &= _pixel_mask(valid, src.shape[1:], "valid")
    n_cand = int(np.count_nonzero(candidates))
    if n_cand < min_pixels:
        raise ValueError(
            f"only {n_cand} valid pixel(s) are available; at least min_pixels={min_pixels} "
            "are required (check valid and nodata)"
        )

    if method == "irmad":
        res = _irmad(src, ref, ~candidates, candidates, max_iter=max_iter, tol=tol)
        invariant = candidates & (res.no_change_prob > min_prob)
    elif method == "pca":
        invariant = _pif_pca(src, ref, candidates, n_sigma)
    else:
        invariant = _pif_percentile(src, ref, candidates, percentile)

    n_inv = int(np.count_nonzero(invariant))
    if n_inv < min_pixels:
        hint = {
            "irmad": "lower min_prob",
            "pca": "raise n_sigma",
            "percentile": "raise percentile",
        }[method]
        raise ValueError(
            f"only {n_inv} pseudo-invariant pixel(s) selected with method={method!r}; "
            f"at least min_pixels={min_pixels} are required ({hint}, or use more pixels)"
        )
    return _result(src, bad_s, ref, invariant, _fit_all(src, ref, invariant, regression), squeeze)


# --------------------------------------------------------------------------- #
# IR-MAD
# --------------------------------------------------------------------------- #
def _consistency_factor(p: int) -> float:
    """Scale that makes the reweighted MAD covariance consistent (see :func:`irmad`).

    For Gaussian no-change data with weights ``w = P(chi2_p > z)``, the weighted
    covariance is ``c * Σ`` with ``c = E[z w] / (p E[w]) = 2 P(chi2_{p+2} < chi2'_p)``,
    which equals ``2 I_{1/2}((p + 2) / 2, p / 2)``.
    """
    return float(2.0 * special.betainc((p + 2) / 2.0, p / 2.0, 0.5))


def _check_pd(cov: np.ndarray, name: str) -> None:
    ev = np.linalg.eigvalsh(cov)
    if not ev[-1] > 0 or ev[0] <= 1e-12 * ev[-1]:
        raise ValueError(
            f"the band covariance of {name} is singular: a band is constant or bands are "
            "linear combinations of each other over the valid pixels; drop redundant bands"
        )


class _CCA(NamedTuple):
    mean_x: np.ndarray
    mean_y: np.ndarray
    a: np.ndarray  # (p, p) columns are canonical vectors of X
    b: np.ndarray
    rho: np.ndarray  # ascending
    scale: np.ndarray  # 1 / variance of each MAD variate, including consistency factor


def _cca(mean: np.ndarray, cov: np.ndarray, p: int, factor: float) -> _CCA:
    sxx, syy, sxy = cov[:p, :p], cov[p:, p:], cov[:p, p:]
    _check_pd(sxx, "before_stack")
    _check_pd(syy, "after_stack")
    m = sxy @ linalg.solve(syy, sxy.T, assume_a="pos")
    r2, a = linalg.eigh((m + m.T) / 2.0, sxx)  # ascending; a' Sxx a = I
    rho = np.sqrt(np.clip(r2, 0.0, 1.0))
    sign = np.sign((sxx @ a).sum(axis=0))  # positive correlation of U_i with the X bands
    a *= np.where(sign == 0, 1.0, sign)
    b = linalg.solve(syy, sxy.T @ a, assume_a="pos")
    norm = np.sqrt(np.clip(np.einsum("ij,ij->j", b, syy @ b), 0.0, None))
    weak = norm <= 1e-10
    if weak.any():  # rho ~ 0: b is not determined by a; solve its own eigenproblem
        n = sxy.T @ linalg.solve(sxx, sxy, assume_a="pos")
        _, b2 = linalg.eigh((n + n.T) / 2.0, syy)
        b[:, weak] = b2[:, weak]
        norm[weak] = 1.0
    b /= norm
    corr = np.einsum("ij,ij->j", a, sxy @ b)
    b *= np.where(corr < 0, -1.0, 1.0)
    if np.any(1.0 - rho <= 1e-12):
        raise ValueError(
            "after_stack is an exact linear function of before_stack (canonical "
            "correlation 1), so there is no noise level to test change against"
        )
    scale = factor / (2.0 * (1.0 - rho))
    return _CCA(mean[:p], mean[p:], a, b, rho, scale)


def _irmad(
    b: np.ndarray,
    a: np.ndarray,
    bad: np.ndarray,
    fit: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> IRMADResult:
    """IR-MAD on prepared ``(p, H, W)`` stacks; ``bad`` = invalid, ``fit`` = used in statistics."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol}")
    p, height, width = b.shape
    n_fit = int(np.count_nonzero(fit))
    need = max(_MIN_FIT_PIXELS, 2 * p + 2)
    if n_fit < need:
        raise ValueError(f"irmad needs at least {need} valid pixels for {p} band(s), got {n_fit}")
    x_all = b.reshape(p, -1)
    y_all = a.reshape(p, -1)
    fit_flat = fit.ravel()
    n_total = fit_flat.size

    def chunks(sel: np.ndarray):
        for start in range(0, n_total, _CHUNK):
            stop = min(start + _CHUNK, n_total)
            s = sel[start:stop]
            if s.any():
                xb = x_all[:, start:stop][:, s].astype(np.float64)
                yb = y_all[:, start:stop][:, s].astype(np.float64)
                yield start, s, xb, yb

    def chi2_of(cca: _CCA, xb: np.ndarray, yb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mad = cca.a.T @ (xb - cca.mean_x[:, None]) - cca.b.T @ (yb - cca.mean_y[:, None])
        return mad, cca.scale @ (mad * mad)

    def accumulate(cca: _CCA | None, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted mean and covariance of [X; Y] with weights from ``cca`` (or 1)."""
        total = 0.0
        s1 = np.zeros(2 * p)
        s2 = np.zeros((2 * p, 2 * p))
        for _, _, xb, yb in chunks(fit_flat):
            z = np.vstack([xb, yb]) - shift[:, None]
            w = (
                np.ones(z.shape[1])
                if cca is None
                else special.gammaincc(p / 2.0, chi2_of(cca, xb, yb)[1] / 2.0)
            )
            total += float(w.sum())
            s1 += z @ w
            s2 += (z * w) @ z.T
        if not total > 0:
            raise ValueError(
                "irmad: every pixel has zero no-change probability; the images are too "
                "different to estimate a no-change relationship"
            )
        d = s1 / total
        return shift + d, s2 / total - np.outer(d, d)

    shift = np.zeros(2 * p)
    for _, _, xb, yb in chunks(fit_flat):
        shift = np.concatenate([xb.mean(axis=1), yb.mean(axis=1)])
        break
    c_p = _consistency_factor(p)

    mean, cov = accumulate(None, shift)
    cca = _cca(mean, cov, p, 1.0)  # plain MAD: uniform weights are consistent as is
    n_iter, converged = 1, False
    while n_iter < max_iter:
        mean, cov = accumulate(cca, mean)
        new = _cca(mean, cov, p, c_p)
        n_iter += 1
        delta = float(np.max(np.abs(new.rho - cca.rho)))
        cca = new
        if delta < tol:
            converged = True
            break

    dt = np.result_type(b.dtype, a.dtype)
    mad_out = np.full((p, n_total), np.nan, dtype=dt)
    chi2_out = np.full(n_total, np.nan, dtype=dt)
    prob_out = np.full(n_total, np.nan, dtype=dt)
    for start, s, xb, yb in chunks(~bad.ravel()):
        mad, z = chi2_of(cca, xb, yb)
        cols = np.flatnonzero(s) + start
        mad_out[:, cols] = mad
        chi2_out[cols] = z
        prob_out[cols] = special.gammaincc(p / 2.0, z / 2.0)  # chi2(p) survival
    return IRMADResult(
        mad_out.reshape(p, height, width),
        chi2_out.reshape(height, width),
        prob_out.reshape(height, width),
        cca.rho.copy(),
        n_iter,
        converged,
    )


def _prepare_irmad(
    before_stack: Any, after_stack: Any, valid: Any, nodata: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b, bad_b, _ = _stack(before_stack, "before_stack", nodata)
    a, bad_a, _ = _stack(after_stack, "after_stack", nodata)
    _check_same_shape(b, a, ("before_stack", "after_stack"))
    ok = _joint_valid(bad_b, bad_a, shape=b.shape[1:])
    fit = ok if valid is None else ok & _pixel_mask(valid, b.shape[1:], "valid")
    return b, a, ~ok, fit


def irmad(
    before_stack: np.ndarray,
    after_stack: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    valid: np.ndarray | None = None,
    nodata: float | None = None,
) -> IRMADResult:
    """Iteratively Reweighted Multivariate Alteration Detection (Nielsen 2007).

    MAD transforms the two stacks with a canonical correlation analysis (CCA): the
    MAD variates ``U_i - V_i`` are differences of maximally correlated linear
    combinations of the bands. Because CCA is invariant to linear transformations
    of either stack, per-band gains and offsets (exposure, sun angle, sensor
    calibration) do not show up as change. IR-MAD iterates the CCA, weighting each
    pixel by its probability of no change, so that the no-change relationship is
    estimated from unchanged pixels only.

    Calibration
    -----------
    The published scheme computes the chi-square statistic from the *weighted*
    MAD variances ``2(1 - rho_i)``. Down-weighting pixels by their p-value shrinks
    those variances below the true no-change variances, so the statistic is far
    too large: in simulations the false-alarm rate at ``alpha = 0.01`` is above
    50%. This implementation multiplies the statistic by the consistency factor
    ``c_p = 2 I_{1/2}((p + 2) / 2, p / 2)``, the exact weighted-to-true covariance
    ratio for Gaussian no-change data. This factor is used for both the weights
    and the output. As a result, ``chi2`` of unchanged pixels follows
    ``chi2(bands)`` and the false-alarm rate of :func:`irmad_change` matches
    ``alpha``. The first iteration has uniform weights and needs no factor.

    Parameters
    ----------
    before_stack, after_stack : numpy.ndarray
        Co-registered ``(bands, H, W)`` stacks of the same shape, or 2-D images.
        The bands need not be the same in both stacks.
    max_iter : int, default 50
        Maximum number of CCAs (1 gives plain MAD).
    tol : float, default 1e-6
        Stop when no canonical correlation changes by more than this.
    valid : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of pixels used to estimate the statistics (for
        example to exclude clouds or water). Outputs are still computed for every
        valid pixel.
    nodata : float, optional
        Sentinel value; a pixel is invalid if any band in either stack is invalid.

    Returns
    -------
    IRMADResult
        ``(mad_variates, chi2, no_change_prob, canonical_correlations, n_iter,
        converged)``.

    Raises
    ------
    ValueError
        If there are too few valid pixels (at least ``max(10, 2 * bands + 2)``), a
        band covariance is singular (constant or duplicated bands), or ``after`` is
        an exact linear function of ``before`` (no noise).

    Notes
    -----
    Statistics are accumulated in ``float64`` over pixel chunks, so memory stays
    proportional to the image size. The canonical vectors are signed so that each
    ``U_i`` correlates positively with the sum of the before bands and ``V_i``
    correlates positively with ``U_i``, which makes the output deterministic. The
    MAD variates are ordered by ascending canonical correlation, so the first
    variate has the most change variance. The chi-square calibration assumes
    approximately Gaussian no-change noise. With heavy-tailed noise, for example
    from misregistration at edges, the false-alarm rate is higher.

    References
    ----------
    A. A. Nielsen, "The regularized iteratively reweighted MAD method for change
    detection in multi- and hyperspectral data", IEEE Trans. Image Processing
    16(2), 463-478, 2007.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.radiometry import irmad
    >>> rng = np.random.default_rng(0)
    >>> before = rng.normal(size=(3, 50, 50))
    >>> after = 0.7 * before + 0.1 + 0.05 * rng.normal(size=(3, 50, 50))
    >>> after[:, 10:20, 10:20] += 2.0
    >>> res = irmad(before, after)
    >>> res.converged, bool(res.chi2[10:20, 10:20].min() > res.chi2[30:, 30:].max())
    (True, True)
    """
    b, a, bad, fit = _prepare_irmad(before_stack, after_stack, valid, nodata)
    return _irmad(b, a, bad, fit, max_iter=max_iter, tol=tol)


def irmad_change(
    before_stack: np.ndarray,
    after_stack: np.ndarray,
    *,
    alpha: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-6,
    valid: np.ndarray | None = None,
    nodata: float | None = None,
) -> np.ndarray:
    """Boolean change mask from :func:`irmad` at false-alarm rate ``alpha``.

    A pixel is flagged when its chi-square statistic exceeds the
    ``1 - alpha`` quantile of the chi-square distribution with ``bands`` degrees
    of freedom, so about ``alpha`` of the unchanged pixels are flagged.

    Parameters
    ----------
    before_stack, after_stack : numpy.ndarray
        Co-registered ``(bands, H, W)`` stacks or 2-D images.
    alpha : float, default 0.01
        False-alarm rate (significance level) in (0, 1).
    max_iter, tol, valid, nodata
        As in :func:`irmad`.

    Returns
    -------
    numpy.ndarray
        Boolean ``(H, W)`` mask; False where invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.radiometry import irmad_change
    >>> rng = np.random.default_rng(0)
    >>> before = rng.normal(size=(3, 50, 50))
    >>> after = 1.5 * before - 0.2 + 0.05 * rng.normal(size=(3, 50, 50))
    >>> after[:, 10:20, 10:20] += 2.0
    >>> mask = irmad_change(before, after, alpha=0.001)
    >>> bool(mask[10:20, 10:20].all()), int(mask.sum()) - 100 < 10
    (True, True)
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    res = irmad(before_stack, after_stack, max_iter=max_iter, tol=tol, valid=valid, nodata=nodata)
    threshold = stats.chi2.ppf(1.0 - alpha, res.mad_variates.shape[0])
    chi2 = res.chi2
    return np.greater(chi2, threshold, where=np.isfinite(chi2), out=np.zeros(chi2.shape, bool))
