"""
Change detection for co-registered rasters.

This module turns two images of the same area, acquired at different times, into
a per-pixel *change magnitude*, a boolean *change mask* and summary statistics.
Every function is vectorized with NumPy/SciPy and works on arrays of any size.

Workflow
--------
1. Measure change per pixel: :func:`difference`, :func:`ratio` (log-ratio, robust
   for SAR), :func:`normalized_difference_change`, :func:`change_vector_analysis`
   (multi-band) or :func:`pca_change` (multi-band).
2. Turn the magnitude into a mask: :func:`compute_threshold` /
   :func:`threshold_change` with Otsu, mean + k·std, percentile or a fixed value.
3. Clean the mask: :func:`clean_mask` removes speckle and fills holes.
4. Summarize: :func:`change_summary` reports pixel counts, percentages and areas.

:func:`detect_changes` runs the whole pipeline in one call. For classified maps
(post-classification comparison) use :func:`transition_matrix` and
:func:`classify_change`.

Conventions
-----------
* Change is always measured as ``after`` relative to ``before``.
* Multi-band stacks are shaped ``(bands, height, width)`` (the rasterio layout).
* Invalid pixels are NaN or ±inf, equal to ``nodata`` or masked in a
  :class:`numpy.ma.MaskedArray`. They come out as NaN in magnitudes and are
  never flagged as change in masks.
* Integer inputs are processed as ``float32`` (``float64`` for 32/64-bit
  integers); ``float32`` and ``float64`` inputs keep their precision. Inputs are
  never modified.
* Undefined arithmetic (division by zero, log of zero) yields NaN without
  emitting ``RuntimeWarning``.
* Everything is deterministic.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
from scipy import ndimage

from .utils import validate_array

__all__ = [
    "CHANGE_LABELS",
    "CHANGE_NODATA",
    "GAINED",
    "LOST",
    "NO_CHANGE",
    "STABLE",
    "CVAResult",
    "ChangeResult",
    "PCAChangeResult",
    "TransitionMatrix",
    "change_summary",
    "change_vector_analysis",
    "classify_change",
    "clean_mask",
    "compute_threshold",
    "detect_changes",
    "difference",
    "normalized_difference_change",
    "otsu_threshold",
    "pca_change",
    "ratio",
    "threshold_change",
    "transition_matrix",
]

#: Class codes produced by :func:`classify_change`.
NO_CHANGE = 0  #: absent in both dates
GAINED = 1  #: absent before, present after
LOST = 2  #: present before, absent after
STABLE = 3  #: present in both dates
CHANGE_NODATA = 255  #: invalid pixel in either date

#: Human-readable names for the :func:`classify_change` codes, suitable for
#: the ``labels`` argument of :func:`change_summary`.
CHANGE_LABELS: dict[int, str] = {
    NO_CHANGE: "no_change",
    GAINED: "gained",
    LOST: "lost",
    STABLE: "stable",
}

_THRESHOLD_METHODS = ("otsu", "std", "percentile")
_DETECT_METHODS = ("difference", "ratio", "normalized_difference", "cva", "pca")
_CHUNK = 1 << 20  # pixels per chunk for PCA statistics


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
class CVAResult(NamedTuple):
    """Output of :func:`change_vector_analysis`.

    Attributes
    ----------
    magnitude : numpy.ndarray
        Euclidean length of the change vector, shape ``(H, W)``; NaN where invalid.
    direction : numpy.ndarray
        Angle of the change vector in the plane of the first two bands, in degrees
        ``[0, 360)`` measured from the band-0 axis towards the band-1 axis; NaN
        where invalid, 0 where there is no change.
    sector : numpy.ndarray
        ``int32`` sector code: bit ``i`` is set when band ``i`` increased. With two
        bands, 0 = both decreased, 1 = band 0 up, 2 = band 1 up, 3 = both up.
        ``-1`` where invalid.
    """

    magnitude: np.ndarray
    direction: np.ndarray
    sector: np.ndarray


class PCAChangeResult(NamedTuple):
    """Output of :func:`pca_change`.

    Attributes
    ----------
    components : numpy.ndarray
        Principal-component scores of the difference image, shape
        ``(n_components, H, W)``, ordered by decreasing variance; NaN where invalid.
    explained_variance_ratio : numpy.ndarray
        Fraction of the difference-image variance explained by each component.
    loadings : numpy.ndarray
        Component vectors, shape ``(n_components, bands)``. Each row is signed so
        that its largest-magnitude entry is positive (deterministic output).
    mean : numpy.ndarray
        Per-band mean of the difference image that was removed before projection.
    """

    components: np.ndarray
    explained_variance_ratio: np.ndarray
    loadings: np.ndarray
    mean: np.ndarray


class TransitionMatrix(NamedTuple):
    """Output of :func:`transition_matrix` (a "from-to" change matrix).

    Attributes
    ----------
    counts : numpy.ndarray
        ``int64`` array of shape ``(k, k)``. ``counts[i, j]`` is the number of
        pixels that were ``classes[i]`` before and ``classes[j]`` after.
    classes : numpy.ndarray
        Class values labelling the rows (before) and columns (after).
    """

    counts: np.ndarray
    classes: np.ndarray

    @property
    def total(self) -> int:
        """Number of pixels counted in the matrix."""
        return int(self.counts.sum())

    @property
    def changed(self) -> int:
        """Number of pixels whose class changed (off-diagonal sum)."""
        return int(self.counts.sum() - np.trace(self.counts))

    def normalized(self, by: str = "all") -> np.ndarray:
        """Return the matrix as fractions.

        Parameters
        ----------
        by : {"all", "before", "after"}
            ``"all"`` divides by the total, ``"before"`` makes each row sum to 1
            (where did each original class go?), ``"after"`` makes each column sum
            to 1 (where did each final class come from?). Empty rows/columns are 0.
        """
        counts = self.counts.astype(np.float64)
        if by == "all":
            denom: Any = counts.sum()
        elif by == "before":
            denom = counts.sum(axis=1, keepdims=True)
        elif by == "after":
            denom = counts.sum(axis=0, keepdims=True)
        else:
            raise ValueError(f"by must be 'all', 'before' or 'after', got {by!r}")
        return np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0)

    def areas(self, pixel_size: Any) -> np.ndarray:
        """Return the matrix in area units (squared CRS units, usually m²).

        ``pixel_size`` accepts the same forms as in :func:`change_summary`.
        """
        area = _pixel_area(pixel_size)
        if area is None:
            raise ValueError("pixel_size is required to compute areas")
        return self.counts * area


class ChangeResult(NamedTuple):
    """Output of :func:`detect_changes`.

    Attributes
    ----------
    magnitude : numpy.ndarray
        Non-negative change magnitude, shape ``(H, W)``; NaN where invalid.
    mask : numpy.ndarray
        Boolean change mask, shape ``(H, W)``; always False where invalid.
    threshold : float
        Threshold applied to ``magnitude`` (pixels with ``magnitude > threshold``
        are change, before mask cleanup).
    method : str
        Change measure that produced ``magnitude``.
    """

    magnitude: np.ndarray
    mask: np.ndarray
    threshold: float
    method: str

    def summary(self, pixel_size: Any = None) -> dict[str, Any]:
        """Summarize the change mask; see :func:`change_summary`.

        Invalid pixels (NaN magnitude) are reported as nodata, not as unchanged.
        """
        return change_summary(self.mask, pixel_size=pixel_size, valid=np.isfinite(self.magnitude))


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _float_dtype(dtype: np.dtype) -> np.dtype:
    result = np.result_type(dtype, np.float32)
    if result.itemsize > 8:
        return np.dtype(np.float64)
    return result


def _check_kind(array: np.ndarray, name: str) -> None:
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must have a boolean, integer or float dtype, got {array.dtype}")


def _prepare(array: Any, name: str, nodata: float | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(float data, invalid mask or None)`` without copying float input."""
    invalid = None
    if isinstance(array, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(array)
        invalid = mask if mask.any() else None
        array = np.asarray(array.data)
    validate_array(array, name)
    _check_kind(array, name)
    data = array.astype(_float_dtype(array.dtype), copy=False)
    bad = ~np.isfinite(data)
    if nodata is not None and not np.isnan(nodata):
        bad |= data == nodata
    if invalid is not None:
        bad |= invalid
    return data, (bad if bad.any() else None)


def _prepare_stack(
    array: Any, name: str, nodata: float | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Like :func:`_prepare` for ``(bands, H, W)`` stacks; 2-D input is one band."""
    data, bad = _prepare(array, name, nodata)
    if data.ndim == 2:
        data = data[np.newaxis]
        bad = None if bad is None else bad[np.newaxis]
    elif data.ndim != 3:
        raise ValueError(
            f"{name} must be a 2-D image or a 3-D stack shaped (bands, height, width), "
            f"got shape {data.shape}"
        )
    return data, (None if bad is None else bad.any(axis=0))


def _check_same_shape(a: np.ndarray, b: np.ndarray, names: tuple[str, str]) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same shape, got {a.shape} and {b.shape}; "
            "make sure both rasters are co-registered (same grid and extent)"
        )


def _union(*masks: np.ndarray | None) -> np.ndarray | None:
    present = [m for m in masks if m is not None]
    if not present:
        return None
    out = present[0].copy()
    for m in present[1:]:
        out |= m
    return out


def _set_nan(out: np.ndarray, *bad: np.ndarray | None) -> np.ndarray:
    for m in bad:
        if m is not None:
            out[m] = np.nan
    return out


def _pair(
    before: Any, after: Any, nodata: float | None
) -> tuple[np.ndarray, np.ndarray, np.dtype, np.ndarray | None]:
    """Prepare two images: ``(before, after, output dtype, invalid mask or None)``."""
    b, bad_b = _prepare(before, "before", nodata)
    a, bad_a = _prepare(after, "after", nodata)
    _check_same_shape(b, a, ("before", "after"))
    return b, a, np.result_type(b.dtype, a.dtype), _union(bad_b, bad_a)


def _finite_values(values: Any, name: str = "values") -> np.ndarray:
    data = np.asarray(values.filled(np.nan) if np.ma.isMaskedArray(values) else values)
    _check_kind(data, name)
    data = data.astype(_float_dtype(data.dtype), copy=False)
    finite = np.isfinite(data)
    out = data.ravel() if finite.all() else data[finite]
    if out.size == 0:
        raise ValueError(f"{name} contains no finite values")
    return out


def _to_bool(mask: Any, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(bool mask, invalid or None)``; NaN / masked entries are invalid."""
    invalid = None
    if isinstance(mask, np.ma.MaskedArray):
        m = np.ma.getmaskarray(mask)
        invalid = m if m.any() else None
        mask = np.asarray(mask.data)
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(mask)}")
    if mask.size == 0:
        raise ValueError(f"{name} cannot be empty")
    _check_kind(mask, name)
    if mask.dtype == bool:
        return mask, invalid
    if mask.dtype.kind == "f":
        nan = np.isnan(mask)
        if nan.any():
            invalid = nan if invalid is None else invalid | nan
            return (mask != 0) & ~nan, invalid
    return mask != 0, invalid


def _pixel_area(pixel_size: Any) -> float | None:
    """Area of one pixel from a size, (x, y) pair, Affine, rasterio meta or dataset."""
    if pixel_size is None:
        return None
    if isinstance(pixel_size, Mapping):
        if "transform" not in pixel_size:
            raise ValueError("pixel_size mapping must contain a 'transform' entry (rasterio meta)")
        crs = pixel_size.get("crs")
        transform = pixel_size["transform"]
        if getattr(transform, "is_identity", False) and (crs is None or pixel_size.get("gcps")):
            # rasterio reports an identity transform for rasters without a geotransform
            # (GCP-only drone images or plain images): its "pixel size" of 1 is not a
            # ground size, and reporting 1 m² per pixel would be silently wrong.
            raise ValueError(
                "pixel_size metadata has no geotransform (identity transform"
                + (" with GCPs" if pixel_size.get("gcps") else " and no CRS")
                + "), so pixel areas are unknown. Rectify/align the raster first "
                "(farq.georef.rectify / align_pair) or pass the pixel size explicitly."
            )
        if crs is not None and getattr(crs, "is_geographic", False):
            warnings.warn(
                "The raster uses a geographic CRS (degrees); areas are in squared degrees, "
                "not m². Reproject to a projected CRS for meaningful areas.",
                UserWarning,
                stacklevel=3,
            )
        return _pixel_area(pixel_size["transform"])
    if all(hasattr(pixel_size, attr) for attr in ("a", "b", "d", "e")):  # affine.Affine
        t = pixel_size
        return abs(float(t.a) * float(t.e) - float(t.b) * float(t.d))
    if hasattr(pixel_size, "transform") and not isinstance(pixel_size, np.ndarray):
        return _pixel_area(
            {"transform": pixel_size.transform, "crs": getattr(pixel_size, "crs", None)}
        )
    if np.isscalar(pixel_size) and not isinstance(pixel_size, (bool, str)):
        size = float(pixel_size)
        if not np.isfinite(size) or size <= 0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}")
        return size * size
    if isinstance(pixel_size, (tuple, list, np.ndarray)) and len(pixel_size) == 2:
        x, y = (abs(float(v)) for v in pixel_size)
        if not (np.isfinite(x) and np.isfinite(y)) or x == 0 or y == 0:
            raise ValueError(f"pixel_size must be non-zero and finite, got {pixel_size}")
        return x * y
    raise TypeError(
        "pixel_size must be a number, an (x, y) pair, an affine.Affine transform, "
        f"a rasterio meta/profile dict or an open dataset, got {type(pixel_size)}"
    )


def _value_counts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized unique/count; uses bincount for small non-negative integers."""
    if values.size == 0:
        return values[:0], np.zeros(0, dtype=np.int64)
    if values.dtype == bool:
        counts = np.bincount(values.view(np.uint8), minlength=2)
        return np.array([False, True]), counts.astype(np.int64)
    if values.dtype.kind in "iu":
        lo, hi = int(values.min()), int(values.max())
        if lo >= 0 and hi <= 1 << 20:
            counts = np.bincount(values.astype(np.intp, copy=False), minlength=hi + 1)
            present = np.flatnonzero(counts)
            return present.astype(values.dtype), counts[present].astype(np.int64)
    uniq, counts = np.unique(values, return_counts=True)
    return uniq, counts.astype(np.int64)


# --------------------------------------------------------------------------- #
# Pixel-wise change measures
# --------------------------------------------------------------------------- #
def difference(
    before: np.ndarray,
    after: np.ndarray,
    *,
    nodata: float | None = None,
    absolute: bool = False,
) -> np.ndarray:
    """Image differencing: ``after - before``.

    Parameters
    ----------
    before, after : numpy.ndarray
        Co-registered images (any matching shape, e.g. ``(H, W)`` or
        ``(bands, H, W)``). Masked arrays are supported.
    nodata : float, optional
        Sentinel value marking invalid pixels in either input. NaN/inf are always
        treated as invalid.
    absolute : bool, default False
        Return ``|after - before|`` instead of the signed difference.

    Returns
    -------
    numpy.ndarray
        Float array (``float32`` unless an input is ``float64``/wide integer) with
        NaN where either input is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import difference
    >>> difference(np.array([[1.0, 2.0]]), np.array([[3.0, 1.0]]))
    array([[ 2., -1.]])
    """
    b, a, dt, bad = _pair(before, after, nodata)
    with np.errstate(invalid="ignore", over="ignore"):
        out = np.subtract(a, b, dtype=dt)
        if absolute:
            np.abs(out, out=out)
    return _set_nan(out, bad)


def ratio(
    before: np.ndarray,
    after: np.ndarray,
    *,
    log: bool = True,
    nodata: float | None = None,
) -> np.ndarray:
    """Image ratioing: ``ln(after / before)`` (or ``after / before``).

    The log-ratio is the standard change measure for SAR intensity because it
    turns multiplicative speckle into additive noise and treats increases and
    decreases symmetrically. Multiply by ``10 / ln(10)`` to express it in dB.

    Parameters
    ----------
    before, after : numpy.ndarray
        Co-registered, non-negative images (e.g. reflectance or SAR intensity).
    log : bool, default True
        Return the natural-log ratio. If False, return the plain ratio.
    nodata : float, optional
        Sentinel value marking invalid pixels in either input.

    Returns
    -------
    numpy.ndarray
        Float array; NaN where an input is invalid or the ratio is undefined
        (``before == 0``, or a zero/negative ratio when ``log=True``).

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import ratio
    >>> np.round(ratio(np.array([1.0, 2.0, 0.0]), np.array([np.e, 2.0, 1.0])), 3)
    array([ 1.,  0., nan])
    """
    b, a, dt, bad = _pair(before, after, nodata)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.divide(a, b, dtype=dt)
        if log:
            np.log(out, out=out)
    out[~np.isfinite(out)] = np.nan
    return _set_nan(out, bad)


def normalized_difference_change(
    before: np.ndarray,
    after: np.ndarray,
    *,
    nodata: float | None = None,
) -> np.ndarray:
    """Normalized difference ``(after - before) / (after + before)``.

    Bounded in ``[-1, 1]`` for non-negative inputs, which makes thresholds
    comparable between scenes with different brightness.

    Parameters
    ----------
    before, after : numpy.ndarray
        Co-registered, non-negative images.
    nodata : float, optional
        Sentinel value marking invalid pixels in either input.

    Returns
    -------
    numpy.ndarray
        Float array; NaN where an input is invalid or ``after + before == 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import normalized_difference_change
    >>> normalized_difference_change(np.array([1.0, 2.0]), np.array([3.0, 2.0]))
    array([0.5, 0. ])
    """
    b, a, dt, bad = _pair(before, after, nodata)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.subtract(a, b, dtype=dt)
        out /= np.add(a, b, dtype=dt)
    out[~np.isfinite(out)] = np.nan
    return _set_nan(out, bad)


def change_vector_analysis(
    before_stack: np.ndarray,
    after_stack: np.ndarray,
    *,
    nodata: float | None = None,
) -> CVAResult:
    """Change Vector Analysis (CVA) for multi-band images.

    The per-pixel change vector is ``after_stack - before_stack`` across bands.
    Its length measures *how much* a pixel changed; its direction tells *how*.

    Parameters
    ----------
    before_stack, after_stack : numpy.ndarray
        Co-registered stacks shaped ``(bands, H, W)``; a 2-D array is one band.
    nodata : float, optional
        Sentinel value; a pixel is invalid if any band in either stack is invalid.

    Returns
    -------
    CVAResult
        ``(magnitude, direction, sector)``; see :class:`CVAResult`.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import change_vector_analysis
    >>> before = np.zeros((2, 1, 2))
    >>> after = np.array([[[3.0, 0.0]], [[4.0, -1.0]]])
    >>> cva = change_vector_analysis(before, after)
    >>> cva.magnitude
    array([[5., 1.]])
    >>> cva.direction.round(1)
    array([[ 53.1, 270. ]])
    """
    b, bad_b = _prepare_stack(before_stack, "before_stack", nodata)
    a, bad_a = _prepare_stack(after_stack, "after_stack", nodata)
    _check_same_shape(b, a, ("before_stack", "after_stack"))
    n_bands = b.shape[0]
    if n_bands > 31:
        raise ValueError(f"change_vector_analysis supports at most 31 bands, got {n_bands}")
    dt = np.result_type(b.dtype, a.dtype)
    bad = _union(bad_b, bad_a)

    with np.errstate(invalid="ignore", over="ignore"):
        sumsq = np.zeros(b.shape[1:], dtype=dt)
        sector = np.zeros(b.shape[1:], dtype=np.int32)
        d0 = d1 = None
        for i in range(n_bands):
            d = np.subtract(a[i], b[i], dtype=dt)
            sector |= (d > 0).astype(np.int32) << i
            if i == 0:
                d0 = d
            elif i == 1:
                d1 = d
            sumsq += d * d
        magnitude = np.sqrt(sumsq, out=sumsq)
        if d1 is None:
            d1 = np.zeros_like(d0)
        direction = np.degrees(np.arctan2(d1, d0)).astype(dt, copy=False)
        direction %= 360.0
        direction[magnitude == 0] = 0.0

    if bad is not None:
        magnitude[bad] = np.nan
        direction[bad] = np.nan
        sector[bad] = -1
    return CVAResult(magnitude, direction, sector)


def pca_change(
    before_stack: np.ndarray,
    after_stack: np.ndarray,
    *,
    n_components: int | None = None,
    standardize: bool = False,
    nodata: float | None = None,
) -> PCAChangeResult:
    """Principal Component Analysis of the multi-band difference image.

    PCA decorrelates the band differences: the first component concentrates the
    dominant change signal while global offsets (e.g. illumination) are removed
    by centering. Statistics are accumulated in ``float64`` over pixel chunks, so
    memory stays proportional to the image, not to ``bands²·pixels``.

    Parameters
    ----------
    before_stack, after_stack : numpy.ndarray
        Co-registered stacks shaped ``(bands, H, W)``; a 2-D array is one band.
    n_components : int, optional
        Number of components to return (default: all bands).
    standardize : bool, default False
        Scale each band difference to unit variance first (correlation PCA);
        useful when bands have very different ranges.
    nodata : float, optional
        Sentinel value; invalid pixels are excluded from the statistics and are
        NaN in the output.

    Returns
    -------
    PCAChangeResult
        ``(components, explained_variance_ratio, loadings, mean)``.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import pca_change
    >>> rng = np.random.default_rng(0)
    >>> before = rng.normal(size=(3, 50, 50))
    >>> after = before.copy()
    >>> after[:, 10:20, 10:20] += 5.0
    >>> res = pca_change(before, after, n_components=1)
    >>> res.components.shape
    (1, 50, 50)
    """
    b, bad_b = _prepare_stack(before_stack, "before_stack", nodata)
    a, bad_a = _prepare_stack(after_stack, "after_stack", nodata)
    _check_same_shape(b, a, ("before_stack", "after_stack"))
    n_bands, height, width = b.shape
    if n_components is None:
        n_components = n_bands
    if not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= n_bands:
        raise ValueError(f"n_components must be an integer in [1, {n_bands}], got {n_components}")

    dt = np.result_type(b.dtype, a.dtype)
    bad = _union(bad_b, bad_a)
    with np.errstate(invalid="ignore", over="ignore"):
        diff = np.subtract(a, b, dtype=dt).reshape(n_bands, -1)
    valid = None if bad is None else ~bad.ravel()
    n_valid = diff.shape[1] if valid is None else int(valid.sum())
    if n_valid < 2:
        raise ValueError("pca_change needs at least 2 valid pixels")

    def chunks():
        for start in range(0, diff.shape[1], _CHUNK):
            block = diff[:, start : start + _CHUNK].astype(np.float64)
            if valid is not None:
                block = block[:, valid[start : start + _CHUNK]]
            yield block

    mean = sum(blk.sum(axis=1) for blk in chunks()) / n_valid
    cov = np.zeros((n_bands, n_bands))
    for blk in chunks():
        blk -= mean[:, None]
        cov += blk @ blk.T
    cov /= n_valid - 1

    scale = np.ones(n_bands)
    if standardize:
        std = np.sqrt(np.diag(cov))
        scale = np.where(std > 0, std, 1.0)
        cov = cov / np.outer(scale, scale)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    loadings = eigvecs[:, order].T  # (bands, bands), rows are components
    signs = np.sign(loadings[np.arange(n_bands), np.argmax(np.abs(loadings), axis=1)])
    loadings *= np.where(signs == 0, 1.0, signs)[:, None]
    total = eigvals.sum()
    evr = eigvals / total if total > 0 else np.zeros_like(eigvals)

    loadings = loadings[:n_components]
    projection = loadings / scale[None, :]
    components = np.empty((n_components, diff.shape[1]), dtype=dt)
    for start in range(0, diff.shape[1], _CHUNK):
        stop = start + _CHUNK
        block = diff[:, start:stop].astype(np.float64) - mean[:, None]
        components[:, start:stop] = projection @ block
    components = components.reshape(n_components, height, width)
    if bad is not None:
        components[:, bad] = np.nan
    return PCAChangeResult(components, evr[:n_components], loadings, mean)


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #
def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold, computed from a histogram in pure NumPy.

    Chooses the value that maximizes the between-class variance of the two groups
    it separates. NaN and ±inf are ignored. When several split points are equally
    good (an empty gap between two modes), the middle of the gap is returned.
    Classify with ``values > threshold``.

    Parameters
    ----------
    values : array_like
        Values to split (any shape).
    bins : int, default 256
        Histogram bins; more bins give a finer threshold.

    Returns
    -------
    float
        The threshold. If all finite values are equal, that value is returned.

    Raises
    ------
    ValueError
        If there are no finite values or ``bins < 2``.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import otsu_threshold
    >>> round(otsu_threshold(np.array([0.0, 0.1, 0.2, 0.9, 1.0, np.nan])), 2)
    0.55
    """
    if int(bins) < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")
    v = _finite_values(values)
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        return lo
    hist, edges = np.histogram(v, bins=int(bins), range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = np.cumsum(hist * centers) / w1
        m2 = (np.cumsum((hist * centers)[::-1]) / w2[::-1])[::-1]
        between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    between = np.nan_to_num(between, nan=-1.0)
    best = np.flatnonzero(between == between.max())
    # Take the first plateau of maxima and return the middle of the gap it spans.
    breaks = np.flatnonzero(np.diff(best) != 1)
    first = best[0]
    last = best[breaks[0]] if breaks.size else best[-1]
    return float((centers[first] + centers[last + 1]) / 2.0)


def compute_threshold(
    values: np.ndarray,
    method: str | float = "otsu",
    *,
    k: float = 2.0,
    percentile: float = 95.0,
    bins: int = 256,
) -> float:
    """Compute a change threshold from data.

    Parameters
    ----------
    values : array_like
        Change magnitudes (NaN/inf ignored).
    method : {"otsu", "std", "percentile"} or float, default "otsu"
        * ``"otsu"`` — :func:`otsu_threshold`; best for bimodal histograms.
        * ``"std"`` — ``mean + k * std``; flags statistical outliers.
        * ``"percentile"`` — the given percentile; flags a fixed share of pixels.
        * a number — used as-is.
    k : float, default 2.0
        Number of standard deviations for ``"std"``.
    percentile : float, default 95.0
        Percentile in ``[0, 100]`` for ``"percentile"``.
    bins : int, default 256
        Histogram bins for ``"otsu"``.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import compute_threshold
    >>> compute_threshold(np.arange(101.0), "percentile", percentile=90)
    90.0
    """
    if isinstance(method, (bool, np.bool_)):
        raise TypeError("method must be a string or a number, not a bool")
    if isinstance(method, (int, float, np.integer, np.floating)):
        value = float(method)
        if not np.isfinite(value):
            raise ValueError(f"threshold must be finite, got {method}")
        return value
    if method == "otsu":
        return otsu_threshold(values, bins=bins)
    if method == "std":
        v = _finite_values(values)
        return float(v.mean(dtype=np.float64) + k * v.std(dtype=np.float64))
    if method == "percentile":
        if not 0.0 <= float(percentile) <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        return float(np.percentile(_finite_values(values), percentile))
    raise ValueError(
        f"Unknown threshold method {method!r}; use one of {_THRESHOLD_METHODS} or a number"
    )


def threshold_change(
    magnitude: np.ndarray,
    method: str | float = "otsu",
    *,
    k: float = 2.0,
    percentile: float = 95.0,
    absolute: bool = False,
    bins: int = 256,
) -> np.ndarray:
    """Turn a change magnitude into a boolean change mask.

    Pixels with ``magnitude > threshold`` are change; NaN pixels never are.

    Parameters
    ----------
    magnitude : numpy.ndarray
        Change magnitude (e.g. from :func:`difference` or CVA).
    method : {"otsu", "std", "percentile"} or float, default "otsu"
        See :func:`compute_threshold`.
    k, percentile, bins
        See :func:`compute_threshold`.
    absolute : bool, default False
        Threshold ``|magnitude|``; use this for *signed* measures such as a
        difference or log-ratio, so that both increases and decreases count.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the shape of ``magnitude``.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import threshold_change
    >>> threshold_change(np.array([-3.0, 0.1, 2.5, np.nan]), 1.0, absolute=True)
    array([ True, False,  True, False])
    """
    data = np.ma.filled(magnitude, np.nan) if np.ma.isMaskedArray(magnitude) else magnitude
    if not isinstance(data, np.ndarray):
        raise TypeError(f"magnitude must be a numpy array, got {type(magnitude)}")
    _check_kind(data, "magnitude")
    if absolute:
        data = np.abs(data)
    t = compute_threshold(data, method, k=k, percentile=percentile, bins=bins)
    return data > t


# --------------------------------------------------------------------------- #
# Mask cleanup
# --------------------------------------------------------------------------- #
def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def clean_mask(
    mask: np.ndarray,
    *,
    min_size: int = 0,
    connectivity: int = 8,
    fill_holes: bool | int = False,
) -> np.ndarray:
    """Remove small change patches and optionally fill holes in a 2-D mask.

    Parameters
    ----------
    mask : numpy.ndarray
        2-D boolean (or 0/1) mask. NaN entries count as False.
    min_size : int, default 0
        Connected regions with fewer pixels than this are removed.
    connectivity : {4, 8}, default 8
        Pixel neighbourhood for regions (8 includes diagonals). Holes use the
        complementary connectivity, as is standard in digital topology.
    fill_holes : bool or int, default False
        ``True`` fills every background region fully enclosed by change;
        an integer fills only holes with at most that many pixels.

    Returns
    -------
    numpy.ndarray
        Cleaned boolean mask (a new array).

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import clean_mask
    >>> m = np.zeros((6, 6), bool)
    >>> m[1:4, 1:4] = True
    >>> m[2, 2] = False  # a hole
    >>> m[5, 5] = True   # speckle
    >>> cleaned = clean_mask(m, min_size=2, fill_holes=True)
    >>> int(cleaned.sum()), bool(cleaned[5, 5])
    (9, False)
    """
    data, _ = _to_bool(mask, "mask")
    if data.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {data.shape}")
    if int(min_size) < 0:
        raise ValueError(f"min_size must be >= 0, got {min_size}")
    structure = _structure(connectivity)
    out = data.copy()

    if min_size > 1 and out.any():
        labels, _ = ndimage.label(out, structure=structure)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        out = keep[labels]

    if fill_holes is not False and fill_holes is not None:
        if isinstance(fill_holes, (bool, np.bool_)):
            max_hole = np.inf
        else:
            max_hole = int(fill_holes)
            if max_hole < 0:
                raise ValueError(
                    f"fill_holes must be a bool or a non-negative int, got {fill_holes}"
                )
        background = ~out
        if background.any() and max_hole > 0:
            labels, _ = ndimage.label(background, structure=_structure(12 - connectivity))
            sizes = np.bincount(labels.ravel())
            touches_border = np.zeros(sizes.size, dtype=bool)
            for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
                touches_border[edge] = True
            fill = ~touches_border & (sizes <= max_hole)
            fill[0] = False
            out |= fill[labels]
    return out


# --------------------------------------------------------------------------- #
# Categorical change
# --------------------------------------------------------------------------- #
def classify_change(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Compare two binary maps (e.g. water / not water) pixel by pixel.

    Parameters
    ----------
    before_mask, after_mask : numpy.ndarray
        Boolean (or 0/1) masks of the same shape. NaN or masked entries are
        treated as invalid.
    valid : numpy.ndarray, optional
        Extra boolean mask; pixels where it is False are invalid.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array with codes :data:`NO_CHANGE` (0, absent in both),
        :data:`GAINED` (1), :data:`LOST` (2), :data:`STABLE` (3, present in both)
        and :data:`CHANGE_NODATA` (255, invalid). :data:`CHANGE_LABELS` maps codes
        to names.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import classify_change
    >>> classify_change(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    array([0, 1, 2, 3], dtype=uint8)
    """
    b, bad_b = _to_bool(before_mask, "before_mask")
    a, bad_a = _to_bool(after_mask, "after_mask")
    _check_same_shape(b, a, ("before_mask", "after_mask"))
    out = b.astype(np.uint8) * np.uint8(2)
    out += a
    bad = _union(bad_b, bad_a)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        _check_same_shape(b, valid, ("before_mask", "valid"))
        bad = _union(bad, ~valid)
    if bad is not None:
        out[bad] = CHANGE_NODATA
    return out


def transition_matrix(
    class_before: np.ndarray,
    class_after: np.ndarray,
    classes: Any = None,
    *,
    nodata: float | None = None,
) -> TransitionMatrix:
    """Post-classification comparison: count pixels for every from→to pair.

    Parameters
    ----------
    class_before, class_after : numpy.ndarray
        Classified maps of the same shape (integer, boolean or float labels).
    classes : sequence, optional
        Class values (and their order) for the rows/columns. Defaults to the
        sorted union of classes present in either map. Pixels whose class is
        not listed are ignored.
    nodata : float, optional
        Label marking invalid pixels; NaN labels are always ignored.

    Returns
    -------
    TransitionMatrix
        ``counts[i, j]`` = pixels that went from ``classes[i]`` to ``classes[j]``.
        Use ``.normalized(by=...)``, ``.areas(pixel_size)``, ``.changed``.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import transition_matrix
    >>> tm = transition_matrix(np.array([1, 1, 2, 2]), np.array([1, 2, 2, 2]))
    >>> tm.classes
    array([1, 2])
    >>> tm.counts
    array([[1, 1],
           [0, 2]])
    """
    for arr, name in ((class_before, "class_before"), (class_after, "class_after")):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr)}")
        if arr.size == 0:
            raise ValueError(f"{name} cannot be empty")
        _check_kind(arr, name)
    _check_same_shape(class_before, class_after, ("class_before", "class_after"))

    def invalid(arr: np.ndarray) -> np.ndarray | None:
        bad = None
        if arr.dtype.kind == "f":
            bad = ~np.isfinite(arr)
        if nodata is not None and not np.isnan(nodata):
            bad = (arr == nodata) if bad is None else bad | (arr == nodata)
        return bad

    bad = _union(invalid(class_before), invalid(class_after))
    if bad is None or not bad.any():
        b, a = class_before.ravel(), class_after.ravel()
    else:
        b, a = class_before[~bad], class_after[~bad]

    if classes is None:
        cls = np.union1d(_value_counts(b)[0], _value_counts(a)[0])
    else:
        cls = np.asarray(classes).ravel()
        if cls.size == 0:
            raise ValueError("classes cannot be empty")
        if np.unique(cls).size != cls.size:
            raise ValueError("classes must not contain duplicates")
    n = cls.size
    if n == 0:
        raise ValueError("no valid pixels to compare")

    if b.dtype.kind in "biu" and a.dtype.kind in "biu" and b.size:
        lo = min(int(b.min()), int(a.min()))
        size = max(int(b.max()), int(a.max())) + 1
        if lo >= 0 and size * size <= 1 << 24:
            # Fast path for typical land-cover codes: one bincount over the raw
            # (before, after) code pairs, then pick the requested rows/columns.
            flat = b.astype(np.intp) * size
            flat += a
            full = np.bincount(flat, minlength=size * size).reshape(size, size)
            in_range = np.array(
                [float(c).is_integer() and 0 <= c < size for c in cls.tolist()], dtype=bool
            )
            idx = np.where(in_range, cls, 0).astype(np.intp)
            counts = full[np.ix_(idx, idx)].astype(np.int64)
            counts[~in_range, :] = 0
            counts[:, ~in_range] = 0
            return TransitionMatrix(counts, cls)

    order = np.argsort(cls, kind="stable")
    sorted_cls = cls[order]

    def index(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.clip(np.searchsorted(sorted_cls, values), 0, n - 1)
        return order[pos], sorted_cls[pos] == values

    ib, ok_b = index(b)
    ia, ok_a = index(a)
    ok = ok_b & ok_a
    flat = ib * n + ia
    if not ok.all():
        flat = flat[ok]
    counts = np.bincount(flat, minlength=n * n).reshape(n, n).astype(np.int64)
    return TransitionMatrix(counts, cls)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def change_summary(
    data: np.ndarray,
    *,
    pixel_size: Any = None,
    labels: Mapping[Any, str] | None = None,
    nodata: float | None = None,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pixel counts, percentages and areas for a change mask or class map.

    Parameters
    ----------
    data : numpy.ndarray
        A boolean change mask, or a categorical map (e.g. from
        :func:`classify_change`).
    pixel_size : optional
        Pixel footprint, used for areas. One of: a number (square pixels), an
        ``(x, y)`` pair, an ``affine.Affine`` transform, a rasterio
        ``meta``/``profile`` dict or an open rasterio dataset. Areas are in squared
        CRS units — m² for projected CRSs such as UTM (a warning is raised when a
        rasterio meta/dataset reports a geographic CRS).
    labels : mapping, optional
        Names for class values, e.g. :data:`CHANGE_LABELS`. Labelled classes
        appear even with zero pixels. Boolean masks default to
        ``{False: "unchanged", True: "changed"}``.
    nodata : float, optional
        Class value marking invalid pixels (e.g. :data:`CHANGE_NODATA`). NaN is
        always invalid.
    valid : numpy.ndarray, optional
        Boolean mask; pixels where it is False are invalid.

    Returns
    -------
    dict
        ``total_pixels``, ``valid_pixels``, ``nodata_pixels``, ``pixel_area_m2``
        (None if unknown) and ``classes`` — a dict keyed by label with ``value``,
        ``pixels``, ``percent`` (of valid pixels), ``area_m2`` and ``area_km2``
        (None if unknown). Boolean masks also get top-level ``changed_pixels``,
        ``changed_percent``, ``changed_area_m2`` and ``changed_area_km2``.
        All numbers are plain Python ``int``/``float`` (JSON-serializable).

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import change_summary
    >>> mask = np.zeros((10, 10), bool)
    >>> mask[:5, :2] = True
    >>> s = change_summary(mask, pixel_size=30)
    >>> s["changed_pixels"], s["changed_percent"], s["changed_area_km2"]
    (10, 10.0, 0.009)
    """
    if np.ma.isMaskedArray(data):
        mvalid = ~np.ma.getmaskarray(data)
        valid = mvalid if valid is None else (np.asarray(valid, dtype=bool) & mvalid)
        data = np.asarray(data.data)
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a numpy array, got {type(data)}")
    if data.size == 0:
        raise ValueError("data cannot be empty")
    _check_kind(data, "data")
    area = _pixel_area(pixel_size)

    bad = None
    if data.dtype.kind == "f":
        bad = np.isnan(data)
    if nodata is not None and not np.isnan(nodata):
        bad = _union(bad, data == nodata)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        _check_same_shape(data, valid, ("data", "valid"))
        bad = _union(bad, ~valid)
    values = data.ravel() if bad is None else data[~bad]

    is_bool = data.dtype == bool
    if labels is None and is_bool:
        labels = {False: "unchanged", True: "changed"}
    present, counts = _value_counts(values)
    count_of = dict(zip(present.tolist(), counts.tolist()))
    for key in labels or {}:
        if nodata is not None and key == nodata:
            continue
        count_of.setdefault(bool(key) if is_bool else key, 0)

    total = int(data.size)
    n_valid = int(values.size)

    def entry(value: Any, n: int) -> dict[str, Any]:
        pct = 100.0 * n / n_valid if n_valid else 0.0
        a_m2 = None if area is None else float(n * area)
        return {
            "value": value,
            "pixels": int(n),
            "percent": float(pct),
            "area_m2": a_m2,
            "area_km2": None if a_m2 is None else a_m2 / 1e6,
        }

    classes: dict[Any, dict[str, Any]] = {}
    for value in sorted(count_of):
        name = labels.get(value, value) if labels else value
        classes[name] = entry(value, count_of[value])

    summary: dict[str, Any] = {
        "total_pixels": total,
        "valid_pixels": n_valid,
        "nodata_pixels": total - n_valid,
        "pixel_area_m2": area,
        "classes": classes,
    }
    if is_bool:
        changed = entry(True, count_of.get(True, 0))
        summary["changed_pixels"] = changed["pixels"]
        summary["changed_percent"] = changed["percent"]
        summary["changed_area_m2"] = changed["area_m2"]
        summary["changed_area_km2"] = changed["area_km2"]
    return summary


# --------------------------------------------------------------------------- #
# One-call workflow
# --------------------------------------------------------------------------- #
def detect_changes(
    before: np.ndarray,
    after: np.ndarray,
    method: str = "difference",
    threshold: str | float = "otsu",
    *,
    k: float = 2.0,
    percentile: float = 95.0,
    min_size: int = 0,
    connectivity: int = 8,
    fill_holes: bool | int = False,
    nodata: float | None = None,
) -> ChangeResult:
    """Detect changes between two co-registered images in one call.

    Computes a change magnitude, thresholds it and cleans the resulting mask.

    Parameters
    ----------
    before, after : numpy.ndarray
        Images of the same area at two dates. ``(H, W)`` for single-band methods;
        ``(H, W)`` or ``(bands, H, W)`` for ``"cva"`` and ``"pca"``.
    method : {"difference", "ratio", "normalized_difference", "cva", "pca"}
        Change measure. The magnitude is the absolute value of the signed measure:
        ``|after - before|``, ``|ln(after / before)|`` (recommended for SAR),
        ``|normalized difference|``, the CVA vector length, or ``|PC1|`` of the
        difference image.
    threshold : {"otsu", "std", "percentile"} or float, default "otsu"
        Thresholding rule applied to the magnitude; see :func:`compute_threshold`.
    k, percentile
        Parameters for the ``"std"`` and ``"percentile"`` rules.
    min_size : int, default 0
        Remove change regions smaller than this many pixels.
    connectivity : {4, 8}, default 8
        Neighbourhood used by ``min_size`` and ``fill_holes``.
    fill_holes : bool or int, default False
        Fill holes inside change regions (see :func:`clean_mask`).
    nodata : float, optional
        Sentinel value marking invalid pixels in the inputs.

    Returns
    -------
    ChangeResult
        ``(magnitude, mask, threshold, method)``; call ``.summary(pixel_size)`` for
        counts and areas. Invalid pixels are NaN in ``magnitude`` and False in
        ``mask``.

    Examples
    --------
    >>> import numpy as np
    >>> from farq.change import detect_changes
    >>> before = np.full((50, 50), 0.2, dtype=np.float32)
    >>> after = before.copy()
    >>> after[10:20, 10:20] = 0.8          # a 100-pixel change
    >>> result = detect_changes(before, after, method="difference", min_size=5)
    >>> int(result.mask.sum())
    100
    >>> result.summary(pixel_size=10)["changed_area_m2"]
    10000.0
    """
    if method not in _DETECT_METHODS:
        raise ValueError(f"Unknown method {method!r}; use one of {_DETECT_METHODS}")

    if method in ("cva", "pca"):
        if method == "cva":
            magnitude = change_vector_analysis(before, after, nodata=nodata).magnitude
        else:
            magnitude = pca_change(before, after, n_components=1, nodata=nodata).components[0]
            np.abs(magnitude, out=magnitude)
    else:
        before, after = _single_band(before, "before", method), _single_band(after, "after", method)
        if method == "difference":
            magnitude = difference(before, after, nodata=nodata)
        elif method == "ratio":
            magnitude = ratio(before, after, nodata=nodata)
        else:
            magnitude = normalized_difference_change(before, after, nodata=nodata)
        np.abs(magnitude, out=magnitude)

    t = compute_threshold(magnitude, threshold, k=k, percentile=percentile)
    valid = np.isfinite(magnitude)
    mask = magnitude > t
    if min_size > 1 or fill_holes:
        mask = clean_mask(mask, min_size=min_size, connectivity=connectivity, fill_holes=fill_holes)
        mask &= valid
    return ChangeResult(magnitude, mask, float(t), method)


def _single_band(array: Any, name: str, method: str) -> Any:
    shape = getattr(array, "shape", None)
    if shape is not None and len(shape) == 3 and shape[0] == 1:
        return array[0]
    if shape is not None and len(shape) != 2:
        raise ValueError(
            f"method {method!r} expects single-band 2-D arrays, but {name} has shape {shape}; "
            "use method='cva' or method='pca' for multi-band stacks"
        )
    return array
