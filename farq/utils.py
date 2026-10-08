"""
Utility functions for array operations.

This module provides NaN-aware array statistics with input validation. Several
functions intentionally share names with Python builtins (``min``, ``max``, ``sum``)
so they can be used as ``farq.min(array)`` and so on; inside this module the builtins
are therefore shadowed.

Unless noted otherwise, functions ignore NaN values (they use the ``numpy.nan*``
family). Reductions over an axis return NaN, without warnings, for slices that contain
only NaN values.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "count_nonzero",
    "max",
    "mean",
    "median",
    "min",
    "percentile",
    "stats",
    "std",
    "sum",
    "unique",
    "validate_array",
]


def validate_array(array: np.ndarray, name: str = "array", allow_all_nan: bool = False) -> None:
    """
    Validate a numpy array input.

    Args:
        array: Input numpy array to validate.
        name: Name of the array used in error messages.
        allow_all_nan: If False (default), a floating point array whose values are all
            NaN is rejected.

    Raises:
        TypeError: If ``array`` is not a numpy array.
        ValueError: If ``array`` is empty, or contains only NaN values and
            ``allow_all_nan`` is False.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    # Only float/complex arrays can hold NaN. Checking the first element first avoids a
    # full pass (and a temporary boolean array) over large rasters in the common case.
    if (
        not allow_all_nan
        and array.dtype.kind in "fc"
        and np.isnan(array.flat[0])
        and np.isnan(array).all()
    ):
        raise ValueError(f"{name} cannot contain all NaN values")


#: Default upper bound on the number of elements (bands x rows x columns) of an output
#: raster that farq allocates for a user-requested grid (``out_shape``, ``target_shape``,
#: ``resolution``). Override with the ``FARQ_MAX_OUTPUT_PIXELS`` environment variable
#: (``0`` disables the check).
DEFAULT_MAX_OUTPUT_PIXELS = 1 << 32


def _check_output_size(shape: Sequence[int], what: str = "output") -> None:
    """
    Refuse to allocate absurdly large outputs.

    A typo in a resolution (degrees instead of metres, millimetres instead of metres)
    or ``out_shape`` easily requests trillions of pixels; allocating and filling such an
    array would exhaust memory (and may get the process OOM-killed rather than raise).
    """
    raw = os.environ.get("FARQ_MAX_OUTPUT_PIXELS", "").strip()
    try:
        limit = int(raw) if raw else DEFAULT_MAX_OUTPUT_PIXELS
    except ValueError:
        raise ValueError(f"FARQ_MAX_OUTPUT_PIXELS must be an integer, got {raw!r}") from None
    if limit <= 0:
        return
    n = 1
    for s in shape:
        n *= int(s)
    if n > limit:
        dims = " x ".join(str(int(s)) for s in shape)
        raise ValueError(
            f"Requested {what} of {dims} = {n:,} pixels exceeds the safety limit of "
            f"{limit:,}. Check the requested shape/resolution (e.g. degrees vs metres), "
            "or raise the limit with the FARQ_MAX_OUTPUT_PIXELS environment variable."
        )


def _nan_reduce(func: Any, data: np.ndarray, axis: int | None, **kwargs: Any) -> Any:
    """Apply a ``numpy.nan*`` reduction, silencing all-NaN-slice warnings."""
    validate_array(data, name="data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(data, axis=axis, **kwargs)


def stats(
    data: np.ndarray,
    percentiles: Sequence[float] = (0, 25, 50, 75, 100),
    reflectance_scale: float | None = None,
    bins: int = 50,
) -> dict[str, Any]:
    """
    Calculate comprehensive statistics for an array.

    Value statistics (min, max, mean, ...) are computed over finite values only; NaN and
    infinite values are counted separately.

    Args:
        data: Input array.
        percentiles: Percentiles to compute (default: 0, 25, 50, 75, 100).
        reflectance_scale: Scale factor for reflectance data (e.g. ``10000`` for Landsat
            8 SR). If given, the statistics of ``data / reflectance_scale`` are added
            under ``"reflectance_stats"``.
        bins: Number of histogram bins.

    Returns:
        Dictionary with the keys:

        - ``min``, ``max``, ``mean``, ``std``, ``median``, ``range``, ``variance``
        - ``skewness``, ``kurtosis`` (Fisher definition, i.e. 0 for a normal
          distribution; NaN for constant data)
        - ``percentiles``: dict mapping ``str(p)`` to the p-th percentile
        - ``non_zero``: count of finite non-zero values
        - ``zeros``, ``nan``, ``inf``: counts of zero, NaN and infinite values
        - ``valid``: count of finite values
        - ``shape``, ``size``, ``dtype``
        - ``histogram``: dict with ``counts`` and ``bin_edges`` arrays
        - ``percentages``: dict with the percentage of ``valid``, ``nan``, ``zeros`` and
          ``inf`` values
        - ``reflectance_stats`` (only if ``reflectance_scale`` is given)

        Value statistics are NaN if the array contains no finite values.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty or all NaN, or ``bins`` is not positive.
    """
    validate_array(data, name="data")
    if bins < 1:
        raise ValueError("bins must be a positive integer")
    percentiles = list(percentiles)

    arr = np.asarray(data)
    if arr.dtype.kind in "fc":
        n_nan = int(np.count_nonzero(np.isnan(arr)))
        n_inf = int(np.count_nonzero(np.isinf(arr)))
        valid = arr[np.isfinite(arr)] if n_nan or n_inf else arr.ravel()
    else:
        n_nan = n_inf = 0
        valid = arr.ravel()
    if valid.dtype == np.bool_:
        valid = valid.astype(np.uint8)

    result: dict[str, Any] = _value_stats(valid, percentiles)
    n_valid = int(valid.size)
    n_zeros = int(n_valid - np.count_nonzero(valid))
    result.update(
        {
            "non_zero": n_valid - n_zeros,
            "zeros": n_zeros,
            "nan": n_nan,
            "inf": n_inf,
            "valid": n_valid,
            "shape": arr.shape,
            "size": int(arr.size),
            "dtype": str(arr.dtype),
        }
    )

    if n_valid:
        counts, edges = np.histogram(valid, bins=bins)
    else:
        counts, edges = np.zeros(bins, dtype=np.intp), np.full(bins + 1, np.nan)
    result["histogram"] = {"counts": counts, "bin_edges": edges}

    if reflectance_scale is not None:
        if not reflectance_scale > 0:
            raise ValueError("reflectance_scale must be positive")
        scaled = _value_stats(valid / reflectance_scale, percentiles)
        result["reflectance_stats"] = {
            k: scaled[k] for k in ("min", "max", "mean", "std", "median", "percentiles")
        }

    total = arr.size
    result["percentages"] = {
        "valid": 100.0 * n_valid / total,
        "nan": 100.0 * n_nan / total,
        "zeros": 100.0 * n_zeros / total,
        "inf": 100.0 * n_inf / total,
    }
    return result


def _value_stats(valid: np.ndarray, percentiles: list[float]) -> dict[str, Any]:
    """Statistics of a 1D array of finite values (NaN everywhere if it is empty)."""
    if valid.size == 0:
        nan = float("nan")
        return {
            "min": nan,
            "max": nan,
            "mean": nan,
            "std": nan,
            "median": nan,
            "percentiles": {str(p): nan for p in percentiles},
            "range": nan,
            "variance": nan,
            "skewness": nan,
            "kurtosis": nan,
        }

    vmin = float(valid.min())
    vmax = float(valid.max())
    mean_ = float(valid.mean(dtype=np.float64))
    centered = valid.astype(np.float64) - mean_
    sq = centered * centered
    m2 = float(sq.mean())
    if m2 > 0:
        m3 = float((sq * centered).mean())
        m4 = float((sq * sq).mean())
        skewness = m3 / m2**1.5
        kurtosis = m4 / m2**2 - 3.0
    else:
        skewness = kurtosis = float("nan")

    qs = [50.0, *percentiles]
    q_values = np.percentile(valid, qs)
    return {
        "min": vmin,
        "max": vmax,
        "mean": mean_,
        "std": float(np.sqrt(m2)),
        "median": float(q_values[0]),
        "percentiles": {str(p): float(v) for p, v in zip(percentiles, q_values[1:])},
        "range": vmax - vmin,
        "variance": m2,
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
    }


def sum(data: np.ndarray, axis: int | None = None) -> Any:
    """
    Sum of array elements, ignoring NaN values.

    Args:
        data: Input array.
        axis: Axis along which to sum (``None`` for the entire array).

    Returns:
        Sum of the non-NaN elements (a scalar if ``axis`` is None, otherwise an array).

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty or all NaN.
    """
    return _nan_reduce(np.nansum, data, axis)


def mean(data: np.ndarray, axis: int | None = None) -> Any:
    """
    Mean of array elements, ignoring NaN values.

    Args:
        data: Input array.
        axis: Axis along which to compute the mean (``None`` for the entire array).

    Returns:
        Mean of the non-NaN elements. With ``axis``, slices that are all NaN give NaN.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty or all NaN.
    """
    return _nan_reduce(np.nanmean, data, axis)


def std(data: np.ndarray, axis: int | None = None, ddof: int = 0) -> Any:
    """
    Standard deviation of array elements, ignoring NaN values.

    Args:
        data: Input array.
        axis: Axis along which to compute the standard deviation (``None`` for the
            entire array).
        ddof: Delta degrees of freedom (``0`` for the population standard deviation,
            ``1`` for the sample standard deviation).

    Returns:
        Standard deviation of the non-NaN elements. With ``axis``, slices that are all
        NaN give NaN.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty or all NaN.
    """
    return _nan_reduce(np.nanstd, data, axis, ddof=ddof)


def min(data: np.ndarray, axis: int | None = None) -> Any:
    """
    Minimum of array elements, ignoring NaN values.

    Args:
        data: Input array.
        axis: Axis along which to compute the minimum (``None`` for the entire array).

    Returns:
        Minimum of the non-NaN elements. With ``axis``, slices that are all NaN give NaN.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty or all NaN.
    """
    return _nan_reduce(np.nanmin, data, axis)


def max(data: np.ndarray, axis: int | None = None) -> Any:
    """
    Maximum of array elements, ignoring NaN values.

    Args:
        data: Input array.
        axis: Axis along which to compute the maximum (``None`` for the entire array).

    Returns:
        Maximum of the non-NaN elements. With ``axis``, slices that are all NaN give NaN.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty or all NaN.
    """
    return _nan_reduce(np.nanmax, data, axis)


def median(data: np.ndarray, axis: int | None = None) -> Any:
    """
    Median of array elements, ignoring NaN values.

    Args:
        data: Input array.
        axis: Axis along which to compute the median (``None`` for the entire array).

    Returns:
        Median of the non-NaN elements. With ``axis``, slices that are all NaN give NaN.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty or all NaN.
    """
    return _nan_reduce(np.nanmedian, data, axis)


def percentile(
    data: np.ndarray, q: float | Sequence[float] | np.ndarray, axis: int | None = None
) -> Any:
    """
    The q-th percentile(s) of the data, ignoring NaN values.

    Args:
        data: Input array.
        q: Percentile or sequence of percentiles, each in ``[0, 100]``.
        axis: Axis along which to compute the percentiles (``None`` for the entire
            array).

    Returns:
        Percentile value(s) of the non-NaN elements.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty or all NaN, or ``q`` is outside ``[0, 100]``.
    """
    return _nan_reduce(np.nanpercentile, data, axis, q=q)


def count_nonzero(data: np.ndarray, axis: int | None = None) -> Any:
    """
    Count non-zero values in an array.

    Follows numpy semantics: NaN counts as non-zero. To exclude NaN values, use e.g.
    ``count_nonzero(np.nan_to_num(data))``.

    Args:
        data: Input array.
        axis: Axis along which to count (``None`` for the entire array).

    Returns:
        Number of non-zero values (an ``int`` if ``axis`` is None, otherwise an array).

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty.
    """
    validate_array(data, name="data", allow_all_nan=True)
    return np.count_nonzero(data, axis=axis)


def unique(data: np.ndarray, return_counts: bool = False) -> Any:
    """
    Sorted unique elements of an array.

    NaN values are sorted to the end (and collapsed into a single entry on
    numpy >= 1.24).

    Args:
        data: Input array.
        return_counts: If True, also return the number of occurrences of each value.

    Returns:
        Array of unique values, or a tuple ``(values, counts)`` if ``return_counts`` is
        True.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty.
    """
    validate_array(data, name="data", allow_all_nan=True)
    return np.unique(data, return_counts=return_counts)
