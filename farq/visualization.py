"""Visualization helpers for raster analysis.

The functions in this module draw single rasters, side-by-side comparisons, change
maps, value distributions and RGB composites with matplotlib.

Conventions shared by every function:

* Nothing is shown implicitly: each function returns the
  :class:`~matplotlib.figure.Figure` it drew on, so callers can ``savefig`` it,
  tweak it, or display it with ``farq.plt.show()``.
* Pass ``ax=`` (single-panel functions) or ``axes=`` (two-panel functions) to
  draw into existing axes of your own figure. In that case ``figsize`` is ignored,
  the layout of your figure is left untouched and the parent figure of the given
  axes is returned.
* NaN (and +/-inf) values are treated as no-data: they are left blank in images,
  skipped in histograms, ignored when computing color limits and contrast
  stretches, and rendered transparent in RGB composites.
* Histograms and percentile stretches operate on a reproducible random sample of at
  most ``max_samples`` pixels, so they stay fast on very large rasters.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np

from .utils import validate_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

__all__ = [
    "changes",
    "compare",
    "compare_rgb",
    "distribution_comparison",
    "hist",
    "plot",
    "plot_rgb",
]

#: Default maximum number of pixels used for histograms and percentile stretches.
DEFAULT_MAX_SAMPLES = 1_000_000

_SAMPLE_SEED = 0

Bins = Union[int, str, Sequence[float]]
RGBBands = Sequence[np.ndarray]


# --------------------------------------------------------------------------- helpers


def _check_image(data: object, name: str = "data") -> np.ndarray:
    """Validate a 2D raster and return it (bools are promoted to ``uint8``)."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(data).__name__}")
    if data.size == 0:
        raise ValueError(f"{name} is empty")
    if data.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {data.shape}")
    if data.dtype == bool:
        data = data.astype(np.uint8)
    validate_array(data, name=name)
    return data


def _check_scale(reflectance_scale: float | None) -> None:
    if reflectance_scale is not None and reflectance_scale == 0:
        raise ValueError("reflectance_scale must be non-zero")


def _apply_scale(data: np.ndarray, reflectance_scale: float | None) -> np.ndarray:
    """Divide by ``reflectance_scale`` (no copy when it is ``None``)."""
    if reflectance_scale is None:
        return data
    return data / reflectance_scale


def _as_float(data: np.ndarray) -> np.ndarray:
    """Return ``data`` as floats with masked entries replaced by NaN."""
    if isinstance(data, np.ma.MaskedArray):
        return data.astype(float).filled(np.nan)
    if data.dtype.kind in "fc":
        return data
    return data.astype(float)


def _finite_sample(data: np.ndarray, max_samples: int | None) -> tuple[np.ndarray, float]:
    """Return finite values of ``data`` (randomly subsampled) and the sample weight.

    The weight is the number of original pixels represented by each sampled pixel,
    so ``weight * counts`` estimates counts over the full array.
    """
    flat = _as_float(data).ravel()
    weight = 1.0
    if max_samples is not None and flat.size > max_samples:
        rng = np.random.default_rng(_SAMPLE_SEED)
        idx = rng.integers(0, flat.size, size=max_samples)
        weight = flat.size / max_samples
        flat = flat[idx]
    return flat[np.isfinite(flat)], weight


def _finite_range(*arrays: np.ndarray) -> tuple[float | None, float | None]:
    """Return the (min, max) over the finite values of all arrays, or ``(None, None)``."""
    lo, hi = np.inf, -np.inf
    for arr in arrays:
        if arr.dtype.kind in "iub" and not isinstance(arr, np.ma.MaskedArray):
            lo, hi = min(lo, float(arr.min())), max(hi, float(arr.max()))
            continue
        values = _as_float(arr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
            a_lo, a_hi = np.nanmin(values), np.nanmax(values)
        if not (np.isfinite(a_lo) and np.isfinite(a_hi)):
            # Only pay for a filtered copy when infinities (or all-NaN) are present.
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            a_lo, a_hi = finite.min(), finite.max()
        lo = min(lo, float(a_lo))
        hi = max(hi, float(a_hi))
    if lo > hi:
        return None, None
    return lo, hi


def _new_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes, bool]:
    """Return ``(fig, ax, created)``, creating a new figure when ``ax`` is None."""
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, True


def _new_axes_pair(
    axes: Sequence[Axes] | None, figsize: tuple[float, float]
) -> tuple[Figure, Axes, Axes, bool]:
    """Return ``(fig, ax1, ax2, created)``, creating a 1x2 figure when ``axes`` is None."""
    if axes is not None:
        axes = list(np.ravel(np.asarray(axes, dtype=object)))
        if len(axes) != 2:
            raise ValueError(f"axes must contain exactly 2 Axes, got {len(axes)}")
        return axes[0].figure, axes[0], axes[1], False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    return fig, ax1, ax2, True


def _draw_image(
    ax: Axes,
    data: np.ndarray,
    *,
    title: str | None,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    colorbar: bool,
    colorbar_label: str | None,
) -> AxesImage:
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return im


def _draw_hist(
    ax: Axes,
    values: np.ndarray,
    weight: float,
    *,
    bins: Bins,
    density: bool,
    alpha: float,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    weights = None if density or weight == 1.0 else np.full(values.shape, weight)
    ax.hist(values, bins=bins, density=density, alpha=alpha, weights=weights)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else ("Density" if density else "Count"))
    ax.grid(True, alpha=0.3)


def _hist_values(
    data: np.ndarray | Sequence[float],
    name: str,
    reflectance_scale: float | None,
    max_samples: int | None,
) -> tuple[np.ndarray, float]:
    arr = np.asanyarray(data)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    values, weight = _finite_sample(arr, max_samples)
    if values.size == 0:
        raise ValueError(f"{name} contains no finite values")
    return _apply_scale(values, reflectance_scale), weight


def _check_rgb_params(gamma: float, percentile: float | None) -> None:
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if percentile is not None and not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")


def _check_bands(bands: RGBBands, name: str) -> tuple[int, int]:
    if isinstance(bands, np.ndarray) and bands.ndim != 3:
        raise ValueError(f"{name} must be a (3, rows, cols) array or three 2D arrays")
    if len(bands) != 3:
        raise ValueError(
            f"{name} must be a (3, rows, cols) array or a sequence of three 2D arrays "
            f"(red, green, blue), got {len(bands)} bands"
        )
    for band, color in zip(bands, ("red", "green", "blue")):
        _check_image(band, name=f"{name} {color} band" if name else f"{color} band")
    shapes = [b.shape for b in bands]
    if len(set(shapes)) != 1:
        raise ValueError(f"All bands must have the same shape, got {shapes}")
    return shapes[0]


def _rgb_composite(
    bands: RGBBands,
    *,
    scale_factor: float,
    gamma: float,
    percentile: float | None,
    reflectance_scale: float | None,
    max_samples: int | None,
) -> np.ndarray:
    """Build a float32 RGBA image in [0, 1]; pixels with any non-finite band are transparent."""
    height, width = bands[0].shape
    rgba = np.empty((height, width, 4), dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    for i, band in enumerate(bands):
        if isinstance(band, np.ma.MaskedArray):
            channel = band.astype(np.float32).filled(np.nan)
        else:
            channel = np.array(band, dtype=np.float32)  # always a private copy
        if reflectance_scale is not None:
            channel /= reflectance_scale
        finite = np.isfinite(channel)
        valid &= finite
        if percentile is not None:
            sample, _ = _finite_sample(channel, max_samples)
            if sample.size:
                ref = float(np.percentile(sample, percentile))
                if not ref > 0:
                    ref = float(sample.max())
                if ref > 0:
                    channel /= ref
        if scale_factor != 1.0:
            channel *= scale_factor
        channel[~finite] = 0.0
        np.clip(channel, 0.0, 1.0, out=channel)
        rgba[..., i] = channel
    if gamma != 1.0:
        np.power(rgba[..., :3], 1.0 / gamma, out=rgba[..., :3])
    rgba[..., 3] = valid
    return rgba


def _draw_rgb(ax: Axes, rgba: np.ndarray, title: str | None) -> None:
    ax.imshow(rgba, interpolation="nearest")
    if title:
        ax.set_title(title)
    ax.set_axis_off()


# ----------------------------------------------------------------------- public API


def plot(
    data: np.ndarray,
    title: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 8),
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    reflectance_scale: float | None = None,
    *,
    ax: Axes | None = None,
    colorbar: bool = True,
) -> Figure:
    """Plot a single 2D raster.

    Args:
        data: 2D array to display. NaN/inf pixels are left blank.
        title: Axes title.
        cmap: Matplotlib colormap name.
        figsize: Size of the new figure (ignored when ``ax`` is given).
        vmin: Lower color limit (default: data minimum, ignoring NaN/inf).
        vmax: Upper color limit (default: data maximum, ignoring NaN/inf).
        colorbar_label: Label for the colorbar.
        reflectance_scale: Divide the data by this factor before plotting
            (e.g. ``10000`` for Landsat 8 surface reflectance).
        ax: Existing axes to draw into instead of creating a new figure.
        colorbar: Whether to add a colorbar.

    Returns:
        The :class:`~matplotlib.figure.Figure` containing the plot. The image axes is
        ``fig.axes[0]`` for a new figure (or the ``ax`` you passed).

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty, not 2D, all NaN, or ``reflectance_scale`` is 0.
    """
    data = _check_image(data)
    _check_scale(reflectance_scale)
    fig, ax, created = _new_axes(ax, figsize)
    _draw_image(
        ax,
        _apply_scale(data, reflectance_scale),
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
    )
    if created:
        fig.tight_layout()
    return fig


def compare(
    data1: np.ndarray,
    data2: np.ndarray,
    title1: str | None = None,
    title2: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (15, 6),
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    reflectance_scale: float | None = None,
    *,
    axes: Sequence[Axes] | None = None,
    colorbar: bool = True,
) -> Figure:
    """Plot two rasters side by side on a shared color scale.

    Args:
        data1: First 2D array.
        data2: Second 2D array, same shape as ``data1``.
        title1: Title of the left panel.
        title2: Title of the right panel.
        cmap: Matplotlib colormap name.
        figsize: Size of the new figure (ignored when ``axes`` is given).
        vmin: Lower color limit (default: minimum over *both* arrays, ignoring NaN/inf).
        vmax: Upper color limit (default: maximum over *both* arrays, ignoring NaN/inf).
        colorbar_label: Label for both colorbars.
        reflectance_scale: Divide both arrays by this factor before plotting.
        axes: Two existing axes ``(left, right)`` to draw into.
        colorbar: Whether to add a colorbar to each panel.

    Returns:
        The :class:`~matplotlib.figure.Figure` containing both panels.

    Raises:
        TypeError: If an input is not a numpy array.
        ValueError: If inputs are empty, not 2D, all NaN, differ in shape, or
            ``axes`` does not hold exactly two axes.
    """
    data1 = _check_image(data1, "data1")
    data2 = _check_image(data2, "data2")
    if data1.shape != data2.shape:
        raise ValueError(
            f"data1 and data2 must have the same shape, got {data1.shape} and {data2.shape}"
        )
    _check_scale(reflectance_scale)
    plot1 = _apply_scale(data1, reflectance_scale)
    plot2 = _apply_scale(data2, reflectance_scale)

    if vmin is None or vmax is None:
        lo, hi = _finite_range(plot1, plot2)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    fig, ax1, ax2, created = _new_axes_pair(axes, figsize)
    for ax, values, title in ((ax1, plot1, title1), (ax2, plot2, title2)):
        _draw_image(
            ax,
            values,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
        )
    if created:
        fig.tight_layout()
    return fig


def changes(
    data: np.ndarray,
    title: str | None = None,
    cmap: str = "RdYlBu",
    figsize: tuple[float, float] = (10, 8),
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = True,
    colorbar_label: str | None = "Change",
    reflectance_scale: float | None = None,
    *,
    ax: Axes | None = None,
    colorbar: bool = True,
) -> Figure:
    """Plot a change map, by default with color limits symmetric around zero.

    Args:
        data: 2D change array (e.g. ``after - before``). NaN/inf pixels are left blank.
        title: Axes title.
        cmap: Matplotlib colormap name (a diverging map is recommended).
        figsize: Size of the new figure (ignored when ``ax`` is given).
        vmin: Lower color limit.
        vmax: Upper color limit.
        symmetric: If True, missing limits are chosen so the scale is centred on
            zero: with neither limit given, ``[-m, m]`` where ``m`` is the largest
            finite absolute value; with only one given, the other mirrors it.
        colorbar_label: Label for the colorbar.
        reflectance_scale: Divide the data by this factor before plotting.
        ax: Existing axes to draw into instead of creating a new figure.
        colorbar: Whether to add a colorbar.

    Returns:
        The :class:`~matplotlib.figure.Figure` containing the plot.

    Raises:
        TypeError: If ``data`` is not a numpy array.
        ValueError: If ``data`` is empty, not 2D, all NaN, or ``reflectance_scale`` is 0.
    """
    data = _check_image(data)
    _check_scale(reflectance_scale)
    plot_data = _apply_scale(data, reflectance_scale)

    if symmetric:
        if vmin is None and vmax is None:
            lo, hi = _finite_range(plot_data)
            abs_max = 0.0 if lo is None or hi is None else max(abs(lo), abs(hi))
            if abs_max == 0:
                abs_max = 1.0
            vmin, vmax = -abs_max, abs_max
        elif vmin is None and vmax is not None:
            vmin = -abs(vmax)
        elif vmax is None and vmin is not None:
            vmax = abs(vmin)

    fig, ax, created = _new_axes(ax, figsize)
    _draw_image(
        ax,
        plot_data,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
    )
    if created:
        fig.tight_layout()
    return fig


def hist(
    data: np.ndarray | Sequence[float],
    bins: Bins = 50,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    density: bool = True,
    xlabel: str | None = "Value",
    ylabel: str | None = None,
    alpha: float = 0.6,
    reflectance_scale: float | None = None,
    *,
    ax: Axes | None = None,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> Figure:
    """Plot a histogram of the finite values of an array of any shape.

    Args:
        data: Array or list of values; flattened before plotting. NaN/inf are ignored.
        bins: Number of bins, a bin-edge sequence, or a numpy binning strategy name.
        title: Axes title.
        figsize: Size of the new figure (ignored when ``ax`` is given).
        density: Plot a probability density instead of counts.
        xlabel: X-axis label.
        ylabel: Y-axis label (default: ``"Density"`` or ``"Count"``).
        alpha: Bar transparency.
        reflectance_scale: Divide the values by this factor before plotting.
        ax: Existing axes to draw into instead of creating a new figure.
        max_samples: Use a reproducible random sample of at most this many values
            (``None`` uses all). With ``density=False`` the counts are rescaled to
            estimate counts over the full array.

    Returns:
        The :class:`~matplotlib.figure.Figure` containing the histogram.

    Raises:
        ValueError: If ``data`` is empty or has no finite values, or
            ``reflectance_scale`` is 0.
    """
    _check_scale(reflectance_scale)
    values, weight = _hist_values(data, "data", reflectance_scale, max_samples)
    fig, ax, created = _new_axes(ax, figsize)
    _draw_hist(
        ax,
        values,
        weight,
        bins=bins,
        density=density,
        alpha=alpha,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    if created:
        fig.tight_layout()
    return fig


def distribution_comparison(
    data1: np.ndarray | Sequence[float],
    data2: np.ndarray | Sequence[float],
    title1: str | None = None,
    title2: str | None = None,
    bins: Bins = 50,
    figsize: tuple[float, float] = (12, 6),
    density: bool = True,
    xlabel: str | None = "Value",
    ylabel: str | None = None,
    alpha: float = 0.6,
    reflectance_scale: float | None = None,
    *,
    axes: Sequence[Axes] | None = None,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> Figure:
    """Plot histograms of two datasets side by side using identical bin edges.

    Args:
        data1: First dataset (array of any shape, or list). NaN/inf are ignored.
        data2: Second dataset. Shapes need not match.
        title1: Title of the left panel.
        title2: Title of the right panel.
        bins: Number of bins, a bin-edge sequence, or a numpy binning strategy name.
            Edges are computed from both datasets combined so panels are comparable.
        figsize: Size of the new figure (ignored when ``axes`` is given).
        density: Plot probability densities instead of counts.
        xlabel: X-axis label.
        ylabel: Y-axis label (default: ``"Density"`` or ``"Count"``).
        alpha: Bar transparency.
        reflectance_scale: Divide the values by this factor before plotting.
        axes: Two existing axes ``(left, right)`` to draw into.
        max_samples: Per-dataset sample size limit, see :func:`hist`.

    Returns:
        The :class:`~matplotlib.figure.Figure` containing both histograms.

    Raises:
        ValueError: If a dataset is empty or has no finite values, ``axes`` does not
            hold exactly two axes, or ``reflectance_scale`` is 0.
    """
    _check_scale(reflectance_scale)
    values1, weight1 = _hist_values(data1, "data1", reflectance_scale, max_samples)
    values2, weight2 = _hist_values(data2, "data2", reflectance_scale, max_samples)
    edges = np.histogram_bin_edges(np.concatenate([values1, values2]), bins=bins)

    fig, ax1, ax2, created = _new_axes_pair(axes, figsize)
    for ax, values, weight, title in (
        (ax1, values1, weight1, title1),
        (ax2, values2, weight2, title2),
    ):
        _draw_hist(
            ax,
            values,
            weight,
            bins=edges,
            density=density,
            alpha=alpha,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
    if created:
        fig.tight_layout()
    return fig


def plot_rgb(
    red: np.ndarray,
    green: np.ndarray | None = None,
    blue: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    scale_factor: float = 1.0,
    gamma: float = 1.0,
    percentile: float | None = 98.0,
    reflectance_scale: float | None = None,
    *,
    ax: Axes | None = None,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> Figure:
    """Plot an RGB composite with a per-band percentile contrast stretch.

    Each band is divided by its ``percentile``-th percentile (computed over finite
    values only), multiplied by ``scale_factor``, clipped to ``[0, 1]`` and then
    gamma-corrected. Pixels where any band is NaN/inf are drawn transparent.

    Args:
        red: Red band (2D), or a ``(3, rows, cols)`` red/green/blue stack such as
            ``farq.read(path, band=[1, 2, 3])[0]`` (then omit ``green`` and ``blue``).
        green: Green band, same shape as ``red``.
        blue: Blue band, same shape as ``red``.
        title: Axes title.
        figsize: Size of the new figure (ignored when ``ax`` is given).
        scale_factor: Brightness multiplier applied after the stretch.
        gamma: Gamma correction (``> 1`` brightens dark tones).
        percentile: Percentile mapped to full brightness, in ``(0, 100]``. ``None``
            disables the stretch (values are only scaled and clipped to ``[0, 1]``).
        reflectance_scale: Divide the bands by this factor first.
        ax: Existing axes to draw into instead of creating a new figure.
        max_samples: Sample size limit for estimating percentiles.

    Returns:
        The :class:`~matplotlib.figure.Figure` containing the composite.

    Raises:
        TypeError: If a band is not a numpy array.
        ValueError: If bands are empty, not 2D, all NaN, differ in shape, or
            ``gamma``/``percentile``/``reflectance_scale`` are out of range.
    """
    if green is None and blue is None:
        if not isinstance(red, np.ndarray) or red.ndim != 3:
            raise ValueError(
                "Pass red, green and blue 2D arrays, or a single (3, rows, cols) stack "
                "such as farq.read(path, band=[1, 2, 3])[0]"
            )
        bands: RGBBands = red
    elif green is None or blue is None:
        raise ValueError("Pass both green and blue, or a single (3, rows, cols) stack")
    else:
        bands = (red, green, blue)
    _check_bands(bands, "")
    _check_rgb_params(gamma, percentile)
    _check_scale(reflectance_scale)
    rgba = _rgb_composite(
        bands,
        scale_factor=scale_factor,
        gamma=gamma,
        percentile=percentile,
        reflectance_scale=reflectance_scale,
        max_samples=max_samples,
    )
    fig, ax, created = _new_axes(ax, figsize)
    _draw_rgb(ax, rgba, title)
    if created:
        fig.tight_layout()
    return fig


def compare_rgb(
    rgb1: RGBBands,
    rgb2: RGBBands,
    title1: str | None = None,
    title2: str | None = None,
    figsize: tuple[float, float] = (15, 6),
    scale_factor: float = 1.0,
    gamma: float = 1.0,
    percentile: float | None = 98.0,
    reflectance_scale: float | None = None,
    *,
    axes: Sequence[Axes] | None = None,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> Figure:
    """Plot two RGB composites side by side.

    Each image is stretched independently as described in :func:`plot_rgb`.

    Args:
        rgb1: ``(red, green, blue)`` arrays, or a ``(3, rows, cols)`` stack, of the
            first image.
        rgb2: ``(red, green, blue)`` arrays of the second image, same shape as ``rgb1``.
        title1: Title of the left panel.
        title2: Title of the right panel.
        figsize: Size of the new figure (ignored when ``axes`` is given).
        scale_factor: Brightness multiplier applied after the stretch.
        gamma: Gamma correction.
        percentile: Percentile mapped to full brightness, or ``None`` for no stretch.
        reflectance_scale: Divide the bands by this factor first.
        axes: Two existing axes ``(left, right)`` to draw into.
        max_samples: Sample size limit for estimating percentiles.

    Returns:
        The :class:`~matplotlib.figure.Figure` containing both composites.

    Raises:
        TypeError: If a band is not a numpy array.
        ValueError: If an image does not have three bands, bands are empty, not 2D,
            all NaN or differ in shape, the two images differ in shape, or a
            parameter is out of range.
    """
    shape1 = _check_bands(rgb1, "rgb1")
    shape2 = _check_bands(rgb2, "rgb2")
    if shape1 != shape2:
        raise ValueError(f"rgb1 and rgb2 must have the same shape, got {shape1} and {shape2}")
    _check_rgb_params(gamma, percentile)
    _check_scale(reflectance_scale)

    fig, ax1, ax2, created = _new_axes_pair(axes, figsize)
    for ax, bands, title in ((ax1, rgb1, title1), (ax2, rgb2, title2)):
        rgba = _rgb_composite(
            bands,
            scale_factor=scale_factor,
            gamma=gamma,
            percentile=percentile,
            reflectance_scale=reflectance_scale,
            max_samples=max_samples,
        )
        _draw_rgb(ax, rgba, title)
    if created:
        fig.tight_layout()
    return fig
