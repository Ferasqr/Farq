"""
Spectral indices module for the Farq library.

This module provides functions for calculating common spectral indices:

- NDWI (Normalized Difference Water Index, McFeeters 1996)
- MNDWI (Modified Normalized Difference Water Index, Xu 2006)
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil Adjusted Vegetation Index)
- NDBI (Normalized Difference Built-up Index)
- NBR (Normalized Burn Ratio)
- NDMI (Normalized Difference Moisture Index)

RGB-only indices for cameras without a NIR band (e.g. drones): VARI, ExG, ExR, ExGR,
GLI, NGRDI and TGI.

Conventions shared by all index functions:

- Bands are converted to floating point (``float32`` for inputs that fit, such as
  ``uint16`` digital numbers or ``float32`` reflectance, otherwise ``float64``), so
  integer inputs never overflow. Inputs are never modified.
- Pixels where the index is undefined (a zero denominator, e.g. ``0 / 0``) are NaN, and
  NaN inputs propagate to NaN outputs. No ``RuntimeWarning`` is emitted.
- With ``clip=True`` (default), finite results are clipped to ``[-1, 1]``.
- Landsat 8/9 OLI band numbers are given in the docstrings for reference.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from numbers import Real

import numpy as np

from .core import validate_bands

__all__ = [
    "calculate_indices",
    "calculate_normalized_difference",
    "evi",
    "exg",
    "exgr",
    "exr",
    "gli",
    "mndwi",
    "nbr",
    "ndbi",
    "ndmi",
    "ndvi",
    "ndwi",
    "ngrdi",
    "savi",
    "tgi",
    "validate_bands",
    "vari",
]


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, clip: bool) -> np.ndarray:
    """
    Divide in place, writing NaN where the denominator is zero, and optionally clip.

    ``numerator`` must be a freshly allocated float array; it is reused as the output.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=numerator)
    numerator[denominator == 0] = np.nan
    if clip:
        np.clip(numerator, -1.0, 1.0, out=numerator)
    return numerator


def calculate_normalized_difference(
    band1: np.ndarray, band2: np.ndarray, clip: bool = True
) -> np.ndarray:
    """
    Calculate the normalized difference ``(band1 - band2) / (band1 + band2)``.

    Args:
        band1: First band array.
        band2: Second band array, with the same shape as ``band1``.
        clip: Whether to clip values to the ``[-1, 1]`` range (only reachable outside
            that range with negative inputs).

    Returns:
        Floating point array of normalized differences. Pixels where
        ``band1 + band2 == 0`` are NaN.

    Raises:
        TypeError: If a band is not a numeric numpy array.
        ValueError: If a band is empty or the shapes differ.
    """
    b1, b2 = validate_bands(band1, band2)
    return _safe_ratio(b1 - b2, b1 + b2, clip)


def ndvi(
    nir: np.ndarray,
    red: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).

    ``NDVI = (NIR - Red) / (NIR + Red)``

    Args:
        nir: Near-infrared band (Landsat 8/9 B5).
        red: Red band (Landsat 8/9 B4).
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result because the ratio is scale invariant.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        NDVI array in ``[-1, 1]``; values above about 0.2 indicate vegetation.
    """
    nir, red = validate_bands(nir, red)
    _check_scale(reflectance_scale)
    return _safe_ratio(nir - red, nir + red, clip)


def ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Normalized Difference Water Index (NDWI, McFeeters 1996).

    ``NDWI = (Green - NIR) / (Green + NIR)``

    Open water typically has values above 0.

    Args:
        green: Green band (Landsat 8/9 B3).
        nir: Near-infrared band (Landsat 8/9 B5).
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result because the ratio is scale invariant.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        NDWI array in ``[-1, 1]``.
    """
    green, nir = validate_bands(green, nir)
    _check_scale(reflectance_scale)
    return _safe_ratio(green - nir, green + nir, clip)


def mndwi(
    green: np.ndarray,
    swir1: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Modified Normalized Difference Water Index (MNDWI, Xu 2006).

    ``MNDWI = (Green - SWIR1) / (Green + SWIR1)``

    MNDWI separates open water from built-up land better than NDWI. Water typically has
    values above 0.

    Args:
        green: Green band (Landsat 8/9 B3).
        swir1: Short-wave infrared band 1 (Landsat 8/9 B6).
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result because the ratio is scale invariant.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        MNDWI array in ``[-1, 1]``.
    """
    green, swir1 = validate_bands(green, swir1)
    _check_scale(reflectance_scale)
    return _safe_ratio(green - swir1, green + swir1, clip)


def ndbi(
    swir1: np.ndarray,
    nir: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Normalized Difference Built-up Index (NDBI).

    ``NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)``

    Args:
        swir1: Short-wave infrared band 1 (Landsat 8/9 B6).
        nir: Near-infrared band (Landsat 8/9 B5).
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result because the ratio is scale invariant.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        NDBI array in ``[-1, 1]``; higher values indicate built-up areas.
    """
    swir1, nir = validate_bands(swir1, nir)
    _check_scale(reflectance_scale)
    return _safe_ratio(swir1 - nir, swir1 + nir, clip)


def nbr(
    nir: np.ndarray,
    swir2: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Normalized Burn Ratio (NBR).

    ``NBR = (NIR - SWIR2) / (NIR + SWIR2)``

    Args:
        nir: Near-infrared band (Landsat 8/9 B5).
        swir2: Short-wave infrared band 2 (Landsat 8/9 B7).
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result because the ratio is scale invariant.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        NBR array in ``[-1, 1]``; lower values indicate burned areas.
    """
    nir, swir2 = validate_bands(nir, swir2)
    _check_scale(reflectance_scale)
    return _safe_ratio(nir - swir2, nir + swir2, clip)


def ndmi(
    nir: np.ndarray,
    swir1: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Normalized Difference Moisture Index (NDMI).

    ``NDMI = (NIR - SWIR1) / (NIR + SWIR1)``

    Args:
        nir: Near-infrared band (Landsat 8/9 B5).
        swir1: Short-wave infrared band 1 (Landsat 8/9 B6).
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result because the ratio is scale invariant.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        NDMI array in ``[-1, 1]``; higher values indicate higher moisture content.
    """
    nir, swir1 = validate_bands(nir, swir1)
    _check_scale(reflectance_scale)
    return _safe_ratio(nir - swir1, nir + swir1, clip)


def evi(
    red: np.ndarray,
    nir: np.ndarray,
    blue: np.ndarray,
    reflectance_scale: float | None = None,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Enhanced Vegetation Index (EVI, Huete et al. 2002).

    ``EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)``

    Unlike the normalized difference indices, EVI depends on the absolute reflectance
    values: bands must be surface reflectance in ``[0, 1]``. Pass ``reflectance_scale``
    (e.g. ``10000``) if the bands are stored as scaled integers.

    Args:
        red: Red band (Landsat 8/9 B4).
        nir: Near-infrared band (Landsat 8/9 B5).
        blue: Blue band (Landsat 8/9 B2).
        reflectance_scale: Scale factor to convert the bands to reflectance.
        G: Gain factor (default 2.5).
        C1: Aerosol resistance coefficient for the red band (default 6.0).
        C2: Aerosol resistance coefficient for the blue band (default 7.5).
        L: Canopy background adjustment (default 1.0).
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        EVI array; pixels with a zero denominator are NaN.

    Raises:
        TypeError: If a band or coefficient has the wrong type.
        ValueError: If bands are invalid, ``G`` is not positive or ``L`` is negative.
    """
    red, nir, blue = validate_bands(red, nir, blue, reflectance_scale=reflectance_scale)
    _check_coefficients(G=G, C1=C1, C2=C2, L=L)
    if L < 0:
        raise ValueError("L must be non-negative")
    if G <= 0:
        raise ValueError("G must be positive")

    dtype = nir.dtype.type
    denominator = nir + dtype(C1) * red
    denominator -= dtype(C2) * blue
    denominator += dtype(L)
    numerator = nir - red
    numerator *= dtype(G)
    return _safe_ratio(numerator, denominator, clip)


def savi(
    nir: np.ndarray,
    red: np.ndarray,
    reflectance_scale: float | None = None,
    L: float = 0.5,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Soil Adjusted Vegetation Index (SAVI, Huete 1988).

    ``SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L)``

    Like EVI, SAVI depends on absolute reflectance values in ``[0, 1]``; pass
    ``reflectance_scale`` for scaled integer data.

    Args:
        nir: Near-infrared band (Landsat 8/9 B5).
        red: Red band (Landsat 8/9 B4).
        reflectance_scale: Scale factor to convert the bands to reflectance.
        L: Soil brightness correction factor in ``[0, 1]`` (default 0.5). ``L = 0``
            gives NDVI.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        SAVI array; pixels with a zero denominator are NaN.

    Raises:
        TypeError: If a band or ``L`` has the wrong type.
        ValueError: If bands are invalid or ``L`` is outside ``[0, 1]``.
    """
    nir, red = validate_bands(nir, red, reflectance_scale=reflectance_scale)
    _check_coefficients(L=L)
    if not 0 <= L <= 1:
        raise ValueError("L must be between 0 and 1")

    dtype = nir.dtype.type
    numerator = nir - red
    numerator *= dtype(1 + L)
    denominator = nir + red
    denominator += dtype(L)
    return _safe_ratio(numerator, denominator, clip)


# --------------------------------------------------------------------------- RGB indices
# Visible-band indices for consumer RGB cameras (e.g. drones) without a NIR band. They
# accept 8-bit image bands directly. Ratio-based indices are scale invariant, so
# ``reflectance_scale`` only matters for :func:`tgi`.


def _chromatic_index(
    weights: tuple[float, float, float],
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    reflectance_scale: float | None,
) -> np.ndarray:
    """``wr*r + wg*g + wb*b`` on chromatic coordinates ``r = R / (R + G + B)`` etc."""
    red, green, blue = validate_bands(red, green, blue)
    _check_scale(reflectance_scale)
    dtype = red.dtype.type
    wr, wg, wb = (dtype(w) for w in weights)
    numerator = wr * red
    numerator += wg * green
    numerator += wb * blue
    total = red + green
    total += blue
    return _safe_ratio(numerator, total, clip=False)


def exg(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    reflectance_scale: float | None = None,
) -> np.ndarray:
    """
    Calculate the Excess Green index (ExG, Woebbecke et al. 1995).

    ``ExG = 2g - r - b`` on chromatic coordinates ``r = R / (R + G + B)``,
    ``g = G / (R + G + B)``, ``b = B / (R + G + B)``.

    Args:
        red: Red band.
        green: Green band.
        blue: Blue band.
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result.

    Returns:
        ExG array in ``[-1, 2]``; vegetation typically has values above about 0.1.
        Pixels with ``R + G + B == 0`` are NaN.
    """
    return _chromatic_index((-1.0, 2.0, -1.0), red, green, blue, reflectance_scale)


def exr(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    reflectance_scale: float | None = None,
) -> np.ndarray:
    """
    Calculate the Excess Red index (ExR, Meyer et al. 1998).

    ``ExR = 1.4r - g`` on chromatic coordinates (see :func:`exg`).

    Args:
        red: Red band.
        green: Green band.
        blue: Blue band (used for the chromatic normalization).
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result.

    Returns:
        ExR array in ``[-1, 1.4]``; soil and residue have higher values than vegetation.
        Pixels with ``R + G + B == 0`` are NaN.
    """
    return _chromatic_index((1.4, -1.0, 0.0), red, green, blue, reflectance_scale)


def exgr(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    reflectance_scale: float | None = None,
) -> np.ndarray:
    """
    Calculate the Excess Green minus Excess Red index (ExGR, Meyer & Neto 2008).

    ``ExGR = ExG - ExR = 3g - 2.4r - b`` on chromatic coordinates (see :func:`exg`).

    Args:
        red: Red band.
        green: Green band.
        blue: Blue band.
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result.

    Returns:
        ExGR array in ``[-2.4, 3]``; values above 0 indicate vegetation.
        Pixels with ``R + G + B == 0`` are NaN.
    """
    return _chromatic_index((-2.4, 3.0, -1.0), red, green, blue, reflectance_scale)


def gli(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Green Leaf Index (GLI, Louhaichi et al. 2001).

    ``GLI = (2G - R - B) / (2G + R + B)``

    Args:
        red: Red band.
        green: Green band.
        blue: Blue band.
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        GLI array in ``[-1, 1]``; positive values indicate green vegetation.
    """
    red, green, blue = validate_bands(red, green, blue)
    _check_scale(reflectance_scale)
    two_green = green + green
    numerator = two_green - red
    numerator -= blue
    denominator = two_green
    denominator += red
    denominator += blue
    return _safe_ratio(numerator, denominator, clip)


def ngrdi(
    red: np.ndarray,
    green: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Normalized Green Red Difference Index (NGRDI, Tucker 1979).

    ``NGRDI = (G - R) / (G + R)``

    Args:
        red: Red band.
        green: Green band.
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        NGRDI array in ``[-1, 1]``; positive values indicate green vegetation.
    """
    red, green = validate_bands(red, green)
    _check_scale(reflectance_scale)
    return _safe_ratio(green - red, green + red, clip)


def vari(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the Visible Atmospherically Resistant Index (VARI, Gitelson et al. 2002).

    ``VARI = (G - R) / (G + R - B)``

    The denominator can approach zero for some colours, producing extreme values;
    ``clip=True`` (default) limits the result to ``[-1, 1]``.

    Args:
        red: Red band.
        green: Green band.
        blue: Blue band.
        reflectance_scale: Accepted for API consistency and validated; it does not
            change the result.
        clip: Whether to clip values to ``[-1, 1]``.

    Returns:
        VARI array; pixels with ``G + R - B == 0`` are NaN.
    """
    red, green, blue = validate_bands(red, green, blue)
    _check_scale(reflectance_scale)
    denominator = green + red
    denominator -= blue
    return _safe_ratio(green - red, denominator, clip)


def tgi(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    reflectance_scale: float | None = None,
    *,
    wavelengths: tuple[float, float, float] = (670.0, 550.0, 480.0),
) -> np.ndarray:
    """
    Calculate the Triangular Greenness Index (TGI, Hunt et al. 2011).

    ``TGI = -0.5 * [(λr - λb)(R - G) - (λr - λg)(R - B)]``

    With the default band centres (670, 550, 480 nm) this is
    ``TGI = -0.5 * [190 (R - G) - 120 (R - B)]``. TGI is not normalized: it scales with
    the input, so pass ``reflectance_scale`` (e.g. ``255`` for 8-bit images) to get
    reflectance-like values.

    Args:
        red: Red band.
        green: Green band.
        blue: Blue band.
        reflectance_scale: Optional scale factor to divide the bands by.
        wavelengths: Centre wavelengths ``(red, green, blue)`` in nm of the sensor.

    Returns:
        TGI array; higher values indicate more chlorophyll.
    """
    red, green, blue = validate_bands(red, green, blue, reflectance_scale=reflectance_scale)
    if len(wavelengths) != 3:
        raise ValueError("wavelengths must be a (red, green, blue) tuple")
    lr, lg, lb = wavelengths
    _check_coefficients(red_wavelength=lr, green_wavelength=lg, blue_wavelength=lb)
    dtype = red.dtype.type
    out = red - green
    out *= dtype(-0.5 * (lr - lb))
    rb = red - blue
    rb *= dtype(0.5 * (lr - lg))
    out += rb
    return out


def _check_coefficients(**coefficients: float) -> None:
    for name, value in coefficients.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")


def _check_scale(reflectance_scale: float | None) -> None:
    if reflectance_scale is not None:
        _check_coefficients(reflectance_scale=reflectance_scale)
        if not reflectance_scale > 0:
            raise ValueError(f"reflectance_scale must be positive, got {reflectance_scale}")


# Index name -> (required band names, function)
_INDICES: dict[str, tuple[tuple[str, ...], Callable[..., np.ndarray]]] = {
    "ndvi": (("nir", "red"), ndvi),
    "ndwi": (("green", "nir"), ndwi),
    "mndwi": (("green", "swir1"), mndwi),
    "evi": (("red", "nir", "blue"), evi),
    "savi": (("nir", "red"), savi),
    "ndbi": (("swir1", "nir"), ndbi),
    "nbr": (("nir", "swir2"), nbr),
    "ndmi": (("nir", "swir1"), ndmi),
    "vari": (("red", "green", "blue"), vari),
    "exg": (("red", "green", "blue"), exg),
    "exr": (("red", "green", "blue"), exr),
    "exgr": (("red", "green", "blue"), exgr),
    "gli": (("red", "green", "blue"), gli),
    "ngrdi": (("red", "green"), ngrdi),
    "tgi": (("red", "green", "blue"), tgi),
}


def calculate_indices(
    bands: Mapping[str, np.ndarray],
    indices: str | Iterable[str],
    reflectance_scale: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Calculate multiple spectral indices at once.

    Args:
        bands: Mapping of band name to array. Recognised names are ``"blue"``,
            ``"green"``, ``"red"``, ``"nir"``, ``"swir1"`` and ``"swir2"``.
        indices: Index name or names to calculate (case-insensitive): ``"ndvi"``,
            ``"ndwi"``, ``"mndwi"``, ``"evi"``, ``"savi"``, ``"ndbi"``, ``"nbr"``,
            ``"ndmi"``, and the RGB-only ``"vari"``, ``"exg"``, ``"exr"``, ``"exgr"``,
            ``"gli"``, ``"ngrdi"``, ``"tgi"``.
        reflectance_scale: Scale factor for reflectance data (only affects EVI, SAVI
            and TGI).

    Returns:
        Dictionary mapping each lower-case index name to its array.

    Raises:
        ValueError: If an index name is unknown or a required band is missing.

    Example:
        >>> bands = {"red": red, "nir": nir, "green": green}
        >>> result = calculate_indices(bands, ["ndvi", "ndwi"], reflectance_scale=10000)
        >>> ndvi_array = result["ndvi"]
    """
    names = [indices] if isinstance(indices, str) else list(indices)

    plan = []
    for name in names:
        key = name.lower()
        if key not in _INDICES:
            raise ValueError(f"Unknown index: {name!r}. Available: {', '.join(_INDICES)}")
        required, func = _INDICES[key]
        missing = [band for band in required if band not in bands]
        if missing:
            raise ValueError(f"Missing required bands for {key}: {missing}")
        plan.append((key, required, func))

    return {
        key: func(*(bands[b] for b in required), reflectance_scale=reflectance_scale)
        for key, required, func in plan
    }
