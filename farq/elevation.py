"""
Elevation change, terrain derivatives and cut/fill volumes from DEMs and DSMs.

Drone photogrammetry, LiDAR and stereo satellites produce digital elevation models
(DEMs, or DSMs when they include buildings and vegetation). Differencing two of them
gives a *DEM of difference* (DoD), the standard tool for measuring earthworks,
stockpiles, mining progress, erosion and deposition. This module covers the whole
workflow:

1. **Terrain derivatives**: :func:`slope`, :func:`aspect` and :func:`hillshade`, with
   the pixel spacing taken from the geotransform.
2. **Co-registration**: :func:`vertical_offset` estimates a vertical bias over stable
   ground; :func:`coregister_dem` implements the Nuth & Kääb (2011) horizontal and
   vertical co-registration and :func:`shift_dem` applies a shift.
3. **Differencing**: :func:`elevation_change` computes ``after - before``.
4. **Uncertainty**: :func:`level_of_detection` and :func:`significant_change`
   threshold the DoD at a minimum level of detection (Brasington et al. 2003;
   Wheaton et al. 2010).
5. **Volumes**: :func:`volume_change` reports cut, fill and net volumes with
   uncorrelated or spatially correlated (Rolstad et al. 2009) uncertainties;
   :func:`stockpile_volume` measures a pile above a base surface fitted to its toe.

Conventions
-----------
* DEMs are 2-D ``(rows, cols)`` arrays (a ``(1, rows, cols)`` stack is accepted).
  NaN, ±inf, masked entries and the optional ``nodata`` value are invalid. Invalid
  pixels are NaN in every output and are never counted in volumes.
* Change is ``after - before``: positive = fill (deposition, material added), negative
  = cut (erosion, excavation).
* Grid geometry comes from ``meta``: a rasterio metadata dict (with ``crs`` and
  ``transform``), an open rasterio dataset, an ``affine.Affine`` transform (assumed to
  be in metres), a pixel size in metres, or an ``(xres, yres)`` pair in metres.
  Horizontal units of projected CRSs are converted to metres (e.g. US survey feet).
  Geographic CRSs (degrees), metadata without a CRS and GCP-only metadata are refused:
  reproject or rectify first (:func:`farq.align_pair`, :func:`farq.rectify`).
* Elevations must be in **metres**; volumes are in m³ and areas in m².
* Inputs are never modified, and no ``RuntimeWarning`` is emitted.

References
----------
Brasington, J., Langham, J., Rumsby, B. (2003). Methodological sensitivity of
morphometric estimates of coarse fluvial sediment transport. Geomorphology 53.

Nuth, C., Kääb, A. (2011). Co-registration and bias corrections of satellite elevation
data sets for quantifying glacier thickness change. The Cryosphere 5, 271-290.

Rolstad, C., Haug, T., Denby, B. (2009). Spatially integrated geodetic glacier mass
balance and its uncertainty based on geostatistical analysis. J. Glaciology 55(192).

Wheaton, J. M., Brasington, J., Darby, S. E., Sear, D. A. (2010). Accounting for
uncertainty in DEMs from repeat topographic surveys. Earth Surf. Process. Landf. 35.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, NamedTuple, Union

import numpy as np
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.transform import Affine
from scipy import ndimage

from .georef import apply_shift
from .utils import validate_array

__all__ = [
    "DEMCoregistration",
    "StockpileResult",
    "VerticalOffset",
    "VolumeResult",
    "aspect",
    "coregister_dem",
    "elevation_change",
    "hillshade",
    "level_of_detection",
    "shift_dem",
    "significant_change",
    "slope",
    "stockpile_volume",
    "vertical_offset",
    "volume_change",
]

#: Scale factor turning a median absolute deviation into a Gaussian standard deviation.
_NMAD_SCALE = 1.4826
#: Aspect bin width (degrees) for the Nuth & Kääb fit.
_ASPECT_BIN = 10.0
#: Minimum number of pixels in an aspect bin for it to enter the Nuth & Kääb fit.
_MIN_BIN_PIXELS = 10

Geometry = Union[Mapping[str, Any], Affine, float, tuple[float, float], Any]


# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------


class VerticalOffset(NamedTuple):
    """Output of :func:`vertical_offset`; unpacks as ``offset, nmad = ...``.

    Attributes
    ----------
    offset : float
        Systematic vertical bias of ``after`` relative to ``before`` over stable ground
        (metres). Subtract it from ``after`` (or from the DoD) to remove the bias.
    nmad : float
        Normalized median absolute deviation of the elevation differences over stable
        ground (a robust standard deviation), a measure of the random DoD error.
    """

    offset: float
    nmad: float


@dataclass(frozen=True)
class DEMCoregistration:
    """Output of :func:`coregister_dem`.

    The fitted model is ``dem(x, y) ≈ reference(x - dx, y - dy) + dz``: ``dem`` is the
    reference surface moved by ``dx`` metres east, ``dy`` metres north and ``dz``
    metres up. ``dem`` (the attribute) is the input DEM with that offset removed, on
    the same grid as the reference.

    Attributes
    ----------
    dx, dy, dz : float
        Estimated offset of the input DEM relative to the reference (metres; ``dx``
        east, ``dy`` north, ``dz`` up).
    dem : numpy.ndarray
        Co-registered DEM (float, NaN where shifted in from outside the input).
    iterations : int
        Number of horizontal iterations run.
    converged : bool
        True if the last horizontal update was below the tolerance.
    nmad_before, nmad_after : float
        NMAD of ``dem - reference`` over the stable pixels before and after
        co-registration (metres).
    n_pixels : int
        Number of stable pixels used in the last horizontal fit.
    """

    dx: float
    dy: float
    dz: float
    dem: np.ndarray = field(repr=False)
    iterations: int
    converged: bool
    nmad_before: float
    nmad_after: float
    n_pixels: int


@dataclass(frozen=True)
class VolumeResult:
    """Output of :func:`volume_change`. All values are plain Python numbers.

    Attributes
    ----------
    cut_m3, fill_m3 : float
        Volume removed (elevation decreased) and added (elevation increased), both
        reported as positive numbers.
    net_m3 : float
        ``fill_m3 - cut_m3``: positive for a net gain of material.
    cut_area_m2, fill_area_m2 : float
        Planimetric area of the cut and fill pixels.
    unchanged_area_m2 : float
        Valid area without detectable change (``|dh| <= lod``, or ``dh == 0``).
    valid_area_m2 : float
        Area of all valid pixels inside the region of interest.
    nodata_area_m2 : float
        Area inside the region of interest without a valid DoD (or LoD) value. Volumes
        exclude it; a large value means the volumes are incomplete.
    pixel_area_m2 : float
        Ground area of one pixel.
    cut_uncertainty_m3, fill_uncertainty_m3, uncertainty_m3 : float or None
        One-sigma uncertainty of ``cut_m3``, ``fill_m3`` and ``net_m3`` (None when no
        ``sigma`` was given). See :func:`volume_change` for the formulas.
    """

    cut_m3: float
    fill_m3: float
    net_m3: float
    cut_area_m2: float
    fill_area_m2: float
    unchanged_area_m2: float
    valid_area_m2: float
    nodata_area_m2: float
    pixel_area_m2: float
    cut_uncertainty_m3: float | None = None
    fill_uncertainty_m3: float | None = None
    uncertainty_m3: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Return the fields as a standard-JSON dict (non-finite numbers become None)."""
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class StockpileResult:
    """Output of :func:`stockpile_volume`. All values are plain Python numbers.

    Attributes
    ----------
    volume_m3 : float
        Volume of material above the base surface.
    below_base_m3 : float
        Volume of voids below the base surface inside the footprint (normally close to
        zero; a large value suggests a poor base or a misplaced footprint).
    net_m3 : float
        ``volume_m3 - below_base_m3``.
    area_m2 : float
        Planimetric footprint area with valid elevations.
    max_height_m : float
        Largest height above the base surface.
    base : str or float
        The base surface used: ``"plane"``, ``"lowest"``, ``"mean"``, ``"surface"``
        (array base) or the fixed base elevation.
    base_elevation_m : float
        Mean base elevation under the footprint.
    base_slope_deg : float
        Slope of the base plane (0 for horizontal bases and NaN for an array base).
    base_rmse_m : float
        RMS difference between the toe ring elevations and the base (NaN for fixed or
        array bases). A large value means the toe is not planar (or the footprint cuts
        through the pile), so the base is uncertain.
    missing_area_m2 : float
        Footprint area without valid elevations, excluded from the volume.
    uncertainty_m3 : float or None
        One-sigma volume uncertainty from ``sigma`` (see :func:`volume_change`).
    """

    volume_m3: float
    below_base_m3: float
    net_m3: float
    area_m2: float
    max_height_m: float
    base: str | float
    base_elevation_m: float
    base_slope_deg: float
    base_rmse_m: float
    missing_area_m2: float
    uncertainty_m3: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as a standard-JSON dict (NaN, e.g. ``base_rmse_m`` of a fixed
        base, becomes None, so ``json.dumps(..., allow_nan=False)`` works)."""
        return _json_safe(asdict(self))


# --------------------------------------------------------------------------------------
# Input helpers
# --------------------------------------------------------------------------------------


def _json_safe(values: dict[str, Any]) -> dict[str, Any]:
    """Replace NaN and ±inf by None: ``json.dumps`` would write them as the non-standard
    tokens ``NaN``/``Infinity``, which strict JSON parsers (browsers, jq, PostGIS) reject."""
    return {
        k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in values.items()
    }


def _float_dtype(dtype: np.dtype) -> np.dtype:
    result = np.result_type(dtype, np.float32)
    return np.dtype(np.float64) if result.itemsize > 8 else result


def _as_dem(array: Any, name: str, nodata: float | None = None) -> np.ndarray:
    """Return a 2-D float DEM with NaN for every invalid pixel (never a view to modify)."""
    invalid = None
    if isinstance(array, np.ma.MaskedArray):
        invalid = np.ma.getmaskarray(array)
        array = np.asarray(array.data)
    validate_array(array, name=name, allow_all_nan=True)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have an integer or float dtype, got {array.dtype}")
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
        invalid = None if invalid is None else invalid[0]
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D elevation raster (rows, cols), got shape {array.shape}"
        )
    data = array.astype(_float_dtype(array.dtype), copy=True)
    bad = ~np.isfinite(data)
    if nodata is not None and not math.isnan(nodata):
        bad |= array == nodata
    if invalid is not None:
        bad |= invalid
    data[bad] = np.nan
    return data


def _as_bool_mask(mask: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Boolean mask of ``shape``; NaN / masked entries are False."""
    if isinstance(mask, np.ma.MaskedArray):
        mask = mask.filled(False)
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(mask).__name__}")
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape != shape:
        raise ValueError(
            f"{name} must have the raster's shape {shape}, got {mask.shape}; "
            "rasterize polygons onto the DEM grid first"
        )
    if mask.dtype == bool:
        return mask
    if mask.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be boolean or numeric, got {mask.dtype}")
    if mask.dtype.kind == "f":
        return np.isfinite(mask) & (mask != 0)
    return mask != 0


def _check_same_shape(a: np.ndarray, b: np.ndarray, names: tuple[str, str]) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{names[0]} {a.shape} and {names[1]} {b.shape} must have the same shape; put "
            "both DEMs on one grid first (farq.align_pair)"
        )


def _crs_unit_factor(crs_value: Any) -> float:
    """Metres per horizontal CRS unit; refuse geographic or invalid CRSs."""
    try:
        crs = crs_value if isinstance(crs_value, CRS) else CRS.from_user_input(crs_value)
    except (CRSError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid CRS {crs_value!r}: {e}") from e
    if crs.is_geographic:
        raise ValueError(
            f"{crs} is a geographic CRS (degrees): pixel sizes, slopes and volumes are "
            "undefined. Reproject the DEMs to a projected CRS such as UTM first "
            "(e.g. farq.align_pair(..., dst_crs=...))."
        )
    try:
        _, factor = crs.linear_units_factor
    except CRSError:
        return 1.0  # local/engineering CRS without declared units: assume metres
    return float(factor) if factor and math.isfinite(factor) and factor > 0 else 1.0


def _linear(meta: Geometry, shape: tuple[int, int] | None = None) -> np.ndarray:
    """2x2 matrix mapping (col, row) pixel steps to (east, north) metres."""
    if isinstance(meta, (bool, str)) or meta is None:
        raise TypeError(
            "meta must be a rasterio metadata dict, a dataset, an affine transform, a "
            f"pixel size in metres or an (xres, yres) pair, got {meta!r}"
        )
    if not isinstance(meta, Mapping) and hasattr(meta, "transform") and hasattr(meta, "crs"):
        meta = {
            "transform": meta.transform,
            "crs": meta.crs,
            "width": getattr(meta, "width", None),
            "height": getattr(meta, "height", None),
        }
    if isinstance(meta, Mapping):
        t = meta.get("transform")
        if t is None:
            if meta.get("gcps"):
                raise ValueError(
                    "meta is georeferenced by GCPs only; rectify the DEM onto a regular "
                    "grid first (farq.rectify or farq.align_pair)"
                )
            raise ValueError("meta has no 'transform': the DEM is not georeferenced")
        t = t if isinstance(t, Affine) else Affine(*tuple(t)[:6])
        crs = meta.get("crs")
        if t.is_identity and (crs is None or meta.get("gcps")):
            raise ValueError(
                "meta has no geotransform (identity transform"
                + (" with GCPs" if meta.get("gcps") else " and no CRS")
                + "): the pixel size is unknown. Rectify the DEM (farq.rectify / "
                "align_pair) or pass the pixel size in metres instead of meta."
            )
        if crs is None:
            raise ValueError(
                "meta has no 'crs', so the horizontal units are unknown; pass a CRS or "
                "the pixel size in metres instead of meta"
            )
        factor = _crs_unit_factor(crs)
        h, w = meta.get("height"), meta.get("width")
        if shape is not None and (
            (h is not None and int(h) != shape[0]) or (w is not None and int(w) != shape[1])
        ):
            raise ValueError(f"raster shape {shape} does not match its metadata ({h}, {w})")
        jac = np.array([[t.a, t.b], [t.d, t.e]], dtype=np.float64) * factor
    elif all(hasattr(meta, k) for k in ("a", "b", "d", "e")):
        affine: Any = meta
        jac = np.array([[affine.a, affine.b], [affine.d, affine.e]], dtype=np.float64)
    elif np.isscalar(meta):
        size = float(meta)  # type: ignore[arg-type]
        if not (math.isfinite(size) and size > 0):
            raise ValueError(f"pixel size must be positive and finite, got {meta!r}")
        jac = np.array([[size, 0.0], [0.0, -size]])
    elif isinstance(meta, (tuple, list, np.ndarray)) and len(meta) == 2:
        xres, yres = (abs(float(v)) for v in meta)
        if not (math.isfinite(xres) and math.isfinite(yres) and xres > 0 and yres > 0):
            raise ValueError(f"pixel size must be positive and finite, got {meta!r}")
        jac = np.array([[xres, 0.0], [0.0, -yres]])
    else:
        raise TypeError(
            "meta must be a rasterio metadata dict, a dataset, an affine transform, a "
            f"pixel size in metres or an (xres, yres) pair, got {type(meta).__name__}"
        )
    det = float(np.linalg.det(jac))
    if not (np.isfinite(jac).all() and abs(det) > 0):
        raise ValueError("the geotransform is degenerate (zero pixel area)")
    return jac


def _pixel_area_m2(jac: np.ndarray) -> float:
    # Explicit 2x2 determinant: np.linalg.det goes through an LU factorization and
    # returns e.g. 100.00000000000004 for 10 m pixels, so areas and volumes would not
    # match the exact pixel areas of farq.change_summary and farq.polygonize.
    (a, b), (d, e) = jac.tolist()
    return abs(a * e - b * d)


def _sigma_array(sigma: Any, shape: tuple[int, ...], name: str) -> np.ndarray | float:
    """Validate a scalar or per-pixel non-negative standard deviation."""
    if np.isscalar(sigma) and not isinstance(sigma, (bool, str)):
        value = float(sigma)
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be a finite number >= 0, got {sigma!r}")
        return value
    if isinstance(sigma, np.ma.MaskedArray):
        sigma = sigma.astype(np.float64).filled(np.nan)
    if not isinstance(sigma, np.ndarray):
        raise TypeError(f"{name} must be a number or a numpy array, got {type(sigma).__name__}")
    if sigma.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric, got {sigma.dtype}")
    arr = sigma.astype(np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must be a scalar or have shape {shape}, got {arr.shape}")
    with np.errstate(invalid="ignore"):
        if np.any(arr < 0):
            raise ValueError(f"{name} must be >= 0")
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _nmad(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(_NMAD_SCALE * np.median(np.abs(values - np.median(values))))


# --------------------------------------------------------------------------------------
# Terrain derivatives
# --------------------------------------------------------------------------------------


def _pixel_diff(z: np.ndarray, axis: int) -> np.ndarray:
    """NaN-aware derivative along ``axis`` per pixel step.

    Central differences where both neighbours are valid, one-sided differences where
    only one is (raster edges and nodata holes), NaN otherwise.
    """
    fwd = np.full(z.shape, np.nan, dtype=z.dtype)
    bwd = np.full(z.shape, np.nan, dtype=z.dtype)
    if z.shape[axis] > 1:
        step = np.diff(z, axis=axis)
        if axis == 0:
            fwd[:-1] = step
            bwd[1:] = step
        else:
            fwd[:, :-1] = step
            bwd[:, 1:] = step
    out = 0.5 * (fwd + bwd)
    one_sided = np.isnan(out)
    out[one_sided] = np.where(np.isnan(fwd[one_sided]), bwd[one_sided], fwd[one_sided])
    out[np.isnan(z)] = np.nan
    return out


def _gradient(z: np.ndarray, jac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elevation gradient ``(dz/dx east, dz/dy north)`` in metres per metre."""
    g_col = _pixel_diff(z, 1)
    g_row = _pixel_diff(z, 0)
    # dz/dcol = gx*a + gy*d and dz/drow = gx*b + gy*e, i.e. [gc, gr] = J^T [gx, gy].
    inv = np.linalg.inv(jac.T)
    gx = inv[0, 0] * g_col + inv[0, 1] * g_row
    gy = inv[1, 0] * g_col + inv[1, 1] * g_row
    return gx, gy


def slope(
    dem: np.ndarray,
    meta: Geometry,
    *,
    units: Literal["degrees", "radians", "percent"] = "degrees",
    nodata: float | None = None,
) -> np.ndarray:
    """
    Terrain slope from a DEM.

    Derivatives are central differences with the pixel spacing (and any rotation) of the
    geotransform. At raster edges and next to nodata holes one-sided differences are
    used, so valid pixels keep a slope wherever at least one neighbour per axis is
    valid.

    Parameters
    ----------
    dem : numpy.ndarray
        Elevations in metres, shape (rows, cols).
    meta : dict, dataset, Affine, float or (float, float)
        Grid geometry (see the module notes). Geographic CRSs are refused.
    units : {"degrees", "radians", "percent"}
        Output units (percent = 100 · rise / run).
    nodata : float, optional
        Elevation value marking invalid pixels (NaN is always invalid).

    Returns
    -------
    numpy.ndarray
        Slope (float64), NaN where the DEM is invalid or isolated.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.add.outer(np.zeros(4), np.arange(4.0))  # rises 1 m per 1 m pixel eastwards
    >>> float(slope(z, 1.0)[1, 1])
    45.0
    """
    if units not in ("degrees", "radians", "percent"):
        raise ValueError(f"units must be 'degrees', 'radians' or 'percent', got {units!r}")
    z = _as_dem(dem, "dem", nodata).astype(np.float64, copy=False)
    gx, gy = _gradient(z, _linear(meta, z.shape))
    rise = np.hypot(gx, gy)
    if units == "percent":
        return 100.0 * rise
    out = np.arctan(rise)
    return np.degrees(out) if units == "degrees" else out


def aspect(dem: np.ndarray, meta: Geometry, *, nodata: float | None = None) -> np.ndarray:
    """
    Terrain aspect: the compass direction a slope faces (steepest descent).

    Parameters
    ----------
    dem : numpy.ndarray
        Elevations in metres, shape (rows, cols).
    meta : dict, dataset, Affine, float or (float, float)
        Grid geometry (see the module notes).
    nodata : float, optional
        Elevation value marking invalid pixels.

    Returns
    -------
    numpy.ndarray
        Aspect in degrees clockwise from grid north, in ``[0, 360)``: 0 = north-facing,
        90 = east-facing. NaN for flat pixels (undefined aspect) and invalid pixels.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.add.outer(np.zeros(3), -np.arange(3.0))  # descends towards the east
    >>> float(aspect(z, 1.0)[1, 1])
    90.0
    """
    z = _as_dem(dem, "dem", nodata).astype(np.float64, copy=False)
    gx, gy = _gradient(z, _linear(meta, z.shape))
    out = np.degrees(np.arctan2(-gx, -gy)) % 360.0
    out[(gx == 0) & (gy == 0)] = np.nan
    return out


def hillshade(
    dem: np.ndarray,
    meta: Geometry,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Shaded relief of a DEM, for display under elevation-change maps.

    Parameters
    ----------
    dem : numpy.ndarray
        Elevations in metres, shape (rows, cols).
    meta : dict, dataset, Affine, float or (float, float)
        Grid geometry (see the module notes).
    azimuth : float
        Direction of the light source in degrees clockwise from north (default 315 =
        north-west, the cartographic convention).
    altitude : float
        Sun elevation above the horizon in degrees, in [0, 90].
    z_factor : float
        Vertical exaggeration (> 0).
    nodata : float, optional
        Elevation value marking invalid pixels.

    Returns
    -------
    numpy.ndarray
        Illumination in ``[0, 1]`` (cosine of the incidence angle; 0 = in shadow), NaN
        where the slope is undefined. Multiply by 255 for an 8-bit image.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(hillshade(np.zeros((3, 3)), 1.0, altitude=30.0)[1, 1]), 3)
    0.5
    """
    if not (math.isfinite(altitude) and 0.0 <= altitude <= 90.0):
        raise ValueError(f"altitude must be in [0, 90] degrees, got {altitude!r}")
    if not math.isfinite(azimuth):
        raise ValueError(f"azimuth must be finite, got {azimuth!r}")
    if not (math.isfinite(z_factor) and z_factor > 0):
        raise ValueError(f"z_factor must be positive, got {z_factor!r}")
    z = _as_dem(dem, "dem", nodata).astype(np.float64, copy=False)
    gx, gy = _gradient(z, _linear(meta, z.shape))
    gx *= z_factor
    gy *= z_factor
    zen = math.radians(90.0 - altitude)
    az = math.radians(azimuth)
    # Surface normal (-gx, -gy, 1) / |.| dotted with the unit vector towards the sun.
    light = -gx * math.sin(zen) * math.sin(az) - gy * math.sin(zen) * math.cos(az) + math.cos(zen)
    out = light / np.sqrt(1.0 + gx * gx + gy * gy)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# Differencing and vertical bias
# --------------------------------------------------------------------------------------


def elevation_change(
    before_dem: np.ndarray, after_dem: np.ndarray, *, nodata: float | None = None
) -> np.ndarray:
    """
    DEM of difference (DoD): ``after_dem - before_dem``.

    Parameters
    ----------
    before_dem, after_dem : numpy.ndarray
        DEMs on the same grid (see :func:`farq.align_pair`), shape (rows, cols).
        Masked arrays are supported.
    nodata : float, optional
        Elevation value marking invalid pixels in either DEM.

    Returns
    -------
    numpy.ndarray
        Elevation change in the DEM units (float32 unless an input is float64 or a wide
        integer): positive = fill/deposition, negative = cut/erosion. NaN where either
        DEM is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> elevation_change(np.array([[10.0, 12.0]]), np.array([[10.5, np.nan]]))
    array([[0.5, nan]])
    """
    b = _as_dem(before_dem, "before_dem", nodata)
    a = _as_dem(after_dem, "after_dem", nodata)
    _check_same_shape(b, a, ("before_dem", "after_dem"))
    return np.subtract(a, b, dtype=np.result_type(a.dtype, b.dtype))


def _stable_dh(
    before_dem: Any, after_dem: Any, stable_mask: Any, nodata: float | None
) -> np.ndarray:
    dh = elevation_change(before_dem, after_dem, nodata=nodata).astype(np.float64)
    keep = np.isfinite(dh)
    if stable_mask is not None:
        keep &= _as_bool_mask(stable_mask, dh.shape, "stable_mask")
    return dh[keep]


def vertical_offset(
    before_dem: np.ndarray,
    after_dem: np.ndarray,
    *,
    stable_mask: np.ndarray | None = None,
    method: Literal["median", "nmad_trimmed"] = "median",
    nodata: float | None = None,
    min_pixels: int = 100,
) -> VerticalOffset:
    """
    Systematic vertical bias of ``after_dem`` relative to ``before_dem``.

    Two surveys of the same unchanged ground rarely agree exactly: GNSS and GCP height
    errors shift a whole drone DEM up or down by a few centimetres, which over a large
    area is a large false volume. Estimate the bias over *stable* ground (roads, rock,
    buildings' surroundings, areas outside the works) and subtract it.

    This estimates a vertical offset only; use :func:`coregister_dem` when the DEMs may
    also be shifted horizontally (any slope then creates false elevation change).

    Parameters
    ----------
    before_dem, after_dem : numpy.ndarray
        DEMs on the same grid, shape (rows, cols).
    stable_mask : numpy.ndarray, optional
        Boolean mask of ground assumed unchanged. Default: all valid pixels, which is
        robust with ``"median"`` as long as less than half of the area changed.
    method : {"median", "nmad_trimmed"}
        ``"median"``: the median of the differences. ``"nmad_trimmed"``: iteratively
        discard differences more than 3 NMAD from the median, then take the mean of the
        rest (more precise when the stable ground is clean, still robust to outliers).
    nodata : float, optional
        Elevation value marking invalid pixels.
    min_pixels : int
        Minimum number of valid stable pixels (default 100).

    Returns
    -------
    VerticalOffset
        ``(offset, nmad)`` named tuple. Correct with ``after_dem - offset``.

    Raises
    ------
    ValueError
        Fewer than ``min_pixels`` valid stable pixels or an unknown ``method``.

    Examples
    --------
    >>> import numpy as np
    >>> before = np.zeros((20, 20))
    >>> after = before + 0.12
    >>> after[:5, :5] += 3.0  # real change, ignored by the robust estimate
    >>> offset, nmad = vertical_offset(before, after)
    >>> round(offset, 3), round(nmad, 3)
    (0.12, 0.0)
    """
    if method not in ("median", "nmad_trimmed"):
        raise ValueError(f"method must be 'median' or 'nmad_trimmed', got {method!r}")
    dh = _stable_dh(before_dem, after_dem, stable_mask, nodata)
    if dh.size < max(int(min_pixels), 1):
        raise ValueError(
            f"only {dh.size} valid stable pixels (need at least {min_pixels}); check the "
            "DEM overlap and stable_mask"
        )
    if method == "median":
        return VerticalOffset(float(np.median(dh)), _nmad(dh))
    kept = dh
    for _ in range(20):
        med, nmad = float(np.median(kept)), _nmad(kept)
        inliers = kept[np.abs(kept - med) <= 3.0 * nmad] if nmad > 0 else kept[kept == med]
        if inliers.size == kept.size or inliers.size < 2:
            break
        kept = inliers
    return VerticalOffset(float(np.mean(kept)), _nmad(kept))


# --------------------------------------------------------------------------------------
# Nuth & Kääb co-registration
# --------------------------------------------------------------------------------------


def shift_dem(
    dem: np.ndarray,
    meta: Geometry,
    dx: float,
    dy: float,
    dz: float = 0.0,
    *,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Remove a 3-D offset from a DEM: the inverse of ``dem(x, y) = ref(x - dx, y - dy) + dz``.

    Use it to apply a :func:`coregister_dem` result estimated on a subset (or a coarser
    copy) to the full DEM on the same grid.

    Parameters
    ----------
    dem : numpy.ndarray
        DEM, shape (rows, cols).
    meta : dict, dataset, Affine, float or (float, float)
        Grid geometry (see the module notes).
    dx, dy, dz : float
        Offset of ``dem`` in metres (east, north, up), e.g. from :func:`coregister_dem`.
    nodata : float, optional
        Elevation value marking invalid pixels.

    Returns
    -------
    numpy.ndarray
        Resampled (bilinear) and vertically corrected DEM (float), NaN where no input
        data covers a pixel.

    Examples
    --------
    >>> r = coregister_dem(ref, dem, meta)  # doctest: +SKIP
    >>> full = shift_dem(full_dem, meta, r.dx, r.dy, r.dz)  # doctest: +SKIP
    """
    for name, value in (("dx", dx), ("dy", dy), ("dz", dz)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    z = _as_dem(dem, "dem", nodata)
    jac = _linear(meta, z.shape)
    return _shift(z, jac, dx, dy) - z.dtype.type(dz)


def _shift(z: np.ndarray, jac: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """``out(x, y) = z(x + dx, y + dy)`` on the same grid (bilinear, NaN outside)."""
    if dx == 0 and dy == 0:
        return z.copy()
    dcol, drow = np.linalg.solve(jac, np.array([dx, dy]))
    return apply_shift(z, (-float(drow), -float(dcol)), order=1)


def _nk_fit(
    dh: np.ndarray, tan_slope: np.ndarray, aspect_rad: np.ndarray
) -> tuple[float, float, float]:
    """One Nuth & Kääb fit of dh / tan(slope) = a cos(b - aspect) + c on aspect bins.

    ``a cos(b - aspect) = a cos(b) cos(aspect) + a sin(b) sin(aspect)``, which is linear
    in ``(a cos b, a sin b, c)``; ``a sin b`` is the east and ``a cos b`` the north
    offset of the DEM.
    """
    y = dh / tan_slope
    nbins = round(360.0 / _ASPECT_BIN)
    idx = np.minimum((aspect_rad / (2 * math.pi) * nbins).astype(np.intp), nbins - 1)
    rows = []
    for k in np.unique(idx):
        sel = idx == k
        if np.count_nonzero(sel) < _MIN_BIN_PIXELS:
            continue
        asp = aspect_rad[sel]
        mean_aspect = math.atan2(float(np.sin(asp).mean()), float(np.cos(asp).mean()))
        rows.append((math.cos(mean_aspect), math.sin(mean_aspect), 1.0, float(np.median(y[sel]))))
    if len(rows) < 3:
        raise ValueError(
            "the stable terrain has too few aspects for a Nuth & Kääb fit; it needs slopes "
            "facing several directions (use vertical_offset for flat or planar terrain)"
        )
    table = np.array(rows)
    coef, _, rank, _ = np.linalg.lstsq(table[:, :3], table[:, 3], rcond=None)
    if rank < 3:
        raise ValueError(
            "the stable terrain has too few aspects for a Nuth & Kääb fit; use "
            "vertical_offset for flat or planar terrain"
        )
    north, east, c = (float(v) for v in coef)
    return east, north, c


def coregister_dem(
    reference_dem: np.ndarray,
    dem: np.ndarray,
    meta: Geometry,
    *,
    stable_mask: np.ndarray | None = None,
    nodata: float | None = None,
    max_iterations: int = 10,
    tolerance: float = 0.01,
    min_slope: float = 2.0,
    max_slope: float = 70.0,
) -> DEMCoregistration:
    """
    Horizontal and vertical DEM co-registration after Nuth & Kääb (2011).

    A horizontal shift between two DEMs creates false elevation change on every slope:
    ``dh = tan(slope) · a · cos(b - aspect)`` for a shift of length ``a`` towards
    azimuth ``b``. The method fits this cosine to ``dh / tan(slope)`` against aspect
    over stable terrain (using per-aspect-bin medians, so outliers and real change have
    little influence), shifts the DEM by the estimate and repeats until the update is
    negligible. The vertical bias is then the median of the remaining differences.

    Slope and aspect come from the reference DEM (:func:`slope`, :func:`aspect`). Only
    a translation is modelled: no rotation, tilt or elevation-dependent bias.

    Parameters
    ----------
    reference_dem : numpy.ndarray
        Reference DEM (usually the "before" survey), shape (rows, cols).
    dem : numpy.ndarray
        DEM to co-register onto the reference, on the same grid.
    meta : dict, dataset, Affine, float or (float, float)
        Common grid geometry (see the module notes).
    stable_mask : numpy.ndarray, optional
        Boolean mask of terrain assumed unchanged (exclude the works, stockpiles,
        vegetation, water). Default: all valid pixels.
    nodata : float, optional
        Elevation value marking invalid pixels.
    max_iterations : int
        Maximum number of horizontal iterations (default 10).
    tolerance : float
        Stop when the horizontal update is below this fraction of a pixel (default
        0.01).
    min_slope, max_slope : float
        Slope range in degrees used for the fit (default 2-70). Gentle slopes carry
        little horizontal information and amplify noise; cliffs are unreliable.

    Returns
    -------
    DEMCoregistration
        Offsets ``dx`` (east), ``dy`` (north), ``dz`` (up) of ``dem`` relative to the
        reference in metres, the co-registered DEM, the iteration count and the NMAD of
        the differences before and after.

    Raises
    ------
    ValueError
        Different shapes, too few stable sloping pixels or too few aspects (flat or
        planar terrain cannot constrain a horizontal shift).

    Warns
    -----
    UserWarning
        If the iteration did not converge within ``max_iterations``.

    Examples
    --------
    >>> result = coregister_dem(dem_2023, dem_2024, meta, stable_mask=~works)  # doctest: +SKIP
    >>> result.dx, result.dy, result.dz  # metres east, north, up  # doctest: +SKIP
    >>> dod = elevation_change(dem_2023, result.dem)  # doctest: +SKIP
    """
    if isinstance(max_iterations, bool) or int(max_iterations) < 1:
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations!r}")
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    if not (0.0 <= min_slope < max_slope <= 90.0):
        raise ValueError("slopes must satisfy 0 <= min_slope < max_slope <= 90 degrees")
    ref = _as_dem(reference_dem, "reference_dem", nodata).astype(np.float64, copy=False)
    mov = _as_dem(dem, "dem", nodata).astype(np.float64, copy=False)
    _check_same_shape(ref, mov, ("reference_dem", "dem"))
    jac = _linear(meta, ref.shape)
    stable = np.ones(ref.shape, dtype=bool)
    if stable_mask is not None:
        stable = _as_bool_mask(stable_mask, ref.shape, "stable_mask")

    gx, gy = _gradient(ref, jac)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect_rad = np.arctan2(-gx, -gy) % (2 * math.pi)
    usable = stable & (slope_deg >= min_slope) & (slope_deg <= max_slope)
    tan_slope = np.tan(np.radians(np.where(usable, slope_deg, 45.0)))
    pixel = math.sqrt(_pixel_area_m2(jac))

    def differences(dx: float, dy: float) -> np.ndarray:
        return _shift(mov, jac, dx, dy) - ref

    dh = differences(0.0, 0.0)
    valid0 = stable & np.isfinite(dh)
    nmad_before = _nmad(dh[valid0])
    dx = dy = 0.0
    converged = False
    iterations = n_used = 0
    while iterations < int(max_iterations):
        iterations += 1
        sel = usable & np.isfinite(dh)
        n_used = int(np.count_nonzero(sel))
        if n_used < 3 * _MIN_BIN_PIXELS:
            raise ValueError(
                f"only {n_used} stable pixels with slopes between {min_slope} and "
                f"{max_slope} degrees; Nuth & Kääb co-registration needs sloping terrain"
            )
        d = dh[sel]
        d = d - np.median(d)  # remove the vertical bias before the horizontal fit
        east, north, _ = _nk_fit(d, tan_slope[sel], aspect_rad[sel])
        dx += east
        dy += north
        dh = differences(dx, dy)
        if math.hypot(east, north) < tolerance * pixel:
            converged = True
            break
    if not converged:
        warnings.warn(
            f"Nuth & Kääb co-registration did not converge in {max_iterations} iterations "
            f"(last update {math.hypot(east, north):.3g} m); check stable_mask",
            UserWarning,
            stacklevel=2,
        )
    final = stable & np.isfinite(dh)
    if not final.any():
        raise ValueError("no valid stable pixels remain after shifting the DEM")
    dz = float(np.median(dh[final]))
    corrected = _shift(mov, jac, dx, dy) - dz
    return DEMCoregistration(
        dx=float(dx),
        dy=float(dy),
        dz=dz,
        dem=corrected,
        iterations=iterations,
        converged=converged,
        nmad_before=nmad_before,
        nmad_after=_nmad(dh[final] - dz),
        n_pixels=n_used,
    )


# --------------------------------------------------------------------------------------
# Level of detection
# --------------------------------------------------------------------------------------


def level_of_detection(
    sigma_before: float | np.ndarray,
    sigma_after: float | np.ndarray | None = None,
    *,
    confidence: float = 0.95,
) -> float | np.ndarray:
    """
    Minimum level of detection (LoD) of a DEM of difference.

    ``LoD = t · sqrt(σ_before² + σ_after²)``, where ``t`` is the two-sided standard
    normal quantile for ``confidence`` (1.96 at 95 %, Brasington et al. 2003; Wheaton
    et al. 2010). Elevation changes smaller than the LoD cannot be distinguished from
    survey noise.

    Parameters
    ----------
    sigma_before : float or numpy.ndarray
        Vertical standard deviation (metres) of the before DEM: a number (e.g. the
        check-point RMSE of the survey) or a per-pixel array (e.g. from a fuzzy model
        of point density and slope). NaN pixels give a NaN LoD.
    sigma_after : float or numpy.ndarray, optional
        Same for the after DEM. Default: equal to ``sigma_before``. If you already have
        the standard deviation of the *difference* itself (for example the NMAD over
        stable ground from :func:`vertical_offset`), pass it as ``sigma_before`` with
        ``sigma_after=0``.
    confidence : float
        Confidence level in (0, 1) (default 0.95).

    Returns
    -------
    float or numpy.ndarray
        The LoD in metres: a float for scalar sigmas, otherwise an array.

    Examples
    --------
    >>> round(level_of_detection(0.05), 4)  # two surveys of 5 cm vertical accuracy
    0.1386
    >>> round(level_of_detection(0.04, 0.03, confidence=0.68), 4)
    0.0497
    """
    if not (isinstance(confidence, (int, float, np.floating)) and 0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    from scipy.stats import norm

    t = float(norm.ppf(0.5 + confidence / 2.0))
    s1 = _sigma_any(sigma_before, "sigma_before")
    s2 = s1 if sigma_after is None else _sigma_any(sigma_after, "sigma_after")
    if isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray) and s1.shape != s2.shape:
        raise ValueError(
            f"sigma_before {s1.shape} and sigma_after {s2.shape} must have the same shape"
        )
    lod = t * np.sqrt(np.square(s1) + np.square(s2))
    return float(lod) if np.ndim(lod) == 0 else lod


def _sigma_any(sigma: Any, name: str) -> float | np.ndarray:
    if np.isscalar(sigma) and not isinstance(sigma, (bool, str)):
        return _sigma_array(sigma, (), name)
    if isinstance(sigma, (np.ndarray, np.ma.MaskedArray)):
        return _sigma_array(sigma, sigma.shape, name)
    raise TypeError(f"{name} must be a number or a numpy array, got {type(sigma).__name__}")


def _lod_array(lod: Any, shape: tuple[int, ...]) -> np.ndarray | float:
    if np.isscalar(lod) and not isinstance(lod, (bool, str)):
        return _sigma_array(lod, shape, "lod")
    if isinstance(lod, np.ndarray) and lod.ndim == 3 and lod.shape[0] == 1:
        lod = lod[0]
    return _sigma_array(lod, shape, "lod")


def significant_change(dod: np.ndarray, lod: float | np.ndarray) -> np.ndarray:
    """
    Threshold a DEM of difference at its level of detection.

    Parameters
    ----------
    dod : numpy.ndarray
        Elevation change (from :func:`elevation_change`), shape (rows, cols).
    lod : float or numpy.ndarray
        Level of detection (from :func:`level_of_detection`), a number or a per-pixel
        array of the same shape.

    Returns
    -------
    numpy.ndarray
        Float copy of ``dod`` where changes with ``|dh| <= lod`` are set to 0 (no
        detectable change). NaN where ``dod`` or ``lod`` is NaN. Use
        ``result > 0`` / ``result < 0`` for significant fill / cut masks.

    Examples
    --------
    >>> import numpy as np
    >>> significant_change(np.array([[0.05, -0.30, 0.20, np.nan]]), 0.1)
    array([[ 0. , -0.3,  0.2,  nan]])
    """
    dh = _as_dem(dod, "dod")
    limit = _lod_array(lod, dh.shape)
    out = dh.copy()
    with np.errstate(invalid="ignore"):
        below = np.abs(dh) <= limit
    out[below] = 0.0
    if isinstance(limit, np.ndarray):
        out[np.isnan(limit)] = np.nan
    return out


# --------------------------------------------------------------------------------------
# Volumes
# --------------------------------------------------------------------------------------


def _volume_sigma(
    sigma: float | np.ndarray,
    sel: np.ndarray,
    pixel_area: float,
    correlation_length: float | None,
) -> float:
    """One-sigma uncertainty (m³) of the summed volume over the pixels in ``sel``."""
    n = int(np.count_nonzero(sel))
    if n == 0:
        return 0.0
    if isinstance(sigma, np.ndarray):
        var_sum = float(np.sum(np.square(sigma[sel])))
    else:
        var_sum = n * sigma * sigma
    uncorrelated = pixel_area * math.sqrt(var_sum) if math.isfinite(var_sum) else math.nan
    if correlation_length is None or not math.isfinite(var_sum):
        return uncorrelated
    area = n * pixel_area
    sigma_rms = math.sqrt(var_sum / n)
    # Rolstad et al. (2009): spherical variogram with range L, integrated over a disc
    # with the same area A (radius r = sqrt(A / pi)).
    radius = math.sqrt(area / math.pi)
    ratio = radius / correlation_length
    if ratio < 1.0:
        factor = 1.0 - ratio + ratio**3 / 5.0
    else:
        factor = math.pi * correlation_length**2 / (5.0 * area)
    correlated = area * sigma_rms * math.sqrt(factor)
    return max(correlated, uncorrelated)


def volume_change(
    dod: np.ndarray,
    meta: Geometry,
    *,
    lod: float | np.ndarray | None = None,
    mask: np.ndarray | None = None,
    sigma: float | np.ndarray | None = None,
    correlation_length: float | None = None,
) -> VolumeResult:
    """
    Cut, fill and net volumes of a DEM of difference.

    ``fill = A · Σ dh`` over pixels with ``dh > lod`` and ``cut = A · Σ |dh|`` over
    pixels with ``dh < -lod`` (``A`` = pixel area). Without ``lod`` every non-zero
    change counts.

    Uncertainty (``sigma`` = one-sigma vertical error of the DoD per pixel, e.g. the
    NMAD over stable ground or ``sqrt(σ_before² + σ_after²)``), for a set of ``n``
    pixels with total area ``S = n · A``:

    * uncorrelated errors: ``σ_V = A · sqrt(Σ σ_i²)`` (``= σ · A · sqrt(n)`` for a
      constant ``σ``). This underestimates real DEM errors, which are spatially
      correlated.
    * with ``correlation_length`` ``L`` (range of a spherical variogram, from a
      variogram of the DoD over stable ground), after Rolstad et al. (2009), treating
      the area as a disc of radius ``r = sqrt(S / π)``:
      ``σ_S² = σ²·(1 - r/L + r³/(5L³))`` if ``r < L``, else ``σ_S² = σ²·πL²/(5S)``;
      ``σ_V = S · σ_S``, with ``σ`` the RMS of ``σ_i``. As positively correlated
      errors cannot average out faster than independent ones, the larger of the two
      estimates is reported.

    The uncertainty is computed over the pixels that make up each volume: the cut
    pixels, the fill pixels and, for the net volume, all counted pixels (all valid
    pixels when ``lod`` is None, otherwise cut and fill pixels).

    Parameters
    ----------
    dod : numpy.ndarray
        Elevation change in metres (``after - before``), shape (rows, cols).
    meta : dict, dataset, Affine, float or (float, float)
        Grid geometry (see the module notes). A number is the pixel *side* in metres,
        as in :func:`farq.change_summary`. Geographic CRSs and metadata without a CRS
        are refused.
    lod : float or numpy.ndarray, optional
        Level of detection (:func:`level_of_detection`); smaller changes are ignored.
        Pixels with a NaN LoD are treated as nodata.
    mask : numpy.ndarray, optional
        Boolean region of interest on the DoD grid (e.g. a rasterized site boundary).
    sigma : float or numpy.ndarray, optional
        One-sigma vertical error of the DoD in metres (scalar or per pixel).
    correlation_length : float, optional
        Spatial correlation range of the DoD errors in metres (requires ``sigma``).

    Returns
    -------
    VolumeResult
        Volumes (m³), areas (m²) and uncertainties; ``to_dict()`` gives a JSON-ready
        dict.

    Raises
    ------
    ValueError
        Geographic or missing CRS, shape mismatches, negative ``sigma``/``lod``.

    Examples
    --------
    >>> import numpy as np
    >>> dod = np.zeros((10, 10))
    >>> dod[:2] = 0.5    # 20 px of fill
    >>> dod[-1] = -1.0   # 10 px of cut
    >>> v = volume_change(dod, 0.5)  # 0.5 m pixels: 0.25 m² each
    >>> v.fill_m3, v.cut_m3, v.net_m3, v.cut_area_m2
    (2.5, 2.5, 0.0, 2.5)
    """
    dh = _as_dem(dod, "dod").astype(np.float64, copy=False)
    jac = _linear(meta, dh.shape)
    pixel_area = _pixel_area_m2(jac)
    roi = np.ones(dh.shape, dtype=bool) if mask is None else _as_bool_mask(mask, dh.shape, "mask")
    valid = roi & np.isfinite(dh)
    limit: float | np.ndarray = 0.0
    if lod is not None:
        limit = _lod_array(lod, dh.shape)
        if isinstance(limit, np.ndarray):
            valid &= np.isfinite(limit)
    if correlation_length is not None:
        if sigma is None:
            raise ValueError("correlation_length requires sigma")
        if not (math.isfinite(correlation_length) and correlation_length > 0):
            raise ValueError(f"correlation_length must be positive, got {correlation_length!r}")
    sig = None if sigma is None else _sigma_array(sigma, dh.shape, "sigma")

    with np.errstate(invalid="ignore"):
        fill = valid & (dh > limit)
        cut = valid & (dh < -limit)
    fill_v = float(np.sum(dh[fill])) * pixel_area
    cut_v = abs(float(np.sum(dh[cut]))) * pixel_area  # abs: no -0.0 without cut
    n_valid = int(np.count_nonzero(valid))
    n_fill, n_cut = int(np.count_nonzero(fill)), int(np.count_nonzero(cut))

    unc: tuple[float | None, ...] = (None, None, None)
    if sig is not None:
        counted = valid if lod is None else (fill | cut)
        unc = tuple(
            _volume_sigma(sig, s, pixel_area, correlation_length) for s in (cut, fill, counted)
        )
    return VolumeResult(
        cut_m3=cut_v,
        fill_m3=fill_v,
        net_m3=fill_v - cut_v,
        cut_area_m2=n_cut * pixel_area,
        fill_area_m2=n_fill * pixel_area,
        unchanged_area_m2=(n_valid - n_cut - n_fill) * pixel_area,
        valid_area_m2=n_valid * pixel_area,
        nodata_area_m2=int(np.count_nonzero(roi & ~valid)) * pixel_area,
        pixel_area_m2=pixel_area,
        cut_uncertainty_m3=unc[0],
        fill_uncertainty_m3=unc[1],
        uncertainty_m3=unc[2],
    )


def stockpile_volume(
    dem: np.ndarray,
    mask: np.ndarray,
    meta: Geometry,
    *,
    base: Literal["plane", "lowest", "mean"] | float | np.ndarray = "plane",
    ring_width: int = 1,
    nodata: float | None = None,
    sigma: float | np.ndarray | None = None,
    correlation_length: float | None = None,
) -> StockpileResult:
    """
    Volume of a stockpile (or any mound) above a base surface.

    The base is estimated from the *toe ring*: valid pixels just outside the footprint
    ``mask`` (within ``ring_width`` pixels), which should be the ground the pile sits
    on. Draw the footprint polygon along the toe of the pile, on bare ground.

    Parameters
    ----------
    dem : numpy.ndarray
        Surface elevations in metres (DSM from a drone survey), shape (rows, cols).
    mask : numpy.ndarray
        Boolean footprint of the stockpile on the DEM grid (e.g. a rasterized polygon).
    meta : dict, dataset, Affine, float or (float, float)
        Grid geometry (see the module notes). Geographic CRSs are refused.
    base : {"plane", "lowest", "mean"}, float or numpy.ndarray
        ``"plane"`` (default): least-squares plane through the toe ring (suits piles on
        sloping ground). ``"lowest"``: horizontal plane at the lowest toe elevation
        (conservative, larger volume). ``"mean"``: horizontal plane at the mean toe
        elevation. A number: a fixed base elevation (e.g. a pad level). An array: a base
        surface on the same grid (e.g. a survey before the pile was placed).
    ring_width : int
        Width of the toe ring in pixels (default 1).
    nodata : float, optional
        Elevation value marking invalid pixels.
    sigma, correlation_length : optional
        Vertical error of the DEM (and base) in metres and its correlation range; the
        uncertainty follows :func:`volume_change`. Base fitting errors are not included
        (check ``base_rmse_m``).

    Returns
    -------
    StockpileResult
        Volume above the base (m³), voids below it, footprint area, maximum height and
        base quality figures; ``to_dict()`` gives a JSON-ready dict.

    Raises
    ------
    ValueError
        Empty footprint, too few valid toe pixels (fewer than 3, or collinear, for
        ``"plane"``), geographic CRS, invalid ``base``.

    Examples
    --------
    >>> import numpy as np
    >>> dem = np.full((20, 20), 50.0)
    >>> pile = np.zeros((20, 20), bool)
    >>> pile[5:15, 5:15] = True
    >>> dem[pile] += 2.0  # a 10 x 10 px box, 2 m high
    >>> r = stockpile_volume(dem, pile, 0.5)
    >>> round(r.volume_m3, 6), r.area_m2, round(r.max_height_m, 6)
    (50.0, 25.0, 2.0)
    """
    z = _as_dem(dem, "dem", nodata).astype(np.float64, copy=False)
    jac = _linear(meta, z.shape)
    pixel_area = _pixel_area_m2(jac)
    foot = _as_bool_mask(mask, z.shape, "mask")
    if not foot.any():
        raise ValueError("mask (the stockpile footprint) is empty")
    if isinstance(ring_width, bool) or int(ring_width) < 1:
        raise ValueError(f"ring_width must be a positive integer, got {ring_width!r}")

    valid = foot & np.isfinite(z)
    rows, cols = np.nonzero(valid)
    base_name: str | float
    base_slope = 0.0
    base_rmse = math.nan
    if isinstance(base, np.ndarray):
        surface = _as_dem(base, "base", nodata).astype(np.float64, copy=False)
        _check_same_shape(z, surface, ("dem", "base"))
        base_values = surface[rows, cols]
        valid_base = np.isfinite(base_values)
        rows, cols, base_values = rows[valid_base], cols[valid_base], base_values[valid_base]
        base_name, base_slope = "surface", math.nan
    elif isinstance(base, str):
        if base not in ("plane", "lowest", "mean"):
            raise ValueError(
                f"base must be 'plane', 'lowest', 'mean', a number or an array, got {base!r}"
            )
        structure = ndimage.generate_binary_structure(2, 2)
        ring = (
            ndimage.binary_dilation(foot, structure=structure, iterations=int(ring_width))
            & ~foot
            & np.isfinite(z)
        )
        r_rows, r_cols = np.nonzero(ring)
        ring_z = z[r_rows, r_cols]
        if ring_z.size < 3:
            raise ValueError(
                f"only {ring_z.size} valid DEM pixels around the footprint; the toe ring "
                "needs at least 3 (is the footprint at the raster edge or in nodata?)"
            )
        base_name = base
        if base == "plane":
            # Fit z = c0 + c1*col + c2*row on the ring (centred for conditioning).
            c0r, c0c = float(r_rows.mean()), float(r_cols.mean())
            design = np.column_stack([np.ones(ring_z.size), r_cols - c0c, r_rows - c0r]).astype(
                np.float64
            )
            coef, _, rank, _ = np.linalg.lstsq(design, ring_z, rcond=None)
            if rank < 3:
                raise ValueError(
                    "the toe ring pixels are collinear; cannot fit a base plane "
                    "(use base='lowest' or 'mean')"
                )
            base_values = coef[0] + coef[1] * (cols - c0c) + coef[2] * (rows - c0r)
            ring_fit = design @ coef
            base_rmse = float(np.sqrt(np.mean((ring_z - ring_fit) ** 2)))
            # Plane gradient in map units: [dz/dcol, dz/drow] = J^T [gx, gy].
            gx, gy = np.linalg.solve(jac.T, coef[1:])
            base_slope = math.degrees(math.atan(math.hypot(gx, gy)))
        else:
            level = float(ring_z.min()) if base == "lowest" else float(ring_z.mean())
            base_values = np.full(rows.size, level)
            base_rmse = float(np.sqrt(np.mean((ring_z - level) ** 2)))
    elif np.isscalar(base) and not isinstance(base, bool):
        level = float(base)
        if not math.isfinite(level):
            raise ValueError(f"base elevation must be finite, got {base!r}")
        base_values = np.full(rows.size, level)
        base_name = level
    else:
        raise TypeError(
            f"base must be 'plane', 'lowest', 'mean', a number or an array, got {type(base)}"
        )

    height = z[rows, cols] - base_values
    above = height > 0
    volume = float(np.sum(height[above])) * pixel_area
    below = abs(float(np.sum(height[~above]))) * pixel_area
    n = int(height.size)
    uncertainty = None
    if sigma is not None:
        if correlation_length is not None and not (
            math.isfinite(correlation_length) and correlation_length > 0
        ):
            raise ValueError(f"correlation_length must be positive, got {correlation_length!r}")
        sig = _sigma_array(sigma, z.shape, "sigma")
        sel = np.zeros(z.shape, dtype=bool)
        sel[rows, cols] = True
        uncertainty = _volume_sigma(sig, sel, pixel_area, correlation_length)
    return StockpileResult(
        volume_m3=volume,
        below_base_m3=below,
        net_m3=volume - below,
        area_m2=n * pixel_area,
        max_height_m=float(height.max()) if n else math.nan,
        base=base_name,
        base_elevation_m=float(base_values.mean()) if n else math.nan,
        base_slope_deg=base_slope,
        base_rmse_m=base_rmse,
        missing_area_m2=(int(np.count_nonzero(foot)) - n) * pixel_area,
        uncertainty_m3=uncertainty,
    )
