"""
Georeferencing, alignment and co-registration of rasters, including drone imagery.

Pixel-wise change detection needs both rasters on one common grid: the same CRS, the
same pixel size and the same pixel origin. Satellite products usually ship with an affine
geotransform, but drone images are often georeferenced only by ground control points
(GCPs), are very high resolution (centimetre pixels), are RGB(A) ``uint8`` and two
flights over the same area never share the same extent, resolution or alignment. This
module provides the tools to get such inputs onto a common grid:

- **GCP handling**: :func:`has_gcps`, :func:`read_gcps`, :func:`make_gcps` and
  :func:`gcp_residuals` (least-squares fit quality, RMSE and leave-one-out errors to
  spot bad GCPs).
- **Rectification**: :func:`georeference` (array + GCPs) and :func:`rectify` (file)
  warp a GCP-referenced image onto a regular north-up grid with a polynomial
  (order 1-3) or thin plate spline transform.
- **Alignment**: :func:`align` reprojects a raster onto a reference grid and
  :func:`align_pair` puts two rasters on a common grid cropped to their overlap.
  Both accept GCP-referenced inputs directly (a single resampling step).
- **Co-registration**: :func:`coregister` estimates a residual sub-pixel *translation*
  between two aligned images by phase correlation and :func:`apply_shift` applies it.
- **Pixel geometry**: :func:`pixel_size` and :func:`pixel_area`.

Metadata dictionaries follow the rasterio profile convention (``crs``, ``transform``,
``width``, ``height``, ``count``, ``dtype``, ``nodata``) with the optional extra keys
``gcps`` (list of :class:`rasterio.control.GroundControlPoint`) and ``gcps_crs``, as
produced by :func:`farq.read`.

Nodata conventions
------------------
Invalid pixels are never turned into valid data. Floating point arrays use NaN as the
nodata marker (a ``nodata`` value in their metadata is *not* applied, because it often
belongs to the original integer data; pass ``nodata=`` explicitly to mask it). Integer
arrays use ``nodata=`` or, failing that, ``metadata["nodata"]``. Output pixels that
fall outside the source footprint are NaN for floating point outputs, or ``nodata`` for
integer outputs. Integer inputs without any nodata value are promoted to ``float32`` so
that out-of-footprint pixels can be NaN; pass ``nodata`` (for example ``0`` for RGB
imagery with a black border) to keep the compact integer dtype.

GCP pixel coordinates follow the GDAL convention: ``(row, col) = (0, 0)`` is the
*top-left corner* of the top-left pixel, so the centre of that pixel is ``(0.5, 0.5)``.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import CRSError, NotGeoreferencedWarning, RasterioError
from rasterio.io import DatasetReaderBase
from rasterio.transform import Affine, array_bounds
from rasterio.warp import calculate_default_transform, reproject
from scipy import fft as sp_fft
from scipy import ndimage

from .core import write as _core_write
from .utils import _check_output_size, validate_array

__all__ = [
    "GCPResiduals",
    "align",
    "align_pair",
    "apply_shift",
    "coregister",
    "gcp_residuals",
    "georeference",
    "has_gcps",
    "make_gcps",
    "pixel_area",
    "pixel_size",
    "read_gcps",
    "rectify",
]

PathLike = Union[str, "os.PathLike[str]"]
Resolution = Union[float, tuple[float, float]]
Method = Literal["polynomial", "tps"]

#: Minimum number of GCPs for each polynomial order: (order + 1)(order + 2) / 2.
_MIN_GCPS = {1: 3, 2: 6, 3: 10}
#: Singular value ratio below which GCP geometry is considered degenerate (collinear).
_DEGENERATE_TOL = 1e-6


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _resampling(value: str | Resampling) -> Resampling:
    if isinstance(value, Resampling):
        return value
    if isinstance(value, str):
        try:
            return Resampling[value.lower()]
        except KeyError:
            valid = ", ".join(m.name for m in Resampling)
            raise ValueError(f"Unknown resampling {value!r}; use one of: {valid}") from None
    raise TypeError("resampling must be a rasterio.enums.Resampling member or its name")


def _as_crs(crs: Any, name: str = "crs") -> CRS:
    if crs is None:
        raise ValueError(f"{name} is missing: a coordinate reference system is required")
    if isinstance(crs, CRS):
        return crs
    try:
        return CRS.from_user_input(crs)
    except (CRSError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid {name}: {crs!r} ({e})") from e


def _as_resolution(resolution: Resolution | None) -> tuple[float, float] | None:
    if resolution is None:
        return None
    if isinstance(resolution, (int, float, np.integer, np.floating)):
        res = (float(resolution), float(resolution))
    else:
        values = tuple(float(r) for r in resolution)
        if len(values) != 2:
            raise ValueError("resolution must be a number or an (xres, yres) pair")
        res = (values[0], values[1])
    if not all(math.isfinite(r) and r > 0 for r in res):
        raise ValueError(f"resolution must be positive and finite, got {resolution!r}")
    return res


def _as_transform(value: Any) -> Affine | None:
    if value is None:
        return None
    if isinstance(value, Affine):
        return value
    try:
        return Affine(*tuple(value)[:6])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid transform: {value!r}") from e


def _as_bands(array: np.ndarray, name: str) -> np.ndarray:
    """Validate a 2D or (bands, H, W) array and return it as a 3D view."""
    validate_array(array, name=name, allow_all_nan=True)
    if array.ndim == 2:
        return array[np.newaxis]
    if array.ndim == 3:
        return array
    raise ValueError(f"{name} must be 2D (H, W) or 3D (bands, H, W), got shape {array.shape}")


def _source_dtype(dtype: np.dtype) -> np.dtype:
    """Dtype GDAL can read the source as (bool and float16 are not supported)."""
    if dtype.kind == "b":
        return np.dtype(np.uint8)
    if dtype.kind == "f" and dtype.itemsize < 4:
        return np.dtype(np.float32)
    if dtype.kind not in "uif":
        raise TypeError(f"Unsupported array dtype {dtype}; expected a real numeric dtype")
    return dtype


def _check_nodata(nodata: float | None, dtype: np.dtype) -> None:
    if nodata is None or dtype.kind == "f":
        return
    info = np.iinfo(dtype)
    if not (math.isfinite(nodata) and float(nodata).is_integer()):
        raise ValueError(f"nodata {nodata!r} is not representable in integer dtype {dtype}")
    if not info.min <= nodata <= info.max:
        raise ValueError(f"nodata {nodata!r} is out of range for dtype {dtype}")


def _nodata_policy(
    dtype: np.dtype, nodata: float | None, meta_nodata: float | None
) -> tuple[float | None, np.dtype, float]:
    """
    Return ``(src_nodata, out_dtype, dst_nodata)`` following the module conventions.
    """
    dtype = _source_dtype(np.dtype(dtype))
    if dtype.kind == "f":
        src = np.nan if nodata is None else float(nodata)
        return src, dtype, np.nan
    src_nodata = nodata if nodata is not None else meta_nodata
    if src_nodata is None or (isinstance(src_nodata, float) and math.isnan(src_nodata)):
        return None, np.dtype(np.float32), np.nan
    _check_nodata(src_nodata, dtype)
    return src_nodata, dtype, src_nodata


def _is_north_up(t: Affine) -> bool:
    return t.b == 0 and t.d == 0 and t.a > 0 and t.e < 0


def _has_valid_transform(meta: Mapping[str, Any]) -> bool:
    t = _as_transform(meta.get("transform"))
    if t is None:
        return False
    # rasterio reports the identity matrix for rasters that are not georeferenced by a
    # geotransform (GCP-only or unreferenced files).
    return not (t.is_identity and (meta.get("crs") is None or meta.get("gcps")))


def _out_meta(
    dtype: np.dtype, nodata: float, count: int, height: int, width: int, crs: CRS, t: Affine
) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "dtype": np.dtype(dtype).name,
        "nodata": nodata,
        "width": int(width),
        "height": int(height),
        "count": int(count),
        "crs": crs,
        "transform": t,
    }


# --------------------------------------------------------------------------------------
# GCP handling
# --------------------------------------------------------------------------------------


def has_gcps(source: PathLike | DatasetReaderBase | Mapping[str, Any]) -> bool:
    """
    Return True if a raster is georeferenced by ground control points.

    Parameters
    ----------
    source : str, os.PathLike, rasterio dataset or dict
        A raster path, an open rasterio dataset, or a metadata dictionary (as returned
        by :func:`farq.read`, with an optional ``"gcps"`` key).

    Returns
    -------
    bool
        True if at least one GCP is attached.

    Examples
    --------
    >>> has_gcps("flight_2024.tif")  # doctest: +SKIP
    True
    """
    if isinstance(source, Mapping):
        return bool(source.get("gcps"))
    if isinstance(source, DatasetReaderBase):
        return bool(source.gcps[0])
    if isinstance(source, (str, os.PathLike)):
        with rasterio.open(os.fspath(source)) as src:
            return bool(src.gcps[0])
    raise TypeError(
        f"source must be a path, a rasterio dataset or a metadata dict, got {type(source).__name__}"
    )


def read_gcps(path: PathLike) -> tuple[list[GroundControlPoint], CRS | None]:
    """
    Read the ground control points stored in a raster file.

    Parameters
    ----------
    path : str or os.PathLike
        Raster file path.

    Returns
    -------
    gcps : list of rasterio.control.GroundControlPoint
        The GCPs (pixel ``row``/``col`` and map ``x``/``y``/``z``).
    crs : rasterio.crs.CRS or None
        CRS of the GCP map coordinates (None if the file does not declare one).

    Raises
    ------
    ValueError
        If the file has no GCPs.

    Examples
    --------
    >>> gcps, crs = read_gcps("drone.tif")  # doctest: +SKIP
    """
    path = os.fspath(path)
    with rasterio.open(path) as src:
        gcps, crs = src.gcps
    if not gcps:
        raise ValueError(f"{path} has no ground control points")
    return list(gcps), crs


def make_gcps(
    pixels: Any,
    coords: Any,
    *,
    ids: Sequence[str] | None = None,
) -> list[GroundControlPoint]:
    """
    Build ground control points from measured pixel/map coordinate pairs.

    Parameters
    ----------
    pixels : array_like, shape (N, 2)
        Pixel positions as ``(row, col)`` pairs (GDAL convention: ``(0, 0)`` is the
        top-left corner of the image; fractional values are allowed).
    coords : array_like, shape (N, 2) or (N, 3)
        Map coordinates ``(x, y)`` or ``(x, y, z)`` in the GCP CRS (``x`` = easting or
        longitude, ``y`` = northing or latitude).
    ids : sequence of str, optional
        GCP identifiers (default ``"1"``, ``"2"``, ...).

    Returns
    -------
    list of rasterio.control.GroundControlPoint

    Raises
    ------
    ValueError
        On mismatched lengths, wrong shapes or non-finite coordinates.

    Examples
    --------
    >>> gcps = make_gcps([(0, 0), (0, 999), (749, 0)],
    ...                  [(500000, 4000000), (500050, 4000000), (500000, 3999962.5)])
    >>> len(gcps)
    3
    """
    pix = np.asarray(pixels, dtype=np.float64)
    xyz = np.asarray(coords, dtype=np.float64)
    if pix.ndim != 2 or pix.shape[1] != 2:
        raise ValueError(f"pixels must have shape (N, 2) as (row, col), got {pix.shape}")
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3):
        raise ValueError(f"coords must have shape (N, 2) or (N, 3), got {xyz.shape}")
    if len(pix) != len(xyz):
        raise ValueError(f"pixels ({len(pix)}) and coords ({len(xyz)}) differ in length")
    if len(pix) == 0:
        raise ValueError("at least one GCP is required")
    if not (np.isfinite(pix).all() and np.isfinite(xyz).all()):
        raise ValueError("GCP pixel and map coordinates must be finite")
    if ids is None:
        ids = [str(i + 1) for i in range(len(pix))]
    elif len(ids) != len(pix):
        raise ValueError(f"ids ({len(ids)}) and pixels ({len(pix)}) differ in length")
    z = xyz[:, 2] if xyz.shape[1] == 3 else np.zeros(len(xyz))
    return [
        GroundControlPoint(
            row=float(r), col=float(c), x=float(x), y=float(y), z=float(zz), id=str(i)
        )
        for (r, c), (x, y), zz, i in zip(pix, xyz[:, :2], z, ids)
    ]


def _normalize_gcps(gcps: Any) -> list[GroundControlPoint]:
    """Validate a GCP sequence; replace missing ``z`` (which GDAL cannot parse) by 0."""
    if gcps is None or len(gcps) == 0:
        raise ValueError("no ground control points given")
    out = []
    for g in gcps:
        if not isinstance(g, GroundControlPoint):
            raise TypeError(
                "gcps must be rasterio GroundControlPoint objects (see make_gcps), "
                f"got {type(g).__name__}"
            )
        values = (g.row, g.col, g.x, g.y)
        if any(v is None or not math.isfinite(v) for v in values):
            raise ValueError(f"GCP {g.id!r} has missing or non-finite coordinates")
        if g.z is None:
            g = GroundControlPoint(row=g.row, col=g.col, x=g.x, y=g.y, z=0.0, id=g.id, info=g.info)
        out.append(g)
    return out


def _gcp_arrays(gcps: Sequence[GroundControlPoint]) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel ``(col, row)`` and map ``(x, y)`` coordinates as (N, 2) arrays."""
    pix = np.array([(g.col, g.row) for g in gcps], dtype=np.float64)
    world = np.array([(g.x, g.y) for g in gcps], dtype=np.float64)
    return pix, world


def _normalize_coords(p: np.ndarray) -> np.ndarray:
    """Centre and scale coordinates for a well-conditioned polynomial fit."""
    centred = p - p.mean(axis=0)
    scale = np.abs(centred).max()
    return centred / scale if scale > 0 else centred


def _poly_design(p: np.ndarray, order: int) -> np.ndarray:
    """Design matrix with all monomials ``u**i * v**j`` for ``i + j <= order``."""
    u, v = p[:, 0], p[:, 1]
    cols = [u**i * v ** (k - i) for k in range(order + 1) for i in range(k, -1, -1)]
    return np.stack(cols, axis=1)


def _is_degenerate(design: np.ndarray) -> bool:
    s = np.linalg.svd(design, compute_uv=False)
    return bool(s[-1] <= _DEGENERATE_TOL * s[0])


def _validate_gcp_geometry(gcps: Sequence[GroundControlPoint], order: int) -> None:
    if order not in _MIN_GCPS:
        raise ValueError(f"polynomial order must be 1, 2 or 3, got {order!r}")
    need = _MIN_GCPS[order]
    if len(gcps) < need:
        raise ValueError(
            f"a polynomial of order {order} needs at least {need} GCPs, got {len(gcps)}"
        )
    pix, world = _gcp_arrays(gcps)
    for coords, what in ((pix, "pixel"), (world, "map")):
        if _is_degenerate(_poly_design(_normalize_coords(coords), order)):
            hint = "collinear or duplicated" if order == 1 else "degenerate (e.g. collinear)"
            raise ValueError(
                f"GCP {what} coordinates are {hint}; cannot fit an order-{order} polynomial. "
                "Use well-spread, non-collinear GCPs (or a lower order)."
            )


def _warp_options(
    gcps: Sequence[GroundControlPoint], method: Method, order: int | None
) -> dict[str, Any]:
    """Validate GCP geometry and return the GDAL transformer options."""
    if method == "polynomial":
        order = 1 if order is None else order
        _validate_gcp_geometry(gcps, order)
        return {"SRC_METHOD": "GCP_POLYNOMIAL", "MAX_GCP_ORDER": order}
    if method == "tps":
        if order is not None:
            raise ValueError("order applies to method='polynomial' only, not 'tps'")
        _validate_gcp_geometry(gcps, 1)
        return {"SRC_METHOD": "GCP_TPS"}
    raise ValueError(f"method must be 'polynomial' or 'tps', got {method!r}")


@dataclass(frozen=True)
class GCPResiduals:
    """
    Quality report of a least-squares polynomial fit through a set of GCPs.

    All distances are in GCP map units (e.g. metres) unless noted.

    Attributes
    ----------
    order : int
        Polynomial order of the fit.
    ids : list of str
        GCP identifiers, in input order.
    residuals : numpy.ndarray, shape (N, 2)
        Observed minus fitted map coordinates ``(dx, dy)`` per GCP.
    errors : numpy.ndarray, shape (N,)
        Euclidean residual length per GCP.
    loo_errors : numpy.ndarray, shape (N,)
        Leave-one-out error per GCP: the error at that GCP of a fit computed *without*
        it. A bad GCP drags a least-squares fit towards itself, which hides it in
        ``errors``; the leave-one-out error does not. NaN when it cannot be computed
        (no redundancy).
    studentized : numpy.ndarray, shape (N,)
        Externally studentized residual per GCP: its residual divided by the noise level
        estimated *without* it, corrected for the GCP's leverage. Comparable across
        GCPs regardless of position (corner GCPs naturally have larger leave-one-out
        errors). NaN when ``dof < 2``.
    rmse : float
        Root mean square of ``errors``.
    max_error : float
        Largest entry of ``errors``.
    dof : int
        Redundancy: number of GCPs minus number of polynomial coefficients. With
        ``dof == 0`` the fit is exact by construction and residuals carry no information.
    pixel_size : float
        Mean ground size of one pixel implied by the linear part of the fit.
    rmse_pixels : float
        ``rmse`` expressed in pixels.
    """

    order: int
    ids: list[str]
    residuals: np.ndarray = field(repr=False)
    errors: np.ndarray
    loo_errors: np.ndarray
    studentized: np.ndarray
    rmse: float
    max_error: float
    dof: int
    pixel_size: float
    rmse_pixels: float

    def outliers(self, threshold: float | None = None, *, alpha: float = 0.05) -> list[int]:
        """
        Indices of suspicious GCPs, worst first.

        By default each GCP is tested with its externally studentized residual against
        an F distribution with a Bonferroni correction for the number of GCPs, so the
        chance of flagging any correct GCP is about ``alpha`` (assuming Gaussian GCP
        noise). This needs ``dof >= 2``; with less redundancy nothing can be tested and
        an empty list is returned.

        Several bad GCPs can mask each other: remove the worst one, refit with
        :func:`gcp_residuals` and check again.

        Parameters
        ----------
        threshold : float, optional
            Instead of the statistical test, flag GCPs whose leave-one-out error (plain
            residual where unavailable) exceeds this distance in map units.
        alpha : float
            Family-wise false-alarm rate of the default test, in (0, 1).
        """
        if threshold is not None:
            errs = np.where(np.isfinite(self.loo_errors), self.loo_errors, self.errors)
            idx = np.flatnonzero(errs > threshold)
            return [int(i) for i in idx[np.argsort(-errs[idx])]]
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if self.dof < 2:
            return []
        from scipy.stats import f as f_dist

        n = len(self.errors)
        # t_i**2 / 2 ~ F(2, 2 * (dof - 1)): two coordinates per GCP.
        critical = float(f_dist.ppf(1.0 - alpha / n, 2, 2 * (self.dof - 1)))
        score = np.where(np.isfinite(self.studentized), self.studentized**2 / 2.0, -np.inf)
        idx = np.flatnonzero(score > critical)
        return [int(i) for i in idx[np.argsort(-score[idx])]]


def gcp_residuals(gcps: Sequence[GroundControlPoint], order: int = 1) -> GCPResiduals:
    """
    Fit a pixel-to-map polynomial through GCPs and report per-GCP residuals.

    Use this to judge georeferencing quality before rectifying and to find mistyped or
    misplaced GCPs: a GCP with a large leave-one-out error disagrees with the others.

    Parameters
    ----------
    gcps : sequence of rasterio.control.GroundControlPoint
        The ground control points.
    order : {1, 2, 3}
        Polynomial order (needs at least 3, 6 or 10 GCPs respectively).

    Returns
    -------
    GCPResiduals
        Residuals, RMSE, leave-one-out errors and :meth:`GCPResiduals.outliers`.

    Raises
    ------
    ValueError
        Too few GCPs for ``order``, or collinear/degenerate GCPs.

    Warns
    -----
    UserWarning
        If there are exactly as many GCPs as coefficients (residuals are trivially 0).

    Examples
    --------
    >>> report = gcp_residuals(gcps, order=1)  # doctest: +SKIP
    >>> report.rmse, report.outliers()  # doctest: +SKIP
    (0.031, [4])
    """
    gcps = _normalize_gcps(gcps)
    _validate_gcp_geometry(gcps, order)
    pix, world = _gcp_arrays(gcps)
    design = _poly_design(_normalize_coords(pix), order)
    n, k = design.shape
    q, r = np.linalg.qr(design)
    coef = np.linalg.solve(r, q.T @ world)
    residuals = world - design @ coef
    errors = np.hypot(residuals[:, 0], residuals[:, 1])

    # Leave-one-out residuals from the hat matrix diagonal: e_i / (1 - h_ii).
    leverage = np.einsum("ij,ij->i", q, q)
    denom = 1.0 - leverage
    with np.errstate(divide="ignore", invalid="ignore"):
        loo = np.where(denom > 1e-9, errors / np.where(denom > 1e-9, denom, 1.0), np.nan)

    dof = n - k
    # Externally studentized residuals: noise variance per coordinate re-estimated
    # without GCP i (SSE_(i) = SSE - e_i**2 / (1 - h_ii)), then scaled by leverage.
    studentized = np.full(n, np.nan)
    if dof >= 2:
        sse = float(np.sum(errors**2))
        ok = denom > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            sse_i = sse - errors**2 / np.where(ok, denom, 1.0)
            sigma2 = np.maximum(sse_i, 0.0) / (2.0 * (dof - 1))
        # Floor the noise so exactly-placed GCPs don't yield 0/0.
        floor = (1e-6 * max(float(np.ptp(world, axis=0).max()), 1e-12)) ** 2
        sigma2 = np.maximum(sigma2, floor)
        studentized = np.where(ok, errors / np.sqrt(sigma2 * np.where(ok, denom, 1.0)), np.nan)
    if dof == 0:
        warnings.warn(
            f"{n} GCPs exactly determine an order-{order} polynomial; residuals are zero "
            "by construction. Add more GCPs to assess accuracy.",
            UserWarning,
            stacklevel=2,
        )
    # Ground pixel size from the linear (affine) fit in raw pixel units.
    affine = np.linalg.lstsq(_poly_design(pix - pix.mean(axis=0), 1), world, rcond=None)[0]
    psize = math.sqrt(abs(np.linalg.det(affine[1:3])))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return GCPResiduals(
        order=order,
        ids=[str(g.id) for g in gcps],
        residuals=residuals,
        errors=errors,
        loo_errors=loo,
        studentized=studentized,
        rmse=rmse,
        max_error=float(errors.max()),
        dof=dof,
        pixel_size=psize,
        rmse_pixels=rmse / psize if psize > 0 else float("nan"),
    )


def _affine_from_gcps(gcps: Sequence[GroundControlPoint]) -> Affine:
    """Least-squares affine pixel(col, row) -> map(x, y) transform from GCPs."""
    pix, world = _gcp_arrays(gcps)
    design = np.column_stack([pix, np.ones(len(pix))])
    (a, d), (b, e), (c, f) = np.linalg.lstsq(design, world, rcond=None)[0]
    return Affine(a, b, c, d, e, f)


# --------------------------------------------------------------------------------------
# Sources, grids and warping
# --------------------------------------------------------------------------------------


@dataclass
class _Source:
    """Georeferencing of a source raster: an affine transform or GCPs."""

    crs: CRS
    width: int
    height: int
    transform: Affine | None = None
    gcps: list[GroundControlPoint] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def geo_kwargs(self) -> dict[str, Any]:
        if self.gcps is not None:
            return {"gcps": self.gcps, **self.options}
        return {"src_transform": self.transform}


@dataclass(frozen=True)
class _Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, bottom, right, top = array_bounds(self.height, self.width, self.transform)
        return left, bottom, right, top

    @property
    def res(self) -> tuple[float, float]:
        return self.transform.a, -self.transform.e


def _source_from_meta(
    meta: Mapping[str, Any],
    height: int,
    width: int,
    method: Method,
    order: int | None,
    name: str,
) -> _Source:
    if not isinstance(meta, Mapping):
        raise TypeError(f"{name} metadata must be a dict, got {type(meta).__name__}")
    mh, mw = meta.get("height"), meta.get("width")
    if (mh is not None and mh != height) or (mw is not None and mw != width):
        raise ValueError(
            f"{name} array shape ({height}, {width}) does not match its metadata ({mh}, {mw})"
        )
    if _has_valid_transform(meta):
        crs = _as_crs(meta.get("crs"), f"{name} CRS")
        return _Source(crs, width, height, transform=_as_transform(meta["transform"]))
    if meta.get("gcps"):
        gcps = _normalize_gcps(meta["gcps"])
        crs = _as_crs(meta.get("gcps_crs") or meta.get("crs"), f"{name} GCP CRS")
        return _Source(crs, width, height, gcps=gcps, options=_warp_options(gcps, method, order))
    raise ValueError(
        f"{name} is not georeferenced: its metadata has neither a valid 'transform' "
        "(with 'crs') nor 'gcps'"
    )


def _native_grid(src: _Source, dst_crs: CRS, resolution: tuple[float, float] | None) -> _Grid:
    """North-up grid covering ``src`` in ``dst_crs`` (optionally at ``resolution``)."""
    t = src.transform
    if t is not None and _is_north_up(t) and src.crs == dst_crs:
        if resolution is None:
            return _Grid(dst_crs, t, src.width, src.height)
        xres, yres = resolution
        w = max(1, math.ceil(src.width * t.a / xres - 1e-9))
        h = max(1, math.ceil(src.height * -t.e / yres - 1e-9))
        return _Grid(dst_crs, Affine(xres, 0, t.c, 0, -yres, t.f), w, h)
    kwargs: dict[str, Any] = {}
    if src.gcps is not None:
        kwargs = {"gcps": src.gcps, **src.options}
    else:
        left, bottom, right, top = array_bounds(src.height, src.width, t)
        kwargs = {"left": left, "bottom": bottom, "right": right, "top": top}
    try:
        dst_t, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, resolution=resolution, **kwargs
        )
    except RasterioError as e:
        raise ValueError(f"Cannot compute an output grid: {e}") from e
    return _Grid(dst_crs, dst_t, int(w), int(h))


def _integer_offset(src: _Source, grid: _Grid) -> tuple[int, int] | None:
    """Row/col offset of ``grid`` inside the source pixel grid, if it is a pure subgrid."""
    t = src.transform
    if t is None or src.crs != grid.crs or not _is_north_up(t):
        return None
    g = grid.transform
    if not (math.isclose(t.a, g.a, rel_tol=1e-9) and math.isclose(t.e, g.e, rel_tol=1e-9)):
        return None
    if g.b != 0 or g.d != 0:
        return None
    col = (g.c - t.c) / t.a
    row = (g.f - t.f) / t.e
    rc, rr = round(col), round(row)
    if abs(col - rc) > 1e-6 or abs(row - rr) > 1e-6:
        return None
    return rr, rc


def _warp(
    bands: np.ndarray,
    src: _Source,
    grid: _Grid,
    resampling: Resampling,
    nodata: float | None,
    meta_nodata: float | None,
) -> tuple[np.ndarray, float]:
    """Warp a (bands, H, W) array onto ``grid``; return the (bands, h, w) result."""
    in_dtype = _source_dtype(bands.dtype)
    src_nodata, out_dtype, dst_nodata = _nodata_policy(in_dtype, nodata, meta_nodata)
    bands = bands.astype(in_dtype, copy=False)
    count = bands.shape[0]
    _check_output_size((count, grid.height, grid.width), "output grid")
    out = np.full((count, grid.height, grid.width), dst_nodata, dtype=out_dtype)

    offset = _integer_offset(src, grid)
    if offset is not None:
        # Pure crop/pad on an identical pixel grid: copy values, no resampling.
        r0, c0 = offset
        sr0, sc0 = max(r0, 0), max(c0, 0)
        sr1, sc1 = min(r0 + grid.height, src.height), min(c0 + grid.width, src.width)
        if sr1 > sr0 and sc1 > sc0:
            block = out[:, sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0]
            block[...] = bands[:, sr0:sr1, sc0:sc1]
            if src_nodata is not None and not math.isnan(src_nodata):
                block[bands[:, sr0:sr1, sc0:sc1] == src_nodata] = dst_nodata
        return out, dst_nodata

    reproject(
        bands,
        out,
        src_crs=src.crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=dst_nodata,
        resampling=resampling,
        **src.geo_kwargs,
    )
    return out, dst_nodata


def _squeeze(out: np.ndarray, was_2d: bool) -> np.ndarray:
    return out[0] if was_2d else out


# --------------------------------------------------------------------------------------
# Rectification
# --------------------------------------------------------------------------------------


def georeference(
    array: np.ndarray,
    gcps: Sequence[GroundControlPoint],
    crs: CRS | str | int | None,
    *,
    dst_crs: CRS | str | int | None = None,
    resolution: Resolution | None = None,
    method: Method = "polynomial",
    order: int | None = None,
    resampling: str | Resampling = "bilinear",
    nodata: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Rectify a GCP-referenced image onto a regular north-up grid.

    Parameters
    ----------
    array : numpy.ndarray
        Image of shape (H, W) or (bands, H, W), e.g. an RGB(A) ``uint8`` drone image.
    gcps : sequence of rasterio.control.GroundControlPoint
        Ground control points (see :func:`make_gcps` / :func:`read_gcps`).
    crs : CRS, str or int
        CRS of the GCP map coordinates (e.g. ``32633`` or ``"EPSG:32633"``).
    dst_crs : CRS, str or int, optional
        Output CRS (default ``crs``).
    resolution : float or (float, float), optional
        Output pixel size in ``dst_crs`` units. Default: estimated from the GCPs (about
        the native resolution). Coarser values downsample large orthomosaics and save
        memory; use ``resampling="average"`` when downsampling strongly.
    method : {"polynomial", "tps"}
        ``"polynomial"`` fits a global polynomial of ``order``; ``"tps"`` (thin plate
        spline) passes exactly through every GCP and suits locally distorted images
        with many accurate GCPs, but also reproduces any GCP error exactly.
    order : {1, 2, 3}, optional
        Polynomial order (default 1 = affine; needs ≥3, ≥6 or ≥10 GCPs). Higher
        orders can extrapolate wildly outside the area covered by GCPs.
    resampling : str or rasterio.enums.Resampling
        Resampling algorithm (default ``"bilinear"``; ``"nearest"`` for class maps).
    nodata : float, optional
        Input value marking invalid pixels (see the module notes on nodata).

    Returns
    -------
    rectified : numpy.ndarray
        Rectified array with the same number of dimensions as ``array``.
    metadata : dict
        Profile of the result: ``crs``, ``transform``, ``width``, ``height``,
        ``count``, ``dtype``, ``nodata`` and ``driver``.

    Raises
    ------
    ValueError
        Missing CRS, too few or collinear GCPs, bad parameters.

    Examples
    --------
    >>> gcps = make_gcps(pixels, coords)  # doctest: +SKIP
    >>> print(gcp_residuals(gcps).rmse)  # doctest: +SKIP
    >>> rgb_geo, meta = georeference(rgb, gcps, "EPSG:32633", resolution=0.05)  # doctest: +SKIP
    """
    bands = _as_bands(array, "array")
    gcps = _normalize_gcps(gcps)
    src_crs = _as_crs(crs, "GCP crs")
    out_crs = src_crs if dst_crs is None else _as_crs(dst_crs, "dst_crs")
    src = _Source(
        src_crs,
        bands.shape[2],
        bands.shape[1],
        gcps=gcps,
        options=_warp_options(gcps, method, order),
    )
    grid = _native_grid(src, out_crs, _as_resolution(resolution))
    out, dst_nodata = _warp(bands, src, grid, _resampling(resampling), nodata, None)
    meta = _out_meta(
        out.dtype, dst_nodata, out.shape[0], grid.height, grid.width, out_crs, grid.transform
    )
    return _squeeze(out, array.ndim == 2), meta


def rectify(
    path: PathLike,
    out_path: PathLike | None = None,
    *,
    bands: int | Sequence[int] | None = None,
    gcps: Sequence[GroundControlPoint] | None = None,
    gcps_crs: CRS | str | int | None = None,
    dst_crs: CRS | str | int | None = None,
    resolution: Resolution | None = None,
    method: Method = "polynomial",
    order: int | None = None,
    resampling: str | Resampling = "bilinear",
    nodata: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Rectify a GCP-referenced raster file, optionally writing a GeoTIFF.

    When the file's own GCPs are used, GDAL streams the source from disk, so only the
    output grid is held in memory; pass a coarse ``resolution`` for very large
    orthomosaics.

    Parameters
    ----------
    path : str or os.PathLike
        Input raster.
    out_path : str or os.PathLike, optional
        If given, the result is also written there as a compressed GeoTIFF.
    bands : int or sequence of int, optional
        1-based band index/indices to rectify (default: all bands). An int returns a
        2D array.
    gcps, gcps_crs : optional
        GCPs (and their CRS) to use instead of those stored in the file, e.g. points
        measured by the user with :func:`make_gcps`.
    dst_crs, resolution, method, order, resampling, nodata
        As in :func:`georeference`. ``nodata`` defaults to the file's nodata value.

    Returns
    -------
    rectified : numpy.ndarray
    metadata : dict
        As in :func:`georeference`.

    Raises
    ------
    ValueError
        If the file has no GCPs (and none are given) or its GCP CRS is unknown.

    Examples
    --------
    >>> arr, meta = rectify("raw.tif", "rectified.tif", resolution=0.1)  # doctest: +SKIP
    """
    path = os.fspath(path)
    resamp = _resampling(resampling)
    res = _as_resolution(resolution)
    with warnings.catch_warnings():
        # A raw image without any georeferencing is expected here (GCPs may be given).
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        ds = rasterio.open(path)
    with ds:
        file_gcps, file_gcps_crs = ds.gcps
        use_file_gcps = gcps is None
        if use_file_gcps and not file_gcps:
            raise ValueError(f"{path} has no ground control points; pass gcps= explicitly")
        gcps = _normalize_gcps(file_gcps if use_file_gcps else gcps)
        crs = _as_crs(gcps_crs if gcps_crs is not None else file_gcps_crs, f"GCP CRS of {path}")
        out_crs = crs if dst_crs is None else _as_crs(dst_crs, "dst_crs")
        indexes = list(range(1, ds.count + 1)) if bands is None else bands
        idx_list = [indexes] if isinstance(indexes, int) else list(indexes)
        for i in idx_list:
            if not 1 <= i <= ds.count:
                raise ValueError(f"band {i} out of range 1..{ds.count} for {path}")
        options = _warp_options(gcps, method, order)
        src = _Source(crs, ds.width, ds.height, gcps=gcps, options=options)
        grid = _native_grid(src, out_crs, res)
        if use_file_gcps and ds.crs is None and not ds.rpcs:
            # Stream from disk; GDAL uses the dataset's own GCPs.
            dtype = _source_dtype(np.dtype(ds.dtypes[idx_list[0] - 1]))
            src_nodata, out_dtype, dst_nodata = _nodata_policy(dtype, nodata, ds.nodata)
            _check_output_size((len(idx_list), grid.height, grid.width), "output grid")
            out = np.full((len(idx_list), grid.height, grid.width), dst_nodata, out_dtype)
            reproject(
                rasterio.band(ds, idx_list),
                out,
                src_crs=crs,
                src_nodata=src_nodata,
                dst_transform=grid.transform,
                dst_crs=out_crs,
                dst_nodata=dst_nodata,
                resampling=resamp,
                **src.options,
            )
        else:
            data = ds.read(idx_list)
            out, dst_nodata = _warp(data, src, grid, resamp, nodata, ds.nodata)
    meta = _out_meta(
        out.dtype, dst_nodata, out.shape[0], grid.height, grid.width, out_crs, grid.transform
    )
    if out_path is not None:
        _write(out_path, out, meta)
    return _squeeze(out, isinstance(indexes, int)), meta


def _write(path: PathLike, data: np.ndarray, meta: dict[str, Any]) -> None:
    # farq.write writes atomically (temporary file + rename), so a failure never leaves
    # a truncated GeoTIFF behind.
    options: dict[str, Any] = {"compress": "deflate"}
    if meta["width"] >= 256 and meta["height"] >= 256:
        options.update(tiled=True, blockxsize=256, blockysize=256)
    _core_write(path, data, meta, nodata=meta["nodata"], **options)


# --------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------


def _grid_from_meta(meta: Mapping[str, Any], name: str) -> _Grid:
    """Target grid described by metadata (GCP metadata yields its native grid)."""
    if not isinstance(meta, Mapping):
        raise TypeError(f"{name} must be a dict, got {type(meta).__name__}")
    try:
        width, height = int(meta["width"]), int(meta["height"])
    except KeyError as e:
        raise ValueError(f"{name} must define 'width' and 'height'") from e
    src = _source_from_meta(meta, height, width, "polynomial", None, name)
    if src.transform is not None:
        return _Grid(src.crs, src.transform, width, height)
    return _native_grid(src, src.crs, None)


def align(
    array: np.ndarray,
    meta: Mapping[str, Any],
    reference_meta: Mapping[str, Any],
    *,
    resampling: str | Resampling = "bilinear",
    nodata: float | None = None,
    method: Method = "polynomial",
    order: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Reproject a raster onto the grid of a reference raster.

    Parameters
    ----------
    array : numpy.ndarray
        Raster of shape (H, W) or (bands, H, W).
    meta : dict
        Its metadata: ``crs`` + ``transform``, or ``gcps`` + ``gcps_crs`` for a
        GCP-referenced image (warped directly, in a single resampling step).
    reference_meta : dict
        Metadata of the reference raster; its ``crs``, ``transform``, ``width`` and
        ``height`` define the output grid.
    resampling : str or rasterio.enums.Resampling
        Resampling algorithm (default ``"bilinear"``).
    nodata : float, optional
        Input nodata value (see the module notes on nodata).
    method, order
        GCP transform options, used only when ``meta`` is GCP-referenced (see
        :func:`georeference`).

    Returns
    -------
    aligned : numpy.ndarray
        Array of shape (height, width) or (bands, height, width) of the reference.
        Areas not covered by ``array`` are nodata/NaN.
    metadata : dict
        The reference grid with this result's ``dtype``, ``count`` and ``nodata``.

    Examples
    --------
    >>> after_on_before, m = align(after, after_meta, before_meta)  # doctest: +SKIP
    """
    bands = _as_bands(array, "array")
    src = _source_from_meta(meta, bands.shape[1], bands.shape[2], method, order, "meta")
    grid = _grid_from_meta(reference_meta, "reference_meta")
    meta_nodata = meta.get("nodata")
    out, dst_nodata = _warp(bands, src, grid, _resampling(resampling), nodata, meta_nodata)
    out_meta = _out_meta(
        out.dtype, dst_nodata, out.shape[0], grid.height, grid.width, grid.crs, grid.transform
    )
    return _squeeze(out, array.ndim == 2), out_meta


def _snap(lo: float, hi: float, origin: float, res: float) -> tuple[float, int]:
    """Snap [lo, hi] inwards onto the grid ``origin + k * res``; return (start, n)."""
    k0 = math.ceil((lo - origin) / res - 1e-6)
    k1 = math.floor((hi - origin) / res + 1e-6)
    return origin + k0 * res, k1 - k0


def align_pair(
    before: np.ndarray,
    before_meta: Mapping[str, Any],
    after: np.ndarray,
    after_meta: Mapping[str, Any],
    *,
    target: Literal["before", "after", "coarsest", "finest"] = "before",
    dst_crs: CRS | str | int | None = None,
    resolution: Resolution | None = None,
    resampling: str | Resampling = "bilinear",
    nodata: float | None = None,
    method: Method = "polynomial",
    order: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Put two rasters on one common pixel grid, cropped to their overlap.

    Use this before any pixel-wise change detection between images with different
    extents, resolutions, CRSs or georeferencing (affine transform or GCPs).

    Parameters
    ----------
    before, after : numpy.ndarray
        Rasters of shape (H, W) or (bands, H, W). Shapes may differ.
    before_meta, after_meta : dict
        Their metadata (``crs`` + ``transform``, or ``gcps`` + ``gcps_crs``).
    target : {"before", "after", "coarsest", "finest"}
        Which pixel size and grid origin to use. ``"before"``/``"after"`` keep that
        raster's grid, so it is only cropped, never resampled. ``"coarsest"`` (most
        robust for change detection between different sensors/flights) and
        ``"finest"`` pick the pixel size per axis and snap to the grid of the raster
        that provides it.
    dst_crs : CRS, str or int, optional
        Common CRS (default: the CRS of ``before``, or of ``after`` if
        ``target="after"``).
    resolution : float or (float, float), optional
        Explicit common pixel size, overriding ``target``'s pixel size.
    resampling : str or rasterio.enums.Resampling
        Resampling algorithm (default ``"bilinear"``; prefer ``"average"`` when
        downsampling a lot, ``"nearest"`` for categorical data).
    nodata : float, optional
        Input nodata value applied to both rasters (see the module notes).
    method, order
        GCP transform options for GCP-referenced inputs (see :func:`georeference`).

    Returns
    -------
    before_aligned, after_aligned : numpy.ndarray
        Arrays with identical spatial shape covering exactly the overlap. If the two
        outputs would have different dtypes or different nodata values they are both
        promoted to floating point, with NaN as nodata.
    metadata : dict
        Common grid profile (``crs``, ``transform``, ``width``, ``height``, ``dtype``,
        ``nodata``; ``count`` is that of ``before``).

    Raises
    ------
    ValueError
        If the rasters do not overlap by at least one pixel, or on invalid inputs.

    Examples
    --------
    >>> b, a, meta = align_pair(rgb_2023, m23, rgb_2024, m24, target="coarsest")  # doctest: +SKIP
    >>> shift = coregister(b, a)  # doctest: +SKIP
    >>> a = apply_shift(a, shift)  # doctest: +SKIP
    """
    if target not in ("before", "after", "coarsest", "finest"):
        raise ValueError(
            f"target must be 'before', 'after', 'coarsest' or 'finest', got {target!r}"
        )
    b3, a3 = _as_bands(before, "before"), _as_bands(after, "after")
    src_b = _source_from_meta(before_meta, b3.shape[1], b3.shape[2], method, order, "before")
    src_a = _source_from_meta(after_meta, a3.shape[1], a3.shape[2], method, order, "after")
    if dst_crs is not None:
        crs = _as_crs(dst_crs, "dst_crs")
    else:
        crs = src_a.crs if target == "after" else src_b.crs
    grid_b, grid_a = _native_grid(src_b, crs, None), _native_grid(src_a, crs, None)

    # Pixel size and grid anchor.
    (xb, yb), (xa, ya) = grid_b.res, grid_a.res
    if target == "after":
        anchor, xres, yres = grid_a, xa, ya
    elif target == "coarsest":
        xres, yres = max(xb, xa), max(yb, ya)
        anchor = grid_a if (xa > xb and ya >= yb) or (ya > yb and xa >= xb) else grid_b
    elif target == "finest":
        xres, yres = min(xb, xa), min(yb, ya)
        anchor = grid_a if (xa < xb and ya <= yb) or (ya < yb and xa <= xb) else grid_b
    else:
        anchor, xres, yres = grid_b, xb, yb
    explicit = _as_resolution(resolution)
    if explicit is not None:
        xres, yres = explicit

    # Intersection, snapped inwards onto the anchor grid.
    lb, bb, rb, tb = grid_b.bounds
    la, ba, ra, ta = grid_a.bounds
    left, right = max(lb, la), min(rb, ra)
    bottom, top = max(bb, ba), min(tb, ta)
    if left >= right or bottom >= top:
        raise ValueError(
            "The rasters do not overlap: before bounds "
            f"{(lb, bb, rb, tb)}, after bounds {(la, ba, ra, ta)} in {crs}"
        )
    x0, width = _snap(left, right, anchor.transform.c, xres)
    # Rows run downwards from the anchor's top edge.
    k0 = math.ceil((anchor.transform.f - top) / yres - 1e-6)
    k1 = math.floor((anchor.transform.f - bottom) / yres + 1e-6)
    y0, height = anchor.transform.f - k0 * yres, k1 - k0
    if width < 1 or height < 1:
        raise ValueError("The rasters overlap by less than one output pixel")
    grid = _Grid(crs, Affine(xres, 0.0, x0, 0.0, -yres, y0), width, height)

    resamp = _resampling(resampling)
    out_b, nd_b = _warp(b3, src_b, grid, resamp, nodata, before_meta.get("nodata"))
    out_a, nd_a = _warp(a3, src_a, grid, resamp, nodata, after_meta.get("nodata"))
    if out_b.dtype != out_a.dtype or not _same_nodata(nd_b, nd_a):
        # One metadata dict must describe both outputs: with different dtypes or
        # different nodata values (e.g. 0 and 255), promote both to float with NaN.
        common = np.result_type(out_b.dtype, out_a.dtype, np.float32)
        out_b = _promote(out_b, nd_b, common)
        out_a = _promote(out_a, nd_a, common)
        nd_b = np.nan
    meta = _out_meta(out_b.dtype, nd_b, out_b.shape[0], height, width, crs, grid.transform)
    return _squeeze(out_b, before.ndim == 2), _squeeze(out_a, after.ndim == 2), meta


def _same_nodata(a: float, b: float) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def _promote(out: np.ndarray, nodata: float, dtype: np.dtype) -> np.ndarray:
    if out.dtype == dtype:
        return out
    result = out.astype(dtype)
    if not (isinstance(nodata, float) and math.isnan(nodata)):
        result[out == nodata] = np.nan
    return result


# --------------------------------------------------------------------------------------
# Co-registration (translation only)
# --------------------------------------------------------------------------------------


def _to_gray(array: np.ndarray, name: str) -> np.ndarray:
    validate_array(array, name=name)
    if array.ndim == 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(array, axis=0, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D or 3D (bands, H, W), got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def _standardize(img: np.ndarray, name: str) -> np.ndarray:
    """Fill NaN with the mean and scale to zero mean, unit variance (float32)."""
    valid = np.isfinite(img)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 0.1 * img.size:
        raise ValueError(f"{name} has too few valid pixels for co-registration")
    mean = float(img[valid].mean())
    out = np.where(valid, img - mean, 0.0).astype(np.float32)
    std = float(out[valid].std())
    if std == 0:
        raise ValueError(f"{name} is constant; it has no texture to co-register")
    out /= std
    return out


def _upsampled_dft(data: np.ndarray, size: int, factor: int, offsets: np.ndarray) -> np.ndarray:
    """Matrix-multiply DFT of ``data`` on a ``size``-square region upsampled by ``factor``."""
    im2pi = 2j * np.pi
    for n_items, offset in list(zip(data.shape, offsets))[::-1]:
        kernel = (np.arange(size) - offset)[:, None] * np.fft.fftfreq(n_items, factor)
        data = np.tensordot(np.exp(-im2pi * kernel), data, axes=(1, -1))
    return data


def _phase_correlate(
    f_ref: np.ndarray,
    mov: np.ndarray,
    window: np.ndarray | None,
    upsample: int,
    whitening: float,
) -> np.ndarray:
    """One phase correlation pass; ``f_ref`` is the FFT of the (windowed) reference."""
    tapered = mov * window if window is not None else mov
    product = f_ref * sp_fft.fft2(tapered, workers=-1).conj()
    if whitening > 0:
        magnitude = np.abs(product)
        np.maximum(magnitude, 100 * np.finfo(magnitude.dtype).eps, out=magnitude)
        product /= magnitude if whitening == 1 else magnitude**whitening
        del magnitude
    corr = np.abs(sp_fft.ifft2(product, workers=-1))
    shape = np.array(corr.shape)
    shifts = np.array(np.unravel_index(np.argmax(corr), corr.shape), dtype=np.float64)
    del corr
    wrap = shifts > np.trunc(shape / 2)
    shifts[wrap] -= shape[wrap]
    if upsample > 1:
        shifts = np.round(shifts * upsample) / upsample
        region = math.ceil(upsample * 1.5)
        dftshift = np.trunc(region / 2.0)
        offsets = dftshift - shifts * upsample
        up = _upsampled_dft(product.conj(), region, upsample, offsets).conj()
        peak = np.array(np.unravel_index(np.argmax(np.abs(up)), up.shape), dtype=np.float64)
        shifts += (peak - dftshift) / upsample
    return shifts


def coregister(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    upsample: int = 10,
    window: bool = True,
    whitening: float = 0.5,
    refine: int = 1,
) -> tuple[float, float]:
    """
    Estimate the sub-pixel translation that aligns ``moving`` with ``reference``.

    Uses FFT-based phase correlation with matrix-multiply DFT refinement around the
    correlation peak (Guizar-Sicairos et al., 2008). Use it after :func:`align_pair`
    to remove the small residual misalignment (GNSS or GCP error) between two drone
    flights that otherwise shows up as false change along every edge.

    Only a *translation* is estimated: rotation, scale differences and local
    distortions (e.g. relief displacement, moving objects) are not modelled. The
    inputs must already be on the same grid, and the shift should be well below half
    the image size. For very large orthomosaics, estimate the shift on a downsampled
    copy or a representative crop (the FFT needs a few times the image memory).

    Parameters
    ----------
    reference, moving : numpy.ndarray
        Images of identical shape, (H, W) or (bands, H, W); multi-band images are
        averaged into one band. NaN pixels are filled with the image mean.
    upsample : int
        Sub-pixel precision is ``1 / upsample`` pixels (default 10 = 0.1 px).
    window : bool
        Taper both images with a Hann window to suppress edge effects (default True).
    whitening : float
        Exponent of the cross-power spectrum normalization, in [0, 1]. ``1`` is classic
        phase correlation (most robust to illumination differences), ``0`` is plain
        cross-correlation. The default ``0.5`` is a robust compromise that is also
        accurate for smooth imagery.
    refine : int
        Number of refinement passes: ``moving`` is shifted by the current estimate and
        the residual shift is measured again. This removes the bias of the window
        towards zero shift (default 1).

    Returns
    -------
    (dy, dx) : tuple of float
        Shift in pixels (rows, columns) to apply to ``moving`` with
        :func:`apply_shift`. If ``moving`` is ``reference`` displaced by ``(sy, sx)``
        the result is ``(-sy, -sx)``.

    Raises
    ------
    ValueError
        Shape mismatch, constant images, too few valid pixels or invalid parameters.

    Examples
    --------
    >>> dy, dx = coregister(before, after)  # doctest: +SKIP
    >>> after_fixed = apply_shift(after, (dy, dx))  # doctest: +SKIP
    """
    if isinstance(upsample, bool) or not isinstance(upsample, (int, np.integer)) or upsample < 1:
        raise ValueError(f"upsample must be a positive integer, got {upsample!r}")
    if not 0 <= whitening <= 1:
        raise ValueError(f"whitening must be in [0, 1], got {whitening!r}")
    if isinstance(refine, bool) or not isinstance(refine, (int, np.integer)) or refine < 0:
        raise ValueError(f"refine must be a non-negative integer, got {refine!r}")
    ref = _to_gray(reference, "reference")
    mov = _to_gray(moving, "moving")
    if ref.shape != mov.shape:
        raise ValueError(f"reference {ref.shape} and moving {mov.shape} shapes differ")
    if min(ref.shape) < 8:
        raise ValueError("images must be at least 8x8 pixels for co-registration")
    win = None
    if window:
        win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1])).astype(np.float32)
    ref_std = _standardize(ref, "reference")
    f_ref = sp_fft.fft2(ref_std * win if win is not None else ref_std, workers=-1)
    del ref_std
    mov = _standardize(mov, "moving")
    shift = _phase_correlate(f_ref, mov, win, upsample, whitening)
    for _ in range(refine):
        shifted = ndimage.shift(mov, shift, order=1, mode="nearest")
        shift = shift + _phase_correlate(f_ref, shifted, win, upsample, whitening)
    return round(float(shift[0]), 6), round(float(shift[1]), 6)


def apply_shift(
    array: np.ndarray,
    shift: tuple[float, float],
    *,
    order: int = 1,
) -> np.ndarray:
    """
    Translate an image by a (sub-pixel) shift, marking uncovered pixels as NaN.

    Parameters
    ----------
    array : numpy.ndarray
        Image of shape (H, W) or (bands, H, W).
    shift : (float, float)
        ``(dy, dx)`` in pixels, e.g. from :func:`coregister`.
    order : int
        Spline interpolation order (0 = nearest, 1 = bilinear (default), 3 = cubic).

    Returns
    -------
    numpy.ndarray
        Floating point array (``float32`` for integer input) of the same shape. Pixels
        shifted in from outside the image, or interpolated from NaN pixels, are NaN.

    Examples
    --------
    >>> after_fixed = apply_shift(after, coregister(before, after))  # doctest: +SKIP
    """
    validate_array(array, name="array", allow_all_nan=True)
    if array.ndim not in (2, 3):
        raise ValueError(f"array must be 2D or 3D (bands, H, W), got shape {array.shape}")
    dy, dx = (float(s) for s in shift)
    if not (math.isfinite(dy) and math.isfinite(dx)):
        raise ValueError(f"shift must be finite, got {shift!r}")
    if not 0 <= order <= 5:
        raise ValueError("order must be between 0 and 5")
    dtype = array.dtype if array.dtype.kind == "f" and array.dtype.itemsize >= 4 else np.float32
    full_shift = (dy, dx) if array.ndim == 2 else (0.0, dy, dx)
    data = array.astype(dtype, copy=False)
    valid = np.isfinite(data)
    if not valid.all():
        data = np.where(valid, data, 0).astype(dtype, copy=False)
    out = ndimage.shift(data, full_shift, order=order, mode="constant", cval=0.0)
    out = out.astype(dtype, copy=False)
    coverage = ndimage.shift(
        valid.astype(np.float32), full_shift, order=min(order, 1), mode="constant", cval=0.0
    )
    out[coverage < 1 - 1e-3] = np.nan
    return out


# --------------------------------------------------------------------------------------
# Pixel geometry
# --------------------------------------------------------------------------------------


def _linear_transform(meta: Mapping[str, Any]) -> Affine:
    if not isinstance(meta, Mapping):
        raise TypeError(f"meta must be a dict, got {type(meta).__name__}")
    if _has_valid_transform(meta):
        return _as_transform(meta["transform"])
    if meta.get("gcps"):
        gcps = _normalize_gcps(meta["gcps"])
        _validate_gcp_geometry(gcps, 1)
        return _affine_from_gcps(gcps)
    raise ValueError("meta is not georeferenced: it has neither a valid 'transform' nor 'gcps'")


def pixel_size(meta: Mapping[str, Any]) -> tuple[float, float]:
    """
    Ground size of one pixel in CRS units.

    Parameters
    ----------
    meta : dict
        Raster metadata with a ``transform``, or ``gcps`` (then estimated from an
        affine least-squares fit through the GCPs).

    Returns
    -------
    (xres, yres) : tuple of float
        Pixel width and height (always positive; rotation is accounted for). The units
        are those of the CRS: metres for UTM, *degrees* for geographic CRSs such as
        EPSG:4326.

    Examples
    --------
    >>> pixel_size({"transform": Affine(0.05, 0, 5e5, 0, -0.05, 4e6), "crs": 32633})
    (0.05, 0.05)
    """
    t = _linear_transform(meta)
    return math.hypot(t.a, t.d), math.hypot(t.b, t.e)


def pixel_area(meta: Mapping[str, Any]) -> float:
    """
    Ground area of one pixel in squared CRS units (e.g. m² for UTM).

    Parameters
    ----------
    meta : dict
        Raster metadata with ``crs`` (or ``gcps_crs``) and ``transform`` (or ``gcps``).

    Returns
    -------
    float
        Pixel area.

    Raises
    ------
    ValueError
        If the CRS is missing or geographic (degrees are not a unit of length; reproject
        to a projected CRS such as UTM first, e.g. with :func:`align_pair`'s
        ``dst_crs``).

    Examples
    --------
    >>> round(pixel_area({"transform": Affine(0.05, 0, 5e5, 0, -0.05, 4e6), "crs": 32633}), 6)
    0.0025
    """
    t = _linear_transform(meta)
    crs_value = (
        meta.get("crs") if _has_valid_transform(meta) else (meta.get("gcps_crs") or meta.get("crs"))
    )
    crs = _as_crs(crs_value, "CRS")
    if crs.is_geographic:
        raise ValueError(
            f"{crs} is a geographic CRS with pixel sizes in degrees; pixel areas are "
            "undefined. Reproject to a projected CRS (e.g. UTM) first."
        )
    return abs(t.a * t.e - t.b * t.d)
