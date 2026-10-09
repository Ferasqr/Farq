"""
Vector export: turn change masks and class maps into polygons for GIS.

The functions in this module convert raster regions (connected pixels with the same
value) into polygons that QGIS, ArcGIS and other GIS software can open.

Workflow
--------
1. :func:`polygonize` traces the regions of a mask or class map into GeoJSON-like
   feature dicts, with per-polygon area, perimeter, pixel count and centroid.
2. :func:`to_geojson` builds a GeoJSON ``FeatureCollection`` (RFC 7946: lon/lat on
   WGS 84 by default) and optionally writes it to a file. It needs no extra
   dependency.
3. :func:`write_vector` writes GeoJSON, GeoPackage, Shapefile or FlatGeobuf. Formats
   other than GeoJSON need the optional ``pyogrio`` (preferred) or ``fiona`` package:
   ``pip install farq[vector]``.

:func:`changes_to_vector` does all three in one call for the output of
:func:`farq.detect_changes` or :func:`farq.classify_change`.

Conventions
-----------
* NaN, ±inf, the ``nodata`` value and masked entries are invalid and never become
  polygons. Boolean masks produce polygons for ``True`` regions only.
* Polygons follow pixel edges exactly. Rings are closed and follow the RFC 7946
  right-hand rule (exterior counter-clockwise, holes clockwise).
* ``area_m2``, ``perimeter_m`` and ``pixel_count`` describe the traced pixel region
  (before any simplification): ``area_m2 == pixel_count * pixel area`` and the
  perimeter is the length of the pixel-edge boundary, holes included. They are in
  m² and metres: CRS units other than metres (e.g. US survey feet) are converted,
  as in :func:`farq.change_summary` and :mod:`farq.elevation`; coordinates stay in
  the CRS units. Rasters in a geographic CRS (degrees) and rasters without a
  geotransform are refused with ``ValueError``.
* Simplification (``simplify=``) is optional; see :func:`polygonize` for its
  topology caveats.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import secrets
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain
from numbers import Real
from typing import Any

import numpy as np

__all__ = ["changes_to_vector", "polygonize", "to_geojson", "write_vector"]

#: Output drivers by file extension.
_EXTENSIONS = {
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".gpkg": "GPKG",
    ".shp": "ESRI Shapefile",
    ".fgb": "FlatGeobuf",
}
_DRIVER_ALIASES = {
    "geojson": "GeoJSON",
    "gpkg": "GPKG",
    "geopackage": "GPKG",
    "esri shapefile": "ESRI Shapefile",
    "shapefile": "ESRI Shapefile",
    "shp": "ESRI Shapefile",
    "flatgeobuf": "FlatGeobuf",
    "fgb": "FlatGeobuf",
}
# Component files of a Shapefile (removed when an existing Shapefile is replaced).
_SHAPEFILE_PARTS = frozenset(
    {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".qmd", ".shp.xml"}
)
# Shapefile (dBASE) field names are limited to 10 characters.
_SHAPEFILE_NAMES = {
    "perimeter_m": "perim_m",
    "pixel_count": "pixels",
    "mean_magnitude": "mean_mag",
    "max_magnitude": "max_mag",
}
# Schema used when an empty feature list is written to an OGR format.
_DEFAULT_FIELDS = {
    "value": 0.0,
    "label": "",
    "pixel_count": 0,
    "area_m2": 0.0,
    "perimeter_m": 0.0,
    "centroid_x": 0.0,
    "centroid_y": 0.0,
}
# Dtypes that GDAL polygonizes natively with exact integer values (rasterio >= 1.3).
_NATIVE_DTYPES = frozenset({np.dtype(t) for t in (np.uint8, np.uint16, np.int16, np.int32)})
_INSTALL_HINT = "pip install farq[vector]"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def _georeference(georef: Any) -> tuple[Any, Any]:
    """Return ``(Affine, CRS or None)`` from a rasterio meta, dataset or Affine."""
    from affine import Affine
    from rasterio.crs import CRS

    if isinstance(georef, Affine):
        transform, crs, gcps = georef, None, None
    elif isinstance(georef, Mapping):
        if "transform" not in georef:
            if georef.get("gcps"):
                raise ValueError(
                    "meta is georeferenced only by GCPs; rectify it first "
                    "(farq.rectify / farq.align_pair) to get a geotransform"
                )
            raise ValueError("meta must contain a 'transform' entry (rasterio meta/profile)")
        transform, crs, gcps = georef["transform"], georef.get("crs"), georef.get("gcps")
    elif hasattr(georef, "transform") and hasattr(georef, "crs"):  # open rasterio dataset
        transform, crs = georef.transform, georef.crs
        gcps = getattr(georef, "gcps", ([], None))[0]
    else:
        raise TypeError(
            "meta must be a rasterio meta/profile dict, an open rasterio dataset or an "
            f"affine.Affine transform, got {type(georef).__name__}"
        )
    if not isinstance(transform, Affine):
        raise TypeError(f"transform must be an affine.Affine, got {type(transform).__name__}")
    if transform.is_identity and (crs is None or gcps):
        raise ValueError(
            "the raster has no geotransform (identity transform"
            + (" with GCPs" if gcps else " and no CRS")
            + "), so polygons would be in pixel coordinates. Rectify/align the raster first "
            "(farq.rectify / farq.align_pair) or pass metadata with a real transform."
        )
    values = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    if not all(math.isfinite(v) for v in values) or transform.determinant == 0:
        raise ValueError(f"transform must be finite and invertible, got {transform!r}")
    if crs is not None:
        try:
            crs = CRS.from_user_input(crs)
        except Exception as exc:
            raise ValueError(f"invalid crs {crs!r}: {exc}") from exc
    return transform, crs


def _as_crs(crs: Any) -> Any:
    """CRS-like value (CRS, EPSG code, string, WKT or rasterio meta) -> CRS or None."""
    from rasterio.crs import CRS

    if isinstance(crs, Mapping) and not isinstance(crs, CRS):
        crs = crs.get("crs")
    if crs is None:
        return None
    try:
        return CRS.from_user_input(crs)
    except Exception as exc:
        raise ValueError(f"invalid crs {crs!r}: {exc}") from exc


def _as_2d(data: Any, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(array, invalid or None)`` for a 2-D or single-band 3-D array."""
    invalid = None
    if isinstance(data, np.ma.MaskedArray):
        m = np.ma.getmaskarray(data)
        invalid = m if m.any() else None
        data = np.asarray(data.data)
    if not isinstance(data, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(data).__name__}")
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
        invalid = None if invalid is None else invalid[0]
    if data.ndim != 2:
        raise ValueError(f"{name} must be 2-D (rows, cols) or (1, rows, cols), got {data.shape}")
    if data.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if data.dtype.kind not in "biuf":
        raise TypeError(f"{name} must have a boolean or numeric dtype, got {data.dtype}")
    return data, invalid


def _check_shape(data: np.ndarray, georef: Any) -> None:
    if isinstance(georef, Mapping):
        h, w = georef.get("height"), georef.get("width")
        if h is not None and w is not None and (int(h), int(w)) != data.shape:
            raise ValueError(f"data shape {data.shape} does not match meta height/width ({h}, {w})")


def _positive(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (Real, np.number)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value}")
    return value


@contextlib.contextmanager
def _gc_paused() -> Iterator[None]:
    """Pause the cyclic garbage collector while building many small containers.

    Building hundreds of thousands of feature dicts triggers repeated full
    collections that roughly double the run time; nothing created here is cyclic.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _py(value: Any) -> Any:
    """numpy scalar -> plain Python scalar (bool -> int for GIS attribute tables)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    return value


# --------------------------------------------------------------------------- #
# Tracing and metrics
# --------------------------------------------------------------------------- #
def _encode(data: np.ndarray, include: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(codes for rasterio, value table or None)``.

    Dtypes GDAL handles exactly are passed through (value table None). Anything else
    (bool, float, 64-bit, int8, uint32) is mapped to ``int32`` codes ``1..n`` that
    index the sorted unique values, so float values come back exactly.
    """
    if data.dtype == bool:
        return data.view(np.uint8), None
    if data.dtype in _NATIVE_DTYPES:
        return data, None
    uniq, inverse = np.unique(data[include], return_inverse=True)
    if uniq.size >= np.iinfo(np.int32).max:
        raise ValueError("too many distinct values to polygonize")
    codes = np.zeros(data.shape, dtype=np.int32)
    codes[include] = inverse.reshape(-1).astype(np.int32) + 1
    return codes, uniq


def _trace(
    codes: np.ndarray, include: np.ndarray, connectivity: int
) -> tuple[list[float], list[list[list[tuple[float, float]]]]]:
    """Polygonize in pixel coordinates (x = col, y = row); return values and rings."""
    from rasterio.features import shapes

    values: list[float] = []
    polys: list[list[list[tuple[float, float]]]] = []
    for geom, value in shapes(codes, mask=include, connectivity=connectivity):
        values.append(value)
        polys.append(geom["coordinates"])
    return values, polys


def _ring_sums(terms: np.ndarray, ends: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sum per-segment ``terms`` (length N-1) over each ring's segments."""
    terms = terms.copy()
    terms[ends[:-1] - 1] = 0.0  # segment joining the last vertex of a ring to the next ring
    return np.add.reduceat(np.append(terms, 0.0), starts)


class _Traced:
    """Flattened rings of all traced polygons, with vectorized metrics."""

    def __init__(self, polys: list[list[list[tuple[float, float]]]], transform: Any) -> None:
        ring_len = np.fromiter((len(r) for p in polys for r in p), dtype=np.intp)
        rings_per_poly = np.fromiter((len(p) for p in polys), dtype=np.intp, count=len(polys))
        xy = np.array(list(chain.from_iterable(chain.from_iterable(polys))), dtype=np.float64)
        xy = xy.reshape(-1, 2)
        ends = np.cumsum(ring_len)
        starts = ends - ring_len
        self.ring_starts, self.ring_ends = starts, ends
        self.poly_ring_starts = np.cumsum(rings_per_poly) - rings_per_poly

        x, y = xy[:, 0], xy[:, 1]
        dx, dy = np.diff(x), np.diff(y)
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        ring_area2 = _ring_sums(cross, ends, starts)  # 2 x signed area, pixel units
        mx = _ring_sums((x[:-1] + x[1:]) * cross, ends, starts)
        my = _ring_sums((y[:-1] + y[1:]) * cross, ends, starts)
        t = transform
        seg = np.hypot(t.a * dx + t.b * dy, t.d * dx + t.e * dy)
        ring_perim = _ring_sums(seg, ends, starts)

        # The first ring of each polygon is the exterior, the others are holes.
        is_exterior = np.zeros(ring_len.size, dtype=bool)
        is_exterior[self.poly_ring_starts] = True
        sign = np.where(is_exterior, 1.0, -1.0) * np.sign(ring_area2)
        area2 = np.add.reduceat(sign * ring_area2, self.poly_ring_starts)
        self.pixel_count = np.rint(area2 / 2.0).astype(np.int64)
        self.perimeter = np.add.reduceat(ring_perim, self.poly_ring_starts)
        cx = np.add.reduceat(sign * mx, self.poly_ring_starts) / (3.0 * area2)
        cy = np.add.reduceat(sign * my, self.poly_ring_starts) / (3.0 * area2)
        self.centroid_x = t.a * cx + t.b * cy + t.c
        self.centroid_y = t.d * cx + t.e * cy + t.f

        # World coordinates and RFC 7946 orientation (exterior counter-clockwise in
        # x/y; an affine with negative determinant mirrors the pixel-space winding).
        self.world = np.column_stack((t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f))
        world_ccw = np.sign(ring_area2) * np.sign(transform.determinant) > 0
        self.reverse = world_ccw != is_exterior

    def _ring_ranges(self, poly: int) -> range:
        first = self.poly_ring_starts[poly]
        last = (
            self.poly_ring_starts[poly + 1]
            if poly + 1 < self.poly_ring_starts.size
            else self.ring_starts.size
        )
        return range(first, last)

    def rings(self, poly: int) -> list[np.ndarray]:
        """World-coordinate rings of one polygon as arrays, correctly oriented."""
        out = []
        for r in self._ring_ranges(poly):
            ring = self.world[self.ring_starts[r] : self.ring_ends[r]]
            out.append(ring[::-1] if self.reverse[r] else ring)
        return out

    def ring_lists(self, polys: Sequence[int]) -> list[list[list[list[float]]]]:
        """World-coordinate rings of several polygons as nested lists (one tolist call)."""
        coords = self.world.tolist()
        starts, ends, reverse = (
            a.tolist() for a in (self.ring_starts, self.ring_ends, self.reverse)
        )
        out = []
        for poly in polys:
            rings = []
            for r in self._ring_ranges(poly):
                ring = coords[starts[r] : ends[r]]
                rings.append(ring[::-1] if reverse[r] else ring)
            out.append(rings)
        return out


# --------------------------------------------------------------------------- #
# Simplification
# --------------------------------------------------------------------------- #
def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    ap = points - a
    if denom == 0.0:
        return np.hypot(ap[:, 0], ap[:, 1])
    t = np.clip(ap @ ab / denom, 0.0, 1.0)
    d = ap - t[:, None] * ab
    return np.hypot(d[:, 0], d[:, 1])


def _douglas_peucker(ring: np.ndarray, tolerance: float) -> np.ndarray | None:
    """Simplify a closed ring; ``None`` if it collapses below 3 distinct vertices."""
    n = ring.shape[0]
    if n <= 4:
        return ring
    pts = ring - ring[0]  # local coordinates for precision
    far = int(np.argmax(np.hypot(pts[:-1, 0], pts[:-1, 1])))
    keep = np.zeros(n, dtype=bool)
    keep[[0, far, n - 1]] = True
    stack = [(0, far), (far, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        d = _segment_distance(pts[i + 1 : j], pts[i], pts[j])
        k = int(np.argmax(d))
        if d[k] > tolerance:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    out = ring[keep]
    return out if out.shape[0] >= 4 else None


def _simplify_dp(polys: list[list[np.ndarray]], tolerance: float) -> list[list[np.ndarray]]:
    out = []
    for rings in polys:
        exterior = _douglas_peucker(rings[0], tolerance)
        new = [rings[0] if exterior is None else exterior]
        for hole in rings[1:]:
            simplified = _douglas_peucker(hole, tolerance)
            if simplified is not None:  # holes smaller than the tolerance are dropped
                new.append(simplified)
        out.append(new)
    return out


def _simplify_coverage(polys: list[list[np.ndarray]], tolerance: float) -> list[list[np.ndarray]]:
    try:
        import shapely
    except ImportError as exc:
        raise ImportError(
            f"preserve_topology=True requires shapely>=2.1 ({_INSTALL_HINT} or "
            "pip install 'shapely>=2.1')"
        ) from exc
    if not hasattr(shapely, "coverage_simplify"):
        raise ImportError(
            f"preserve_topology=True requires shapely>=2.1, found {shapely.__version__}"
        )
    if not polys:
        return []
    geoms = np.empty(len(polys), dtype=object)
    geoms[:] = [shapely.Polygon(rings[0], rings[1:]) for rings in polys]
    simplified = shapely.coverage_simplify(geoms, tolerance)
    out = []
    for geom, rings in zip(simplified, polys):
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            out.append(rings)
            continue
        new = [np.asarray(geom.exterior.coords)]
        new.extend(np.asarray(h.coords) for h in geom.interiors)
        out.append([_orient(r, i == 0) for i, r in enumerate(new)])
    return out


def _signed_area2(ring: np.ndarray) -> float:
    p = ring - ring[0]
    return float(np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]))


def _orient(ring: np.ndarray, exterior: bool) -> np.ndarray:
    """Exterior counter-clockwise, holes clockwise (RFC 7946 right-hand rule)."""
    return ring if (_signed_area2(ring) > 0) == exterior else ring[::-1]


# --------------------------------------------------------------------------- #
# Polygonize
# --------------------------------------------------------------------------- #
def _build_features(
    codes: np.ndarray,
    include: np.ndarray,
    transform: Any,
    crs: Any,
    *,
    connectivity: int,
    min_area: float | None,
    simplify: float | None,
    preserve_topology: bool,
    describe: Any,
) -> list[dict[str, Any]]:
    """Shared engine: trace ``codes`` and build features.

    ``describe(code) -> dict`` returns the leading properties of a polygon.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if crs is not None and crs.is_geographic:
        raise ValueError(
            f"{crs} is a geographic CRS (degrees), so polygon areas and perimeters in "
            "metres are undefined. Reproject the raster to a projected CRS (e.g. UTM, "
            "farq.align with dst_crs=...) before polygonizing; to_geojson then writes "
            "WGS 84 lon/lat."
        )
    if not include.any():
        return []
    from .change import _metres_per_unit

    with _gc_paused():
        return _features(
            codes,
            include,
            transform,
            _metres_per_unit(crs),
            connectivity=connectivity,
            min_area=min_area,
            simplify=simplify,
            preserve_topology=preserve_topology,
            describe=describe,
        )


def _features(
    codes: np.ndarray,
    include: np.ndarray,
    transform: Any,
    unit: float,
    *,
    connectivity: int,
    min_area: float | None,
    simplify: float | None,
    preserve_topology: bool,
    describe: Any,
) -> list[dict[str, Any]]:
    values, polys = _trace(codes, include, connectivity)
    traced = _Traced(polys, transform)
    t = transform
    pixel_area = abs(t.a * t.e - t.b * t.d) * unit * unit  # m² (unit = metres per CRS unit)
    area = traced.pixel_count * pixel_area
    keep = np.arange(len(values))
    if min_area:
        keep = keep[area >= min_area]
    if simplify:
        arrays = [traced.rings(i) for i in keep]
        if preserve_topology:
            arrays = _simplify_coverage(arrays, simplify)
        else:
            arrays = _simplify_dp(arrays, simplify)
        geometries = [[r.tolist() for r in poly] for poly in arrays]
    else:
        geometries = traced.ring_lists(keep.tolist())

    cache: dict[float, dict[str, Any]] = {}
    pixels = traced.pixel_count.tolist()
    areas = area.tolist()
    perimeters = (traced.perimeter * unit).tolist()
    cxs, cys = traced.centroid_x.tolist(), traced.centroid_y.tolist()
    features = []
    for fid, (i, coords) in enumerate(zip(keep.tolist(), geometries)):
        code = values[i]
        if code not in cache:
            cache[code] = describe(code)
        props = dict(cache[code])
        props["pixel_count"] = pixels[i]
        props["area_m2"] = areas[i]
        props["perimeter_m"] = perimeters[i]
        props["centroid_x"] = cxs[i]
        props["centroid_y"] = cys[i]
        features.append(
            {
                "type": "Feature",
                "id": fid,
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": props,
            }
        )
    return features


def polygonize(
    data: np.ndarray,
    meta: Any,
    *,
    valid: np.ndarray | None = None,
    connectivity: int = 4,
    values: Any = None,
    min_area: float | None = None,
    simplify: float | None = None,
    preserve_topology: bool = False,
    labels: Mapping[Any, str] | None = None,
    nodata: float | None = None,
) -> list[dict[str, Any]]:
    """Trace connected regions of a mask or class map into polygons.

    Parameters
    ----------
    data : numpy.ndarray
        2-D (or ``(1, rows, cols)``) boolean mask or numeric class map. Boolean masks
        produce polygons for ``True`` regions; other arrays produce one polygon per
        connected region of equal value. NaN, ±inf and masked entries are invalid.
    meta : dict, rasterio dataset or affine.Affine
        Georeferencing: a rasterio meta/profile dict (``transform`` and ``crs``), an
        open dataset, or a bare ``Affine`` (CRS unknown: areas are in squared CRS
        units, assumed to be metres). Refused with ``ValueError``: rasters with no
        geotransform (identity transform without a CRS, or GCP-only) and geographic
        CRSs (degrees; reproject to a projected CRS such as UTM first).
    valid : numpy.ndarray, optional
        Boolean mask; pixels where it is False are excluded.
    connectivity : {4, 8}
        Pixel connectivity. ``4`` (default) yields OGC-valid polygons. With ``8``,
        diagonally touching pixels join one polygon whose ring touches itself at a
        vertex (accepted by ArcGIS, but invalid by OGC rules).
    values : scalar or sequence, optional
        Only polygonize these values. Default: ``True`` for boolean masks, every valid
        value otherwise.
    min_area : float, optional
        Drop polygons smaller than this area in m² (see the module notes on units).
    simplify : float, optional
        Douglas–Peucker tolerance in CRS units. Each ring is simplified on its own,
        so neighbouring polygons may develop small gaps or overlaps along shared
        edges, and a simplified ring may self-intersect. Holes that collapse are
        dropped; an exterior that would collapse is kept unsimplified. ``None`` or
        0 keeps the exact pixel edges.
    preserve_topology : bool
        Simplify all polygons together with :func:`shapely.coverage_simplify`, which
        keeps shared edges identical (no gaps or overlaps). Requires ``shapely>=2.1``
        and ``connectivity=4``.
    labels : mapping, optional
        Names for values, e.g. :data:`farq.CHANGE_LABELS`. Boolean masks default to
        ``{True: "changed", False: "unchanged"}``; unlabelled values use ``str(value)``.
    nodata : float, optional
        Value marking invalid pixels (e.g. :data:`farq.CHANGE_NODATA`).

    Returns
    -------
    list of dict
        GeoJSON-like features (``type``, ``id``, ``geometry``, ``properties``) in the
        raster CRS. Geometries are ``Polygon`` with ``[x, y]`` coordinates. Properties:
        ``value`` (``1`` for boolean masks), ``label``, ``pixel_count``, ``area_m2``,
        ``perimeter_m`` (pixel-edge length including holes) and ``centroid_x``/
        ``centroid_y`` (area centroid in the raster CRS; it can lie outside a
        non-convex polygon). All values are plain Python
        types. Metrics describe the pixel region, before simplification.

    Raises
    ------
    TypeError, ValueError
        Invalid input, unreferenced metadata or a geographic CRS.

    Examples
    --------
    >>> import numpy as np
    >>> from rasterio.transform import from_origin
    >>> mask = np.zeros((5, 5), bool)
    >>> mask[1:4, 1:4] = True
    >>> meta = {"transform": from_origin(500000, 4000000, 10, 10), "crs": "EPSG:32633"}
    >>> feature = polygonize(mask, meta)[0]
    >>> props = feature["properties"]
    >>> props["pixel_count"], props["area_m2"], props["perimeter_m"]
    (9, 900.0, 120.0)
    """
    data, invalid = _as_2d(data, "data")
    _check_shape(data, meta)
    transform, crs = _georeference(meta)
    include = np.ones(data.shape, dtype=bool) if invalid is None else ~invalid
    if data.dtype.kind == "f":
        include &= np.isfinite(data)
    if nodata is not None and data.dtype != bool and not np.isnan(nodata):
        include &= data != nodata
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != data.shape:
            raise ValueError(f"valid shape {valid.shape} does not match data shape {data.shape}")
        include &= valid
    is_bool = data.dtype == bool
    if values is None:
        if is_bool:
            include &= data
    else:
        selected = np.atleast_1d(np.asarray(values))
        if selected.size == 0:
            raise ValueError("values cannot be empty")
        include &= np.isin(data, selected)
    if labels is None and is_bool:
        labels = {True: "changed", False: "unchanged"}
    if labels is not None and not isinstance(labels, Mapping):
        raise TypeError(f"labels must be a mapping, got {type(labels).__name__}")
    min_area = _positive(min_area, "min_area")
    simplify = _positive(simplify, "simplify")

    codes, table = _encode(data, include)

    def describe(code: float) -> dict[str, Any]:
        value = table[int(code) - 1] if table is not None else code
        value = _py(np.asarray(value).astype(data.dtype)[()])
        label = labels.get(value) if labels else None
        return {"value": value, "label": str(value) if label is None else str(label)}

    return _build_features(
        codes,
        include,
        transform,
        crs,
        connectivity=connectivity,
        min_area=min_area,
        simplify=simplify,
        preserve_topology=preserve_topology,
        describe=describe,
    )


# --------------------------------------------------------------------------- #
# GeoJSON
# --------------------------------------------------------------------------- #
def _feature_list(features: Any) -> list[Mapping[str, Any]]:
    if isinstance(features, Mapping):
        if features.get("type") != "FeatureCollection":
            raise ValueError("features must be a list of features or a FeatureCollection")
        features = features.get("features", [])
    if isinstance(features, (str, bytes)) or not isinstance(features, Iterable):
        raise TypeError("features must be a list of GeoJSON-like feature dicts")
    out = list(features)
    for i, feat in enumerate(out):
        if not isinstance(feat, Mapping) or not isinstance(feat.get("geometry"), Mapping):
            raise ValueError(f"feature {i} is not a GeoJSON-like dict with a 'geometry'")
    return out


_PLAIN = (str, int, type(None))


def _clean_value(value: Any) -> Any:
    """JSON-safe property value: plain Python values, non-finite floats -> None (also
    inside lists and dicts, which ``json.dumps(allow_nan=False)`` would reject)."""
    kind = type(value)
    if kind is float:
        return value if value - value == 0.0 else None  # False for NaN and ±inf
    if kind in _PLAIN:
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):  # GeoJSON allows arrays and objects as values
        return [_clean_value(v) for v in value]
    if isinstance(value, Mapping):
        return {k if type(k) is str else str(k): _clean_value(v) for k, v in value.items()}
    value = _py(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean_properties(props: Mapping[str, Any] | None) -> dict[str, Any]:
    if not props:
        return {}
    return {k if type(k) is str else str(k): _clean_value(v) for k, v in props.items()}


def _other_geometry(
    geom: Mapping[str, Any], src: Any, dst: Any, precision: int | None
) -> dict[str, Any]:
    """Copy (and reproject) a point or line geometry."""
    if src is not None:
        from rasterio.warp import transform_geom

        geom = transform_geom(src, dst, geom)
    arr = np.asarray(geom["coordinates"], dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("geometry has non-finite coordinates (failed reprojection?)")
    if precision is not None:
        arr = np.round(arr, precision)
    return {"type": geom["type"], "coordinates": arr.tolist()}


def _geometries(
    geoms: Sequence[Mapping[str, Any]], src: Any, dst: Any, precision: int | None
) -> list[dict[str, Any]]:
    """Copy geometries: reproject ``src`` -> ``dst`` (unless ``src`` is None), apply
    the RFC 7946 winding order and round. Polygon rings are processed in one
    vectorized pass over all coordinates.
    """
    out: list[Any] = [None] * len(geoms)
    polygonal: list[tuple[int, str, int]] = []  # (index, type, number of polygons)
    rings_per_poly: list[int] = []
    rings: list[Any] = []
    for i, geom in enumerate(geoms):
        kind, coords = geom.get("type"), geom.get("coordinates")
        if kind in ("Point", "MultiPoint", "LineString", "MultiLineString"):
            out[i] = _other_geometry(geom, src, dst, precision)
            continue
        if kind not in ("Polygon", "MultiPolygon") or not coords:
            raise ValueError(f"feature {i}: unsupported or empty geometry of type {kind!r}")
        polys = [coords] if kind == "Polygon" else coords
        polygonal.append((i, kind, len(polys)))
        for poly in polys:
            if not poly:
                raise ValueError(f"feature {i}: empty polygon")
            rings_per_poly.append(len(poly))
            rings.extend(poly)
    if not polygonal:
        return out

    ring_len = np.fromiter((len(r) for r in rings), dtype=np.intp, count=len(rings))
    if (ring_len < 4).any():
        raise ValueError("polygon rings must have at least 4 positions (closed rings)")
    try:
        xy = np.array(list(chain.from_iterable(rings)), dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"invalid polygon coordinates: {exc}") from None
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("polygon coordinates must be [x, y] positions")
    ends = np.cumsum(ring_len)
    starts = ends - ring_len
    geom_rings = np.add.reduceat(
        np.asarray(rings_per_poly), np.cumsum([0] + [n for _, _, n in polygonal])[:-1]
    )
    geom_starts = starts[np.cumsum(geom_rings) - geom_rings]
    cut = np.zeros(len(polygonal), dtype=bool)
    if src is not None:
        from rasterio.warp import transform

        xs, ys = transform(src, dst, xy[:, 0], xy[:, 1])
        xy[:, 0], xy[:, 1] = xs, ys
        if np.isfinite(xy[:, 0]).all():
            lon = xy[:, 0]
            span = np.maximum.reduceat(lon, geom_starts) - np.minimum.reduceat(lon, geom_starts)
            cut = span > 180.0  # crosses the antimeridian: let GDAL split it
    finite = np.isfinite(xy).all(axis=1)
    if not finite.all():
        bad_geom = np.searchsorted(geom_starts, np.flatnonzero(~finite), side="right") - 1
        if not cut[bad_geom].all():
            raise ValueError("geometry has non-finite coordinates (failed reprojection?)")

    local = xy[:, :2] - np.repeat(xy[starts, :2], ring_len, axis=0)
    x, y = local[:, 0], local[:, 1]
    area2 = _ring_sums(x[:-1] * y[1:] - x[1:] * y[:-1], ends, starts)
    is_exterior = np.zeros(ring_len.size, dtype=bool)
    is_exterior[np.cumsum(rings_per_poly) - rings_per_poly] = True
    reverse = (area2 > 0) != is_exterior
    if precision is not None:
        xy = np.round(xy, precision)
    values = xy.tolist()

    r = p = 0
    for k, (i, kind, n_polys) in enumerate(polygonal):
        polys_out = []
        for _ in range(n_polys):
            poly_out = []
            for _ in range(rings_per_poly[p]):
                ring = values[starts[r] : ends[r]]
                poly_out.append(ring[::-1] if reverse[r] else ring)
                r += 1
            polys_out.append(poly_out)
            p += 1
        if cut[k]:
            out[i] = _cut_antimeridian(polys_out, precision)
        elif kind == "Polygon":
            out[i] = {"type": kind, "coordinates": polys_out[0]}
        else:
            out[i] = {"type": kind, "coordinates": polys_out}
    return out


def _cut_antimeridian(polys: list[list[list[list[float]]]], precision: int | None) -> dict:
    """Split lon/lat polygons that cross the antimeridian (RFC 7946 section 3.1.9).

    Longitudes are first unwrapped to [0, 360) so each polygon is continuous. With
    shapely installed the polygons are cut at 180° into a MultiPolygon; without it the
    unwrapped polygon (longitudes above 180) is returned with a warning.
    """
    unwrapped: list[list[np.ndarray]] = []
    for poly in polys:
        rings: list[np.ndarray] = []
        for ring in poly:
            arr = np.array(ring, dtype=np.float64)
            arr[:, 0] %= 360.0
            rings.append(_orient(arr, not rings))
        unwrapped.append(rings)
    try:
        import shapely
    except ImportError:
        warnings.warn(
            "A polygon crosses the antimeridian; it is written with longitudes above 180° "
            "instead of being cut in two as RFC 7946 recommends. Install shapely to cut "
            f"it ({_INSTALL_HINT}).",
            UserWarning,
            stacklevel=4,
        )
        coords = [
            [np.round(r, precision) if precision is not None else r for r in p] for p in unwrapped
        ]
        return {"type": "MultiPolygon", "coordinates": [[r.tolist() for r in p] for p in coords]}

    pieces = []
    halves = (
        (shapely.box(0.0, -90.0, 180.0, 90.0), 0.0),
        (shapely.box(180.0, -90.0, 360.0, 90.0), -360.0),
    )
    for rings in unwrapped:
        poly = shapely.make_valid(shapely.Polygon(rings[0], rings[1:]))
        for box, shift in halves:
            part = shapely.intersection(poly, box)
            for geom in shapely.get_parts(part):
                if geom.geom_type != "Polygon" or geom.is_empty:
                    continue
                geom_rings = [np.asarray(geom.exterior.coords)]
                geom_rings += [np.asarray(h.coords) for h in geom.interiors]
                new = []
                for j, ring in enumerate(geom_rings):
                    ring = ring.copy()
                    ring[:, 0] += shift
                    ring = _orient(ring, j == 0)
                    if precision is not None:
                        ring = np.round(ring, precision)
                    new.append(ring.tolist())
                pieces.append(new)
    return {"type": "MultiPolygon", "coordinates": pieces}


def _crs_member(crs: Any) -> dict[str, Any]:
    epsg = crs.to_epsg()
    name = f"urn:ogc:def:crs:EPSG::{epsg}" if epsg else crs.to_wkt()
    return {"type": "name", "properties": {"name": name}}


def _atomic_target(path: str) -> tuple[str, str]:
    target = os.path.realpath(path)
    directory, name = os.path.split(target)
    _, ext = os.path.splitext(name)
    return target, os.path.join(directory, f".{name}.{secrets.token_hex(8)}.tmp{ext}")


def _discard(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def to_geojson(
    features: Any,
    path: str | os.PathLike[str] | None = None,
    *,
    crs: Any = None,
    to_wgs84: bool = True,
    precision: int | None = None,
) -> dict[str, Any]:
    """Build a GeoJSON ``FeatureCollection`` and optionally write it to a file.

    Parameters
    ----------
    features : list of dict or FeatureCollection dict
        GeoJSON-like features, e.g. from :func:`polygonize`. They are not modified.
    path : str or os.PathLike, optional
        Output file (``.geojson``). Written atomically as UTF-8: a temporary file in
        the same directory replaces ``path`` only when complete.
    crs : optional
        CRS of the input coordinates: a ``rasterio.crs.CRS``, EPSG code, string, WKT
        or a rasterio meta dict. Required when ``to_wgs84=True`` (the default).
    to_wgs84 : bool
        ``True`` (default) reprojects coordinates to WGS 84 longitude/latitude as RFC
        7946 requires. ``False`` keeps the input coordinates and records ``crs`` in the
        legacy ``"crs"`` member, which GDAL, QGIS and ArcGIS honour but RFC 7946 does
        not define.
    precision : int, optional
        Round coordinates to this many decimals (7 decimals of a degree is about 1 cm).

    Returns
    -------
    dict
        The ``FeatureCollection``, JSON-serializable (non-finite property values
        become ``null``). Polygon rings follow the right-hand rule (exterior
        counter-clockwise, holes clockwise). Properties are copied unchanged, so
        ``centroid_x``/``centroid_y`` from :func:`polygonize` stay in the source CRS.

    Examples
    --------
    >>> fc = to_geojson(features, "changes.geojson", crs=meta["crs"])  # doctest: +SKIP
    """
    feats = _feature_list(features)
    if precision is not None and (isinstance(precision, bool) or not isinstance(precision, int)):
        raise TypeError(f"precision must be an int or None, got {type(precision).__name__}")
    src = _as_crs(crs)
    out_crs = src
    reproject_from = dst = None
    if to_wgs84:
        if src is None:
            raise ValueError(
                "crs is required to reproject to WGS 84 (RFC 7946). Pass crs=meta['crs'], "
                "or to_wgs84=False to keep the native coordinates."
            )
        from rasterio.crs import CRS

        dst = CRS.from_epsg(4326)
        reproject_from = None if src == dst else src
        out_crs = None

    with _gc_paused():
        geoms = _geometries([f["geometry"] for f in feats], reproject_from, dst, precision)
        out_features = []
        for feat, geom in zip(feats, geoms):
            out: dict[str, Any] = {"type": "Feature"}
            if feat.get("id") is not None:
                out["id"] = _clean_value(feat["id"])
            out["geometry"] = geom
            out["properties"] = _clean_properties(feat.get("properties"))
            out_features.append(out)
    collection: dict[str, Any] = {"type": "FeatureCollection"}
    if out_crs is not None:
        collection["crs"] = _crs_member(out_crs)
    collection["features"] = out_features

    if path is not None:
        text = json.dumps(collection, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        target, tmp = _atomic_target(os.fspath(path))
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, target)
        except BaseException:
            _discard(tmp)
            raise
    return collection


# --------------------------------------------------------------------------- #
# OGR formats (GeoPackage, Shapefile, FlatGeobuf)
# --------------------------------------------------------------------------- #
def _resolve_driver(path: str, driver: str | None) -> str:
    if driver is None:
        ext = os.path.splitext(path)[1].lower()
        if ext not in _EXTENSIONS:
            raise ValueError(
                f"cannot infer the vector format from extension {ext!r}; use one of "
                f"{sorted(_EXTENSIONS)} or pass driver="
            )
        return _EXTENSIONS[ext]
    if not isinstance(driver, str):
        raise TypeError(f"driver must be a string, got {type(driver).__name__}")
    resolved = _DRIVER_ALIASES.get(driver.lower())
    if resolved is None:
        raise ValueError(
            f"unsupported driver {driver!r}; use 'GeoJSON', 'GPKG', 'ESRI Shapefile' or "
            "'FlatGeobuf'"
        )
    return resolved


def _engine(engine: str | None) -> str:
    if engine not in (None, "pyogrio", "fiona"):
        raise ValueError(f"engine must be None, 'pyogrio' or 'fiona', got {engine!r}")
    for name in (engine,) if engine else ("pyogrio", "fiona"):
        try:
            __import__(name)
        except ImportError:
            continue
        return name
    raise ImportError(
        f"Writing GeoPackage, Shapefile or FlatGeobuf requires "
        f"{engine or 'pyogrio or fiona'}. Install it with: {_INSTALL_HINT} "
        f"(or pip install {engine or 'pyogrio'}). GeoJSON needs no extra package."
    )


def _field_types(feats: list[dict[str, Any]]) -> dict[str, str]:
    """Infer ``int``/``float``/``str`` field types from the (cleaned) properties."""
    names: dict[str, None] = {}
    for f in feats:
        names.update(dict.fromkeys(f["properties"]))
    if not feats:
        names = dict.fromkeys(_DEFAULT_FIELDS)
    types = {}
    for name in names:
        seen = [f["properties"].get(name) for f in feats] if feats else [_DEFAULT_FIELDS[name]]
        seen = [v for v in seen if v is not None]
        if seen and all(isinstance(v, int) for v in seen):
            missing = feats and any(f["properties"].get(name) is None for f in feats)
            types[name] = "float" if missing else "int"
        elif seen and all(isinstance(v, (int, float)) for v in seen):
            types[name] = "float"
        else:
            types[name] = "str"
    return types


def _field_names(names: Sequence[str], driver: str) -> dict[str, str]:
    if driver != "ESRI Shapefile":
        return {n: n for n in names}
    out: dict[str, str] = {}
    used: set[str] = set()
    for n in names:
        short = _SHAPEFILE_NAMES.get(n, n)[:10]
        i = 1
        while short.lower() in used:
            suffix = f"_{i}"
            short = _SHAPEFILE_NAMES.get(n, n)[: 10 - len(suffix)] + suffix
            i += 1
        used.add(short.lower())
        out[n] = short
    return out


def _wkb(geom: Mapping[str, Any]) -> bytes:
    """Little-endian WKB for a Polygon / MultiPolygon GeoJSON geometry."""
    import struct

    def polygon(rings: Any) -> bytes:
        parts = [struct.pack("<BII", 1, 3, len(rings))]
        for ring in rings:
            arr = np.ascontiguousarray(np.asarray(ring, dtype="<f8")[:, :2])
            parts.append(struct.pack("<I", arr.shape[0]))
            parts.append(arr.tobytes())
        return b"".join(parts)

    if geom["type"] == "Polygon":
        return polygon(geom["coordinates"])
    polys = geom["coordinates"]
    return struct.pack("<BII", 1, 6, len(polys)) + b"".join(polygon(p) for p in polys)


def _write_ogr(
    feats: list[dict[str, Any]],
    path: str,
    driver: str,
    crs: Any,
    layer: str | None,
    engine: str,
) -> None:
    for i, f in enumerate(feats):
        if f["geometry"]["type"] not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"feature {i}: only Polygon/MultiPolygon geometries can be written, "
                f"got {f['geometry']['type']}"
            )
    multi = any(f["geometry"]["type"] == "MultiPolygon" for f in feats)
    geometry_type = "MultiPolygon" if multi else "Polygon"
    if multi:  # one geometry type per layer
        for f in feats:
            if f["geometry"]["type"] == "Polygon":
                f["geometry"] = {
                    "type": "MultiPolygon",
                    "coordinates": [f["geometry"]["coordinates"]],
                }
    types = _field_types(feats)
    names = _field_names(list(types), driver)
    wkt = None if crs is None else crs.to_wkt()

    if engine == "pyogrio":
        from pyogrio.raw import write

        with _gc_paused():
            geometry = np.array([_wkb(f["geometry"]) for f in feats], dtype=object)

        dtypes = {"int": np.int64, "float": np.float64, "str": object}
        field_data = []
        for name, kind in types.items():
            column = [f["properties"].get(name) for f in feats]
            if kind == "float":
                column = [np.nan if v is None else v for v in column]
            field_data.append(np.array(column, dtype=dtypes[kind]))
        write(
            path,
            geometry,
            field_data,
            [names[n] for n in types],
            layer=layer,
            driver=driver,
            geometry_type=geometry_type,
            crs=wkt,
            encoding="UTF-8" if driver == "ESRI Shapefile" else None,
        )
        return

    import fiona

    schema = {
        "geometry": geometry_type,
        "properties": {names[n]: kind for n, kind in types.items()},
    }
    records = []
    for f in feats:
        props = {names[n]: f["properties"].get(n) for n in types}
        records.append(
            fiona.Feature.from_dict(
                {"type": "Feature", "geometry": f["geometry"], "properties": props}
            )
        )
    options: dict[str, Any] = {"driver": driver, "schema": schema, "crs_wkt": wkt or ""}
    if layer is not None:
        options["layer"] = layer
    if driver == "ESRI Shapefile":
        options["encoding"] = "utf-8"
    with fiona.open(path, "w", **options) as dst:
        dst.writerecords(records)


def write_vector(
    features: Any,
    path: str | os.PathLike[str],
    *,
    crs: Any,
    driver: str | None = None,
    layer: str | None = None,
    to_wgs84: bool | None = None,
    engine: str | None = None,
) -> None:
    """Write features to GeoJSON, GeoPackage, Shapefile or FlatGeobuf.

    Parameters
    ----------
    features : list of dict or FeatureCollection dict
        Polygon features, e.g. from :func:`polygonize`. They are not modified.
    path : str or os.PathLike
        Output file. An existing file is replaced (the whole file, not one layer).
        The file is written under a temporary name in the same directory and then
        renamed, so a failed write leaves an existing file intact (for a Shapefile,
        each component file is renamed in turn and stale components are removed).
    crs : optional
        CRS of the feature coordinates (``CRS``, EPSG code, string, WKT or a rasterio
        meta dict). ``None`` writes no CRS (not allowed for WGS 84 GeoJSON).
    driver : str, optional
        ``"GeoJSON"``, ``"GPKG"``, ``"ESRI Shapefile"`` or ``"FlatGeobuf"``. Inferred
        from the extension by default (``.geojson``/``.json``, ``.gpkg``, ``.shp``,
        ``.fgb``).
    layer : str, optional
        Layer name (GeoPackage, FlatGeobuf). Defaults to the file name. Shapefile
        layers are always named after the file.
    to_wgs84 : bool, optional
        Reproject to WGS 84 lon/lat. Defaults to ``True`` for GeoJSON (RFC 7946) and
        ``False`` for the other formats, which store the CRS natively.
    engine : {"pyogrio", "fiona"}, optional
        Library for the non-GeoJSON formats. By default pyogrio is used if installed,
        otherwise fiona. Install one with ``pip install farq[vector]``.

    Notes
    -----
    Shapefile field names are limited to 10 characters: ``perimeter_m`` is written
    as ``perim_m``, ``pixel_count`` as ``pixels``, ``mean_magnitude`` as ``mean_mag``
    and ``max_magnitude`` as ``max_mag``; other long names are truncated. Fields are
    written as 64-bit integer, double or string. Boolean properties become integers.

    Raises
    ------
    ImportError
        A non-GeoJSON format was requested but neither pyogrio nor fiona is installed.
    """
    path = os.fspath(path)
    drv = _resolve_driver(path, driver)
    if to_wgs84 is None:
        to_wgs84 = drv == "GeoJSON"
    if drv == "GeoJSON":
        if layer is not None:
            raise ValueError("GeoJSON files have no layers; layer must be None")
        to_geojson(features, path, crs=crs, to_wgs84=to_wgs84)
        return
    eng = _engine(engine)
    src = _as_crs(crs)
    if to_wgs84 and src is None:
        raise ValueError("crs is required to reproject to WGS 84")
    collection = to_geojson(features, crs=src, to_wgs84=bool(to_wgs84))
    feats = collection["features"]
    out_crs = _as_crs(4326) if to_wgs84 else src
    target, tmp = _atomic_target(path)
    if drv == "ESRI Shapefile":
        _write_shapefile(feats, target, tmp, out_crs, eng)
        return
    if layer is None:
        layer = os.path.splitext(os.path.basename(target))[0]
    try:
        _write_ogr(feats, tmp, drv, out_crs, layer, eng)
        os.replace(tmp, target)
    except BaseException:
        _discard(tmp)
        raise


def _write_shapefile(
    feats: list[dict[str, Any]], target: str, tmp: str, crs: Any, eng: str
) -> None:
    """Write a Shapefile under a temporary name, then rename each component file.

    Component files of an existing Shapefile at ``target`` that the new one does not
    have (e.g. an outdated ``.prj`` or spatial index) are removed.
    """
    tmp_stem, target_stem = os.path.splitext(tmp)[0], os.path.splitext(target)[0]
    directory = os.path.dirname(tmp)
    prefix = os.path.basename(tmp_stem) + "."

    def components() -> list[str]:
        return [n for n in os.listdir(directory) if n.startswith(prefix)]

    try:
        _write_ogr(feats, tmp, "ESRI Shapefile", crs, None, eng)
    except BaseException:
        for name in components():
            _discard(os.path.join(directory, name))
        raise
    written = {name[len(prefix) - 1 :] for name in components()}  # ".shp", ".dbf", ...
    for ext in _SHAPEFILE_PARTS - {e.lower() for e in written}:
        for candidate in (target_stem + ext, target_stem + ext.upper()):
            _discard(candidate)
    for ext in sorted(written, key=lambda e: e.lower() == ".shp"):  # .shp last
        os.replace(tmp_stem + ext, target_stem + ext)


# --------------------------------------------------------------------------- #
# One call for change-detection output
# --------------------------------------------------------------------------- #
def changes_to_vector(
    change: Any,
    meta: Any,
    path: str | os.PathLike[str] | None = None,
    *,
    values: Any = None,
    labels: Mapping[Any, str] | None = None,
    nodata: float | None = None,
    valid: np.ndarray | None = None,
    min_area: float | None = None,
    connectivity: int = 4,
    simplify: float | None = None,
    preserve_topology: bool = False,
    driver: str | None = None,
    layer: str | None = None,
    to_wgs84: bool | None = None,
    engine: str | None = None,
) -> list[dict[str, Any]]:
    """Polygonize change-detection output and optionally write it to a file.

    Parameters
    ----------
    change : ChangeResult, numpy.ndarray
        One of:

        * a :class:`farq.ChangeResult` from :func:`farq.detect_changes`: polygons of
          its ``mask``, with ``mean_magnitude`` and ``max_magnitude`` per polygon;
        * a boolean change mask: polygons of the ``True`` regions;
        * a class map from :func:`farq.classify_change` (any other numeric array, e.g.
          read back with ``farq.read(..., masked=True)``). Defaults: ``labels`` =
          :data:`farq.CHANGE_LABELS`, ``nodata`` = :data:`farq.CHANGE_NODATA` and
          ``values`` = ``(GAINED, LOST)``.
    meta : dict, rasterio dataset or affine.Affine
        Georeferencing of the grid (see :func:`polygonize`); its ``crs`` is used for
        writing.
    path : str or os.PathLike, optional
        Output file; the format follows the extension (see :func:`write_vector`).
        Without ``path`` the features are only returned.
    values, labels, nodata, valid, min_area, connectivity, simplify, preserve_topology
        As in :func:`polygonize`.
    driver, layer, to_wgs84, engine
        As in :func:`write_vector`.

    Returns
    -------
    list of dict
        The features in the raster CRS (as from :func:`polygonize`).

    Examples
    --------
    >>> result = farq.detect_changes(before, after, min_size=5)  # doctest: +SKIP
    >>> farq.changes_to_vector(result, meta, "changes.gpkg", min_area=500)  # doctest: +SKIP
    """
    common: dict[str, Any] = {
        "connectivity": connectivity,
        "min_area": min_area,
        "simplify": simplify,
        "preserve_topology": preserve_topology,
    }
    if hasattr(change, "magnitude") and hasattr(change, "mask"):
        features = _change_result_features(change, meta, valid=valid, **common)
    else:
        data, _ = _as_2d(change, "change")
        if data.dtype != bool:
            from .change import CHANGE_LABELS, CHANGE_NODATA, GAINED, LOST

            labels = CHANGE_LABELS if labels is None else labels
            nodata = CHANGE_NODATA if nodata is None else nodata
            values = (GAINED, LOST) if values is None else values
        features = polygonize(
            change, meta, valid=valid, values=values, labels=labels, nodata=nodata, **common
        )
    if path is not None:
        write_vector(
            features,
            path,
            crs=_georeference(meta)[1],
            driver=driver,
            layer=layer,
            to_wgs84=to_wgs84,
            engine=engine,
        )
    return features


def _change_result_features(
    result: Any,
    meta: Any,
    *,
    valid: np.ndarray | None,
    connectivity: int,
    min_area: float | None,
    simplify: float | None,
    preserve_topology: bool,
) -> list[dict[str, Any]]:
    """Features of a ChangeResult mask with per-polygon magnitude statistics."""
    from scipy import ndimage

    mask, _ = _as_2d(np.asarray(result.mask), "result.mask")
    magnitude, _ = _as_2d(np.asarray(result.magnitude), "result.magnitude")
    if magnitude.shape != mask.shape:
        raise ValueError("result.mask and result.magnitude must have the same shape")
    _check_shape(mask, meta)
    transform, crs = _georeference(meta)
    include = (mask != 0) & np.isfinite(magnitude)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != mask.shape:
            raise ValueError(f"valid shape {valid.shape} does not match mask shape {mask.shape}")
        include &= valid
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    components, n = ndimage.label(include, structure=structure)
    components = components.astype(np.int32, copy=False)
    index = np.arange(1, n + 1)
    filled = np.where(include, magnitude, 0.0)
    if n:
        mean = ndimage.mean(filled, components, index)
        peak = ndimage.maximum(filled, components, index)
    else:
        mean = peak = np.zeros(0)

    def describe(code: float) -> dict[str, Any]:
        i = int(code) - 1
        return {
            "value": 1,
            "label": "changed",
            "mean_magnitude": float(mean[i]),
            "max_magnitude": float(peak[i]),
        }

    return _build_features(
        components,
        include,
        transform,
        crs,
        connectivity=connectivity,
        min_area=_positive(min_area, "min_area"),
        simplify=_positive(simplify, "simplify"),
        preserve_topology=preserve_topology,
        describe=describe,
    )
