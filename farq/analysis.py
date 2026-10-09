"""Water-body statistics, change analysis and shape metrics for binary water masks.

The functions in this module work on 2-D water masks (``True``/non-zero = water).
Floating-point masks may contain ``NaN`` to mark invalid pixels (clouds, nodata), and
masked entries of a :class:`numpy.ma.MaskedArray` are invalid too. Invalid pixels are
treated as *unknown*: they are never counted as water and are excluded from coverage
and change statistics.

Units
-----
``pixel_size`` is given in metres, either as one number (square pixels) or as a
``(width, height)`` pair. Areas are reported in square kilometres and lengths
(perimeters) in kilometres, unless a function documents otherwise.

Shape metrics
-------------
All per-object metrics are computed in a single vectorised pass over the
labelled image (``numpy.bincount``), so cost is linear in the number of pixels
and independent of the number of water bodies.

* ``perimeter`` is the exact length of the pixel-edge boundary ("crack length"):
  every pixel side that separates the object from a non-object pixel (including
  the outer image border and the boundary of interior holes). A 3x3 square of
  1 m pixels has perimeter 12 m.
* ``compactness`` is the isoperimetric quotient ``4*pi*area / perimeter**2``.
  With the pixel-edge perimeter a square scores ``pi/4`` (~0.785), the maximum
  for rasterised shapes; elongated or ragged shapes approach 0.
* ``elongation`` is ``sqrt(lambda1 / lambda2)`` of the second-moment
  (inertia) matrix of the object, treating each pixel as a filled rectangle.
  It equals ``length / width`` for an axis-aligned rectangle and is 1 for a
  square or disc.
* ``orientation`` is the angle of the major axis in degrees, in ``[-90, 90]``,
  measured counter-clockwise from the column (x/east) axis with rows
  increasing downwards (i.e. as the image is normally displayed).
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any, Union

import numpy as np
from scipy import ndimage

__all__ = [
    "calculate_shape_metrics",
    "get_water_bodies",
    "water_change",
    "water_stats",
]

PixelSize = Union[float, "tuple[float, float]"]

_M2_PER_KM2 = 1_000_000.0
_M_PER_KM = 1_000.0


# --------------------------------------------------------------------------- #
# Validation helpers (shared with farq.ml)
# --------------------------------------------------------------------------- #
def _parse_pixel_size(pixel_size: Any) -> tuple[float, float]:
    """Return ``(width, height)`` of a pixel from a scalar or a 2-sequence."""
    if isinstance(pixel_size, (bool, np.bool_)):
        raise TypeError("pixel_size must be a number or a (width, height) pair, not a bool")
    if isinstance(pixel_size, Real):
        dx = dy = float(pixel_size)
    elif isinstance(pixel_size, (Sequence, np.ndarray)) and not isinstance(pixel_size, str):
        if len(pixel_size) != 2:
            raise ValueError(
                f"pixel_size sequence must have exactly 2 elements (width, height), "
                f"got {len(pixel_size)}"
            )
        if not all(isinstance(p, Real) and not isinstance(p, (bool, np.bool_)) for p in pixel_size):
            raise TypeError("pixel_size elements must be numbers")
        dx, dy = float(pixel_size[0]), float(pixel_size[1])
    else:
        kind = type(pixel_size).__name__
        raise TypeError(f"pixel_size must be a number or a (width, height) pair, got {kind}")
    if not (np.isfinite(dx) and np.isfinite(dy)) or dx <= 0 or dy <= 0:
        raise ValueError(f"pixel_size must be positive and finite, got {pixel_size!r}")
    return dx, dy


def _as_mask(array: Any, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a 2-D array to a boolean water mask.

    Returns ``(mask, valid)``. ``valid`` is ``None`` when every pixel is valid,
    otherwise a boolean array that is ``False`` where the input was ``NaN``.
    Boolean inputs are returned without copying.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}")
    if isinstance(array, np.ma.MaskedArray):
        # Masked pixels are invalid (unknown), like NaN.
        masked = np.ma.getmaskarray(array)
        mask, valid = _as_mask(np.ma.getdata(array), name)
        if not masked.any():
            return mask, valid
        mask = mask & ~masked
        return mask, (~masked if valid is None else valid & ~masked)
    if array.dtype == bool:
        return array, None
    if np.issubdtype(array.dtype, np.floating):
        valid = ~np.isnan(array)
        mask = (array != 0) & valid
        return mask, (None if valid.all() else valid)
    if np.issubdtype(array.dtype, np.integer):
        return array != 0, None
    raise TypeError(f"{name} must have a boolean or numeric dtype, got {array.dtype}")


def _structure(connectivity: int) -> np.ndarray:
    if connectivity not in (1, 2):
        raise ValueError(
            f"connectivity must be 1 (4-neighbour) or 2 (8-neighbour), got {connectivity!r}"
        )
    return ndimage.generate_binary_structure(2, connectivity)


def _label(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    labeled, n = ndimage.label(mask, structure=_structure(connectivity), output=np.int32)
    return labeled, int(n)


def _object_metrics(labeled: np.ndarray, n: int, dx: float, dy: float) -> dict[str, np.ndarray]:
    """Vectorised per-object metrics for labels ``1..n`` of a 2-D label image.

    Returns arrays of length ``n`` (index ``i`` holds label ``i + 1``):
    ``pixel_count``, ``area`` (dx*dy units), ``perimeter`` (dx/dy units),
    ``compactness``, ``elongation`` and ``orientation`` (degrees).
    """
    size = n + 1
    rows, cols = np.nonzero(labeled)
    lab = labeled[rows, cols]
    counts = np.bincount(lab, minlength=size).astype(np.float64)
    safe = np.where(counts > 0, counts, 1.0)

    # Coordinates in metres: x east (columns), y north (rows increase downwards).
    x = cols * dx
    y = rows * -dy
    del rows, cols
    mx = np.bincount(lab, weights=x, minlength=size) / safe
    my = np.bincount(lab, weights=y, minlength=size) / safe
    x -= mx[lab]
    y -= my[lab]
    # Each pixel is a filled dx-by-dy rectangle: add its own variance (s^2 / 12).
    mu20 = np.bincount(lab, weights=x * x, minlength=size) / safe + dx * dx / 12.0
    mu02 = np.bincount(lab, weights=y * y, minlength=size) / safe + dy * dy / 12.0
    mu11 = np.bincount(lab, weights=x * y, minlength=size) / safe
    del x, y, lab

    root = np.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11**2)
    lam1 = (mu20 + mu02 + root) / 2.0
    lam2 = (mu20 + mu02 - root) / 2.0
    elongation = np.sqrt(lam1 / np.maximum(lam2, np.finfo(float).tiny))
    orientation = 0.5 * np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02))

    # Pixel-edge perimeter: every pixel has 2 vertical sides (length dy) and
    # 2 horizontal sides (length dx); sides shared by two pixels of the same
    # object are interior and are removed.
    left, right = labeled[:, :-1], labeled[:, 1:]
    same = (left == right) & (right != 0)
    h_pairs = np.bincount(right[same], minlength=size)
    up, down = labeled[:-1, :], labeled[1:, :]
    same = (up == down) & (down != 0)
    v_pairs = np.bincount(down[same], minlength=size)
    perimeter = 2.0 * counts * (dx + dy) - 2.0 * h_pairs * dy - 2.0 * v_pairs * dx

    area = counts * (dx * dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        compactness = np.where(perimeter > 0, 4.0 * np.pi * area / perimeter**2, 0.0)

    return {
        "pixel_count": counts[1:].astype(np.int64),
        "area": area[1:],
        "perimeter": perimeter[1:],
        "compactness": compactness[1:],
        "elongation": elongation[1:],
        "orientation": orientation[1:],
    }


def _body_dicts(metrics: dict[str, np.ndarray], shapes: bool) -> list[dict[str, Any]]:
    """Convert per-object metric arrays (km units) to a list of plain dicts."""
    keys = ["area", "pixel_count"]
    if shapes:
        keys += ["perimeter", "compactness", "elongation", "orientation"]
    columns = {k: metrics[k].tolist() for k in keys}
    return [dict(zip(keys, values)) for values in zip(*(columns[k] for k in keys))]


def _metrics_km(labeled: np.ndarray, n: int, dx: float, dy: float) -> dict[str, np.ndarray]:
    m = _object_metrics(labeled, n, dx, dy)
    m["area"] = m["area"] / _M2_PER_KM2
    m["perimeter"] = m["perimeter"] / _M_PER_KM
    return m


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def calculate_shape_metrics(
    water_body_mask: np.ndarray, pixel_size: PixelSize = 1.0
) -> dict[str, float]:
    """Calculate shape metrics for a single water body.

    All non-zero pixels of ``water_body_mask`` are treated as one object (even if
    they are not connected). See the module docstring for metric definitions.

    Args:
        water_body_mask: 2-D binary mask of the water body.
        pixel_size: Pixel size as a number or ``(width, height)``. The default
            ``1.0`` reports lengths in pixels and areas in pixels squared.

    Returns:
        Dict with ``area`` (pixel_size units squared), ``perimeter``
        (pixel_size units), ``compactness``, ``elongation`` and ``orientation``
        (degrees). An empty mask yields zeros, elongation 1 and orientation 0.

    Raises:
        TypeError: If the mask is not a numeric/boolean numpy array.
        ValueError: If the mask is empty or not 2-D, or ``pixel_size`` is invalid.

    Example:
        >>> m = np.zeros((5, 5), bool); m[1:4, 1:4] = True
        >>> calculate_shape_metrics(m)["perimeter"]
        12.0
    """
    mask, _ = _as_mask(water_body_mask, "water_body_mask")
    dx, dy = _parse_pixel_size(pixel_size)
    if not mask.any():
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "compactness": 0.0,
            "elongation": 1.0,
            "orientation": 0.0,
        }
    m = _object_metrics(mask.view(np.uint8), 1, dx, dy)
    return {
        k: float(m[k][0]) for k in ("area", "perimeter", "compactness", "elongation", "orientation")
    }


def water_stats(
    water_mask: np.ndarray,
    pixel_size: PixelSize = 30.0,
    calculate_shapes: bool = False,
    *,
    connectivity: int = 1,
) -> dict[str, Any]:
    """Calculate water surface statistics from a water mask.

    Args:
        water_mask: 2-D water mask (``True``/non-zero = water). ``NaN`` pixels of
            a float mask are treated as invalid and excluded from the coverage
            denominator.
        pixel_size: Pixel size in metres, a number or ``(width, height)``.
            Default 30.0 (Landsat).
        calculate_shapes: Also compute per-body shape metrics.
        connectivity: 1 for 4-neighbour, 2 for 8-neighbour connected bodies.

    Returns:
        Dict with:

        - ``total_area``: water area (km²)
        - ``coverage_percent``: water pixels / valid pixels * 100
        - ``num_water_bodies``: number of connected water bodies
        - ``mean_body_size``: mean body area (km²)
        - ``largest_body``: largest body area (km²)
        - ``shape_metrics`` (only if ``calculate_shapes``): dict with
          ``mean_compactness``, ``mean_elongation`` and ``body_metrics``, a list
          (ordered by label) of per-body dicts with ``area`` (km²),
          ``pixel_count``, ``perimeter`` (km), ``compactness``, ``elongation``
          and ``orientation``.

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If ``water_mask`` is empty/not 2-D or ``pixel_size`` invalid.
    """
    mask, valid = _as_mask(water_mask, "water_mask")
    dx, dy = _parse_pixel_size(pixel_size)
    pixel_area = dx * dy / _M2_PER_KM2

    labeled, n = _label(mask, connectivity)
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)[1:]
    water_pixels = int(sizes.sum())
    n_valid = mask.size if valid is None else int(np.count_nonzero(valid))

    stats: dict[str, Any] = {
        "total_area": water_pixels * pixel_area,
        "coverage_percent": (100.0 * water_pixels / n_valid) if n_valid else 0.0,
        "num_water_bodies": n,
        "mean_body_size": float(sizes.mean() * pixel_area) if n else 0.0,
        "largest_body": float(sizes.max() * pixel_area) if n else 0.0,
    }

    if calculate_shapes and n > 0:
        metrics = _metrics_km(labeled, n, dx, dy)
        stats["shape_metrics"] = {
            "mean_compactness": float(metrics["compactness"].mean()),
            "mean_elongation": float(metrics["elongation"].mean()),
            "body_metrics": _body_dicts(metrics, shapes=True),
        }
    return stats


def _remove_small(mask: np.ndarray, min_pixels: float, connectivity: int) -> np.ndarray:
    """Drop connected components of ``mask`` with fewer than ``min_pixels`` pixels."""
    labeled, n = _label(mask, connectivity)
    if n == 0:
        return mask
    keep = np.bincount(labeled.ravel(), minlength=n + 1) >= min_pixels
    keep[0] = False
    return keep[labeled]


def water_change(
    mask1: np.ndarray,
    mask2: np.ndarray,
    pixel_size: PixelSize = 30.0,
    min_change_area: float | None = None,
    *,
    connectivity: int = 1,
) -> dict[str, Any]:
    """Analyse changes between two water masks of the same scene.

    Pixels that are ``NaN`` in either (float) mask are treated as unknown and
    never counted as gained, lost or stable water.

    Args:
        mask1: Earlier water mask (``True``/non-zero = water).
        mask2: Later water mask.
        pixel_size: Pixel size in metres, a number or ``(width, height)``.
        min_change_area: Minimum area in m² of a connected patch of gain or loss.
            Smaller patches are discarded as noise (they count as no change).
        connectivity: Pixel connectivity (1 or 2) used for ``min_change_area``.

    Returns:
        Dict with:

        - ``gained_area`` / ``lost_area`` / ``net_change``: km²
        - ``change_percent``: net change relative to the earlier water area
          (``inf`` if there was no water before but some was gained, else 0)
        - ``change_mask``: int8 array, 1 = gained, -1 = lost, 0 = no change
        - ``stable_water``: bool array, water in both masks

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If the masks differ in shape, are empty/not 2-D, or
            ``pixel_size``/``min_change_area`` are invalid.
    """
    m1, valid1 = _as_mask(mask1, "mask1")
    m2, valid2 = _as_mask(mask2, "mask2")
    if m1.shape != m2.shape:
        raise ValueError(f"Input masks must have the same shape, got {m1.shape} and {m2.shape}")
    dx, dy = _parse_pixel_size(pixel_size)
    pixel_area = dx * dy / _M2_PER_KM2

    gained = ~m1 & m2
    lost = m1 & ~m2
    stable = m1 & m2
    before = m1
    if valid1 is not None or valid2 is not None:
        valid = valid1 if valid2 is None else (valid2 if valid1 is None else valid1 & valid2)
        gained &= valid
        lost &= valid
        stable &= valid
        before = m1 & valid

    if min_change_area is not None:
        if not np.isfinite(min_change_area) or min_change_area < 0:
            raise ValueError(f"min_change_area must be >= 0, got {min_change_area!r}")
        min_pixels = min_change_area / (dx * dy)
        gained = _remove_small(gained, min_pixels, connectivity)
        lost = _remove_small(lost, min_pixels, connectivity)

    gained_px = int(np.count_nonzero(gained))
    lost_px = int(np.count_nonzero(lost))
    before_px = int(np.count_nonzero(before))
    gained_area = gained_px * pixel_area
    lost_area = lost_px * pixel_area
    net_change = gained_area - lost_area
    if before_px > 0:
        change_percent = 100.0 * (gained_px - lost_px) / before_px
    else:
        change_percent = float("inf") if gained_px > 0 else 0.0

    change_mask = gained.view(np.int8) - lost.view(np.int8)

    return {
        "gained_area": gained_area,
        "lost_area": lost_area,
        "net_change": net_change,
        "change_percent": change_percent,
        "change_mask": change_mask,
        "stable_water": stable,
    }


def get_water_bodies(
    water_mask: np.ndarray,
    pixel_size: PixelSize = 30.0,
    min_area: float | None = None,
    calculate_shapes: bool = False,
    *,
    connectivity: int = 1,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Label individual water bodies and calculate their characteristics.

    Args:
        water_mask: 2-D water mask (``True``/non-zero = water; ``NaN`` = no water).
        pixel_size: Pixel size in metres, a number or ``(width, height)``.
        min_area: Minimum body area in m². Smaller bodies are removed and the
            remaining bodies are relabelled ``1..n`` (in original label order).
        calculate_shapes: Also compute shape metrics for each body.
        connectivity: 1 for 4-neighbour, 2 for 8-neighbour connected bodies.

    Returns:
        ``(labeled, characteristics)``: an ``int32`` label image (0 = background)
        and a dict mapping each label to a dict with ``area`` (km²),
        ``pixel_count`` and, if ``calculate_shapes``, ``perimeter`` (km),
        ``compactness``, ``elongation`` and ``orientation`` (degrees).

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If ``water_mask`` is empty/not 2-D or ``pixel_size``/
            ``min_area`` are invalid.
    """
    mask, _ = _as_mask(water_mask, "water_mask")
    dx, dy = _parse_pixel_size(pixel_size)

    if min_area is not None and (not np.isfinite(min_area) or min_area < 0):
        raise ValueError(f"min_area must be >= 0, got {min_area!r}")

    labeled, n = _label(mask, connectivity)

    if min_area is not None and n > 0:
        counts = np.bincount(labeled.ravel(), minlength=n + 1)
        keep = counts >= min_area / (dx * dy)
        keep[0] = False
        label_map = np.zeros(n + 1, dtype=np.int32)
        label_map[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
        labeled = label_map[labeled]
        n = int(keep.sum())

    if n == 0:
        return labeled, {}

    if calculate_shapes:
        metrics = _metrics_km(labeled, n, dx, dy)
    else:
        counts = np.bincount(labeled.ravel(), minlength=n + 1)[1:]
        metrics = {"pixel_count": counts, "area": counts * (dx * dy / _M2_PER_KM2)}
    bodies = _body_dicts(metrics, shapes=calculate_shapes)
    return labeled, dict(zip(range(1, n + 1), bodies))
