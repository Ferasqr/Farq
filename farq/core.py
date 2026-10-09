"""
Core functionality for the Farq library.

This module provides fundamental operations for raster data processing:

- Raster file I/O (:func:`read`, :func:`write`)
- Raster resampling (:func:`resample`)
- Band validation (:func:`validate_bands`)

All functions validate their inputs and raise clear, specific errors.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import warnings
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Union

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from .utils import _check_output_size, validate_array

__all__ = ["read", "resample", "validate_bands", "write"]

PathLike = Union[str, "os.PathLike[str]"]

# Prefixes for paths that GDAL can read but that do not exist on the local filesystem.
_REMOTE_PREFIXES = ("/vsi", "http://", "https://", "s3://", "gs://", "az://", "ftp://")


def validate_bands(*bands: np.ndarray, reflectance_scale: float | None = None) -> list[np.ndarray]:
    """
    Validate band arrays and prepare them for spectral index calculations.

    Each band must be a non-empty numeric numpy array, and all bands must have the
    same shape. Bands are returned as floating point arrays so that index arithmetic
    cannot overflow or wrap around (e.g. ``uint16`` Landsat digital numbers). The common
    float dtype is ``float32`` when every input fits in it losslessly (e.g. ``uint8``,
    ``uint16``, ``int16``, ``float32``) and ``float64`` otherwise. Arrays that already
    have that dtype are returned without copying. Masked arrays are converted to plain
    arrays with masked pixels set to NaN.

    Args:
        *bands: One or more band arrays to validate.
        reflectance_scale: Optional scale factor to divide every band by (e.g. ``10000``
            for Landsat 8 Collection 1 surface reflectance). Must be a positive number.

    Returns:
        List of validated (and optionally scaled) floating point band arrays, in the same
        order as the input.

    Raises:
        TypeError: If any band is not a numpy array or has a non-numeric dtype, or if
            ``reflectance_scale`` is not a number.
        ValueError: If no bands are given, a band is empty, the band shapes differ, or
            ``reflectance_scale`` is not positive.
    """
    if not bands:
        raise ValueError("No bands provided")

    for i, band in enumerate(bands):
        if not isinstance(band, np.ndarray):
            raise TypeError(f"Band {i} must be a numpy array, got {type(band).__name__}")
        if band.size == 0:
            raise ValueError(f"Band {i} cannot be empty")
        if band.dtype.kind not in "biuf":
            raise TypeError(f"Band {i} must have a real numeric dtype, got {band.dtype}")

    shape = bands[0].shape
    for band in bands[1:]:
        if band.shape != shape:
            raise ValueError(f"Band shapes do not match: {shape} != {band.shape}")

    if reflectance_scale is not None:
        if isinstance(reflectance_scale, bool) or not isinstance(reflectance_scale, Real):
            raise TypeError(
                f"reflectance_scale must be a number, got {type(reflectance_scale).__name__}"
            )
        if not reflectance_scale > 0:
            raise ValueError(f"reflectance_scale must be positive, got {reflectance_scale}")

    dtype = np.result_type(*(b.dtype for b in bands), np.float32)

    validated = []
    for band in bands:
        if isinstance(band, np.ma.MaskedArray):
            out = np.ma.filled(band.astype(dtype), np.nan)
        else:
            out = np.asarray(band, dtype=dtype)
        if reflectance_scale is not None:
            scale = dtype.type(reflectance_scale)
            if out is band or np.shares_memory(out, band):
                out = out / scale
            else:  # ``out`` is a fresh copy we own: scale it in place
                out /= scale
        validated.append(out)
    return validated


def _check_target_shape(target_shape: Any) -> tuple[int, int]:
    if not isinstance(target_shape, (tuple, list)) or len(target_shape) != 2:
        raise TypeError("target_shape must be a tuple of (height, width)")
    if not all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in target_shape):
        raise ValueError("target_shape dimensions must be positive integers")
    height, width = (int(x) for x in target_shape)
    if height <= 0 or width <= 0:
        raise ValueError("target_shape dimensions must be positive integers")
    return height, width


def resample(
    array: np.ndarray,
    target_shape: tuple[int, int],
    method: Resampling | str = Resampling.bilinear,
) -> np.ndarray:
    """
    Resample an array to a target shape using GDAL resampling.

    Works on single-band ``(height, width)`` arrays and multi-band
    ``(bands, height, width)`` arrays (each band is resampled to ``target_shape``).
    NaN values in floating point arrays are treated as nodata, so they do not
    contaminate neighbouring pixels when averaging. The output has the same dtype as
    the input; ``bool`` and ``float16`` arrays, which GDAL cannot handle directly, are
    processed as ``uint8`` and ``float32`` respectively and converted back.

    Args:
        array: Input array of shape ``(height, width)`` or ``(bands, height, width)``.
        target_shape: Desired output shape as ``(height, width)``.
        method: Resampling method, either a :class:`rasterio.enums.Resampling` member or
            its name (e.g. ``"nearest"``, ``"bilinear"``, ``"average"``).

    Returns:
        Resampled array with shape ``target_shape`` (or ``(bands, *target_shape)`` for
        3D input).

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If the array is empty, has an unsupported number of dimensions, the
            target shape is invalid, or the method name is unknown.
    """
    validate_array(array)
    if array.ndim not in (2, 3):
        raise ValueError(f"array must be 2D or 3D, got {array.ndim}D")
    height, width = _check_target_shape(target_shape)

    method = _resampling_method(method)
    _check_output_size((*array.shape[:-2], height, width), "resampled array")

    if isinstance(array, np.ma.MaskedArray):
        array = np.ma.filled(array.astype(np.result_type(array.dtype, np.float32)), np.nan)

    original_dtype = array.dtype
    if original_dtype == np.bool_:
        work = array.view(np.uint8)
    elif original_dtype == np.float16:
        work = array.astype(np.float32)
    else:
        work = array

    is_3d = work.ndim == 3
    count = work.shape[0] if is_3d else 1
    src_h, src_w = work.shape[-2:]
    profile: dict[str, Any] = {
        "driver": "MEM",
        "height": src_h,
        "width": src_w,
        "count": count,
        "dtype": work.dtype,
    }
    if work.dtype.kind == "f":
        profile["nodata"] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile() as memfile, memfile.open(**profile) as dataset:
            if is_3d:
                dataset.write(work)
                out = dataset.read(out_shape=(count, height, width), resampling=method)
            else:
                dataset.write(work, 1)
                out = dataset.read(1, out_shape=(height, width), resampling=method)

    if original_dtype == np.bool_:
        return out.astype(bool)
    if out.dtype != original_dtype:
        return out.astype(original_dtype)
    return out


def _resampling_method(method: Resampling | str) -> Resampling:
    if isinstance(method, Resampling):
        return method
    if isinstance(method, str):
        try:
            return Resampling[method.lower()]
        except KeyError:
            valid = ", ".join(m.name for m in Resampling)
            raise ValueError(f"Unknown resampling method {method!r}; use one of: {valid}") from None
    raise TypeError("method must be a rasterio.enums.Resampling member or its name")


def _rescale_metadata(meta: dict[str, Any], src_h: int, src_w: int, h: int, w: int) -> None:
    """Update metadata in place for a raster resampled from (src_h, src_w) to (h, w)."""
    sy, sx = src_h / h, src_w / w
    meta["height"], meta["width"] = h, w
    transform = meta.get("transform")
    if transform is not None and not transform.is_identity:
        a, b, c, d, e, f = transform[:6]
        meta["transform"] = Affine(a * sx, b * sy, c, d * sx, e * sy, f)
    elif transform is not None and "gcps" not in meta:
        meta["transform"] = Affine.scale(sx, sy)
    if "gcps" in meta:
        meta["gcps"] = [
            GroundControlPoint(
                row=g.row / sy, col=g.col / sx, x=g.x, y=g.y, z=g.z, id=g.id, info=g.info
            )
            for g in meta["gcps"]
        ]


def _is_local_path(path: str) -> bool:
    return not path.startswith(_REMOTE_PREFIXES) and "://" not in path


def read(
    filepath: PathLike,
    band: int | Sequence[int] | None = 1,
    *,
    masked: bool = False,
    out_shape: tuple[int, int] | None = None,
    resampling: Resampling | str = Resampling.average,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read raster data from a file.

    Args:
        filepath: Path (``str`` or :class:`os.PathLike`) or GDAL-readable URL of the
            raster.
        band: 1-based index of the band to read (default ``1``, returns a 2D array). A
            sequence of indices, or ``None`` for all bands, returns a 3D array of shape
            ``(bands, height, width)``.
        masked: If True, pixels flagged as nodata by the dataset (nodata value, alpha
            band or internal mask) are set to NaN. Integer data is converted to a float
            dtype that can hold it (``float32`` for 8/16-bit data, ``float64`` otherwise).
        out_shape: Optional ``(height, width)`` to resample to while reading (decimated
            read, useful for previewing huge orthomosaics). The returned metadata's
            ``height``, ``width`` and ``transform`` (and GCP pixel positions) are updated
            to match.
        resampling: Resampling method used with ``out_shape``, as a
            :class:`rasterio.enums.Resampling` member or its name (default ``average``).

    Returns:
        Tuple ``(data, metadata)`` where ``data`` is the raster array and ``metadata`` is
        a copy of the dataset's ``meta`` dictionary (``driver``, ``dtype``, ``nodata``,
        ``width``, ``height``, ``count``, ``crs``, ``transform``). If the dataset is
        georeferenced by ground control points (e.g. raw drone imagery), ``metadata``
        also has ``gcps`` (list of :class:`rasterio.control.GroundControlPoint`) and
        ``gcps_crs``; :func:`write` preserves them.

    Raises:
        FileNotFoundError: If a local file does not exist.
        IndexError: If a requested band does not exist.
        ValueError: If the file cannot be opened as a raster.
        RuntimeError: If any other error occurs while reading.
    """
    path = os.fspath(filepath)
    if _is_local_path(path) and not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    target = _check_target_shape(out_shape) if out_shape is not None else None
    method = _resampling_method(resampling)

    try:
        with warnings.catch_warnings():
            # GCP-referenced rasters have no geotransform; the GCPs are returned instead.
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            src = rasterio.open(path)
        with src:
            if band is None:
                indexes: int | list[int] = list(src.indexes)
            elif isinstance(band, (bool, np.bool_)):
                raise TypeError("band must be an int, a sequence of ints or None, not a bool")
            elif isinstance(band, (int, np.integer)):
                indexes = int(band)
            else:
                indexes = [int(b) for b in band]
            requested = [indexes] if isinstance(indexes, int) else indexes
            if not requested:
                raise ValueError("band must not be an empty sequence")
            for b in requested:
                if not 1 <= b <= src.count:
                    raise IndexError(
                        f"Band {b} out of range; {path} has {src.count} band(s) (1-based)"
                    )

            kwargs: dict[str, Any] = {}
            if target is not None:
                n = len(requested)
                _check_output_size((n, *target), "out_shape")
                kwargs["out_shape"] = target if isinstance(indexes, int) else (n, *target)
                kwargs["resampling"] = method
            if masked:
                data = src.read(indexes, masked=True, **kwargs)
                dtype = np.result_type(data.dtype, np.float32)
                data = np.ma.filled(data.astype(dtype, copy=False), np.nan)
            else:
                data = src.read(indexes, **kwargs)
            metadata = src.meta.copy()
            gcps, gcps_crs = src.gcps
            if gcps:
                metadata["gcps"] = list(gcps)
                metadata["gcps_crs"] = gcps_crs
            if target is not None:
                _rescale_metadata(metadata, src.height, src.width, *target)
    except (FileNotFoundError, IndexError, ValueError, TypeError, MemoryError):
        raise
    except RasterioIOError as e:
        raise ValueError(f"Unable to read raster file {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading file {path}: {e}") from e
    return data, metadata


def _nodata_fits(nodata: float, dtype: np.dtype) -> bool:
    if dtype.kind == "f":
        return True
    if np.isnan(nodata) or np.isinf(nodata):
        return False
    info = np.iinfo(dtype)
    return info.min <= nodata <= info.max and float(nodata).is_integer()


_UNSET: Any = object()


def write(
    filepath: PathLike,
    data: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
    *,
    dtype: str | np.dtype | None = None,
    nodata: float | None = _UNSET,
    **profile: Any,
) -> None:
    """
    Write raster data to a file.

    GeoTIFFs (the default driver) are written atomically: data goes to a temporary file
    in the destination directory that replaces ``filepath`` only once it is complete,
    so a failure never leaves a truncated file or destroys an existing one.

    The metadata (typically from :func:`read`) is copied, never modified. Its
    ``height``, ``width``, ``count`` and ``dtype`` entries are updated to match
    ``data``, and ``driver`` defaults to ``"GTiff"``. Data types GDAL cannot store are
    converted: ``bool`` is written as ``uint8`` and ``float16`` as ``float32``.

    Args:
        filepath: Output path (``str`` or :class:`os.PathLike`).
        data: Array of shape ``(height, width)`` for a single band or
            ``(bands, height, width)`` for multiple bands. Masked arrays are filled with
            the nodata value (NaN for float data when no nodata is set).
        metadata: Raster metadata such as ``crs``, ``transform`` and ``nodata``. May be
            omitted to write an un-georeferenced raster. Ground control points in
            ``metadata["gcps"]`` / ``metadata["gcps_crs"]`` (as returned by :func:`read`)
            are written to the file, unless ``metadata["transform"]`` is a real
            (non-identity) geotransform, which then takes precedence.
        dtype: Optional output dtype; ``data`` is cast to it. Defaults to the dtype of
            ``data``.
        nodata: Optional nodata value overriding ``metadata["nodata"]``. Pass ``None``
            to write without a nodata value. When it is not given, ``metadata["nodata"]``
            is used, except that a float result written with the metadata of *integer*
            data (e.g. an index computed from ``uint8`` imagery whose nodata is ``0``)
            gets NaN as nodata, so valid zeros are not turned into nodata. NaN values are
            stored as the nodata value when it is finite; writing NaN/inf to an integer
            dtype requires a nodata value.
        **profile: Additional creation options passed to :func:`rasterio.open`, e.g.
            ``compress="deflate"``. They override entries in ``metadata``.

    Raises:
        TypeError: If ``data`` is not a numpy array or ``metadata`` is not a mapping.
        ValueError: If ``data`` is empty or not 2D/3D, the nodata value cannot be
            represented in the output dtype, or NaN/inf values would be written to an
            integer dtype without a nodata value.
        RuntimeError: If writing the file fails.
    """
    validate_array(data, name="data", allow_all_nan=True)
    if data.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")

    meta: dict[str, Any] = dict(metadata) if metadata is not None else {}
    meta.update(profile)
    explicit_nodata = nodata is not _UNSET or "nodata" in profile
    if nodata is not _UNSET:
        meta["nodata"] = nodata
    meta.setdefault("driver", "GTiff")
    gcps = meta.pop("gcps", None) or None
    gcps_crs = meta.pop("gcps_crs", None)
    if gcps is not None:
        transform = meta.get("transform")
        if transform is None or transform.is_identity:
            # GCP-referenced raster: the identity transform and empty CRS from read()
            # are placeholders and must not be written alongside the GCPs.
            meta.pop("transform", None)
            if meta.get("crs") is None:
                meta.pop("crs", None)
        else:
            # A real geotransform (e.g. after rectification) supersedes stale GCPs.
            gcps = None

    out_dtype = np.dtype(dtype) if dtype is not None else data.dtype
    if out_dtype == np.bool_:
        out_dtype = np.dtype(np.uint8)
    elif out_dtype == np.float16:
        out_dtype = np.dtype(np.float32)
    if out_dtype.kind not in "uif":
        raise TypeError(f"Cannot write dtype {out_dtype}; use an integer or float dtype")

    fill_value = meta.get("nodata")
    if not explicit_nodata and out_dtype.kind == "f" and _is_finite_number(fill_value):
        src_dtype = metadata.get("dtype") if metadata is not None else None
        if src_dtype is not None and np.dtype(src_dtype).kind in "biu":
            # A nodata value inherited from integer source data (e.g. 0 for uint8
            # imagery read with masked=True) is meaningless for a derived float
            # product such as an index, where 0 is a legitimate value: writing it
            # would silently turn valid pixels into nodata. NaN marks invalid pixels.
            fill_value = meta["nodata"] = np.nan

    mask = None
    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)
        if not mask.any():
            mask = None
        if mask is not None and fill_value is None:
            if out_dtype.kind != "f":
                raise ValueError(
                    "Writing a masked integer array requires a nodata value "
                    "(set metadata['nodata'] or pass nodata=...)"
                )
            fill_value = meta["nodata"] = np.nan
        data = np.ma.getdata(data)

    if fill_value is not None and not _nodata_fits(float(fill_value), out_dtype):
        raise ValueError(
            f"nodata value {fill_value!r} cannot be represented in dtype {out_dtype}; "
            "pass nodata=... or dtype=... to write()"
        )

    arr = _cast_for_write(data, out_dtype, fill_value)
    if mask is not None:
        arr = arr.copy() if np.shares_memory(arr, data) else arr
        arr[mask] = fill_value
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    meta.update(count=arr.shape[0], height=arr.shape[1], width=arr.shape[2], dtype=out_dtype.name)

    path = os.fspath(filepath)
    target, tmp = _temporary_path(path, str(meta["driver"]))
    try:
        with warnings.catch_warnings():
            if meta.get("crs") is None and meta.get("transform") is None:
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(tmp or target, "w", **meta) as dst:
                if gcps is not None:
                    dst.gcps = (gcps, gcps_crs)
                dst.write(arr)
        if tmp is not None:
            _commit(tmp, target)
    except MemoryError:
        _discard(tmp)
        raise
    except Exception as e:
        _discard(tmp)
        raise RuntimeError(f"Error writing file {path}: {e}") from e


def _is_finite_number(value: Any) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _cast_for_write(data: np.ndarray, out_dtype: np.dtype, fill_value: Any) -> np.ndarray:
    """Cast ``data`` to ``out_dtype``, storing NaN (and inf for integers) as nodata."""
    if data.dtype.kind not in "fc":
        return data.astype(out_dtype, copy=False)
    if out_dtype.kind == "f":
        if fill_value is None or np.isnan(float(fill_value)):
            return data.astype(out_dtype, copy=False)
        bad = np.isnan(data)
    else:
        bad = ~np.isfinite(data)
    if not bad.any():
        return data.astype(out_dtype, copy=False)
    if fill_value is None:
        raise ValueError(
            f"data contains NaN/inf values that cannot be stored in dtype {out_dtype}; "
            "pass nodata=... (or a float dtype) to write()"
        )
    if out_dtype.kind == "f":
        arr = data.astype(out_dtype)  # always a copy: it is modified below
    else:
        # Avoid the undefined float -> int cast (and its RuntimeWarning) for NaN/inf.
        arr = np.where(bad, 0, data).astype(out_dtype)
    arr[bad] = fill_value
    return arr


# Drivers that write a single file, which can safely be written to a temporary name in
# the destination directory and then renamed over the destination atomically.
_ATOMIC_DRIVERS = frozenset({"GTIFF", "COG"})


def _temporary_path(path: str, driver: str) -> tuple[str, str | None]:
    """Return ``(final path, temporary path or None)`` for an atomic write."""
    if driver.upper() not in _ATOMIC_DRIVERS or not _is_local_path(path):
        return path, None
    # Write through a symlink to its target instead of replacing the link itself.
    target = os.path.realpath(path)
    directory, name = os.path.split(target)
    _, ext = os.path.splitext(name)
    # Unpredictable name in the destination directory (same filesystem, so the final
    # os.replace is atomic). GDAL creates the file, honouring the process umask.
    tmp = os.path.join(directory, f".{name}.{secrets.token_hex(8)}.tmp{ext}")
    return target, tmp


def _sidecars(path: str) -> list[str]:
    """Auxiliary files GDAL may create next to ``path`` (e.g. ``.aux.xml``, ``.msk``)."""
    return [path + suffix for suffix in (".aux.xml", ".msk", ".ovr")]


def _commit(tmp: str, target: str) -> None:
    for side, final_side in zip(_sidecars(tmp), _sidecars(target)):
        if os.path.exists(side):
            os.replace(side, final_side)
        elif os.path.exists(final_side):
            # A stale sidecar of the overwritten file would describe the wrong data.
            os.remove(final_side)
    os.replace(tmp, target)


def _discard(tmp: str | None) -> None:
    if tmp is None:
        return
    for leftover in (tmp, *_sidecars(tmp)):
        with contextlib.suppress(FileNotFoundError):
            os.remove(leftover)
