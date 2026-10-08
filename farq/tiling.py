"""
Out-of-core (tiled) processing of rasters larger than memory.

Every function in this module reads its inputs block by block from disk and
writes its result block by block to a tiled, compressed GeoTIFF, so memory use
is bounded by the block size, not by the raster size. A 50,000 x 50,000 drone
orthomosaic can be processed on a laptop.

- :func:`iter_windows` splits a raster into blocks, optionally with an overlap
  (halo) for neighbourhood operations.
- :func:`map_blocks` applies any NumPy function to matching blocks of one or
  more co-registered files and writes the result.
- :func:`index_file` computes any farq spectral/RGB index (NDVI, NDWI, VARI, ...)
  file to file.
- :func:`detect_changes_file` is the out-of-core :func:`farq.change.detect_changes`:
  a global threshold estimated from a reproducible pixel sample, and small-region
  removal / hole filling that is exact across block borders.
- :func:`summarize_file` computes min, max, mean, std and a histogram with
  streaming (mergeable) statistics.

Conventions
-----------
* Inputs are read with ``masked=True`` by default: nodata, alpha bands and
  internal masks become NaN, and integer data is converted to ``float32``
  (``float64`` for 32/64-bit integers), exactly as :func:`farq.read` does.
* Arrays are shaped ``(rows, cols)`` for one band and ``(bands, rows, cols)``
  for several.
* All inputs of one call must share the same grid (CRS, transform and size).
  Use :func:`farq.align_pair` / :func:`farq.align` (in memory) or ``gdalwarp``
  (for files larger than memory) to put them on one grid first.
* Outputs are written atomically: data goes to a temporary file in the
  destination directory that replaces the destination only when complete, so a
  failure never leaves a partial file or destroys an existing one.
* Results do not depend on ``block_size`` (except where documented) or on
  ``n_jobs``: blocks are always written in the same order.
"""

from __future__ import annotations

import math
import os
import queue
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, NamedTuple, TypeVar, Union

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import change as _change
from .core import (
    _cast_for_write,
    _commit,
    _discard,
    _is_local_path,
    _nodata_fits,
    _temporary_path,
)
from .utils import _check_output_size

__all__ = [
    "Block",
    "detect_changes_file",
    "index_file",
    "iter_windows",
    "map_blocks",
    "summarize_file",
]

PathLike = Union[str, "os.PathLike[str]"]
BandSpec = Union[int, Sequence[int], None]
ProgressCallback = Callable[[int, int], Any]

_T = TypeVar("_T")
_UNSET: Any = object()

#: Creation options for output GeoTIFFs; ``**profile`` arguments override them.
_DEFAULT_PROFILE: dict[str, Any] = {
    "driver": "GTiff",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "compress": "deflate",
    "BIGTIFF": "IF_SAFER",
}

_TILED_METHODS = ("difference", "ratio", "normalized_difference", "cva")


class Block(NamedTuple):
    """One block of a raster, as yielded by :func:`iter_windows`.

    Attributes
    ----------
    number : int
        Position of the block in row-major order (0, 1, 2, ...). (Not called
        ``index``, which would shadow :meth:`tuple.index`.)
    read_window : rasterio.windows.Window
        Window to read: the block plus up to ``overlap`` pixels on each side,
        clipped to the raster.
    write_window : rasterio.windows.Window
        Window this block is responsible for. The write windows of all blocks
        tile the raster exactly, without gaps or overlap.
    inner : tuple of slice
        ``(row_slice, col_slice)`` selecting the ``write_window`` part of an
        array read with ``read_window``: ``array[..., inner[0], inner[1]]``.
    """

    number: int
    read_window: Window
    write_window: Window
    inner: tuple[slice, slice]


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _block_shape(block_size: Any) -> tuple[int, int]:
    if isinstance(block_size, (tuple, list)):
        if len(block_size) != 2:
            raise ValueError(f"block_size must be an int or a (rows, cols) pair, got {block_size}")
        return _positive_int(block_size[0], "block_size"), _positive_int(
            block_size[1], "block_size"
        )
    size = _positive_int(block_size, "block_size")
    return size, size


def _n_jobs(n_jobs: Any) -> int:
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)):
        raise TypeError(f"n_jobs must be an integer, got {type(n_jobs).__name__}")
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer or -1 (all CPUs), got {n_jobs}")
    return int(n_jobs)


def _align(size: int, unit: int, extent: int) -> int:
    """Round ``size`` down to a multiple of ``unit`` (when that is meaningful)."""
    if unit <= 1 or unit >= extent or size < unit:
        return size
    return (size // unit) * unit


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def iter_windows(
    width: int,
    height: int,
    block_size: int | tuple[int, int] = 1024,
    overlap: int = 0,
    *,
    align_to: tuple[int, int] | None = None,
) -> Iterator[Block]:
    """Split a ``height x width`` raster into blocks, in row-major order.

    Parameters
    ----------
    width, height : int
        Raster size in pixels.
    block_size : int or (int, int), default 1024
        Block size in pixels (``(rows, cols)`` for non-square blocks). Blocks at
        the right and bottom edges are smaller when the size is not a multiple.
    overlap : int, default 0
        Halo in pixels added on each side of ``read_window`` (clipped to the
        raster), for neighbourhood operations such as filters.
    align_to : (int, int), optional
        Internal block shape ``(rows, cols)`` of the file, e.g.
        ``dataset.block_shapes[0]``. ``block_size`` is rounded *down* to a
        multiple of it (where it is at least one internal block), so that every
        file block is decoded once. Strip-organised files (blocks as wide as the
        raster) are only aligned along rows.

    Yields
    ------
    Block
        ``(number, read_window, write_window, inner)``; see :class:`Block`.

    Examples
    --------
    >>> from farq.tiling import iter_windows
    >>> blocks = list(iter_windows(5, 3, block_size=2, overlap=1))
    >>> len(blocks)
    6
    >>> blocks[4].read_window, blocks[4].inner
    (Window(col_off=1, row_off=1, width=4, height=2), (slice(1, 2, None), slice(1, 3, None)))
    """
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    rows, cols = _block_shape(block_size)
    overlap = _non_negative_int(overlap, "overlap")
    if align_to is not None:
        a_rows, a_cols = (int(v) for v in align_to)
        rows = _align(rows, a_rows, height)
        cols = _align(cols, a_cols, width)

    index = 0
    for r0 in range(0, height, rows):
        r1 = min(r0 + rows, height)
        rr0, rr1 = max(0, r0 - overlap), min(height, r1 + overlap)
        for c0 in range(0, width, cols):
            c1 = min(c0 + cols, width)
            cc0, cc1 = max(0, c0 - overlap), min(width, c1 + overlap)
            yield Block(
                index,
                Window(cc0, rr0, cc1 - cc0, rr1 - rr0),
                Window(c0, r0, c1 - c0, r1 - r0),
                (slice(r0 - rr0, r1 - rr0), slice(c0 - cc0, c1 - cc0)),
            )
            index += 1


def _grid_shape(width: int, height: int, blocks: Sequence[Block]) -> tuple[int, int]:
    """Number of block rows and columns of a block list from :func:`iter_windows`."""
    n_cols = sum(1 for b in blocks if b.write_window.row_off == 0)
    return len(blocks) // n_cols, n_cols


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _open(path: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        return rasterio.open(path)


class _Input(NamedTuple):
    path: str
    indexes: int | list[int]


def _parse_inputs(inputs: Any, bands: BandSpec) -> list[_Input]:
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    if not isinstance(inputs, Sequence) or not inputs:
        raise ValueError("inputs must be a non-empty sequence of raster paths")
    parsed = []
    for item in inputs:
        spec = bands
        if isinstance(item, tuple) and len(item) == 2 and not isinstance(item[1], str):
            item, spec = item
        if not isinstance(item, (str, os.PathLike)):
            raise TypeError(
                f"inputs must contain paths or (path, bands) pairs, got {type(item).__name__}"
            )
        parsed.append(_Input(os.fspath(item), _indexes(spec)))
    return parsed


def _indexes(spec: Any) -> int | list[int]:
    if isinstance(spec, (bool, np.bool_)):
        raise TypeError("bands must be an int, a sequence of ints or None, not a bool")
    if spec is None:
        return []  # all bands, resolved when the file is opened
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    indexes = [int(b) for b in spec]
    if not indexes:
        raise ValueError("bands must not be an empty sequence")
    return indexes


class _Grid(NamedTuple):
    width: int
    height: int
    crs: Any
    transform: Any
    gcps: list[Any] | None
    gcps_crs: Any
    block_shape: tuple[int, int]


def _check_inputs(specs: list[_Input]) -> tuple[_Grid, list[_Input]]:
    """Open every input once: check files, bands and that all share one grid."""
    grid: _Grid | None = None
    resolved = []
    for spec in specs:
        if _is_local_path(spec.path) and not os.path.exists(spec.path):
            raise FileNotFoundError(f"File not found: {spec.path}")
        with _open(spec.path) as src:
            indexes = spec.indexes
            if indexes == []:
                indexes = list(src.indexes)
            for b in [indexes] if isinstance(indexes, int) else indexes:
                if not 1 <= b <= src.count:
                    raise IndexError(
                        f"Band {b} out of range; {spec.path} has {src.count} band(s) (1-based)"
                    )
            gcps, gcps_crs = src.gcps
            this = _Grid(
                src.width,
                src.height,
                src.crs,
                src.transform,
                list(gcps) or None,
                gcps_crs,
                tuple(src.block_shapes[0]),
            )
        resolved.append(_Input(spec.path, indexes))
        if grid is None:
            grid = this
            first = spec.path
            continue
        problem = None
        if (this.height, this.width) != (grid.height, grid.width):
            problem = (
                f"size {this.height}x{this.width} (rows x cols) differs from "
                f"{grid.height}x{grid.width}"
            )
        elif this.crs != grid.crs:
            problem = f"CRS {this.crs} differs from {grid.crs}"
        elif not _same_transform(this.transform, grid.transform):
            problem = (
                f"transform {tuple(this.transform)[:6]} differs from {tuple(grid.transform)[:6]}"
            )
        if problem is not None:
            raise ValueError(
                f"{spec.path} is not on the same pixel grid as {first}: {problem}. "
                "All inputs must share CRS, transform and size. Put them on one grid first, "
                "e.g. with farq.align_pair / farq.align (in memory) and farq.write, or with "
                "gdalwarp for rasters larger than memory."
            )
    assert grid is not None
    return grid, resolved


def _same_transform(t1: Any, t2: Any) -> bool:
    a = np.array(tuple(t1)[:6], dtype=np.float64)
    b = np.array(tuple(t2)[:6], dtype=np.float64)
    pixel = max(abs(a[0]), abs(a[1]), abs(a[3]), abs(a[4]), 1e-300)
    return bool(np.all(np.abs(a - b) <= 1e-6 * pixel))


class _Reader:
    """Reads matching windows of several inputs from a pool of dataset handles.

    All handles are opened in the calling thread: :func:`_open` uses
    :class:`warnings.catch_warnings`, which swaps the process-wide warning filters
    and is not thread-safe, so entering it from worker threads could leave a
    filter installed (or drop one of the caller's) after the call returns. Each
    :meth:`read` checks out one set of handles (one per input file) for its
    duration, so a handle is never used by two threads at once. With
    ``n_handles`` equal to the number of worker threads a read never waits.
    """

    def __init__(self, specs: list[_Input], masked: bool, n_handles: int = 1) -> None:
        self.specs = specs
        self.masked = masked
        self._pool: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._opened: list[Any] = []
        try:
            for _ in range(max(1, n_handles)):
                datasets: dict[str, Any] = {}
                for spec in specs:
                    if spec.path not in datasets:
                        datasets[spec.path] = _open(spec.path)
                        self._opened.append(datasets[spec.path])
                self._pool.put(datasets)
        except BaseException:
            self.close()
            raise

    def read(self, window: Window) -> list[np.ndarray]:
        datasets = self._pool.get()
        try:
            out = []
            for spec in self.specs:
                src = datasets[spec.path]
                if self.masked:
                    data = src.read(spec.indexes, window=window, masked=True)
                    dtype = np.result_type(data.dtype, np.float32)
                    out.append(np.ma.filled(data.astype(dtype, copy=False), np.nan))
                else:
                    out.append(src.read(spec.indexes, window=window))
            return out
        finally:
            self._pool.put(datasets)

    def close(self) -> None:
        for src in self._opened:
            src.close()
        self._opened.clear()


def _check_block_memory(specs: list[_Input], blocks: Sequence[Block]) -> None:
    rows = max(b.read_window.height for b in blocks)
    cols = max(b.read_window.width for b in blocks)
    n_bands = sum(1 if isinstance(s.indexes, int) else len(s.indexes) for s in specs)
    _check_output_size((n_bands, rows, cols), "block (bands x rows x cols)")


# --------------------------------------------------------------------------- #
# Block engine
# --------------------------------------------------------------------------- #
class _Progress:
    def __init__(self, callback: ProgressCallback | None, total: int) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("progress must be a callable progress(done, total) or None")
        self.callback = callback
        self.total = total
        self.done = 0

    def tick(self) -> None:
        self.done += 1
        if self.callback is not None:
            self.callback(self.done, self.total)


def _run(
    blocks: Sequence[Block],
    work: Callable[[Block], _T],
    consume: Callable[[Block, _T], None],
    n_jobs: int,
    progress: _Progress,
) -> None:
    """Run ``work`` on every block (in threads) and ``consume`` results in block order.

    At most ``2 * n_jobs`` blocks are in flight, which bounds memory use.
    """
    if n_jobs == 1 or len(blocks) == 1:
        for block in blocks:
            consume(block, work(block))
            progress.tick()
        return
    pending: deque[tuple[Block, Future[_T]]] = deque()
    todo = iter(blocks)
    with ThreadPoolExecutor(max_workers=n_jobs, thread_name_prefix="farq-tiling") as pool:
        try:
            for block in todo:
                pending.append((block, pool.submit(work, block)))
                if len(pending) >= 2 * n_jobs:
                    break
            while pending:
                block, future = pending.popleft()
                result = future.result()
                following = next(todo, None)
                if following is not None:
                    pending.append((following, pool.submit(work, following)))
                consume(block, result)
                progress.tick()
        except BaseException:
            for _, future in pending:
                future.cancel()
            raise


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
class _Writer:
    """Atomic block-wise GeoTIFF writer on the grid of the inputs."""

    def __init__(
        self,
        path: PathLike,
        grid: _Grid,
        count: int,
        dtype: np.dtype,
        nodata: float | None,
        profile: Mapping[str, Any],
    ) -> None:
        self.dtype = dtype
        self.nodata = nodata
        self.count = count
        meta: dict[str, Any] = {**_DEFAULT_PROFILE, **profile}
        if str(meta.get("driver", "GTiff")).upper() != "GTIFF":
            raise ValueError(
                f"Block-wise output must be a GeoTIFF (driver='GTiff'), got {meta['driver']!r}"
            )
        meta.update(
            width=grid.width,
            height=grid.height,
            count=count,
            dtype=dtype.name,
            nodata=nodata,
        )
        gcps = None
        if grid.gcps and grid.transform.is_identity:
            gcps = (grid.gcps, grid.gcps_crs)
        else:
            meta["crs"] = grid.crs
            meta["transform"] = grid.transform
        self.meta: dict[str, Any] = {
            k: meta[k]
            for k in ("driver", "dtype", "nodata", "width", "height", "count")
            if k in meta
        }
        self.meta["crs"] = grid.crs
        self.meta["transform"] = grid.transform
        if gcps is not None:
            self.meta["gcps"], self.meta["gcps_crs"] = list(gcps[0]), gcps[1]
        self.path = os.fspath(path)
        self.target, self.tmp = _temporary_path(self.path, "GTiff")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                self.dst = rasterio.open(self.tmp or self.target, "w", **meta)
        except BaseException:
            _discard(self.tmp)
            raise
        if gcps is not None:
            self.dst.gcps = gcps

    def write(self, window: Window, data: np.ndarray) -> None:
        arr = _cast_for_write(data, self.dtype, self.nodata)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        self.dst.write(arr, window=window)

    def commit(self) -> None:
        self.dst.close()
        if self.tmp is not None:
            _commit(self.tmp, self.target)

    def discard(self) -> None:
        self.dst.close()
        _discard(self.tmp)


def _with_threads(profile: Mapping[str, Any], n_jobs: int) -> dict[str, Any]:
    """Let GDAL compress output tiles with ``n_jobs`` threads unless told otherwise."""
    out = dict(profile)
    if n_jobs > 1 and not any(str(k).upper() == "NUM_THREADS" for k in out):
        out["NUM_THREADS"] = n_jobs
    return out


def _out_dtype(dtype: Any, default: np.dtype) -> np.dtype:
    out = np.dtype(dtype) if dtype is not None else np.dtype(default)
    if out == np.bool_:
        return np.dtype(np.uint8)
    if out == np.float16:
        return np.dtype(np.float32)
    if out.kind not in "uif":
        raise TypeError(f"Cannot write dtype {out}; use an integer or float dtype")
    return out


def _out_nodata(nodata: Any, dtype: np.dtype) -> float | None:
    if nodata is _UNSET:
        return float("nan") if dtype.kind == "f" else None
    if nodata is not None and not _nodata_fits(float(nodata), dtype):
        raise ValueError(f"nodata value {nodata!r} cannot be represented in dtype {dtype}")
    return nodata


# --------------------------------------------------------------------------- #
# map_blocks
# --------------------------------------------------------------------------- #
def map_blocks(
    func: Callable[..., np.ndarray],
    inputs: Sequence[PathLike | tuple[PathLike, BandSpec]] | PathLike,
    out_path: PathLike,
    *,
    bands: BandSpec = 1,
    block_size: int | tuple[int, int] = 1024,
    overlap: int = 0,
    dtype: str | np.dtype | None = None,
    nodata: float | None = _UNSET,
    masked: bool = True,
    n_jobs: int = 1,
    progress: ProgressCallback | None = None,
    **profile: Any,
) -> dict[str, Any]:
    """Apply a NumPy function block by block to co-registered rasters on disk.

    For every block, the matching window of each input is read and passed to
    ``func(*arrays)``; the non-overlapping part of its result is written to
    ``out_path``. Memory use is proportional to the block size, not to the
    raster size.

    Parameters
    ----------
    func : callable
        ``func(*arrays) -> numpy.ndarray`` with one array per input. It must
        return an array of shape ``(rows, cols)`` or ``(bands, rows, cols)``
        with the spatial shape of its inputs (which include the halo). Results
        may differ between blocks only through their data: the number of
        output bands must be the same for every block. ``bool`` results are
        written as ``uint8``.
    inputs : sequence of path or (path, bands)
        Raster files on one common grid (same CRS, transform and size). A pair
        ``(path, bands)`` overrides ``bands`` for that input, e.g.
        ``[("ortho.tif", 1), ("ortho.tif", 2)]``. A single path is accepted.
    out_path : str or os.PathLike
        Output GeoTIFF (tiled, deflate-compressed, BigTIFF when needed), on the
        grid of the inputs. Written atomically.
    bands : int, sequence of int or None, default 1
        Bands read from each input, as in :func:`farq.read`: an int gives 2-D
        ``(rows, cols)`` arrays, a sequence (or ``None`` for all bands) gives
        3-D ``(bands, rows, cols)`` arrays.
    block_size : int or (int, int), default 1024
        Block size in pixels; rounded down to a multiple of the first input's
        internal tile size. Memory use is roughly
        ``(rows + 2 * overlap) * (cols + 2 * overlap) * bytes per pixel`` for each
        input band and each of the up to ``2 * n_jobs`` blocks in flight.
    overlap : int, default 0
        Halo in pixels added around each block (clipped at the raster edge), so
        that neighbourhood operations (filters, morphology) give the same result
        as on the full raster, provided the halo is at least the operation's
        radius.
    dtype : str or numpy.dtype, optional
        Output dtype; defaults to the dtype of ``func``'s result on the first
        block.
    nodata : float or None, optional
        Output nodata value. Defaults to NaN for float outputs and to no nodata
        for integer outputs. NaN results are stored as ``nodata`` when it is
        finite; writing NaN to an integer dtype requires a nodata value.
    masked : bool, default True
        Read nodata pixels as NaN (integer data become float, as in
        :func:`farq.read`). If False, raw values are passed to ``func``.
    n_jobs : int, default 1
        Number of worker threads (``-1`` for all CPUs). GDAL reads and most NumPy
        operations release the GIL. The output is identical for any ``n_jobs``.
        ``func`` must be thread-safe when ``n_jobs > 1``; in particular it must not
        use :class:`warnings.catch_warnings`, which changes process-wide state
        (this includes the ``farq.mean``/``farq.std``-style NaN reductions; use
        ``numpy.nanmean`` and friends with :func:`numpy.errstate` instead).
    progress : callable, optional
        Called as ``progress(done, total)`` after each block is written.
    **profile
        GeoTIFF creation options overriding the defaults (``tiled=True``,
        ``blockxsize=256``, ``blockysize=256``, ``compress="deflate"``,
        ``BIGTIFF="IF_SAFER"``, and ``NUM_THREADS=n_jobs`` for compression when
        ``n_jobs > 1``), e.g. ``compress="zstd"`` or ``predictor=2``.

    Returns
    -------
    dict
        Metadata of the written file (``driver``, ``dtype``, ``nodata``,
        ``width``, ``height``, ``count``, ``crs``, ``transform``, plus ``gcps``
        and ``gcps_crs`` for GCP-referenced inputs), usable with
        :func:`farq.write`.

    Raises
    ------
    FileNotFoundError
        If an input does not exist.
    IndexError
        If a requested band does not exist.
    ValueError
        If the inputs are not on the same grid, a parameter is invalid, or
        ``func`` returns an array of the wrong shape.
    TypeError
        If ``func`` does not return a numeric numpy array.

    Examples
    --------
    Smooth a large raster with a 5x5 mean filter (radius 2, so ``overlap=2``)::

        from scipy import ndimage
        from farq.tiling import map_blocks

        map_blocks(lambda a: ndimage.uniform_filter(a, 5), ["dem.tif"], "dem_smooth.tif",
                   overlap=2, block_size=2048, n_jobs=4)
    """
    if not callable(func):
        raise TypeError("func must be callable")
    specs = _parse_inputs(inputs, bands)
    n_jobs = _n_jobs(n_jobs)
    profile = _with_threads(profile, n_jobs)
    overlap = _non_negative_int(overlap, "overlap")
    grid, specs = _check_inputs(specs)
    blocks = list(
        iter_windows(grid.width, grid.height, block_size, overlap, align_to=grid.block_shape)
    )
    _check_block_memory(specs, blocks)
    tracker = _Progress(progress, len(blocks))
    reader = _Reader(specs, masked, min(n_jobs, len(blocks)))
    state: dict[str, Any] = {}

    def work(block: Block) -> np.ndarray:
        arrays = reader.read(block.read_window)
        result = func(*arrays)
        return _crop_result(result, block)

    def consume(block: Block, result: np.ndarray) -> None:
        writer: _Writer | None = state.get("writer")
        if writer is None:
            count = 1 if result.ndim == 2 else result.shape[0]
            out_dtype = _out_dtype(dtype, result.dtype)
            writer = state["writer"] = _Writer(
                out_path, grid, count, out_dtype, _out_nodata(nodata, out_dtype), profile
            )
            state["ndim"] = result.ndim
        if result.ndim != state["ndim"] or (result.ndim == 3 and result.shape[0] != writer.count):
            raise ValueError(
                f"func returned an array of shape {result.shape} for block {block.number}, "
                f"but {writer.count} band(s) with {state['ndim']} dimensions on the first block"
            )
        if np.ma.isMaskedArray(result):
            if writer.nodata is None:
                raise ValueError(
                    "func returned a masked array, which requires an output nodata value"
                )
            result = np.ma.filled(result, writer.nodata)
        writer.write(block.write_window, result)

    try:
        _run(blocks, work, consume, n_jobs, tracker)
    except BaseException:
        reader.close()
        if "writer" in state:
            state["writer"].discard()
        raise
    reader.close()
    writer = state["writer"]
    writer.commit()
    return dict(writer.meta)


def _crop_result(result: Any, block: Block) -> np.ndarray:
    if not isinstance(result, np.ndarray):
        raise TypeError(f"func must return a numpy array, got {type(result).__name__}")
    if result.dtype.kind not in "biuf":
        raise TypeError(f"func must return a boolean or numeric array, got dtype {result.dtype}")
    expected = (block.read_window.height, block.read_window.width)
    if result.ndim not in (2, 3) or result.shape[-2:] != expected:
        raise ValueError(
            f"func must return an array of shape {expected} or (bands, *{expected}) "
            f"(the spatial shape of its inputs), got {result.shape}"
        )
    return result[..., block.inner[0], block.inner[1]]


# --------------------------------------------------------------------------- #
# index_file
# --------------------------------------------------------------------------- #
def index_file(
    index_name: str,
    band_paths: Mapping[str, PathLike | tuple[PathLike, int]],
    out_path: PathLike,
    *,
    block_size: int | tuple[int, int] = 1024,
    n_jobs: int = 1,
    progress: ProgressCallback | None = None,
    dtype: str | np.dtype | None = None,
    profile: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute a farq spectral or RGB index block by block, file to file.

    The result equals ``farq.<index>(...)`` on the full rasters read with
    ``farq.read(..., masked=True)``: nodata pixels are NaN in the output.

    Parameters
    ----------
    index_name : str
        Index name (case-insensitive): ``"ndvi"``, ``"ndwi"``, ``"mndwi"``,
        ``"evi"``, ``"savi"``, ``"ndbi"``, ``"nbr"``, ``"ndmi"``, or the RGB-only
        ``"vari"``, ``"exg"``, ``"exr"``, ``"exgr"``, ``"gli"``, ``"ngrdi"``,
        ``"tgi"``.
    band_paths : mapping
        Band name (``"blue"``, ``"green"``, ``"red"``, ``"nir"``, ``"swir1"``,
        ``"swir2"``) to a file path (band 1 is read) or a ``(path, band)`` pair,
        e.g. ``{"red": ("ortho.tif", 1), "green": ("ortho.tif", 2), ...}`` for an
        RGB orthomosaic. All files must share one grid. Bands not needed by the
        index are ignored.
    out_path : str or os.PathLike
        Output GeoTIFF (float, NaN nodata).
    block_size, n_jobs, progress
        See :func:`map_blocks`.
    dtype : str or numpy.dtype, optional
        Output dtype (default: the index dtype, normally ``float32``).
    profile : mapping, optional
        GeoTIFF creation options; see :func:`map_blocks`.
    **kwargs
        Passed to the index function, e.g. ``reflectance_scale=10000``,
        ``clip=False`` or ``L=0.5`` for SAVI.

    Returns
    -------
    dict
        Metadata of the written file; see :func:`map_blocks`.

    Raises
    ------
    ValueError
        If the index is unknown, a required band is missing, or the files are
        not on the same grid.

    Examples
    --------
    ::

        from farq.tiling import index_file

        index_file("vari", {"red": ("ortho.tif", 1), "green": ("ortho.tif", 2),
                            "blue": ("ortho.tif", 3)}, "vari.tif", n_jobs=4)
    """
    from .indices import _INDICES

    if not isinstance(index_name, str):
        raise TypeError(f"index_name must be a string, got {type(index_name).__name__}")
    key = index_name.lower()
    if key not in _INDICES:
        raise ValueError(f"Unknown index: {index_name!r}. Available: {', '.join(_INDICES)}")
    if not isinstance(band_paths, Mapping):
        raise TypeError(f"band_paths must be a mapping, got {type(band_paths).__name__}")
    required, index_func = _INDICES[key]
    missing = [b for b in required if b not in band_paths]
    if missing:
        raise ValueError(f"Missing required bands for {key}: {missing}")

    inputs: list[tuple[PathLike, BandSpec]] = []
    for name in required:
        source = band_paths[name]
        if isinstance(source, tuple):
            if len(source) != 2 or not isinstance(source[1], (int, np.integer)):
                raise ValueError(
                    f"band_paths[{name!r}] must be a path or a (path, band) pair with an int band"
                )
            inputs.append((source[0], int(source[1])))
        else:
            inputs.append((source, 1))

    def compute(*arrays: np.ndarray) -> np.ndarray:
        return index_func(*arrays, **kwargs)

    return map_blocks(
        compute,
        inputs,
        out_path,
        block_size=block_size,
        dtype=dtype,
        n_jobs=n_jobs,
        progress=progress,
        **dict(profile or {}),
    )


# --------------------------------------------------------------------------- #
# Connected components across blocks
# --------------------------------------------------------------------------- #
class _BlockLabels(NamedTuple):
    sizes: np.ndarray  # pixels per local label (index 0 = background)
    border: np.ndarray  # local labels touching the block edge (sorted, > 0)
    edges: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # top, bottom, left, right


def _label_block(mask: np.ndarray, structure: np.ndarray) -> tuple[np.ndarray, _BlockLabels]:
    labels, n = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    edges = (labels[0], labels[-1], labels[:, 0], labels[:, -1])
    border = np.unique(np.concatenate(edges))
    return labels, _BlockLabels(sizes, border[border > 0], edges)


def _edge_pairs(a: np.ndarray, b: np.ndarray, diagonal: bool) -> list[np.ndarray]:
    """Node pairs of facing pixels on two adjacent block edges (-1 = no component)."""
    pairs = [np.stack([a, b])]
    if diagonal and a.size > 1:
        pairs.append(np.stack([a[:-1], b[1:]]))
        pairs.append(np.stack([a[1:], b[:-1]]))
    out = np.concatenate(pairs, axis=1)
    out = out[:, (out[0] >= 0) & (out[1] >= 0)]
    return [np.unique(out, axis=1)] if out.size else []


class _Components:
    """Global connected components of a mask that is only available block by block.

    Pass 1 (:meth:`add`, blocks in row-major order) labels each block and links
    labels that touch across block edges. Components fully inside a block are
    resolved locally; only labels on block edges become graph nodes, so memory
    is proportional to the total length of block edges, not to the raster.
    """

    def __init__(self, grid_shape: tuple[int, int], diagonal: bool) -> None:
        self.n_rows, self.n_cols = grid_shape
        self.diagonal = diagonal
        self.offsets: list[int] = []
        self.borders: list[np.ndarray] = []
        self.sizes: list[np.ndarray] = []
        self.touch: list[np.ndarray] = []
        self.pairs: list[np.ndarray] = []
        self.n_nodes = 0
        self._prev_bottom: list[np.ndarray | None] = [None] * self.n_cols
        self._cur_bottom: list[np.ndarray | None] = [None] * self.n_cols
        self._last_right: np.ndarray | None = None

    def add(self, index: int, info: _BlockLabels) -> None:
        row, col = divmod(index, self.n_cols)
        if col == 0 and row > 0:
            self._prev_bottom, self._cur_bottom = self._cur_bottom, [None] * self.n_cols
        lut = np.full(info.sizes.size, -1, dtype=np.int64)
        lut[info.border] = self.n_nodes + np.arange(info.border.size)
        top, bottom, left, right = (lut[e] for e in info.edges)

        on_image_edge = (
            row == 0,
            row == self.n_rows - 1,
            col == 0,
            col == self.n_cols - 1,
        )
        image_edges = [e for e, flag in zip(info.edges, on_image_edge) if flag]
        if image_edges:
            touch = np.isin(info.border, np.concatenate(image_edges))
        else:
            touch = np.zeros(info.border.size, dtype=bool)

        if col > 0 and self._last_right is not None:
            self.pairs += _edge_pairs(self._last_right, left, self.diagonal)
        if row > 0:
            above = self._prev_bottom[col]
            assert above is not None
            self.pairs += _edge_pairs(above, top, self.diagonal)
            if self.diagonal:
                corners = []
                upper_left = self._prev_bottom[col - 1] if col > 0 else None
                if upper_left is not None:
                    corners.append((upper_left[-1], top[0]))
                upper_right = self._prev_bottom[col + 1] if col + 1 < self.n_cols else None
                if upper_right is not None:
                    corners.append((upper_right[0], top[-1]))
                for u, v in corners:
                    if u >= 0 and v >= 0:
                        self.pairs.append(np.array([[u], [v]], dtype=np.int64))
        self._cur_bottom[col] = bottom
        self._last_right = right

        self.offsets.append(self.n_nodes)
        self.borders.append(info.border)
        self.sizes.append(info.sizes[info.border].astype(np.int64))
        self.touch.append(touch)
        self.n_nodes += int(info.border.size)

    def resolve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the total size and image-border flag of each node's component."""
        n = self.n_nodes
        if n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
        if self.pairs:
            pairs = np.concatenate(self.pairs, axis=1)
            graph = coo_matrix(
                (np.ones(pairs.shape[1], dtype=np.int8), (pairs[0], pairs[1])), shape=(n, n)
            )
            _, comp = connected_components(graph, directed=False)
        else:
            comp = np.arange(n)
        sizes = np.concatenate(self.sizes)
        touch = np.concatenate(self.touch)
        comp_size = np.bincount(comp, weights=sizes).astype(np.int64)
        comp_touch = np.bincount(comp, weights=touch).astype(np.int64) > 0
        self.pairs = []
        return comp_size[comp], comp_touch[comp]

    def node_slice(self, index: int) -> tuple[np.ndarray, slice]:
        border = self.borders[index]
        start = self.offsets[index]
        return border, slice(start, start + border.size)


# --------------------------------------------------------------------------- #
# detect_changes_file
# --------------------------------------------------------------------------- #
class _Sample:
    """Uniform random sample of ``k`` values (bottom-k keys), mergeable across blocks."""

    def __init__(self, k: int | None) -> None:
        self.k = k
        self.keys: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.buffered = 0
        self.cutoff = np.inf
        self.seen = 0

    def add(self, keys: np.ndarray, values: np.ndarray, positions: np.ndarray, n: int) -> None:
        self.seen += n
        if self.k is not None and np.isfinite(self.cutoff):
            keep = keys < self.cutoff
            keys, values, positions = keys[keep], values[keep], positions[keep]
        if keys.size == 0:
            return
        self.keys.append(keys)
        self.values.append(values)
        self.positions.append(positions)
        self.buffered += keys.size
        if self.k is not None and self.buffered > 2 * self.k:
            self._consolidate()

    def _consolidate(self) -> None:
        keys = np.concatenate(self.keys)
        values = np.concatenate(self.values)
        positions = np.concatenate(self.positions)
        if self.k is not None and keys.size > self.k:
            keep = np.argpartition(keys, self.k - 1)[: self.k]
            keys, values, positions = keys[keep], values[keep], positions[keep]
            self.cutoff = float(keys.max())
        self.keys, self.values, self.positions = [keys], [values], [positions]
        self.buffered = keys.size

    def result(self) -> np.ndarray:
        """Sampled values in raster (row-major) order."""
        if not self.keys:
            return np.zeros(0, dtype=np.float64)
        self._consolidate()
        order = np.argsort(self.positions[0], kind="stable")
        return self.values[0][order]


def _block_keys(
    seed: int, block: Block, valid_count: int, k: int | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Random keys for a block's valid pixels and, if ``k`` is set, which to keep."""
    rng = np.random.default_rng([int(seed), block.number])
    keys = rng.random(valid_count)
    if k is not None and valid_count > k:
        keep = np.argpartition(keys, k - 1)[:k]
        return keys[keep], keep
    return keys, None


def _magnitude(before: np.ndarray, after: np.ndarray, method: str) -> np.ndarray:
    """Change magnitude of one block, exactly as :func:`farq.change.detect_changes`."""
    shape = before.shape[-2:]
    if np.isnan(before).all() or np.isnan(after).all():
        # farq.change rejects all-NaN inputs; a block entirely outside the data
        # (e.g. the corner of an orthomosaic) simply has no valid magnitude.
        dtype = _change._float_dtype(np.result_type(before.dtype, after.dtype))
        return np.full(shape, np.nan, dtype=dtype)
    if method == "cva":
        return _change.change_vector_analysis(before, after).magnitude
    if method == "difference":
        magnitude = _change.difference(before, after)
    elif method == "ratio":
        magnitude = _change.ratio(before, after)
    else:
        magnitude = _change.normalized_difference_change(before, after)
    np.abs(magnitude, out=magnitude)
    return magnitude


def _pixel_area_of(grid: _Grid, pixel_size: Any) -> float | None:
    if pixel_size is not None:
        return _change._pixel_area(pixel_size)
    if grid.transform.is_identity and (grid.crs is None or grid.gcps):
        return None  # no geotransform: the ground size of a pixel is unknown
    return _change._pixel_area({"transform": grid.transform, "crs": grid.crs})


def detect_changes_file(
    before_path: PathLike,
    after_path: PathLike,
    out_path: PathLike,
    *,
    method: str = "difference",
    threshold: str | float = "otsu",
    k: float = 2.0,
    percentile: float = 95.0,
    min_size: int = 0,
    connectivity: int = 8,
    fill_holes: bool | int = False,
    bands: BandSpec = 1,
    sample_size: int | None = 1_000_000,
    seed: int = 0,
    magnitude_path: PathLike | None = None,
    pixel_size: Any = None,
    block_size: int | tuple[int, int] = 1024,
    n_jobs: int = 1,
    progress: ProgressCallback | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Out-of-core :func:`farq.change.detect_changes` for rasters larger than memory.

    The rasters are processed block by block in several passes:

    1. If ``threshold`` is a rule (``"otsu"``, ``"std"``, ``"percentile"``), the
       change magnitude is computed for every block and a uniform random sample
       of ``sample_size`` valid pixels is drawn (reproducible from ``seed``,
       independent of ``block_size`` layout order and ``n_jobs``). One *global*
       threshold is computed from it with :func:`farq.change.compute_threshold`.
       If the raster has at most ``sample_size`` valid pixels, all of them are
       used and the threshold is exactly the in-memory one.
    2. If ``min_size > 1`` (and again if ``fill_holes``), connected regions are
       labelled per block and linked across block edges, giving exact global
       region sizes (no tiling artefacts at block borders).
    3. The final mask is written.

    With a numeric threshold, or when the sample covers all valid pixels, the
    output equals ``detect_changes(before, after, ...)`` on the full rasters
    read with ``farq.read(..., masked=True)`` exactly, pixel for pixel.

    Parameters
    ----------
    before_path, after_path : str or os.PathLike
        Co-registered rasters of the two dates (same CRS, transform and size).
    out_path : str or os.PathLike
        Output change mask: ``uint8`` GeoTIFF with 1 = change, 0 = no change and
        255 (:data:`farq.CHANGE_NODATA`, the nodata value) where either date is
        invalid.
    method : {"difference", "ratio", "normalized_difference", "cva"}
        Change measure; see :func:`farq.change.detect_changes`. ``"pca"`` and
        ``"irmad"`` need global statistics and are not available block-wise.
    threshold : {"otsu", "std", "percentile"} or float, default "otsu"
        Threshold rule or a fixed value.
    k, percentile
        Parameters of the ``"std"`` and ``"percentile"`` rules.
    min_size : int, default 0
        Remove change regions smaller than this many pixels (exact across
        blocks).
    connectivity : {4, 8}, default 8
        Neighbourhood used by ``min_size`` and ``fill_holes``.
    fill_holes : bool or int, default False
        Fill holes inside change regions (``True``) or holes of at most this many
        pixels (int), as :func:`farq.change.clean_mask` (exact across blocks).
    bands : int, sequence of int or None, default 1
        Bands to read from both files. Use a sequence or ``None`` (all bands)
        with ``method="cva"``.
    sample_size : int or None, default 1_000_000
        Number of pixels used to estimate the threshold. ``None`` uses every
        valid pixel (exact, but needs memory for all of them).
    seed : int, default 0
        Seed of the pixel sample.
    magnitude_path : str or os.PathLike, optional
        Also write the change magnitude (float, NaN nodata) to this file.
    pixel_size : optional
        Pixel footprint for areas (see :func:`farq.change_summary`). Defaults to
        the files' geotransform; areas are None for rasters without one. Rasters
        in a geographic CRS (degrees) raise ``ValueError`` before any processing,
        unless ``pixel_size`` is given in metres.
    block_size, n_jobs, progress
        See :func:`map_blocks`. ``progress(done, total)`` counts blocks over all
        passes.
    profile : mapping, optional
        GeoTIFF creation options for the output(s); see :func:`map_blocks`.

    Returns
    -------
    dict
        ``method``, ``threshold`` (the value applied), ``threshold_method``
        (the rule, or ``"fixed"``), ``threshold_exact`` (False if estimated from
        a sample), ``sampled_pixels`` (None for a fixed threshold),
        ``total_pixels``, ``valid_pixels``, ``nodata_pixels``,
        ``changed_pixels``, ``changed_percent`` (of valid pixels),
        ``pixel_area_m2``, ``changed_area_m2`` and ``changed_area_km2`` (None if
        the pixel area is unknown). All values are JSON-serializable.

    Raises
    ------
    ValueError
        If the files are not on one grid, use a geographic CRS without an
        explicit ``pixel_size``, a parameter is invalid, or there are no valid
        pixels to estimate a threshold from.

    Examples
    --------
    ::

        from farq.tiling import detect_changes_file

        summary = detect_changes_file("ndwi_2020.tif", "ndwi_2024.tif", "change.tif",
                                      threshold="otsu", min_size=10, n_jobs=4)
        print(summary["changed_area_km2"], summary["threshold"])
    """
    if method not in _TILED_METHODS:
        if method in ("pca", "irmad"):
            raise ValueError(
                f"method={method!r} needs statistics of the whole raster and is not available "
                f"block-wise; use one of {_TILED_METHODS} (e.g. 'cva' for multi-band stacks)"
            )
        raise ValueError(f"Unknown method {method!r}; use one of {_TILED_METHODS}")
    if method != "cva" and not isinstance(_indexes(bands), int):
        raise ValueError(
            f"method {method!r} expects single-band rasters; pass an int for bands, "
            "or use method='cva' for multi-band stacks"
        )
    min_size = _non_negative_int(min_size, "min_size")
    structure = _change._structure(connectivity)
    max_hole = _max_hole(fill_holes)
    if sample_size is not None:
        sample_size = _positive_int(sample_size, "sample_size")
        _check_output_size((sample_size,), "sample_size")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    fixed = None
    if isinstance(threshold, (bool, np.bool_)):
        raise TypeError("threshold must be a string or a number, not a bool")
    if isinstance(threshold, (int, float, np.integer, np.floating)):
        fixed = _change.compute_threshold(np.zeros(0), threshold)
    elif threshold not in _change._THRESHOLD_METHODS:
        raise ValueError(
            f"Unknown threshold method {threshold!r}; use one of "
            f"{_change._THRESHOLD_METHODS} or a number"
        )
    elif threshold == "percentile" and not 0.0 <= float(percentile) <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    n_jobs = _n_jobs(n_jobs)
    profile = _with_threads(profile or {}, n_jobs)
    if magnitude_path is not None and os.path.realpath(os.fspath(magnitude_path)) == (
        os.path.realpath(os.fspath(out_path))
    ):
        raise ValueError("magnitude_path must be a different file from out_path")

    specs = _parse_inputs([before_path, after_path], bands)
    grid, specs = _check_inputs(specs)
    area = _pixel_area_of(grid, pixel_size)
    blocks = list(iter_windows(grid.width, grid.height, block_size, align_to=grid.block_shape))
    _check_block_memory(specs, blocks)
    grid_shape = _grid_shape(grid.width, grid.height, blocks)
    n_passes = 1 + (fixed is None) + (min_size > 1) + (max_hole > 0)
    tracker = _Progress(progress, len(blocks) * n_passes)
    reader = _Reader(specs, masked=True, n_handles=min(n_jobs, len(blocks)))

    def magnitude_of(block: Block) -> np.ndarray:
        before, after = reader.read(block.read_window)
        return _magnitude(before, after, method)

    writers: list[_Writer] = []
    try:
        # Pass 1: global threshold from a reproducible uniform pixel sample.
        if fixed is None:
            sample = _Sample(sample_size)

            def sample_work(block: Block) -> tuple[Any, ...]:
                mag = magnitude_of(block).ravel()
                where = np.flatnonzero(np.isfinite(mag))
                n_valid = int(where.size)
                keys, keep = _block_keys(seed, block, where.size, sample_size)
                if keep is not None:
                    where = where[keep]
                w = block.write_window
                rows, cols = np.divmod(where, w.width)
                positions = (rows + w.row_off).astype(np.int64) * grid.width + (cols + w.col_off)
                return keys, mag[where], positions, n_valid

            def sample_consume(block: Block, res: tuple[Any, ...]) -> None:
                sample.add(*res)

            _run(blocks, sample_work, sample_consume, n_jobs, tracker)
            values = sample.result()
            if values.size == 0:
                raise ValueError(
                    "The change magnitude has no valid pixels (both rasters are entirely "
                    "nodata where they overlap); cannot estimate a threshold"
                )
            t = _change.compute_threshold(values, threshold, k=k, percentile=percentile)
            sampled = int(values.size)
            exact = sampled == sample.seen
        else:
            t, sampled, exact = fixed, None, True

        def raw_mask(block: Block) -> tuple[np.ndarray, np.ndarray]:
            mag = magnitude_of(block)
            return mag, mag > t

        # Pass 2: exact removal of regions smaller than min_size across blocks.
        regions = None
        if min_size > 1:
            regions = _Components(grid_shape, connectivity == 8)

            def regions_work(block: Block) -> _BlockLabels:
                return _label_block(raw_mask(block)[1], structure)[1]

            _run(blocks, regions_work, lambda b, r: regions.add(b.number, r), n_jobs, tracker)
            region_size, _ = regions.resolve()
            region_keep = region_size >= min_size

        def cleaned_mask(block: Block) -> tuple[np.ndarray, np.ndarray]:
            mag, mask = raw_mask(block)
            if regions is not None:
                labels, info = _label_block(mask, structure)
                keep = info.sizes >= min_size
                keep[0] = False
                border, nodes = regions.node_slice(block.number)
                keep[border] = region_keep[nodes]
                mask = keep[labels]
            return mag, mask

        # Pass 3: exact hole filling across blocks.
        holes = None
        hole_structure = _change._structure(12 - connectivity)
        if max_hole > 0:
            holes = _Components(grid_shape, connectivity == 4)

            def holes_work(block: Block) -> _BlockLabels:
                return _label_block(~cleaned_mask(block)[1], hole_structure)[1]

            _run(blocks, holes_work, lambda b, r: holes.add(b.number, r), n_jobs, tracker)
            hole_size, hole_touch = holes.resolve()
            hole_fill = ~hole_touch & (hole_size <= max_hole)

        def final_work(block: Block) -> tuple[np.ndarray, np.ndarray]:
            mag, mask = cleaned_mask(block)
            if holes is not None:
                labels, info = _label_block(~mask, hole_structure)
                fill = info.sizes <= max_hole
                fill[0] = False
                border, nodes = holes.node_slice(block.number)
                fill[border] = hole_fill[nodes]
                mask = mask | fill[labels]
            valid = np.isfinite(mag)
            mask &= valid
            out = mask.astype(np.uint8)
            out[~valid] = _change.CHANGE_NODATA
            return out, mag

        counts = {"valid": 0, "changed": 0}

        def final_consume(block: Block, res: tuple[np.ndarray, np.ndarray]) -> None:
            out, mag = res
            if not writers:
                writers.append(
                    _Writer(
                        out_path,
                        grid,
                        1,
                        np.dtype(np.uint8),
                        _change.CHANGE_NODATA,
                        profile,
                    )
                )
                if magnitude_path is not None:
                    writers.append(
                        _Writer(magnitude_path, grid, 1, mag.dtype, float("nan"), profile)
                    )
            counts["valid"] += int(np.count_nonzero(out != _change.CHANGE_NODATA))
            counts["changed"] += int(np.count_nonzero(out == 1))
            writers[0].write(block.write_window, out)
            if magnitude_path is not None:
                writers[1].write(block.write_window, mag)

        _run(blocks, final_work, final_consume, n_jobs, tracker)
    except BaseException:
        reader.close()
        for writer in writers:
            writer.discard()
        raise
    reader.close()
    for writer in writers:
        writer.commit()

    total = grid.width * grid.height
    n_valid, n_changed = counts["valid"], counts["changed"]
    changed_m2 = None if area is None else float(n_changed * area)
    return {
        "method": method,
        "threshold": float(t),
        "threshold_method": "fixed" if fixed is not None else str(threshold),
        "threshold_exact": bool(exact),
        "sampled_pixels": sampled,
        "total_pixels": int(total),
        "valid_pixels": n_valid,
        "nodata_pixels": int(total - n_valid),
        "changed_pixels": n_changed,
        "changed_percent": 100.0 * n_changed / n_valid if n_valid else 0.0,
        "pixel_area_m2": None if area is None else float(area),
        "changed_area_m2": changed_m2,
        "changed_area_km2": None if changed_m2 is None else changed_m2 / 1e6,
    }


def _max_hole(fill_holes: Any) -> float:
    if fill_holes is None or fill_holes is False:
        return 0
    if isinstance(fill_holes, (bool, np.bool_)):
        return math.inf if fill_holes else 0
    if not isinstance(fill_holes, (int, np.integer)) or fill_holes < 0:
        raise ValueError(f"fill_holes must be a bool or a non-negative int, got {fill_holes}")
    return int(fill_holes)


# --------------------------------------------------------------------------- #
# summarize_file
# --------------------------------------------------------------------------- #
class _Moments(NamedTuple):
    n: int
    mean: float
    m2: float
    total: float
    vmin: float
    vmax: float
    n_nan: int
    n_inf: int

    @classmethod
    def of(cls, data: np.ndarray) -> _Moments:
        finite = np.isfinite(data) if data.dtype.kind == "f" else None
        values = data.ravel() if finite is None or finite.all() else data[finite]
        n_nan = 0 if finite is None else int(np.count_nonzero(np.isnan(data)))
        n_inf = 0 if finite is None else int(data.size - values.size - n_nan)
        if values.size == 0:
            return cls(0, 0.0, 0.0, 0.0, math.inf, -math.inf, n_nan, n_inf)
        v = values.astype(np.float64)
        mean = float(v.mean())
        centered = v - mean
        return cls(
            int(v.size),
            mean,
            float(np.dot(centered, centered)),
            float(v.sum()),
            float(v.min()),
            float(v.max()),
            n_nan,
            n_inf,
        )

    def merge(self, other: _Moments) -> _Moments:
        """Chan et al. parallel update of count, mean and sum of squared deviations."""
        if other.n == 0 or self.n == 0:
            base = self if other.n == 0 else other
            return base._replace(
                n_nan=self.n_nan + other.n_nan,
                n_inf=self.n_inf + other.n_inf,
                vmin=min(self.vmin, other.vmin),
                vmax=max(self.vmax, other.vmax),
            )
        n = self.n + other.n
        delta = other.mean - self.mean
        return _Moments(
            n,
            self.mean + delta * other.n / n,
            self.m2 + other.m2 + delta * delta * self.n * other.n / n,
            self.total + other.total,
            min(self.vmin, other.vmin),
            max(self.vmax, other.vmax),
            self.n_nan + other.n_nan,
            self.n_inf + other.n_inf,
        )


def summarize_file(
    path: PathLike,
    band: BandSpec = 1,
    *,
    bins: int | None = 50,
    masked: bool = True,
    block_size: int | tuple[int, int] = 1024,
    n_jobs: int = 1,
    progress: ProgressCallback | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Statistics of a raster file computed block by block.

    Mean and standard deviation are accumulated with a numerically stable
    streaming (Welford/Chan) update in ``float64``, so they agree with
    :func:`farq.stats` on the full raster to floating point rounding. The
    histogram is exact (same bin edges and counts as :func:`farq.stats`); it
    needs a second pass over the file.

    Parameters
    ----------
    path : str or os.PathLike
        Raster file.
    band : int, sequence of int or None, default 1
        Band(s) to summarize; ``None`` for all bands.
    bins : int or None, default 50
        Histogram bins over ``[min, max]``. ``None`` skips the histogram (and the
        second pass).
    masked : bool, default True
        Treat the file's nodata (and masks) as NaN, as ``farq.read(masked=True)``.
    block_size, n_jobs, progress
        See :func:`map_blocks`.

    Returns
    -------
    dict or list of dict
        For an int ``band`` a dict, otherwise one dict per band, with
        ``band``, ``shape``, ``size``, ``valid`` (finite values), ``nan``,
        ``inf``, ``min``, ``max``, ``range``, ``mean``, ``std`` (population),
        ``variance``, ``sum``, ``percentages`` (``valid``, ``nan``, ``inf``) and
        ``histogram`` (``counts`` and ``bin_edges`` arrays; omitted when
        ``bins`` is None). Value statistics are NaN when there are no finite
        values.

    Examples
    --------
    ::

        from farq.tiling import summarize_file

        s = summarize_file("ortho.tif", band=None, bins=None, n_jobs=4)
        print([(b["band"], b["mean"]) for b in s])
    """
    if bins is not None:
        bins = _positive_int(bins, "bins")
    n_jobs = _n_jobs(n_jobs)
    specs = _parse_inputs([path], band)
    grid, specs = _check_inputs(specs)
    indexes = specs[0].indexes
    band_list = [indexes] if isinstance(indexes, int) else list(indexes)
    blocks = list(iter_windows(grid.width, grid.height, block_size, align_to=grid.block_shape))
    _check_block_memory(specs, blocks)
    read_specs = [_Input(specs[0].path, band_list)]
    tracker = _Progress(progress, len(blocks) * (2 if bins is not None else 1))
    reader = _Reader(read_specs, masked, min(n_jobs, len(blocks)))
    empty = _Moments(0, 0.0, 0.0, 0.0, math.inf, -math.inf, 0, 0)
    moments = [empty] * len(band_list)
    counts: list[np.ndarray] = []
    edges: list[np.ndarray] = []

    value_dtype: list[np.dtype] = []

    def moments_work(block: Block) -> list[_Moments]:
        data = reader.read(block.read_window)[0]
        if not value_dtype:
            value_dtype.append(np.dtype(np.uint8) if data.dtype == np.bool_ else data.dtype)
        return [_Moments.of(layer) for layer in data]

    def moments_consume(block: Block, result: list[_Moments]) -> None:
        for i, m in enumerate(result):
            moments[i] = moments[i].merge(m)

    try:
        _run(blocks, moments_work, moments_consume, n_jobs, tracker)
        if bins is not None:
            for m in moments:
                if m.n:
                    # Same dtype as the data, so edges and bin assignment match
                    # np.histogram on the whole array (as used by farq.stats).
                    lo_hi = np.array([m.vmin, m.vmax]).astype(value_dtype[0])
                    edges.append(np.histogram_bin_edges(lo_hi, bins=bins))
                else:
                    edges.append(np.full(bins + 1, np.nan))
                counts.append(np.zeros(bins, dtype=np.intp))

            def hist_work(block: Block) -> list[np.ndarray]:
                data = reader.read(block.read_window)[0]
                out = []
                for i, layer in enumerate(data):
                    m = moments[i]
                    if not m.n:
                        out.append(np.zeros(bins, dtype=np.intp))
                        continue
                    values = layer[np.isfinite(layer)] if layer.dtype.kind == "f" else layer
                    if values.dtype == np.bool_:
                        values = values.astype(np.uint8)
                    lo, hi = np.array([m.vmin, m.vmax]).astype(values.dtype)
                    out.append(np.histogram(values, bins=bins, range=(lo, hi))[0])
                return out

            def hist_consume(block: Block, result: list[np.ndarray]) -> None:
                for i, c in enumerate(result):
                    counts[i] += c

            _run(blocks, hist_work, hist_consume, n_jobs, tracker)
    finally:
        reader.close()

    size = grid.width * grid.height
    results = []
    for i, (b, m) in enumerate(zip(band_list, moments)):
        nan = float("nan")
        variance = m.m2 / m.n if m.n else nan
        entry: dict[str, Any] = {
            "band": b,
            "shape": (grid.height, grid.width),
            "size": size,
            "valid": m.n,
            "nan": m.n_nan,
            "inf": m.n_inf,
            "min": m.vmin if m.n else nan,
            "max": m.vmax if m.n else nan,
            "range": m.vmax - m.vmin if m.n else nan,
            "mean": m.mean if m.n else nan,
            "std": math.sqrt(variance) if m.n else nan,
            "variance": variance,
            "sum": m.total if m.n else 0.0,
            "percentages": {
                "valid": 100.0 * m.n / size,
                "nan": 100.0 * m.n_nan / size,
                "inf": 100.0 * m.n_inf / size,
            },
        }
        if bins is not None:
            entry["histogram"] = {"counts": counts[i], "bin_edges": edges[i]}
        results.append(entry)
    return results[0] if isinstance(indexes, int) else results
