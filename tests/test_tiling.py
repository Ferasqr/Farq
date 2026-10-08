"""Tests for farq.tiling (out-of-core block-wise processing)."""

from __future__ import annotations

import json
import os
import tracemalloc
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy import ndimage

import farq
from farq import tiling
from farq.change import detect_changes

CRS_UTM = CRS.from_epsg(32633)
TRANSFORM = from_origin(500000.0, 4000000.0, 10.0, 10.0)
NODATA = -9999.0
HEIGHT, WIDTH = 1300, 1500  # deliberately not multiples of the block size


@pytest.fixture(autouse=True)
def _no_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def write_tif(path, data, *, nodata=None, transform=TRANSFORM, crs=CRS_UTM, tiled=True, **kw):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[np.newaxis]
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        **kw,
    }
    if tiled:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def read_all(path, band=1):
    data, _ = farq.read(path, band, masked=True)
    return data


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """Two dates with irregular change patches crossing block borders, and nodata."""
    tmp = tmp_path_factory.mktemp("scene")
    rng = np.random.default_rng(0)
    before = (0.2 + 0.02 * rng.standard_normal((HEIGHT, WIDTH))).astype(np.float32)
    blobs = ndimage.gaussian_filter(rng.standard_normal((HEIGHT, WIDTH)), 6)
    blobs = (blobs / blobs.std()).astype(np.float32)
    after = before + np.where(blobs > 1.0, 0.5, 0.0).astype(np.float32)
    after += (0.02 * rng.standard_normal((HEIGHT, WIDTH))).astype(np.float32)
    before[:300, :300] = NODATA  # a whole 256x256 block without any valid pixel
    after[700:760, 900:1400] = NODATA  # a nodata stripe crossing several blocks
    before[rng.random((HEIGHT, WIDTH)) < 0.01] = NODATA  # scattered nodata
    b = write_tif(tmp / "before.tif", before, nodata=NODATA)
    a = write_tif(tmp / "after.tif", after, nodata=NODATA)
    return b, a


# --------------------------------------------------------------------------- #
# iter_windows
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("overlap", [0, 3])
@pytest.mark.parametrize("block_size", [256, (100, 333), 2000])
def test_iter_windows_tiles_raster_exactly(block_size, overlap):
    cover = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    blocks = list(tiling.iter_windows(WIDTH, HEIGHT, block_size, overlap))
    assert [b.number for b in blocks] == list(range(len(blocks)))
    for b in blocks:
        w, r = b.write_window, b.read_window
        cover[w.row_off : w.row_off + w.height, w.col_off : w.col_off + w.width] += 1
        # inner slices map the read window onto the write window
        assert r.row_off + b.inner[0].start == w.row_off
        assert r.col_off + b.inner[1].start == w.col_off
        assert b.inner[0].stop - b.inner[0].start == w.height
        assert b.inner[1].stop - b.inner[1].start == w.width
        # halo is clipped to the raster
        assert r.row_off >= 0 and r.col_off >= 0
        assert r.row_off + r.height <= HEIGHT and r.col_off + r.width <= WIDTH
        assert w.row_off - r.row_off == min(overlap, w.row_off)
    assert (cover == 1).all()


def test_iter_windows_alignment_and_validation():
    blocks = list(tiling.iter_windows(1000, 1000, 1000, align_to=(256, 256)))
    assert blocks[0].write_window.width == 768  # rounded down to a multiple of 256
    # smaller than one internal tile, or strip-organised: left alone
    assert next(tiling.iter_windows(1000, 1000, 100, align_to=(256, 256))).write_window.width == 100
    strips = next(tiling.iter_windows(1000, 1000, 300, align_to=(16, 1000)))
    assert (strips.write_window.height, strips.write_window.width) == (288, 300)
    with pytest.raises(ValueError, match="block_size must be positive"):
        list(tiling.iter_windows(10, 10, 0))
    with pytest.raises(ValueError, match="overlap"):
        list(tiling.iter_windows(10, 10, 4, -1))
    with pytest.raises(TypeError, match="width"):
        list(tiling.iter_windows(10.5, 10))


# --------------------------------------------------------------------------- #
# map_blocks
# --------------------------------------------------------------------------- #
def test_map_blocks_matches_in_memory(scene, tmp_path):
    b, a = scene
    out = tmp_path / "diff.tif"
    meta = tiling.map_blocks(lambda x, y: y - x, [b, a], out, block_size=256)
    expected = read_all(a) - read_all(b)
    got = read_all(out)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.float32
    assert meta["crs"] == CRS_UTM and meta["transform"] == TRANSFORM
    assert (meta["height"], meta["width"], meta["count"]) == (HEIGHT, WIDTH, 1)
    assert np.isnan(meta["nodata"])
    with rasterio.open(out) as src:
        assert src.compression is not None
        assert src.block_shapes[0] == (256, 256)


def test_map_blocks_overlap_neighbourhood_operation(scene, tmp_path):
    b, _ = scene
    out = tmp_path / "max.tif"

    def local_max(x):  # NaN-free input: comparisons with NaN depend on scan order
        return ndimage.maximum_filter(np.nan_to_num(x, nan=-1.0), size=7)

    tiling.map_blocks(local_max, [b], out, block_size=(200, 300), overlap=3)
    expected = local_max(read_all(b))
    np.testing.assert_array_equal(read_all(out), expected)

    # Without a halo, the block borders differ: shows the halo is what makes it exact.
    tiling.map_blocks(local_max, [b], out, block_size=200)
    assert not np.array_equal(read_all(out), expected)


def test_map_blocks_multiband_and_band_pairs(tmp_path):
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(3, 517, 389), dtype=np.uint8)
    src = write_tif(tmp_path / "rgb.tif", rgb, tiled=False)
    out = tmp_path / "out.tif"
    meta = tiling.map_blocks(
        lambda red, stack: np.stack([red, stack.sum(axis=0)]),
        [(src, 1), (src, [2, 3])],
        out,
        block_size=128,
        masked=False,
        dtype="uint16",
    )
    assert meta["count"] == 2 and meta["dtype"] == "uint16" and meta["nodata"] is None
    with rasterio.open(out) as ds:
        got = ds.read()
    np.testing.assert_array_equal(got[0], rgb[0])
    np.testing.assert_array_equal(got[1], rgb[1].astype(np.uint16) + rgb[2])

    # bands=None reads all bands as a 3-D stack; bool results are written as uint8
    tiling.map_blocks(lambda s: s[0] > s[1], src, out, bands=None, block_size=(64, 1000))
    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == "uint8"
        np.testing.assert_array_equal(ds.read(1), (rgb[0] > rgb[1]).astype(np.uint8))


def test_map_blocks_nodata_handling(tmp_path):
    data = np.arange(300 * 200, dtype=np.int16).reshape(300, 200)
    data[50:60, 20:180] = -1
    src = write_tif(tmp_path / "int.tif", data, nodata=-1)
    out = tmp_path / "out.tif"
    tiling.map_blocks(lambda x: x * 2, [src], out, block_size=64)
    got = read_all(out)
    assert got.dtype == np.float32
    assert np.isnan(got[50:60, 20:180]).all()
    np.testing.assert_array_equal(got[100], data[100] * 2.0)

    # integer output: NaN is stored as the given nodata value
    tiling.map_blocks(lambda x: x, [src], out, block_size=64, dtype="int16", nodata=-32768)
    with rasterio.open(out) as ds:
        assert ds.nodata == -32768
        raw = ds.read(1)
    assert (raw[50:60, 20:180] == -32768).all()
    with pytest.raises(ValueError, match="nodata"):
        tiling.map_blocks(lambda x: x, [src], out, block_size=64, dtype="int16")


def test_map_blocks_grid_mismatch(tmp_path):
    data = np.ones((40, 50), dtype=np.float32)
    ref = write_tif(tmp_path / "ref.tif", data)
    shifted = write_tif(
        tmp_path / "shifted.tif", data, transform=from_origin(500005.0, 4000000.0, 10.0, 10.0)
    )
    other_crs = write_tif(tmp_path / "crs.tif", data, crs=CRS.from_epsg(32634))
    other_size = write_tif(tmp_path / "size.tif", np.ones((41, 50), np.float32))
    out = tmp_path / "out.tif"
    for bad, what in [(shifted, "transform"), (other_crs, "CRS"), (other_size, "size")]:
        with pytest.raises(ValueError, match=f"same pixel grid.*{what}.*align_pair"):
            tiling.map_blocks(lambda x, y: x + y, [ref, bad], out)
    assert not out.exists()

    with pytest.raises(FileNotFoundError):
        tiling.map_blocks(lambda x: x, [tmp_path / "missing.tif"], out)
    with pytest.raises(IndexError, match="Band 2"):
        tiling.map_blocks(lambda x: x, [ref], out, bands=2)
    with pytest.raises(ValueError, match="shape"):
        tiling.map_blocks(lambda x: x[:-1], [ref], out)
    with pytest.raises(TypeError, match="numpy array"):
        tiling.map_blocks(lambda x: 1.0, [ref], out)
    with pytest.raises(ValueError, match="n_jobs"):
        tiling.map_blocks(lambda x: x, [ref], out, n_jobs=0)
    with pytest.raises(ValueError, match="GeoTIFF"):
        tiling.map_blocks(lambda x: x, [ref], out, driver="PNG")


def test_map_blocks_parallel_is_deterministic(scene, tmp_path):
    b, a = scene

    def func(x, y):
        return np.stack([np.log1p(np.abs(y - x)), ndimage.median_filter(y, 3)])

    calls = []
    out1, out4 = tmp_path / "j1.tif", tmp_path / "j4.tif"
    tiling.map_blocks(func, [b, a], out1, block_size=256, overlap=1, n_jobs=1)
    tiling.map_blocks(
        func,
        [b, a],
        out4,
        block_size=256,
        overlap=1,
        n_jobs=4,
        progress=lambda done, total: calls.append((done, total)),
    )
    with rasterio.open(out1) as d1, rasterio.open(out4) as d4:
        np.testing.assert_array_equal(d1.read(), d4.read())
    assert calls == [(i, 36) for i in range(1, 37)]


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_map_blocks_failure_is_atomic(scene, tmp_path, n_jobs):
    b, _ = scene
    out = tmp_path / "out.tif"
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("boom")
        return x

    with pytest.raises(RuntimeError, match="boom"):
        tiling.map_blocks(flaky, [b], out, block_size=256, n_jobs=n_jobs)
    assert os.listdir(tmp_path) == []

    # an existing output survives a failed run untouched
    out.write_bytes(b"previous")
    calls["n"] = 0
    with pytest.raises(RuntimeError, match="boom"):
        tiling.map_blocks(flaky, [b], out, block_size=256, n_jobs=n_jobs)
    assert out.read_bytes() == b"previous"
    assert os.listdir(tmp_path) == ["out.tif"]


def test_map_blocks_memory_is_bounded(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.random((2048, 2048), dtype=np.float32)  # 16 MiB, 64 blocks of 256x256
    src = write_tif(tmp_path / "big.tif", data)
    del data
    out = tmp_path / "out.tif"
    tracemalloc.start()
    try:
        tiling.map_blocks(lambda x: x * 2 + 1, [src], out, block_size=256)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    full = 2048 * 2048 * 4
    assert peak < full / 4, f"peak {peak / 2**20:.1f} MiB is not bounded by the block size"
    stats = tiling.summarize_file(out, bins=None)
    assert stats["valid"] == 2048 * 2048
    assert stats["min"] >= 1.0 and stats["max"] <= 3.0


def test_map_blocks_keeps_gcps(gcp_tiff_path, tmp_path):
    path, data, gcps, gcps_crs = gcp_tiff_path
    out = tmp_path / "out.tif"
    meta = tiling.map_blocks(lambda x: x + 1, [path], out, masked=False, block_size=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(out) as ds:
            got_gcps, got_crs = ds.gcps
            np.testing.assert_array_equal(ds.read(1), data[0] + 1)
    assert len(got_gcps) == len(gcps) == len(meta["gcps"])
    assert got_crs == gcps_crs


# --------------------------------------------------------------------------- #
# index_file
# --------------------------------------------------------------------------- #
def test_index_file_matches_in_memory(tmp_path):
    rng = np.random.default_rng(4)
    nir = rng.integers(0, 10000, size=(700, 900), dtype=np.uint16)
    red = rng.integers(0, 10000, size=(700, 900), dtype=np.uint16)
    nir[:10] = 0  # nodata
    red[100:110, 200:300] = 0
    n = write_tif(tmp_path / "nir.tif", nir, nodata=0)
    r = write_tif(tmp_path / "red.tif", red, nodata=0)
    out = tmp_path / "savi.tif"
    tiling.index_file(
        "SAVI", {"nir": n, "red": r, "blue": n}, out, block_size=256, reflectance_scale=10000
    )
    expected = farq.savi(read_all(n), read_all(r), reflectance_scale=10000)
    np.testing.assert_array_equal(read_all(out), expected)
    assert np.isnan(read_all(out)[:10]).all()

    with pytest.raises(ValueError, match="Missing required bands"):
        tiling.index_file("ndvi", {"nir": n}, out)
    with pytest.raises(ValueError, match="Unknown index"):
        tiling.index_file("foo", {"nir": n}, out)


def test_index_file_rgb_orthomosaic(tmp_path):
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(3, 600, 700), dtype=np.uint8)
    ortho = write_tif(tmp_path / "ortho.tif", rgb, nodata=0)
    out = tmp_path / "vari.tif"
    bands = {"red": (ortho, 1), "green": (ortho, 2), "blue": (ortho, 3)}
    tiling.index_file("vari", bands, out, block_size=200, n_jobs=2, clip=False)
    expected = farq.vari(*read_all(ortho, [1, 2, 3]), clip=False)
    np.testing.assert_array_equal(read_all(out), expected)


# --------------------------------------------------------------------------- #
# detect_changes_file
# --------------------------------------------------------------------------- #
def _in_memory(b, a, **kwargs):
    return detect_changes(read_all(b), read_all(a), **kwargs)


def _assert_same_mask(out, result):
    with rasterio.open(out) as ds:
        got = ds.read(1)
        assert ds.nodata == farq.CHANGE_NODATA and ds.dtypes[0] == "uint8"
    np.testing.assert_array_equal(got == 1, result.mask)
    np.testing.assert_array_equal(got == farq.CHANGE_NODATA, ~np.isfinite(result.magnitude))
    return got


@pytest.mark.parametrize(
    ("min_size", "connectivity", "fill_holes"),
    [(0, 8, False), (300, 8, False), (300, 4, False), (50, 8, True), (40, 4, 60)],
)
def test_detect_changes_file_exact_with_fixed_threshold(
    scene, tmp_path, min_size, connectivity, fill_holes
):
    b, a = scene
    kwargs = {"min_size": min_size, "connectivity": connectivity, "fill_holes": fill_holes}
    expected = _in_memory(b, a, threshold=0.25, **kwargs)
    out, mag = tmp_path / "mask.tif", tmp_path / "mag.tif"
    summary = tiling.detect_changes_file(
        b, a, out, threshold=0.25, block_size=256, magnitude_path=mag, **kwargs
    )
    _assert_same_mask(out, expected)
    np.testing.assert_array_equal(read_all(mag), expected.magnitude)

    ref = expected.summary(pixel_size={"transform": TRANSFORM, "crs": CRS_UTM})
    for key in ("total_pixels", "valid_pixels", "nodata_pixels", "changed_pixels"):
        assert summary[key] == ref[key], key
    assert summary["changed_percent"] == pytest.approx(ref["changed_percent"])
    assert summary["changed_area_m2"] == pytest.approx(ref["changed_area_m2"])
    assert summary["changed_area_km2"] == pytest.approx(ref["changed_area_km2"])
    assert summary["pixel_area_m2"] == 100.0
    assert summary["threshold"] == 0.25 and summary["threshold_method"] == "fixed"
    assert summary["threshold_exact"] and summary["sampled_pixels"] is None
    json.dumps(summary)
    if min_size:
        # the cleanup really removed regions, including ones crossing block borders
        assert expected.mask.sum() < _in_memory(b, a, threshold=0.25).mask.sum()


@pytest.mark.parametrize("threshold", ["otsu", "std", "percentile"])
def test_detect_changes_file_exact_when_sample_covers_all(scene, tmp_path, threshold):
    b, a = scene
    expected = _in_memory(b, a, threshold=threshold, min_size=20)
    out = tmp_path / "mask.tif"
    summary = tiling.detect_changes_file(
        b, a, out, threshold=threshold, min_size=20, sample_size=None, block_size=256
    )
    assert summary["threshold"] == expected.threshold
    assert summary["threshold_exact"]
    assert summary["sampled_pixels"] == summary["valid_pixels"]
    _assert_same_mask(out, expected)


def test_detect_changes_file_sampled_otsu_is_close(scene, tmp_path):
    b, a = scene
    expected = _in_memory(b, a, threshold="otsu")
    out = tmp_path / "mask.tif"
    summary = tiling.detect_changes_file(b, a, out, sample_size=100_000, block_size=256)
    assert not summary["threshold_exact"]
    assert summary["sampled_pixels"] == 100_000
    assert summary["threshold"] == pytest.approx(expected.threshold, rel=0.05)
    with rasterio.open(out) as ds:
        got = ds.read(1) == 1
    disagreement = np.count_nonzero(got != expected.mask) / expected.mask.size
    assert disagreement < 1e-3

    # reproducible, and independent of n_jobs; a different seed gives another sample
    out2 = tmp_path / "mask2.tif"
    again = tiling.detect_changes_file(b, a, out2, sample_size=100_000, block_size=256, n_jobs=4)
    assert again["threshold"] == summary["threshold"]
    other = tiling.detect_changes_file(b, a, out2, sample_size=100_000, block_size=256, seed=1)
    assert other["threshold"] != summary["threshold"]


def test_detect_changes_file_parallel_identical(scene, tmp_path):
    b, a = scene
    kwargs = {"threshold": "otsu", "min_size": 100, "fill_holes": 30, "block_size": 256}
    out1, out4 = tmp_path / "j1.tif", tmp_path / "j4.tif"
    calls = []
    s1 = tiling.detect_changes_file(b, a, out1, n_jobs=1, **kwargs)
    s4 = tiling.detect_changes_file(
        b, a, out4, n_jobs=4, progress=lambda d, t: calls.append((d, t)), **kwargs
    )
    assert s1 == s4
    with rasterio.open(out1) as d1, rasterio.open(out4) as d4:
        np.testing.assert_array_equal(d1.read(), d4.read())
    assert calls[-1] == (36 * 4, 36 * 4) and len(calls) == 36 * 4


def test_detect_changes_file_methods(tmp_path):
    rng = np.random.default_rng(6)
    before = rng.uniform(0.1, 1.0, size=(3, 333, 444)).astype(np.float32)
    after = before * rng.uniform(0.9, 1.1, size=before.shape).astype(np.float32)
    after[:, 100:150, 200:300] *= 3
    before[0, :5] = 0  # zero -> ratio undefined, but a valid pixel
    b = write_tif(tmp_path / "b.tif", before)
    a = write_tif(tmp_path / "a.tif", after)
    out = tmp_path / "out.tif"
    for method in ("ratio", "normalized_difference"):
        tiling.detect_changes_file(b, a, out, method=method, threshold=0.2, block_size=100)
        _assert_same_mask(out, detect_changes(before[0], after[0], method=method, threshold=0.2))
    tiling.detect_changes_file(
        b, a, out, method="cva", bands=None, threshold="otsu", sample_size=None, block_size=100
    )
    _assert_same_mask(out, detect_changes(before, after, method="cva", threshold="otsu"))

    with pytest.raises(ValueError, match="pca"):
        tiling.detect_changes_file(b, a, out, method="pca")
    with pytest.raises(ValueError, match="single-band"):
        tiling.detect_changes_file(b, a, out, bands=[1, 2])
    with pytest.raises(ValueError, match="Unknown threshold"):
        tiling.detect_changes_file(b, a, out, threshold="foo")
    with pytest.raises(ValueError, match="connectivity"):
        tiling.detect_changes_file(b, a, out, connectivity=6)


def test_detect_changes_file_all_nodata(tmp_path):
    data = np.full((50, 60), NODATA, dtype=np.float32)
    b = write_tif(tmp_path / "b.tif", data, nodata=NODATA)
    a = write_tif(tmp_path / "a.tif", data, nodata=NODATA)
    out = tmp_path / "out.tif"
    with pytest.raises(ValueError, match="no valid pixels"):
        tiling.detect_changes_file(b, a, out)
    assert not out.exists()
    summary = tiling.detect_changes_file(b, a, out, threshold=1.0)
    assert summary["valid_pixels"] == 0 and summary["changed_percent"] == 0.0
    with rasterio.open(out) as ds:
        assert (ds.read(1) == farq.CHANGE_NODATA).all()


def test_detect_changes_file_without_geotransform(tmp_path):
    data = np.zeros((20, 30), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        b = write_tif(tmp_path / "b.tif", data, transform=None, crs=None, tiled=False)
        a = write_tif(tmp_path / "a.tif", data + 1, transform=None, crs=None, tiled=False)
        summary = tiling.detect_changes_file(b, a, tmp_path / "o.tif", threshold=0.5)
    assert summary["changed_pixels"] == 600
    assert summary["pixel_area_m2"] is None and summary["changed_area_km2"] is None
    summary = tiling.detect_changes_file(b, a, tmp_path / "o.tif", threshold=0.5, pixel_size=2.0)
    assert summary["changed_area_m2"] == 2400.0


# --------------------------------------------------------------------------- #
# summarize_file
# --------------------------------------------------------------------------- #
def test_summarize_file_matches_stats(scene):
    b, _ = scene
    expected = farq.stats(read_all(b), bins=40)
    got = tiling.summarize_file(b, bins=40, block_size=256, n_jobs=2)
    for key in ("valid", "nan", "inf", "size", "min", "max", "range"):
        assert got[key] == expected[key], key
    for key in ("mean", "std", "variance"):
        assert got[key] == pytest.approx(expected[key], rel=1e-10), key
    np.testing.assert_array_equal(got["histogram"]["counts"], expected["histogram"]["counts"])
    np.testing.assert_array_equal(got["histogram"]["bin_edges"], expected["histogram"]["bin_edges"])
    assert got["percentages"]["valid"] == pytest.approx(expected["percentages"]["valid"])
    assert got["shape"] == (HEIGHT, WIDTH) and got["band"] == 1


def test_summarize_file_multiband_and_raw(tmp_path):
    rng = np.random.default_rng(7)
    data = rng.integers(0, 1000, size=(2, 300, 211), dtype=np.uint16)
    path = write_tif(tmp_path / "m.tif", data, nodata=0)
    stats = tiling.summarize_file(path, band=None, bins=16, block_size=64)
    assert [s["band"] for s in stats] == [1, 2]
    for i, s in enumerate(stats):
        ref = farq.stats(read_all(path, i + 1), bins=16)
        assert s["valid"] == ref["valid"] and s["nan"] == ref["nan"]
        assert s["mean"] == pytest.approx(ref["mean"], rel=1e-12)
        np.testing.assert_array_equal(s["histogram"]["counts"], ref["histogram"]["counts"])
    raw = tiling.summarize_file(path, band=2, masked=False, bins=None)
    assert raw["valid"] == data[1].size and raw["min"] == data[1].min()
    assert "histogram" not in raw
    assert raw["sum"] == float(data[1].sum(dtype=np.float64))


@pytest.mark.parametrize("anti", [False, True])
def test_detect_changes_file_regions_through_block_corners(tmp_path, anti):
    """A diagonal line crosses block corners; only 8-connectivity links it."""
    after = np.eye(16, dtype=np.float32)
    if anti:
        after = after[:, ::-1].copy()
    b = write_tif(tmp_path / "b.tif", np.zeros((16, 16), np.float32), tiled=False)
    a = write_tif(tmp_path / "a.tif", after, tiled=False)
    out = tmp_path / "out.tif"
    for connectivity, kept in ((8, 16), (4, 0)):
        summary = tiling.detect_changes_file(
            b, a, out, threshold=0.5, min_size=16, connectivity=connectivity, block_size=4
        )
        assert summary["changed_pixels"] == kept
    # a ring around a hole that spans four blocks: filled only with fill_holes
    ring = np.zeros((16, 16), np.float32)
    ring[2:14, 2:14] = 1
    ring[5:11, 5:11] = 0
    a = write_tif(tmp_path / "ring.tif", ring, tiled=False)
    summary = tiling.detect_changes_file(b, a, out, threshold=0.5, fill_holes=True, block_size=4)
    assert summary["changed_pixels"] == 144
    summary = tiling.detect_changes_file(b, a, out, threshold=0.5, fill_holes=35, block_size=4)
    assert summary["changed_pixels"] == 144 - 36


def test_detect_changes_file_memory_is_bounded(tmp_path):
    rng = np.random.default_rng(8)
    before = rng.random((2048, 2048), dtype=np.float32)
    after = before + (ndimage.gaussian_filter(rng.random((2048, 2048)), 4) > 0.5)
    b = write_tif(tmp_path / "b.tif", before)
    a = write_tif(tmp_path / "a.tif", after.astype(np.float32))
    del before, after
    tracemalloc.start()
    try:
        summary = tiling.detect_changes_file(
            b, a, tmp_path / "o.tif", sample_size=20_000, min_size=50, block_size=256
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert summary["changed_pixels"] > 0
    assert peak < 2048 * 2048 * 4 / 4, f"peak {peak / 2**20:.1f} MiB"


@pytest.mark.parametrize("crs", ["EPSG:4326", 4326, CRS.from_epsg(4326)])
def test_detect_changes_file_geographic_crs_raises_before_processing(tmp_path, crs):
    data = np.zeros((20, 30), dtype=np.float32)
    geo = from_origin(10.0, 50.0, 0.001, 0.001)
    b = write_tif(tmp_path / "b.tif", data, transform=geo, crs=crs, tiled=False)
    a = write_tif(tmp_path / "a.tif", data + 1, transform=geo, crs=crs, tiled=False)
    out = tmp_path / "o.tif"
    calls = []
    with pytest.raises(ValueError, match="geographic"):
        tiling.detect_changes_file(b, a, out, threshold=0.5, progress=lambda *x: calls.append(x))
    assert not out.exists() and calls == []
    summary = tiling.detect_changes_file(b, a, out, threshold=0.5, pixel_size=(80.0, 110.0))
    assert summary["changed_area_m2"] == 600 * 80.0 * 110.0
