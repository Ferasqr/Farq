"""Tests for farq.georef: GCP georeferencing, alignment and co-registration."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine, from_origin
from scipy import ndimage

import farq
from farq import georef
from farq.georef import (
    GCPResiduals,
    align,
    align_pair,
    apply_shift,
    coregister,
    gcp_residuals,
    georeference,
    has_gcps,
    make_gcps,
    pixel_area,
    pixel_size,
    read_gcps,
    rectify,
)

UTM = CRS.from_epsg(32633)
# 5 cm drone pixels in UTM 33N.
AFFINE = Affine(0.05, 0.0, 500000.0, 0.0, -0.05, 4000000.0)
H, W = 80, 120


def gcps_from_affine(affine: Affine, height: int = H, width: int = W, n: int = 3):
    """Exact GCPs on an n x n grid of pixel positions (GDAL corner convention)."""
    rows = np.linspace(0, height, n)
    cols = np.linspace(0, width, n)
    pixels = [(r, c) for r in rows for c in cols]
    coords = [
        (affine.a * c + affine.b * r + affine.c, affine.d * c + affine.e * r + affine.f)
        for r, c in pixels
    ]
    return make_gcps(pixels, coords)


def compose(t1: Affine, t2: Affine) -> Affine:
    """Matrix product ``t1 @ t2`` (portable across affine versions)."""
    m = np.asarray(t1, dtype=float).reshape(3, 3) @ np.asarray(t2, dtype=float).reshape(3, 3)
    return Affine(*m.ravel()[:6])


def ramp(height: int, width: int, transform: Affine) -> np.ndarray:
    """A linear function of map coordinates sampled at pixel centres (float64)."""
    rows, cols = np.mgrid[0:height, 0:width] + 0.5
    x = transform.c + cols * transform.a + rows * transform.b
    y = transform.f + cols * transform.d + rows * transform.e
    return (x - 500000.0) + 2.0 * (4000000.0 - y)


def write_gcp_tif(path, data: np.ndarray, gcps, crs=UTM, nodata=None) -> None:
    data3 = data[np.newaxis] if data.ndim == 2 else data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data3.shape[2],
            height=data3.shape[1],
            count=data3.shape[0],
            dtype=data3.dtype,
            nodata=nodata,
        ) as dst:
            dst.write(data3)
            dst.gcps = (gcps, crs)


def gcp_meta(data: np.ndarray, gcps, crs=UTM) -> dict:
    """Metadata as farq.read returns it for a GCP-only raster."""
    return {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "nodata": None,
        "width": data.shape[-1],
        "height": data.shape[-2],
        "count": 1 if data.ndim == 2 else data.shape[0],
        "crs": None,
        "transform": Affine.identity(),
        "gcps": gcps,
        "gcps_crs": crs,
    }


def geo_meta(data: np.ndarray, transform: Affine, crs=UTM, nodata=None) -> dict:
    return {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "nodata": nodata,
        "width": data.shape[-1],
        "height": data.shape[-2],
        "count": 1 if data.ndim == 2 else data.shape[0],
        "crs": crs,
        "transform": transform,
    }


@pytest.fixture
def rgb() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(1, 256, size=(3, H, W), dtype=np.uint8)


def smooth_texture(shape, seed=1, sigma=2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(size=shape), sigma).astype(np.float32)


# ------------------------------------------------------------------------------ GCPs


class TestGCPs:
    def test_make_gcps(self):
        gcps = make_gcps([(0, 0), (10.5, 20)], [(1.0, 2.0), (3.0, 4.0)])
        assert len(gcps) == 2
        assert isinstance(gcps[0], GroundControlPoint)
        assert (gcps[1].row, gcps[1].col, gcps[1].z) == (10.5, 20.0, 0.0)
        assert [g.id for g in gcps] == ["1", "2"]
        gcps3 = make_gcps([(0, 0)], [(1, 2, 7)], ids=["A"])
        assert gcps3[0].z == 7.0 and gcps3[0].id == "A"

    @pytest.mark.parametrize(
        ("pixels", "coords"),
        [
            ([(0, 0)], [(1, 2), (3, 4)]),
            ([(0, 0, 0)], [(1, 2)]),
            ([(0, 0)], [(1,)]),
            ([(0, np.nan)], [(1, 2)]),
            (np.empty((0, 2)), np.empty((0, 2))),
        ],
    )
    def test_make_gcps_invalid(self, pixels, coords):
        with pytest.raises(ValueError):
            make_gcps(pixels, coords)

    def test_has_and_read_gcps(self, tmp_path, rgb):
        gcps = gcps_from_affine(AFFINE)
        path = tmp_path / "drone.tif"
        write_gcp_tif(path, rgb, gcps)
        assert has_gcps(path) and has_gcps(str(path))
        with rasterio.open(path) as ds:
            assert has_gcps(ds)
        read, crs = read_gcps(path)
        assert crs == UTM and len(read) == len(gcps)
        assert has_gcps({"gcps": gcps}) and not has_gcps({"transform": AFFINE})

        plain = tmp_path / "plain.tif"
        with rasterio.open(plain, "w", **geo_meta(rgb[0], AFFINE)) as dst:
            dst.write(rgb[:1])
        assert not has_gcps(plain)
        with pytest.raises(ValueError, match="no ground control points"):
            read_gcps(plain)
        with pytest.raises(TypeError):
            has_gcps(42)


# ------------------------------------------------------------------------- Residuals


class TestResiduals:
    def test_exact_gcps_have_zero_residuals(self):
        report = gcp_residuals(gcps_from_affine(AFFINE), order=1)
        assert isinstance(report, GCPResiduals)
        assert report.rmse < 1e-6 and report.max_error < 1e-6
        assert report.dof == 9 - 3
        assert report.pixel_size == pytest.approx(0.05)

    def test_corrupted_gcp_is_detected(self):
        gcps = gcps_from_affine(AFFINE)
        g = gcps[4]
        gcps[4] = GroundControlPoint(
            row=g.row, col=g.col, x=g.x + 2.0, y=g.y - 1.0, z=0.0, id="bad"
        )
        report = gcp_residuals(gcps, order=1)
        assert report.rmse > 0.1
        assert np.argmax(report.loo_errors) == 4
        assert report.loo_errors[4] == pytest.approx(math.hypot(2.0, 1.0), rel=1e-6)
        assert report.outliers()[0] == 4
        assert report.ids[4] == "bad"

    def test_higher_order_and_warning(self):
        gcps = gcps_from_affine(AFFINE, n=4)  # 16 GCPs
        assert gcp_residuals(gcps, order=3).rmse < 1e-6
        with pytest.warns(UserWarning, match="by construction"):
            report = gcp_residuals([gcps[0], gcps[3], gcps[12]], order=1)
        assert report.dof == 0

    @pytest.mark.parametrize(("order", "n"), [(1, 2), (2, 5), (3, 9)])
    def test_too_few_gcps(self, order, n):
        gcps = gcps_from_affine(AFFINE, n=4)[:n]
        with pytest.raises(ValueError, match=f"at least {georef._MIN_GCPS[order]} GCPs"):
            gcp_residuals(gcps, order=order)

    def test_collinear_gcps(self):
        gcps = make_gcps([(i, i) for i in range(5)], [(i, i) for i in range(5)])
        with pytest.raises(ValueError, match="collinear"):
            gcp_residuals(gcps)

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="order"):
            gcp_residuals(gcps_from_affine(AFFINE), order=4)


# ----------------------------------------------------------------------- Georeference


class TestGeoreference:
    def test_recovers_affine(self):
        data = ramp(H, W, AFFINE).astype(np.float32)
        out, meta = georeference(data, gcps_from_affine(AFFINE), "EPSG:32633")
        assert meta["crs"] == UTM
        assert meta["transform"].almost_equals(AFFINE, precision=1e-6)
        assert out.shape == (H, W) == (meta["height"], meta["width"])
        assert meta["dtype"] == "float32" and math.isnan(meta["nodata"])
        np.testing.assert_allclose(out[1:-1, 1:-1], data[1:-1, 1:-1], atol=1e-3)

    def test_rgb_uint8_and_resolution(self, rgb):
        out, meta = georeference(rgb, gcps_from_affine(AFFINE), UTM, resolution=0.1)
        assert out.shape == (3, H // 2, W // 2)
        assert meta["count"] == 3
        assert meta["transform"].a == pytest.approx(0.1)
        assert meta["transform"].e == pytest.approx(-0.1)
        # No nodata given: promoted to float32 so outside pixels can be NaN.
        assert out.dtype == np.float32

        out8, meta8 = georeference(
            rgb, gcps_from_affine(AFFINE), UTM, nodata=0, resampling="nearest"
        )
        assert out8.dtype == np.uint8 and meta8["nodata"] == 0
        np.testing.assert_array_equal(out8, rgb)

    def test_rotated_gcps_tps_and_order2(self):
        theta = math.radians(25)
        rot = compose(AFFINE, Affine.rotation(math.degrees(theta)))
        gcps = gcps_from_affine(rot, n=4)
        data = np.ones((H, W), dtype=np.float32)
        for kwargs in ({}, {"order": 2}, {"method": "tps"}):
            out, meta = georeference(data, gcps, UTM, **kwargs)
            t = meta["transform"]
            assert t.b == 0 and t.d == 0  # north-up
            left, bottom, right, top = rasterio.transform.array_bounds(
                meta["height"], meta["width"], t
            )
            xs, ys = [g.x for g in gcps], [g.y for g in gcps]
            assert left <= min(xs) + 0.1 and right >= max(xs) - 0.1
            assert bottom <= min(ys) + 0.1 and top >= max(ys) - 0.1
            # The rotated footprint leaves NaN corners; valid values stay exactly 1.
            assert np.isnan(out).any()
            assert np.nanmin(out) == pytest.approx(1.0) and np.nanmax(out) == pytest.approx(1.0)

    def test_dst_crs(self):
        out, meta = georeference(
            np.ones((H, W), np.float32), gcps_from_affine(AFFINE), UTM, dst_crs=4326
        )
        assert meta["crs"] == CRS.from_epsg(4326)
        assert np.isfinite(out).any()

    def test_invalid_inputs(self, rgb):
        gcps = gcps_from_affine(AFFINE)
        with pytest.raises(ValueError, match="crs is missing"):
            georeference(rgb, gcps, None)
        with pytest.raises(ValueError, match="at least 6 GCPs"):
            georeference(rgb, gcps[:5], UTM, order=2)
        with pytest.raises(ValueError, match="order applies"):
            georeference(rgb, gcps, UTM, method="tps", order=2)
        with pytest.raises(ValueError, match="method"):
            georeference(rgb, gcps, UTM, method="spline")
        with pytest.raises(ValueError, match="resampling"):
            georeference(rgb, gcps, UTM, resampling="magic")
        with pytest.raises(ValueError, match="resolution"):
            georeference(rgb, gcps, UTM, resolution=-1)
        with pytest.raises(ValueError, match="2D"):
            georeference(rgb[np.newaxis], gcps, UTM)
        with pytest.raises(ValueError, match="out of range"):
            georeference(rgb, gcps, UTM, nodata=300)
        with pytest.raises(TypeError):
            georeference(rgb, [(0, 0, 1, 2)], UTM)

    def test_nan_propagates(self):
        data = ramp(H, W, AFFINE).astype(np.float32)
        data[30:40, 50:60] = np.nan
        out, _ = georeference(data, gcps_from_affine(AFFINE), UTM)
        assert np.isnan(out[32:38, 52:58]).all()
        assert np.isfinite(out[5:20, 5:40]).all()


class TestRectify:
    def test_file_roundtrip(self, tmp_path, rgb):
        src = tmp_path / "raw.tif"
        dst = tmp_path / "rect.tif"
        write_gcp_tif(src, rgb, gcps_from_affine(AFFINE), nodata=0)
        out, _ = rectify(src, dst, resampling="nearest")
        assert out.dtype == np.uint8 and out.shape == (3, H, W)
        np.testing.assert_array_equal(out, rgb)
        with rasterio.open(dst) as ds:
            assert ds.transform.almost_equals(AFFINE, precision=1e-6)
            assert ds.crs == UTM and ds.count == 3
            np.testing.assert_array_equal(ds.read(), rgb)

    def test_band_selection_and_downsampling(self, tmp_path, rgb):
        src = tmp_path / "raw.tif"
        write_gcp_tif(src, rgb, gcps_from_affine(AFFINE))
        out, _ = rectify(str(src), bands=2, resolution=0.2, resampling="average")
        assert out.ndim == 2 and out.shape == (H // 4, W // 4)
        assert out.dtype == np.float32
        with pytest.raises(ValueError, match="out of range"):
            rectify(src, bands=[4])

    def test_user_supplied_gcps(self, tmp_path, rgb):
        plain = tmp_path / "plain.tif"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(
                plain, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8"
            ) as ds:
                ds.write(rgb)
        with pytest.raises(ValueError, match="no ground control points"):
            rectify(plain)
        out, meta = rectify(
            plain, gcps=gcps_from_affine(AFFINE), gcps_crs=32633, nodata=0, resampling="nearest"
        )
        np.testing.assert_array_equal(out, rgb)
        assert meta["transform"].almost_equals(AFFINE, precision=1e-6)


# ----------------------------------------------------------------------------- Align


class TestAlign:
    def test_subgrid_is_exact_crop(self):
        data = ramp(H, W, AFFINE)
        ref_t = compose(AFFINE, Affine.translation(10, 5))
        ref = {"crs": UTM, "transform": ref_t, "width": 50, "height": 40}
        out, meta = align(data, geo_meta(data, AFFINE), ref)
        assert out.shape == (40, 50)
        np.testing.assert_array_equal(out, data[5:45, 10:60])
        assert meta["transform"] == ref_t and meta["dtype"] == "float64"

    def test_partial_cover_is_nan(self):
        data = ramp(H, W, AFFINE)
        ref_t = compose(AFFINE, Affine.translation(-10, 0))
        ref = {"crs": UTM, "transform": ref_t, "width": 30, "height": 10}
        out, _ = align(data, geo_meta(data, AFFINE), ref)
        assert np.isnan(out[:, :10]).all()
        np.testing.assert_array_equal(out[:, 10:], data[:10, :20])

    def test_integer_nodata_kept(self, rgb):
        rgb = rgb.copy()
        rgb[:, 0, 0] = 0
        ref_t = compose(AFFINE, Affine.translation(-2, -2))
        ref = {"crs": UTM, "transform": ref_t, "width": 10, "height": 10}
        out, meta = align(rgb, geo_meta(rgb, AFFINE, nodata=0), ref)
        assert out.dtype == np.uint8 and meta["nodata"] == 0
        assert (out[:, :2, :] == 0).all() and (out[:, 2, 2] == 0).all()
        np.testing.assert_array_equal(out[:, 2:, 2:][:, 1:, 1:], rgb[:, 1:8, 1:8])

    def test_float_nodata_value_masked_when_given(self):
        data = ramp(H, W, AFFINE)
        data[0, 0] = -9999
        ref = {"crs": UTM, "transform": AFFINE, "width": 4, "height": 4}
        out, _ = align(data, geo_meta(data, AFFINE, nodata=-9999), ref, nodata=-9999)
        assert np.isnan(out[0, 0]) and np.isfinite(out[1:, 1:]).all()

    def test_resampled_to_coarser_grid(self):
        data = ramp(H, W, AFFINE)
        t2 = Affine(0.1, 0, 500000.3, 0, -0.1, 3999999.7)
        ref = {"crs": UTM, "transform": t2, "width": 40, "height": 25}
        out, _ = align(data, geo_meta(data, AFFINE), ref)
        np.testing.assert_allclose(out[1:-1, 1:-1], ramp(25, 40, t2)[1:-1, 1:-1], atol=1e-6)

    def test_gcp_input(self):
        data = ramp(H, W, AFFINE).astype(np.float32)
        meta = gcp_meta(data, gcps_from_affine(AFFINE))
        ref_t = compose(AFFINE, Affine.translation(20, 10))
        ref = {"crs": UTM, "transform": ref_t, "width": 40, "height": 30}
        out, _ = align(data, meta, ref)
        np.testing.assert_allclose(out, data[10:40, 20:60], atol=1e-3)
        # GCP metadata can also serve as the reference grid.
        out2, m2 = align(data, geo_meta(data, AFFINE), meta)
        assert m2["transform"].almost_equals(AFFINE, precision=1e-6)
        np.testing.assert_allclose(out2, data, atol=1e-6)

    def test_errors(self):
        data = np.ones((H, W), np.float32)
        ref = {"crs": UTM, "transform": AFFINE, "width": 4, "height": 4}
        with pytest.raises(ValueError, match="not georeferenced"):
            align(data, {"width": W, "height": H}, ref)
        with pytest.raises(ValueError, match="does not match"):
            align(data, geo_meta(np.ones((5, 5)), AFFINE), ref)
        with pytest.raises(ValueError, match="CRS is missing"):
            align(data, {**geo_meta(data, AFFINE), "crs": None, "transform": AFFINE}, ref)
        with pytest.raises(ValueError, match="width"):
            align(data, geo_meta(data, AFFINE), {"crs": UTM, "transform": AFFINE})


class TestAlignPair:
    @pytest.fixture
    def pair(self):
        # before: 5 cm, 120x80 at (500000, 4000000) -> x 500000..500006, y 3999996..4000000
        before = ramp(H, W, AFFINE)
        # after: 10 cm, offset by (1.03, -0.97) m, 70x50 -> x 500001.03..500008.03
        t_after = Affine(0.1, 0, 500001.03, 0, -0.1, 3999999.03)
        after = ramp(50, 70, t_after)
        return before, geo_meta(before, AFFINE), after, geo_meta(after, t_after), t_after

    def test_target_before(self, pair):
        before, bm, after, am, _ = pair
        b, a, meta = align_pair(before, bm, after, am)
        assert b.shape == a.shape == (meta["height"], meta["width"])
        t = meta["transform"]
        assert t.a == pytest.approx(0.05) and t.e == pytest.approx(-0.05)
        left, bottom, right, top = rasterio.transform.array_bounds(b.shape[0], b.shape[1], t)
        # Overlap x 500001.03..500006, y 3999996..3999999.03, snapped inwards to before grid.
        assert left == pytest.approx(500001.05) and right == pytest.approx(500006.0)
        assert top == pytest.approx(3999999.0) and bottom == pytest.approx(3999996.03, abs=0.05)
        assert bottom >= 3999996.0 - 1e-9
        # before is only cropped, never resampled.
        r0, c0 = round((AFFINE.f - top) / 0.05), round((left - AFFINE.c) / 0.05)
        np.testing.assert_array_equal(b, before[r0 : r0 + b.shape[0], c0 : c0 + b.shape[1]])
        # after is resampled; for a linear ramp bilinear is exact away from the edges.
        np.testing.assert_allclose(a[2:-2, 2:-2], b[2:-2, 2:-2], atol=1e-6)

    @pytest.mark.parametrize(("target", "res"), [("coarsest", 0.1), ("finest", 0.05)])
    def test_targets(self, pair, target, res):
        before, bm, after, am, t_after = pair
        b, a, meta = align_pair(before, bm, after, am, target=target)
        assert b.shape == a.shape
        assert meta["transform"].a == pytest.approx(res)
        if target == "coarsest":
            # Snapped onto the after grid: after is a pure crop.
            off = (meta["transform"].c - t_after.c) / 0.1
            assert off == pytest.approx(round(off), abs=1e-6)
        np.testing.assert_allclose(a[2:-2, 2:-2], b[2:-2, 2:-2], atol=1e-6)

    def test_explicit_resolution_and_after_target(self, pair):
        before, bm, after, am, _ = pair
        b, a, meta = align_pair(before, bm, after, am, target="after", resolution=0.25)
        assert meta["transform"].a == pytest.approx(0.25)
        assert b.shape == a.shape

    def test_rgb_and_gcp_input(self, rgb):
        before = rgb
        bm = gcp_meta(rgb, gcps_from_affine(AFFINE))
        t_after = compose(AFFINE, Affine.translation(30, 20))
        after = rgb[:, 20:, 30:].copy()
        am = geo_meta(after, t_after, nodata=0)
        b, a, meta = align_pair(before, bm, after, am, resampling="nearest")
        assert b.shape == a.shape == (3, H - 20, W - 30)
        np.testing.assert_array_equal(b[:, 1:-1, 1:-1], a[:, 1:-1, 1:-1].astype(np.float32))
        # Different output dtypes (float32 vs uint8) are promoted to float.
        assert b.dtype == a.dtype == np.float32 and math.isnan(meta["nodata"])

    def test_different_crs(self, pair):
        before, bm, after, am, _ = pair
        b, a, meta = align_pair(before, bm, after, am, dst_crs=32634)
        assert meta["crs"] == CRS.from_epsg(32634)
        assert b.shape == a.shape
        both = np.isfinite(b) & np.isfinite(a)
        assert both.sum() > 0.5 * b.size
        np.testing.assert_allclose(a[both], b[both], atol=0.25)

    def test_no_overlap(self, pair):
        before, bm, after, am, _ = pair
        far = {**am, "transform": Affine(0.1, 0, 600000, 0, -0.1, 4000000)}
        with pytest.raises(ValueError, match="do not overlap"):
            align_pair(before, bm, after, far)

    def test_bad_target(self, pair):
        before, bm, after, am, _ = pair
        with pytest.raises(ValueError, match="target"):
            align_pair(before, bm, after, am, target="middle")


# ------------------------------------------------------------------------ Coregister


class TestCoregister:
    @pytest.mark.parametrize("true_shift", [(3.4, -2.25), (-7.0, 5.6), (0.0, 0.0)])
    def test_recovers_subpixel_shift(self, true_shift):
        ref = smooth_texture((128, 160))
        moving = ndimage.shift(ref, true_shift, order=3, mode="reflect")
        dy, dx = coregister(ref, moving)
        assert dy == pytest.approx(-true_shift[0], abs=0.1)
        assert dx == pytest.approx(-true_shift[1], abs=0.1)

    @pytest.mark.parametrize("true_shift", [(12.3, -40.7), (3.4, -2.25), (-0.35, 0.7)])
    def test_recovers_shift_without_wraparound_content(self, true_shift):
        # Cropping the interior means content truly enters/leaves at the borders, as in
        # two real flights; the refinement pass removes the window bias.
        big = smooth_texture((400, 460), seed=5, sigma=3.0)
        moved = ndimage.shift(big, true_shift, order=3, mode="reflect")
        ref, moving = big[60:-60, 60:-60], moved[60:-60, 60:-60]
        shift = coregister(ref, moving, upsample=20)
        assert shift == pytest.approx((-true_shift[0], -true_shift[1]), abs=0.06)

    def test_classic_phase_correlation(self):
        ref = smooth_texture((128, 128), seed=7)
        moving = ndimage.shift(ref, (-5.0, 9.0), order=3, mode="reflect")
        assert coregister(ref, moving, whitening=1.0, refine=2) == pytest.approx(
            (5.0, -9.0), abs=0.15
        )

    def test_multiband_nan_and_apply(self):
        ref = smooth_texture((3, 128, 128), seed=3, sigma=(0, 2, 2))
        moving = ndimage.shift(ref, (0, 3.4, -2.25), order=3, mode="reflect")
        moving[:, 10:20, 10:20] = np.nan
        shift = coregister(ref, moving, upsample=20)
        assert shift == pytest.approx((-3.4, 2.25), abs=0.06)
        fixed = apply_shift(moving, shift, order=3)
        inner = (slice(None), slice(30, 100), slice(30, 100))
        before_err = np.nanmean(np.abs(moving[inner] - ref[inner]))
        after_err = np.nanmean(np.abs(fixed[inner] - ref[inner]))
        assert after_err < 0.1 * before_err
        assert np.isnan(fixed[:, 10:20, 10:20]).any()

    def test_errors(self):
        ref = smooth_texture((64, 64))
        with pytest.raises(ValueError, match="shapes differ"):
            coregister(ref, ref[:32])
        with pytest.raises(ValueError, match="constant"):
            coregister(np.ones((64, 64)), ref)
        with pytest.raises(ValueError, match="upsample"):
            coregister(ref, ref, upsample=0)
        with pytest.raises(ValueError, match="whitening"):
            coregister(ref, ref, whitening=2)
        with pytest.raises(ValueError, match="refine"):
            coregister(ref, ref, refine=-1)
        with pytest.raises(ValueError, match="valid pixels"):
            bad = ref.copy()
            bad[:60] = np.nan
            coregister(ref, bad)


class TestApplyShift:
    def test_edges_are_nan_and_int_promoted(self):
        data = np.arange(100, dtype=np.uint8).reshape(10, 10)
        out = apply_shift(data, (2, -1))
        assert out.dtype == np.float32
        assert np.isnan(out[:2]).all() and np.isnan(out[:, -1]).all()
        np.testing.assert_array_equal(out[2:, :-1], data[:-2, 1:])

    def test_float64_kept_and_validation(self):
        data = np.ones((8, 8))
        assert apply_shift(data, (0.5, 0.5)).dtype == np.float64
        with pytest.raises(ValueError):
            apply_shift(data, (np.nan, 0))
        with pytest.raises(ValueError):
            apply_shift(np.ones(5), (1, 1))


# -------------------------------------------------------------------- Pixel geometry


class TestPixelGeometry:
    def test_projected(self):
        meta = {"crs": UTM, "transform": from_origin(500000, 4000000, 0.05, 0.1)}
        assert pixel_size(meta) == pytest.approx((0.05, 0.1))
        assert pixel_area(meta) == pytest.approx(0.005)

    def test_rotated(self):
        t = compose(AFFINE, Affine.rotation(30))
        meta = {"crs": UTM, "transform": t}
        assert pixel_size(meta) == pytest.approx((0.05, 0.05))
        assert pixel_area(meta) == pytest.approx(0.0025)

    def test_gcps(self):
        meta = gcp_meta(np.zeros((H, W)), gcps_from_affine(AFFINE))
        assert pixel_size(meta) == pytest.approx((0.05, 0.05))
        assert pixel_area(meta) == pytest.approx(0.0025)

    def test_geographic_and_missing(self):
        meta = {"crs": CRS.from_epsg(4326), "transform": from_origin(15, 45, 1e-6, 1e-6)}
        assert pixel_size(meta) == pytest.approx((1e-6, 1e-6))
        with pytest.raises(ValueError, match="geographic"):
            pixel_area(meta)
        with pytest.raises(ValueError, match="CRS is missing"):
            pixel_area({"transform": AFFINE})
        with pytest.raises(ValueError, match="not georeferenced"):
            pixel_size({"crs": UTM})


# ----------------------------------------------------------------------------- Guards


class TestOutputSizeGuard:
    def test_absurd_resolution_rejected_before_allocation(self, rgb):
        # 1e-5 m pixels over a 6 m x 4 m image would be ~2.4e11 pixels per band.
        with pytest.raises(ValueError, match="safety limit"):
            georeference(rgb, gcps_from_affine(AFFINE), UTM, resolution=1e-5)

    def test_align_pair_resolution_guard(self):
        meta = {"crs": UTM, "transform": AFFINE, "width": W, "height": H}
        data = np.ones((H, W), np.float32)
        with pytest.raises(ValueError, match="FARQ_MAX_OUTPUT_PIXELS"):
            align_pair(data, meta, data, meta, resolution=1e-6)

    def test_rectify_failure_leaves_no_partial_file(self, tmp_path, rgb, monkeypatch):
        src = tmp_path / "raw.tif"
        write_gcp_tif(src, rgb, gcps_from_affine(AFFINE), nodata=0)
        monkeypatch.setenv("FARQ_MAX_OUTPUT_PIXELS", "100")
        with pytest.raises(ValueError, match="safety limit"):
            rectify(src, tmp_path / "rect.tif")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["raw.tif"]


class TestOutlierTest:
    """outliers() must find a single bad GCP anywhere without flagging clean ones."""

    PIX = np.array([[0, 0], [0, 900], [900, 0], [900, 900], [450, 300], [200, 700]], float)

    def _gcps(self, bad=None, offset=2.0, seed=0):
        rng = np.random.default_rng(seed)
        coords = np.column_stack(
            [500000 + self.PIX[:, 1] * 0.05, 4000000 - self.PIX[:, 0] * 0.05]
        ) + rng.normal(0, 0.01, (len(self.PIX), 2))
        if bad is not None:
            coords[bad, 0] += offset
        return farq.make_gcps(self.PIX, coords)

    def test_clean_gcps_not_flagged(self):
        assert farq.gcp_residuals(self._gcps()).outliers() == []

    @pytest.mark.parametrize("bad", range(6))
    def test_bad_gcp_ranked_first_and_refit_is_clean(self, bad):
        gcps = self._gcps(bad)
        flagged = farq.gcp_residuals(gcps).outliers()
        assert flagged and flagged[0] == bad
        rest = [g for i, g in enumerate(gcps) if i != bad]
        assert farq.gcp_residuals(rest).outliers() == []

    def test_false_alarm_rate_matches_alpha(self):
        rng = np.random.default_rng(3)
        pix = rng.uniform(0, 1000, (12, 2))
        alarms = 0
        for _ in range(400):
            coords = np.column_stack([pix[:, 1] * 0.05, -pix[:, 0] * 0.05])
            coords = coords + rng.normal(0, 0.02, coords.shape)
            alarms += bool(farq.gcp_residuals(farq.make_gcps(pix, coords)).outliers())
        assert alarms / 400 < 0.1

    def test_low_redundancy_returns_empty(self):
        gcps = self._gcps()[:4]
        report = farq.gcp_residuals(gcps)
        assert report.dof == 1
        assert np.isnan(report.studentized).all()
        assert report.outliers() == []

    def test_explicit_threshold_and_alpha_validation(self):
        report = farq.gcp_residuals(self._gcps(2, offset=5.0))
        assert report.outliers(threshold=1.0)[0] == 2
        with pytest.raises(ValueError, match="alpha"):
            report.outliers(alpha=1.5)
