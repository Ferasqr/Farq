"""Tests for farq.elevation: DoD, terrain derivatives, co-registration and volumes."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

from farq.elevation import (
    DEMCoregistration,
    StockpileResult,
    VerticalOffset,
    VolumeResult,
    aspect,
    coregister_dem,
    elevation_change,
    hillshade,
    level_of_detection,
    shift_dem,
    significant_change,
    slope,
    stockpile_volume,
    vertical_offset,
    volume_change,
)

UTM = CRS.from_epsg(32633)
X0, Y0 = 500000.0, 4000000.0


def _meta(res: float | tuple[float, float], shape: tuple[int, int], crs=UTM) -> dict:
    xres, yres = (res, res) if np.isscalar(res) else res
    return {
        "crs": crs,
        "transform": from_origin(X0, Y0, xres, yres),
        "height": shape[0],
        "width": shape[1],
    }


def _coords(shape: tuple[int, int], res: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre map coordinates relative to the grid origin (x east, y north)."""
    rows, cols = np.indices(shape, dtype=np.float64)
    return (cols + 0.5) * res, -(rows + 0.5) * res


# --------------------------------------------------------------------------------------
# elevation_change
# --------------------------------------------------------------------------------------


class TestElevationChange:
    def test_after_minus_before_with_nan(self):
        before = np.array([[10.0, 12.0, np.nan], [5.0, np.inf, 1.0]])
        after = np.array([[10.5, 11.0, 3.0], [np.nan, 2.0, 1.0]])
        before_copy, after_copy = before.copy(), after.copy()
        dod = elevation_change(before, after)
        np.testing.assert_array_equal(dod, np.array([[0.5, -1.0, np.nan], [np.nan, np.nan, 0.0]]))
        np.testing.assert_array_equal(before, before_copy)  # inputs untouched
        np.testing.assert_array_equal(after, after_copy)

    def test_nodata_masked_and_int(self):
        before = np.array([[-9999, 10], [20, 30]], dtype=np.int16)
        after = np.ma.masked_array(
            np.array([[1, 12], [25, 0]], dtype=np.int16), mask=[[0, 0], [0, 1]]
        )
        dod = elevation_change(before, after, nodata=-9999)
        assert dod.dtype == np.float32
        np.testing.assert_array_equal(dod, [[np.nan, 2.0], [5.0, np.nan]])

    def test_single_band_stack_and_dtype(self):
        before = np.zeros((1, 3, 3), dtype=np.float32)
        after = np.ones((3, 3), dtype=np.float64)
        dod = elevation_change(before, after)
        assert dod.shape == (3, 3) and dod.dtype == np.float64

    def test_errors(self):
        with pytest.raises(ValueError, match="same shape"):
            elevation_change(np.zeros((3, 3)), np.zeros((3, 4)))
        with pytest.raises(ValueError, match="2-D"):
            elevation_change(np.zeros((2, 3, 3)), np.zeros((2, 3, 3)))
        with pytest.raises(TypeError):
            elevation_change([[1.0]], np.zeros((1, 1)))
        with pytest.raises(TypeError, match="dtype"):
            elevation_change(np.zeros((2, 2), bool), np.zeros((2, 2), bool))


# --------------------------------------------------------------------------------------
# Terrain derivatives
# --------------------------------------------------------------------------------------


class TestSlopeAspect:
    @pytest.mark.parametrize("res", [2.0, (1.0, 3.0)])
    def test_plane(self, res):
        shape = (12, 15)
        xres, yres = (res, res) if np.isscalar(res) else res
        rows, cols = np.indices(shape, dtype=np.float64)
        x, y = (cols + 0.5) * xres, -(rows + 0.5) * yres
        dem = 100.0 + 0.1 * x + 0.2 * y  # rises to the east and north
        meta = _meta(res, shape)
        expected = math.degrees(math.atan(math.hypot(0.1, 0.2)))
        s = slope(dem, meta)
        np.testing.assert_allclose(s, expected, atol=1e-9)  # edges included
        np.testing.assert_allclose(slope(dem, meta, units="percent"), 100 * math.hypot(0.1, 0.2))
        np.testing.assert_allclose(slope(dem, meta, units="radians"), math.radians(expected))
        # Facing down the gradient: towards south-west, atan2(-0.1, -0.2).
        expected_aspect = math.degrees(math.atan2(-0.1, -0.2)) % 360
        np.testing.assert_allclose(aspect(dem, meta), expected_aspect, atol=1e-9)

    @pytest.mark.parametrize(
        ("dem", "expected"),
        [
            (np.add.outer(np.arange(5.0), np.zeros(5)), 0.0),  # rows go south: lower north
            (np.add.outer(np.zeros(5), -np.arange(5.0)), 90.0),
            (np.add.outer(-np.arange(5.0), np.zeros(5)), 180.0),
            (np.add.outer(np.zeros(5), np.arange(5.0)), 270.0),
        ],
    )
    def test_aspect_cardinal(self, dem, expected):
        np.testing.assert_allclose(aspect(dem, 1.0), expected)

    def test_flat_aspect_nan(self):
        assert np.isnan(aspect(np.zeros((4, 4)), 1.0)).all()
        np.testing.assert_array_equal(slope(np.zeros((4, 4)), 1.0), 0.0)

    def test_rotated_transform(self):
        # A grid rotated by 30 degrees: slope and aspect must follow map coordinates.
        theta = math.radians(30)
        t = Affine(math.cos(theta), math.sin(theta), X0, math.sin(theta), -math.cos(theta), Y0)
        rows, cols = np.indices((10, 10), dtype=np.float64)
        x = t.a * cols + t.b * rows
        dem = 0.3 * x  # rises east only
        meta = {"crs": UTM, "transform": t, "height": 10, "width": 10}
        np.testing.assert_allclose(slope(dem, meta), math.degrees(math.atan(0.3)), atol=1e-9)
        np.testing.assert_allclose(aspect(dem, meta), 270.0, atol=1e-9)

    def test_nan_aware_edges(self):
        x, _ = _coords((6, 6), 1.0)
        dem = 0.5 * x
        dem[2, 2] = np.nan
        s = slope(dem, 1.0)
        assert np.isnan(s[2, 2])
        valid = np.isfinite(dem)
        # Neighbours of the hole keep an exact slope from one-sided differences.
        np.testing.assert_allclose(s[valid], math.degrees(math.atan(0.5)))
        isolated = np.full((3, 3), np.nan)
        isolated[1, 1] = 5.0
        assert np.isnan(slope(isolated, 1.0)).all()

    def test_feet_crs_converted_to_metres(self):
        # EPSG:2263 has US survey feet: a 1 ft pixel is 0.3048 m, so z (m) rising 0.3048 m
        # per pixel is a 45 degree slope.
        meta = {"crs": CRS.from_epsg(2263), "transform": from_origin(0, 0, 1, 1)}
        dem = np.add.outer(np.zeros(4), np.arange(4.0)) * 0.30480060960121924
        np.testing.assert_allclose(slope(dem, meta), 45.0)

    def test_errors(self):
        dem = np.zeros((4, 4))
        with pytest.raises(ValueError, match="units"):
            slope(dem, 1.0, units="grad")
        with pytest.raises(ValueError, match="geographic"):
            slope(dem, _meta(0.0001, (4, 4), crs=CRS.from_epsg(4326)))
        with pytest.raises(ValueError, match="does not match"):
            slope(dem, _meta(1.0, (5, 4)))
        with pytest.raises(ValueError, match="positive"):
            aspect(dem, -1.0)


class TestHillshade:
    def test_flat_and_range(self):
        np.testing.assert_allclose(hillshade(np.zeros((3, 3)), 1.0, altitude=30.0), 0.5)
        x, y = _coords((20, 20), 1.0)
        dem = 5 * np.sin(x / 3) + np.cos(y / 4)
        hs = hillshade(dem, 1.0)
        assert np.nanmin(hs) >= 0 and np.nanmax(hs) <= 1

    def test_facing_the_sun(self):
        x, y = _coords((5, 5), 1.0)
        facing_nw = x - y  # descends towards the west and north, where the sun is
        facing_se = y - x
        lit = hillshade(facing_nw, 1.0)
        shaded = hillshade(facing_se, 1.0)
        assert (lit > shaded).all()
        # Slope facing the sun with slope = 90 - altitude: light hits perpendicularly.
        steep = (x - y) / math.sqrt(2)  # gradient magnitude 1 -> 45 degree slope
        np.testing.assert_allclose(hillshade(steep, 1.0, altitude=45.0), 1.0)
        np.testing.assert_allclose(
            hillshade(steep, 1.0, azimuth=135.0, altitude=45.0), 0.0, atol=1e-12
        )

    def test_nan_and_errors(self):
        dem = np.zeros((3, 3))
        dem[0, 0] = np.nan
        assert np.isnan(hillshade(dem, 1.0)[0, 0])
        with pytest.raises(ValueError, match="altitude"):
            hillshade(dem, 1.0, altitude=95)
        with pytest.raises(ValueError, match="z_factor"):
            hillshade(dem, 1.0, z_factor=0)


# --------------------------------------------------------------------------------------
# Vertical offset and Nuth & Kääb co-registration
# --------------------------------------------------------------------------------------


class TestVerticalOffset:
    def test_recovers_offset_with_outliers(self, rng):
        before = rng.normal(100, 5, (100, 100))
        after = before + 0.15 + rng.normal(0, 0.03, before.shape)
        after[:30, :30] += 4.0  # 9 % real change
        after[50, 50] = np.nan
        for method in ("median", "nmad_trimmed"):
            result = vertical_offset(before, after, method=method)
            assert isinstance(result, VerticalOffset)
            offset, nmad = result
            assert offset == pytest.approx(0.15, abs=0.01)
            assert nmad == pytest.approx(0.03, abs=0.01)
        # The trimmed mean is not pulled by the outliers.
        trimmed = vertical_offset(before, after, method="nmad_trimmed").offset
        assert trimmed == pytest.approx(0.15, abs=0.002)

    def test_stable_mask(self):
        before = np.zeros((20, 20))
        after = np.full((20, 20), 2.0)  # most of the area changed by +2
        after[:, :5] = 0.1
        stable = np.zeros((20, 20), bool)
        stable[:, :5] = True
        assert vertical_offset(before, after, stable_mask=stable).offset == pytest.approx(0.1)
        assert vertical_offset(before, after).offset == pytest.approx(2.0)

    def test_errors(self):
        with pytest.raises(ValueError, match="valid stable pixels"):
            vertical_offset(np.zeros((5, 5)), np.zeros((5, 5)))
        with pytest.raises(ValueError, match="method"):
            vertical_offset(np.zeros((20, 20)), np.zeros((20, 20)), method="mean")
        with pytest.raises(ValueError, match="shape"):
            vertical_offset(np.zeros((20, 20)), np.zeros((20, 20)), stable_mask=np.ones(3, bool))


def _hilly(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        30 * np.sin(x / 57.0) * np.cos(y / 43.0)
        + 20 * np.exp(-2 * ((x - 200) ** 2 + (y + 150) ** 2) / 80**2)
        + 10 * np.sin((x + y) / 31.0)
        + 0.02 * x
    )


class TestCoregisterDEM:
    shape = (200, 240)
    res = 2.0

    def _pair(self, shift, rng, noise=0.1):
        x, y = _coords(self.shape, self.res)
        sx, sy, sz = shift
        ref = _hilly(x, y)
        dem = _hilly(x - sx, y - sy) + sz + rng.normal(0, noise, self.shape)
        return ref, dem

    @pytest.mark.parametrize("shift", [(3.1, -1.7, 0.8), (-2.4, 4.3, -0.5)])
    def test_recovers_known_shift(self, rng, shift):
        ref, dem = self._pair(shift, rng)
        works = np.zeros(self.shape, bool)
        works[50:70, 50:90] = True
        dem[works] += 5.0  # real change, excluded through stable_mask
        result = coregister_dem(ref, dem, _meta(self.res, self.shape), stable_mask=~works)
        assert isinstance(result, DEMCoregistration)
        assert result.converged
        assert result.dx == pytest.approx(shift[0], abs=0.1)
        assert result.dy == pytest.approx(shift[1], abs=0.1)
        assert result.dz == pytest.approx(shift[2], abs=0.05)
        assert result.nmad_after < 0.15 < result.nmad_before
        # The corrected DEM matches the reference away from the edges and the works.
        dod = elevation_change(ref, result.dem)
        core = (slice(10, -10), slice(10, -10))
        assert np.nanmedian(np.abs(dod[core][~works[core]])) < 0.1
        assert np.nanmedian(dod[works]) == pytest.approx(5.0, abs=0.2)
        # shift_dem reproduces the corrected DEM.
        again = shift_dem(dem, _meta(self.res, self.shape), result.dx, result.dy, result.dz)
        np.testing.assert_allclose(again, result.dem, equal_nan=True)

    def test_planar_terrain_rejected(self):
        x, y = _coords((50, 50), 1.0)
        ref = 0.2 * x + 0.1 * y
        with pytest.raises(ValueError, match="aspects"):
            coregister_dem(ref, ref + 1.0, 1.0)

    def test_flat_terrain_rejected(self):
        with pytest.raises(ValueError, match="sloping terrain"):
            coregister_dem(np.zeros((50, 50)), np.ones((50, 50)), 1.0)

    def test_no_shift(self, rng):
        ref, dem = self._pair((0.0, 0.0, 0.0), rng, noise=0.0)
        result = coregister_dem(ref, dem, self.res)
        assert result.iterations == 1
        assert abs(result.dx) < 1e-6 and abs(result.dy) < 1e-6 and abs(result.dz) < 1e-9

    def test_parameter_errors(self):
        dem = np.zeros((20, 20))
        with pytest.raises(ValueError, match="max_iterations"):
            coregister_dem(dem, dem, 1.0, max_iterations=0)
        with pytest.raises(ValueError, match="slope"):
            coregister_dem(dem, dem, 1.0, min_slope=50, max_slope=10)
        with pytest.raises(ValueError, match="same shape"):
            coregister_dem(dem, np.zeros((20, 21)), 1.0)
        with pytest.raises(ValueError, match="finite"):
            shift_dem(dem, 1.0, np.nan, 0.0)


# --------------------------------------------------------------------------------------
# Level of detection
# --------------------------------------------------------------------------------------


class TestLevelOfDetection:
    def test_scalar(self):
        assert level_of_detection(0.05) == pytest.approx(1.959964 * 0.05 * math.sqrt(2))
        assert level_of_detection(0.03, 0.04) == pytest.approx(1.959964 * 0.05)
        assert level_of_detection(0.1, 0) == pytest.approx(0.1959964)
        assert level_of_detection(0.1, 0, confidence=0.6827) == pytest.approx(0.1, rel=1e-3)
        assert isinstance(level_of_detection(0.05), float)

    def test_array(self):
        s1 = np.array([[0.03, np.nan], [0.0, 0.1]])
        lod = level_of_detection(s1, 0.04)
        expected = 1.959964 * np.sqrt(s1**2 + 0.04**2)
        np.testing.assert_allclose(lod, expected, rtol=1e-6)
        assert np.isnan(lod[0, 1])

    def test_errors(self):
        with pytest.raises(ValueError, match="confidence"):
            level_of_detection(0.1, confidence=1.0)
        with pytest.raises(ValueError, match=">= 0"):
            level_of_detection(-0.1)
        with pytest.raises(ValueError, match=">= 0"):
            level_of_detection(np.array([0.1, -0.1]))
        with pytest.raises(ValueError, match="same shape"):
            level_of_detection(np.zeros(3), np.zeros(4))
        with pytest.raises(TypeError):
            level_of_detection("0.1")

    def test_significant_change(self):
        dod = np.array([[0.05, -0.3, 0.2, np.nan], [-0.1, 0.1, 0.11, -0.11]])
        out = significant_change(dod, 0.1)
        np.testing.assert_array_equal(out, [[0, -0.3, 0.2, np.nan], [0, 0, 0.11, -0.11]])
        lod = np.array([[0.5, 0.1, np.nan, 0.0], [0.0, 0.0, 0.0, 0.0]])
        out = significant_change(dod, lod)
        np.testing.assert_array_equal(out, [[0, -0.3, np.nan, np.nan], dod[1]])
        np.testing.assert_array_equal(dod[0, :3], [0.05, -0.3, 0.2])  # input untouched


# --------------------------------------------------------------------------------------
# volume_change
# --------------------------------------------------------------------------------------


class TestVolumeChange:
    def test_exact_cut_and_fill(self):
        dod = np.zeros((10, 10))
        dod[:2, :] = 0.5  # 20 px fill
        dod[5, :4] = -2.0  # 4 px cut
        dod[9, 9] = np.nan
        meta = _meta(0.5, (10, 10))
        v = volume_change(dod, meta)
        assert isinstance(v, VolumeResult)
        assert v.pixel_area_m2 == pytest.approx(0.25)
        assert v.fill_m3 == pytest.approx(20 * 0.5 * 0.25)
        assert v.cut_m3 == pytest.approx(4 * 2.0 * 0.25)
        assert v.net_m3 == pytest.approx(2.5 - 2.0)
        assert v.fill_area_m2 == pytest.approx(5.0)
        assert v.cut_area_m2 == pytest.approx(1.0)
        assert v.valid_area_m2 == pytest.approx(99 * 0.25)
        assert v.unchanged_area_m2 == pytest.approx(75 * 0.25)
        assert v.nodata_area_m2 == pytest.approx(0.25)
        assert v.uncertainty_m3 is None
        json.dumps(v.to_dict())

    def test_lod_and_mask(self):
        dod = np.array([[0.05, 0.3, -0.08, -0.5], [1.0, 1.0, 1.0, 1.0]])
        v = volume_change(dod, 1.0, lod=0.1)
        assert v.fill_m3 == pytest.approx(4.3)
        assert v.cut_m3 == pytest.approx(0.5)
        assert v.unchanged_area_m2 == pytest.approx(2.0)
        roi = np.array([[True, True, True, True], [False] * 4])
        v = volume_change(dod, 1.0, lod=0.1, mask=roi)
        assert v.fill_m3 == pytest.approx(0.3) and v.cut_m3 == pytest.approx(0.5)
        assert v.valid_area_m2 == pytest.approx(4.0)
        # A per-pixel LoD; NaN LoD pixels count as nodata.
        lod = np.array([[0.0, 0.5, 0.0, np.nan], [0.0] * 4])
        v = volume_change(dod, 1.0, lod=lod)
        assert v.fill_m3 == pytest.approx(4.05) and v.cut_m3 == pytest.approx(0.08)
        assert v.nodata_area_m2 == pytest.approx(1.0)

    def test_uncorrelated_uncertainty(self):
        dod = np.zeros((10, 10))
        dod[:4] = 1.0  # 40 fill px
        dod[-1] = -1.0  # 10 cut px
        v = volume_change(dod, 2.0, sigma=0.1)
        assert v.fill_uncertainty_m3 == pytest.approx(0.1 * 4.0 * math.sqrt(40))
        assert v.cut_uncertainty_m3 == pytest.approx(0.1 * 4.0 * math.sqrt(10))
        assert v.uncertainty_m3 == pytest.approx(0.1 * 4.0 * math.sqrt(100))
        v = volume_change(dod, 2.0, sigma=0.1, lod=0.5)
        assert v.uncertainty_m3 == pytest.approx(0.1 * 4.0 * math.sqrt(50))
        sigma = np.full((10, 10), 0.1)
        sigma[0, 0] = 0.3
        v = volume_change(dod, 2.0, sigma=sigma)
        expected = 4.0 * math.sqrt(39 * 0.01 + 0.09)
        assert v.fill_uncertainty_m3 == pytest.approx(expected)

    def test_correlated_uncertainty(self):
        dod = np.ones((100, 100))  # 10 000 m² at 1 m pixels
        area = 1e4
        # Range much larger than the area: nearly fully correlated.
        L = 1000.0
        r = math.sqrt(area / math.pi)
        expected = area * 0.1 * math.sqrt(1 - r / L + (r / L) ** 3 / 5)
        v = volume_change(dod, 1.0, sigma=0.1, correlation_length=L)
        assert v.uncertainty_m3 == pytest.approx(expected)
        assert v.uncertainty_m3 > 0.9 * area * 0.1
        # Range smaller than the area: sigma * sqrt(pi L^2 / 5A).
        L = 20.0
        expected = area * 0.1 * math.sqrt(math.pi * L**2 / (5 * area))
        v = volume_change(dod, 1.0, sigma=0.1, correlation_length=L)
        assert v.uncertainty_m3 == pytest.approx(expected)
        # Never below the uncorrelated estimate.
        v = volume_change(dod, 1.0, sigma=0.1, correlation_length=0.01)
        assert v.uncertainty_m3 == pytest.approx(0.1 * math.sqrt(1e4))

    def test_geometry_errors(self):
        dod = np.ones((4, 4))
        with pytest.raises(ValueError, match="geographic"):
            volume_change(dod, _meta(1e-5, (4, 4), crs=CRS.from_epsg(4326)))
        with pytest.raises(ValueError, match="geographic"):
            volume_change(dod, _meta(1e-5, (4, 4), crs="EPSG:4326"))
        with pytest.raises(ValueError, match="no 'crs'"):
            volume_change(dod, {"transform": from_origin(0, 0, 1, 1)})
        with pytest.raises(ValueError, match="no geotransform"):
            volume_change(dod, {"transform": Affine.identity(), "crs": None})
        gcps = [GroundControlPoint(row=0, col=0, x=1, y=1, z=0)]
        with pytest.raises(ValueError, match="no geotransform"):
            volume_change(dod, {"transform": Affine.identity(), "crs": UTM, "gcps": gcps})
        with pytest.raises(ValueError, match="GCPs only"):
            volume_change(dod, {"gcps": gcps, "gcps_crs": UTM})
        with pytest.raises(TypeError, match="meta"):
            volume_change(dod, "EPSG:32633")
        with pytest.raises(ValueError, match="positive"):
            volume_change(dod, 0.0)

    def test_parameter_errors(self):
        dod = np.ones((4, 4))
        with pytest.raises(ValueError, match="requires sigma"):
            volume_change(dod, 1.0, correlation_length=10)
        with pytest.raises(ValueError, match="correlation_length"):
            volume_change(dod, 1.0, sigma=0.1, correlation_length=-1)
        with pytest.raises(ValueError, match=">= 0"):
            volume_change(dod, 1.0, lod=-0.1)
        with pytest.raises(ValueError, match="shape"):
            volume_change(dod, 1.0, sigma=np.ones((3, 3)))
        with pytest.raises(ValueError, match="shape"):
            volume_change(dod, 1.0, mask=np.ones((3, 3), bool))

    def test_meta_variants(self):
        dod = np.ones((4, 4))
        expected = 16 * 0.5 * 2.0
        assert volume_change(dod, (0.5, 2.0)).fill_m3 == pytest.approx(expected)
        assert volume_change(dod, from_origin(0, 0, 0.5, 2.0)).fill_m3 == pytest.approx(expected)
        meta = _meta((0.5, 2.0), (4, 4))
        assert volume_change(dod, meta).fill_m3 == pytest.approx(expected)
        feet = {"crs": CRS.from_epsg(2263), "transform": from_origin(0, 0, 1, 1)}
        assert volume_change(dod, feet).pixel_area_m2 == pytest.approx(0.3048006**2)


# --------------------------------------------------------------------------------------
# stockpile_volume
# --------------------------------------------------------------------------------------


def _cone(shape, res, radius, height, centre):
    x, y = _coords(shape, res)
    r = np.hypot(x - centre[0], y - centre[1])
    return np.clip(height * (1 - r / radius), 0, None), r


class TestStockpileVolume:
    shape = (240, 240)
    res = 0.25

    def test_cone_on_flat_ground(self):
        cone, r = _cone(self.shape, self.res, 20.0, 10.0, (30.0, -30.0))
        dem = 100.0 + cone
        mask = r <= 20.5
        meta = _meta(self.res, self.shape)
        analytic = math.pi * 20.0**2 * 10.0 / 3.0
        for base in ("plane", "lowest", "mean", 100.0):
            result = stockpile_volume(dem, mask, meta, base=base)
            assert isinstance(result, StockpileResult)
            assert result.volume_m3 == pytest.approx(analytic, rel=5e-3)
            assert result.below_base_m3 == pytest.approx(0.0, abs=1e-6)
            assert result.max_height_m == pytest.approx(10.0, abs=0.1)
            assert result.base_elevation_m == pytest.approx(100.0)
        json.dumps(result.to_dict())

    def test_cone_on_tilted_ground(self):
        x, y = _coords(self.shape, self.res)
        ground = 50.0 + 0.05 * x - 0.03 * y
        cone, r = _cone(self.shape, self.res, 20.0, 10.0, (30.0, -30.0))
        dem = ground + cone
        mask = r <= 20.5
        result = stockpile_volume(dem, mask, _meta(self.res, self.shape), base="plane")
        analytic = math.pi * 20.0**2 * 10.0 / 3.0
        assert result.volume_m3 == pytest.approx(analytic, rel=5e-3)
        assert result.base_rmse_m == pytest.approx(0.0, abs=1e-9)
        assert result.base_slope_deg == pytest.approx(
            math.degrees(math.atan(math.hypot(0.05, 0.03)))
        )
        # A horizontal base at the lowest toe point overestimates on sloping ground.
        lowest = stockpile_volume(dem, mask, _meta(self.res, self.shape), base="lowest")
        assert lowest.volume_m3 > 1.2 * analytic
        assert lowest.base_rmse_m > 0.5

    def test_box_on_tilted_ground_exact(self):
        shape = (40, 50)
        x, y = _coords(shape, 0.5)
        dem = 10.0 + 0.1 * x + 0.2 * y
        box = np.zeros(shape, bool)
        box[10:30, 15:35] = True
        dem[box] += 3.0
        result = stockpile_volume(dem, box, _meta(0.5, shape), ring_width=2)
        assert result.volume_m3 == pytest.approx(400 * 3.0 * 0.25)
        assert result.area_m2 == pytest.approx(100.0)
        assert result.net_m3 == pytest.approx(result.volume_m3)

    def test_array_base_nodata_and_voids(self):
        shape = (20, 20)
        ground = np.full(shape, 5.0)
        pile = np.zeros(shape, bool)
        pile[5:15, 5:15] = True
        dem = ground.copy()
        dem[pile] += 1.0
        dem[6, 6] = np.nan  # a hole in the survey
        dem[10, 10] = 4.0  # a pit 1 m below the base
        result = stockpile_volume(dem, pile, 1.0, base=ground)
        assert result.base == "surface"
        assert result.volume_m3 == pytest.approx(98.0)
        assert result.below_base_m3 == pytest.approx(1.0)
        assert result.net_m3 == pytest.approx(97.0)
        assert result.missing_area_m2 == pytest.approx(1.0)
        assert result.area_m2 == pytest.approx(99.0)

    def test_uncertainty(self):
        shape = (20, 20)
        dem = np.full(shape, 5.0)
        pile = np.zeros(shape, bool)
        pile[5:15, 5:15] = True
        dem[pile] += 1.0
        result = stockpile_volume(dem, pile, 0.5, sigma=0.05)
        assert result.uncertainty_m3 == pytest.approx(0.05 * 0.25 * 10)

    def test_errors(self):
        shape = (20, 20)
        dem = np.full(shape, 5.0)
        pile = np.zeros(shape, bool)
        with pytest.raises(ValueError, match="empty"):
            stockpile_volume(dem, pile, 1.0)
        pile[5:10, 5:10] = True
        with pytest.raises(ValueError, match="base must be"):
            stockpile_volume(dem, pile, 1.0, base="median")
        with pytest.raises(ValueError, match="ring_width"):
            stockpile_volume(dem, pile, 1.0, ring_width=0)
        with pytest.raises(ValueError, match="geographic"):
            stockpile_volume(dem, pile, _meta(1e-5, shape, crs=CRS.from_epsg(4326)))
        everything = np.ones(shape, bool)
        with pytest.raises(ValueError, match="toe ring"):
            stockpile_volume(dem, everything, 1.0)
        line = np.zeros(shape, bool)
        line[0, :] = True
        line[1, :] = True
        dem2 = dem.copy()
        dem2[3:, :] = np.nan  # the ring is a single row: collinear
        with pytest.raises(ValueError, match="collinear"):
            stockpile_volume(dem2, line, 1.0)
        assert stockpile_volume(dem2, line, 1.0, base="mean").volume_m3 == pytest.approx(0.0)
        with pytest.raises(ValueError, match="finite"):
            stockpile_volume(dem, pile, 1.0, base=np.inf)
