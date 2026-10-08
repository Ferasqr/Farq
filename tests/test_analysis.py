"""Tests for farq.analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

import farq
from farq.analysis import (
    calculate_shape_metrics,
    get_water_bodies,
    water_change,
    water_stats,
)


def _square(n: int = 3, pad: int = 1) -> np.ndarray:
    mask = np.zeros((n + 2 * pad, n + 2 * pad), dtype=bool)
    mask[pad : pad + n, pad : pad + n] = True
    return mask


# --------------------------------------------------------------------------- #
# calculate_shape_metrics
# --------------------------------------------------------------------------- #
class TestShapeMetrics:
    def test_square(self):
        m = calculate_shape_metrics(_square(3))
        assert m["area"] == 9
        assert m["perimeter"] == 12
        assert m["compactness"] == pytest.approx(math.pi / 4)
        assert m["elongation"] == pytest.approx(1.0)
        assert m["orientation"] == pytest.approx(0.0)

    def test_square_touching_border(self):
        assert calculate_shape_metrics(np.ones((3, 3), bool))["perimeter"] == 12

    def test_single_pixel(self):
        m = calculate_shape_metrics(_square(1))
        assert m["perimeter"] == 4
        assert m["elongation"] == pytest.approx(1.0)

    def test_horizontal_rectangle(self):
        mask = np.zeros((6, 10), bool)
        mask[2:4, 1:7] = True  # 2 rows x 6 cols
        m = calculate_shape_metrics(mask)
        assert m["area"] == 12
        assert m["perimeter"] == 16
        assert m["elongation"] == pytest.approx(3.0)
        assert m["orientation"] == pytest.approx(0.0)

    def test_vertical_line(self):
        mask = np.zeros((7, 3), bool)
        mask[1:6, 1] = True
        m = calculate_shape_metrics(mask)
        assert m["elongation"] == pytest.approx(5.0)
        assert abs(m["orientation"]) == pytest.approx(90.0)

    def test_diagonal_orientation(self):
        rising = np.eye(6, dtype=bool)[::-1]  # bottom-left to top-right
        falling = np.eye(6, dtype=bool)
        assert calculate_shape_metrics(rising)["orientation"] == pytest.approx(45.0)
        assert calculate_shape_metrics(falling)["orientation"] == pytest.approx(-45.0)
        assert calculate_shape_metrics(rising)["elongation"] > 3

    def test_hole_adds_inner_perimeter(self):
        ring = _square(3)
        ring[2, 2] = False
        m = calculate_shape_metrics(ring)
        assert m["area"] == 8
        assert m["perimeter"] == 16

    def test_anisotropic_pixels(self):
        m = calculate_shape_metrics(_square(3), pixel_size=(2.0, 3.0))
        assert m["area"] == pytest.approx(54.0)
        # 6 horizontal edges of width 2 + 6 vertical edges of height 3
        assert m["perimeter"] == pytest.approx(30.0)
        # a 6 m wide x 9 m tall rectangle
        assert m["elongation"] == pytest.approx(1.5)
        assert abs(m["orientation"]) == pytest.approx(90.0)

    def test_scale_invariance(self):
        rng = np.random.default_rng(0)
        mask = rng.random((20, 20)) > 0.5
        a = calculate_shape_metrics(mask)
        b = calculate_shape_metrics(mask, pixel_size=30)
        assert b["area"] == pytest.approx(a["area"] * 900)
        assert b["perimeter"] == pytest.approx(a["perimeter"] * 30)
        for key in ("compactness", "elongation", "orientation"):
            assert b[key] == pytest.approx(a[key])

    def test_empty(self):
        m = calculate_shape_metrics(np.zeros((4, 4), bool))
        assert m == {
            "area": 0.0,
            "perimeter": 0.0,
            "compactness": 0.0,
            "elongation": 1.0,
            "orientation": 0.0,
        }

    def test_numeric_and_nan_masks(self):
        mask = _square(3).astype(float)
        mask[0, 0] = np.nan  # NaN is not water
        assert calculate_shape_metrics(mask)["area"] == 9
        assert calculate_shape_metrics(_square(3).astype(np.uint8))["area"] == 9

    def test_returns_python_floats(self):
        m = calculate_shape_metrics(_square(3))
        assert all(type(v) is float for v in m.values())

    @pytest.mark.parametrize(
        ("value", "exc"),
        [
            ([[1, 0]], TypeError),
            (np.array([]), ValueError),
            (np.zeros((2, 2, 2)), ValueError),
            (np.array([["a"]]), TypeError),
        ],
    )
    def test_invalid_input(self, value, exc):
        with pytest.raises(exc):
            calculate_shape_metrics(value)


# --------------------------------------------------------------------------- #
# water_stats
# --------------------------------------------------------------------------- #
def _two_bodies() -> np.ndarray:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True  # 9 px
    mask[7, 7] = True  # 1 px
    return mask


class TestWaterStats:
    def test_known_values(self):
        s = water_stats(_two_bodies(), pixel_size=30.0)
        px = 900 / 1e6
        assert s["num_water_bodies"] == 2
        assert s["total_area"] == pytest.approx(10 * px)
        assert s["coverage_percent"] == pytest.approx(10.0)
        # Regression: background bin used to be counted as an empty body.
        assert s["mean_body_size"] == pytest.approx(5 * px)
        assert s["largest_body"] == pytest.approx(9 * px)
        assert "shape_metrics" not in s

    def test_shapes(self):
        s = water_stats(_two_bodies(), pixel_size=10.0, calculate_shapes=True)
        sm = s["shape_metrics"]
        assert len(sm["body_metrics"]) == 2
        first = sm["body_metrics"][0]
        assert first["pixel_count"] == 9
        assert first["area"] == pytest.approx(9 * 100 / 1e6)
        assert first["perimeter"] == pytest.approx(12 * 10 / 1e3)  # km
        assert sm["mean_compactness"] == pytest.approx(math.pi / 4)
        assert sm["mean_elongation"] == pytest.approx(1.0)

    def test_empty_mask(self):
        s = water_stats(np.zeros((5, 5), bool), calculate_shapes=True)
        assert s["num_water_bodies"] == 0
        assert s["total_area"] == 0
        assert s["mean_body_size"] == 0
        assert s["largest_body"] == 0
        assert "shape_metrics" not in s

    def test_rectangular_pixels(self):
        s = water_stats(_two_bodies(), pixel_size=(10.0, 20.0))
        assert s["total_area"] == pytest.approx(10 * 200 / 1e6)

    @pytest.mark.parametrize("ps", [[10, 20], np.array([10.0, 20.0]), (np.int64(10), 20)])
    def test_pixel_size_sequences(self, ps):
        assert water_stats(_two_bodies(), pixel_size=ps)["total_area"] == pytest.approx(0.002)

    def test_numpy_scalar_pixel_size(self):
        s = water_stats(_two_bodies(), pixel_size=np.int32(30))
        assert s["total_area"] == pytest.approx(10 * 900 / 1e6)

    def test_connectivity(self):
        mask = np.eye(4, dtype=bool)
        assert water_stats(mask)["num_water_bodies"] == 4
        assert water_stats(mask, connectivity=2)["num_water_bodies"] == 1
        with pytest.raises(ValueError, match="connectivity"):
            water_stats(mask, connectivity=3)

    def test_nan_pixels_excluded_from_coverage(self):
        mask = np.zeros((4, 5))
        mask[0, :] = 1.0
        mask[1, :] = np.nan
        s = water_stats(mask, pixel_size=1)
        assert s["total_area"] == pytest.approx(5e-6)
        assert s["coverage_percent"] == pytest.approx(100 * 5 / 15)

    def test_does_not_modify_input(self):
        mask = _two_bodies().astype(float)
        copy = mask.copy()
        water_stats(mask, calculate_shapes=True)
        np.testing.assert_array_equal(mask, copy)

    @pytest.mark.parametrize(
        ("ps", "exc"),
        [(0, ValueError), (-30, ValueError), (float("nan"), ValueError), ((1, 2, 3), ValueError),
         ("30", TypeError), (True, TypeError), (None, TypeError), ((1, "a"), TypeError)],
    )  # fmt: skip
    def test_invalid_pixel_size(self, ps, exc):
        with pytest.raises(exc):
            water_stats(_two_bodies(), pixel_size=ps)

    def test_invalid_mask(self):
        with pytest.raises(TypeError):
            water_stats([[1, 0]])
        with pytest.raises(ValueError):
            water_stats(np.zeros((0, 3)))
        with pytest.raises(ValueError, match="2-D"):
            water_stats(np.zeros(5))


# --------------------------------------------------------------------------- #
# water_change
# --------------------------------------------------------------------------- #
class TestWaterChange:
    def test_known_change(self):
        m1 = np.zeros((4, 4), bool)
        m2 = np.zeros((4, 4), bool)
        m1[0, :] = True  # 4 px
        m2[0, :2] = True  # lost 2
        m2[3, :3] = True  # gained 3
        r = water_change(m1, m2, pixel_size=10)
        px = 100 / 1e6
        assert r["gained_area"] == pytest.approx(3 * px)
        assert r["lost_area"] == pytest.approx(2 * px)
        assert r["net_change"] == pytest.approx(px)
        assert r["change_percent"] == pytest.approx(25.0)
        assert r["change_mask"].dtype == np.int8
        assert r["change_mask"].sum() == 1
        assert r["change_mask"][0, 3] == -1
        assert r["change_mask"][3, 0] == 1
        assert r["stable_water"].sum() == 2
        assert r["stable_water"].dtype == bool

    def test_no_initial_water(self):
        empty = np.zeros((3, 3), bool)
        full = np.ones((3, 3), bool)
        assert water_change(empty, full)["change_percent"] == float("inf")
        assert water_change(empty, empty)["change_percent"] == 0.0

    def test_min_change_area_removes_small_patches(self):
        m1 = np.zeros((10, 10), bool)
        m2 = m1.copy()
        m2[0, 0] = True  # 1 px gain (noise)
        m2[5:8, 5:8] = True  # 9 px gain
        r = water_change(m1, m2, pixel_size=1, min_change_area=4)
        assert r["gained_area"] == pytest.approx(9e-6)
        assert r["change_mask"][0, 0] == 0
        assert r["change_mask"][6, 6] == 1
        # Regression: min_change_area below one pixel must keep everything.
        r = water_change(m1, m2, pixel_size=30, min_change_area=1)
        assert r["gained_area"] == pytest.approx(10 * 900 / 1e6)

    def test_min_change_area_thin_patch_kept(self):
        # A 1-px wide but long change is a real change and must survive.
        m1 = np.zeros((5, 20), bool)
        m2 = m1.copy()
        m2[2, :] = True
        r = water_change(m1, m2, pixel_size=1, min_change_area=10)
        assert r["gained_area"] == pytest.approx(20e-6)

    def test_nan_pixels_are_unknown(self):
        m1 = np.array([[1.0, 1.0, 0.0, np.nan]])
        m2 = np.array([[1.0, np.nan, 1.0, 1.0]])
        r = water_change(m1, m2, pixel_size=1000)
        assert r["lost_area"] == 0
        assert r["gained_area"] == pytest.approx(1.0)
        np.testing.assert_array_equal(r["change_mask"], [[0, 0, 1, 0]])
        np.testing.assert_array_equal(r["stable_water"], [[True, False, False, False]])
        assert r["change_percent"] == pytest.approx(100.0)

    def test_errors(self):
        with pytest.raises(ValueError, match="same shape"):
            water_change(np.zeros((2, 2)), np.zeros((3, 3)))
        with pytest.raises(TypeError):
            water_change(np.zeros((2, 2)), [[0, 0], [0, 0]])
        with pytest.raises(ValueError):
            water_change(np.zeros((2, 2)), np.zeros((2, 2)), min_change_area=-1)


# --------------------------------------------------------------------------- #
# get_water_bodies
# --------------------------------------------------------------------------- #
class TestGetWaterBodies:
    def test_labels_and_areas(self):
        labeled, bodies = get_water_bodies(_two_bodies(), pixel_size=30)
        assert labeled.dtype == np.int32
        assert labeled.max() == 2
        assert set(bodies) == {1, 2}
        assert bodies[1]["area"] == pytest.approx(9 * 900 / 1e6)
        assert bodies[1]["pixel_count"] == 9
        assert bodies[2]["area"] == pytest.approx(900 / 1e6)
        assert "perimeter" not in bodies[1]

    def test_min_area_relabels(self):
        mask = np.zeros((10, 10), bool)
        mask[0, 0] = True  # label 1, removed
        mask[3:6, 3:6] = True  # label 2 -> 1
        labeled, bodies = get_water_bodies(mask, pixel_size=10, min_area=200)
        assert set(bodies) == {1}
        assert labeled[0, 0] == 0
        assert labeled[4, 4] == 1
        assert np.count_nonzero(labeled) == 9

    def test_min_area_removes_all(self):
        labeled, bodies = get_water_bodies(_two_bodies(), pixel_size=1, min_area=1000)
        assert bodies == {}
        assert not labeled.any()

    def test_shapes_match_single_body_metrics(self):
        rng = np.random.default_rng(42)
        mask = rng.random((40, 50)) > 0.6
        labeled, bodies = get_water_bodies(mask, pixel_size=(20, 30), calculate_shapes=True)
        assert len(bodies) == labeled.max()
        for label in (1, len(bodies) // 2, len(bodies)):
            single = calculate_shape_metrics(labeled == label, pixel_size=(20, 30))
            body = bodies[label]
            assert body["area"] == pytest.approx(single["area"] / 1e6)
            assert body["perimeter"] == pytest.approx(single["perimeter"] / 1e3)
            for key in ("compactness", "elongation", "orientation"):
                assert body[key] == pytest.approx(single[key], abs=1e-9)

    def test_empty(self):
        labeled, bodies = get_water_bodies(np.zeros((3, 3), bool))
        assert bodies == {}
        assert labeled.shape == (3, 3)

    def test_connectivity(self):
        _, bodies = get_water_bodies(np.eye(3, dtype=bool), connectivity=2)
        assert len(bodies) == 1

    def test_invalid_min_area(self):
        with pytest.raises(ValueError):
            get_water_bodies(_two_bodies(), min_area=-5)


# --------------------------------------------------------------------------- #
# masked arrays
# --------------------------------------------------------------------------- #
class TestMaskedArrays:
    def test_masked_pixels_are_invalid_in_stats(self):
        data = np.ma.masked_array(np.ones((3, 3)), mask=np.eye(3, dtype=bool))
        stats = water_stats(data, pixel_size=1000)
        assert stats["total_area"] == pytest.approx(6.0)  # 6 water pixels of 1 km²
        assert stats["coverage_percent"] == pytest.approx(100.0)

    def test_masked_pixels_never_change(self):
        before = np.ma.masked_array(np.ones((2, 2)), mask=[[True, False], [False, False]])
        after = np.zeros((2, 2))
        result = water_change(before, after, pixel_size=1000)
        assert result["lost_area"] == pytest.approx(3.0)
        assert result["change_mask"][0, 0] == 0

    def test_masked_water_not_labelled(self):
        data = np.ma.masked_array(np.ones((1, 3), bool), mask=[[False, True, False]])
        labeled, bodies = get_water_bodies(data, pixel_size=1)
        assert labeled.tolist() == [[1, 0, 2]] and len(bodies) == 2


def test_get_water_bodies_rejects_negative_min_area_without_bodies():
    with pytest.raises(ValueError, match="min_area"):
        farq.get_water_bodies(np.zeros((5, 5), dtype=bool), min_area=-1)
