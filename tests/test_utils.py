"""Tests for farq.utils NaN-aware statistics and validation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import farq
from farq import utils


@pytest.fixture
def data():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def with_nan():
    return np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])


# --------------------------------------------------------------------------- validate_array


class TestValidateArray:
    def test_valid(self):
        utils.validate_array(np.ones(3))
        utils.validate_array(np.array([1, 2], dtype=np.int64))
        utils.validate_array(np.array([np.nan, 1.0]))
        utils.validate_array(np.array([True]))

    def test_type_error_message_uses_name(self):
        with pytest.raises(TypeError, match="band"):
            utils.validate_array([1, 2], name="band")

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            utils.validate_array(np.array([]))

    def test_all_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            utils.validate_array(np.full((2, 2), np.nan))
        utils.validate_array(np.full((2, 2), np.nan), allow_all_nan=True)

    def test_leading_nan_not_all(self):
        utils.validate_array(np.array([np.nan, np.nan, 1.0]))

    def test_non_numeric_dtype_ok(self):
        utils.validate_array(np.array(["a", "b"]))


# --------------------------------------------------------------------------- reductions


def test_basic_reductions(data):
    assert farq.min(data) == 1
    assert farq.max(data) == 6
    assert farq.mean(data) == 3.5
    assert farq.sum(data) == 21
    assert farq.median(data) == 3.5
    assert np.isclose(farq.std(data), np.std(data))


def test_std_ddof(data):
    assert np.isclose(farq.std(data, ddof=1), np.std(data, ddof=1))


def test_reductions_ignore_nan(with_nan, no_warn):
    valid = np.array([1.0, 3.0, 4.0, 5.0])
    assert farq.min(with_nan) == 1
    assert farq.max(with_nan) == 5
    assert farq.sum(with_nan) == 13
    assert farq.mean(with_nan) == valid.mean()
    assert np.isclose(farq.std(with_nan), valid.std())
    assert farq.median(with_nan) == 3.5
    assert farq.percentile(with_nan, 50) == 3.5


@pytest.fixture
def no_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_axis(data):
    np.testing.assert_array_equal(farq.min(data, axis=0), [1, 2, 3])
    np.testing.assert_array_equal(farq.max(data, axis=1), [3, 6])
    np.testing.assert_array_equal(farq.sum(data, axis=0), [5, 7, 9])
    np.testing.assert_allclose(farq.mean(data, axis=1), [2, 5])
    np.testing.assert_allclose(farq.median(data, axis=0), [2.5, 3.5, 4.5])
    np.testing.assert_allclose(farq.percentile(data, 50, axis=1), [2, 5])


@pytest.mark.parametrize("func", [farq.min, farq.max, farq.mean, farq.std, farq.median])
def test_all_nan_slice_gives_nan_silently(func, no_warn):
    arr = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    out = func(arr, axis=0)
    assert np.isnan(out[0]) and np.isfinite(out[1])


@pytest.mark.parametrize(
    "func",
    [
        farq.min,
        farq.max,
        farq.mean,
        farq.std,
        farq.sum,
        farq.median,
        lambda a: farq.percentile(a, 50),
    ],
)
def test_all_nan_raises(func):
    with pytest.raises(ValueError):
        func(np.full(3, np.nan))


@pytest.mark.parametrize(
    "func",
    [
        farq.min,
        farq.max,
        farq.mean,
        farq.std,
        farq.sum,
        farq.median,
        farq.count_nonzero,
        farq.unique,
        lambda a: farq.percentile(a, 50),
        farq.stats,
    ],
)
def test_invalid_input(func):
    with pytest.raises(TypeError):
        func([1, 2, 3])
    with pytest.raises(ValueError):
        func(np.array([]))


def test_percentile_multiple(data):
    np.testing.assert_allclose(farq.percentile(data, [0, 100]), [1, 6])


def test_percentile_out_of_range(data):
    with pytest.raises(ValueError):
        farq.percentile(data, 150)


def test_negative_and_large_values():
    assert farq.min(np.array([-1, -4])) == -4
    assert farq.mean(np.array([1e6, 2e6, 3e6, 4e6])) == 2.5e6


def test_count_nonzero():
    assert farq.count_nonzero(np.array([0, 1, 2, 0])) == 2
    np.testing.assert_array_equal(farq.count_nonzero(np.eye(2), axis=0), [1, 1])
    # numpy semantics: NaN is non-zero, all-NaN accepted
    assert farq.count_nonzero(np.array([np.nan, 0.0])) == 1
    assert farq.count_nonzero(np.full(2, np.nan)) == 2


def test_unique():
    np.testing.assert_array_equal(farq.unique(np.array([3, 1, 3])), [1, 3])
    vals, counts = farq.unique(np.array([3, 1, 3]), return_counts=True)
    np.testing.assert_array_equal(vals, [1, 3])
    np.testing.assert_array_equal(counts, [1, 2])


def test_builtins_not_affected():
    import builtins

    assert builtins.min(1, 2) == 1
    assert utils.min is not builtins.min


# --------------------------------------------------------------------------- stats


class TestStats:
    def test_keys_and_values(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, np.nan, np.inf])
        s = farq.stats(arr)
        assert s["min"] == 0 and s["max"] == 3
        assert s["mean"] == 1.5 and s["median"] == 1.5
        assert np.isclose(s["std"], np.std([0, 1, 2, 3]))
        assert np.isclose(s["variance"], np.var([0, 1, 2, 3]))
        assert s["range"] == 3
        assert s["nan"] == 1 and s["inf"] == 1
        assert s["valid"] == 4 and s["zeros"] == 1 and s["non_zero"] == 3
        assert s["size"] == 6 and s["shape"] == (6,) and s["dtype"] == "float64"
        assert s["percentiles"] == {"0": 0.0, "25": 0.75, "50": 1.5, "75": 2.25, "100": 3.0}
        assert s["percentages"]["nan"] == pytest.approx(100 / 6)
        assert s["percentages"]["valid"] == pytest.approx(400 / 6)
        assert "reflectance_stats" not in s

    def test_skew_kurtosis_match_scipy(self, rng):
        sp_stats = pytest.importorskip("scipy.stats")
        arr = rng.gamma(2.0, size=1000)
        s = farq.stats(arr)
        assert np.isclose(s["skewness"], sp_stats.skew(arr))
        assert np.isclose(s["kurtosis"], sp_stats.kurtosis(arr))

    def test_constant_data(self):
        s = farq.stats(np.zeros((3, 3)))
        assert s["std"] == 0 and np.isnan(s["skewness"]) and np.isnan(s["kurtosis"])
        assert s["zeros"] == 9 and s["non_zero"] == 0

    def test_histogram(self):
        s = farq.stats(np.array([0.0, 0.5, 1.0, np.nan]), bins=2)
        np.testing.assert_array_equal(s["histogram"]["counts"], [1, 2])
        np.testing.assert_allclose(s["histogram"]["bin_edges"], [0, 0.5, 1])

    def test_reflectance_scale(self):
        s = farq.stats(np.array([0, 5000, 10000], dtype=np.uint16), reflectance_scale=10000)
        r = s["reflectance_stats"]
        assert r["min"] == 0 and r["max"] == 1 and r["mean"] == 0.5
        assert r["percentiles"]["50"] == 0.5

    def test_integer_and_bool(self):
        s = farq.stats(np.array([[1, 2], [3, 4]], dtype=np.uint8))
        assert s["mean"] == 2.5 and s["nan"] == 0 and s["dtype"] == "uint8"
        b = farq.stats(np.array([True, False, True]))
        assert b["non_zero"] == 2 and b["mean"] == pytest.approx(2 / 3)

    def test_custom_percentiles(self):
        s = farq.stats(np.arange(101.0), percentiles=[10, 90])
        assert s["percentiles"] == {"10": 10.0, "90": 90.0}

    def test_only_inf(self):
        s = farq.stats(np.array([np.inf, -np.inf]))
        assert s["valid"] == 0 and np.isnan(s["mean"])

    def test_does_not_modify_input(self):
        arr = np.array([1.0, np.nan])
        farq.stats(arr)
        assert arr[0] == 1.0 and np.isnan(arr[1])

    def test_invalid(self):
        with pytest.raises(ValueError):
            farq.stats(np.full(2, np.nan))
        with pytest.raises(ValueError):
            farq.stats(np.ones(2), bins=0)
        with pytest.raises(ValueError):
            farq.stats(np.ones(2), reflectance_scale=0)
