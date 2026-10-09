"""Tests for farq.change (change detection)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

from farq import change as ch

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def scene():
    """A noisy 60x60 scene with a 10x10 block that brightens strongly."""
    rng = np.random.default_rng(42)
    before = (0.2 + 0.01 * rng.standard_normal((60, 60))).astype(np.float32)
    after = (before + 0.01 * rng.standard_normal((60, 60))).astype(np.float32)
    after[20:30, 30:40] += 0.5
    truth = np.zeros((60, 60), bool)
    truth[20:30, 30:40] = True
    return before, after, truth


@pytest.fixture(autouse=True)
def _no_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


# --------------------------------------------------------------------------- #
# Pixel-wise measures
# --------------------------------------------------------------------------- #


def test_difference_signed_and_absolute():
    b = np.array([[1.0, 5.0], [2.0, 2.0]])
    a = np.array([[3.0, 1.0], [2.0, 0.0]])
    np.testing.assert_array_equal(ch.difference(b, a), [[2, -4], [0, -2]])
    np.testing.assert_array_equal(ch.difference(b, a, absolute=True), [[2, 4], [0, 2]])


def test_difference_does_not_modify_inputs_and_dtypes():
    b = np.array([[1, 2]], dtype=np.uint16)
    a = np.array([[3, 1]], dtype=np.uint16)
    out = ch.difference(b, a)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [[2, -1]])  # no uint wrap-around
    np.testing.assert_array_equal(b, [[1, 2]])
    assert ch.difference(b.astype(np.float64), a).dtype == np.float64


def test_difference_nodata_nan_and_masked():
    b = np.array([1.0, -9999.0, np.nan, 4.0, 5.0])
    a = np.ma.array([2.0, 2.0, 2.0, np.inf, 7.0], mask=[0, 0, 0, 0, 1])
    out = ch.difference(b, a, nodata=-9999)
    assert out[0] == 1.0
    assert np.isnan(out[1:]).all()


def test_shape_mismatch_message():
    with pytest.raises(ValueError, match="same shape"):
        ch.difference(np.zeros((3, 3)), np.zeros((3, 4)))


def test_invalid_input_types():
    with pytest.raises(TypeError):
        ch.difference([[1, 2]], np.zeros((1, 2)))
    with pytest.raises(ValueError):
        ch.difference(np.array([]), np.array([]))
    with pytest.raises(TypeError, match="dtype"):
        ch.difference(np.array([1 + 1j]), np.array([1 + 1j]))


def test_ratio_log_and_undefined():
    b = np.array([1.0, 2.0, 0.0, 1.0, 0.0])
    a = np.array([np.e, 2.0, 1.0, 0.0, 0.0])
    out = ch.ratio(b, a)
    np.testing.assert_allclose(out[:2], [1.0, 0.0])
    assert np.isnan(out[2:]).all()
    plain = ch.ratio(b, a, log=False)
    np.testing.assert_allclose(plain[[0, 1, 3]], [np.e, 1.0, 0.0])
    assert np.isnan(plain[[2, 4]]).all()


def test_ratio_symmetric():
    b = np.array([1.0, 4.0])
    a = np.array([4.0, 1.0])
    out = ch.ratio(b, a)
    assert out[0] == pytest.approx(-out[1])


def test_normalized_difference_change():
    b = np.array([1.0, 2.0, 0.0, 0.0])
    a = np.array([3.0, 2.0, 5.0, 0.0])
    out = ch.normalized_difference_change(b, a)
    np.testing.assert_allclose(out[:3], [0.5, 0.0, 1.0])
    assert np.isnan(out[3])


# --------------------------------------------------------------------------- #
# CVA and PCA
# --------------------------------------------------------------------------- #


def test_cva_magnitude_direction_sector():
    before = np.zeros((2, 2, 2))
    after = np.array([[[3.0, -1.0], [0.0, 0.0]], [[4.0, 0.0], [-2.0, 0.0]]])
    res = ch.change_vector_analysis(before, after)
    np.testing.assert_allclose(res.magnitude, [[5, 1], [2, 0]])
    np.testing.assert_allclose(res.direction, [[np.degrees(np.arctan2(4, 3)), 180], [270, 0]])
    np.testing.assert_array_equal(res.sector, [[3, 0], [0, 0]])
    assert res.sector.dtype == np.int32


def test_cva_multiband_and_nodata():
    before = np.zeros((3, 2, 2), dtype=np.float32)
    after = np.ones((3, 2, 2), dtype=np.float32)
    after[2, 0, 0] = np.nan
    res = ch.change_vector_analysis(before, after)
    assert res.magnitude.dtype == np.float32
    assert np.isnan(res.magnitude[0, 0]) and np.isnan(res.direction[0, 0])
    assert res.sector[0, 0] == -1
    np.testing.assert_allclose(res.magnitude[1, 1], np.sqrt(3), rtol=1e-6)
    assert res.sector[1, 1] == 0b111


def test_cva_accepts_2d_and_rejects_bad_ndim():
    res = ch.change_vector_analysis(np.zeros((2, 2)), np.full((2, 2), -2.0))
    np.testing.assert_allclose(res.magnitude, 2.0)
    np.testing.assert_allclose(res.direction, 180.0)
    with pytest.raises(ValueError, match="bands, height, width"):
        ch.change_vector_analysis(np.zeros(4), np.zeros(4))


def test_pca_change_detects_block():
    rng = np.random.default_rng(0)
    before = rng.normal(size=(4, 40, 40))
    after = before + 0.05 * rng.normal(size=before.shape)
    after[:, 5:15, 5:15] += np.array([3.0, 2.0, 1.0, 0.5])[:, None, None]
    res = ch.pca_change(before, after)
    assert res.components.shape == (4, 40, 40)
    assert res.loadings.shape == (4, 4)
    assert res.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert res.explained_variance_ratio[0] > 0.95
    assert np.all(np.diff(res.explained_variance_ratio) <= 1e-12)
    pc1 = np.abs(res.components[0])
    assert pc1[5:15, 5:15].min() > pc1[20:, 20:].max()
    # deterministic sign: largest loading of each component is positive
    idx = np.argmax(np.abs(res.loadings), axis=1)
    assert np.all(res.loadings[np.arange(4), idx] > 0)
    # deterministic output
    res2 = ch.pca_change(before, after)
    np.testing.assert_array_equal(res.components, res2.components)


def test_pca_change_nodata_and_options():
    rng = np.random.default_rng(1)
    before = rng.normal(size=(2, 10, 10)).astype(np.float32)
    after = before * 1.0
    after[0, 3, 3] = -9999
    after[1, 2:4, 2:4] += 10
    res = ch.pca_change(before, after, n_components=1, standardize=True, nodata=-9999)
    assert res.components.shape == (1, 10, 10)
    assert res.components.dtype == np.float32
    assert np.isnan(res.components[0, 3, 3])
    assert np.isfinite(res.components[0]).sum() == 99
    with pytest.raises(ValueError, match="n_components"):
        ch.pca_change(before, after, n_components=3)


def test_pca_matches_numpy_reference():
    rng = np.random.default_rng(3)
    before = rng.normal(size=(3, 20, 20))
    after = rng.normal(size=(3, 20, 20))
    res = ch.pca_change(before, after)
    d = (after - before).reshape(3, -1)
    cov = np.cov(d)
    w = np.linalg.eigvalsh(cov)[::-1]
    np.testing.assert_allclose(res.explained_variance_ratio, w / w.sum(), rtol=1e-10)
    # total variance preserved by the orthogonal projection
    assert res.components.reshape(3, -1).var(axis=1, ddof=1).sum() == pytest.approx(w.sum())


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #


def test_otsu_bimodal():
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(0, 1, 5000), rng.normal(10, 1, 5000)])
    t = ch.otsu_threshold(values)
    assert 3.5 < t < 6.5


def test_otsu_two_values_returns_gap_middle():
    values = np.array([0.0] * 10 + [1.0] * 3)
    assert ch.otsu_threshold(values) == pytest.approx(0.5)


def test_otsu_ignores_nan_and_inf():
    values = np.array([0.0, 0.0, 1.0, 1.0, np.nan, np.inf, -np.inf])
    assert ch.otsu_threshold(values) == pytest.approx(0.5)


def test_otsu_edge_cases():
    assert ch.otsu_threshold(np.full(10, 3.0)) == 3.0
    with pytest.raises(ValueError, match="finite"):
        ch.otsu_threshold(np.array([np.nan, np.nan]))
    with pytest.raises(ValueError, match="bins"):
        ch.otsu_threshold(np.arange(5.0), bins=1)


def test_compute_threshold_methods():
    v = np.arange(101.0)
    assert ch.compute_threshold(v, "percentile", percentile=90) == pytest.approx(90.0)
    assert ch.compute_threshold(v, "std", k=1) == pytest.approx(v.mean() + v.std())
    assert ch.compute_threshold(v, 12) == 12.0
    assert ch.compute_threshold(v, np.float32(1.5)) == 1.5
    with pytest.raises(ValueError, match="Unknown threshold"):
        ch.compute_threshold(v, "magic")
    with pytest.raises(ValueError, match="percentile"):
        ch.compute_threshold(v, "percentile", percentile=150)
    with pytest.raises(TypeError):
        ch.compute_threshold(v, True)


def test_threshold_change_mask():
    m = np.array([-3.0, 0.1, 2.5, np.nan])
    np.testing.assert_array_equal(ch.threshold_change(m, 1.0), [False, False, True, False])
    np.testing.assert_array_equal(
        ch.threshold_change(m, 1.0, absolute=True), [True, False, True, False]
    )
    out = ch.threshold_change(np.zeros((3, 3)))  # no change at all
    assert out.dtype == bool and not out.any()


# --------------------------------------------------------------------------- #
# Mask cleanup
# --------------------------------------------------------------------------- #


def test_clean_mask_min_size_and_connectivity():
    m = np.zeros((8, 8), bool)
    m[0:3, 0:3] = True  # 9 px
    m[5, 5] = m[6, 6] = True  # diagonal pair
    m[7, 0] = True  # single
    out8 = ch.clean_mask(m, min_size=2, connectivity=8)
    assert out8[0:3, 0:3].all() and out8[5, 5] and out8[6, 6] and not out8[7, 0]
    out4 = ch.clean_mask(m, min_size=2, connectivity=4)
    assert not out4[5, 5] and not out4[6, 6]
    assert out4.sum() == 9
    np.testing.assert_array_equal(ch.clean_mask(m), m)
    assert ch.clean_mask(m) is not m


def test_clean_mask_fill_holes():
    m = np.zeros((9, 9), bool)
    m[1:8, 1:8] = True
    m[3:6, 3:6] = False  # 9-pixel hole
    m[0, 4] = False  # border background stays background
    filled = ch.clean_mask(m, fill_holes=True)
    assert filled[1:8, 1:8].all()
    assert not filled[0, 0]
    assert not ch.clean_mask(m, fill_holes=8)[4, 4]  # hole too big
    assert ch.clean_mask(m, fill_holes=9)[4, 4]


def test_clean_mask_hole_connectivity():
    # Diagonal gap: the inner pixel touches the outside only diagonally.
    m = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        bool,
    )
    # 8-connected foreground -> 4-connected background -> (1, 1) is a hole
    assert ch.clean_mask(m, fill_holes=True, connectivity=8)[1, 1]
    # 4-connected foreground -> 8-connected background -> not enclosed
    assert not ch.clean_mask(m, fill_holes=True, connectivity=4)[1, 1]


def test_clean_mask_validation():
    with pytest.raises(ValueError, match="2-D"):
        ch.clean_mask(np.zeros((2, 2, 2), bool))
    with pytest.raises(ValueError, match="connectivity"):
        ch.clean_mask(np.zeros((2, 2), bool), min_size=2, connectivity=6)
    with pytest.raises(ValueError, match="min_size"):
        ch.clean_mask(np.zeros((2, 2), bool), min_size=-1)
    out = ch.clean_mask(np.array([[np.nan, 1.0], [0.0, 1.0]]))
    np.testing.assert_array_equal(out, [[False, True], [False, True]])


# --------------------------------------------------------------------------- #
# Categorical change
# --------------------------------------------------------------------------- #


def test_classify_change_codes():
    before = np.array([[0, 0], [1, 1]])
    after = np.array([[0, 1], [0, 1]])
    out = ch.classify_change(before, after)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[ch.NO_CHANGE, ch.GAINED], [ch.LOST, ch.STABLE]])
    assert ch.CHANGE_LABELS[ch.GAINED] == "gained"


def test_classify_change_invalid():
    before = np.array([1.0, np.nan, 0.0, 1.0])
    after = np.array([1.0, 1.0, 1.0, 0.0])
    valid = np.array([True, True, True, False])
    out = ch.classify_change(before, after, valid=valid)
    np.testing.assert_array_equal(out, [ch.STABLE, ch.CHANGE_NODATA, ch.GAINED, ch.CHANGE_NODATA])
    with pytest.raises(ValueError, match="same shape"):
        ch.classify_change(np.zeros(3), np.zeros(4))


def test_transition_matrix_counts():
    before = np.array([[1, 1, 2], [2, 3, 3]])
    after = np.array([[1, 2, 2], [3, 3, 1]])
    tm = ch.transition_matrix(before, after)
    np.testing.assert_array_equal(tm.classes, [1, 2, 3])
    expected = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    np.testing.assert_array_equal(tm.counts, expected)
    assert tm.counts.dtype == np.int64
    assert tm.total == 6 and tm.changed == 3
    np.testing.assert_allclose(tm.normalized("before").sum(axis=1), 1.0)
    np.testing.assert_allclose(tm.normalized("after").sum(axis=0), 1.0)
    assert tm.normalized().sum() == pytest.approx(1.0)
    np.testing.assert_allclose(tm.areas(30), expected * 900.0)
    with pytest.raises(ValueError):
        tm.normalized("rows")


def test_transition_matrix_matches_bruteforce():
    rng = np.random.default_rng(5)
    before = rng.integers(0, 5, size=(50, 50))
    after = rng.integers(0, 5, size=(50, 50))
    tm = ch.transition_matrix(before, after)
    brute = np.zeros((5, 5), int)
    for b, a in zip(before.ravel(), after.ravel()):
        brute[b, a] += 1
    np.testing.assert_array_equal(tm.counts, brute)


def test_transition_matrix_classes_order_nodata_and_unlisted():
    before = np.array([10, 20, 20, 0, 30, 10])
    after = np.array([20, 20, 10, 10, 10, 0])
    tm = ch.transition_matrix(before, after, classes=[20, 10], nodata=0)
    np.testing.assert_array_equal(tm.classes, [20, 10])
    # pairs counted: 10->20, 20->20, 20->10 ; 0 is nodata, 30 unlisted
    np.testing.assert_array_equal(tm.counts, [[1, 1], [1, 0]])
    empty_class = ch.transition_matrix(before, after, classes=[10, 20, 99])
    assert empty_class.counts[2].sum() == 0 and empty_class.counts[:, 2].sum() == 0
    np.testing.assert_allclose(empty_class.normalized("before")[2], 0.0)
    with pytest.raises(ValueError, match="duplicates"):
        ch.transition_matrix(before, after, classes=[1, 1])


def test_transition_matrix_float_labels_with_nan():
    before = np.array([1.0, 2.0, np.nan, 2.0])
    after = np.array([1.0, 1.0, 2.0, np.nan])
    tm = ch.transition_matrix(before, after)
    np.testing.assert_array_equal(tm.classes, [1.0, 2.0])
    np.testing.assert_array_equal(tm.counts, [[1, 0], [1, 0]])


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def test_change_summary_bool_with_affine():
    mask = np.zeros((10, 10), bool)
    mask[:4, :5] = True  # 20 px
    transform = from_origin(500000, 4000000, 30, 30)
    s = ch.change_summary(mask, pixel_size=transform)
    assert s["total_pixels"] == 100 and s["valid_pixels"] == 100
    assert s["pixel_area_m2"] == 900.0
    assert s["changed_pixels"] == 20
    assert s["changed_percent"] == pytest.approx(20.0)
    assert s["changed_area_m2"] == pytest.approx(18000.0)
    assert s["changed_area_km2"] == pytest.approx(0.018)
    assert s["classes"]["unchanged"]["pixels"] == 80


@pytest.mark.parametrize(
    "pixel_size, area",
    [
        (10, 100.0),
        (2.5, 6.25),
        ((10, 20), 200.0),
        ((10, -20), 200.0),
        (Affine(10, 0, 0, 0, -20, 0), 200.0),
        (Affine(10, 1, 0, 1, -20, 0), 201.0),  # rotated / sheared grid
        ({"transform": from_origin(0, 0, 5, 5), "crs": CRS.from_epsg(32633)}, 25.0),
    ],
)
def test_pixel_size_forms(pixel_size, area):
    s = ch.change_summary(np.ones((2, 2), bool), pixel_size=pixel_size)
    assert s["pixel_area_m2"] == pytest.approx(area)
    assert s["changed_area_m2"] == pytest.approx(4 * area)


@pytest.mark.parametrize("crs", [CRS.from_epsg(4326), "EPSG:4326", 4326, "OGC:CRS84"])
def test_pixel_size_geographic_crs_rejected(crs):
    """Any spelling of a geographic CRS is refused: areas would be in degrees²."""
    meta = {"transform": from_origin(0, 0, 0.001, 0.001), "crs": crs}
    with pytest.raises(ValueError, match="geographic"):
        ch.change_summary(np.ones((2, 2), bool), pixel_size=meta)


@pytest.mark.parametrize("crs", ["EPSG:32633", 32633, CRS.from_epsg(32633)])
def test_pixel_size_projected_crs_spellings(crs):
    meta = {"transform": from_origin(0, 0, 10, 10), "crs": crs}
    assert ch.change_summary(np.ones((2, 2), bool), pixel_size=meta)["changed_area_m2"] == 400


def test_pixel_size_bad_values():
    with pytest.raises(ValueError, match="invalid CRS"):
        ch.change_summary(
            np.ones((2, 2), bool),
            pixel_size={"transform": from_origin(0, 0, 1, 1), "crs": "not-a-crs"},
        )
    with pytest.raises(ValueError):
        ch.change_summary(np.ones((2, 2), bool), pixel_size=0)
    with pytest.raises(ValueError, match="transform"):
        ch.change_summary(np.ones((2, 2), bool), pixel_size={"crs": None})
    with pytest.raises(TypeError):
        ch.change_summary(np.ones((2, 2), bool), pixel_size="30m")


def test_pixel_size_meta_without_geotransform_rejected():
    """GCP-only / unreferenced rasters report an identity transform, not 1 m pixels."""
    mask = np.ones((2, 2), bool)
    gcp_meta = {"transform": Affine.identity(), "crs": None, "gcps": ["placeholder"]}
    with pytest.raises(ValueError, match="geotransform"):
        ch.change_summary(mask, pixel_size=gcp_meta)
    with pytest.raises(ValueError, match="no CRS"):
        ch.change_summary(mask, pixel_size={"transform": Affine.identity(), "crs": None})
    # An explicit identity transform in a projected CRS is a genuine 1 m grid.
    s = ch.change_summary(mask, pixel_size={"transform": Affine.identity(), "crs": "EPSG:32633"})
    assert s["pixel_area_m2"] == 1.0


def test_change_summary_no_pixel_size():
    s = ch.change_summary(np.zeros((3, 3), bool))
    assert s["pixel_area_m2"] is None and s["changed_area_m2"] is None
    assert s["changed_pixels"] == 0 and s["changed_percent"] == 0.0
    assert set(s["classes"]) == {"unchanged", "changed"}


def test_change_summary_categorical_with_labels():
    before = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])
    after = np.array([[0, 1, 0, 1], [1, 1, 0, 0]])
    valid = np.ones_like(before, bool)
    valid[1, 3] = False
    codes = ch.classify_change(before, after, valid=valid)
    s = ch.change_summary(codes, pixel_size=10, labels=ch.CHANGE_LABELS, nodata=ch.CHANGE_NODATA)
    assert s["valid_pixels"] == 7 and s["nodata_pixels"] == 1
    assert s["classes"]["no_change"]["pixels"] == 2
    assert s["classes"]["gained"]["pixels"] == 1
    assert s["classes"]["lost"]["pixels"] == 1
    assert s["classes"]["stable"]["pixels"] == 3
    assert s["classes"]["gained"]["area_m2"] == 100.0
    assert sum(c["percent"] for c in s["classes"].values()) == pytest.approx(100.0)
    assert "changed_pixels" not in s


def test_change_summary_includes_zero_count_labels_and_masked():
    data = np.ma.array([1, 1, 3, 3], mask=[0, 0, 0, 1])
    s = ch.change_summary(data, labels=ch.CHANGE_LABELS)
    assert s["classes"]["lost"]["pixels"] == 0
    assert s["classes"]["stable"]["pixels"] == 1
    assert s["nodata_pixels"] == 1


def test_change_summary_json_serializable():
    import json

    s = ch.change_summary(np.eye(3, dtype=bool), pixel_size=(30, 30))
    json.dumps(s)


# --------------------------------------------------------------------------- #
# detect_changes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["difference", "ratio", "normalized_difference"])
def test_detect_changes_single_band_methods(scene, method):
    before, after, truth = scene
    res = ch.detect_changes(before, after, method=method)
    assert isinstance(res, ch.ChangeResult)
    assert res.method == method
    assert res.mask.dtype == bool and res.mask.shape == truth.shape
    assert res.magnitude.dtype == np.float32
    np.testing.assert_array_equal(res.mask, truth)
    magnitude, mask, threshold, name = res  # tuple unpacking
    assert magnitude is res.magnitude and mask is res.mask
    assert np.isfinite(threshold) and name == method


@pytest.mark.parametrize("method", ["cva", "pca"])
def test_detect_changes_multiband_methods(method):
    rng = np.random.default_rng(11)
    before = (0.3 + 0.01 * rng.standard_normal((3, 50, 50))).astype(np.float32)
    after = before + (0.01 * rng.standard_normal((3, 50, 50))).astype(np.float32)
    after[:, 10:20, 25:40] += np.array([0.3, -0.2, 0.4], dtype=np.float32)[:, None, None]
    truth = np.zeros((50, 50), bool)
    truth[10:20, 25:40] = True
    res = ch.detect_changes(before, after, method=method)
    np.testing.assert_array_equal(res.mask, truth)


def test_detect_changes_nodata_never_flagged(scene):
    before, after, truth = scene
    before = before.copy()
    after = after.copy()
    before[22, 32] = -9999  # inside the changed block
    after[0, 0] = np.nan
    res = ch.detect_changes(before, after, nodata=-9999, fill_holes=True)
    assert not res.mask[22, 32] and not res.mask[0, 0]
    assert np.isnan(res.magnitude[22, 32]) and np.isnan(res.magnitude[0, 0])
    assert res.mask.sum() == truth.sum() - 1
    s = res.summary(pixel_size=10)
    assert s["nodata_pixels"] == 2
    assert s["changed_pixels"] == 99
    assert s["changed_area_m2"] == pytest.approx(9900.0)


def test_detect_changes_min_size_removes_speckle(scene):
    before, after, truth = scene
    after = after.copy()
    after[50, 5] += 0.5  # isolated single-pixel change
    noisy = ch.detect_changes(before, after)
    assert noisy.mask[50, 5]
    clean = ch.detect_changes(before, after, min_size=4)
    np.testing.assert_array_equal(clean.mask, truth)


def test_detect_changes_threshold_options(scene):
    before, after, truth = scene
    fixed = ch.detect_changes(before, after, threshold=0.25)
    assert fixed.threshold == 0.25
    np.testing.assert_array_equal(fixed.mask, truth)
    pct = ch.detect_changes(before, after, threshold="percentile", percentile=99)
    assert pct.mask.sum() == pytest.approx(36, abs=1)
    std = ch.detect_changes(before, after, threshold="std", k=3)
    assert std.mask[truth].all()


def test_detect_changes_no_change_and_decrease():
    flat = np.full((20, 20), 0.5, dtype=np.float32)
    res = ch.detect_changes(flat, flat.copy())
    assert not res.mask.any()
    after = flat.copy()
    after[:5, :5] = 0.1  # decrease must be detected too (magnitude is absolute)
    res = ch.detect_changes(flat, after)
    assert res.mask[:5, :5].all() and res.mask.sum() == 25


def test_detect_changes_input_validation():
    with pytest.raises(ValueError, match="Unknown method"):
        ch.detect_changes(np.zeros((2, 2)), np.zeros((2, 2)), method="magic")
    with pytest.raises(ValueError, match="cva"):
        ch.detect_changes(np.zeros((3, 4, 4)), np.zeros((3, 4, 4)), method="difference")
    # single-band 3-D stack is accepted
    b = np.zeros((1, 4, 4))
    a = np.zeros((1, 4, 4))
    a[0, 0, 0] = 1
    res = ch.detect_changes(b, a)
    assert res.mask.shape == (4, 4) and res.mask[0, 0]
    with pytest.raises(ValueError, match="no finite"):
        ch.detect_changes(np.full((2, 2), -1.0), np.full((2, 2), 1.0), nodata=-1)


def test_detect_changes_is_deterministic(scene):
    before, after, _ = scene
    r1 = ch.detect_changes(before, after, method="ratio", min_size=3)
    r2 = ch.detect_changes(before, after, method="ratio", min_size=3)
    np.testing.assert_array_equal(r1.magnitude, r2.magnitude)
    np.testing.assert_array_equal(r1.mask, r2.mask)
    assert r1.threshold == r2.threshold


def test_public_api():
    for name in ch.__all__:
        assert hasattr(ch, name), name
