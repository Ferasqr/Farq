"""Tests for farq.indices spectral index calculations."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import farq
from farq import indices

ND_CASES = [
    # (function, positional band order, formula as (a - b) / (a + b) of positional indices)
    (indices.ndvi, ("nir", "red"), (0, 1)),
    (indices.ndwi, ("green", "nir"), (0, 1)),
    (indices.mndwi, ("green", "swir1"), (0, 1)),
    (indices.ndbi, ("swir1", "nir"), (0, 1)),
    (indices.nbr, ("nir", "swir2"), (0, 1)),
    (indices.ndmi, ("nir", "swir1"), (0, 1)),
]


@pytest.fixture
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _nd(a, b):
    return (a - b) / (a + b)


@pytest.fixture
def bands(rng):
    names = ("blue", "green", "red", "nir", "swir1", "swir2")
    return {n: rng.uniform(0.01, 0.6, size=(20, 30)) for n in names}


# --------------------------------------------------------------------------- formulas


@pytest.mark.parametrize(("func", "names", "_"), ND_CASES)
def test_normalized_difference_formulas(func, names, _, bands):
    a, b = bands[names[0]], bands[names[1]]
    np.testing.assert_allclose(func(a, b), _nd(a, b))


def test_ndwi_mcfeeters_sign():
    green = np.array([100, 1000, 500, 2000], dtype=float)
    nir = np.array([1000, 100, 500, 2000], dtype=float)
    result = farq.ndwi(green, nir)
    np.testing.assert_allclose(result, [-0.81818182, 0.81818182, 0.0, 0.0])


def test_ndwi_water_positive():
    green = np.array([[0.3, 0.1]])
    nir = np.array([[0.05, 0.3]])
    water = farq.ndwi(green, nir) > 0
    np.testing.assert_array_equal(water, [[True, False]])


def test_mndwi_known_value():
    np.testing.assert_allclose(farq.indices.mndwi(np.array([0.3]), np.array([0.1])), [0.5])


def test_ndvi_vegetation():
    nir = np.array([0.5, 0.1])
    red = np.array([0.05, 0.3])
    np.testing.assert_allclose(farq.ndvi(nir, red), [0.45 / 0.55, -0.5])


def test_evi_formula(bands):
    r, n, b = bands["red"], bands["nir"], bands["blue"]
    expected = 2.5 * (n - r) / (n + 6.0 * r - 7.5 * b + 1.0)
    np.testing.assert_allclose(farq.evi(r, n, b, clip=False), expected, rtol=1e-12)


def test_evi_known_value():
    # Typical healthy vegetation: red 0.05, nir 0.4, blue 0.03
    out = farq.evi(np.array([0.05]), np.array([0.4]), np.array([0.03]))
    np.testing.assert_allclose(out, [2.5 * 0.35 / (0.4 + 0.3 - 0.225 + 1.0)])


def test_evi_custom_coefficients(bands):
    r, n, b = bands["red"], bands["nir"], bands["blue"]
    expected = 2.0 * (n - r) / (n + 5.0 * r - 7.0 * b + 0.5)
    np.testing.assert_allclose(
        farq.evi(r, n, b, G=2.0, C1=5.0, C2=7.0, L=0.5, clip=False), expected, rtol=1e-12
    )


def test_evi_reflectance_scale():
    red, nir, blue = (np.array([v], dtype=np.uint16) for v in (500, 4000, 300))
    scaled = farq.evi(red, nir, blue, reflectance_scale=10000)
    direct = farq.evi(np.array([0.05]), np.array([0.4]), np.array([0.03]))
    np.testing.assert_allclose(scaled, direct, rtol=1e-6)


def test_savi_formula(bands):
    n, r = bands["nir"], bands["red"]
    for L in (0.0, 0.25, 0.5, 1.0):
        expected = (1 + L) * (n - r) / (n + r + L)
        np.testing.assert_allclose(farq.savi(n, r, L=L), expected, rtol=1e-12)


def test_savi_l0_equals_ndvi(bands):
    n, r = bands["nir"], bands["red"]
    np.testing.assert_allclose(farq.savi(n, r, L=0.0), farq.ndvi(n, r))


def test_savi_reflectance_scale():
    n = np.array([4000], dtype=np.uint16)
    r = np.array([500], dtype=np.uint16)
    np.testing.assert_allclose(
        farq.savi(n, r, reflectance_scale=10000),
        farq.savi(np.array([0.4]), np.array([0.05])),
        rtol=1e-6,
    )


def test_calculate_normalized_difference_order():
    out = farq.calculate_normalized_difference(np.array([3.0]), np.array([1.0]))
    np.testing.assert_allclose(out, [0.5])


# --------------------------------------------------------------------------- edge cases


@pytest.mark.parametrize(("func", "_n", "_"), ND_CASES)
def test_zero_division_gives_nan_without_warning(func, _n, _, no_warnings):
    a = np.array([0.0, 0.0, 0.2])
    b = np.array([0.0, -0.0, 0.2])
    out = func(a, b)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == 0.0


def test_nonzero_over_zero_sum_is_nan(no_warnings):
    # a + b == 0 with a != b (only possible with negative input)
    out = farq.calculate_normalized_difference(np.array([0.1]), np.array([-0.1]))
    assert np.isnan(out[0])


def test_evi_zero_denominator_nan(no_warnings):
    # nir + 6*red - 7.5*blue + 1 == 0
    out = farq.evi(np.array([0.0]), np.array([0.5]), np.array([0.2]))
    assert np.isnan(out[0])


def test_savi_zero_denominator_nan(no_warnings):
    out = farq.savi(np.array([0.0]), np.array([0.0]), L=0.0)
    assert np.isnan(out[0])


@pytest.mark.parametrize(("func", "_n", "_"), ND_CASES)
def test_nan_propagates(func, _n, _, no_warnings):
    a = np.array([np.nan, 0.3])
    b = np.array([0.2, np.nan])
    assert np.isnan(func(a, b)).all()


def test_nan_propagates_evi(no_warnings):
    out = farq.evi(np.array([np.nan, 0.1]), np.array([0.4, 0.4]), np.array([0.03, 0.03]))
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_clip_negative_inputs():
    nir = np.array([0.2, -0.3])
    red = np.array([-0.1, 0.2])
    out = farq.ndvi(nir, red)
    assert np.all((out >= -1) & (out <= 1))
    unclipped = farq.ndvi(nir, red, clip=False)
    np.testing.assert_allclose(unclipped, _nd(nir, red))
    assert np.abs(unclipped).max() > 1


def test_evi_clip_option():
    # Bright red/blue imbalance gives EVI beyond [-1, 1]
    red, nir, blue = np.array([0.0]), np.array([0.9]), np.array([0.2])
    assert farq.evi(red, nir, blue)[0] == 1.0
    assert farq.evi(red, nir, blue, clip=False)[0] > 1.0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_integer_inputs_do_not_wrap(dtype):
    # uint16 subtraction would wrap around if done in the input dtype
    nir = np.array([100, 2000], dtype=dtype) if dtype != np.uint8 else np.array([10, 200], dtype)
    red = np.array([2000, 100], dtype=dtype) if dtype != np.uint8 else np.array([200, 10], dtype)
    out = farq.ndvi(nir, red)
    assert out.dtype == np.float32
    assert out[0] < 0 < out[1]
    np.testing.assert_allclose(out[0], -out[1])


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.uint16, np.float32),
        (np.int32, np.float64),
        (np.float16, np.float32),
    ],
)
def test_output_dtype(dtype, expected):
    a = np.array([1, 2], dtype=dtype)
    b = np.array([2, 1], dtype=dtype)
    for func in (farq.ndvi, farq.ndwi, farq.ndbi, farq.nbr, farq.ndmi, indices.mndwi, farq.savi):
        assert func(a, b).dtype == expected
    assert farq.evi(a, b, a).dtype == expected


@pytest.mark.parametrize("func", [farq.ndvi, farq.ndwi, farq.savi, indices.mndwi])
def test_inputs_not_modified(func):
    a = np.array([0.1, 0.0], dtype=np.float32)
    b = np.array([0.3, 0.0], dtype=np.float32)
    a0, b0 = a.copy(), b.copy()
    out = func(a, b)
    np.testing.assert_array_equal(a, a0)
    np.testing.assert_array_equal(b, b0)
    assert out is not a and out is not b


def test_evi_inputs_not_modified():
    r, n, b = np.array([0.1]), np.array([0.4]), np.array([0.05])
    farq.evi(r, n, b, reflectance_scale=1.0)
    assert (r[0], n[0], b[0]) == (0.1, 0.4, 0.05)


def test_masked_array_input():
    nir = np.ma.masked_array([0.5, 0.5], mask=[True, False])
    red = np.array([0.1, 0.1])
    out = farq.ndvi(nir, red)
    assert np.isnan(out[0]) and np.isclose(out[1], 0.4 / 0.6)


def test_3d_input():
    a = np.full((2, 3, 3), 0.3)
    b = np.full((2, 3, 3), 0.1)
    np.testing.assert_allclose(farq.ndvi(a, b), 0.5)


def test_reflectance_scale_does_not_change_nd(bands):
    a, b = bands["nir"], bands["red"]
    np.testing.assert_array_equal(farq.ndvi(a, b, 10000), farq.ndvi(a, b))


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "func",
    [
        farq.ndvi,
        farq.ndwi,
        farq.ndbi,
        farq.nbr,
        farq.ndmi,
        indices.mndwi,
        farq.savi,
        farq.calculate_normalized_difference,
    ],
)
def test_mismatched_shapes(func):
    with pytest.raises(ValueError, match="shapes"):
        func(np.ones((2, 2)), np.ones((2, 3)))


@pytest.mark.parametrize("func", [farq.ndvi, farq.ndwi, indices.mndwi, farq.savi])
def test_wrong_types(func):
    with pytest.raises(TypeError):
        func([0.1, 0.2], np.array([0.1, 0.2]))


def test_empty_band():
    with pytest.raises(ValueError):
        farq.ndvi(np.array([]), np.array([]))


@pytest.mark.parametrize("scale", [0, -10])
def test_invalid_reflectance_scale(scale):
    with pytest.raises(ValueError):
        farq.ndvi(np.ones(2), np.ones(2), reflectance_scale=scale)
    with pytest.raises(ValueError):
        farq.evi(np.ones(2), np.ones(2), np.ones(2), reflectance_scale=scale)


def test_evi_parameter_validation():
    r = n = b = np.ones(2)
    with pytest.raises(ValueError):
        farq.evi(r, n, b, G=0)
    with pytest.raises(ValueError):
        farq.evi(r, n, b, L=-1)
    with pytest.raises(TypeError):
        farq.evi(r, n, b, C1="6")
    # numpy scalars are accepted
    farq.evi(r, n, b, G=np.float32(2.5), C1=np.float64(6))


def test_savi_parameter_validation():
    with pytest.raises(ValueError):
        farq.savi(np.ones(2), np.ones(2), L=1.5)
    with pytest.raises(TypeError):
        farq.savi(np.ones(2), np.ones(2), L=None)


# --------------------------------------------------------------------------- calculate_indices


def test_calculate_indices_all(bands):
    names = ["ndvi", "ndwi", "mndwi", "evi", "savi", "ndbi", "nbr", "ndmi"]
    result = farq.calculate_indices(bands, names)
    assert set(result) == set(names)
    np.testing.assert_array_equal(result["ndwi"], farq.ndwi(bands["green"], bands["nir"]))
    np.testing.assert_array_equal(
        result["evi"], farq.evi(bands["red"], bands["nir"], bands["blue"])
    )
    np.testing.assert_array_equal(result["mndwi"], indices.mndwi(bands["green"], bands["swir1"]))


def test_calculate_indices_single_string_case_insensitive(bands):
    result = farq.calculate_indices(bands, "NDVI")
    assert list(result) == ["ndvi"]


def test_calculate_indices_reflectance_scale():
    bands = {
        k: np.array([v], dtype=np.uint16) for k, v in {"red": 500, "nir": 4000, "blue": 300}.items()
    }
    result = farq.calculate_indices(bands, ["evi", "ndvi"], reflectance_scale=10000)
    np.testing.assert_allclose(
        result["evi"], farq.evi(np.array([0.05]), np.array([0.4]), np.array([0.03])), rtol=1e-6
    )


def test_calculate_indices_errors(bands):
    with pytest.raises(ValueError, match="Unknown index"):
        farq.calculate_indices(bands, ["foo"])
    with pytest.raises(ValueError, match="Missing required bands"):
        farq.calculate_indices({"red": bands["red"]}, ["ndvi"])


# --------------------------------------------------------------------------- RGB indices

RGB3 = [indices.vari, indices.exg, indices.exr, indices.exgr, indices.gli, indices.tgi]


@pytest.fixture
def rgb(rng):
    return tuple(rng.uniform(0.05, 0.9, size=(10, 12)) for _ in range(3))


def test_vari_formula(rgb):
    r, g, b = rgb
    np.testing.assert_allclose(indices.vari(r, g, b, clip=False), (g - r) / (g + r - b))


def test_vari_known_value_and_clip():
    r, g, b = np.array([0.2, 0.2]), np.array([0.4, 0.3]), np.array([0.1, 0.45])
    np.testing.assert_allclose(indices.vari(r, g, b, clip=False), [0.2 / 0.5, 0.1 / 0.05])
    np.testing.assert_allclose(indices.vari(r, g, b), [0.4, 1.0])


def test_exg_formula(rgb):
    r, g, b = rgb
    t = r + g + b
    np.testing.assert_allclose(indices.exg(r, g, b), 2 * g / t - r / t - b / t)


def test_exr_formula(rgb):
    r, g, b = rgb
    t = r + g + b
    np.testing.assert_allclose(indices.exr(r, g, b), 1.4 * r / t - g / t)


def test_exgr_is_exg_minus_exr(rgb):
    r, g, b = rgb
    np.testing.assert_allclose(
        indices.exgr(r, g, b), indices.exg(r, g, b) - indices.exr(r, g, b), atol=1e-12
    )


def test_chromatic_indices_known_values():
    # Pure green pixel: r=0, g=1, b=0
    r, g, b = np.array([0], np.uint8), np.array([200], np.uint8), np.array([0], np.uint8)
    assert indices.exg(r, g, b)[0] == pytest.approx(2.0)
    assert indices.exr(r, g, b)[0] == pytest.approx(-1.0)
    assert indices.exgr(r, g, b)[0] == pytest.approx(3.0)
    # Grey pixel: r=g=b=1/3
    grey = np.array([90], np.uint8)
    assert indices.exg(grey, grey, grey)[0] == pytest.approx(0.0, abs=1e-6)
    assert indices.exr(grey, grey, grey)[0] == pytest.approx(0.4 / 3, rel=1e-6)


def test_chromatic_scale_invariant(rgb):
    r, g, b = rgb
    for func in (indices.exg, indices.exr, indices.exgr, indices.gli, indices.vari):
        np.testing.assert_allclose(func(r * 255, g * 255, b * 255), func(r, g, b), rtol=1e-10)


def test_gli_formula(rgb):
    r, g, b = rgb
    np.testing.assert_allclose(indices.gli(r, g, b), (2 * g - r - b) / (2 * g + r + b))


def test_ngrdi_formula(rgb):
    r, g, _ = rgb
    np.testing.assert_allclose(indices.ngrdi(r, g), (g - r) / (g + r))


def test_tgi_formula(rgb):
    r, g, b = rgb
    expected = -0.5 * (190 * (r - g) - 120 * (r - b))
    np.testing.assert_allclose(indices.tgi(r, g, b), expected)


def test_tgi_scale_and_wavelengths():
    r, g, b = (np.array([v], np.uint8) for v in (51, 102, 25))
    np.testing.assert_allclose(
        indices.tgi(r, g, b, reflectance_scale=255),
        indices.tgi(np.array([0.2]), np.array([0.4]), np.array([25 / 255])),
        rtol=1e-6,
    )
    custom = indices.tgi(
        np.array([0.2]), np.array([0.4]), np.array([0.1]), wavelengths=(660, 560, 470)
    )
    np.testing.assert_allclose(custom, -0.5 * (190 * (-0.2) - 100 * 0.1))
    with pytest.raises(ValueError):
        indices.tgi(r, g, b, wavelengths=(1, 2))


@pytest.mark.parametrize("func", [f for f in RGB3 if f is not indices.exr])
def test_rgb_uint8_input(func):
    r = np.array([[10, 250]], np.uint8)
    g = np.array([[250, 10]], np.uint8)
    b = np.array([[5, 5]], np.uint8)
    out = func(r, g, b)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out[0, 0] > out[0, 1]  # the green pixel scores higher


def test_exr_uint8_ordering():
    r = np.array([10, 250], np.uint8)
    g = np.array([250, 10], np.uint8)
    b = np.array([5, 5], np.uint8)
    out = indices.exr(r, g, b)
    assert out[0] < out[1]


def test_ngrdi_uint8_no_wrap():
    out = indices.ngrdi(np.array([250], np.uint8), np.array([10], np.uint8))
    assert out.dtype == np.float32 and out[0] == pytest.approx(-240 / 260)


@pytest.mark.parametrize(
    "func", [indices.vari, indices.exg, indices.exr, indices.exgr, indices.gli]
)
def test_rgb_black_pixel_nan_no_warning(func, no_warnings):
    zero = np.zeros(2, np.uint8)
    out = func(zero, zero, zero)
    assert np.isnan(out).all()


def test_ngrdi_zero_nan(no_warnings):
    assert np.isnan(indices.ngrdi(np.zeros(1), np.zeros(1))[0])


@pytest.mark.parametrize("func", RGB3)
def test_rgb_nan_propagates(func, no_warnings):
    out = func(np.array([np.nan, 0.2]), np.array([0.3, 0.4]), np.array([0.1, 0.1]))
    assert np.isnan(out[0]) and np.isfinite(out[1])


@pytest.mark.parametrize("func", RGB3)
def test_rgb_mismatched_shapes(func):
    with pytest.raises(ValueError, match="shapes"):
        func(np.ones((2, 2)), np.ones((2, 2)), np.ones((3, 2)))


@pytest.mark.parametrize("func", [indices.vari, indices.gli, indices.ngrdi])
def test_rgb_clip_range(func, rng):
    bands = [rng.uniform(-0.2, 1, 200) for _ in range(3)]
    args = bands[:2] if func is indices.ngrdi else bands
    out = func(*args)
    finite = out[np.isfinite(out)]
    assert finite.min() >= -1 and finite.max() <= 1


def test_calculate_indices_rgb():
    rgb_bands = {
        "red": np.array([[10, 200]], np.uint8),
        "green": np.array([[200, 10]], np.uint8),
        "blue": np.array([[20, 20]], np.uint8),
    }
    names = ["vari", "exg", "exr", "exgr", "gli", "ngrdi", "tgi"]
    result = farq.calculate_indices(rgb_bands, names)
    assert list(result) == names
    np.testing.assert_array_equal(result["exg"], indices.exg(*rgb_bands.values()))
    np.testing.assert_array_equal(
        result["ngrdi"], indices.ngrdi(rgb_bands["red"], rgb_bands["green"])
    )
    with pytest.raises(ValueError, match="Missing"):
        farq.calculate_indices({"red": rgb_bands["red"]}, ["ngrdi"])
