"""Tests for farq.masking (quality-band masks)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

from farq import masking as mk


@pytest.fixture(autouse=True)
def _no_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def qa_value(*bits: int, **conf: int) -> int:
    """Build a Landsat QA_PIXEL value bit by bit (conf: cloud/shadow/snow/cirrus=0-3)."""
    value = sum(1 << b for b in bits)
    offsets = {"cloud": 8, "shadow": 10, "snow": 12, "cirrus": 14}
    for name, level in conf.items():
        value |= level << offsets[name]
    return value


LOW = {"cloud": 1, "shadow": 1, "snow": 1, "cirrus": 1}

# Well-known Landsat 8/9 Collection 2 QA_PIXEL values, built from their bits.
CLEAR_LAND = qa_value(6, **LOW)
CLEAR_WATER = qa_value(6, 7, **LOW)
CLOUD_HIGH = qa_value(3, **{**LOW, "cloud": 3})
SHADOW = qa_value(4, 6, **{**LOW, "shadow": 3})
CIRRUS = qa_value(2, 6, **{**LOW, "cirrus": 3})
SNOW = qa_value(5, 6, **{**LOW, "snow": 3})
DILATED = qa_value(1, **{**LOW, "cloud": 2})
FILL = 1


def test_known_landsat_values():
    assert CLEAR_LAND == 21824
    assert CLEAR_WATER == 21952
    assert CLOUD_HIGH == 22280
    assert SHADOW == 23888
    assert CIRRUS == 54596
    assert SNOW == 30048
    assert DILATED == 22018
    # Landsat 4-7 clear land (no cirrus fields).
    assert qa_value(6, cloud=1, shadow=1, snow=1) == 5440


# --------------------------------------------------------------------------- #
# decode_bits
# --------------------------------------------------------------------------- #
def test_decode_bits_single_and_fields():
    qa = np.array([CLEAR_LAND, CLOUD_HIGH], dtype=np.uint16)
    np.testing.assert_array_equal(mk.decode_bits(qa, 3), [0, 1])
    np.testing.assert_array_equal(mk.decode_bits(qa, 6), [1, 0])
    np.testing.assert_array_equal(mk.decode_bits(qa, 8, 2), [1, 3])
    np.testing.assert_array_equal(mk.decode_bits(qa, 14, 2), [1, 1])
    assert mk.decode_bits(qa, 0).dtype == np.uint8
    assert mk.decode_bits(qa, 0, 12).dtype == np.uint16
    # Top bit of a uint16 and every bit of a full-width field.
    top = np.array([0x8000, 0xFFFF], dtype=np.uint16)
    np.testing.assert_array_equal(mk.decode_bits(top, 15), [1, 1])
    np.testing.assert_array_equal(mk.decode_bits(top, 0, 16), [0x8000, 0xFFFF])


def test_decode_bits_matches_python_for_all_uint16():
    qa = np.arange(1 << 16, dtype=np.uint16)
    for bit, width in [(0, 1), (3, 1), (8, 2), (14, 2), (5, 7)]:
        expected = (np.arange(1 << 16) >> bit) & ((1 << width) - 1)
        np.testing.assert_array_equal(mk.decode_bits(qa, bit, width), expected)


def test_decode_bits_shapes_and_signed():
    qa = np.full((2, 3, 4), 8, dtype=np.int32)
    out = mk.decode_bits(qa, 3)
    assert out.shape == (2, 3, 4) and out.all()
    with pytest.raises(ValueError, match="negative"):
        mk.decode_bits(np.array([-1], dtype=np.int16), 0)


@pytest.mark.parametrize(
    ("qa", "bit", "width", "error"),
    [
        (np.array([1.0]), 0, 1, TypeError),
        (np.array([True]), 0, 1, TypeError),
        ([1, 2], 0, 1, TypeError),
        (np.array([], dtype=np.uint16), 0, 1, ValueError),
        (np.array([1], dtype=np.uint16), 16, 1, ValueError),
        (np.array([1], dtype=np.uint16), 15, 2, ValueError),
        (np.array([1], dtype=np.uint8), 8, 1, ValueError),
        (np.array([1], dtype=np.uint16), -1, 1, ValueError),
        (np.array([1], dtype=np.uint16), 0, 0, ValueError),
        (np.array([1], dtype=np.uint16), 1.5, 1, TypeError),
        (np.array([1], dtype=np.uint16), True, 1, TypeError),
    ],
)
def test_decode_bits_errors(qa, bit, width, error):
    with pytest.raises(error):
        mk.decode_bits(qa, bit, width)


# --------------------------------------------------------------------------- #
# Landsat QA_PIXEL
# --------------------------------------------------------------------------- #
def test_decode_landsat_qa():
    d = mk.decode_landsat_qa(np.array([CLOUD_HIGH, SHADOW, FILL], dtype=np.uint16))
    np.testing.assert_array_equal(d["cloud"], [True, False, False])
    np.testing.assert_array_equal(d["cloud_shadow"], [False, True, False])
    np.testing.assert_array_equal(d["fill"], [False, False, True])
    np.testing.assert_array_equal(d["clear"], [False, True, False])
    np.testing.assert_array_equal(d["cloud_confidence"], [3, 1, 0])
    np.testing.assert_array_equal(d["cloud_shadow_confidence"], [1, 3, 0])
    assert d["cloud"].dtype == bool and d["cloud_confidence"].dtype == np.uint8


def test_landsat_qa_mask_defaults():
    qa = np.array(
        [CLEAR_LAND, CLEAR_WATER, CLOUD_HIGH, SHADOW, CIRRUS, SNOW, DILATED, FILL],
        dtype=np.uint16,
    )
    expected = [False, False, True, True, True, False, True, True]
    np.testing.assert_array_equal(mk.landsat_qa_mask(qa), expected)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("cloud", CLOUD_HIGH),
        ("shadow", SHADOW),
        ("cirrus", CIRRUS),
        ("dilated", DILATED),
        ("fill", FILL),
    ],
)
def test_landsat_qa_mask_flags_toggle(flag, value):
    qa = np.array([value], dtype=np.uint16)
    assert mk.landsat_qa_mask(qa)[0]
    assert not mk.landsat_qa_mask(qa, **{flag: False})[0]


def test_landsat_qa_mask_snow_and_water_opt_in():
    qa = np.array([SNOW, CLEAR_WATER, CLEAR_LAND], dtype=np.uint16)
    np.testing.assert_array_equal(mk.landsat_qa_mask(qa), [False, False, False])
    np.testing.assert_array_equal(mk.landsat_qa_mask(qa, snow=True), [True, False, False])
    np.testing.assert_array_equal(mk.landsat_qa_mask(qa, water=True), [False, True, False])


def test_landsat_qa_mask_landsat_4_7_values():
    # Landsat 4-7: clear land 5440, water 5504 (clear bit unset over water, a known
    # Collection 2 issue), high-confidence cloud.
    l7 = np.array([5440, 5504, qa_value(3, cloud=3, shadow=1, snow=1)], dtype=np.uint16)
    np.testing.assert_array_equal(mk.landsat_qa_mask(l7), [False, False, True])


def test_landsat_qa_mask_min_confidence():
    medium_cloud = qa_value(6, **{**LOW, "cloud": 2})  # not flagged in bit 3
    high_shadow_conf = qa_value(6, **{**LOW, "shadow": 3})  # without the shadow bit
    qa = np.array([CLEAR_LAND, medium_cloud, high_shadow_conf], dtype=np.uint16)
    np.testing.assert_array_equal(mk.landsat_qa_mask(qa), [False, False, False])
    np.testing.assert_array_equal(
        mk.landsat_qa_mask(qa, min_confidence="medium"), [False, True, True]
    )
    np.testing.assert_array_equal(
        mk.landsat_qa_mask(qa, min_confidence=mk.Confidence.HIGH), [False, False, True]
    )
    # "low" masks every pixel with a low confidence (nearly all clear pixels).
    assert mk.landsat_qa_mask(qa, min_confidence="low").all()
    # Disabled categories ignore their confidence field.
    np.testing.assert_array_equal(
        mk.landsat_qa_mask(qa, shadow=False, min_confidence="high"), [False, False, False]
    )
    # Medium is reserved for shadow/snow/cirrus: "medium" means >= high there.
    reserved = np.array([qa_value(6, shadow=2)], dtype=np.uint16)
    assert not mk.landsat_qa_mask(reserved, min_confidence="medium")[0]


@pytest.mark.parametrize("level", ["none", 0, 4, "extreme", 1.5])
def test_landsat_qa_mask_bad_confidence(level):
    with pytest.raises((ValueError, TypeError)):
        mk.landsat_qa_mask(np.array([CLEAR_LAND], dtype=np.uint16), min_confidence=level)


def test_landsat_qa_mask_float_and_masked_input():
    qa = np.array([[CLEAR_LAND, np.nan], [CLOUD_HIGH, CLEAR_WATER]], dtype=np.float32)
    np.testing.assert_array_equal(mk.landsat_qa_mask(qa), [[False, True], [True, False]])
    ma = np.ma.MaskedArray(np.array([CLEAR_LAND, CLEAR_LAND], np.uint16), mask=[False, True])
    np.testing.assert_array_equal(mk.landsat_qa_mask(ma), [False, True])
    with pytest.raises(ValueError, match="whole numbers"):
        mk.landsat_qa_mask(np.array([1.5]))
    with pytest.raises(ValueError, match="0, 65535"):
        mk.landsat_qa_mask(np.array([70000.0]))
    with pytest.raises(ValueError, match="16 bits"):
        mk.landsat_qa_mask(np.array([70000], dtype=np.uint32))
    with pytest.raises(TypeError):
        mk.landsat_qa_mask(np.array(["a"]))


def test_landsat_qa_mask_does_not_modify_input():
    qa = np.array([CLOUD_HIGH, CLEAR_LAND], dtype=np.uint16)
    copy = qa.copy()
    mk.landsat_qa_mask(qa, min_confidence="high")
    np.testing.assert_array_equal(qa, copy)


# --------------------------------------------------------------------------- #
# Landsat QA_RADSAT and scaling
# --------------------------------------------------------------------------- #
def test_landsat_radsat_mask_oli():
    radsat = np.array([0, 1 << 2, 1 << 4, 1 << 8, 1 << 11, 1 << 7], dtype=np.uint16)
    # all bands + terrain occlusion; bit 7 is unused.
    np.testing.assert_array_equal(
        mk.landsat_radsat_mask(radsat), [False, True, True, True, True, False]
    )
    np.testing.assert_array_equal(
        mk.landsat_radsat_mask(radsat, bands=[3, 5], terrain_occlusion=False),
        [False, True, True, False, False, False],
    )
    np.testing.assert_array_equal(
        mk.landsat_radsat_mask(radsat, bands=9, terrain_occlusion=False),
        [False, False, False, True, False, False],
    )


def test_landsat_radsat_mask_tm_etm():
    radsat = np.array([1 << 5, 1 << 8, 1 << 9, 1 << 11], dtype=np.uint16)
    np.testing.assert_array_equal(
        mk.landsat_radsat_mask(radsat, sensor="etm"), [True, True, True, False]
    )
    np.testing.assert_array_equal(
        mk.landsat_radsat_mask(radsat, sensor="etm", bands=["6H"], dropped_pixel=False),
        [False, True, False, False],
    )
    np.testing.assert_array_equal(
        mk.landsat_radsat_mask(radsat, sensor="TM"), [True, False, True, False]
    )
    with pytest.raises(ValueError, match="no QA_RADSAT bit"):
        mk.landsat_radsat_mask(radsat, sensor="tm", bands=["6H"])
    with pytest.raises(ValueError, match="no QA_RADSAT bit"):
        mk.landsat_radsat_mask(radsat, bands=[8])
    with pytest.raises(ValueError, match="sensor"):
        mk.landsat_radsat_mask(radsat, sensor="msi")


def test_landsat_c2_scale():
    dn = np.array([[0, 7273], [43636, 10000]], dtype=np.uint16)
    sr = mk.landsat_c2_scale(dn)
    assert sr.dtype == np.float32
    assert np.isnan(sr[0, 0])
    np.testing.assert_allclose(sr[0, 1], 7273 * 2.75e-5 - 0.2, atol=1e-6)
    np.testing.assert_allclose(sr[1], [43636 * 2.75e-5 - 0.2, 0.075], atol=1e-6)
    st = mk.landsat_c2_scale(np.array([0, 44000], dtype=np.uint16), "st")
    assert np.isnan(st[0])
    np.testing.assert_allclose(st[1], 44000 * 0.00341802 + 149.0, rtol=1e-6)
    np.testing.assert_allclose(mk.landsat_c2_scale(np.array([0]), fill=None), [-0.2])
    assert mk.landsat_c2_scale(np.array([1], dtype=np.int32)).dtype == np.float64
    with pytest.raises(ValueError, match="kind"):
        mk.landsat_c2_scale(dn, "toa")


# --------------------------------------------------------------------------- #
# Sentinel-2
# --------------------------------------------------------------------------- #
def test_scl_enum_values():
    assert [int(c) for c in mk.SCLClass] == list(range(12))
    assert mk.SCLClass.CLOUD_SHADOW == 3
    assert mk.SCLClass.CLOUD_MEDIUM == 8
    assert mk.SCLClass.CLOUD_HIGH == 9
    assert mk.SCLClass.THIN_CIRRUS == 10
    assert mk.SCLClass.SNOW_ICE == 11
    assert set(mk.DEFAULT_S2_BAD_CLASSES) == {0, 1, 3, 8, 9, 10}


def test_sentinel2_scl_mask_every_class():
    scl = np.arange(12, dtype=np.uint8).reshape(3, 4)
    mask = mk.sentinel2_scl_mask(scl)
    for code in range(12):
        assert mask.ravel()[code] == (code in {0, 1, 3, 8, 9, 10}), code
    for code in range(12):
        only = mk.sentinel2_scl_mask(scl, classes=[code])
        assert only.sum() == 1 and only.ravel()[code]
    for code, name in mk.SCL_NAMES.items():
        np.testing.assert_array_equal(
            mk.sentinel2_scl_mask(scl, classes=[name]), mk.sentinel2_scl_mask(scl, classes=[code])
        )


def test_sentinel2_scl_mask_names_aliases_enum():
    scl = np.array([3, 9, 10, 11, 2, 4], dtype=np.uint8)
    np.testing.assert_array_equal(
        mk.sentinel2_scl_mask(scl, classes={"Cloud_High", "snow", "cirrus", "cast-shadow"}),
        [False, True, True, True, True, False],
    )
    np.testing.assert_array_equal(
        mk.sentinel2_scl_mask(scl, classes=mk.SCLClass.CLOUD_SHADOW),
        [True, False, False, False, False, False],
    )
    assert not mk.sentinel2_scl_mask(scl, classes=[]).any()
    with pytest.raises(ValueError, match="Unknown SCL class"):
        mk.sentinel2_scl_mask(scl, classes=["clouds"])
    with pytest.raises(ValueError, match="0-11"):
        mk.sentinel2_scl_mask(scl, classes=[12])
    with pytest.raises(ValueError, match="0-11"):
        mk.sentinel2_scl_mask(np.array([12], dtype=np.uint8))


def test_sentinel2_scl_mask_nodata_float():
    scl = np.array([4.0, np.nan, 9.0])
    np.testing.assert_array_equal(mk.sentinel2_scl_mask(scl), [False, True, True])


def test_sentinel2_scl_mask_upsample():
    scl = np.array([[4, 9], [3, 5]], dtype=np.uint8)
    mask = mk.sentinel2_scl_mask(scl, target_shape=(4, 4))
    expected = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], bool)
    np.testing.assert_array_equal(mask, expected)
    mask60 = mk.sentinel2_scl_mask(scl, target_shape=(12, 12))  # 60 m -> 10 m
    assert mask60.shape == (12, 12) and mask60.sum() == 72
    with pytest.raises(ValueError, match="integer multiple"):
        mk.sentinel2_scl_mask(scl, target_shape=(5, 4))
    with pytest.raises(ValueError, match="integer multiple"):
        mk.sentinel2_scl_mask(scl, target_shape=(1, 1))
    with pytest.raises(TypeError, match="target_shape"):
        mk.sentinel2_scl_mask(scl, target_shape=4)
    with pytest.raises(ValueError):
        mk.sentinel2_scl_mask(scl, target_shape=(0, 4))


def test_upsample_mask():
    m = np.array([[True, False]])
    np.testing.assert_array_equal(mk.upsample_mask(m, (2, 4)), [[1, 1, 0, 0], [1, 1, 0, 0]])
    assert mk.upsample_mask(m, (1, 2)) is m
    with pytest.raises(ValueError, match="2-D"):
        mk.upsample_mask(np.zeros((1, 2, 2), bool), (2, 2))


def test_sentinel2_cloud_probability_mask():
    prob = np.array([[0, 49], [50, 100]], dtype=np.uint8)
    np.testing.assert_array_equal(
        mk.sentinel2_cloud_probability_mask(prob), [[False, False], [True, True]]
    )
    np.testing.assert_array_equal(
        mk.sentinel2_cloud_probability_mask(prob, 20), [[False, True], [True, True]]
    )
    up = mk.sentinel2_cloud_probability_mask(prob, target_shape=(4, 4))
    assert up.shape == (4, 4) and up.sum() == 8
    f = np.array([np.nan, 10.0])
    np.testing.assert_array_equal(mk.sentinel2_cloud_probability_mask(f), [True, False])
    with pytest.raises(ValueError, match="percent"):
        mk.sentinel2_cloud_probability_mask(np.array([101], dtype=np.uint8))
    with pytest.raises(ValueError, match="threshold"):
        mk.sentinel2_cloud_probability_mask(prob, 0)
    with pytest.raises(TypeError, match="threshold"):
        mk.sentinel2_cloud_probability_mask(prob, "50")


def test_sentinel2_l2a_scale():
    dn = np.array([0, 1000, 3500, 500], dtype=np.uint16)
    out = mk.sentinel2_l2a_scale(dn)
    assert out.dtype == np.float32
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [0.0, 0.25, -0.05], atol=1e-7)
    old = mk.sentinel2_l2a_scale(np.array([2500], dtype=np.uint16), offset=0)
    np.testing.assert_allclose(old, [0.25])
    # Harmonization: the same surface gives the same reflectance in both baselines.
    np.testing.assert_allclose(
        mk.sentinel2_l2a_scale(np.array([3500], np.uint16)),
        mk.sentinel2_l2a_scale(np.array([2500], np.uint16), offset=0),
    )
    np.testing.assert_allclose(mk.sentinel2_l2a_scale(np.array([0]), nodata=None), [-0.1])
    with pytest.raises(ValueError, match="quantification"):
        mk.sentinel2_l2a_scale(dn, quantification=0)
    with pytest.raises(TypeError, match="offset"):
        mk.sentinel2_l2a_scale(dn, offset=np.nan)
    masked = np.ma.MaskedArray(np.array([2000, 2000], np.uint16), mask=[True, False])
    res = mk.sentinel2_l2a_scale(masked)
    assert np.isnan(res[0]) and res[1] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# HLS
# --------------------------------------------------------------------------- #
def test_hls_fmask_mask():
    low_aer = 1 << 6
    fm = np.array(
        [
            low_aer,  # clear
            low_aer | 1 << 1,  # cloud
            low_aer | 1 << 2,  # adjacent
            low_aer | 1 << 3,  # shadow
            low_aer | 1 << 4,  # snow
            low_aer | 1 << 5,  # water
            3 << 6,  # high aerosol
            2 << 6,  # moderate aerosol
            255,  # fill
            0,  # climatology aerosol, clear
        ],
        dtype=np.uint8,
    )
    expected = [False, True, True, True, False, False, True, False, True, False]
    np.testing.assert_array_equal(mk.hls_fmask_mask(fm), expected)
    np.testing.assert_array_equal(mk.hls_fmask_mask(fm, snow=True)[4], True)
    np.testing.assert_array_equal(mk.hls_fmask_mask(fm, water=True)[5], True)
    assert mk.hls_fmask_mask(fm, aerosol="moderate")[7]
    assert not mk.hls_fmask_mask(fm, aerosol=None)[6]
    assert mk.hls_fmask_mask(fm, cloud=False, adjacent=False, shadow=False, aerosol=None)[8]
    assert not mk.hls_fmask_mask(
        fm, cloud=False, adjacent=False, shadow=False, aerosol=None, fill=False
    )[8]
    assert mk.hls_fmask_mask(np.array([1], np.uint8), cirrus=True)[0]
    with pytest.raises(ValueError, match="aerosol"):
        mk.hls_fmask_mask(fm, aerosol="climatology")
    with pytest.raises(ValueError, match="8 bits"):
        mk.hls_fmask_mask(np.array([256], dtype=np.uint16))


# --------------------------------------------------------------------------- #
# buffer_mask
# --------------------------------------------------------------------------- #
def _point(shape=(11, 11)):
    m = np.zeros(shape, bool)
    m[shape[0] // 2, shape[1] // 2] = True
    return m


def _disk(radius, shape=(11, 11)):
    yy, xx = np.indices(shape)
    cy, cx = shape[0] // 2, shape[1] // 2
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2 + 1e-9


@pytest.mark.parametrize("radius", [1, 1.5, 2, 2.5, 3, 4.2])
def test_buffer_mask_pixels_is_disk(radius):
    np.testing.assert_array_equal(mk.buffer_mask(_point(), radius), _disk(radius))


def test_buffer_mask_matches_binary_dilation():
    from scipy import ndimage

    rng = np.random.default_rng(0)
    m = rng.random((60, 50)) > 0.97
    r = 3
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = yy**2 + xx**2 <= r * r
    np.testing.assert_array_equal(mk.buffer_mask(m, r), ndimage.binary_dilation(m, structure=disk))


def test_buffer_mask_distance_metres():
    meta = {"crs": CRS.from_epsg(32633), "transform": from_origin(5e5, 4e6, 30.0, 30.0)}
    expected = _disk(2)
    np.testing.assert_array_equal(mk.buffer_mask(_point(), distance=60, pixel_size=meta), expected)
    np.testing.assert_array_equal(
        mk.buffer_mask(_point(), distance=60, pixel_size=meta["transform"]), expected
    )
    np.testing.assert_array_equal(mk.buffer_mask(_point(), distance=60, pixel_size=30), expected)
    # 10 m Sentinel-2 pixels: 25 m -> radius 2.5 px.
    np.testing.assert_array_equal(
        mk.buffer_mask(_point(), distance=25, pixel_size=(10, 10)), _disk(2.5)
    )


def test_buffer_mask_anisotropic_pixels():
    # 10 m wide, 20 m tall pixels: a 20 m buffer reaches 2 columns but only 1 row.
    out = mk.buffer_mask(_point(), distance=20, pixel_size=(10, 20))
    assert out[5, 3] and out[5, 7] and not out[5, 2]
    assert out[4, 5] and out[6, 5] and not out[3, 5]


def test_buffer_mask_geographic_rejected():
    meta = {"crs": CRS.from_epsg(4326), "transform": Affine(0.001, 0, 10, 0, -0.001, 50)}
    with pytest.raises(ValueError, match="geographic"):
        mk.buffer_mask(_point(), distance=100, pixel_size=meta)


def test_buffer_mask_blocks_and_edges():
    # Large enough to be processed in several row blocks: compare with a direct EDT.
    from scipy import ndimage

    m = np.zeros((3000, 2000), bool)
    m[2094:2096, 10] = True  # just above a block boundary (block = 2**22 // 2000 = 2097 rows)
    m[0, 1999] = True
    m[2999, 0] = True
    out = mk.buffer_mask(m, 7)
    ref = ndimage.distance_transform_edt(~m) <= 7
    np.testing.assert_array_equal(out, ref)


def test_buffer_mask_trivial_and_errors():
    m = _point()
    assert mk.buffer_mask(m, 0).sum() == 1
    assert mk.buffer_mask(m, 0.5).sum() == 1
    assert not mk.buffer_mask(np.zeros((4, 4), bool), 3).any()
    out = mk.buffer_mask(m, 1)
    assert out is not m and m.sum() == 1  # input untouched
    with pytest.raises(ValueError, match="exactly one"):
        mk.buffer_mask(m)
    with pytest.raises(ValueError, match="exactly one"):
        mk.buffer_mask(m, 1, distance=30, pixel_size=30)
    with pytest.raises(ValueError, match="pixel_size"):
        mk.buffer_mask(m, distance=30)
    with pytest.raises(ValueError, match="only used"):
        mk.buffer_mask(m, 1, pixel_size=30)
    with pytest.raises(ValueError, match=">= 0"):
        mk.buffer_mask(m, -1)
    with pytest.raises(TypeError):
        mk.buffer_mask(m, "2")
    with pytest.raises(ValueError, match="2-D"):
        mk.buffer_mask(np.zeros((2, 3, 3), bool), 1)
    with pytest.raises(ValueError, match="positive"):
        mk.buffer_mask(m, distance=30, pixel_size=0)
    with pytest.raises(TypeError, match="pixel_size"):
        mk.buffer_mask(m, distance=30, pixel_size="30m")


# --------------------------------------------------------------------------- #
# apply_mask, combine_masks, clear_fraction, valid_overlap
# --------------------------------------------------------------------------- #
def test_apply_mask_2d_and_multiband():
    img = np.arange(2 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3)
    original = img.copy()
    mask = np.zeros((3, 3), bool)
    mask[1, 1] = True
    out = mk.apply_mask(img, mask)
    assert out.dtype == np.float32
    assert np.isnan(out[:, 1, 1]).all()
    assert np.isfinite(out).sum() == 16
    np.testing.assert_array_equal(img, original)
    # Per-band mask.
    per_band = np.zeros((2, 3, 3), bool)
    per_band[0, 0, 0] = True
    out = mk.apply_mask(img, per_band)
    assert np.isnan(out[0, 0, 0]) and out[1, 0, 0] == 9
    # 2-D float64 input keeps float64 and is copied.
    f = np.ones((3, 3))
    out = mk.apply_mask(f, mask, fill=-9999)
    assert out.dtype == np.float64 and out[1, 1] == -9999 and f[1, 1] == 1
    out = mk.apply_mask(f, np.zeros((3, 3), bool))
    assert out is not f and not np.shares_memory(out, f)


def test_apply_mask_masked_array_and_errors():
    data = np.ma.MaskedArray(np.ones((2, 2), np.uint8), mask=[[True, False], [False, False]])
    out = mk.apply_mask(data, np.zeros((2, 2), bool))
    assert np.isnan(out[0, 0]) and np.isfinite(out).sum() == 3
    with pytest.raises(ValueError, match="mask shape"):
        mk.apply_mask(np.ones((4, 4)), np.zeros((2, 2), bool))
    with pytest.raises(ValueError, match="rows, cols"):
        mk.apply_mask(np.ones(4), np.zeros(4, bool))
    with pytest.raises(TypeError, match="fill"):
        mk.apply_mask(np.ones((2, 2)), np.zeros((2, 2), bool), fill=None)
    with pytest.raises(TypeError):
        mk.apply_mask(np.ones((2, 2), complex), np.zeros((2, 2), bool))
    with pytest.raises(TypeError):
        mk.apply_mask([[1, 2]], np.zeros((1, 2), bool))


def test_combine_masks():
    a = np.array([True, False, False, False])
    b = np.array([0, 1, 0, 0], dtype=np.uint8)
    c = np.array([0.0, 0.0, np.nan, 0.0])
    out = mk.combine_masks(a, None, b, c)
    np.testing.assert_array_equal(out, [True, True, True, False])
    assert out is not a
    np.testing.assert_array_equal(a, [True, False, False, False])
    with pytest.raises(ValueError, match="at least one"):
        mk.combine_masks(None)
    with pytest.raises(ValueError, match="shapes differ"):
        mk.combine_masks(a, np.zeros(3, bool))


def test_clear_fraction():
    m = np.array([True, False, False, False])
    assert mk.clear_fraction(m) == 0.75
    assert mk.clear_fraction(m, valid=np.array([True, True, False, False])) == 0.5
    assert np.isnan(mk.clear_fraction(m, valid=np.zeros(4, bool)))
    with pytest.raises(ValueError, match="valid shape"):
        mk.clear_fraction(m, valid=np.ones(3, bool))


def test_valid_overlap():
    before = np.array([[True, False], [False, False]])
    after = np.array([[False, False], [True, False]])
    ov = mk.valid_overlap(before, after)
    assert isinstance(ov, mk.MaskOverlap)
    np.testing.assert_array_equal(ov.valid, [[False, True], [False, True]])
    assert (ov.n_valid, ov.n_total, ov.fraction) == (2, 4, 0.5)
    assert ov.before_clear == 0.75 and ov.after_clear == 0.75
    fp = np.array([[True, True], [False, False]])
    ov = mk.valid_overlap(before, after, footprint=fp)
    assert (ov.n_valid, ov.n_total, ov.fraction) == (1, 2, 0.5)
    assert ov.before_clear == 0.5 and ov.after_clear == 1.0
    with pytest.raises(ValueError, match="same shape"):
        mk.valid_overlap(before, np.zeros(4, bool))


# --------------------------------------------------------------------------- #
# End to end: clouds never appear as change
# --------------------------------------------------------------------------- #
def test_clouds_do_not_appear_as_change():
    import farq

    rng = np.random.default_rng(1)
    shape = (40, 40)
    dn_before = (10000 + 50 * rng.standard_normal(shape)).astype(np.uint16)
    dn_after = (dn_before + 50 * rng.standard_normal(shape)).astype(np.uint16)
    dn_after[5:10, 5:10] = 15000  # real change
    dn_after[25:35, 20:35] = 40000  # bright cloud
    qa_before = np.full(shape, CLEAR_LAND, np.uint16)
    qa_after = qa_before.copy()
    qa_after[26:34, 21:34] = CLOUD_HIGH  # detector misses the cloud edge

    m_before = mk.landsat_qa_mask(qa_before)
    m_after = mk.buffer_mask(mk.landsat_qa_mask(qa_after), 1.5)
    ov = mk.valid_overlap(m_before, m_after)
    mask = mk.combine_masks(m_before, m_after)
    before = mk.apply_mask(mk.landsat_c2_scale(dn_before), mask)
    after = mk.apply_mask(mk.landsat_c2_scale(dn_after), mask)

    result = farq.detect_changes(before, after, threshold="otsu")
    assert not result.mask[25:35, 20:35].any()
    assert result.mask[5:10, 5:10].all()
    assert result.mask.sum() == 25
    assert ov.n_valid == int(np.isfinite(result.magnitude).sum())
