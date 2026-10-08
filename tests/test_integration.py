"""End-to-end tests across module boundaries.

These exercise the full workflows a user runs: read rasters from disk, put them on a
common grid, compute an index, detect change, summarize areas and write the result,
checking that the nodata/NaN conventions of every module agree with each other.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

import farq
from farq import change, georef, indices

UTM = CRS.from_epsg(32633)


# These tests require farq itself to be warning-free. Deprecation notices raised inside
# third-party code (rasterio's use of affine's ``*`` operator; NumPy 2.5's masked-array
# shape assignment hit by older rasterio releases) are outside farq's control.
_IGNORE_RASTERIO_DEPRECATIONS = pytest.mark.filterwarnings(
    r"ignore::DeprecationWarning:(rasterio|numpy\.ma)"
)
_IGNORE_RASTERIO_PENDING = pytest.mark.filterwarnings("ignore::PendingDeprecationWarning:rasterio")


def _write_raw(path, data, **meta):
    """Write a test fixture with rasterio directly (independent of farq.write)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        gcps = meta.pop("gcps", None)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            count=data.shape[0],
            height=data.shape[1],
            width=data.shape[2],
            dtype=data.dtype.name,
            **meta,
        ) as dst:
            if gcps is not None:
                dst.gcps = gcps
            dst.write(data)


# --------------------------------------------------------------------------- satellite


def _scene(x0, res, shape, lake_east):
    """Green/NIR uint16 scene: a lake from x=500300 to x=lake_east, land elsewhere."""
    xs = x0 + (np.arange(shape[1]) + 0.5) * res
    water = np.broadcast_to((xs >= 500300) & (xs < lake_east), shape)
    green = np.where(water, 3000, 1000).astype(np.uint16)
    nir = np.where(water, 500, 3000).astype(np.uint16)
    return np.stack([green, nir])


@pytest.fixture
def satellite_pair(tmp_path):
    """Two 2-band (green, NIR) uint16 GeoTIFFs with different grids and nodata."""
    # Before: 30 m pixels, nodata 0, lake 300 m wide.
    before = _scene(500000, 30.0, (40, 40), lake_east=500600)
    before[:, 5, 5] = 0
    b_path = tmp_path / "before.tif"
    _write_raw(
        b_path,
        before,
        crs=UTM,
        transform=from_origin(500000, 4000000, 30.0, 30.0),
        nodata=0,
    )
    # After: 15 m pixels, shifted extent, nodata 65535, the lake grew to 600 m.
    after = _scene(500090, 15.0, (80, 80), lake_east=500900)
    after[:, 10:12, 60:62] = 65535
    a_path = tmp_path / "after.tif"
    _write_raw(
        a_path,
        after,
        crs=UTM,
        transform=from_origin(500090, 3999910, 15.0, 15.0),
        nodata=65535,
    )
    return b_path, a_path


@_IGNORE_RASTERIO_DEPRECATIONS
@_IGNORE_RASTERIO_PENDING
@pytest.mark.filterwarnings("error")
def test_satellite_end_to_end(tmp_path, satellite_pair):
    b_path, a_path = satellite_pair
    before, b_meta = farq.read(b_path, band=None, masked=True)
    after, a_meta = farq.read(a_path, band=None, masked=True)
    assert before.dtype == np.float32 and np.isnan(before[:, 5, 5]).all()
    assert np.isnan(after[:, 10:12, 60:62]).all()

    b_al, a_al, meta = farq.align_pair(
        before, b_meta, after, a_meta, target="coarsest", resampling="average"
    )
    assert b_al.shape == a_al.shape == (2, meta["height"], meta["width"])
    assert meta["transform"].a == 30.0 and meta["crs"] == UTM
    assert np.isnan(meta["nodata"]) and meta["dtype"] == "float32"
    # The overlap starts at x=500090, y=3999910 (snapped onto the 30 m grid of before).
    assert (meta["transform"].c, meta["transform"].f) == (500090.0, 3999910.0)
    # Nodata survives alignment as NaN; it never becomes a valid value.
    assert np.isnan(b_al[:, 2, 2]).all()  # before's (5, 5) pixel on the new grid
    assert not np.isin(a_al[np.isfinite(a_al)], [0, 65535]).any()

    ndwi_b = indices.ndwi(*b_al)
    ndwi_a = indices.ndwi(*a_al)
    result = farq.detect_changes(ndwi_b, ndwi_a, method="difference", threshold=0.5)
    assert np.isnan(result.magnitude[2, 2]) and not result.mask[2, 2]

    xs = meta["transform"].c + (np.arange(meta["width"]) + 0.5) * 30.0
    grown = np.broadcast_to((xs >= 500600) & (xs < 500900), result.mask.shape)
    valid = np.isfinite(result.magnitude)
    np.testing.assert_array_equal(result.mask[valid], grown[valid])

    summary = farq.change_summary(result.mask, pixel_size=meta, valid=valid)
    assert summary["pixel_area_m2"] == 900.0
    assert summary["changed_pixels"] == int(grown[valid].sum())
    assert summary["changed_area_m2"] == 900.0 * summary["changed_pixels"]
    assert summary["nodata_pixels"] == int((~valid).sum()) > 0
    assert result.summary(meta) == summary

    # Write the magnitude and the mask, read them back.
    mag_path = tmp_path / "magnitude.tif"
    mask_path = tmp_path / "mask.tif"
    farq.write(mag_path, result.magnitude, meta, compress="deflate")
    farq.write(mask_path, result.mask, meta, nodata=None)
    mag, mag_meta = farq.read(mag_path, masked=True)
    np.testing.assert_array_equal(mag, result.magnitude)  # NaN positions included
    assert mag_meta["transform"] == meta["transform"] and mag_meta["crs"] == UTM
    mask, mask_meta = farq.read(mask_path)
    assert mask_meta["dtype"] == "uint8"
    np.testing.assert_array_equal(mask.astype(bool), result.mask)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


@_IGNORE_RASTERIO_DEPRECATIONS
@_IGNORE_RASTERIO_PENDING
@pytest.mark.filterwarnings("error")
def test_index_zero_survives_write_with_integer_source_metadata(tmp_path):
    """A derived float product must not inherit the integer source's nodata value.

    uint8 imagery with nodata=0 read with masked=True, NDWI computed, written with the
    source metadata: valid NDWI values of exactly 0 must not turn into nodata.
    """
    data = np.array([[[0, 10], [20, 30]], [[0, 10], [5, 40]]], dtype=np.uint8)
    src = tmp_path / "src.tif"
    _write_raw(src, data, crs=UTM, transform=from_origin(0, 0, 10, 10), nodata=0)
    green, meta = farq.read(src, 1, masked=True)
    nir, _ = farq.read(src, 2, masked=True)
    ndwi = indices.ndwi(green, nir)
    assert ndwi[0, 1] == 0.0 and np.isnan(ndwi[0, 0])

    out = tmp_path / "ndwi.tif"
    farq.write(out, ndwi, meta)
    back, back_meta = farq.read(out, masked=True)
    assert np.isnan(back_meta["nodata"])
    np.testing.assert_array_equal(back, ndwi)


# --------------------------------------------------------------------------- drone


def _drone_flight(path, offset_x, plant_cols):
    """3-band uint8 RGB drone image (10 cm pixels) referenced only by 4 GCPs."""
    rows, cols = 40, 50
    rgb = np.empty((3, rows, cols), np.uint8)
    rgb[0], rgb[1], rgb[2] = 120, 100, 80  # bare soil
    rgb[:, :, plant_cols] = np.array([40, 160, 40], np.uint8)[:, None, None]
    x0, y0, res = 500000.0 + offset_x, 4000000.0, 0.1
    gcps = [
        GroundControlPoint(row=r, col=c, x=x0 + c * res, y=y0 - r * res, z=0.0, id=str(i))
        for i, (r, c) in enumerate([(0, 0), (0, cols), (rows, 0), (rows, cols)], 1)
    ]
    _write_raw(path, rgb, gcps=(gcps, UTM))
    return rgb


@pytest.fixture
def drone_pair(tmp_path):
    # Second flight is shifted 0.5 m east; vegetation spread from 10 to 20 columns.
    b_path, a_path = tmp_path / "flight1.tif", tmp_path / "flight2.tif"
    _drone_flight(b_path, 0.0, slice(10, 20))
    _drone_flight(a_path, 0.5, slice(5, 25))  # x = 500001.0 .. 500003.0 after the shift
    return b_path, a_path


@_IGNORE_RASTERIO_DEPRECATIONS
@_IGNORE_RASTERIO_PENDING
@pytest.mark.filterwarnings("error")
def test_drone_gcp_end_to_end(tmp_path, drone_pair):
    b_path, a_path = drone_pair
    before, b_meta = farq.read(b_path, band=None)
    after, a_meta = farq.read(a_path, band=None)
    assert before.dtype == np.uint8 and b_meta["gcps"] and b_meta["transform"].is_identity
    # The raw metadata has no ground pixel size: areas must not be silently wrong.
    with pytest.raises(ValueError, match="geotransform"):
        farq.change_summary(np.zeros((40, 50), bool), pixel_size=b_meta)

    b_al, a_al, meta = farq.align_pair(
        before, b_meta, after, a_meta, resolution=0.1, resampling="nearest"
    )
    assert b_al.shape == a_al.shape and b_al.shape[0] == 3
    assert meta["crs"] == UTM
    np.testing.assert_allclose(georef.pixel_size(meta), (0.1, 0.1))

    exg_b = indices.exg(*b_al)
    exg_a = indices.exg(*a_al)
    vari_b = indices.vari(*b_al)
    assert np.isfinite(vari_b).any()
    result = farq.detect_changes(exg_b, exg_a, threshold=0.3)

    xs = meta["transform"].c + (np.arange(meta["width"]) + 0.5) * 0.1
    veg_b = (xs > 500001.0) & (xs < 500002.0)
    veg_a = (xs > 500001.0) & (xs < 500003.0)
    expected = np.broadcast_to(veg_a & ~veg_b, result.mask.shape)
    valid = np.isfinite(result.magnitude)
    assert valid.all()  # the overlap is fully covered by both flights
    np.testing.assert_array_equal(result.mask, expected)

    summary = farq.change_summary(result.mask, pixel_size=meta, valid=valid)
    assert summary["pixel_area_m2"] == pytest.approx(0.01)
    assert summary["changed_area_m2"] == pytest.approx(0.01 * expected.sum())

    out = tmp_path / "exg_change.tif"
    farq.write(out, result.magnitude, meta)
    back, back_meta = farq.read(out, masked=True)
    np.testing.assert_array_equal(back, result.magnitude)
    assert back_meta["transform"] == meta["transform"]
    assert "gcps" not in back_meta


@_IGNORE_RASTERIO_DEPRECATIONS
@_IGNORE_RASTERIO_PENDING
@pytest.mark.filterwarnings("error")
def test_drone_rectify_then_align(tmp_path, drone_pair):
    b_path, a_path = drone_pair
    rect_b = tmp_path / "flight1_rect.tif"
    arr, rmeta = georef.rectify(b_path, rect_b, resolution=0.1, resampling="nearest")
    assert arr.dtype == np.float32 and np.isnan(rmeta["nodata"])
    rectified, meta_b = farq.read(rect_b, band=None, masked=True)
    np.testing.assert_array_equal(rectified, arr)
    assert not meta_b["transform"].is_identity

    after, meta_a = farq.read(a_path, band=None)
    b_al, a_al, meta = farq.align_pair(rectified, meta_b, after, meta_a, resampling="nearest")
    result = farq.detect_changes(indices.exg(*b_al), indices.exg(*a_al), threshold=0.3)
    assert result.mask.any()
    farq.write(tmp_path / "out.tif", result.mask, meta, nodata=None)
    mask, _ = farq.read(tmp_path / "out.tif")
    np.testing.assert_array_equal(mask.astype(bool), result.mask)


# --------------------------------------------------------------------------- nodata


@_IGNORE_RASTERIO_DEPRECATIONS
@_IGNORE_RASTERIO_PENDING
@pytest.mark.filterwarnings("error")
def test_align_pair_with_different_integer_nodata_values():
    meta = {
        "crs": UTM,
        "transform": from_origin(0, 0, 10, 10),
        "width": 2,
        "height": 2,
        "dtype": "uint8",
    }
    before = np.array([[1, 2], [3, 0]], np.uint8)
    after = np.array([[255, 7], [8, 9]], np.uint8)
    after_meta = {**meta, "nodata": 255, "transform": from_origin(10, 0, 10, 10)}
    b, a, out_meta = farq.align_pair(before, {**meta, "nodata": 0}, after, after_meta)
    # One metadata dict describes both outputs, so they must share a nodata value.
    assert np.isnan(out_meta["nodata"]) and b.dtype == a.dtype == np.float32
    np.testing.assert_array_equal(b, [[2.0], [np.nan]])
    np.testing.assert_array_equal(a, [[np.nan], [8.0]])


@_IGNORE_RASTERIO_DEPRECATIONS
@_IGNORE_RASTERIO_PENDING
@pytest.mark.filterwarnings("error")
def test_change_module_accepts_analysis_inputs():
    """Masks from change feed analysis (pixel sizes in metres there, areas in km²)."""
    before = np.zeros((10, 10), np.float32)
    after = before.copy()
    after[2:5, 2:5] = 1.0
    after[0, 0] = np.nan
    result = farq.detect_changes(before, after, threshold=0.5)
    stats = farq.water_stats(result.mask, pixel_size=30)
    assert stats["total_area"] == pytest.approx(9 * 900 / 1e6)
    summary = change.change_summary(result.mask, pixel_size=30)
    assert summary["changed_area_km2"] == pytest.approx(stats["total_area"])
