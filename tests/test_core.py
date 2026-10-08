"""Tests for farq.core: read, write, resample and validate_bands."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling

import farq
from farq import core, indices

# --------------------------------------------------------------------------- validate_bands


class TestValidateBands:
    def test_same_object_in_core_and_indices(self):
        assert core.validate_bands is indices.validate_bands
        assert farq.validate_bands is core.validate_bands

    def test_returns_float_list(self):
        a = np.ones((2, 2), dtype=np.uint16)
        b = np.ones((2, 2), dtype=np.uint8)
        out = core.validate_bands(a, b)
        assert isinstance(out, list) and len(out) == 2
        assert all(x.dtype == np.float32 for x in out)

    def test_float64_preserved_without_copy(self):
        a = np.ones((3, 3))
        (out,) = core.validate_bands(a)
        assert out is a

    def test_float32_preserved_without_copy(self):
        a = np.ones((3, 3), dtype=np.float32)
        b = np.ones((3, 3), dtype=np.float32)
        out = core.validate_bands(a, b)
        assert out[0] is a and out[1] is b

    def test_wide_ints_promote_to_float64(self):
        (out,) = core.validate_bands(np.ones(3, dtype=np.int32))
        assert out.dtype == np.float64

    def test_mixed_dtypes_common_dtype(self):
        out = core.validate_bands(np.ones(3, np.float32), np.ones(3, np.float64))
        assert {x.dtype for x in out} == {np.dtype(np.float64)}

    def test_reflectance_scale(self):
        a = np.array([10000, 5000], dtype=np.uint16)
        (out,) = core.validate_bands(a, reflectance_scale=10000)
        np.testing.assert_allclose(out, [1.0, 0.5])
        assert a[0] == 10000  # input untouched

    def test_scale_does_not_mutate_float_input(self):
        a = np.array([2.0, 4.0])
        core.validate_bands(a, reflectance_scale=2)
        np.testing.assert_array_equal(a, [2.0, 4.0])

    def test_masked_array_filled_with_nan(self):
        a = np.ma.masked_array([1, 2, 3], mask=[False, True, False])
        (out,) = core.validate_bands(a)
        assert not isinstance(out, np.ma.MaskedArray)
        assert np.isnan(out[1]) and out[0] == 1

    def test_all_nan_band_allowed(self):
        (out,) = core.validate_bands(np.full(3, np.nan))
        assert np.isnan(out).all()

    @pytest.mark.parametrize(
        ("bands", "exc"),
        [
            ((), ValueError),
            (([1, 2],), TypeError),
            ((np.array([]),), ValueError),
            ((np.array(["a", "b"]),), TypeError),
            ((np.ones((2, 2)), np.ones((2, 3))), ValueError),
        ],
    )
    def test_invalid(self, bands, exc):
        with pytest.raises(exc):
            core.validate_bands(*bands)

    @pytest.mark.parametrize(
        ("scale", "exc"), [(0, ValueError), (-1, ValueError), ("x", TypeError)]
    )
    def test_invalid_scale(self, scale, exc):
        with pytest.raises(exc):
            core.validate_bands(np.ones(2), reflectance_scale=scale)


# --------------------------------------------------------------------------- resample


class TestResample:
    def test_upsample_shape(self):
        out = farq.resample(np.array([[1, 2], [3, 4]], dtype=np.float32), (3, 3))
        assert out.shape == (3, 3) and out.dtype == np.float32

    def test_downsample_average(self):
        data = np.arange(16, dtype=np.float64).reshape(4, 4)
        out = farq.resample(data, (2, 2), Resampling.average)
        np.testing.assert_allclose(out, [[2.5, 4.5], [10.5, 12.5]])

    def test_method_by_name(self):
        data = np.arange(16, dtype=np.float64).reshape(4, 4)
        np.testing.assert_array_equal(
            farq.resample(data, (2, 2), "average"), farq.resample(data, (2, 2), Resampling.average)
        )

    def test_nearest_preserves_values(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        out = farq.resample(data, (4, 4), Resampling.nearest)
        np.testing.assert_array_equal(out, np.kron(data, np.ones((2, 2), dtype=np.uint8)))

    def test_list_and_numpy_int_target(self):
        data = np.ones((4, 4))
        assert farq.resample(data, [2, 2]).shape == (2, 2)
        assert farq.resample(data, (np.int64(2), np.int32(3))).shape == (2, 3)

    @pytest.mark.parametrize("dtype", [bool, np.int8, np.uint8, np.int16, np.int64, np.float16])
    def test_dtypes_roundtrip(self, dtype):
        data = (np.arange(16).reshape(4, 4) % 2).astype(dtype)
        out = farq.resample(data, (8, 8), Resampling.nearest)
        assert out.dtype == np.dtype(dtype)
        assert out.shape == (8, 8)

    def test_bool_nearest_values(self):
        data = np.eye(2, dtype=bool)
        out = farq.resample(data, (4, 4), "nearest")
        np.testing.assert_array_equal(out, np.kron(data, np.ones((2, 2), dtype=bool)))

    def test_3d(self):
        data = np.stack([np.zeros((4, 4)), np.ones((4, 4))]).astype(np.float32)
        out = farq.resample(data, (2, 2))
        assert out.shape == (2, 2, 2)
        np.testing.assert_allclose(out[1], 1.0)

    def test_nan_treated_as_nodata(self):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        data[1, 1] = np.nan
        out = farq.resample(data, (2, 2), "average")
        assert not np.isnan(out).any()
        assert np.isclose(out[0, 0], (0 + 1 + 4) / 3)

    def test_no_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            farq.resample(np.ones((4, 4)), (2, 2))

    def test_input_not_modified(self):
        data = np.arange(16, dtype=np.float64).reshape(4, 4)
        before = data.copy()
        farq.resample(data, (8, 8))
        np.testing.assert_array_equal(data, before)

    @pytest.mark.parametrize(
        ("array", "shape", "exc"),
        [
            ([[1, 2]], (2, 2), TypeError),
            (np.array([]), (2, 2), ValueError),
            (np.ones(4), (2, 2), ValueError),
            (np.ones((4, 4)), 4, TypeError),
            (np.ones((4, 4)), (2, 2, 2), TypeError),
            (np.ones((4, 4)), (0, 2), ValueError),
            (np.ones((4, 4)), (2.5, 2), ValueError),
            (np.ones((4, 4)), (True, 2), ValueError),
        ],
    )
    def test_invalid_inputs(self, array, shape, exc):
        with pytest.raises(exc):
            farq.resample(array, shape)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown resampling"):
            farq.resample(np.ones((4, 4)), (2, 2), "bogus")
        with pytest.raises(TypeError):
            farq.resample(np.ones((4, 4)), (2, 2), 1.5)


# --------------------------------------------------------------------------- read


class TestRead:
    def test_default_reads_band_one_2d(self, geotiff_path):
        path, data = geotiff_path
        arr, meta = farq.read(path)
        assert arr.ndim == 2
        np.testing.assert_array_equal(arr, data[0])
        assert arr.dtype == np.uint16
        assert meta["count"] == 3 and meta["crs"].to_epsg() == 32633
        assert meta["transform"].a == 30.0
        assert meta["nodata"] == 0

    def test_str_path(self, geotiff_path):
        path, data = geotiff_path
        arr, _ = farq.read(str(path))
        np.testing.assert_array_equal(arr, data[0])

    def test_select_band(self, geotiff_path):
        path, data = geotiff_path
        arr, _ = farq.read(path, band=2)
        np.testing.assert_array_equal(arr, data[1])

    def test_multiple_bands(self, geotiff_path):
        path, data = geotiff_path
        arr, _ = farq.read(path, band=[3, 1])
        assert arr.shape == (2, 6, 5)
        np.testing.assert_array_equal(arr[0], data[2])
        np.testing.assert_array_equal(arr[1], data[0])

    def test_all_bands(self, geotiff_path):
        path, data = geotiff_path
        arr, _ = farq.read(path, band=None)
        np.testing.assert_array_equal(arr, data)

    def test_masked(self, geotiff_path):
        path, data = geotiff_path
        arr, meta = farq.read(path, masked=True)
        assert arr.dtype == np.float32
        assert np.isnan(arr[0, 0])
        assert np.isnan(arr).sum() == 1
        np.testing.assert_array_equal(arr[1:], data[0, 1:])
        assert meta["dtype"] == "uint16"  # metadata describes the file

    def test_masked_all_bands(self, geotiff_path):
        path, _ = geotiff_path
        arr, _ = farq.read(path, band=None, masked=True)
        assert arr.shape == (3, 6, 5)
        assert np.isnan(arr[:, 0, 0]).all()

    @pytest.mark.parametrize("band", [0, 4, [1, 5]])
    def test_band_out_of_range(self, geotiff_path, band):
        path, _ = geotiff_path
        with pytest.raises(IndexError):
            farq.read(path, band=band)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            farq.read(tmp_path / "nonexistent.tif")

    def test_not_a_raster(self, tmp_path):
        bad = tmp_path / "bad.tif"
        bad.write_text("not a raster")
        with pytest.raises(ValueError):
            farq.read(bad)


# --------------------------------------------------------------------------- write


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
class TestWrite:
    def test_roundtrip_2d_preserves_georeference(self, tmp_path, geo_metadata):
        data = np.arange(30, dtype=np.uint16).reshape(6, 5) + 1
        path = tmp_path / "out.tif"
        farq.write(path, data, geo_metadata)
        arr, meta = farq.read(path)
        np.testing.assert_array_equal(arr, data)
        assert meta["crs"] == geo_metadata["crs"]
        assert meta["transform"] == geo_metadata["transform"]
        assert meta["nodata"] == 0

    def test_does_not_mutate_metadata(self, tmp_path, geo_metadata):
        before = dict(geo_metadata)
        farq.write(tmp_path / "a.tif", np.ones((3, 4), dtype=np.float32), geo_metadata)
        assert geo_metadata == before

    def test_updates_dtype_and_shape(self, tmp_path, geo_metadata):
        # Metadata from a uint16 6x5 file, writing a float32 3x4 NDWI result
        data = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)
        path = tmp_path / "ndwi.tif"
        farq.write(path, data, geo_metadata)
        arr, meta = farq.read(path)
        assert meta["dtype"] == "float32"
        assert (meta["height"], meta["width"], meta["count"]) == (3, 4, 1)
        np.testing.assert_array_equal(arr, data)

    def test_3d_roundtrip(self, tmp_path, geo_metadata):
        data = np.random.default_rng(0).random((3, 6, 5))
        path = tmp_path / "multi.tif"
        farq.write(path, data, geo_metadata)
        arr, meta = farq.read(path, band=None)
        assert meta["count"] == 3 and meta["dtype"] == "float64"
        np.testing.assert_array_equal(arr, data)

    def test_read_write_roundtrip_multiband(self, tmp_path, geotiff_path):
        path, data = geotiff_path
        arr, meta = farq.read(path, band=None)
        out = tmp_path / "copy.tif"
        farq.write(out, arr, meta)
        arr2, meta2 = farq.read(out, band=None)
        np.testing.assert_array_equal(arr2, data)
        assert meta2 == meta

    def test_bool_written_as_uint8(self, tmp_path):
        data = np.eye(4, dtype=bool)
        path = tmp_path / "mask.tif"
        farq.write(path, data)
        arr, meta = farq.read(path)
        assert meta["dtype"] == "uint8"
        np.testing.assert_array_equal(arr, data.astype(np.uint8))

    def test_float16_written_as_float32(self, tmp_path):
        path = tmp_path / "f16.tif"
        farq.write(path, np.ones((2, 2), dtype=np.float16))
        assert farq.read(path)[1]["dtype"] == "float32"

    def test_without_metadata_no_warning(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            farq.write(tmp_path / "plain.tif", np.ones((2, 2)))

    def test_explicit_dtype(self, tmp_path):
        path = tmp_path / "cast.tif"
        farq.write(path, np.array([[1.7, 2.2]]), dtype="uint8")
        arr, meta = farq.read(path)
        assert meta["dtype"] == "uint8"
        np.testing.assert_array_equal(arr, [[1, 2]])

    def test_nodata_override_and_profile(self, tmp_path, geo_metadata):
        data = np.array([[np.nan, 0.5]], dtype=np.float32)
        path = tmp_path / "nan.tif"
        farq.write(path, data, geo_metadata, nodata=np.nan, compress="deflate")
        with rasterio.open(path) as src:
            assert np.isnan(src.nodata)
            assert src.compression.name.lower() == "deflate"
        arr, _ = farq.read(path, masked=True)
        assert np.isnan(arr[0, 0]) and arr[0, 1] == 0.5

    def test_masked_array_input(self, tmp_path):
        data = np.ma.masked_array([[1.0, 2.0]], mask=[[True, False]])
        path = tmp_path / "masked.tif"
        farq.write(path, data)
        arr, meta = farq.read(path, masked=True)
        assert np.isnan(meta["nodata"])
        assert np.isnan(arr[0, 0]) and arr[0, 1] == 2.0

    def test_masked_int_without_nodata_raises(self, tmp_path):
        data = np.ma.masked_array([[1, 2]], mask=[[True, False]])
        with pytest.raises(ValueError, match="nodata"):
            farq.write(tmp_path / "x.tif", data)

    def test_incompatible_nodata_raises(self, tmp_path, geo_metadata):
        with pytest.raises(ValueError, match="nodata"):
            farq.write(tmp_path / "x.tif", np.ones((2, 2), np.uint8), geo_metadata, nodata=-9999)

    def test_all_nan_allowed(self, tmp_path):
        farq.write(tmp_path / "nan.tif", np.full((2, 2), np.nan))

    @pytest.mark.parametrize(
        ("data", "meta", "exc"),
        [
            ([[1, 2]], None, TypeError),
            (np.array([]), None, ValueError),
            (np.ones(3), None, ValueError),
            (np.ones((1, 1, 1, 1)), None, ValueError),
            (np.ones((2, 2)), "meta", TypeError),
        ],
    )
    def test_invalid_inputs(self, tmp_path, data, meta, exc):
        with pytest.raises(exc):
            farq.write(tmp_path / "x.tif", data, meta)

    def test_write_failure_is_runtime_error(self, tmp_path):
        with pytest.raises(RuntimeError):
            farq.write(tmp_path / "missing_dir" / "x.tif", np.ones((2, 2)))


# --------------------------------------------------------------------------- GCPs / drones


@pytest.mark.filterwarnings("error::rasterio.errors.NotGeoreferencedWarning")
class TestGCPs:
    def test_read_includes_gcps(self, gcp_tiff_path):
        path, data, gcps, crs = gcp_tiff_path
        arr, meta = farq.read(path, band=None)
        np.testing.assert_array_equal(arr, data)
        assert arr.dtype == np.uint8 and arr.shape == (3, 8, 10)
        assert len(meta["gcps"]) == 4
        assert [(g.row, g.col, g.x, g.y) for g in meta["gcps"]] == [
            (g.row, g.col, g.x, g.y) for g in gcps
        ]
        assert meta["gcps_crs"] == crs

    def test_georeferenced_file_has_no_gcp_keys(self, geotiff_path):
        _, meta = farq.read(geotiff_path[0])
        assert "gcps" not in meta and "gcps_crs" not in meta

    def test_default_read_unchanged(self, gcp_tiff_path):
        path, data, _, _ = gcp_tiff_path
        arr, _ = farq.read(path)
        np.testing.assert_array_equal(arr, data[0])

    def test_write_roundtrip_preserves_gcps(self, tmp_path, gcp_tiff_path):
        path, data, gcps, crs = gcp_tiff_path
        arr, meta = farq.read(path, band=None)
        before = dict(meta)
        out = tmp_path / "copy.tif"
        farq.write(out, arr, meta)
        assert meta == before  # not mutated
        with rasterio.open(out) as src:
            out_gcps, out_crs = src.gcps
            assert src.transform.is_identity
            np.testing.assert_array_equal(src.read(), data)
        assert out_crs == crs
        assert [(g.row, g.col, g.x, g.y) for g in out_gcps] == [
            (g.row, g.col, g.x, g.y) for g in gcps
        ]

    def test_index_result_keeps_gcps(self, tmp_path, gcp_tiff_path):
        path, _, gcps, _ = gcp_tiff_path
        rgb, meta = farq.read(path, band=None)
        out = tmp_path / "exg.tif"
        farq.write(out, indices.exg(*rgb), meta)
        result, meta2 = farq.read(out)
        assert result.dtype == np.float32 and len(meta2["gcps"]) == len(gcps)

    def test_real_transform_supersedes_gcps(self, tmp_path, gcp_tiff_path, geo_metadata):
        path, _, _, _ = gcp_tiff_path
        arr, meta = farq.read(path)
        meta = {**meta, "transform": geo_metadata["transform"], "crs": geo_metadata["crs"]}
        out = tmp_path / "rectified.tif"
        farq.write(out, arr, meta)
        with rasterio.open(out) as src:
            assert not src.gcps[0]
            assert src.transform == geo_metadata["transform"]

    def test_out_shape_rescales_gcps(self, gcp_tiff_path):
        path, _, _, _ = gcp_tiff_path
        arr, meta = farq.read(path, band=None, out_shape=(4, 5))
        assert arr.shape == (3, 4, 5)
        assert (meta["height"], meta["width"]) == (4, 5)
        corner = meta["gcps"][3]
        assert (corner.row, corner.col, corner.x) == (4, 5, 500010.0)


class TestOutShape:
    def test_downsample_2d_updates_transform(self, geotiff_path):
        path, data = geotiff_path
        arr, meta = farq.read(path, band=2, out_shape=(3, 5), resampling="nearest")
        assert arr.shape == (3, 5) and arr.dtype == np.uint16
        np.testing.assert_array_equal(arr, data[1, 1::2, :])  # pixel centres
        _, full = farq.read(path)
        assert (meta["height"], meta["width"]) == (3, 5)
        assert meta["transform"].a == full["transform"].a  # x unchanged
        assert meta["transform"].e == 2 * full["transform"].e  # y pixel twice as big
        assert meta["transform"].c == full["transform"].c
        assert meta["transform"].f == full["transform"].f

    def test_downsample_multiband_average(self, tmp_path, geo_metadata):
        data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
        path = tmp_path / "f.tif"
        farq.write(path, data, geo_metadata, nodata=None)
        arr, meta = farq.read(path, band=None, out_shape=(2, 2))
        assert arr.shape == (2, 2, 2)
        np.testing.assert_allclose(arr[0], [[2.5, 4.5], [10.5, 12.5]])
        assert meta["transform"].a == 60.0

    def test_downsample_masked(self, geotiff_path):
        path, _ = geotiff_path
        arr, _ = farq.read(path, out_shape=(6, 5), masked=True)
        assert np.isnan(arr[0, 0]) and arr.dtype == np.float32

    def test_written_downsample_georeference_consistent(self, tmp_path, geotiff_path):
        path, _ = geotiff_path
        arr, meta = farq.read(path, out_shape=(3, 5))
        out = tmp_path / "small.tif"
        farq.write(out, arr, meta)
        with rasterio.open(path) as full, rasterio.open(out) as small:
            assert full.bounds == small.bounds

    @pytest.mark.parametrize(
        ("kwargs", "exc"),
        [
            ({"out_shape": (0, 5)}, ValueError),
            ({"out_shape": 5}, TypeError),
            ({"out_shape": (3, 5), "resampling": "bogus"}, ValueError),
        ],
    )
    def test_invalid(self, geotiff_path, kwargs, exc):
        with pytest.raises(exc):
            farq.read(geotiff_path[0], **kwargs)


# --------------------------------------------------------------------------- write: nodata


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
class TestWriteNodataConventions:
    def test_float_result_does_not_inherit_integer_nodata(self, tmp_path, geo_metadata):
        # geo_metadata describes uint16 data with nodata=0; 0.0 is a valid index value.
        data = np.array([[np.nan, 0.0], [0.5, -0.5]], dtype=np.float32)
        path = tmp_path / "index.tif"
        farq.write(path, data, geo_metadata)
        arr, meta = farq.read(path, masked=True)
        assert np.isnan(meta["nodata"])
        np.testing.assert_array_equal(arr, data)

    def test_explicit_nodata_is_respected_and_nan_stored_as_it(self, tmp_path):
        data = np.array([[np.nan, 1.5]], dtype=np.float32)
        path = tmp_path / "f.tif"
        farq.write(path, data, nodata=-9999)
        with rasterio.open(path) as src:
            assert src.nodata == -9999
            np.testing.assert_array_equal(src.read(1), [[-9999, 1.5]])
        arr, _ = farq.read(path, masked=True)
        np.testing.assert_array_equal(arr, data)

    def test_float_metadata_nodata_kept_and_nan_filled(self, tmp_path, geo_metadata):
        meta = {**geo_metadata, "dtype": "float32", "nodata": -9999.0, "height": 1, "width": 2}
        farq.write(tmp_path / "f.tif", np.array([[np.nan, 2.0]], np.float32), meta)
        with rasterio.open(tmp_path / "f.tif") as src:
            assert src.nodata == -9999
            np.testing.assert_array_equal(src.read(1), [[-9999, 2.0]])

    def test_nan_to_integer_dtype_uses_nodata(self, tmp_path):
        path = tmp_path / "u8.tif"
        farq.write(path, np.array([[np.nan, 3.0, np.inf]]), dtype="uint8", nodata=255)
        arr, meta = farq.read(path)
        assert meta["nodata"] == 255
        np.testing.assert_array_equal(arr, [[255, 3, 255]])

    def test_nan_to_integer_dtype_without_nodata_raises(self, tmp_path):
        with pytest.raises(ValueError, match="NaN"):
            farq.write(tmp_path / "u8.tif", np.array([[np.nan, 3.0]]), dtype="uint8")
        assert not list(tmp_path.iterdir())

    def test_masked_float_to_integer(self, tmp_path):
        data = np.ma.masked_array([[np.nan, 2.0]], mask=[[True, False]])
        path = tmp_path / "m.tif"
        farq.write(path, data, dtype="uint8", nodata=0)
        np.testing.assert_array_equal(farq.read(path)[0], [[0, 2]])


# --------------------------------------------------------------------------- write: files


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
class TestAtomicWrite:
    def test_failed_write_keeps_existing_file_and_leaves_no_temp(self, tmp_path, monkeypatch):
        path = tmp_path / "out.tif"
        farq.write(path, np.ones((2, 2), np.uint8))
        original = path.read_bytes()

        class Boom(Exception):
            pass

        real_open = rasterio.open

        def failing_open(fp, mode="r", **kwargs):
            if mode == "w":
                dst = real_open(fp, mode, **kwargs)

                def broken_write(*args, **kw):
                    raise Boom("disk full")

                dst.write = broken_write
                return dst
            return real_open(fp, mode, **kwargs)

        monkeypatch.setattr(core.rasterio, "open", failing_open)
        with pytest.raises(RuntimeError, match="disk full"):
            farq.write(path, np.zeros((5, 5), np.uint8))
        monkeypatch.undo()
        assert path.read_bytes() == original
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.tif"]

    def test_overwrite_replaces_content(self, tmp_path):
        path = tmp_path / "out.tif"
        farq.write(path, np.ones((2, 2), np.uint8))
        farq.write(path, np.full((3, 4), 7, np.uint8))
        arr, _ = farq.read(path)
        assert arr.shape == (3, 4) and (arr == 7).all()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.tif"]

    def test_writes_through_symlink(self, tmp_path):
        target = tmp_path / "data" / "real.tif"
        target.parent.mkdir()
        farq.write(target, np.ones((2, 2), np.uint8))
        link = tmp_path / "link.tif"
        link.symlink_to(target)
        farq.write(link, np.full((2, 2), 9, np.uint8))
        assert link.is_symlink()
        assert (farq.read(target)[0] == 9).all()


class TestSizeGuards:
    def test_read_out_shape_too_large(self, geotiff_path):
        with pytest.raises(ValueError, match="FARQ_MAX_OUTPUT_PIXELS"):
            farq.read(geotiff_path[0], out_shape=(10**6, 10**6))

    def test_resample_too_large(self):
        with pytest.raises(ValueError, match="safety limit"):
            farq.resample(np.ones((2, 2), np.float32), (10**6, 10**6))

    def test_limit_configurable(self, geotiff_path, monkeypatch):
        monkeypatch.setenv("FARQ_MAX_OUTPUT_PIXELS", "10")
        with pytest.raises(ValueError, match="safety limit of 10"):
            farq.read(geotiff_path[0], out_shape=(4, 4))
        monkeypatch.setenv("FARQ_MAX_OUTPUT_PIXELS", "0")  # disabled
        assert farq.read(geotiff_path[0], out_shape=(4, 4))[0].shape == (4, 4)

    def test_bool_band_rejected(self, geotiff_path):
        with pytest.raises(TypeError, match="bool"):
            farq.read(geotiff_path[0], band=True)
