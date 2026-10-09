"""End-to-end pipelines across the 0.3 modules (masking, radiometry, elevation,
tiling, vector) on synthetic GeoTIFFs, plus regression tests for defects found
while integrating them.

The pipelines check that the modules agree on their shared conventions: pixel area
from the raster metadata, refusal of geographic CRSs wherever areas are computed,
NaN as nodata, ``(bands, H, W)`` stacks, change = after - before, and the uint8
change masks written by :func:`farq.detect_changes_file` (1 change, 0 no change,
:data:`farq.CHANGE_NODATA` invalid) being read back and summarized/vectorized with
the same counts and areas.
"""

from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from rasterio.transform import from_origin

import farq
from farq import elevation, masking, radiometry, tiling, vector

UTM = CRS.from_epsg(32633)


def _write(path, data, *, transform=None, crs=UTM, nodata=None, gcps=None, **extra):
    data = np.asarray(data)
    stack = data[np.newaxis] if data.ndim == 2 else data
    meta = {
        "driver": "GTiff",
        "dtype": stack.dtype.name,
        "count": stack.shape[0],
        "height": stack.shape[1],
        "width": stack.shape[2],
        "nodata": nodata,
        **extra,
    }
    if gcps is None:
        meta.update(crs=crs, transform=transform)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(stack)
            if gcps is not None:
                dst.gcps = (gcps, crs)
    return path


def _ogr_available() -> bool:
    try:
        import pyogrio  # noqa: F401
    except ImportError:
        return False
    return True


def _read_vector(path):
    """(geometries as shapely objects, properties dict of columns) of a written file."""
    shapely = pytest.importorskip("shapely")
    if str(path).endswith(".geojson"):
        with open(path, encoding="utf-8") as fh:
            fc = json.load(fh)
        geoms = [shapely.geometry.shape(f["geometry"]) for f in fc["features"]]
        names = fc["features"][0]["properties"].keys() if fc["features"] else []
        cols = {n: [f["properties"][n] for f in fc["features"]] for n in names}
        return geoms, cols, fc.get("crs")
    from pyogrio.raw import read

    meta, _, wkb, fields = read(path)
    geoms = list(shapely.from_wkb(wkb))
    cols = {n: list(c) for n, c in zip(meta["fields"], fields)}
    return geoms, cols, meta["crs"]


# --------------------------------------------------------------------------- #
# Landsat-style: QA mask -> buffer -> apply -> align -> MNDWI -> PIF -> vector
# --------------------------------------------------------------------------- #
def _landsat_scene(rng, shape, water, cloud=None, gain=1.0, offset=0.0):
    """uint16 Collection 2 SR digital numbers (green, swir1) and a QA_PIXEL band."""
    refl_green = rng.uniform(0.05, 0.12, shape)
    refl_swir = rng.uniform(0.15, 0.30, shape)
    refl_green[water] = rng.uniform(0.08, 0.10, water.sum())
    refl_swir[water] = rng.uniform(0.01, 0.02, water.sum())
    stack = np.stack([refl_green, refl_swir]) * gain + offset
    dn = np.round((stack + 0.2) / 2.75e-05).astype(np.uint16)
    qa = np.full(shape, 21824, dtype=np.uint16)  # clear land (Landsat 8)
    qa[water] = 21952  # clear water
    if cloud is not None:
        dn[:, cloud] = 30000  # bright cloud tops: a huge false "change"
        qa[cloud] = 22280  # high-confidence cloud
    dn[:, :, :2] = 0  # fill columns (SR fill value 0)
    qa[:, :2] = 1  # QA fill bit
    return dn, qa


def test_landsat_pipeline_mask_align_pif_vector(tmp_path, rng):
    h, w = 90, 110
    t_before = from_origin(500000.0, 4000000.0, 30.0, 30.0)
    t_after = from_origin(500000.0 + 3 * 30.0, 4000000.0 - 2 * 30.0, 30.0, 30.0)  # shifted
    yy, xx = np.mgrid[:h, :w]
    lake = (yy - 40) ** 2 + (xx - 50) ** 2 < 12**2
    # The lake grows by a 10 x 14 px bay in the after scene (in before-grid coordinates).
    bay = np.zeros((h, w), bool)
    bay[25:35, 64:78] = True
    cloud_b = np.zeros((h, w), bool)
    cloud_b[70:80, 10:22] = True
    dn_b, qa_b = _landsat_scene(rng, (h, w), lake, cloud_b)
    # After: different gain/offset (sensor/illumination) on the after grid.
    lake_a = np.roll(np.roll(lake | bay, -2, axis=0), -3, axis=1)
    cloud_a = np.zeros((h, w), bool)
    cloud_a[5:15, 85:100] = True
    dn_a, qa_a = _landsat_scene(rng, (h, w), lake_a, cloud_a, gain=1.15, offset=0.01)

    paths = {}
    for name, dn, qa, t in (("b", dn_b, qa_b, t_before), ("a", dn_a, qa_a, t_after)):
        paths[name] = (
            _write(tmp_path / f"sr_{name}.tif", dn, transform=t, nodata=0),
            _write(tmp_path / f"qa_{name}.tif", qa, transform=t, nodata=1),
        )

    stacks = {}
    for name, (sr_path, qa_path) in paths.items():
        dn, meta = farq.read(sr_path, band=None)
        qa, qa_meta = farq.read(qa_path, masked=True)  # float, fill -> NaN
        assert dn.shape == (2, h, w) and qa.dtype.kind == "f"
        mask = masking.landsat_qa_mask(qa)
        assert mask[:, :2].all()  # NaN (QA fill) is masked
        mask = masking.buffer_mask(mask, distance=60.0, pixel_size=qa_meta)  # 2 px
        sr = masking.landsat_c2_scale(dn)  # fill (0) -> NaN
        stacks[name] = (masking.apply_mask(sr, mask), meta)
        assert stacks[name][0].shape == (2, h, w)

    before, after, meta = farq.align_pair(*stacks["b"], *stacks["a"])
    assert before.shape == after.shape and before.shape[0] == 2
    assert meta["transform"] == stacks["b"][1]["transform"] @ rasterio.Affine.translation(3, 2)

    mndwi_b = farq.mndwi(before[0], before[1])
    mndwi_a = farq.mndwi(after[0], after[1])
    valid = np.isfinite(mndwi_b) & np.isfinite(mndwi_a)
    result = farq.detect_changes(mndwi_b, mndwi_a, normalize="pif", min_size=4)
    assert np.array_equal(np.isfinite(result.magnitude), valid)  # NaN = nodata everywhere
    assert not result.mask[~valid].any()
    # The bay (shifted onto the aligned grid) is detected; clouds are not.
    bay_aligned = bay[2 : 2 + before.shape[1], 3 : 3 + before.shape[2]]
    detected = result.mask & bay_aligned
    assert detected.sum() >= 0.8 * (bay_aligned & valid).sum()
    assert (result.mask & ~bay_aligned).sum() <= 5
    # Direction: water gained, so MNDWI increased (after - before > 0) on the bay.
    assert np.nanmedian((mndwi_a - mndwi_b)[detected]) > 0

    summary = result.summary(meta)
    assert summary["pixel_area_m2"] == 900.0
    out = tmp_path / ("changes.gpkg" if _ogr_available() else "changes.geojson")
    features = farq.changes_to_vector(result, meta, out)
    assert sum(f["properties"]["pixel_count"] for f in features) == summary["changed_pixels"]
    assert math.isclose(
        sum(f["properties"]["area_m2"] for f in features), summary["changed_area_m2"]
    )

    geoms, cols, crs = _read_vector(out)
    assert len(geoms) == len(features)
    assert sum(cols["pixel_count"]) == summary["changed_pixels"]
    assert math.isclose(sum(cols["area_m2"]), summary["changed_area_m2"])
    if out.suffix == ".gpkg":  # native UTM coordinates: geometry areas are exact
        assert "32633" in crs
        assert math.isclose(sum(g.area for g in geoms), summary["changed_area_m2"])
    assert np.isfinite(cols["mean_magnitude"]).all()


def test_geographic_crs_refused_wherever_areas_are_computed(tmp_path):
    wgs = {"crs": CRS.from_epsg(4326), "transform": from_origin(10.0, 50.0, 1e-4, 1e-4)}
    mask = np.zeros((6, 6), bool)
    mask[2:4, 2:4] = True
    dem = np.add.outer(np.arange(6.0), np.arange(6.0))
    with pytest.raises(ValueError, match="geographic"):
        farq.change_summary(mask, pixel_size=wgs)
    with pytest.raises(ValueError, match="geographic"):
        vector.polygonize(mask, wgs)
    with pytest.raises(ValueError, match="geographic"):
        masking.buffer_mask(mask, distance=30.0, pixel_size=wgs)
    with pytest.raises(ValueError, match="geographic"):
        elevation.volume_change(dem, wgs)
    with pytest.raises(ValueError, match="geographic"):
        elevation.slope(dem, wgs)
    a = _write(tmp_path / "a.tif", dem.astype(np.float32), **wgs)
    b = _write(tmp_path / "b.tif", dem.astype(np.float32) + 1, **wgs)
    with pytest.raises(ValueError, match="geographic"):
        farq.detect_changes_file(a, b, tmp_path / "c.tif")
    assert not (tmp_path / "c.tif").exists()
    # An explicit pixel size in metres is the documented escape hatch.
    s = farq.detect_changes_file(a, b, tmp_path / "c.tif", threshold=0.5, pixel_size=10.0)
    assert s["changed_area_m2"] == 3600.0


# --------------------------------------------------------------------------- #
# Large raster: tiled change detection with clouds -> uint8 mask -> vector/summary
# --------------------------------------------------------------------------- #
@pytest.fixture
def cloudy_pair(tmp_path):
    rng = np.random.default_rng(7)
    h, w = 450, 520  # several 128 px blocks in each direction
    t = from_origin(600000.0, 5000000.0, 10.0, 10.0)
    before = rng.normal(0.1, 0.03, (h, w)).astype(np.float32)
    after = before + rng.normal(0.0, 0.03, (h, w)).astype(np.float32)
    after[100:180, 120:300] += 0.6  # crosses block borders
    after[300:305, 400:405] += 0.6  # small region (25 px)
    after[20:22, 20:22] += 0.6  # tiny region (4 px), removed by min_size
    yy, xx = np.mgrid[:h, :w]
    cloud_b = (yy - 150) ** 2 + (xx - 260) ** 2 < 40**2  # partly over the change
    cloud_a = (yy - 380) ** 2 + (xx - 100) ** 2 < 50**2
    before = masking.apply_mask(before, cloud_b)
    after = masking.apply_mask(after, masking.buffer_mask(cloud_a, 2))
    before[:, :64] = np.nan  # outside the footprint: whole blocks of NaN
    after[:, :64] = np.nan
    pb = _write(tmp_path / "before.tif", before, transform=t, nodata=float("nan"), tiled=True)
    pa = _write(tmp_path / "after.tif", after, transform=t, nodata=float("nan"), tiled=True)
    return pb, pa, before, after


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_tiled_change_detection_to_vector_and_summary(tmp_path, cloudy_pair, n_jobs):
    pb, pa, before, after = cloudy_pair
    out = tmp_path / "change.tif"
    s = farq.detect_changes_file(
        pb,
        pa,
        out,
        min_size=10,
        fill_holes=True,
        sample_size=None,
        block_size=128,
        n_jobs=n_jobs,
    )
    json.dumps(s, allow_nan=False)
    expected = farq.detect_changes(before, after, min_size=10, fill_holes=True)
    assert s["threshold_exact"] and math.isclose(s["threshold"], expected.threshold)

    # Raw read: uint8 with nodata 255 (CHANGE_NODATA).
    raw, meta = farq.read(out)
    assert raw.dtype == np.uint8 and meta["nodata"] == farq.CHANGE_NODATA
    assert np.array_equal(raw == 1, expected.mask)
    assert np.array_equal(raw != farq.CHANGE_NODATA, np.isfinite(expected.magnitude))
    exp_summary = expected.summary(meta)
    for key in ("valid_pixels", "nodata_pixels", "changed_pixels", "changed_area_m2"):
        assert s[key] == exp_summary[key], key

    # change_summary on the raw class map, with the sentinel as nodata.
    cs = farq.change_summary(raw, pixel_size=meta, nodata=farq.CHANGE_NODATA)
    assert cs["valid_pixels"] == s["valid_pixels"]
    assert cs["classes"][1]["area_m2"] == s["changed_area_m2"]
    assert farq.CHANGE_NODATA not in cs["classes"]
    # Masked read: the sentinel becomes NaN and is treated as nodata as well.
    as_float, _ = farq.read(out, masked=True)
    assert np.isnan(as_float).sum() == s["nodata_pixels"]
    cs2 = farq.change_summary(as_float, pixel_size=meta)
    assert cs2["valid_pixels"] == s["valid_pixels"]
    assert cs2["classes"][1.0]["pixels"] == s["changed_pixels"]
    stats = farq.summarize_file(out, bins=None, block_size=128)
    assert stats["valid"] == s["valid_pixels"]
    assert math.isclose(stats["mean"] * 100, s["changed_percent"])

    # Vector export of the written mask: the 255 sentinel never becomes a polygon.
    path = tmp_path / ("change.fgb" if _ogr_available() else "change.geojson")
    for data in (raw, as_float):
        feats = farq.changes_to_vector(data, meta, path, values=1)
        assert all(f["properties"]["value"] == 1 for f in feats)
        assert sum(f["properties"]["pixel_count"] for f in feats) == s["changed_pixels"]
        assert math.isclose(sum(f["properties"]["area_m2"] for f in feats), s["changed_area_m2"])
    # The same through a boolean mask + validity (the labels are then "changed").
    feats = farq.polygonize(raw == 1, meta, valid=raw != farq.CHANGE_NODATA)
    assert {f["properties"]["label"] for f in feats} == {"changed"}
    assert math.isclose(sum(f["properties"]["area_m2"] for f in feats), s["changed_area_m2"])
    _, cols, _ = _read_vector(path)
    assert sum(cols["pixel_count"]) == s["changed_pixels"]
    assert not [n for n in os.listdir(tmp_path) if ".tmp" in n]  # no temp files left


def test_tiled_all_nan_inputs(tmp_path):
    t = from_origin(0.0, 1000.0, 1.0, 1.0)
    nan = np.full((70, 90), np.nan, dtype=np.float32)
    pb = _write(tmp_path / "b.tif", nan, transform=t, nodata=float("nan"))
    pa = _write(tmp_path / "a.tif", nan, transform=t, nodata=float("nan"))
    with pytest.raises(ValueError, match="no valid pixels"):
        farq.detect_changes_file(pb, pa, tmp_path / "c.tif", block_size=32)
    assert sorted(os.listdir(tmp_path)) == ["a.tif", "b.tif"]
    s = farq.detect_changes_file(pb, pa, tmp_path / "c.tif", threshold=0.1, block_size=32)
    assert s["valid_pixels"] == 0 and s["changed_percent"] == 0.0
    raw, meta = farq.read(tmp_path / "c.tif")
    assert (raw == farq.CHANGE_NODATA).all()
    assert farq.changes_to_vector(raw, meta) == []
    assert farq.change_summary(raw, pixel_size=meta, nodata=255)["valid_pixels"] == 0


# --------------------------------------------------------------------------- #
# Drone DSM: GCP-only DEM -> align -> co-register -> DoD -> LoD -> volumes -> vector
# --------------------------------------------------------------------------- #
def _terrain(x, y):
    """Smooth hills with slopes in every direction (metres)."""
    return 100.0 + 3.0 * np.sin(x / 4.0) * np.cos(y / 5.0) + 2.0 * np.cos((x + y) / 6.0) + 0.02 * x


def test_drone_dsm_volume_pipeline(tmp_path):
    res = 0.25
    h, w = 160, 180
    x0, y0 = 400000.0, 3000040.0
    t = from_origin(x0, y0, res, res)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = x0 + cols * res, y0 - rows * res
    before = _terrain(x - x0, y0 - y).astype(np.float32)

    # After survey: the same terrain, 0.5 m east / 0.25 m north and 0.15 m up (GNSS
    # error), plus a stockpile (fill) and a pit (cut). It is referenced by GCPs only.
    dx, dy, dz = 0.5, 0.25, 0.15
    after = _terrain(x - dx - x0, y0 - (y - dy)) + dz
    pile = (rows - 50) ** 2 + (cols - 60) ** 2 < 15**2
    pit = (np.abs(rows - 115) < 10) & (np.abs(cols - 130) < 12)
    after = after + np.where(pile, 2.0 - np.hypot(rows - 50, cols - 60) / 10.0, 0.0)
    after = (after - np.where(pit, 1.5, 0.0)).astype(np.float32)
    after[:3, :] = np.nan  # a strip without data
    gcps = [
        GroundControlPoint(row=r, col=c, x=x0 + c * res, y=y0 - r * res, z=0.0, id=str(i))
        for i, (r, c) in enumerate([(0, 0), (0, w), (h, 0), (h, w), (h / 2, w / 2)])
    ]
    pb = _write(tmp_path / "dsm_before.tif", before, transform=t, nodata=-9999.0)
    pa = _write(tmp_path / "dsm_after.tif", after, gcps=gcps, nodata=float("nan"))

    dem_b, meta_b = farq.read(pb, masked=True)
    dem_a, meta_a = farq.read(pa, masked=True)
    assert meta_a["transform"].is_identity and meta_a.get("gcps")
    with pytest.raises(ValueError, match=r"GCP|geotransform"):
        elevation.volume_change(dem_a, meta_a)  # no ground pixel size yet
    with pytest.raises(ValueError, match=r"GCP|geotransform"):
        vector.polygonize(np.isfinite(dem_a), meta_a)

    dem_b, dem_a, meta = farq.align_pair(dem_b, meta_b, dem_a, meta_a)
    assert dem_b.shape == dem_a.shape == (h, w)
    works = masking.buffer_mask(pile | pit, 4)
    co = elevation.coregister_dem(dem_b, dem_a, meta, stable_mask=~works)
    assert co.dx == pytest.approx(dx, abs=0.05)
    assert co.dy == pytest.approx(dy, abs=0.05)
    assert co.dz == pytest.approx(dz, abs=0.02)
    assert co.nmad_after < co.nmad_before

    dod = elevation.elevation_change(dem_b, co.dem)
    offset, _ = elevation.vertical_offset(dem_b, co.dem, stable_mask=~works)
    assert abs(offset) < 0.02
    lod = elevation.level_of_detection(0.03)
    sig = elevation.significant_change(dod, lod)
    vol = elevation.volume_change(dod, meta, lod=lod, sigma=0.03 * math.sqrt(2))
    json.dumps(vol.to_dict(), allow_nan=False)
    pile_volume = float(np.sum(np.clip(2.0 - np.hypot(rows - 50, cols - 60) / 10.0, 0, None)[pile]))
    assert vol.fill_m3 == pytest.approx(pile_volume * res * res, rel=0.05)
    assert vol.cut_m3 == pytest.approx(1.5 * pit.sum() * res * res, rel=0.05)

    # Polygonize the significant cut (2) / fill (1) zones and measure each polygon.
    zones = np.where(sig > 0, 1.0, np.where(sig < 0, 2.0, 0.0))
    zones[np.isnan(sig)] = np.nan
    feats = vector.polygonize(zones, meta, values=[1.0, 2.0], labels={1.0: "fill", 2.0: "cut"})
    fill_sum = cut_sum = area_sum = 0.0
    for f in feats:
        inside = rasterize(
            [(f["geometry"], 1)], out_shape=dod.shape, transform=meta["transform"], dtype="uint8"
        ).astype(bool)
        assert inside.sum() == f["properties"]["pixel_count"]
        v = elevation.volume_change(dod, meta, lod=lod, mask=inside)
        fill_sum += v.fill_m3
        cut_sum += v.cut_m3
        area_sum += f["properties"]["area_m2"]
        if f["properties"]["label"] == "fill":
            assert v.cut_m3 == 0.0 and v.fill_m3 > 0
        else:
            assert v.fill_m3 == 0.0 and v.cut_m3 > 0
    assert fill_sum == pytest.approx(vol.fill_m3, rel=1e-9)
    assert cut_sum == pytest.approx(vol.cut_m3, rel=1e-9)
    assert area_sum == pytest.approx(vol.fill_area_m2 + vol.cut_area_m2, rel=1e-12)
    assert {f["properties"]["label"] for f in feats} == {"fill", "cut"}


# --------------------------------------------------------------------------- #
# IR-MAD through detect_changes vs radiometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nodata", [None, -9999.0])
def test_detect_changes_irmad_matches_irmad_change(nodata):
    rng = np.random.default_rng(3)
    before = rng.normal(0.3, 0.05, size=(4, 60, 70)).astype(np.float32)
    after = (1.3 * before - 0.02 + 0.01 * rng.normal(size=before.shape)).astype(np.float32)
    after[:, 10:25, 30:50] += 0.2
    if nodata is not None:
        before[:, :5, :5] = nodata
    else:
        before[1, :5, :5] = np.nan
    for alpha in (0.01, 0.001):
        res = farq.detect_changes(before, after, method="irmad", alpha=alpha, nodata=nodata)
        mask = radiometry.irmad_change(before, after, alpha=alpha, nodata=nodata)
        assert np.array_equal(res.mask, mask)
        assert res.mask[10:25, 30:50].mean() > 0.95
        assert np.isnan(res.magnitude[:5, :5]).all() and not res.mask[:5, :5].any()
    # Masked arrays behave like NaN.
    masked = np.ma.masked_invalid(np.where(before == (nodata or np.nan), np.nan, before))
    res = farq.detect_changes(masked, after, method="irmad")
    assert np.array_equal(res.mask, radiometry.irmad_change(masked, after))


# --------------------------------------------------------------------------- #
# Regression tests
# --------------------------------------------------------------------------- #
def test_tiling_threads_do_not_leak_warning_filters(tmp_path):
    """Worker threads must not enter warnings.catch_warnings (not thread-safe): it
    left a global 'ignore NotGeoreferencedWarning' filter behind after the call."""
    t = from_origin(0.0, 64.0, 1.0, 1.0)
    path = _write(
        tmp_path / "a.tif",
        np.ones((64, 64), np.float32),
        transform=t,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    )
    tiling.summarize_file(path, block_size=16)  # warm-up: lazy imports add filters
    before = list(warnings.filters)
    for _ in range(5):
        tiling.map_blocks(lambda a: a, [path], tmp_path / "o.tif", block_size=16, n_jobs=8)
        tiling.summarize_file(path, block_size=16, n_jobs=8)
        assert warnings.filters == before


def test_tiling_reader_handles_closed_on_error(tmp_path):
    t = from_origin(0.0, 64.0, 1.0, 1.0)
    path = _write(tmp_path / "a.tif", np.ones((64, 64), np.float32), transform=t)
    with pytest.raises(TypeError, match="progress"):
        tiling.map_blocks(lambda a: a, [path], tmp_path / "o.tif", progress=1)

    def boom(a):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        tiling.map_blocks(boom, [path], tmp_path / "o.tif", block_size=16, n_jobs=4)
    assert sorted(os.listdir(tmp_path)) == ["a.tif"]


def test_detect_changes_file_rejects_same_mask_and_magnitude_path(tmp_path):
    t = from_origin(0.0, 10.0, 1.0, 1.0)
    a = _write(tmp_path / "a.tif", np.zeros((10, 10), np.float32), transform=t)
    b = _write(tmp_path / "b.tif", np.ones((10, 10), np.float32), transform=t)
    with pytest.raises(ValueError, match="magnitude_path"):
        farq.detect_changes_file(
            a, b, tmp_path / "c.tif", threshold=0.5, magnitude_path=str(tmp_path / "c.tif")
        )
    assert not (tmp_path / "c.tif").exists()


def test_block_does_not_shadow_tuple_index():
    block = next(tiling.iter_windows(4, 4, 2))
    assert block.number == 0
    assert block.index(block.read_window) == 1  # tuple.index still works
    assert block._fields == ("number", "read_window", "write_window", "inner")


def test_elevation_results_are_strict_json():
    dem = np.full((20, 20), 50.0)
    pile = np.zeros((20, 20), bool)
    pile[5:15, 5:15] = True
    dem[pile] += 2.0
    r = elevation.stockpile_volume(dem, pile, 0.5, base=48.0)  # base_rmse_m is NaN
    assert math.isnan(r.base_rmse_m)
    d = r.to_dict()
    assert d["base_rmse_m"] is None
    json.dumps(d, allow_nan=False)
    assert math.copysign(1.0, r.below_base_m3) == 1.0  # no -0.0
    v = elevation.volume_change(np.full((3, 3), 0.5), 1.0, sigma=np.full((3, 3), np.nan))
    assert math.isnan(v.uncertainty_m3) and v.to_dict()["uncertainty_m3"] is None
    assert math.copysign(1.0, v.cut_m3) == 1.0
    json.dumps(v.to_dict(), allow_nan=False)


def test_user_sized_allocations_are_guarded(monkeypatch):
    monkeypatch.setenv("FARQ_MAX_OUTPUT_PIXELS", "1000000")
    with pytest.raises(ValueError, match="FARQ_MAX_OUTPUT_PIXELS"):
        masking.upsample_mask(np.zeros((2, 2), bool), (2 * 10**6, 2 * 10**6))
    with pytest.raises(ValueError, match="FARQ_MAX_OUTPUT_PIXELS"):
        masking.sentinel2_scl_mask(np.zeros((2, 2), np.uint8), target_shape=(4000, 4000))
    x = np.random.default_rng(0).random((5, 5))
    with pytest.raises(ValueError, match="FARQ_MAX_OUTPUT_PIXELS"):
        radiometry.histogram_match(x, x, n_quantiles=10**13)
    assert masking.upsample_mask(np.zeros((2, 2), bool), (4, 6)).shape == (4, 6)


@pytest.mark.parametrize(
    "transform",
    [lambda a: a.astype(a.dtype.newbyteorder()), lambda a: np.asfortranarray(a), lambda a: a.T.T],
    ids=["big-endian", "fortran", "view"],
)
def test_unusual_array_layouts_across_modules(transform):
    rng = np.random.default_rng(1)
    before = rng.normal(0.2, 0.01, (3, 40, 50))
    after = 1.2 * before + 0.01 * rng.normal(size=before.shape)
    after[:, 5:15, 5:15] += 0.5
    meta = {"transform": from_origin(500000, 4000000, 10, 10), "crs": UTM}
    ref = farq.detect_changes(before[0], after[0], normalize="pif")
    got = farq.detect_changes(transform(before[0]), transform(after[0]), normalize="pif")
    assert np.array_equal(ref.mask, got.mask)
    assert np.array_equal(
        radiometry.irmad_change(before, after), radiometry.irmad_change(transform(before), after)
    )
    m = ref.mask
    assert vector.polygonize(transform(m), meta) == vector.polygonize(m, meta)
    assert vector.polygonize(transform(m.astype(np.int32)), meta) == vector.polygonize(
        m.astype(np.int32), meta
    )
    assert np.array_equal(masking.buffer_mask(transform(m), 2), masking.buffer_mask(m, 2))
    dod = elevation.elevation_change(transform(before[0]), transform(after[0]))
    assert np.allclose(dod, after[0] - before[0])
    assert elevation.volume_change(dod, meta) == elevation.volume_change(after[0] - before[0], meta)
    qa = transform(np.full((4, 4), 22280, dtype=np.uint16))
    assert masking.landsat_qa_mask(qa).all()


def test_one_pixel_and_constant_inputs():
    meta = {"transform": from_origin(500000, 4000000, 10, 10), "crs": UTM}
    one = np.ones((1, 1))
    (f,) = vector.polygonize(one.astype(bool), meta)
    assert f["properties"]["area_m2"] == 100.0
    assert masking.buffer_mask(one.astype(bool), 3).shape == (1, 1)
    assert elevation.volume_change(one, meta).fill_m3 == 100.0
    assert radiometry.histogram_match(one, one * 2)[0, 0] == 2.0
    const = np.full((20, 20), 0.3)
    res = farq.detect_changes(const, const)  # no variation: nothing changes
    assert not res.mask.any()
    with pytest.raises(ValueError, match="constant"):
        radiometry.linear_normalize(const, const)


def test_areas_are_square_metres_for_feet_crs(tmp_path):
    """change_summary / polygonize / detect_changes_file reported ``*_m2`` in ft² for
    CRSs in feet, while farq.elevation converted to metres: one DSM gave two areas."""
    ft = 0.30480060960121924  # US survey foot
    meta = {"crs": CRS.from_epsg(2263), "transform": from_origin(1e6, 2e5, 1.0, 1.0)}
    mask = np.zeros((10, 10), bool)
    mask[2:6, 3:8] = True  # 20 pixels of 1 ft x 1 ft
    expected = 20 * ft * ft
    assert farq.change_summary(mask, pixel_size=meta)["changed_area_m2"] == pytest.approx(expected)
    (f,) = vector.polygonize(mask, meta)
    assert f["properties"]["area_m2"] == pytest.approx(expected)
    assert f["properties"]["perimeter_m"] == pytest.approx(18 * ft)
    assert f["properties"]["centroid_x"] == pytest.approx(1e6 + 5.5)  # CRS units
    assert vector.polygonize(mask, meta, min_area=expected * 1.01) == []
    dod = np.where(mask, 1.0, 0.0)
    v = elevation.volume_change(dod, meta)
    assert v.fill_area_m2 == pytest.approx(expected) and v.fill_m3 == pytest.approx(expected)
    b = _write(tmp_path / "b.tif", np.zeros((10, 10), np.float32), **meta)
    a = _write(tmp_path / "a.tif", dod.astype(np.float32), **meta)
    s = farq.detect_changes_file(b, a, tmp_path / "c.tif", threshold=0.5)
    assert s["changed_area_m2"] == pytest.approx(expected)
    # Metre CRSs and plain pixel sizes are unchanged.
    utm = {"crs": UTM, "transform": from_origin(5e5, 4e6, 10.0, 10.0)}
    assert farq.change_summary(mask, pixel_size=utm)["changed_area_m2"] == 2000.0
    assert farq.change_summary(mask, pixel_size=10.0)["changed_area_m2"] == 2000.0


def test_irmad_alpha_checked_before_normalization():
    x = np.zeros((3, 4, 4))  # would fail inside the normalization
    with pytest.raises(ValueError, match="alpha"):
        farq.detect_changes(x, x, method="irmad", alpha=1.5, normalize="pif")


def test_geojson_nested_non_finite_properties_become_null(tmp_path):
    meta = {"transform": from_origin(500000, 4000000, 10, 10), "crs": UTM}
    (f,) = vector.polygonize(np.ones((2, 2), bool), meta)
    f["properties"].update(
        stats=[1.0, float("nan"), np.float32("inf")],
        info={"a": np.nan, 1: np.int64(2)},
        arr=np.array([0.5, np.nan]),
    )
    fc = vector.to_geojson([f], tmp_path / "x.geojson", crs=UTM)
    props = fc["features"][0]["properties"]
    assert props["stats"] == [1.0, None, None]
    assert props["info"] == {"a": None, "1": 2}
    assert props["arr"] == [0.5, None]
    text = (tmp_path / "x.geojson").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


@pytest.mark.parametrize("threshold", ["otsu", "std", 0.5])
def test_irmad_rejects_any_explicit_threshold(threshold):
    """An explicit threshold="otsu" used to be silently ignored with method="irmad"."""
    rng = np.random.default_rng(0)
    before = rng.normal(size=(3, 20, 20))
    after = 1.1 * before + 0.05 * rng.normal(size=before.shape)
    with pytest.raises(ValueError, match="alpha"):
        farq.detect_changes(before, after, method="irmad", threshold=threshold)
    farq.detect_changes(before, after, method="irmad")  # the default is fine
    import inspect

    default = inspect.signature(farq.detect_changes).parameters["threshold"].default
    assert default == "otsu" and repr(default) == "'otsu'"
    # Explicit "otsu" keeps working for the other methods.
    a = farq.detect_changes(before[0], after[0], threshold="otsu")
    b = farq.detect_changes(before[0], after[0])
    assert a.threshold == b.threshold
