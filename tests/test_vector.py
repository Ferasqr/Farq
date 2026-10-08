"""Tests for farq.vector (polygon export of masks and class maps)."""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

import farq
from farq import vector as vec
from farq.change import CHANGE_LABELS, CHANGE_NODATA, GAINED, LOST, classify_change

UTM = CRS.from_epsg(32633)


@pytest.fixture(autouse=True)
def _no_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def make_meta(shape=(10, 10), res=10.0, crs=UTM, origin=(500000.0, 4000000.0)):
    return {
        "transform": from_origin(origin[0], origin[1], res, res),
        "crs": crs,
        "height": shape[0],
        "width": shape[1],
    }


def signed_area(ring) -> float:
    a = np.asarray(ring, dtype=float)
    a = a - a[0]
    return 0.5 * float(np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1]))


def check_rings(geometry) -> None:
    """Closed rings with >= 4 positions; exterior CCW, holes CW (RFC 7946)."""
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    for poly in polys:
        for i, ring in enumerate(poly):
            assert len(ring) >= 4
            assert ring[0] == ring[-1]
            area = signed_area(ring)
            assert area > 0 if i == 0 else area < 0


# --------------------------------------------------------------------------- #
# polygonize: geometry and metrics
# --------------------------------------------------------------------------- #
def test_square_block_area_perimeter_centroid():
    mask = np.zeros((10, 10), bool)
    mask[1:4, 1:4] = True
    features = vec.polygonize(mask, make_meta())
    assert len(features) == 1
    f = features[0]
    assert f["type"] == "Feature" and f["id"] == 0
    assert f["properties"] == {
        "value": 1,
        "label": "changed",
        "pixel_count": 9,
        "area_m2": 900.0,
        "perimeter_m": 120.0,
        "centroid_x": 500025.0,
        "centroid_y": 3999975.0,
    }
    geom = f["geometry"]
    assert geom["type"] == "Polygon"
    check_rings(geom)
    xs = [p[0] for p in geom["coordinates"][0]]
    ys = [p[1] for p in geom["coordinates"][0]]
    assert (min(xs), max(xs), min(ys), max(ys)) == (500010.0, 500040.0, 3999960.0, 3999990.0)
    assert abs(signed_area(geom["coordinates"][0])) == 900.0


def test_ring_with_hole():
    mask = np.zeros((7, 7), bool)
    mask[1:6, 1:6] = True
    mask[3, 3] = False
    (f,) = vec.polygonize(mask, make_meta((7, 7)))
    props = f["properties"]
    assert props["pixel_count"] == 24
    assert props["area_m2"] == 2400.0
    assert props["perimeter_m"] == 4 * 50.0 + 4 * 10.0  # outer boundary + hole
    rings = f["geometry"]["coordinates"]
    assert len(rings) == 2
    check_rings(f["geometry"])
    # Shoelace over exterior minus hole equals the pixel area.
    assert signed_area(rings[0]) + signed_area(rings[1]) == pytest.approx(2400.0)
    # Symmetric shape: centroid is the centre of the block.
    assert props["centroid_x"] == pytest.approx(500035.0)
    assert props["centroid_y"] == pytest.approx(3999965.0)


def test_diagonal_pixels_connectivity():
    mask = np.eye(3, dtype=bool)
    meta = make_meta((3, 3))
    four = vec.polygonize(mask, meta, connectivity=4)
    eight = vec.polygonize(mask, meta, connectivity=8)
    assert len(four) == 3
    assert all(f["properties"]["pixel_count"] == 1 for f in four)
    assert len(eight) == 1
    assert eight[0]["properties"]["pixel_count"] == 3
    assert eight[0]["properties"]["area_m2"] == 300.0
    assert eight[0]["properties"]["perimeter_m"] == 120.0


@pytest.mark.parametrize("connectivity", [4, 8])
def test_random_mask_counts_match(connectivity):
    rng = np.random.default_rng(0)
    mask = rng.random((60, 70)) > 0.55
    features = vec.polygonize(mask, make_meta(mask.shape, res=2.0), connectivity=connectivity)
    assert sum(f["properties"]["pixel_count"] for f in features) == int(mask.sum())
    assert sum(f["properties"]["area_m2"] for f in features) == pytest.approx(mask.sum() * 4.0)
    assert [f["id"] for f in features] == list(range(len(features)))
    for f in features:
        check_rings(f["geometry"])


def test_perimeter_matches_analysis_crack_length():
    rng = np.random.default_rng(1)
    mask = rng.random((40, 40)) > 0.5
    features = vec.polygonize(mask, make_meta(mask.shape, res=1.0))
    # Total crack length: pixel edges between a True pixel and a False pixel or border.
    padded = np.pad(mask, 1)
    edges = sum(int(np.sum(padded & ~np.roll(padded, s, axis=a))) for a in (0, 1) for s in (1, -1))
    assert sum(f["properties"]["perimeter_m"] for f in features) == pytest.approx(edges)


def test_rotated_and_south_up_transforms():
    mask = np.zeros((6, 6), bool)
    mask[1:3, 1:4] = True  # 2 x 3 pixels
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    rotated = Affine(2 * c, 3 * s, 1000, 2 * s, -3 * c, 2000)  # rotation, 2 x 3 m pixels
    (f,) = vec.polygonize(mask, {"transform": rotated, "crs": UTM})
    assert f["properties"]["pixel_count"] == 6
    assert f["properties"]["area_m2"] == pytest.approx(36.0)
    assert f["properties"]["perimeter_m"] == pytest.approx(2 * (3 * 2.0 + 2 * 3.0))
    check_rings(f["geometry"])
    assert signed_area(f["geometry"]["coordinates"][0]) == pytest.approx(36.0)
    south_up = Affine(10, 0, 0, 0, 10, 0)  # rows increase northwards
    (g,) = vec.polygonize(mask, {"transform": south_up, "crs": UTM})
    check_rings(g["geometry"])
    assert g["properties"]["centroid_x"] == 25.0
    assert g["properties"]["centroid_y"] == 20.0


def test_bare_affine_and_dataset_like():
    mask = np.ones((2, 2), bool)
    t = from_origin(0, 100, 5, 5)
    (f,) = vec.polygonize(mask, t)
    assert f["properties"]["area_m2"] == 100.0

    class Dataset:
        transform = t
        crs = UTM
        gcps = ([], None)

    (g,) = vec.polygonize(mask, Dataset())
    assert g["properties"]["area_m2"] == 100.0


# --------------------------------------------------------------------------- #
# polygonize: values, labels, nodata
# --------------------------------------------------------------------------- #
def test_class_map_from_classify_change_with_labels():
    before = np.zeros((6, 6), bool)
    after = np.zeros((6, 6), bool)
    after[0:2, 0:2] = True  # gained
    before[4:6, 4:6] = True  # lost
    before[0:2, 4:6] = after[0:2, 4:6] = True  # stable
    valid = np.ones((6, 6), bool)
    valid[5, 0] = False
    classes = classify_change(before, after, valid=valid)
    features = vec.polygonize(
        classes, make_meta((6, 6)), labels=CHANGE_LABELS, nodata=CHANGE_NODATA
    )
    by_label = {}
    for f in features:
        by_label.setdefault(f["properties"]["label"], []).append(f["properties"])
    assert set(by_label) == {"gained", "lost", "stable", "no_change"}
    assert by_label["gained"][0]["pixel_count"] == 4
    assert by_label["gained"][0]["value"] == GAINED
    assert by_label["lost"][0]["value"] == LOST
    total = sum(f["properties"]["pixel_count"] for f in features)
    assert total == 35  # the nodata pixel is excluded

    only = vec.polygonize(classes, make_meta((6, 6)), values=[GAINED, LOST], labels=CHANGE_LABELS)
    assert sorted(f["properties"]["label"] for f in only) == ["gained", "lost"]
    for f in only:
        assert isinstance(f["properties"]["value"], int)


def test_float_and_int64_values_are_exact():
    data = np.full((4, 4), 0.1, dtype=np.float64)
    data[:2, :2] = 0.30000000000000004
    data[3, 3] = np.nan
    data[0, 3] = np.inf
    features = vec.polygonize(data, make_meta((4, 4)))
    values = sorted(f["properties"]["value"] for f in features)
    assert values == [0.1, 0.30000000000000004]
    assert sum(f["properties"]["pixel_count"] for f in features) == 14
    big = np.full((3, 3), 2**40, dtype=np.int64)
    big[0, 0] = -(2**40)
    out = vec.polygonize(big, make_meta((3, 3)), labels={2**40: "big"})
    assert {f["properties"]["value"] for f in out} == {2**40, -(2**40)}
    assert {f["properties"]["label"] for f in out} == {"big", str(-(2**40))}


def test_valid_masked_array_and_single_band():
    data = np.ones((1, 4, 4), np.uint8)
    valid = np.ones((4, 4), bool)
    valid[0] = False
    (f,) = vec.polygonize(data, make_meta((4, 4)), valid=valid)
    assert f["properties"]["pixel_count"] == 12
    masked = np.ma.masked_array(np.ones((4, 4), np.int16), mask=~valid)
    (g,) = vec.polygonize(masked, make_meta((4, 4)))
    assert g["properties"]["pixel_count"] == 12
    false_regions = vec.polygonize(valid, make_meta((4, 4)), values=False)
    assert false_regions[0]["properties"]["label"] == "unchanged"
    assert false_regions[0]["properties"]["value"] == 0
    assert false_regions[0]["properties"]["pixel_count"] == 4


def test_min_area_and_empty_result():
    mask = np.zeros((10, 10), bool)
    mask[0, 0] = True
    mask[5:8, 5:8] = True
    features = vec.polygonize(mask, make_meta(), min_area=500)
    assert [f["properties"]["pixel_count"] for f in features] == [9]
    assert features[0]["id"] == 0
    assert vec.polygonize(np.zeros((3, 3), bool), make_meta((3, 3))) == []


def test_input_not_modified():
    data = np.zeros((5, 5), np.float32)
    data[1, 1] = np.nan
    copy = data.copy()
    vec.polygonize(data, make_meta((5, 5)))
    np.testing.assert_array_equal(data, copy)


# --------------------------------------------------------------------------- #
# polygonize: validation
# --------------------------------------------------------------------------- #
def test_rejects_unreferenced_metadata():
    mask = np.ones((3, 3), bool)
    with pytest.raises(ValueError, match="no geotransform"):
        vec.polygonize(mask, {"transform": Affine.identity(), "crs": None})
    with pytest.raises(ValueError, match="no geotransform"):
        vec.polygonize(mask, Affine.identity())
    with pytest.raises(ValueError, match="GCPs"):
        vec.polygonize(mask, {"gcps": [object()], "crs": None})
    with pytest.raises(ValueError, match="invertible"):
        vec.polygonize(mask, Affine(0, 0, 0, 0, 0, 0))
    with pytest.raises(TypeError, match="meta"):
        vec.polygonize(mask, "EPSG:32633")


def test_rejects_geographic_crs():
    mask = np.ones((3, 3), bool)
    meta = {"transform": from_origin(10, 50, 0.001, 0.001), "crs": "EPSG:4326"}
    with pytest.raises(ValueError, match="geographic"):
        vec.polygonize(mask, meta)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"connectivity": 6}, ValueError, "connectivity"),
        ({"min_area": -1}, ValueError, "min_area"),
        ({"simplify": float("nan")}, ValueError, "simplify"),
        ({"min_area": "big"}, TypeError, "min_area"),
        ({"values": []}, ValueError, "values"),
        ({"valid": np.ones((2, 2), bool)}, ValueError, "valid shape"),
        ({"labels": ["a"]}, TypeError, "labels"),
    ],
)
def test_parameter_validation(kwargs, error, match):
    with pytest.raises(error, match=match):
        vec.polygonize(np.ones((3, 3), bool), make_meta((3, 3)), **kwargs)


def test_data_validation():
    meta = make_meta((3, 3))
    with pytest.raises(TypeError, match="numpy array"):
        vec.polygonize([[1, 0]], meta)
    with pytest.raises(ValueError, match="2-D"):
        vec.polygonize(np.ones((2, 3, 3)), meta)
    with pytest.raises(TypeError, match="dtype"):
        vec.polygonize(np.full((3, 3), "a"), meta)
    with pytest.raises(ValueError, match="does not match meta"):
        vec.polygonize(np.ones((4, 3), bool), meta)


# --------------------------------------------------------------------------- #
# Simplification
# --------------------------------------------------------------------------- #
def staircase(n=30):
    return np.tril(np.ones((n, n), bool))


def test_douglas_peucker_simplifies_staircase():
    mask = staircase()
    meta = make_meta(mask.shape, res=1.0)
    (exact,) = vec.polygonize(mask, meta)
    (simple,) = vec.polygonize(mask, meta, simplify=1.0)
    n_exact = len(exact["geometry"]["coordinates"][0])
    n_simple = len(simple["geometry"]["coordinates"][0])
    assert n_simple < n_exact / 5
    check_rings(simple["geometry"])
    # Metrics describe the pixel region, not the simplified ring.
    assert simple["properties"] == exact["properties"]
    assert abs(signed_area(simple["geometry"]["coordinates"][0])) == pytest.approx(450, rel=0.1)
    (same,) = vec.polygonize(mask, meta, simplify=0)
    assert same == exact


def test_douglas_peucker_drops_small_holes_keeps_exterior():
    mask = np.ones((5, 5), bool)
    mask[2, 2] = False
    (f,) = vec.polygonize(mask, make_meta((5, 5), res=1.0), simplify=5.0)
    rings = f["geometry"]["coordinates"]
    assert len(rings) == 1  # 1-pixel hole collapsed
    assert len(rings[0]) == 5  # the square exterior survives
    tiny = np.zeros((3, 3), bool)
    tiny[1, 1] = True
    (g,) = vec.polygonize(tiny, make_meta((3, 3), res=1.0), simplify=100.0)
    assert len(g["geometry"]["coordinates"][0]) == 5


def test_coverage_simplify_keeps_shared_edges():
    shapely = pytest.importorskip("shapely")
    if not hasattr(shapely, "coverage_simplify"):
        pytest.skip("needs shapely>=2.1")
    rng = np.random.default_rng(3)
    classes = np.kron(rng.integers(0, 3, (6, 6)), np.ones((5, 5), int)).astype(np.uint8)
    meta = make_meta(classes.shape, res=1.0)
    features = vec.polygonize(classes, meta, simplify=1.5, preserve_topology=True)
    polys = [shapely.geometry.shape(f["geometry"]) for f in features]
    for f in features:
        check_rings(f["geometry"])
    total = sum(p.area for p in polys)
    union = shapely.union_all(polys).area
    assert total == pytest.approx(union)  # no overlaps
    assert union == pytest.approx(classes.size)  # no gaps


def test_coverage_simplify_without_shapely(monkeypatch):
    monkeypatch.setitem(sys.modules, "shapely", None)
    with pytest.raises(ImportError, match=r"farq\[vector\]"):
        vec.polygonize(np.ones((3, 3), bool), make_meta((3, 3)), simplify=1, preserve_topology=True)


# --------------------------------------------------------------------------- #
# GeoJSON
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_regions():
    mask = np.zeros((10, 10), bool)
    mask[1:4, 1:4] = True
    mask[5:9, 5:9] = True
    mask[6, 6] = False
    meta = make_meta()
    return vec.polygonize(mask, meta), meta


def test_to_geojson_wgs84(two_regions, tmp_path):
    features, meta = two_regions
    before = json.dumps(features)
    path = tmp_path / "changes.geojson"
    fc = vec.to_geojson(features, path, crs=meta)
    assert json.dumps(features) == before  # input not modified
    assert fc["type"] == "FeatureCollection"
    assert "crs" not in fc
    assert len(fc["features"]) == 2
    for f in fc["features"]:
        check_rings(f["geometry"])
        coords = np.array(f["geometry"]["coordinates"][0])
        assert np.all((coords[:, 0] > 14.9) & (coords[:, 0] < 15.1))  # UTM 33N: lon ~ 15
        assert np.all((coords[:, 1] > 36) & (coords[:, 1] < 36.2))
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == fc
    assert [p.name for p in tmp_path.iterdir()] == ["changes.geojson"]
    assert fc["features"][0]["properties"] == features[0]["properties"]


def test_to_geojson_native_crs_and_precision(two_regions):
    features, meta = two_regions
    fc = vec.to_geojson(features, crs=meta["crs"], to_wgs84=False)
    assert fc["crs"] == {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32633"}}
    assert fc["features"][0]["geometry"] == features[0]["geometry"]
    rounded = vec.to_geojson(features, crs=meta, precision=4)
    lon = rounded["features"][0]["geometry"]["coordinates"][0][0][0]
    assert lon == round(lon, 4)
    no_crs = vec.to_geojson(features, to_wgs84=False)
    assert "crs" not in no_crs


def test_to_geojson_wgs84_input_is_not_reprojected():
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],  # clockwise
        },
        "properties": {"a": np.float32(1.5), "b": float("nan"), "c": np.bool_(True)},
    }
    fc = vec.to_geojson([feature], crs=4326)
    geom = fc["features"][0]["geometry"]
    check_rings(geom)  # re-oriented counter-clockwise
    assert fc["features"][0]["properties"] == {"a": 1.5, "b": None, "c": 1}
    json.dumps(fc, allow_nan=False)
    point = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}}
    assert vec.to_geojson({"type": "FeatureCollection", "features": [point]}, crs=4326)["features"][
        0
    ]["geometry"] == {"type": "Point", "coordinates": [1.0, 2.0]}


def test_to_geojson_errors(two_regions, tmp_path):
    features, _ = two_regions
    with pytest.raises(ValueError, match="crs is required"):
        vec.to_geojson(features)
    with pytest.raises(ValueError, match="invalid crs"):
        vec.to_geojson(features, crs="not a crs")
    with pytest.raises(ValueError, match="geometry"):
        vec.to_geojson([{"type": "Feature"}], crs=4326)
    with pytest.raises(TypeError, match="list"):
        vec.to_geojson("features", crs=4326)
    bad = [{"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 0]]]}}]
    with pytest.raises(ValueError, match="at least 4"):
        vec.to_geojson(bad, crs=4326)


def test_to_geojson_failed_write_keeps_existing_file(two_regions, tmp_path):
    features, meta = two_regions
    path = tmp_path / "out.geojson"
    path.write_text("old", encoding="utf-8")
    broken = [dict(features[0], properties={"obj": object()})]
    with pytest.raises(TypeError):
        vec.to_geojson(broken, path, crs=meta)
    assert path.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.geojson"]


def antimeridian_features():
    meta = {"transform": from_origin(500000, 7000000, 1000, 1000), "crs": "EPSG:32660"}
    mask = np.zeros((20, 200), bool)
    mask[5:10, 10:190] = True  # ~177.2°E to ~179.2°W at 63°N
    return vec.polygonize(mask, meta), meta


def test_antimeridian_is_cut_with_shapely():
    pytest.importorskip("shapely")
    features, meta = antimeridian_features()
    fc = vec.to_geojson(features, crs=meta)
    geom = fc["features"][0]["geometry"]
    assert geom["type"] == "MultiPolygon"
    assert len(geom["coordinates"]) == 2
    check_rings(geom)
    lons = [p[0] for poly in geom["coordinates"] for p in poly[0]]
    assert min(lons) >= -180 and max(lons) <= 180
    assert 180.0 in lons and -180.0 in lons


def test_antimeridian_without_shapely_unwraps(monkeypatch):
    features, meta = antimeridian_features()
    monkeypatch.setitem(sys.modules, "shapely", None)
    with pytest.warns(UserWarning, match="antimeridian"):
        fc = vec.to_geojson(features, crs=meta)
    geom = fc["features"][0]["geometry"]
    lons = [p[0] for p in geom["coordinates"][0][0]]
    assert max(lons) > 180 and min(lons) > 170  # continuous, not wrapped
    check_rings(geom)


# --------------------------------------------------------------------------- #
# write_vector
# --------------------------------------------------------------------------- #
def test_write_vector_geojson_needs_no_extra(two_regions, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyogrio", None)
    monkeypatch.setitem(sys.modules, "fiona", None)
    features, meta = two_regions
    vec.write_vector(features, tmp_path / "a.json", crs=meta["crs"])
    data = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert len(data["features"]) == 2
    vec.write_vector(features, tmp_path / "b.geojson", crs=meta["crs"], to_wgs84=False)
    assert "crs" in json.loads((tmp_path / "b.geojson").read_text(encoding="utf-8"))
    with pytest.raises(ImportError, match=r"pip install farq\[vector\]"):
        vec.write_vector(features, tmp_path / "c.gpkg", crs=meta["crs"])


def test_write_vector_driver_errors(two_regions, tmp_path):
    features, meta = two_regions
    with pytest.raises(ValueError, match="extension"):
        vec.write_vector(features, tmp_path / "a.kml", crs=meta["crs"])
    with pytest.raises(ValueError, match="unsupported driver"):
        vec.write_vector(features, tmp_path / "a.x", crs=meta["crs"], driver="KML")
    with pytest.raises(ValueError, match="layer"):
        vec.write_vector(features, tmp_path / "a.geojson", crs=meta["crs"], layer="x")
    with pytest.raises(ValueError, match="engine"):
        vec.write_vector(features, tmp_path / "a.gpkg", crs=meta["crs"], engine="gdal")


@pytest.mark.parametrize("engine", ["pyogrio", "fiona"])
@pytest.mark.parametrize(
    ("ext", "fields"),
    [
        ("gpkg", ["value", "label", "pixel_count", "area_m2", "perimeter_m"]),
        ("fgb", ["value", "label", "pixel_count", "area_m2", "perimeter_m"]),
        ("shp", ["value", "label", "pixels", "area_m2", "perim_m"]),
    ],
)
def test_write_vector_round_trip(two_regions, tmp_path, engine, ext, fields):
    pytest.importorskip(engine)
    pyogrio = pytest.importorskip("pyogrio")
    features, meta = two_regions
    path = tmp_path / f"changes.{ext}"
    path.write_bytes(b"stale")  # replaced, not appended to
    vec.write_vector(features, path, crs=meta["crs"], engine=engine)
    info, _, geometry, field_data = pyogrio.raw.read(path)
    assert CRS.from_user_input(info["crs"]) == UTM
    names = list(info["fields"])
    assert names[:5] == fields
    rows = sorted(zip(*(col.tolist() for col in field_data)))
    assert [r[names.index(fields[2])] for r in rows] == [9, 15]
    assert sorted(r[names.index(fields[3])] for r in rows) == [900.0, 1500.0]
    assert {r[names.index("label")] for r in rows} == {"changed"}
    assert len(geometry) == 2
    if ext != "shp":
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_write_vector_gpkg_layer_and_wgs84(two_regions, tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    features, meta = two_regions
    path = tmp_path / "out.gpkg"
    vec.write_vector(features, path, crs=meta, layer="gained", to_wgs84=True)
    assert [layer[0] for layer in pyogrio.list_layers(path)] == ["gained"]
    info = pyogrio.read_info(path, layer="gained")
    assert CRS.from_user_input(info["crs"]) == CRS.from_epsg(4326)
    vec.write_vector([], tmp_path / "empty.gpkg", crs=meta)
    assert pyogrio.read_info(tmp_path / "empty.gpkg")["features"] == 0


# --------------------------------------------------------------------------- #
# changes_to_vector
# --------------------------------------------------------------------------- #
def test_changes_to_vector_change_result(tmp_path):
    before = np.zeros((20, 20), np.float32)
    after = before.copy()
    after[2:6, 2:6] = 1.0
    after[10:12, 10:13] = 2.0
    after[15, 15] = 3.0
    meta = make_meta((20, 20))
    result = farq.detect_changes(before, after, threshold=0.5)
    path = tmp_path / "changes.geojson"
    features = vec.changes_to_vector(result, meta, path, min_area=500)
    assert len(features) == 2
    by_pixels = {f["properties"]["pixel_count"]: f["properties"] for f in features}
    assert by_pixels[16]["mean_magnitude"] == 1.0
    assert by_pixels[6]["max_magnitude"] == 2.0
    assert by_pixels[16]["label"] == "changed"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["features"]) == 2


def test_changes_to_vector_class_map_defaults(tmp_path):
    before = np.zeros((6, 6), bool)
    after = np.zeros((6, 6), bool)
    after[0:2, 0:2] = True
    before[4:6, 4:6] = True
    before[0:2, 4:6] = after[0:2, 4:6] = True
    classes = classify_change(before, after)
    meta = make_meta((6, 6))
    features = vec.changes_to_vector(classes, meta)
    assert sorted(f["properties"]["label"] for f in features) == ["gained", "lost"]
    # Read back with masked=True gives float32 with NaN nodata: same polygons.
    as_float = np.where(classes == CHANGE_NODATA, np.nan, classes).astype(np.float32)
    again = vec.changes_to_vector(as_float, meta)
    assert [f["properties"] for f in again] == [f["properties"] for f in features]
    mask = vec.changes_to_vector(classes == GAINED, meta)
    assert [f["properties"]["label"] for f in mask] == ["changed"]


def test_changes_to_vector_written_gpkg(tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    classes = np.zeros((6, 6), np.uint8)
    classes[0:2, 0:2] = GAINED
    meta = make_meta((6, 6))
    vec.changes_to_vector(classes, meta, tmp_path / "c.gpkg", layer="change")
    info = pyogrio.read_info(tmp_path / "c.gpkg", layer="change")
    assert info["features"] == 1
    assert CRS.from_user_input(info["crs"]) == UTM


def test_large_mask_is_consistent():
    rng = np.random.default_rng(7)
    mask = rng.random((300, 300)) > 0.6
    features = vec.polygonize(mask, make_meta(mask.shape))
    assert sum(f["properties"]["pixel_count"] for f in features) == int(mask.sum())
    fc = vec.to_geojson(features, crs=UTM)
    assert len(fc["features"]) == len(features)


def test_public_names():
    assert set(vec.__all__) == {"changes_to_vector", "polygonize", "to_geojson", "write_vector"}
    assert os.path.basename(vec.__file__) == "vector.py"
