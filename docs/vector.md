# Vector export guide

`farq.vector` turns change masks and class maps into polygons that QGIS, ArcGIS and
other GIS software can open. Each polygon is a connected region of pixels that have the
same value. It carries its area, perimeter, pixel count and centroid as attributes.

| Function | What it does |
| --- | --- |
| [`polygonize`](#polygonize) | Mask or class map to a list of GeoJSON-like features |
| [`to_geojson`](#to_geojson) | Features to a GeoJSON `FeatureCollection` (WGS 84 by default), optionally written to a file |
| [`write_vector`](#write_vector) | Writes GeoJSON, GeoPackage, Shapefile or FlatGeobuf |
| [`changes_to_vector`](#one-call-changes_to_vector) | Does all of this in one call for `detect_changes` or `classify_change` output |

Every function is available as `farq.<name>` and from `farq.vector`. GeoJSON output
needs no extra package. GeoPackage, Shapefile and FlatGeobuf need
[pyogrio](https://pyogrio.readthedocs.io) (preferred) or
[fiona](https://fiona.readthedocs.io):

```bash
pip install "farq[vector]"   # pyogrio and shapely
```

The examples on this page run as written and share variables. They use a small
synthetic raster in UTM zone 33N with 10 m pixels.

## One call: `changes_to_vector`

```python
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
import farq

meta = {
    "transform": from_origin(500000, 4000000, 10, 10),  # 10 m pixels
    "crs": CRS.from_epsg(32633),
    "height": 50,
    "width": 50,
}
before = np.zeros((50, 50), np.float32)
after = before.copy()
after[5:15, 5:15] = 0.8   # a 10 x 10 pixel patch of change (1 ha)
after[30:33, 40:42] = 0.5  # a small 3 x 2 patch (600 m²)

result = farq.detect_changes(before, after, threshold=0.25)
features = farq.changes_to_vector(result, meta, "changes.geojson", min_area=1000)
for f in features:
    p = f["properties"]
    print(p["label"], p["pixel_count"], p["area_m2"], p["perimeter_m"], p["mean_magnitude"])
# changed 100 10000.0 400.0 0.800000011920929
```

The format follows the file extension: `.geojson`/`.json`, `.gpkg`, `.shp` or `.fgb`.
`changes_to_vector` accepts three kinds of input:

- a `ChangeResult` from `detect_changes`: polygons of `result.mask`, with the
  `mean_magnitude` and `max_magnitude` of each polygon's pixels;
- a boolean change mask: polygons of the `True` regions, labelled `"changed"`;
- a class map from `classify_change` (also after reading it back with
  `farq.read(..., masked=True)`): by default the `gained` and `lost` regions, labelled
  with `farq.CHANGE_LABELS`, with `farq.CHANGE_NODATA` pixels excluded.

```python
water_before = np.zeros((50, 50), bool)
water_after = np.zeros((50, 50), bool)
water_before[20:40, 20:40] = True
water_after[25:45, 20:40] = True  # the lake moved 5 pixels south

classes = farq.classify_change(water_before, water_after)
features = farq.changes_to_vector(classes, meta, "water_change.geojson")
print(sorted((f["properties"]["label"], f["properties"]["area_m2"]) for f in features))
# [('gained', 10000.0), ('lost', 10000.0)]
```

## `polygonize`

```text
polygonize(data, meta, *, valid=None, connectivity=4, values=None, min_area=None,
           simplify=None, preserve_topology=False, labels=None, nodata=None) -> list[dict]
```

`data` is a 2-D boolean mask or numeric class map. NaN, ±inf, `nodata` and masked
entries are invalid and never become polygons, and pixels where `valid` is False are
skipped. Boolean masks produce polygons for `True` only. Class maps produce one polygon
per connected region of each value. Use `values=` to keep only some values.

```python
features = farq.polygonize(classes, meta, values=[farq.GAINED, farq.LOST],
                           labels=farq.CHANGE_LABELS)
f = features[0]
print(f["geometry"]["type"], f["properties"])
# Polygon {'value': 2, 'label': 'lost', 'pixel_count': 100, 'area_m2': 10000.0,
#          'perimeter_m': 500.0, 'centroid_x': 500300.0, 'centroid_y': 3999775.0}
```

Each feature is a GeoJSON-like dict (`type`, `id`, `geometry`, `properties`) in the
raster CRS. The properties are:

| Property | Meaning |
| --- | --- |
| `value` | Pixel value of the region (`1` for boolean masks) |
| `label` | `labels[value]`, `"changed"` for boolean masks, otherwise `str(value)` |
| `pixel_count` | Number of pixels in the region (holes excluded) |
| `area_m2` | `pixel_count` × pixel area, in squared CRS units. This equals the exact shoelace area of the polygon. |
| `perimeter_m` | Length of the pixel-edge boundary, including the boundaries of holes |
| `centroid_x`, `centroid_y` | Area centroid in the raster CRS. It can lie outside a non-convex polygon. |

Notes:

- **Metadata.** `meta` is a rasterio meta dict, an open dataset or an `affine.Affine`.
  Rasters without a geotransform (an identity transform with no CRS, or GCP-only drone
  images) raise `ValueError`: rectify them first with `farq.rectify` or
  `farq.align_pair`. A **geographic CRS** (degrees) also raises `ValueError`, because
  areas and perimeters in metres are undefined. Reproject the raster to a projected CRS
  such as UTM first. `to_geojson` still writes lon/lat. A bare `Affine` carries no CRS,
  so its units are assumed to be metres.
- **Connectivity.** With `connectivity=4` (the default) the polygons are valid OGC
  geometries. With `connectivity=8`, pixels that touch only at a corner join one
  polygon whose ring touches itself at that corner. ArcGIS accepts this, but it is
  invalid by OGC rules, and shapely or QGIS's "Check validity" report it.
- **`min_area`** (m²) drops small polygons. Remove speckle from the raster first
  (`farq.clean_mask`) to keep the polygon count down.
- **Large masks.** Area, perimeter and centroid are computed in one vectorized pass, so
  the cost is dominated by GDAL's tracing. A noisy 2000 × 2000 mask with 425,000
  polygons takes under 10 seconds. Very noisy masks produce many tiny polygons, so clean
  them first.

### Simplification

Polygons follow pixel edges, which gives staircase outlines. `simplify=` (a tolerance in
CRS units) smooths them with Douglas–Peucker:

```python
blob = np.hypot(*np.mgrid[-25:25, -25:25]) < 18  # a rasterized disc
exact = farq.polygonize(blob, meta)[0]
smooth = farq.polygonize(blob, meta, simplify=10)[0]
print(len(exact["geometry"]["coordinates"][0]), len(smooth["geometry"]["coordinates"][0]))
# 77 12
```

- The attributes describe the pixel region and do not change with simplification.
- Each ring is simplified on its own. Neighbouring polygons, such as adjacent classes
  of a class map, can therefore develop small gaps or overlaps along shared edges, and
  a simplified ring can self-intersect. Holes smaller than the tolerance are dropped. An
  exterior ring that would collapse is kept unsimplified.
- `preserve_topology=True` simplifies all polygons together with
  `shapely.coverage_simplify`, so shared edges stay identical. This requires
  `shapely>=2.1`, which `farq[vector]` installs on Python 3.10+ (on Python 3.9 the
  extra installs shapely 2.0 and this option raises a clear error).

## `to_geojson`

```text
to_geojson(features, path=None, *, crs=None, to_wgs84=True, precision=None) -> dict
```

Returns a `FeatureCollection` dict and, if `path` is given, writes it as UTF-8. The
file is written to a temporary name and then renamed, so a failed write never leaves a
truncated file.

```python
fc = farq.to_geojson(features, "water_change.geojson", crs=meta, precision=7)
print(fc["features"][0]["geometry"]["coordinates"][0][0])
# [15.0022231, 36.1429149]
```

- By default the coordinates are reprojected to WGS 84 longitude/latitude, as
  [RFC 7946](https://www.rfc-editor.org/rfc/rfc7946) requires, so `crs` is required.
  `crs` accepts a `CRS`, an EPSG code, a string or a rasterio meta dict.
- Rings follow the right-hand rule: exteriors are counter-clockwise and holes
  clockwise. A polygon that crosses the antimeridian is cut into a `MultiPolygon` at
  180° when shapely is installed. Without shapely, it is kept whole with longitudes
  above 180°, and a warning is raised.
- `to_wgs84=False` keeps the raster coordinates and records the CRS in the legacy
  `"crs"` member. GDAL, QGIS and ArcGIS read that member, but RFC 7946 does not define
  it.
- `precision` rounds coordinates to that many decimals (7 decimals of a degree is
  about 1 cm).
- Properties are copied as they are. `centroid_x`/`centroid_y` therefore stay in the
  raster CRS. NaN property values become `null`.

## `write_vector`

```text
write_vector(features, path, *, crs, driver=None, layer=None, to_wgs84=None,
             engine=None) -> None
```

```python
farq.write_vector(features, "water_change.gpkg", crs=meta["crs"], layer="water")
farq.write_vector(features, "water_change.shp", crs=meta["crs"])
```

- `driver` is inferred from the extension: `.geojson`/`.json` (GeoJSON), `.gpkg`
  (GPKG), `.shp` (ESRI Shapefile) or `.fgb` (FlatGeobuf).
- `to_wgs84` defaults to `True` for GeoJSON and to `False` for the other formats,
  which store the raster CRS natively.
- An existing file is replaced as a whole. This includes a GeoPackage, so you cannot
  add a layer to an existing file. Writes go through a temporary file. For a
  Shapefile, each component file is renamed in turn and stale components are removed.
- Shapefile field names are limited to 10 characters, so `perimeter_m` is written as
  `perim_m`, `pixel_count` as `pixels`, `mean_magnitude` as `mean_mag` and
  `max_magnitude` as `max_mag`.
- `engine="pyogrio"` or `engine="fiona"` selects the library. By default pyogrio is
  used, with fiona as a fallback. If neither is installed, `ImportError` asks you to
  `pip install "farq[vector]"`.

## From files on disk

```python
import farq

classes, meta = farq.read("water_change_classes.tif", masked=True)  # 255 -> NaN
farq.changes_to_vector(classes, meta, "water_change.gpkg", min_area=900,
                       simplify=5, preserve_topology=True)
```

A **binary change mask** written by `farq.detect_changes_file` stores 1 = change,
0 = no change and 255 = nodata. Vectorize it as a mask, not as a class map, so polygons
are not labelled with `CHANGE_LABELS` ("gained"):

```python
import farq

farq.detect_changes_file("ndwi_2020.tif", "ndwi_2024.tif", "change.tif", min_size=10)
raw, meta = farq.read("change.tif")
features = farq.polygonize(raw == 1, meta, valid=raw != farq.CHANGE_NODATA, min_area=900)
farq.write_vector(features, "change.gpkg", crs=meta["crs"])
```
