# Drone imagery guide

Drone data needs some extra steps compared with satellite scenes:

- **Georeferencing by GCPs.** Orthomosaics and raw frames are often georeferenced only
  by ground control points (GCPs) instead of an affine geotransform.
- **Different grids.** Two flights never share the same extent, pixel size or pixel
  origin.
- **Residual misregistration.** GNSS and GCP errors of a few centimetres are several
  pixels at centimetre resolution. Without correction, they show up as false change along
  every edge.
- **RGB only.** Consumer cameras have no NIR band, so NDVI and NDWI are not available.
- **Huge files.** Full-resolution mosaics can have billions of pixels.

`farq.georef` and the RGB indices in `farq.indices` cover these steps. The recipes on
this page run in order and share variables. The example files are two GCP-referenced
3-band `uint8` orthomosaics (`flight_2023.tif` at 5 cm, `flight_2024.tif` at 10 cm) and
an image without any georeferencing (`raw_frame.tif`).

## 1. Inspect the files

```python
import numpy as np
import farq

print(farq.has_gcps("flight_2023.tif"))           # True: GCPs, no geotransform
gcps_23, gcps_crs = farq.read_gcps("flight_2023.tif")
print(len(gcps_23), gcps_crs)
print(gcps_23[0].row, gcps_23[0].col, gcps_23[0].x, gcps_23[0].y)

# Read RGB (bands 1-3; an alpha band 4 would be skipped). The metadata carries the GCPs.
rgb_23, meta_23 = farq.read("flight_2023.tif", band=[1, 2, 3])
rgb_24, meta_24 = farq.read("flight_2024.tif", band=[1, 2, 3])
print(rgb_23.shape, rgb_23.dtype, "gcps" in meta_23)

# Quick preview of a huge mosaic: decimated read; transform and GCP positions are rescaled.
preview, preview_meta = farq.read("flight_2023.tif", band=[1, 2, 3], out_shape=(100, 100))
print(preview.shape, len(preview_meta["gcps"]))
```

GCP pixel positions follow the GDAL convention. `(row, col) = (0, 0)` is the **top-left
corner** of the image, so the centre of the first pixel is `(0.5, 0.5)`. Keep this in
mind when you enter GCPs measured in other software, because some tools use pixel
centres.

## 2. Check GCP quality

```python
report = farq.gcp_residuals(gcps_23, order=1)
print(f"RMSE {report.rmse:.3f} m = {report.rmse_pixels:.2f} px "
      f"(pixel size {report.pixel_size:.3f} m, redundancy {report.dof})")
for gcp_id, err, loo in zip(report.ids, report.errors, report.loo_errors):
    print(f"  {gcp_id}: residual {err:.3f} m, leave-one-out {loo:.3f} m")

# A mistyped coordinate (2 m off) stands out in the leave-one-out errors.
bad = list(gcps_23)
bad[4] = farq.make_gcps([(bad[4].row, bad[4].col)], [(bad[4].x + 2.0, bad[4].y)],
                        ids=[bad[4].id])[0]
bad_report = farq.gcp_residuals(bad)
worst = bad_report.outliers()[0]
print(f"worst GCP: {bad_report.ids[worst]} "
      f"(leave-one-out error {bad_report.loo_errors[worst]:.2f} m, RMSE {bad_report.rmse:.2f} m)")

# Drop it and refit: the RMSE should fall back to the centimetre level.
refit = farq.gcp_residuals([g for i, g in enumerate(bad) if i != worst])
print(f"RMSE without {bad_report.ids[worst]}: {refit.rmse:.3f} m, "
      f"remaining outliers: {refit.outliers()}")
```

Guidelines:

- A least-squares fit pulls toward a bad GCP, which hides it in `errors`. The
  **leave-one-out error** (`loo_errors`, the error at a GCP of a fit computed without it)
  does not have this problem. `outliers()` goes one step further: it compares each GCP's
  residual with the noise level of the *other* GCPs, corrected for position (corner GCPs
  naturally have larger leave-one-out errors), and runs a statistical test whose
  false-alarm rate on clean GCPs is about `alpha` (default 5%). It returns indices worst
  first. Several bad GCPs can mask each other, so remove the worst one, refit, and
  repeat until `outliers()` is empty. You can also pass an absolute `threshold` in map
  units.
- `order=1` (affine) needs 3 GCPs, 2 needs 6 and 3 needs 10. Use more GCPs than the
  minimum: with `dof == 0` the residuals are zero by construction and a warning is
  raised. Higher orders can extrapolate wildly outside the area covered by GCPs.
- Aim for `rmse_pixels` around 1 or below before you trust pixel-level change.

## 3. Rectify (optional)

`align_pair` and `align` rectify GCP-referenced inputs themselves, in a single resampling
step. Rectify separately only if you need a georeferenced file of each flight.

```python
# From the file's own GCPs; writes a compressed, tiled GeoTIFF too.
ortho_23, ortho_meta = farq.rectify("flight_2023.tif", "flight_2023_rectified.tif",
                                    resolution=0.05, resampling="bilinear")
print(ortho_23.shape, ortho_meta["transform"].a, ortho_meta["crs"])

# A frame without georeferencing: measure a few GCPs (pixel (row, col) -> map (x, y)).
pixels = [(0, 0), (0, 500), (500, 0), (500, 500), (250, 250)]
coords = [(500005.0, 3999995.0), (500030.0, 3999995.0), (500005.0, 3999970.0),
          (500030.0, 3999970.0), (500017.5, 3999982.5)]
gcps = farq.make_gcps(pixels, coords, ids=["NW", "NE", "SW", "SE", "C"])
print(f"RMSE {farq.gcp_residuals(gcps).rmse:.4f} m")

raw, _ = farq.read("raw_frame.tif", band=None)
rect, rect_meta = farq.georeference(raw, gcps, "EPSG:32633", resolution=0.1,
                                    resampling="average")
farq.write("raw_frame_georeferenced.tif", rect, rect_meta)

# The same from the file, with user-supplied GCPs:
rect2, _ = farq.rectify("raw_frame.tif", gcps=gcps, gcps_crs="EPSG:32633", resolution=0.1)
```

Use `method="tps"` (thin plate spline) for locally distorted images with many accurate
GCPs. Note that TPS also reproduces every GCP error exactly.

## 4. Put both flights on one grid

```python
before, after, meta = farq.align_pair(
    rgb_23, meta_23, rgb_24, meta_24,
    target="coarsest",        # resample to the 10 cm flight; or "before"/"after"/"finest"
    resampling="average",     # good when downsampling 5 cm -> 10 cm
)
print(before.shape, after.shape, before.dtype, farq.pixel_size(meta))
```

- The output is cropped to the overlap of both flights.
- `uint8` input without a nodata value is promoted to `float32`, so pixels outside a
  footprint can be NaN. To keep `uint8`, pass `nodata=0` (if 0 marks the black border of
  your mosaics).
- `resolution=0.2` forces a common pixel size, and `dst_crs="EPSG:32633"` forces a
  common CRS.
- To align a raster to an existing reference grid, use
  `farq.align(array, meta, reference_meta)`.

## 5. Co-register

```python
shift = farq.coregister(before, after)        # (dy, dx) in pixels, sub-pixel precision
xres, yres = farq.pixel_size(meta)
print(f"residual shift: {shift} px = ({shift[0] * yres:.3f}, {shift[1] * xres:.3f}) m")

# For very large mosaics, estimate on a representative crop: the shift in pixels is the
# same, and the FFT needs a few times the memory of the input.
crop = (slice(None), slice(40, 200), slice(40, 200))
print("shift from a crop:", farq.coregister(before[crop], after[crop]))

after = farq.apply_shift(after, shift)        # float32, NaN where shifted in from outside
print("after correction:", farq.coregister(before, after))  # ~ (0, 0)
```

`coregister` uses FFT phase correlation with sub-pixel refinement. It estimates a single
**translation** for the whole image and does not model rotation, scale or local
distortion. It works best on images with texture (fields, roads, buildings) and needs
both images on the same grid (run `align_pair` first). If the GCPs of the two flights
disagree by more than a translation, improve the GCPs or use a higher polynomial order in
`align_pair(..., order=2)`.

## 6. RGB vegetation indices

```python
red_b, green_b, blue_b = before
red_a, green_a, blue_a = after

exg_b, exg_a = farq.exg(red_b, green_b, blue_b), farq.exg(red_a, green_a, blue_a)
vari_b = farq.vari(red_b, green_b, blue_b)          # clipped to [-1, 1]
gli_b = farq.gli(red_b, green_b, blue_b)
tgi_b = farq.tgi(red_b, green_b, blue_b, reflectance_scale=255)
print(f"vegetation (ExG > 0.1): {np.nanmean(exg_b > 0.1):.1%}")
```

| Index | Notes |
| --- | --- |
| `exg` (2g − r − b) | Robust default for green vegetation. Values roughly > 0.1 are vegetation. |
| `exgr` (ExG − ExR) | Vegetation > 0, a built-in threshold. |
| `vari` | Corrects somewhat for atmosphere and illumination. The denominator can approach 0, so values are clipped by default. |
| `gli`, `ngrdi` | Normalized in [-1, 1]. Vegetation > 0. |
| `tgi` | Chlorophyll-sensitive. Not normalized, so pass `reflectance_scale=255` for 8-bit data. |

All of them propagate NaN and return NaN for black pixels (`R + G + B == 0`).

## 7. Detect, clean and summarize

```python
result = farq.detect_changes(exg_b, exg_a, method="difference", threshold="otsu")
mask = farq.clean_mask(result.mask, min_size=50, fill_holes=200)

summary = farq.change_summary(mask, pixel_size=meta, valid=np.isfinite(result.magnitude))
print(f"changed: {summary['changed_area_m2']:.1f} m² ({summary['changed_percent']:.2f}%)")

# Direction: vegetation lost (ExG decreased) vs gained.
lost = mask & (exg_a < exg_b)
print(f"vegetation lost: {lost.sum() * farq.pixel_area(meta):.1f} m²")

# Binary vegetation maps -> gained / lost / stable vegetation.
valid = np.isfinite(exg_b) & np.isfinite(exg_a)
veg = farq.classify_change(exg_b > 0.1, exg_a > 0.1, valid=valid)
veg_summary = farq.change_summary(veg, pixel_size=meta, labels=farq.CHANGE_LABELS,
                                  nodata=farq.CHANGE_NODATA)
print({k: round(v["area_m2"], 2) for k, v in veg_summary["classes"].items()})
```

Drone areas are small, so use `area_m2`. Summaries need metadata with a real
geotransform, like the output of `align_pair` or `rectify`, or an explicit pixel size.
Pass `valid=` so that NaN pixels outside the overlap count as nodata, not as unchanged.

## 8. Plot and save

```python
fig = farq.compare_rgb(before, after, title1="2023", title2="2024")
fig.savefig("flights.png", dpi=150)

fig = farq.changes(exg_a - exg_b, title="ExG change", colorbar_label="ΔExG")
fig.axes[0].contour(mask, levels=[0.5], colors="k", linewidths=0.5)
fig.savefig("exg_change.png", dpi=150)

farq.write("exg_change_mask.tif", mask, meta, dtype="uint8", nodata=None)
```

`compare_rgb` and `plot_rgb` accept a `(3, rows, cols)` stack as returned by
`farq.read(path, band=[1, 2, 3])`, or three separate 2-D bands.

## Working with very large orthomosaics

- **Preview** with `farq.read(path, out_shape=(h, w))`, which uses GDAL overviews and
  decimated reads.
- **Rectify coarser**: `farq.rectify(path, resolution=0.2, resampling="average")` streams
  the source from disk, so only the output grid is held in memory.
- **Align coarser**: pass `resolution=` to `align_pair`. Change detection at 2-3× the
  native pixel size is usually more robust anyway.
- **Co-register on a crop** (see above), then apply the shift to the full array.
- **Process out of core.** Once the flights are on one grid on disk, `farq.index_file`,
  `farq.detect_changes_file` and `farq.map_blocks` work block by block, so memory does
  not depend on the mosaic size. See the [tiling guide](tiling.md).

## Related guides

- [Radiometric normalization](radiometry.md): every flight has its own exposure and
  white balance. `farq.detect_changes(..., normalize="pif")` or `farq.pif_normalize`
  removes these differences before change is measured.
- [Elevation change and volumes](elevation.md): DSM differencing, cut/fill and stockpile
  volumes from the same photogrammetry surveys.
- [Vector export](vector.md): write the change regions as polygons for GIS.
