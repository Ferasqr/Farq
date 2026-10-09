# Cloud and quality masking

A cloud in one date and not in the other is the largest source of false change in
satellite change detection. Cloud shadows, snow, saturated pixels and fill pixels cause
the same problem. Most products ship a quality band that flags these pixels.
`farq.masking` decodes those bands into boolean masks and sets the flagged pixels to NaN
in your images. NaN is farq's nodata value, so masked pixels are never counted as change.

Every function is available as `farq.<name>` (for example `farq.landsat_qa_mask`) and
from `farq.masking`. No extra dependency is needed: `pip install farq` is enough, and the
GDAL bundled in rasterio's wheels reads the GeoTIFF and JPEG 2000 bands of these products.

- [Conventions](#conventions)
- [Supported quality bands](#supported-quality-bands)
- [Landsat Collection 2 pipeline](#landsat-collection-2-pipeline)
- [Sentinel-2 L2A pipeline](#sentinel-2-l2a-pipeline)
- [Buffering](#buffering)
- [Reference](#reference)

## Conventions

| Topic | Convention |
| --- | --- |
| Mask values | `True` means **masked** (unusable pixel). |
| Applying masks | `apply_mask(data, mask)` returns a **float copy** with masked pixels set to NaN. It accepts `(rows, cols)` or `(bands, rows, cols)` data, and a `(rows, cols)` mask (applied to every band) or a mask with the shape of `data`. |
| Quality band dtype | Integer bands (as read by `farq.read(path)`). Float bands are accepted if every value is a whole number. NaN entries, and masked entries of a masked array, count as nodata and are always masked. |
| Change pairs | Mask the **union** of both dates (`combine_masks`). Change can only be measured where both dates are clear. |
| Inputs | Never modified. No `RuntimeWarning` is emitted. |

## Supported quality bands

| Product | Band | Function | Masked by default |
| --- | --- | --- | --- |
| Landsat 4-9 Collection 2 (L1 and L2) | `QA_PIXEL` | `landsat_qa_mask` | fill, dilated cloud, cirrus, cloud, cloud shadow |
| Landsat 4-9 Collection 2 | `QA_RADSAT` | `landsat_radsat_mask` | saturation in any band, terrain occlusion (8/9), dropped pixels (4-7) |
| Sentinel-2 L2A | `SCL` | `sentinel2_scl_mask` | no data, saturated/defective, cloud shadow, cloud medium and high probability, thin cirrus (classes 0, 1, 3, 8, 9, 10) |
| Sentinel-2 L1C/L2A | `MSK_CLDPRB` | `sentinel2_cloud_probability_mask` | cloud probability ≥ 50 % |
| HLS v2.0 (HLSL30, HLSS30) | `Fmask` | `hls_fmask_mask` | fill (255), cloud, adjacent to cloud/shadow, cloud shadow, high aerosol |

Snow and water are not masked by default, because they are often what you want to
measure. Pass `snow=True` (Landsat, HLS) or add `"snow_ice"` to the SCL `classes` when you
compare a winter scene with a summer scene and snow would show up as change.

Two helpers convert digital numbers to physical units, with fill pixels as NaN:

- `landsat_c2_scale(dn, "sr")`: surface reflectance, `DN * 0.0000275 - 0.2`.
  `landsat_c2_scale(dn, "st")`: surface temperature in kelvin, `DN * 0.00341802 + 149.0`.
- `sentinel2_l2a_scale(dn, offset=-1000, quantification=10000)`: reflectance
  `(DN + offset) / 10000`. Products from processing baseline 04.00 onwards (from
  25 January 2022) carry `BOA_ADD_OFFSET = -1000`. For older products pass `offset=0`.
  If you compare a 2021 scene with a 2023 scene without this correction, every pixel
  appears to change by 0.1 reflectance.

A quick look at what a Landsat 8/9 `QA_PIXEL` value means:

```python
import numpy as np
import farq

qa = np.array([21824, 21952, 22280, 23888, 54596, 1], dtype=np.uint16)
# clear land, clear water, high-confidence cloud, cloud shadow, cirrus, fill
print(farq.landsat_qa_mask(qa))   # [False False  True  True  True  True]
print(farq.decode_bits(qa, 8, 2))  # cloud confidence: [1 1 3 1 1 0]
print(farq.decode_landsat_qa(qa)["water"])
```

## Landsat Collection 2 pipeline

Water change between two Landsat 8/9 Collection 2 Level-2 scenes of the same path/row.
NDWI uses green (`SR_B3`) and NIR (`SR_B5`).

```python
import numpy as np
import farq


def load(scene):
    """Green and NIR reflectance, the scene's mask and its metadata."""
    green, meta = farq.read(f"{scene}_SR_B3.TIF")
    nir, _ = farq.read(f"{scene}_SR_B5.TIF")
    qa, _ = farq.read(f"{scene}_QA_PIXEL.TIF")
    radsat, _ = farq.read(f"{scene}_QA_RADSAT.TIF")
    clouds = farq.landsat_qa_mask(qa)  # fill, cloud, shadow, cirrus
    clouds = farq.buffer_mask(clouds, distance=90, pixel_size=meta)  # + 3 px of 30 m
    mask = farq.combine_masks(clouds, farq.landsat_radsat_mask(radsat, bands=[3, 5]))
    reflectance = farq.landsat_c2_scale(np.stack([green, nir]))  # fill (0) -> NaN
    return reflectance, mask, meta


before, mask_before, meta = load("LC08_L2SP_044034_20200712_20200912_02_T1")
after, mask_after, meta_after = load("LC09_L2SP_044034_20240715_20240716_02_T1")

# Scenes of one path/row are usually on the same 30 m grid but with slightly different
# extents. align_pair crops both to the overlap (NaN is nodata in float data).
if before.shape != after.shape or meta["transform"] != meta_after["transform"]:
    before, after, meta = farq.align_pair(
        farq.apply_mask(before, mask_before), meta,
        farq.apply_mask(after, mask_after), meta_after,
    )
    mask = ~(np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0))
else:
    overlap = farq.valid_overlap(mask_before, mask_after)
    print(f"Usable in both dates: {overlap.fraction:.0%} "
          f"(before {overlap.before_clear:.0%} clear, after {overlap.after_clear:.0%})")
    mask = farq.combine_masks(mask_before, mask_after)
before, after = farq.apply_mask(before, mask), farq.apply_mask(after, mask)

ndwi_before = farq.ndwi(before[0], before[1])
ndwi_after = farq.ndwi(after[0], after[1])
result = farq.detect_changes(ndwi_before, ndwi_after, threshold="otsu", min_size=5)

valid = np.isfinite(result.magnitude)  # clear in both dates
summary = farq.change_summary(result.mask, pixel_size=meta, valid=valid)
print(f"Changed: {summary['changed_area_km2']:.2f} km² of "
      f"{summary['valid_pixels']} clear pixels")
```

Notes:

- `landsat_qa_mask` works for Landsat 4-7 TM/ETM+ as well. They have no cirrus band, so
  their cirrus bits are always 0. For `QA_RADSAT`, pass `sensor="tm"` (Landsat 4/5) or
  `sensor="etm"` (Landsat 7), because the saturation bits differ by sensor.
- The single-bit flags (cloud, shadow, snow, cirrus) mark *high-confidence* detections.
  For a stricter mask, add `min_confidence="medium"`, which also masks pixels whose cloud
  confidence is medium or high. Do not use `"low"`: CFMask gives low confidence to almost
  every clear pixel, so it masks nearly the whole scene.
- CFMask already dilates clouds by a few pixels (bit 1, masked by default). The extra
  `buffer_mask` catches thin cloud edges and the edges of shadows.

## Sentinel-2 L2A pipeline

Vegetation change between two Sentinel-2 L2A acquisitions of one tile. The red (`B04`)
and NIR (`B08`) bands are 10 m, and the scene classification (`SCL`) is 20 m.
`target_shape` repeats each 20 m pixel over its 2 × 2 block of 10 m pixels. This is exact
because both resolutions cover the same tile extent. Tiles of the same MGRS ID share one
grid, so no alignment is needed.

```python
import numpy as np
import farq


def load(granule, offset):
    red, meta = farq.read(f"{granule}_B04_10m.jp2")
    nir, _ = farq.read(f"{granule}_B08_10m.jp2")
    scl, _ = farq.read(f"{granule}_SCL_20m.jp2")
    bad = {"no_data", "saturated_defective", "cloud_shadow", "cloud_medium", "cloud_high",
           "thin_cirrus", "snow_ice"}
    mask = farq.sentinel2_scl_mask(scl, classes=bad, target_shape=red.shape)
    mask = farq.buffer_mask(mask, distance=50, pixel_size=meta)  # 5 pixels of 10 m
    # BOA_ADD_OFFSET from MTD_MSIL2A.xml: -1000 from baseline 04.00, 0 before.
    reflectance = farq.sentinel2_l2a_scale(np.stack([red, nir]), offset=offset)
    return reflectance, mask, meta


before, mask_before, meta = load("T33UUP_20210705T101559", offset=0)      # baseline 03.xx
after, mask_after, _ = load("T33UUP_20240709T101559", offset=-1000)       # baseline 05.10

overlap = farq.valid_overlap(mask_before, mask_after)
if overlap.fraction < 0.3:
    raise SystemExit(f"Only {overlap.fraction:.0%} of the tile is clear in both dates")

mask = farq.combine_masks(mask_before, mask_after)
before, after = farq.apply_mask(before, mask), farq.apply_mask(after, mask)

ndvi_before = farq.ndvi(before[1], before[0])
ndvi_after = farq.ndvi(after[1], after[0])
result = farq.detect_changes(ndvi_before, ndvi_after, threshold="otsu", min_size=10)
summary = farq.change_summary(result.mask, pixel_size=meta, valid=overlap.valid)
print(f"{summary['changed_area_km2']:.2f} km² changed "
      f"({summary['changed_percent']:.1f}% of the clear overlap)")
```

The `MSK_CLDPRB` cloud probability band (20 m, percent) can be used instead of, or as
well as, the SCL. It flags clouds only, not their shadows:

```python
import farq

red, meta = farq.read("T33UUP_20240709T101559_B04_10m.jp2")
prob, _ = farq.read("T33UUP_20240709T101559_MSK_CLDPRB_20m.jp2")
clouds = farq.sentinel2_cloud_probability_mask(prob, threshold=40, target_shape=red.shape)
```

For HLS, `hls_fmask_mask(fmask)` plays the same role. HLS surface reflectance is scaled
by 0.0001 and has a fill value of -9999, so read it with `farq.read(path, masked=True)`
and divide by 10000.

## Buffering

```text
buffer_mask(mask, pixels=None, *, distance=None, pixel_size=None) -> ndarray
```

`buffer_mask` grows a mask by a radius, either in pixels or as a ground `distance` with
`pixel_size` (a number, an `(x, y)` pair, an `affine.Affine`, or rasterio metadata).
Every pixel whose centre is within the radius of a masked pixel centre becomes masked.
This is the same as a dilation with a disk-shaped structuring element. A radius of 1
adds the 4 direct neighbours, and a radius of 1.5 adds the full 3 × 3 neighbourhood.
Non-square pixels give a circle on the ground. A `distance` with metadata in a
geographic CRS (degrees) raises `ValueError`.

The cost does not depend on the radius, because `buffer_mask` uses an exact Euclidean
distance transform. It processes the mask in row blocks, so memory use stays bounded on
full scenes.

## Reference

All names below are available as `farq.<name>` and in `farq.masking`. Parameters after
`*` are keyword-only, and every mask function returns a boolean array with
`True` = masked.

| Function | Returns |
| --- | --- |
| `decode_bits(qa, bit, width=1)` | Bit field `(qa >> bit) & (2**width - 1)`, as the smallest unsigned dtype that holds it. |
| `decode_landsat_qa(qa_pixel)` | dict of every `QA_PIXEL` flag (bool) and confidence field (0-3). |
| `landsat_qa_mask(qa_pixel, *, cloud=True, shadow=True, cirrus=True, snow=False, dilated=True, fill=True, water=False, min_confidence=None)` | Mask. `min_confidence` is `"low"`, `"medium"`, `"high"` or 1-3. |
| `landsat_radsat_mask(qa_radsat, *, sensor="oli", bands=None, terrain_occlusion=True, dropped_pixel=True)` | Mask. `sensor` is `"oli"`, `"tm"` or `"etm"`. |
| `landsat_c2_scale(dn, kind="sr", *, fill=0)` | float32 reflectance (`"sr"`) or kelvin (`"st"`). |
| `sentinel2_scl_mask(scl, *, classes=DEFAULT_S2_BAD_CLASSES, target_shape=None)` | Mask. `classes` takes codes, `SCLClass` members or names from `SCL_NAMES`. |
| `sentinel2_cloud_probability_mask(probability, threshold=50, *, target_shape=None)` | Mask where `probability >= threshold`. |
| `sentinel2_l2a_scale(dn, offset=-1000, quantification=10000, *, nodata=0)` | float32 reflectance. |
| `hls_fmask_mask(fmask, *, cloud=True, adjacent=True, shadow=True, snow=False, water=False, cirrus=False, aerosol="high", fill=True)` | Mask. |
| `upsample_mask(mask, target_shape)` | Mask repeated by an exact integer factor. |
| `buffer_mask(mask, pixels=None, *, distance=None, pixel_size=None)` | Grown mask. |
| `combine_masks(*masks)` | Union of the masks. `None` entries are skipped. |
| `apply_mask(data, mask, *, fill=nan)` | Float copy of `data` with masked pixels set to `fill`. |
| `clear_fraction(mask, valid=None)` | Fraction of (valid) pixels that are not masked. NaN if there are none. |
| `valid_overlap(mask_before, mask_after, *, footprint=None)` | `MaskOverlap(valid, n_valid, n_total, fraction, before_clear, after_clear)`. |

Constants: `LandsatQA` (QA_PIXEL bit positions), `Confidence` (0 none, 1 low, 2 medium,
3 high), `SCLClass` and `SCL_NAMES` (SCL classes 0-11), `DEFAULT_S2_BAD_CLASSES`,
`HLSFmask` (Fmask bit positions).

### Sources

- Landsat `QA_PIXEL`, `QA_RADSAT` and scale factors: USGS, *Landsat 8-9 Collection 2
  Level 2 Science Product Guide* (LSDS-1619), and *Landsat 4-7 Collection 2 Level 2
  Science Product Guide* (LSDS-1618). In `QA_PIXEL`, bit 0 is fill, 1 dilated cloud,
  2 cirrus (8/9 only), 3 cloud, 4 cloud shadow, 5 snow, 6 clear, 7 water, 8-9 cloud
  confidence, 10-11 cloud shadow confidence, 12-13 snow/ice confidence and 14-15 cirrus
  confidence (8/9 only). In the shadow, snow/ice and cirrus confidence fields, the value
  2 is reserved. In `QA_RADSAT` for Landsat 8/9, bits 0-6 are bands 1-7, bit 8 is band 9
  and bit 11 is terrain occlusion. For Landsat 4-7, bits 0-6 are bands 1-7, bit 8 is
  ETM+ band 6H and bit 9 is a dropped pixel.
- Sentinel-2 `SCL`, `MSK_CLDPRB` and `BOA_ADD_OFFSET`: ESA, *Sentinel-2 Products
  Specification Document* and the Sen2Cor / L2A product documentation (processing
  baseline 04.00 introduced the -1000 offset and renamed SCL class 2 "topographic cast
  shadows").
- HLS `Fmask`: NASA LP DAAC, *HLS Product User Guide v2.0*, Table 9. Bit 0 is cirrus
  (reserved, not used), 1 cloud, 2 adjacent to cloud/shadow, 3 cloud shadow, 4 snow/ice,
  5 water, and bits 6-7 the aerosol level (climatology, low, moderate, high).
