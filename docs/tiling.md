# Processing rasters larger than memory

`farq.tiling` processes rasters block by block, straight from disk to disk. Memory use
depends on the block size, not on the raster size, so a 50,000 x 50,000 drone orthomosaic
(2.5 billion pixels, 10 GB per float32 band) can be processed on a laptop.

| Function | In-memory equivalent | Result |
| --- | --- | --- |
| `map_blocks(func, inputs, out_path, ...)` | `func(*arrays)` | GeoTIFF |
| `index_file(name, band_paths, out_path, ...)` | `farq.ndvi(...)`, `farq.vari(...)`, ... | GeoTIFF |
| `detect_changes_file(before, after, out_path, ...)` | `farq.detect_changes(...)` | GeoTIFF mask + summary dict |
| `summarize_file(path, ...)` | `farq.stats(...)` | dict |
| `iter_windows(width, height, ...)` | | block windows |

All of them are available as `farq.<name>` (for example `farq.detect_changes_file`) and
from `farq.tiling`. They need no extra dependency beyond `pip install farq`.

## Conventions

- **Inputs must be on one grid.** Every file passed to a call must have the same CRS,
  transform and size. If they do not, a `ValueError` says what differs. Put the rasters
  on one grid first, with `farq.align_pair` / `farq.align` and `farq.write` when they fit
  in memory, or with `gdalwarp` when they do not.
- **NaN is nodata.** Files are read like `farq.read(path, masked=True)`: nodata values,
  alpha bands and internal masks become NaN, and integer data becomes `float32`
  (`float64` for 32/64-bit integers). Pass `masked=False` to `map_blocks` for raw values.
- **Layout.** A block of one band is a `(rows, cols)` array; several bands are
  `(bands, rows, cols)`.
- **Outputs** are tiled (256 x 256), deflate-compressed GeoTIFFs on the input grid,
  BigTIFF when needed, with NaN as nodata for float data. Keyword arguments such as
  `compress="zstd"` or `predictor=2` override the creation options.
- **Atomic writes.** The output is written to a temporary file next to the destination
  and renamed only when complete. If anything fails (including your function raising
  an exception), no partial file is left and an existing file is not touched.
- **Parallelism.** `n_jobs=4` processes four blocks at a time in threads (GDAL reads and
  most NumPy operations release the GIL) and compresses output tiles with four threads.
  The output is identical for any `n_jobs`. Your function must be thread-safe.
- **Progress.** `progress=callback` is called as `callback(done, total)` after each
  block. Multi-pass functions count the blocks of all passes.
- **Block size.** `block_size=1024` (or a `(rows, cols)` pair) is rounded down to a
  multiple of the input's internal tile size, so each tile is decoded once. Up to
  `2 * n_jobs` blocks are in memory at a time.

## Example data

The examples below run as-is. This snippet writes two co-registered 3000 x 2500 float32
rasters (UTM, 0.1 m pixels) with a nodata corner and a few changed patches:

```python
import numpy as np
import rasterio
from rasterio.transform import from_origin

rng = np.random.default_rng(0)
profile = dict(driver="GTiff", height=3000, width=2500, count=1, dtype="float32",
               crs="EPSG:32633", transform=from_origin(500000, 4000000, 0.1, 0.1),
               nodata=-9999, tiled=True, blockxsize=256, blockysize=256)
before = (0.2 + 0.02 * rng.standard_normal((3000, 2500))).astype("float32")
after = before + (0.02 * rng.standard_normal((3000, 2500))).astype("float32")
after[500:900, 300:700] += 0.5        # a 400 x 400 pixel change
after[2000:2003, 1000:1003] += 0.5    # a 9-pixel speck
before[:200, :200] = -9999            # nodata
for name, data in (("before.tif", before), ("after.tif", after)):
    with rasterio.open(name, "w", **profile) as dst:
        dst.write(data, 1)
```

## Change detection: `detect_changes_file`

```python
import farq

summary = farq.detect_changes_file(
    "before.tif", "after.tif", "change.tif",
    method="difference", threshold="otsu", min_size=20,
    magnitude_path="magnitude.tif", block_size=1024, n_jobs=4,
)
print(f"threshold {summary['threshold']:.3f}, "
      f"{summary['changed_pixels']} changed pixels = {summary['changed_area_m2']:.0f} m²")
# threshold 0.259, 160000 changed pixels = 1600 m²
```

`change.tif` is a `uint8` mask: 1 = change, 0 = no change, and 255
(`farq.CHANGE_NODATA`, the file's nodata value) where either date is invalid.
`magnitude.tif` is optional and holds the float change magnitude.

The returned summary is JSON-serializable: `method`, `threshold`, `threshold_method`,
`threshold_exact`, `sampled_pixels`, `total_pixels`, `valid_pixels`, `nodata_pixels`,
`changed_pixels`, `changed_percent` (of valid pixels), `pixel_area_m2`,
`changed_area_m2` and `changed_area_km2`. Areas come from the files' geotransform (or
the `pixel_size=` argument). They are `None` for rasters without a geotransform. As in
`farq.change_summary`, a raster in a geographic CRS (degrees) raises `ValueError` before
any processing unless you pass `pixel_size=` in metres.

How it works, and how it compares with `farq.detect_changes` on the full arrays:

1. **Threshold** (only for `"otsu"`, `"std"` and `"percentile"`). The magnitude is
   computed for every block, and a uniform random sample of `sample_size` valid pixels
   (default 1,000,000) is drawn. One global threshold is computed from that sample.
   The sample depends only on `seed` (default 0), not on `n_jobs`, so it is
   reproducible. If the raster has no more valid pixels than `sample_size`, or if
   `sample_size=None`, every valid pixel is used and the threshold is exactly the
   in-memory one (`summary["threshold_exact"]` is then True). With a sample, the Otsu
   threshold typically differs from the full-raster one by much less than one histogram
   bin.
2. **`min_size` and `fill_holes`** are exact, with no artefacts at block borders.
   Regions are labelled per block, and labels that touch across block edges (including
   diagonally through block corners with `connectivity=8`) are linked into global
   regions, so every region's full size is known before anything is removed. Only
   regions that touch a block edge are tracked between passes. Memory therefore grows
   with the total length of block edges, not with the raster area.
3. The mask (and the magnitude) are written.

With a numeric threshold, or when the sample covers every valid pixel, the mask is
identical, pixel for pixel, to `farq.detect_changes(before, after, ...)` on arrays read
with `farq.read(..., masked=True)`. Each pass reads the inputs again: a fixed
threshold without cleanup takes 1 pass, and `"otsu"` with `min_size` and `fill_holes`
takes 4.

Supported methods are `"difference"`, `"ratio"`, `"normalized_difference"` and
`"cva"` (multi-band: pass `bands=[1, 2, 3]` or `bands=None`). `"pca"` and `"irmad"` need
statistics of the whole image and are not available block-wise.

## Indices: `index_file`

`index_file` computes any farq index from band files, or from bands of one multi-band
file given as `(path, band)` pairs. Extra keyword arguments go to the index function:

```python
import numpy as np
import rasterio
import farq

# A small 3-band RGB "orthomosaic" (uint8, 0 = nodata)
rgb = np.random.default_rng(1).integers(1, 256, size=(3, 1200, 1500), dtype="uint8")
with rasterio.open("ortho.tif", "w", driver="GTiff", height=1200, width=1500, count=3,
                   dtype="uint8", nodata=0, crs="EPSG:32633",
                   transform=rasterio.transform.from_origin(500000, 4000000, 0.05, 0.05)) as dst:
    dst.write(rgb)

bands = {"red": ("ortho.tif", 1), "green": ("ortho.tif", 2), "blue": ("ortho.tif", 3)}
meta = farq.index_file("vari", bands, "vari.tif", n_jobs=4, clip=True)
print(meta["dtype"], meta["width"], meta["height"])
# float32 1500 1200
```

The result equals `farq.vari(red, green, blue, clip=True)` on the full bands.

## Your own function: `map_blocks`

`map_blocks(func, inputs, out_path)` reads the same window from each input, calls
`func(*arrays)` and writes the result. `func` must return an array with the spatial
shape of its inputs, either `(rows, cols)` or `(bands, rows, cols)`. For neighbourhood
operations, add a halo with `overlap`: each block is read with `overlap` extra pixels on
every side, and only its interior is written. With a halo at least as large as the
operation's radius, the result is the same as on the full raster.

```python
import numpy as np
from scipy import ndimage
import farq

# 1. Pixel-wise: signed difference of two co-registered files (after - before)
farq.map_blocks(lambda before, after: after - before, ["before.tif", "after.tif"], "diff.tif")

# 2. Neighbourhood: 5 x 5 median filter (radius 2 -> overlap=2)
farq.map_blocks(lambda a: ndimage.median_filter(a, size=5), ["after.tif"], "after_median.tif",
                overlap=2, block_size=2048, n_jobs=4)

# 3. Several bands in, several bands out, integer output with a nodata value
def brightness_and_mask(rgb):                 # rgb: (3, rows, cols) float32, NaN = nodata
    brightness = rgb.mean(axis=0)
    return np.stack([brightness, brightness > 128])

farq.map_blocks(brightness_and_mask, [("ortho.tif", [1, 2, 3])], "bright.tif",
                dtype="uint8", nodata=255, progress=lambda done, total: None)
```

`map_blocks` returns the output metadata (`crs`, `transform`, `dtype`, `nodata`, ...) in
the same form as `farq.read`.

## Statistics: `summarize_file`

```python
import farq

s = farq.summarize_file("before.tif", bins=50)
print(s["valid"], s["nan"], round(s["mean"], 3), round(s["std"], 3), s["histogram"]["counts"].sum())
# 7460000 40000 0.2 0.02 7460000

per_band = farq.summarize_file("ortho.tif", band=None, bins=None)   # one dict per band, 1 pass
print([round(b["mean"], 1) for b in per_band])
```

Mean, standard deviation and variance are accumulated with a numerically stable
streaming update (Welford/Chan) in float64. They agree with `farq.stats` on the full
raster to rounding. Count, min, max and the histogram (computed in a second pass, over
`[min, max]`) are exact. Percentiles and the median are not computed, because they would
need the whole raster in memory.

## Low level: `iter_windows`

```python
import rasterio
import farq

with rasterio.open("before.tif") as src:
    for block in farq.iter_windows(src.width, src.height, block_size=1024, overlap=8,
                                   align_to=src.block_shapes[0]):
        data = src.read(1, window=block.read_window)     # block plus halo
        core = data[block.inner]                          # = the block.write_window part
        # ... process, then dst.write(result[block.inner], 1, window=block.write_window)
```

## Memory and performance tips

- Memory is about `(block rows + 2*overlap) * (block cols + 2*overlap) * 4 bytes`
  per input band for each of the up to `2 * n_jobs` blocks in flight, plus temporaries in
  your function and GDAL's block cache. The cache defaults to 5% of RAM; you can lower
  it with the `GDAL_CACHEMAX` environment variable, e.g. `GDAL_CACHEMAX=256` (MB).
- Inputs should be tiled GeoTIFFs or COGs. A striped (untiled) file must be read in
  full-width strips, so use blocks as wide as the raster, e.g.
  `block_size=(256, width)`, or convert it once with
  `gdal_translate -co TILED=YES in.tif out.tif`.
- Larger blocks (2048-4096) reduce overhead. Use `n_jobs=-1` for all CPUs.
- For quick previews of a huge file, `farq.read(path, out_shape=(2000, 2000))` reads a
  decimated copy into memory.
