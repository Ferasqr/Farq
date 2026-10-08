# Elevation change and volumes

`farq.elevation` measures elevation change and cut/fill volumes from two DEMs or DSMs
(drone photogrammetry, LiDAR or stereo satellite), and stockpile volumes from a single
survey. It is meant for earthworks, mining, stockpile inventories and erosion studies.

The standard method is the **DEM of difference (DoD)**: `after - before` on a common grid.
Positive values are fill (material added, deposition). Negative values are cut (material
removed, erosion). A DoD is only as good as the agreement between the two surveys, so the
workflow below has four steps: align the grids, co-register the surfaces on stable
ground, ignore changes below the level of detection, and report volumes with an
uncertainty.

| Function | Purpose |
| --- | --- |
| `slope`, `aspect`, `hillshade` | Terrain derivatives, using the pixel spacing of the geotransform |
| `vertical_offset` | Vertical bias over stable ground: `(offset, nmad)` |
| `coregister_dem`, `shift_dem` | Nuth & Kääb (2011) horizontal and vertical co-registration |
| `elevation_change` | DoD = `after - before` |
| `level_of_detection`, `significant_change` | Threshold the DoD at its level of detection |
| `volume_change` | Cut, fill and net volumes, with uncertainty |
| `stockpile_volume` | Volume above a base surface fitted to the pile's toe |

Every function is available as `farq.<name>` (for example `farq.volume_change`) and from
`farq.elevation`. No extra dependency is needed beyond `pip install farq`.

## Conventions

- DEMs are 2-D arrays. NaN (or `nodata=`, or masked entries) is nodata, and nodata
  pixels never count in volumes.
- **Elevations must be in metres.** Volumes are in m³ and areas in m².
- The grid geometry (`meta`) is a metadata dict from `farq.read` or `farq.align_pair`, an
  open rasterio dataset, an affine transform, or a pixel size in metres. Horizontal units
  of projected CRSs are converted to metres (for example US survey feet).
- Volumes and slopes need a projected CRS. **Geographic CRSs (degrees) are refused**, and
  so is metadata without a CRS or with GCPs only. Reproject with
  `farq.align_pair(..., dst_crs="EPSG:326xx")` or rectify with `farq.rectify` first.

## Example data

The recipes on this page run in order and share variables. This first block writes two
synthetic 20 cm drone DSMs of the same 60 m × 60 m site. Between the surveys a 2 m high
stockpile was placed and a pit was dug, and the second survey is offset by 0.3 m east,
0.2 m south and 5 cm up, a typical GNSS error.

```python
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

import farq

res, n = 0.2, 300
rows, cols = np.indices((n, n)) + 0.5
x, y = cols * res, -rows * res  # metres from the top-left corner


def ground(x, y):
    """Undulating natural ground."""
    return 120 + 1.5 * np.sin(x / 7) * np.cos(y / 9) + 0.8 * np.sin((x - y) / 5) + 0.02 * x


rng = np.random.default_rng(0)
before = ground(x, y) + rng.normal(0, 0.02, (n, n))

dx, dy, dz = 0.3, -0.2, 0.05  # GNSS offset of the second survey
after = ground(x - dx, y - dy) + dz + rng.normal(0, 0.02, (n, n))
pile = np.clip(2.0 - np.hypot(x - dx - 40, y - dy + 20) / 3, 0, 1.6)  # truncated cone
pit = np.hypot(x - 15, y + 40) < 5
after += pile
after[pit] -= 1.0
after[220:245, 230:270] = np.nan  # a gap in the second survey

meta = {"driver": "GTiff", "dtype": "float32", "nodata": np.nan, "count": 1,
        "width": n, "height": n, "crs": CRS.from_epsg(32633),
        "transform": from_origin(500000.0, 4000000.0, res, res)}
farq.write("dsm_2023.tif", before.astype(np.float32), meta)
farq.write("dsm_2024.tif", after.astype(np.float32), meta)
```

## 1. Read and align the surveys

```python
import farq

dsm_23, meta_23 = farq.read("dsm_2023.tif", masked=True)  # nodata -> NaN
dsm_24, meta_24 = farq.read("dsm_2024.tif", masked=True)
before, after, meta = farq.align_pair(dsm_23, meta_23, dsm_24, meta_24)
print(before.shape, farq.pixel_size(meta))  # (300, 300) (0.2, 0.2)
```

`align_pair` puts both surveys on one grid (resampling, cropping to the overlap). Use
`target="coarsest"` when the two surveys have different resolutions.

## 2. Look at the terrain

```python
shade = farq.hillshade(before, meta)       # [0, 1], light from the north-west
slope_deg = farq.slope(before, meta)       # degrees ("radians" and "percent" too)
facing = farq.aspect(before, meta)         # degrees clockwise from north
print(f"median slope {np.nanmedian(slope_deg):.1f}°")

fig = farq.plot(shade, title="Hillshade 2023", cmap="gray")
fig.savefig("hillshade_2023.png")
```

Derivatives use central differences with the true pixel spacing (including rotated or
non-square pixels). At raster edges and next to nodata holes they use one-sided
differences.

## 3. Co-register on stable ground

A 5 cm vertical offset over a 1 ha site is a 500 m³ false volume. A horizontal shift
creates false cut on slopes facing one way and false fill on slopes facing the other way.
Estimate both on **stable ground**: terrain that did not change between the surveys.
Exclude the works, stockpiles, vegetation, vehicles and water.

```python
works = np.zeros(before.shape, bool)
works[50:250, 30:280] = True  # the active area; usually a rasterized polygon
stable = ~works

reg = farq.coregister_dem(before, after, meta, stable_mask=stable)
print(f"offset E {reg.dx:+.2f} m, N {reg.dy:+.2f} m, up {reg.dz:+.3f} m "
      f"after {reg.iterations} iterations; "
      f"NMAD {reg.nmad_before:.3f} m -> {reg.nmad_after:.3f} m")
after_reg = reg.dem  # the 2024 surface, moved onto the 2023 one
```

`coregister_dem` implements Nuth & Kääb (2011). A horizontal shift `(dx, dy)` makes the
elevation difference on a slope follow `dh = tan(slope) · a · cos(b − aspect)`. The
function fits this cosine to `dh / tan(slope)` per aspect bin, shifts the DEM, and
repeats until the update is below 1 % of a pixel. The vertical bias is the median of the
remaining differences. The method needs sloping stable terrain that faces several
directions. On flat or planar sites, correct only the vertical offset:

```python
offset, nmad = farq.vertical_offset(before, after, stable_mask=stable)
after_v = after - offset
print(f"vertical offset {offset:+.3f} m, NMAD {nmad:.3f} m")
```

`method="nmad_trimmed"` drops differences beyond 3 NMAD and averages the rest. It is
more precise on clean stable ground. To apply a shift that you estimated on a crop or a
coarser copy to the full DEM, use `shift_dem(dem, meta, dx, dy, dz)`.

## 4. DEM of difference and level of detection

```python
dod = farq.elevation_change(before, after_reg)  # after - before, NaN if either is nodata

# Survey noise: NMAD over stable ground (sigma of the difference itself), or the
# check-point RMSE of each survey, e.g. farq.level_of_detection(0.03, 0.04).
sigma_dod = reg.nmad_after
lod = farq.level_of_detection(sigma_dod, 0)  # 95 %: 1.96 * sigma
print(f"LoD {lod:.3f} m")

dod_sig = farq.significant_change(dod, lod)  # |dh| <= LoD -> 0, NaN kept
fig = farq.changes(dod_sig, title="Elevation change 2023-2024", colorbar_label="Δh (m)")
fig.savefig("dod.png", dpi=150)
```

`level_of_detection(σ1, σ2)` is `t · sqrt(σ1² + σ2²)`, with `t` = 1.96 for 95 %
confidence (Brasington et al. 2003, Wheaton et al. 2010). If `sigma_after` is omitted, it
equals `sigma_before`, which suits two surveys of the same quality. Pass `sigma_after=0`
when `sigma_before` is already the error of the difference. Both sigmas can be per-pixel
arrays, for example larger on steep slopes or in poorly matched areas.

## 5. Cut and fill volumes

```python
vol = farq.volume_change(dod, meta, lod=lod, mask=works, sigma=sigma_dod,
                         correlation_length=5.0)
print(f"fill {vol.fill_m3:7.1f} ± {vol.fill_uncertainty_m3:.1f} m³ "
      f"over {vol.fill_area_m2:.0f} m²")
print(f"cut  {vol.cut_m3:7.1f} ± {vol.cut_uncertainty_m3:.1f} m³ "
      f"over {vol.cut_area_m2:.0f} m²")
print(f"net  {vol.net_m3:+7.1f} ± {vol.uncertainty_m3:.1f} m³; "
      f"no data over {vol.nodata_area_m2:.1f} m²")
report = vol.to_dict()  # JSON-serializable
```

Cut and fill are both positive numbers, and `net = fill − cut`. Always check
`nodata_area_m2`: gaps in either survey are left out, so the volumes are incomplete.

At 95 % confidence, about 5 % of the unchanged pixels still exceed the LoD by chance.
These isolated specks add a small spurious cut and fill. Remove them with
`farq.clean_mask` before computing volumes:

```python
keep = farq.clean_mask(dod_sig != 0, min_size=25)  # drop patches under 1 m² (25 px)
vol_clean = farq.volume_change(np.where(keep, dod, 0.0), meta, lod=lod, mask=works)
print(f"after cleaning: fill {vol_clean.fill_m3:.1f} m³, cut {vol_clean.cut_m3:.1f} m³")
```

The uncertainty (one sigma) uses the per-pixel DoD error `σ` over the `n` pixels of each
volume, where `A` is the pixel area and `S = n · A`:

- With **uncorrelated** errors: `σ_V = A · sqrt(Σ σ_i²) = σ · A · sqrt(n)`. This is
  optimistic, because DEM errors are spatially correlated.
- With `correlation_length = L`, the range of a spherical variogram of the DoD over
  stable ground (Rolstad et al. 2009), the area is treated as a disc of radius
  `r = sqrt(S/π)`. Then `σ_S² = σ²(1 − r/L + r³/5L³)` if `r < L`, else
  `σ_S² = σ² πL²/(5S)`, and `σ_V = S · σ_S`. The result is never smaller than the
  uncorrelated value.

## 6. Stockpile volume from one survey

You often need the volume of a pile from a single flight, with no survey from before the
pile existed. `stockpile_volume` fits a base surface to the **toe ring**, the valid
pixels just outside the footprint, and sums the height above it. Draw the footprint
along the toe, on bare ground. The example rasterizes a polygon with rasterio:

```python
from rasterio.features import rasterize

t = meta["transform"]
cx, cy = t.c + 40, t.f - 20  # pile centre in map coordinates
angles = np.linspace(0, 2 * np.pi, 64)
ring = [(cx + 6.5 * np.cos(a), cy + 6.5 * np.sin(a)) for a in angles]
footprint = rasterize([{"type": "Polygon", "coordinates": [ring]}],
                      out_shape=after_reg.shape, transform=t).astype(bool)

pile_vol = farq.stockpile_volume(after_reg, footprint, meta, base="plane",
                                 sigma=sigma_dod, correlation_length=5.0)
print(f"stockpile: {pile_vol.volume_m3:.1f} ± {pile_vol.uncertainty_m3:.1f} m³ on "
      f"{pile_vol.area_m2:.1f} m², max height {pile_vol.max_height_m:.2f} m, "
      f"base slope {pile_vol.base_slope_deg:.1f}°, base fit RMSE {pile_vol.base_rmse_m:.3f} m")

# With a survey of the empty pad, use it as the base surface instead.
on_pad = farq.stockpile_volume(after_reg, footprint, meta, base=before)
print(f"above the 2023 surface: {on_pad.volume_m3:.1f} m³")

# Analytic volume of the synthetic pile: a 2 m cone of radius 6 m, truncated at 1.6 m.
analytic = np.pi * 6.0**2 * 2.0 / 3 - np.pi * 1.2**2 * 0.4 / 3
print(f"analytic: {analytic:.1f} m³")
```

Here the plane base gives a smaller volume than the true one: the natural ground under
the pile is curved, and `base_rmse_m` (about 0.1 m) shows that the toe ring is not
planar. Whenever a survey of the empty site exists, pass it as `base`.

`base` choices:

- `"plane"` (default): a least-squares plane through the toe ring. Use it for piles on
  sloping or uneven pads.
- `"lowest"`: a horizontal plane at the lowest toe point. It is conservative and gives
  a larger volume on sloping ground.
- `"mean"`: a horizontal plane at the mean toe elevation.
- A number: a known pad elevation.
- An array: a base surface on the same grid, such as a survey of the empty pad.

`base_rmse_m` is the misfit between the toe ring and the base. A large value means the
ground around the pile is not planar, or the footprint cuts through the pile. Increase
`ring_width` to average over a wider ring. `below_base_m3` reports voids below the base,
which should be close to zero.

## Limitations

- `coregister_dem` estimates a translation only. It does not model tilt, rotation or
  elevation-dependent bias, and it cannot constrain a horizontal shift on flat or planar
  terrain.
- DSMs include vegetation, buildings and machines. Mask them from the stable ground and
  from the volume regions.
- The Rolstad uncertainty assumes a single spherical variogram and a roughly circular
  area. It does not include errors in the base surface or in the footprint delineation.

## References

- Brasington, J., Langham, J., Rumsby, B. (2003). Methodological sensitivity of
  morphometric estimates of coarse fluvial sediment transport. *Geomorphology* 53.
- Nuth, C., Kääb, A. (2011). Co-registration and bias corrections of satellite elevation
  data sets for quantifying glacier thickness change. *The Cryosphere* 5, 271–290.
- Rolstad, C., Haug, T., Denby, B. (2009). Spatially integrated geodetic glacier mass
  balance and its uncertainty based on geostatistical analysis. *Journal of Glaciology*
  55(192).
- Wheaton, J. M., Brasington, J., Darby, S. E., Sear, D. A. (2010). Accounting for
  uncertainty in DEMs from repeat topographic surveys. *Earth Surface Processes and
  Landforms* 35.
