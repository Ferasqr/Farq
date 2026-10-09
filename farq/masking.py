"""
Cloud, cloud-shadow, snow and water masking from satellite quality bands.

Clouds, their shadows and snow change from one acquisition to the next, so an unmasked
cloud is the most common source of false change. This module decodes the quality bands
that ship with the main open satellite products into boolean masks (``True`` = masked,
i.e. *bad* pixel), and applies them as NaN so that every farq change function ignores
those pixels.

Workflow
--------
1. Build a mask per date from its quality band: :func:`landsat_qa_mask` (Landsat
   Collection 2 ``QA_PIXEL``), :func:`landsat_radsat_mask` (``QA_RADSAT``),
   :func:`sentinel2_scl_mask` (Sentinel-2 L2A scene classification),
   :func:`sentinel2_cloud_probability_mask` (``MSK_CLDPRB``) or :func:`hls_fmask_mask`
   (HLS v2.0 ``Fmask``). :func:`decode_bits` extracts arbitrary bit fields.
2. Grow the masks with :func:`buffer_mask` (cloud edges and shadows leak beyond the
   flagged pixels) and merge them with :func:`combine_masks`.
3. Check how much of the pair is usable with :func:`valid_overlap` /
   :func:`clear_fraction`.
4. :func:`apply_mask` sets masked pixels to NaN in a float copy of the image.

:func:`landsat_c2_scale` and :func:`sentinel2_l2a_scale` convert the integer digital
numbers of these products to physical units (reflectance, kelvin), with fill as NaN.

Conventions
-----------
* Masks are boolean arrays where ``True`` means *masked* (unusable).
* Quality bands must be integer arrays. Float quality bands (e.g. read with
  ``farq.read(..., masked=True)``) are accepted when every finite value is a whole
  number; NaN and masked entries of a :class:`numpy.ma.MaskedArray` are nodata and
  always come out masked.
* Inputs are never modified. No ``RuntimeWarning`` is emitted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from enum import IntEnum
from numbers import Integral, Real
from typing import Any, NamedTuple

import numpy as np
from scipy import ndimage

from .utils import _check_output_size

__all__ = [
    "DEFAULT_S2_BAD_CLASSES",
    "SCL_NAMES",
    "Confidence",
    "HLSFmask",
    "LandsatQA",
    "MaskOverlap",
    "SCLClass",
    "apply_mask",
    "buffer_mask",
    "clear_fraction",
    "combine_masks",
    "decode_bits",
    "decode_landsat_qa",
    "hls_fmask_mask",
    "landsat_c2_scale",
    "landsat_qa_mask",
    "landsat_radsat_mask",
    "sentinel2_cloud_probability_mask",
    "sentinel2_l2a_scale",
    "sentinel2_scl_mask",
    "upsample_mask",
    "valid_overlap",
]


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #
class LandsatQA(IntEnum):
    """Bit positions of the Landsat Collection 2 Level-2 ``QA_PIXEL`` band.

    Identical for Landsat 8/9 OLI/TIRS and Landsat 4-7 TM/ETM+, except that the
    cirrus flag (bit 2) and cirrus confidence (bits 14-15) exist only for Landsat 8/9
    (always 0 for Landsat 4-7). The ``*_CONFIDENCE`` members are the lowest bit of a
    2-bit field, see :class:`Confidence`.

    Source: USGS, *Landsat 8-9 Collection 2 Level 2 Science Product Guide* (LSDS-1619),
    and *Landsat 4-7 Collection 2 Level 2 Science Product Guide* (LSDS-1618), section
    "Pixel Quality Assessment (QA_PIXEL) Band".
    """

    FILL = 0  #: 1 = fill (no data)
    DILATED_CLOUD = 1  #: 1 = cloud dilation
    CIRRUS = 2  #: 1 = high-confidence cirrus (Landsat 8/9 only)
    CLOUD = 3  #: 1 = high-confidence cloud
    CLOUD_SHADOW = 4  #: 1 = high-confidence cloud shadow
    SNOW = 5  #: 1 = high-confidence snow cover
    CLEAR = 6  #: 1 = neither the cloud nor the dilated-cloud bit is set
    WATER = 7  #: 1 = water
    CLOUD_CONFIDENCE = 8  #: bits 8-9
    CLOUD_SHADOW_CONFIDENCE = 10  #: bits 10-11
    SNOW_ICE_CONFIDENCE = 12  #: bits 12-13
    CIRRUS_CONFIDENCE = 14  #: bits 14-15 (Landsat 8/9 only)


class Confidence(IntEnum):
    """Values of the 2-bit Landsat Collection 2 ``QA_PIXEL`` confidence fields.

    For cloud confidence (bits 8-9) all four values are used. For cloud-shadow,
    snow/ice and cirrus confidence, ``MEDIUM`` (binary ``10``) is *reserved* and not
    produced, so those fields are ``NONE``, ``LOW`` or ``HIGH`` (LSDS-1619, LSDS-1618).
    """

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class SCLClass(IntEnum):
    """Sentinel-2 Level-2A Scene Classification (``SCL``) values.

    Source: ESA, *Sentinel-2 MSI Level-2A Product Format Specification* /
    *Sen2Cor Software Release Note*, SCL band (20 m and 60 m). From processing
    baseline 04.00 class 2 is labelled "topographic casted shadows" (earlier baselines:
    "dark area pixels").
    """

    NO_DATA = 0
    SATURATED_DEFECTIVE = 1
    DARK_AREA = 2  #: dark area pixels / topographic cast shadows (baseline 04.00+)
    CLOUD_SHADOW = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM = 8  #: cloud, medium probability
    CLOUD_HIGH = 9  #: cloud, high probability
    THIN_CIRRUS = 10
    SNOW_ICE = 11


#: Lower-case names of the SCL classes, accepted by :func:`sentinel2_scl_mask`.
SCL_NAMES: dict[int, str] = {
    SCLClass.NO_DATA: "no_data",
    SCLClass.SATURATED_DEFECTIVE: "saturated_defective",
    SCLClass.DARK_AREA: "dark_area",
    SCLClass.CLOUD_SHADOW: "cloud_shadow",
    SCLClass.VEGETATION: "vegetation",
    SCLClass.NOT_VEGETATED: "not_vegetated",
    SCLClass.WATER: "water",
    SCLClass.UNCLASSIFIED: "unclassified",
    SCLClass.CLOUD_MEDIUM: "cloud_medium",
    SCLClass.CLOUD_HIGH: "cloud_high",
    SCLClass.THIN_CIRRUS: "thin_cirrus",
    SCLClass.SNOW_ICE: "snow_ice",
}
_SCL_ALIASES = {
    "nodata": 0,
    "saturated": 1,
    "defective": 1,
    "cast_shadow": 2,
    "cast_shadows": 2,
    "topographic_shadow": 2,
    "shadow": 3,
    "cloud_shadows": 3,
    "bare": 5,
    "cloud_medium_probability": 8,
    "cloud_high_probability": 9,
    "cirrus": 10,
    "snow": 11,
}

#: SCL classes masked by default: no data, saturated/defective, cloud shadow, cloud
#: (medium and high probability) and thin cirrus. Snow (11), cast shadows (2) and
#: unclassified (7) are kept; add them when they would show up as false change.
DEFAULT_S2_BAD_CLASSES: frozenset[int] = frozenset(
    {
        SCLClass.NO_DATA,
        SCLClass.SATURATED_DEFECTIVE,
        SCLClass.CLOUD_SHADOW,
        SCLClass.CLOUD_MEDIUM,
        SCLClass.CLOUD_HIGH,
        SCLClass.THIN_CIRRUS,
    }
)


class HLSFmask(IntEnum):
    """Bit positions of the HLS v2.0 ``Fmask`` quality band (8-bit; fill value 255).

    Bit 0 (cirrus) is reserved and not used in HLS v2.0. Bits 6-7 are the aerosol
    level: 0 climatology, 1 low, 2 moderate, 3 high.

    Source: NASA LP DAAC, *Harmonized Landsat Sentinel-2 (HLS) Product User Guide,
    Version 2.0*, Table 9.
    """

    CIRRUS = 0
    CLOUD = 1
    ADJACENT = 2  #: adjacent to cloud or cloud shadow
    CLOUD_SHADOW = 3
    SNOW_ICE = 4
    WATER = 5
    AEROSOL_LEVEL = 6  #: bits 6-7


_HLS_FILL = 255
_HLS_AEROSOL = {"climatology": 0, "low": 1, "moderate": 2, "medium": 2, "high": 3}

# Landsat QA_RADSAT bit of each band (USGS LSDS-1619 / LSDS-1618, "Radiometric
# Saturation Quality Assessment (QA_RADSAT) Band").
_RADSAT_BITS: dict[str, dict[Any, int]] = {
    # Landsat 8/9 OLI: bands 1-7 -> bits 0-6, band 9 (cirrus) -> bit 8.
    "oli": {**{b: b - 1 for b in range(1, 8)}, 9: 8},
    # Landsat 4/5 TM: bands 1-7 -> bits 0-6.
    "tm": {b: b - 1 for b in range(1, 8)},
    # Landsat 7 ETM+: bands 1-7 -> bits 0-6 (band 6 = 6L), band 6H -> bit 8.
    "etm": {**{b: b - 1 for b in range(1, 8)}, "6L": 5, "6H": 8},
}
_RADSAT_TERRAIN_OCCLUSION = 11  # Landsat 8/9 only
_RADSAT_DROPPED_PIXEL = 9  # Landsat 4-7 only

# Landsat Collection 2 Level-2 scale factors (LSDS-1619 / LSDS-1618, "Scale factor").
_LANDSAT_SCALE = {"sr": (2.75e-05, -0.2), "st": (0.00341802, 149.0)}


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _check_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _as_array(array: Any, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(plain ndarray, masked-entries mask or None)``."""
    invalid = None
    if isinstance(array, np.ma.MaskedArray):
        m = np.ma.getmaskarray(array)
        invalid = m if m.any() else None
        array = np.asarray(array.data)
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return array, invalid


def _as_qa(qa: Any, name: str, max_bits: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate a quality band; return ``(unsigned integer array, nodata mask or None)``.

    Integer input is viewed as unsigned (negative values raise). Float input must hold
    whole numbers; NaN entries are nodata. Values must fit in ``max_bits`` bits.
    """
    data, invalid = _as_array(qa, name)
    kind = data.dtype.kind
    if kind == "f":
        nan = np.isnan(data)
        finite = data[~nan]
        if np.isinf(finite).any() or (finite != np.floor(finite)).any():
            raise ValueError(f"{name} must contain whole numbers (or NaN for nodata)")
        if finite.size and (finite.min() < 0 or finite.max() >= 2.0**max_bits):
            raise ValueError(
                f"{name} values must be in [0, {2**max_bits - 1}], "
                f"got [{finite.min():g}, {finite.max():g}]"
            )
        out = np.where(nan, 0, data).astype(np.uint64 if max_bits > 32 else np.uint32)
        if nan.any():
            invalid = nan if invalid is None else invalid | nan
        return out, invalid
    if kind not in "iu":
        raise TypeError(f"{name} must have an integer dtype, got {data.dtype}")
    if kind == "i":
        lo = int(data.min())
        if lo < 0:
            raise ValueError(f"{name} must not contain negative values, got {lo}")
        data = data.astype(np.dtype(f"u{data.dtype.itemsize}"))
    if data.dtype.itemsize * 8 > max_bits:
        hi = int(data.max())
        if hi >= 1 << max_bits:
            raise ValueError(
                f"{name} values must fit in {max_bits} bits (<= {(1 << max_bits) - 1}), got {hi}"
            )
    return data, invalid


def _field(qa: np.ndarray, bit: int, width: int = 1) -> np.ndarray:
    """Extract a bit field of an unsigned array (no validation)."""
    shifted = qa >> qa.dtype.type(bit) if bit else qa
    return shifted & qa.dtype.type((1 << width) - 1)


def _with_nodata(mask: np.ndarray, invalid: np.ndarray | None) -> np.ndarray:
    if invalid is not None:
        mask |= invalid
    return mask


def _as_bool_mask(mask: Any, name: str) -> np.ndarray:
    """Boolean mask from bool/0-1/float input; NaN and masked entries count as True."""
    data, invalid = _as_array(mask, name)
    if data.dtype.kind not in "biuf":
        raise TypeError(f"{name} must have a boolean, integer or float dtype, got {data.dtype}")
    if data.dtype == bool:
        out = data.copy() if invalid is not None else data
    elif data.dtype.kind == "f":
        out = (data != 0) | np.isnan(data)
    else:
        out = data != 0
    return _with_nodata(out, invalid)


def _float_dtype(dtype: np.dtype) -> np.dtype:
    result = np.result_type(dtype, np.float32)
    return np.dtype(np.float64) if result.itemsize > 8 else result


def _scale_input(dn: Any, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    data, invalid = _as_array(dn, name)
    if data.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have an integer or float dtype, got {data.dtype}")
    return data, invalid


def _parse_level(value: Any, names: Mapping[str, int], name: str) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in names:
            raise ValueError(f"Unknown {name} {value!r}; expected one of {sorted(names)}")
        return names[key]
    level = _check_int(value, name, 0)
    if level > 3:
        raise ValueError(f"{name} must be between 0 and 3, got {level}")
    return level


def _check_target_shape(target_shape: Any) -> tuple[int, int]:
    if not isinstance(target_shape, (tuple, list)) or len(target_shape) != 2:
        raise TypeError(f"target_shape must be a (height, width) tuple, got {target_shape!r}")
    h, w = (_check_int(v, "target_shape dimension", 1) for v in target_shape)
    return h, w


# --------------------------------------------------------------------------- #
# Bit decoding
# --------------------------------------------------------------------------- #
def decode_bits(qa: np.ndarray, bit: int, width: int = 1) -> np.ndarray:
    """Extract a bit field from an integer quality band.

    Parameters
    ----------
    qa : numpy.ndarray
        Integer quality band of any shape (e.g. ``uint16`` Landsat ``QA_PIXEL``).
        Signed integers are accepted when no value is negative.
    bit : int
        Position of the lowest bit of the field (0 = least significant bit).
    width : int, default 1
        Number of bits in the field (e.g. 2 for Landsat confidence fields).

    Returns
    -------
    numpy.ndarray
        Field values, ``(qa >> bit) & (2**width - 1)``, with the shape of ``qa``.
        The dtype is the smallest unsigned integer holding ``width`` bits (``uint8``
        for up to 8 bits).

    Raises
    ------
    TypeError
        If ``qa`` is not an integer array or ``bit``/``width`` are not integers.
    ValueError
        If ``qa`` is empty or negative, ``width < 1``, or ``bit + width`` exceeds the
        number of bits of the dtype.

    Examples
    --------
    >>> import numpy as np
    >>> qa = np.array([21824, 22280], dtype=np.uint16)  # Landsat 8: clear, cloud
    >>> decode_bits(qa, 3)  # cloud bit
    array([0, 1], dtype=uint8)
    >>> decode_bits(qa, 8, width=2)  # cloud confidence: low, high
    array([1, 3], dtype=uint8)
    """
    data, _ = _as_array(qa, "qa")
    if data.dtype.kind not in "iu":
        raise TypeError(f"qa must have an integer dtype, got {data.dtype}")
    bit = _check_int(bit, "bit", 0)
    width = _check_int(width, "width", 1)
    nbits = data.dtype.itemsize * 8
    if bit + width > nbits:
        raise ValueError(
            f"bit + width = {bit + width} exceeds the {nbits} bits of dtype {data.dtype}"
        )
    data, _ = _as_qa(data, "qa", nbits)
    out_bits = next(n for n in (8, 16, 32, 64) if n >= width)
    return _field(data, bit, width).astype(np.dtype(f"u{out_bits // 8}"))


# --------------------------------------------------------------------------- #
# Landsat Collection 2
# --------------------------------------------------------------------------- #
def decode_landsat_qa(qa_pixel: np.ndarray) -> dict[str, np.ndarray]:
    """Decode every field of a Landsat Collection 2 ``QA_PIXEL`` band.

    Parameters
    ----------
    qa_pixel : numpy.ndarray
        ``QA_PIXEL`` band (``uint16``) of a Landsat 4-9 Collection 2 Level-1 or Level-2
        product.

    Returns
    -------
    dict of str to numpy.ndarray
        Boolean flags ``fill``, ``dilated_cloud``, ``cirrus``, ``cloud``,
        ``cloud_shadow``, ``snow``, ``clear``, ``water`` and ``uint8`` confidence levels
        (0-3, see :class:`Confidence`) ``cloud_confidence``,
        ``cloud_shadow_confidence``, ``snow_ice_confidence``, ``cirrus_confidence``.
        NaN (nodata) entries of a float band decode as 0.

    Notes
    -----
    Bit layout (USGS LSDS-1619 for Landsat 8/9, LSDS-1618 for Landsat 4-7): 0 fill,
    1 dilated cloud, 2 cirrus (8/9 only), 3 cloud, 4 cloud shadow, 5 snow, 6 clear,
    7 water, 8-9 cloud confidence, 10-11 cloud-shadow confidence, 12-13 snow/ice
    confidence, 14-15 cirrus confidence (8/9 only).
    """
    qa, _ = _as_qa(qa_pixel, "qa_pixel", 16)
    flags = {
        "fill": LandsatQA.FILL,
        "dilated_cloud": LandsatQA.DILATED_CLOUD,
        "cirrus": LandsatQA.CIRRUS,
        "cloud": LandsatQA.CLOUD,
        "cloud_shadow": LandsatQA.CLOUD_SHADOW,
        "snow": LandsatQA.SNOW,
        "clear": LandsatQA.CLEAR,
        "water": LandsatQA.WATER,
    }
    out: dict[str, np.ndarray] = {k: _field(qa, b).astype(bool) for k, b in flags.items()}
    confidences = {
        "cloud_confidence": LandsatQA.CLOUD_CONFIDENCE,
        "cloud_shadow_confidence": LandsatQA.CLOUD_SHADOW_CONFIDENCE,
        "snow_ice_confidence": LandsatQA.SNOW_ICE_CONFIDENCE,
        "cirrus_confidence": LandsatQA.CIRRUS_CONFIDENCE,
    }
    for k, b in confidences.items():
        out[k] = _field(qa, b, 2).astype(np.uint8)
    return out


def landsat_qa_mask(
    qa_pixel: np.ndarray,
    *,
    cloud: bool = True,
    shadow: bool = True,
    cirrus: bool = True,
    snow: bool = False,
    dilated: bool = True,
    fill: bool = True,
    water: bool = False,
    min_confidence: int | str | None = None,
) -> np.ndarray:
    """Mask of unusable pixels from a Landsat Collection 2 ``QA_PIXEL`` band.

    Works for Landsat 8/9 OLI/TIRS and Landsat 4-7 TM/ETM+ Collection 2 (Level-1 and
    Level-2). Landsat 4-7 have no cirrus band, so their cirrus bits are always 0 and
    ``cirrus`` has no effect.

    Parameters
    ----------
    qa_pixel : numpy.ndarray
        ``QA_PIXEL`` band, normally ``uint16``. NaN / masked entries are masked.
    cloud, shadow, cirrus, snow, dilated, fill, water : bool
        Which flags to mask: cloud (bit 3), cloud shadow (bit 4), cirrus (bit 2), snow
        (bit 5), dilated cloud (bit 1), fill (bit 0) and water (bit 7). The single-bit
        flags are set by CFMask for *high-confidence* detections.
    min_confidence : {None, "low", "medium", "high"} or int, optional
        Also mask pixels whose confidence field is at least this level (1 low,
        2 medium, 3 high) for each enabled category: cloud (bits 8-9), cloud shadow
        (bits 10-11), snow/ice (bits 12-13) and cirrus (bits 14-15). For shadow,
        snow/ice and cirrus the value 2 is reserved, so ``"medium"`` behaves like
        ``"high"`` there. Note that CFMask sets *low* confidence on almost every
        clear pixel, so ``"low"`` masks nearly the whole scene; ``"medium"`` is the
        useful setting for a more conservative cloud mask.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the shape of ``qa_pixel``; ``True`` = masked.

    Raises
    ------
    TypeError, ValueError
        If ``qa_pixel`` is not an integer band with values in ``[0, 65535]`` or
        ``min_confidence`` is not a valid level.

    Notes
    -----
    Source: USGS *Landsat 8-9 Collection 2 Level 2 Science Product Guide* (LSDS-1619)
    and *Landsat 4-7 Collection 2 Level 2 Science Product Guide* (LSDS-1618), "Pixel
    Quality Assessment (QA_PIXEL) Band". See :class:`LandsatQA`.

    Examples
    --------
    >>> import numpy as np
    >>> qa = np.array([21824, 21952, 22280, 23888, 1], dtype=np.uint16)
    >>> landsat_qa_mask(qa)  # clear land, clear water, cloud, shadow, fill
    array([False, False,  True,  True,  True])
    """
    qa, invalid = _as_qa(qa_pixel, "qa_pixel", 16)
    bits = []
    if fill:
        bits.append(LandsatQA.FILL)
    if dilated:
        bits.append(LandsatQA.DILATED_CLOUD)
    if cirrus:
        bits.append(LandsatQA.CIRRUS)
    if cloud:
        bits.append(LandsatQA.CLOUD)
    if shadow:
        bits.append(LandsatQA.CLOUD_SHADOW)
    if snow:
        bits.append(LandsatQA.SNOW)
    if water:
        bits.append(LandsatQA.WATER)
    flag_bits = sum(1 << b for b in bits)
    mask = (qa & qa.dtype.type(flag_bits)) != 0

    if min_confidence is not None:
        names = {"none": 0, "low": 1, "medium": 2, "high": 3}
        level = _parse_level(min_confidence, names, "min_confidence")
        if level == 0:
            raise ValueError("min_confidence must be at least 1 ('low'); 0 would mask everything")
        fields = [
            (cloud, LandsatQA.CLOUD_CONFIDENCE, False),
            (shadow, LandsatQA.CLOUD_SHADOW_CONFIDENCE, True),
            (snow, LandsatQA.SNOW_ICE_CONFIDENCE, True),
            (cirrus, LandsatQA.CIRRUS_CONFIDENCE, True),
        ]
        for enabled, bit, medium_reserved in fields:
            if enabled:
                field_level = 3 if (medium_reserved and level == 2) else level
                mask |= _field(qa, bit, 2) >= field_level
    return _with_nodata(mask, invalid)


def landsat_radsat_mask(
    qa_radsat: np.ndarray,
    *,
    sensor: str = "oli",
    bands: Iterable[int | str] | None = None,
    terrain_occlusion: bool = True,
    dropped_pixel: bool = True,
) -> np.ndarray:
    """Mask of saturated (and occluded/dropped) pixels from a Landsat ``QA_RADSAT`` band.

    Parameters
    ----------
    qa_radsat : numpy.ndarray
        ``QA_RADSAT`` band (``uint16``) of a Landsat Collection 2 product.
    sensor : {"oli", "tm", "etm"}, default "oli"
        ``"oli"``: Landsat 8/9 (bands 1-7 -> bits 0-6, band 9 -> bit 8, terrain
        occlusion bit 11). ``"tm"``: Landsat 4/5 (bands 1-7 -> bits 0-6, dropped pixel
        bit 9). ``"etm"``: Landsat 7 (bands 1-7 -> bits 0-6 with band 6 = ``"6L"``,
        ``"6H"`` -> bit 8, dropped pixel bit 9).
    bands : iterable of int or str, optional
        Band numbers whose saturation is masked (e.g. ``[3, 5]`` for the NDWI bands of
        Landsat 8). ``None`` (default) masks saturation in any band of the sensor.
    terrain_occlusion : bool, default True
        Mask terrain occlusion (bit 11; Landsat 8/9 only, ignored otherwise).
    dropped_pixel : bool, default True
        Mask dropped pixels (bit 9; Landsat 4-7 only, ignored for ``"oli"``).

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` = masked.

    Notes
    -----
    Source: USGS LSDS-1619 / LSDS-1618, "Radiometric Saturation Quality Assessment
    (QA_RADSAT) Band".

    Examples
    --------
    >>> import numpy as np
    >>> radsat = np.array([0, 1 << 4, 1 << 2], dtype=np.uint16)  # -, band 5, band 3
    >>> landsat_radsat_mask(radsat, bands=[5])
    array([False,  True, False])
    """
    qa, invalid = _as_qa(qa_radsat, "qa_radsat", 16)
    if not isinstance(sensor, str) or sensor.lower() not in _RADSAT_BITS:
        raise ValueError(f"sensor must be one of {sorted(_RADSAT_BITS)}, got {sensor!r}")
    sensor = sensor.lower()
    table = _RADSAT_BITS[sensor]
    if bands is None:
        selected = set(table.values())
    else:
        if isinstance(bands, (str, int)):
            bands = [bands]
        selected = set()
        for band in bands:
            key = band.upper() if isinstance(band, str) else band
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            if isinstance(key, bool) or key not in table:
                valid = ", ".join(str(k) for k in table)
                raise ValueError(f"Band {band!r} has no QA_RADSAT bit for {sensor}; valid: {valid}")
            selected.add(table[key])
    if terrain_occlusion and sensor == "oli":
        selected.add(_RADSAT_TERRAIN_OCCLUSION)
    if dropped_pixel and sensor != "oli":
        selected.add(_RADSAT_DROPPED_PIXEL)
    flag_bits = sum(1 << b for b in selected)
    mask = (qa & qa.dtype.type(flag_bits)) != 0
    return _with_nodata(mask, invalid)


def landsat_c2_scale(dn: np.ndarray, kind: str = "sr", *, fill: float | None = 0) -> np.ndarray:
    """Convert Landsat Collection 2 Level-2 digital numbers to physical units.

    - ``kind="sr"`` (surface reflectance, bands ``SR_B*``):
      ``reflectance = DN * 0.0000275 - 0.2`` (unitless).
    - ``kind="st"`` (surface temperature, band ``ST_B10`` / ``ST_B6``):
      ``temperature = DN * 0.00341802 + 149.0`` (kelvin).

    Parameters
    ----------
    dn : numpy.ndarray
        Integer (normally ``uint16``) Level-2 band of any shape. NaN and masked
        entries stay NaN.
    kind : {"sr", "st"}, default "sr"
        Product type.
    fill : number or None, default 0
        Fill value of the product, returned as NaN. ``None`` disables it.

    Returns
    -------
    numpy.ndarray
        ``float32`` array (``float64`` for 32/64-bit integer or ``float64`` input).
        Reflectance is not clipped: valid SR digital numbers (7273-43636) map to
        about -0.0 to 1.0, but values slightly outside [0, 1] occur.

    Notes
    -----
    Source: USGS LSDS-1619 / LSDS-1618, "Scale factors" (Landsat Collection 2 Level-2
    Science Products). These factors do **not** apply to Collection 1 data
    (scale 0.0001).

    Examples
    --------
    >>> import numpy as np
    >>> landsat_c2_scale(np.array([0, 7273, 43636], dtype=np.uint16)).round(4)
    array([nan,  0.,  1.], dtype=float32)
    """
    if kind not in _LANDSAT_SCALE:
        raise ValueError(f"kind must be 'sr' or 'st', got {kind!r}")
    data, invalid = _scale_input(dn, "dn")
    scale, offset = _LANDSAT_SCALE[kind]
    dtype = _float_dtype(data.dtype)
    out = data.astype(dtype)
    if fill is not None:
        invalid = (data == fill) if invalid is None else invalid | (data == fill)
    out *= dtype.type(scale)
    out += dtype.type(offset)
    if invalid is not None:
        out[invalid] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Sentinel-2
# --------------------------------------------------------------------------- #
def _scl_class(value: Any) -> int:
    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        for code, name in SCL_NAMES.items():
            if name == key:
                return int(code)
        if key in _SCL_ALIASES:
            return _SCL_ALIASES[key]
        raise ValueError(
            f"Unknown SCL class name {value!r}; expected one of {list(SCL_NAMES.values())}"
        )
    code = _check_int(value, "SCL class", 0)
    if code > 11:
        raise ValueError(f"SCL classes are 0-11, got {code}")
    return code


def upsample_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample a 2-D mask by an exact integer factor.

    Used to apply a 20 m (or 60 m) Sentinel-2 quality band to 10 m bands of the same
    tile: each coarse pixel covers exactly ``factor x factor`` fine pixels, so
    repeating it is exact (no resampling error).

    Parameters
    ----------
    mask : numpy.ndarray
        2-D mask (any dtype; the dtype is kept).
    target_shape : tuple of int
        ``(height, width)``; each must be a whole multiple of the mask's size.

    Returns
    -------
    numpy.ndarray
        Array of shape ``target_shape``. The input itself is returned when the shapes
        already match.

    Raises
    ------
    ValueError
        If ``mask`` is not 2-D or ``target_shape`` is not an exact integer multiple
        (e.g. a 20 m band cropped differently from the 10 m band).
    """
    data, _ = _as_array(mask, "mask")
    if data.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {data.shape}")
    h, w = _check_target_shape(target_shape)
    if (h, w) == data.shape:
        return data
    fy, ry = divmod(h, data.shape[0])
    fx, rx = divmod(w, data.shape[1])
    if ry or rx or fy == 0 or fx == 0:
        raise ValueError(
            f"target_shape {(h, w)} is not an integer multiple of the mask shape "
            f"{data.shape}; the quality band and the image must cover the same extent "
            "(e.g. 20 m SCL 5490x5490 -> 10 m 10980x10980)"
        )
    _check_output_size((h, w), "upsampled mask")
    return np.repeat(np.repeat(data, fy, axis=0), fx, axis=1)


def sentinel2_scl_mask(
    scl: np.ndarray,
    *,
    classes: Iterable[int | str] = DEFAULT_S2_BAD_CLASSES,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Mask of unusable pixels from a Sentinel-2 L2A Scene Classification (SCL) band.

    Parameters
    ----------
    scl : numpy.ndarray
        SCL band (``uint8``, values 0-11), 20 m or 60 m. NaN / masked entries are
        masked.
    classes : iterable of int or str, default :data:`DEFAULT_S2_BAD_CLASSES`
        Classes to mask, as codes, :class:`SCLClass` members or names from
        :data:`SCL_NAMES` (e.g. ``{"cloud_high", "cloud_medium", "cloud_shadow",
        "thin_cirrus", "snow_ice"}``). Aliases ``"snow"``, ``"cirrus"``,
        ``"saturated"``, ``"cast_shadow"`` and ``"nodata"`` are also accepted.
    target_shape : tuple of int, optional
        Upsample the mask (nearest neighbour, exact integer factor; see
        :func:`upsample_mask`) to the shape of finer bands, e.g. ``(10980, 10980)``
        to apply the 20 m SCL to 10 m bands of the same tile.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` = masked.

    Raises
    ------
    ValueError
        If ``scl`` holds values outside 0-11, a class is unknown, or ``target_shape``
        is not an integer multiple of the SCL shape.

    Notes
    -----
    Classes (ESA Sentinel-2 L2A Product Specification): 0 no data, 1 saturated or
    defective, 2 dark area pixels (topographic cast shadows from baseline 04.00),
    3 cloud shadows, 4 vegetation, 5 not vegetated, 6 water, 7 unclassified, 8 cloud
    medium probability, 9 cloud high probability, 10 thin cirrus, 11 snow/ice.

    Examples
    --------
    >>> import numpy as np
    >>> scl = np.array([[4, 9], [3, 11]], dtype=np.uint8)
    >>> sentinel2_scl_mask(scl)
    array([[False,  True],
           [ True, False]])
    >>> int(sentinel2_scl_mask(scl, classes={"cloud_high", "snow"}).sum())
    2
    """
    data, invalid = _as_qa(scl, "scl", 8)
    if int(data.max()) > 11:
        raise ValueError(f"scl values must be SCL classes 0-11, got {int(data.max())}")
    if isinstance(classes, (str, int)):
        classes = [classes]
    lut = np.zeros(256, dtype=bool)
    for c in classes:
        lut[_scl_class(c)] = True
    mask = _with_nodata(lut[data], invalid)
    if target_shape is not None:
        mask = upsample_mask(mask, target_shape)
    return mask


def sentinel2_cloud_probability_mask(
    probability: np.ndarray,
    threshold: float = 50,
    *,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Mask from the Sentinel-2 ``MSK_CLDPRB`` cloud probability band (L1C and L2A).

    Parameters
    ----------
    probability : numpy.ndarray
        Cloud probability in percent (``uint8``, 0-100), 20 m or 60 m. NaN / masked
        entries are masked.
    threshold : float, default 50
        Pixels with ``probability >= threshold`` are masked. Must be in (0, 100].
    target_shape : tuple of int, optional
        Upsample to finer bands, see :func:`upsample_mask`.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` = masked.

    Notes
    -----
    ``MSK_CLDPRB`` is the per-pixel cloud probability (0-100 %) written in the
    ``QI_DATA`` folder of L1C (baseline 04.00+) and L2A products (ESA Sentinel-2
    Products Specification Document). It does not flag cloud shadows; combine it with
    the SCL shadow class or a :func:`buffer_mask`.

    Examples
    --------
    >>> import numpy as np
    >>> sentinel2_cloud_probability_mask(np.array([0, 40, 65, 100], dtype=np.uint8))
    array([False, False,  True,  True])
    """
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")
    if not 0 < threshold <= 100:
        raise ValueError(f"threshold must be in (0, 100], got {threshold}")
    data, invalid = _as_array(probability, "probability")
    if data.dtype.kind not in "iuf":
        raise TypeError(f"probability must have an integer or float dtype, got {data.dtype}")
    nan = np.isnan(data) if data.dtype.kind == "f" else None
    finite = data if nan is None else data[~nan]
    if finite.size and (finite.min() < 0 or finite.max() > 100):
        raise ValueError(
            f"probability must be in percent [0, 100], got [{finite.min()}, {finite.max()}]"
        )
    mask = data >= threshold  # NaN compares False
    if nan is not None:
        mask |= nan
    mask = _with_nodata(mask, invalid)
    if target_shape is not None:
        mask = upsample_mask(mask, target_shape)
    return mask


def sentinel2_l2a_scale(
    dn: np.ndarray,
    offset: float = -1000,
    quantification: float = 10000,
    *,
    nodata: float | None = 0,
) -> np.ndarray:
    """Convert Sentinel-2 digital numbers to reflectance: ``(DN + offset) / quantification``.

    Since processing baseline 04.00 (products from 25 January 2022) L2A bands carry a
    ``BOA_ADD_OFFSET`` of ``-1000`` (L1C: ``RADIO_ADD_OFFSET``), so that dark pixels
    can hold negative values. Mixing dates from before and after that baseline without
    harmonizing them creates a 0.1 reflectance jump that looks like change everywhere.

    Parameters
    ----------
    dn : numpy.ndarray
        Integer (normally ``uint16``) band of any shape. NaN and masked entries stay
        NaN.
    offset : float, default -1000
        ``BOA_ADD_OFFSET`` from the product metadata (``MTD_MSIL2A.xml``). Use ``0``
        for products processed with a baseline before 04.00.
    quantification : float, default 10000
        ``BOA_QUANTIFICATION_VALUE`` (``QUANTIFICATION_VALUE`` for L1C).
    nodata : number or None, default 0
        Digital number of nodata pixels (0 in Sentinel-2 products), returned as NaN.
        ``None`` disables it.

    Returns
    -------
    numpy.ndarray
        ``float32`` reflectance (``float64`` for 32/64-bit or ``float64`` input). Not
        clipped: it can be slightly negative over dark targets.

    Examples
    --------
    >>> import numpy as np
    >>> sentinel2_l2a_scale(np.array([0, 1000, 3500], dtype=np.uint16))
    array([ nan, 0.  , 0.25], dtype=float32)
    >>> sentinel2_l2a_scale(np.array([2500], dtype=np.uint16), offset=0)  # baseline < 04.00
    array([0.25], dtype=float32)
    """
    for value, name in ((offset, "offset"), (quantification, "quantification")):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise TypeError(f"{name} must be a finite number, got {value!r}")
    if quantification <= 0:
        raise ValueError(f"quantification must be positive, got {quantification}")
    data, invalid = _scale_input(dn, "dn")
    dtype = _float_dtype(data.dtype)
    out = data.astype(dtype)
    if nodata is not None:
        invalid = (data == nodata) if invalid is None else invalid | (data == nodata)
    out += dtype.type(offset)
    out /= dtype.type(quantification)
    if invalid is not None:
        out[invalid] = np.nan
    return out


# --------------------------------------------------------------------------- #
# HLS
# --------------------------------------------------------------------------- #
def hls_fmask_mask(
    fmask: np.ndarray,
    *,
    cloud: bool = True,
    adjacent: bool = True,
    shadow: bool = True,
    snow: bool = False,
    water: bool = False,
    cirrus: bool = False,
    aerosol: int | str | None = "high",
    fill: bool = True,
) -> np.ndarray:
    """Mask of unusable pixels from the HLS v2.0 ``Fmask`` band (HLSL30 and HLSS30).

    Parameters
    ----------
    fmask : numpy.ndarray
        ``Fmask`` band (``uint8``). NaN / masked entries are masked.
    cloud, adjacent, shadow, snow, water, cirrus : bool
        Which flags to mask: cloud (bit 1), adjacent to cloud/shadow (bit 2), cloud
        shadow (bit 3), snow/ice (bit 4), water (bit 5), cirrus (bit 0; reserved and
        not set in HLS v2.0).
    aerosol : {"low", "moderate", "high"} or int or None, default "high"
        Mask pixels whose aerosol level (bits 6-7: 0 climatology, 1 low, 2 moderate,
        3 high) is at least this level. NASA recommends discarding high-aerosol
        pixels. ``None`` ignores aerosol.
    fill : bool, default True
        Mask the fill value 255.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` = masked.

    Notes
    -----
    Source: NASA LP DAAC, *HLS Product User Guide v2.0*, Table 9 (see
    :class:`HLSFmask`).

    Examples
    --------
    >>> import numpy as np
    >>> fm = np.array([0b01000000, 0b01000010, 0b01001000, 0b11000000], dtype=np.uint8)
    >>> hls_fmask_mask(fm)  # clear, cloud, shadow, high aerosol
    array([False,  True,  True,  True])
    """
    qa, invalid = _as_qa(fmask, "fmask", 8)
    bits = []
    if cirrus:
        bits.append(HLSFmask.CIRRUS)
    if cloud:
        bits.append(HLSFmask.CLOUD)
    if adjacent:
        bits.append(HLSFmask.ADJACENT)
    if shadow:
        bits.append(HLSFmask.CLOUD_SHADOW)
    if snow:
        bits.append(HLSFmask.SNOW_ICE)
    if water:
        bits.append(HLSFmask.WATER)
    mask = (qa & qa.dtype.type(sum(1 << b for b in bits))) != 0
    if aerosol is not None:
        level = _parse_level(aerosol, _HLS_AEROSOL, "aerosol")
        if level == 0:
            raise ValueError("aerosol must be at least 1 ('low'); 0 would mask everything")
        mask |= _field(qa, HLSFmask.AEROSOL_LEVEL, 2) >= level
    if fill:
        mask |= qa == _HLS_FILL
    return _with_nodata(mask, invalid)


# --------------------------------------------------------------------------- #
# Mask operations
# --------------------------------------------------------------------------- #
def _pixel_resolution(pixel_size: Any) -> tuple[float, float]:
    """``(xres, yres)`` in CRS units from a number, pair, Affine or rasterio meta."""
    if pixel_size is None:
        raise ValueError("distance requires pixel_size (a number, (x, y), Affine or meta)")
    if isinstance(pixel_size, Mapping):
        from rasterio.crs import CRS

        from .georef import pixel_size as _meta_pixel_size

        res = _meta_pixel_size(pixel_size)
        crs = pixel_size.get("crs") or pixel_size.get("gcps_crs")
        if crs is not None and CRS.from_user_input(crs).is_geographic:
            raise ValueError(
                "pixel_size metadata uses a geographic CRS (degrees); a distance in metres "
                "cannot be converted to pixels. Reproject first or pass pixels=."
            )
        return res
    if all(hasattr(pixel_size, a) for a in ("a", "b", "d", "e")):  # affine.Affine
        t = pixel_size
        return math.hypot(t.a, t.d), math.hypot(t.b, t.e)
    if isinstance(pixel_size, Real) and not isinstance(pixel_size, bool):
        res = (float(pixel_size), float(pixel_size))
    elif isinstance(pixel_size, (tuple, list, np.ndarray)) and len(pixel_size) == 2:
        res = (abs(float(pixel_size[0])), abs(float(pixel_size[1])))
    else:
        raise TypeError(
            "pixel_size must be a number, an (x, y) pair, an affine.Affine or a rasterio "
            f"meta dict, got {type(pixel_size).__name__}"
        )
    if not all(math.isfinite(r) and r > 0 for r in res):
        raise ValueError(f"pixel_size must be positive and finite, got {pixel_size}")
    return res


def buffer_mask(
    mask: np.ndarray,
    pixels: float | None = None,
    *,
    distance: float | None = None,
    pixel_size: Any = None,
) -> np.ndarray:
    """Grow a mask by a radius (dilation with a disk), to catch cloud edges and shadows.

    Cloud detectors miss the thin, semi-transparent cloud edges and often part of the
    shadow, which then show up as change. Buffering the mask by a few pixels (Landsat
    CFMask already dilates clouds by 3 pixels, see ``dilated`` in
    :func:`landsat_qa_mask`) removes them.

    Give the radius either in pixels (``pixels``) or in ground units (``distance``,
    e.g. metres, together with ``pixel_size``).

    Parameters
    ----------
    mask : numpy.ndarray
        2-D mask (``True``/non-zero = masked; NaN counts as masked).
    pixels : float, optional
        Buffer radius in pixels.
    distance : float, optional
        Buffer radius in CRS units (metres for UTM). Requires ``pixel_size``.
    pixel_size : number, (x, y), affine.Affine or dict, optional
        Pixel size in CRS units, an affine transform, or rasterio metadata with a
        ``transform`` (geographic CRSs are rejected). Non-square pixels give an
        elliptical buffer in pixel space, i.e. a true circle on the ground.

    Returns
    -------
    numpy.ndarray
        New boolean mask: every pixel whose centre lies within the radius of a masked
        pixel centre is masked. Equivalent to a binary dilation with a disk structuring
        element: radius 1 adds the 4 direct neighbours, radius ``sqrt(2)`` the full
        3x3 neighbourhood.

    Raises
    ------
    ValueError
        If neither or both of ``pixels`` and ``distance`` are given, the radius is
        negative, or ``mask`` is not 2-D.

    Notes
    -----
    Implemented with an exact Euclidean distance transform
    (:func:`scipy.ndimage.distance_transform_edt`), processed in row blocks so that
    large scenes need bounded memory and time does not grow with the radius.

    Examples
    --------
    >>> import numpy as np
    >>> m = np.zeros((5, 5), bool)
    >>> m[2, 2] = True
    >>> int(buffer_mask(m, 1).sum()), int(buffer_mask(m, 1.5).sum())
    (5, 9)
    >>> int(buffer_mask(m, distance=60, pixel_size=30).sum())  # 2 pixels
    13
    """
    data = _as_bool_mask(mask, "mask")
    if data.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {data.shape}")
    if (pixels is None) == (distance is None):
        raise ValueError("Give exactly one of pixels= or distance= (with pixel_size=)")
    radius: Any
    if pixels is not None:
        radius = pixels
        name = "pixels"
        xres = yres = 1.0
        if pixel_size is not None:
            raise ValueError("pixel_size is only used with distance=")
    else:
        radius = distance
        name = "distance"
        xres, yres = _pixel_resolution(pixel_size)
    if isinstance(radius, bool) or not isinstance(radius, Real) or not math.isfinite(radius):
        raise TypeError(f"{name} must be a finite number, got {radius!r}")
    if radius < 0:
        raise ValueError(f"{name} must be >= 0, got {radius}")
    radius = float(radius)
    out = data.copy()
    if radius == 0 or not data.any():
        return out

    halo = math.floor(radius / yres + 1e-9)
    if halo == 0 and radius / xres < 1:
        return out  # the radius does not reach any neighbouring pixel centre
    rows, cols = data.shape
    block = max(1, (1 << 22) // cols)
    tol = radius * 1e-9
    for start in range(0, rows, block):
        stop = min(rows, start + block)
        lo, hi = max(0, start - halo), min(rows, stop + halo)
        window = data[lo:hi]
        if not window.any():
            continue
        dist = ndimage.distance_transform_edt(~window, sampling=(yres, xres))
        out[start:stop] |= dist[start - lo : stop - lo] <= radius + tol
    return out


def combine_masks(*masks: np.ndarray | None) -> np.ndarray:
    """Union (logical OR) of several masks; ``None`` entries are skipped.

    Parameters
    ----------
    *masks : numpy.ndarray or None
        Masks of the same shape (bool, 0/1 or float; NaN counts as masked).

    Returns
    -------
    numpy.ndarray
        New boolean mask, ``True`` where any input is masked.

    Raises
    ------
    ValueError
        If no mask is given or the shapes differ.

    Examples
    --------
    >>> import numpy as np
    >>> combine_masks(np.array([True, False, False]), None, np.array([0, 0, 1]))
    array([ True, False,  True])
    """
    present = [_as_bool_mask(m, f"masks[{i}]") for i, m in enumerate(masks) if m is not None]
    if not present:
        raise ValueError("combine_masks needs at least one mask")
    out = present[0].copy()
    for i, m in enumerate(present[1:], start=1):
        if m.shape != out.shape:
            raise ValueError(f"Mask shapes differ: {out.shape} and {m.shape} (mask {i})")
        out |= m
    return out


def apply_mask(data: np.ndarray, mask: np.ndarray, *, fill: float = np.nan) -> np.ndarray:
    """Return a float copy of ``data`` with masked pixels set to ``fill`` (NaN).

    Parameters
    ----------
    data : numpy.ndarray
        Image of shape ``(rows, cols)`` or ``(bands, rows, cols)``. Masked entries of a
        :class:`numpy.ma.MaskedArray` are also set to ``fill``.
    mask : numpy.ndarray
        ``True`` = masked. Either ``(rows, cols)`` (applied to every band) or the same
        shape as ``data`` (per-band mask).
    fill : float, default NaN
        Value written into masked pixels. NaN is farq's nodata value, so masked pixels
        are ignored by every change and statistics function.

    Returns
    -------
    numpy.ndarray
        New array: ``float32`` for integer data of up to 16 bits and ``float32`` input,
        ``float64`` otherwise. ``data`` is never modified.

    Raises
    ------
    TypeError
        If ``data`` is not numeric or ``fill`` is not a number.
    ValueError
        If ``data`` is not 2-D/3-D or the mask shape does not match.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.arange(4, dtype=np.uint16).reshape(2, 2)
    >>> apply_mask(img, np.array([[True, False], [False, False]]))
    array([[nan,  1.],
           [ 2.,  3.]], dtype=float32)
    """
    arr, invalid = _as_array(data, "data")
    if arr.dtype.kind not in "biuf":
        raise TypeError(f"data must have a numeric dtype, got {arr.dtype}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"data must be (rows, cols) or (bands, rows, cols), got shape {arr.shape}")
    if isinstance(fill, bool) or not isinstance(fill, Real):
        raise TypeError(f"fill must be a number, got {type(fill).__name__}")
    m = _as_bool_mask(mask, "mask")
    if m.shape != arr.shape and m.shape != arr.shape[-2:]:
        raise ValueError(
            f"mask shape {m.shape} must be {arr.shape[-2:]} or {arr.shape}; "
            "use sentinel2_scl_mask(..., target_shape=) / upsample_mask for coarser masks"
        )
    out = arr.astype(_float_dtype(arr.dtype))  # always a copy
    if m.shape != arr.shape:
        m = np.broadcast_to(m, arr.shape)
    out[m] = fill
    if invalid is not None:
        out[invalid] = fill
    return out


def clear_fraction(mask: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Fraction of pixels that are not masked.

    Parameters
    ----------
    mask : numpy.ndarray
        ``True`` = masked.
    valid : numpy.ndarray, optional
        Boolean array of the same shape; only these pixels are counted (e.g. the
        image footprint, ``np.isfinite(band)``).

    Returns
    -------
    float
        ``unmasked valid pixels / valid pixels`` in [0, 1], or NaN when no pixel is
        valid.

    Examples
    --------
    >>> import numpy as np
    >>> clear_fraction(np.array([True, False, False, False]))
    0.75
    """
    m = _as_bool_mask(mask, "mask")
    if valid is None:
        total = m.size
        clear = total - int(np.count_nonzero(m))
    else:
        v = _as_bool_mask(valid, "valid")
        if v.shape != m.shape:
            raise ValueError(f"valid shape {v.shape} does not match mask shape {m.shape}")
        total = int(np.count_nonzero(v))
        clear = int(np.count_nonzero(v & ~m))
    return clear / total if total else float("nan")


class MaskOverlap(NamedTuple):
    """Output of :func:`valid_overlap`."""

    valid: np.ndarray  #: bool, ``True`` where both dates are usable
    n_valid: int  #: number of pixels usable in both dates
    n_total: int  #: number of pixels considered
    fraction: float  #: ``n_valid / n_total`` (NaN when ``n_total == 0``)
    before_clear: float  #: clear fraction of the first date
    after_clear: float  #: clear fraction of the second date


def valid_overlap(
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    *,
    footprint: np.ndarray | None = None,
) -> MaskOverlap:
    """Pixels usable in *both* dates of a change pair.

    Change can only be measured where neither date is masked. Check ``fraction``
    before trusting a change summary: a pair with 20 % usable overlap reports change
    for a fifth of the area only.

    Parameters
    ----------
    mask_before, mask_after : numpy.ndarray
        Masks (``True`` = masked) of the two dates on the same grid.
    footprint : numpy.ndarray, optional
        Boolean array; only these pixels are counted (e.g. the area of interest).

    Returns
    -------
    MaskOverlap
        ``valid`` mask (pass it as ``valid=`` to :func:`farq.change_summary` or
        :func:`farq.classify_change`), pixel counts and clear fractions.

    Examples
    --------
    >>> import numpy as np
    >>> ov = valid_overlap(np.array([True, False, False, False]),
    ...                    np.array([False, False, True, False]))
    >>> ov.n_valid, ov.fraction
    (2, 0.5)
    """
    b = _as_bool_mask(mask_before, "mask_before")
    a = _as_bool_mask(mask_after, "mask_after")
    if a.shape != b.shape:
        raise ValueError(
            f"mask_before and mask_after must have the same shape, got {b.shape} and {a.shape}"
        )
    valid = ~(a | b)
    if footprint is not None:
        fp = _as_bool_mask(footprint, "footprint")
        if fp.shape != b.shape:
            raise ValueError(f"footprint shape {fp.shape} does not match mask shape {b.shape}")
        valid &= fp
        n_total = int(np.count_nonzero(fp))
    else:
        n_total = valid.size
    n_valid = int(np.count_nonzero(valid))
    return MaskOverlap(
        valid=valid,
        n_valid=n_valid,
        n_total=n_total,
        fraction=n_valid / n_total if n_total else float("nan"),
        before_clear=clear_fraction(b, footprint),
        after_clear=clear_fraction(a, footprint),
    )
