"""Farq: raster change detection and analysis for satellite and drone imagery.

Every public function is available directly on the package (``farq.read``,
``farq.ndwi``, ``farq.detect_changes``, ...). Submodules are imported lazily on
first use, so ``import farq`` stays fast and heavy dependencies such as
matplotlib and scikit-learn are only loaded when you need them.

Submodules
----------
core
    Reading, writing and resampling rasters.
indices
    Spectral indices (NDWI, NDVI, MNDWI, ...) and RGB-only drone indices.
change
    Change detection: differencing, ratios, CVA, PCA, thresholds, transitions.
georef
    GCP georeferencing, grid alignment and image co-registration.
masking
    Cloud/shadow/snow masks from Landsat QA_PIXEL, Sentinel-2 SCL and HLS Fmask.
radiometry
    Relative radiometric normalization between dates and calibrated IR-MAD.
elevation
    DEM differencing, DEM co-registration, cut/fill and stockpile volumes.
tiling
    Out-of-core, block-wise processing of rasters larger than memory.
vector
    Polygonize change masks and export GeoJSON, GeoPackage, Shapefile, FlatGeobuf.
analysis
    Water-body statistics and shape metrics.
ml
    Feature extraction, classifiers and clustering.
visualization
    Plotting helpers returning matplotlib figures.
utils
    NaN-aware statistics helpers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.3.0"

_SUBMODULES = (
    "analysis",
    "change",
    "core",
    "elevation",
    "georef",
    "indices",
    "masking",
    "ml",
    "radiometry",
    "tiling",
    "utils",
    "vector",
    "visualization",
)

# Public name -> submodule that defines it.
_EXPORTS: dict[str, str] = {}
for _module, _names in {
    "core": ("read", "write", "resample", "validate_bands"),
    "utils": (
        "stats",
        "min",
        "max",
        "mean",
        "std",
        "sum",
        "median",
        "percentile",
        "count_nonzero",
        "unique",
        "validate_array",
    ),
    "indices": (
        "ndwi",
        "mndwi",
        "ndvi",
        "evi",
        "savi",
        "ndbi",
        "nbr",
        "ndmi",
        "vari",
        "exg",
        "exr",
        "exgr",
        "gli",
        "ngrdi",
        "tgi",
        "calculate_indices",
        "calculate_normalized_difference",
    ),
    "change": (
        "detect_changes",
        "ChangeResult",
        "difference",
        "ratio",
        "normalized_difference_change",
        "change_vector_analysis",
        "CVAResult",
        "pca_change",
        "PCAChangeResult",
        "otsu_threshold",
        "compute_threshold",
        "threshold_change",
        "clean_mask",
        "classify_change",
        "transition_matrix",
        "TransitionMatrix",
        "change_summary",
        "NO_CHANGE",
        "GAINED",
        "LOST",
        "STABLE",
        "CHANGE_NODATA",
        "CHANGE_LABELS",
    ),
    "georef": (
        "has_gcps",
        "read_gcps",
        "make_gcps",
        "gcp_residuals",
        "GCPResiduals",
        "georeference",
        "rectify",
        "align",
        "align_pair",
        "coregister",
        "apply_shift",
        "pixel_size",
        "pixel_area",
    ),
    "analysis": (
        "water_stats",
        "water_change",
        "get_water_bodies",
        "calculate_shape_metrics",
    ),
    "ml": (
        "extract_features",
        "train_classifier",
        "predict_raster",
        "save_model",
        "load_model",
        "ModelIntegrityError",
        "detect_changes_ml",
        "augment_training_data",
        "cluster_water_bodies",
        "analyze_water_clusters",
        "optimize_clustering",
    ),
    "elevation": (
        "DEMCoregistration",
        "StockpileResult",
        "VerticalOffset",
        "VolumeResult",
        "aspect",
        "coregister_dem",
        "elevation_change",
        "hillshade",
        "level_of_detection",
        "shift_dem",
        "significant_change",
        "slope",
        "stockpile_volume",
        "vertical_offset",
        "volume_change",
    ),
    "masking": (
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
    ),
    "radiometry": (
        "IRMADResult",
        "NormalizationResult",
        "histogram_match",
        "irmad",
        "irmad_change",
        "linear_normalize",
        "pif_normalize",
    ),
    "tiling": (
        "Block",
        "detect_changes_file",
        "index_file",
        "iter_windows",
        "map_blocks",
        "summarize_file",
    ),
    "vector": (
        "changes_to_vector",
        "polygonize",
        "to_geojson",
        "write_vector",
    ),
    "visualization": (
        "plot",
        "compare",
        "changes",
        "hist",
        "distribution_comparison",
        "plot_rgb",
        "compare_rgb",
    ),
}.items():
    for _name in _names:
        _EXPORTS[_name] = _module
del _module, _names, _name

__all__ = sorted(_EXPORTS) + list(_SUBMODULES)


def _compat(name: str) -> Any:
    """Objects re-exported by farq 0.1 for convenience (``farq.plt`` etc.)."""
    if name == "plt":
        import matplotlib.pyplot as plt

        return plt
    if name == "Resampling":
        from rasterio.enums import Resampling

        return Resampling
    if name == "os":
        import os

        return os
    raise AttributeError(name)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    elif name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        value = getattr(module, name)
    else:
        try:
            value = _compat(name)
        except AttributeError:
            raise AttributeError(f"module 'farq' has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static analysers and IDE completion
    from . import (
        analysis,
        change,
        core,
        elevation,
        georef,
        indices,
        masking,
        ml,
        radiometry,
        tiling,
        utils,
        vector,
        visualization,
    )
    from .analysis import (
        calculate_shape_metrics,
        get_water_bodies,
        water_change,
        water_stats,
    )
    from .change import (
        CHANGE_LABELS,
        CHANGE_NODATA,
        GAINED,
        LOST,
        NO_CHANGE,
        STABLE,
        ChangeResult,
        CVAResult,
        PCAChangeResult,
        TransitionMatrix,
        change_summary,
        change_vector_analysis,
        classify_change,
        clean_mask,
        compute_threshold,
        detect_changes,
        difference,
        normalized_difference_change,
        otsu_threshold,
        pca_change,
        ratio,
        threshold_change,
        transition_matrix,
    )
    from .core import read, resample, validate_bands, write
    from .elevation import (
        DEMCoregistration,
        StockpileResult,
        VerticalOffset,
        VolumeResult,
        aspect,
        coregister_dem,
        elevation_change,
        hillshade,
        level_of_detection,
        shift_dem,
        significant_change,
        slope,
        stockpile_volume,
        vertical_offset,
        volume_change,
    )
    from .georef import (
        GCPResiduals,
        align,
        align_pair,
        apply_shift,
        coregister,
        gcp_residuals,
        georeference,
        has_gcps,
        make_gcps,
        pixel_area,
        pixel_size,
        read_gcps,
        rectify,
    )
    from .indices import (
        calculate_indices,
        calculate_normalized_difference,
        evi,
        exg,
        exgr,
        exr,
        gli,
        mndwi,
        nbr,
        ndbi,
        ndmi,
        ndvi,
        ndwi,
        ngrdi,
        savi,
        tgi,
        vari,
    )
    from .masking import (
        DEFAULT_S2_BAD_CLASSES,
        SCL_NAMES,
        Confidence,
        HLSFmask,
        LandsatQA,
        MaskOverlap,
        SCLClass,
        apply_mask,
        buffer_mask,
        clear_fraction,
        combine_masks,
        decode_bits,
        decode_landsat_qa,
        hls_fmask_mask,
        landsat_c2_scale,
        landsat_qa_mask,
        landsat_radsat_mask,
        sentinel2_cloud_probability_mask,
        sentinel2_l2a_scale,
        sentinel2_scl_mask,
        upsample_mask,
        valid_overlap,
    )
    from .ml import (
        ModelIntegrityError,
        analyze_water_clusters,
        augment_training_data,
        cluster_water_bodies,
        detect_changes_ml,
        extract_features,
        load_model,
        optimize_clustering,
        predict_raster,
        save_model,
        train_classifier,
    )
    from .radiometry import (
        IRMADResult,
        NormalizationResult,
        histogram_match,
        irmad,
        irmad_change,
        linear_normalize,
        pif_normalize,
    )
    from .tiling import (
        Block,
        detect_changes_file,
        index_file,
        iter_windows,
        map_blocks,
        summarize_file,
    )
    from .utils import (
        count_nonzero,
        max,
        mean,
        median,
        min,
        percentile,
        stats,
        std,
        sum,
        unique,
        validate_array,
    )
    from .vector import (
        changes_to_vector,
        polygonize,
        to_geojson,
        write_vector,
    )
    from .visualization import (
        changes,
        compare,
        compare_rgb,
        distribution_comparison,
        hist,
        plot,
        plot_rgb,
    )
