"""Performance and memory tests for farq.

Heavy tests are marked ``@pytest.mark.performance`` and are skipped by default to
keep the regular test run fast. Run them with either::

    python -m pytest -m performance tests/test_performance.py
    FARQ_RUN_PERFORMANCE=1 python -m pytest tests/test_performance.py

The quick smoke tests at the bottom always run; they check that the vectorised
code paths scale (no per-object Python loops) on small inputs.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import ndimage

import farq
from farq.analysis import get_water_bodies, water_change, water_stats
from farq.ml import analyze_water_clusters, cluster_water_bodies, extract_features


@pytest.fixture(autouse=True)
def _performance_opt_in(request: pytest.FixtureRequest) -> None:
    """Skip ``performance``-marked tests unless explicitly requested."""
    if request.node.get_closest_marker("performance") is None:
        return
    selected = "performance" in (request.config.getoption("-m") or "")
    if not selected and os.environ.get("FARQ_RUN_PERFORMANCE") != "1":
        pytest.skip("performance test: use -m performance or FARQ_RUN_PERFORMANCE=1")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _timed(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def _rss_mb() -> float:
    psutil = pytest.importorskip("psutil")
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _blobs(size: int, seed: int = 0) -> np.ndarray:
    """Water mask with thousands of irregular water bodies."""
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.random((size, size)), 4) > 0.52


# --------------------------------------------------------------------------- #
# Heavy tests (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.performance
@pytest.mark.parametrize(("size", "limit"), [(100, 1.0), (1000, 3.0), (5000, 15.0)])
def test_ndwi_and_plot(size: int, limit: float) -> None:
    rng = np.random.default_rng(0)
    green, nir = rng.random((size, size)), rng.random((size, size))
    ndwi, calc_time = _timed(farq.ndwi, green, nir)
    _, plot_time = _timed(farq.plot, ndwi)
    assert calc_time < limit, f"NDWI took {calc_time:.2f}s"
    assert plot_time < limit, f"plot took {plot_time:.2f}s"


@pytest.mark.performance
def test_resample_performance() -> None:
    data = np.random.default_rng(0).random((1000, 1000))
    resampled, elapsed = _timed(farq.resample, data, (500, 500))
    assert resampled.shape == (500, 500)
    assert elapsed < 5.0, f"resample took {elapsed:.2f}s"


@pytest.mark.performance
def test_statistical_operations() -> None:
    data = np.random.default_rng(0).random((5000, 5000))
    _, mean_time = _timed(farq.mean, data)
    _, std_time = _timed(farq.std, data)
    assert mean_time < 3.0, f"mean took {mean_time:.2f}s"
    assert std_time < 3.0, f"std took {std_time:.2f}s"


@pytest.mark.performance
def test_memory_efficiency() -> None:
    rng = np.random.default_rng(0)
    green, nir = rng.random((2000, 2000)), rng.random((2000, 2000))
    start = _rss_mb()
    ndwi = farq.ndwi(green, nir)
    after_ndwi = _rss_mb()
    farq.plot(ndwi)
    after_plot = _rss_mb()
    assert after_ndwi - start < 1000, f"NDWI used {after_ndwi - start:.1f}MB"
    assert after_plot - after_ndwi < 1000, f"plot used {after_plot - after_ndwi:.1f}MB"


@pytest.mark.performance
def test_visualization_memory() -> None:
    data = np.random.default_rng(0).random((1000, 1000))
    start = _rss_mb()
    farq.plot(data)
    after_single = _rss_mb()
    farq.compare(data, data)
    after_compare = _rss_mb()
    assert after_single - start < 1000
    assert after_compare - after_single < 1000


@pytest.mark.performance
def test_water_analysis_large_raster() -> None:
    """2000x2000 mask with ~5000 bodies: per-body loops used to take minutes."""
    mask = _blobs(2000)
    stats, t_stats = _timed(water_stats, mask, calculate_shapes=True)
    assert stats["num_water_bodies"] > 1000
    assert t_stats < 10.0, f"water_stats(calculate_shapes=True) took {t_stats:.2f}s"

    (_, bodies), t_bodies = _timed(get_water_bodies, mask, calculate_shapes=True)
    assert len(bodies) == stats["num_water_bodies"]
    assert t_bodies < 10.0, f"get_water_bodies took {t_bodies:.2f}s"

    _, t_change = _timed(water_change, mask, np.roll(mask, 3, axis=0), min_change_area=5000)
    assert t_change < 5.0, f"water_change took {t_change:.2f}s"

    _, t_clusters = _timed(analyze_water_clusters, mask.astype(np.int8), 1)
    assert t_clusters < 10.0, f"analyze_water_clusters took {t_clusters:.2f}s"


@pytest.mark.performance
def test_ml_large_raster() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((2000, 2000, 4)).astype(np.float32)
    features, t_feat = _timed(extract_features, image)
    assert features.shape == (2000, 2000, 12)
    assert t_feat < 15.0, f"extract_features took {t_feat:.2f}s"

    small = image[:1000, :1000]
    (labels, meta), t_cluster = _timed(cluster_water_bodies, small, sample_size=20_000)
    assert labels.shape == (1000, 1000)
    assert meta["water_cluster"] is not None
    assert t_cluster < 30.0, f"cluster_water_bodies took {t_cluster:.2f}s"


# --------------------------------------------------------------------------- #
# Quick smoke tests (always run)
# --------------------------------------------------------------------------- #
def test_shape_metrics_cost_independent_of_body_count() -> None:
    """Many tiny bodies must not be much slower than one big body (vectorised)."""
    few = np.zeros((300, 300), bool)
    few[50:250, 50:250] = True
    many = np.zeros((300, 300), bool)
    many[::2, ::2] = True  # 22,500 single-pixel bodies

    water_stats(few, calculate_shapes=True)  # warm-up
    _, t_few = _timed(water_stats, few, calculate_shapes=True)
    stats, t_many = _timed(water_stats, many, calculate_shapes=True)
    assert stats["num_water_bodies"] == 150 * 150
    # A per-body loop would be >1000x slower; allow generous slack for noisy CI.
    assert t_many < max(50 * t_few, 1.0), f"{t_many:.3f}s vs {t_few:.3f}s"


def test_extract_features_smoke() -> None:
    image = np.random.default_rng(0).random((200, 200, 3))
    _, elapsed = _timed(extract_features, image, window_size=5)
    assert elapsed < 2.0
