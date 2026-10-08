"""Tests for farq.visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import farq
from farq import visualization as viz

RNG = np.random.default_rng(42)


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure after each test so nothing leaks between tests."""
    yield
    plt.close("all")


def image_axes(fig):
    """Return the non-colorbar axes of a figure."""
    return [ax for ax in fig.axes if "colorbar" not in ax.get_label()]


def colorbar_axes(fig):
    return [ax for ax in fig.axes if "colorbar" in ax.get_label()]


def rgb_bands(shape=(20, 30), scale=1.0):
    return tuple(RNG.random(shape) * scale for _ in range(3))


# ------------------------------------------------------------------------------ plot


def test_plot_basic():
    data = RNG.random((10, 10))
    fig = farq.plot(data, title="Test Plot")
    assert isinstance(fig, Figure)
    axes = image_axes(fig)
    assert len(axes) == 1
    assert axes[0].get_title() == "Test Plot"
    assert len(colorbar_axes(fig)) == 1


def test_plot_with_colormap_and_limits():
    data = np.array([[0, 1], [2, 3]])
    fig = farq.plot(data, cmap="RdYlBu", vmin=0, vmax=3)
    im = image_axes(fig)[0].images[0]
    assert im.get_cmap().name == "RdYlBu"
    assert im.norm.vmin == 0
    assert im.norm.vmax == 3


def test_plot_colorbar_label_and_disable():
    fig = farq.plot(RNG.random((10, 10)), colorbar_label="Values")
    assert colorbar_axes(fig)[0].get_ylabel() == "Values"
    fig = farq.plot(RNG.random((10, 10)), colorbar=False)
    assert colorbar_axes(fig) == []


def test_plot_does_not_close_other_figures():
    other = plt.figure()
    farq.plot(RNG.random((5, 5)))
    assert plt.fignum_exists(other.number)


def test_plot_does_not_show(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("plt.show must not be called")

    monkeypatch.setattr(plt, "show", fail)
    farq.plot(RNG.random((5, 5)))
    farq.compare(RNG.random((5, 5)), RNG.random((5, 5)))
    farq.changes(RNG.random((5, 5)))
    farq.hist(RNG.random(50))
    farq.distribution_comparison(RNG.random(50), RNG.random(50))
    farq.plot_rgb(*rgb_bands())
    farq.compare_rgb(rgb_bands(), rgb_bands())


def test_plot_into_existing_ax():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    result = farq.plot(RNG.random((10, 10)), title="Left", ax=ax1)
    assert result is fig
    assert ax1.get_title() == "Left"
    assert len(ax1.images) == 1
    assert len(ax2.images) == 0
    # Only one extra (colorbar) axes was added; no new figure was created.
    assert len(fig.axes) == 3
    assert plt.get_fignums() == [fig.number]


def test_plot_reflectance_scale_does_not_modify_input():
    data = np.full((4, 4), 5000.0)
    original = data.copy()
    fig = farq.plot(data, reflectance_scale=10000)
    np.testing.assert_allclose(image_axes(fig)[0].images[0].get_array(), 0.5)
    np.testing.assert_array_equal(data, original)


def test_plot_nan_values():
    data = np.array([[1.0, np.nan], [3.0, 4.0]])
    fig = farq.plot(data)
    im = image_axes(fig)[0].images[0]
    assert im.norm.vmin == 1
    assert im.norm.vmax == 4
    fig.canvas.draw()


def test_plot_bool_and_int_inputs():
    farq.plot(RNG.random((5, 5)) > 0.5).canvas.draw()
    farq.plot(RNG.integers(0, 10000, (5, 5), dtype=np.uint16)).canvas.draw()


def test_plot_large_array():
    fig = farq.plot(RNG.random((1000, 1000)))
    assert len(image_axes(fig)) == 1


@pytest.mark.parametrize(
    ("data", "exc"),
    [
        ([1, 2, 3], TypeError),
        (np.array([]), ValueError),
        (np.array([1, 2, 3]), ValueError),
        (np.ones((2, 2, 2)), ValueError),
        (np.full((3, 3), np.nan), ValueError),
    ],
)
def test_plot_invalid_data(data, exc):
    with pytest.raises(exc):
        farq.plot(data)


def test_plot_zero_reflectance_scale():
    with pytest.raises(ValueError, match="reflectance_scale"):
        farq.plot(np.ones((3, 3)), reflectance_scale=0)


# --------------------------------------------------------------------------- compare


def test_compare_plots():
    fig = farq.compare(RNG.random((10, 10)), RNG.random((10, 10)), title1="Plot 1", title2="Plot 2")
    assert isinstance(fig, Figure)
    axes = image_axes(fig)
    assert [ax.get_title() for ax in axes] == ["Plot 1", "Plot 2"]
    assert all(ax.images[0].get_cmap().name == "viridis" for ax in axes)


def test_compare_colorbars():
    fig = farq.compare(RNG.random((10, 10)), RNG.random((10, 10)), colorbar_label="Values")
    cbars = colorbar_axes(fig)
    assert len(cbars) == 2
    assert all(ax.get_ylabel() == "Values" for ax in cbars)


def test_compare_shared_scale_covers_both_arrays():
    data1 = np.array([[0.0, 1.0], [np.nan, 2.0]])
    data2 = np.array([[5.0, -3.0], [np.inf, 1.0]])
    fig = farq.compare(data1, data2)
    for ax in image_axes(fig):
        assert ax.images[0].norm.vmin == -3
        assert ax.images[0].norm.vmax == 5


def test_compare_explicit_limits():
    fig = farq.compare(RNG.random((4, 4)), RNG.random((4, 4)), vmin=-1, vmax=1)
    for ax in image_axes(fig):
        assert (ax.images[0].norm.vmin, ax.images[0].norm.vmax) == (-1, 1)


def test_compare_into_existing_axes():
    fig, axes = plt.subplots(2, 2)
    result = farq.compare(RNG.random((4, 4)), RNG.random((4, 4)), axes=axes[1], colorbar=False)
    assert result is fig
    assert len(axes[1][0].images) == 1
    assert len(axes[1][1].images) == 1
    assert len(axes[0][0].images) == 0
    assert len(fig.axes) == 4


def test_compare_wrong_number_of_axes():
    _, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="exactly 2"):
        farq.compare(RNG.random((4, 4)), RNG.random((4, 4)), axes=axes)


def test_compare_invalid_shapes():
    with pytest.raises(ValueError, match=r"same shape.*\(10, 10\).*\(10, 11\)"):
        farq.compare(RNG.random((10, 10)), RNG.random((10, 11)))


def test_compare_invalid_types():
    with pytest.raises(TypeError, match="data2"):
        farq.compare(RNG.random((3, 3)), [[1, 2], [3, 4]])


# --------------------------------------------------------------------------- changes


def test_changes_symmetric_default():
    data = np.array([[-1.0, 0.5], [3.0, np.nan]])
    fig = farq.changes(data, title="Change")
    ax = image_axes(fig)[0]
    assert ax.get_title() == "Change"
    assert (ax.images[0].norm.vmin, ax.images[0].norm.vmax) == (-3, 3)
    assert colorbar_axes(fig)[0].get_ylabel() == "Change"


def test_changes_symmetric_ignores_inf():
    data = np.array([[-2.0, np.inf], [1.0, -np.inf]])
    norm = image_axes(farq.changes(data))[0].images[0].norm
    assert (norm.vmin, norm.vmax) == (-2, 2)


def test_changes_symmetric_mirrors_single_limit():
    data = RNG.random((4, 4))
    norm = image_axes(farq.changes(data, vmax=0.5))[0].images[0].norm
    assert (norm.vmin, norm.vmax) == (-0.5, 0.5)
    norm = image_axes(farq.changes(data, vmin=-2))[0].images[0].norm
    assert (norm.vmin, norm.vmax) == (-2, 2)


def test_changes_all_zero():
    norm = image_axes(farq.changes(np.zeros((3, 3))))[0].images[0].norm
    assert (norm.vmin, norm.vmax) == (-1, 1)


def test_changes_not_symmetric():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm = image_axes(farq.changes(data, symmetric=False))[0].images[0].norm
    assert (norm.vmin, norm.vmax) == (1, 4)


def test_changes_bool_and_int():
    farq.changes(RNG.random((5, 5)) > 0.5).canvas.draw()
    data = np.array([[-3, 1], [2, 0]], dtype=np.int8)
    norm = image_axes(farq.changes(data))[0].images[0].norm
    assert (norm.vmin, norm.vmax) == (-3, 3)


def test_changes_into_existing_ax_uses_that_figure():
    fig, ax = plt.subplots()
    other = plt.figure()  # becomes pyplot's "current" figure
    result = farq.changes(RNG.random((4, 4)) - 0.5, ax=ax)
    assert result is fig
    assert len(fig.axes) == 2  # image + its colorbar
    assert other.axes == []


def test_changes_invalid():
    with pytest.raises(TypeError):
        farq.changes([[1, 2]])
    with pytest.raises(ValueError):
        farq.changes(np.ones(4))


# ------------------------------------------------------------------------------ hist


def test_hist_basic():
    fig = farq.hist(RNG.random(1000), bins=20, title="H")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "H"
    assert len(ax.patches) == 20
    assert ax.get_ylabel() == "Density"
    assert ax.get_xlabel() == "Value"


def test_hist_counts_and_labels():
    fig = farq.hist([1, 2, 2, 3], bins=3, density=False, xlabel="NDWI")
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Count"
    assert ax.get_xlabel() == "NDWI"
    assert sum(p.get_height() for p in ax.patches) == 4


def test_hist_ignores_nan_and_inf():
    data = np.array([[1.0, np.nan], [np.inf, 2.0], [-np.inf, 3.0]])
    ax = farq.hist(data, bins=3, density=False).axes[0]
    assert sum(p.get_height() for p in ax.patches) == 3


def test_hist_all_nan_raises():
    with pytest.raises(ValueError, match="no finite values"):
        farq.hist(np.full(10, np.nan))


def test_hist_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        farq.hist([])


def test_hist_bool_and_masked():
    farq.hist(RNG.random((10, 10)) > 0.5)
    masked = np.ma.masked_array([1.0, 2.0, 1e9], mask=[False, False, True])
    ax = farq.hist(masked, bins=2, density=False).axes[0]
    assert sum(p.get_height() for p in ax.patches) == 2
    assert ax.get_xlim()[1] < 10


def test_hist_reflectance_scale():
    ax = farq.hist(np.array([5000.0, 10000.0]), bins=2, reflectance_scale=10000).axes[0]
    assert ax.patches[0].get_x() == pytest.approx(0.5)


def test_hist_subsamples_large_input_and_scales_counts():
    data = RNG.random(200_000)
    data[::4] = np.nan  # 25% no-data
    ax = farq.hist(data, bins=10, density=False, max_samples=10_000).axes[0]
    total = sum(p.get_height() for p in ax.patches)
    assert total == pytest.approx(150_000, rel=0.05)


def test_hist_subsampling_is_reproducible():
    data = RNG.random(50_000)
    h1 = [p.get_height() for p in farq.hist(data, max_samples=1000).axes[0].patches]
    h2 = [p.get_height() for p in farq.hist(data, max_samples=1000).axes[0].patches]
    assert h1 == h2


def test_hist_into_existing_ax():
    fig, ax = plt.subplots()
    assert farq.hist(RNG.random(100), ax=ax) is fig
    assert len(ax.patches) == 50


# --------------------------------------------------------------- distribution_comparison


def test_distribution_comparison_basic():
    fig = farq.distribution_comparison(
        RNG.random(100), RNG.random((10, 10)) + 1, title1="A", title2="B", bins=10
    )
    assert isinstance(fig, Figure)
    ax1, ax2 = fig.axes
    assert (ax1.get_title(), ax2.get_title()) == ("A", "B")


def test_distribution_comparison_shared_bins():
    fig = farq.distribution_comparison(np.array([0.0, 1.0]), np.array([10.0, 11.0]), bins=11)
    ax1, ax2 = fig.axes
    left_edges = [p.get_x() for p in ax1.patches]
    right_edges = [p.get_x() for p in ax2.patches]
    assert left_edges == right_edges
    assert left_edges[0] == 0
    assert left_edges[-1] + ax1.patches[-1].get_width() == pytest.approx(11)


def test_distribution_comparison_nan_and_lists():
    data1 = [1.0, np.nan, 2.0]
    data2 = np.array([[np.nan, 3.0], [4.0, np.inf]])
    fig = farq.distribution_comparison(data1, data2, bins=4, density=False)
    for ax in fig.axes:
        assert sum(p.get_height() for p in ax.patches) == 2
        assert ax.get_ylabel() == "Count"


def test_distribution_comparison_invalid():
    with pytest.raises(ValueError, match="data2"):
        farq.distribution_comparison([1.0, 2.0], [])
    with pytest.raises(ValueError, match="data1"):
        farq.distribution_comparison([np.nan], [1.0])


def test_distribution_comparison_into_existing_axes():
    fig, axes = plt.subplots(1, 2)
    assert farq.distribution_comparison(RNG.random(10), RNG.random(10), axes=axes) is fig
    assert all(len(ax.patches) == 50 for ax in axes)


# -------------------------------------------------------------------------- plot_rgb


def test_plot_rgb_basic():
    fig = farq.plot_rgb(*rgb_bands(), title="RGB")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "RGB"
    img = ax.images[0].get_array()
    assert img.shape == (20, 30, 4)
    assert img.min() >= 0
    assert img.max() <= 1


def test_plot_rgb_does_not_modify_input():
    bands = rgb_bands(scale=10000)
    copies = [b.copy() for b in bands]
    farq.plot_rgb(*bands, reflectance_scale=10000, gamma=2.0, scale_factor=1.5)
    for band, original in zip(bands, copies):
        np.testing.assert_array_equal(band, original)


def test_plot_rgb_nan_pixels_are_transparent():
    red, green, blue = rgb_bands((10, 10))
    red[0, 0] = np.nan
    blue[5, 5] = np.inf
    img = farq.plot_rgb(red, green, blue).axes[0].images[0].get_array()
    assert np.isfinite(img).all()
    assert img[0, 0, 3] == 0
    assert img[5, 5, 3] == 0
    assert img[1, 1, 3] == 1
    # The NaN pixel must not poison the stretch of the rest of the band.
    assert img[..., 0].max() == 1


def test_plot_rgb_percentile_stretch():
    band = np.arange(1, 101, dtype=float).reshape(10, 10)
    img = farq.plot_rgb(band, band, band, percentile=50).axes[0].images[0].get_array()
    assert (img[..., :3] == 1).sum() >= 3 * 50
    np.testing.assert_allclose(img[0, 0, :3], 1 / np.percentile(band, 50), rtol=1e-5)


def test_plot_rgb_scale_factor_changes_brightness():
    bands = rgb_bands()
    dim = farq.plot_rgb(*bands, scale_factor=0.5).axes[0].images[0].get_array()
    full = farq.plot_rgb(*bands, scale_factor=1.0).axes[0].images[0].get_array()
    unclipped = full[..., :3] < 1
    np.testing.assert_allclose(dim[..., :3][unclipped], full[..., :3][unclipped] * 0.5, rtol=1e-5)
    assert dim[..., :3].mean() < full[..., :3].mean()


def test_plot_rgb_gamma():
    band = np.full((4, 4), 0.25)
    img = farq.plot_rgb(band, band, band, percentile=None, gamma=2.0).axes[0].images[0].get_array()
    np.testing.assert_allclose(img[..., :3], 0.5)


def test_plot_rgb_zero_band_does_not_produce_nan():
    zeros = np.zeros((5, 5))
    img = farq.plot_rgb(zeros, zeros, RNG.random((5, 5))).axes[0].images[0].get_array()
    assert np.isfinite(img).all()


def test_plot_rgb_integer_bands():
    bands = [RNG.integers(0, 30000, (8, 8), dtype=np.uint16) for _ in range(3)]
    farq.plot_rgb(*bands).canvas.draw()


def test_plot_rgb_into_existing_ax():
    fig, ax = plt.subplots()
    assert farq.plot_rgb(*rgb_bands(), ax=ax) is fig
    assert len(ax.images) == 1


def test_plot_rgb_mismatched_shapes():
    red, green, _ = rgb_bands((5, 5))
    with pytest.raises(ValueError, match="same shape"):
        farq.plot_rgb(red, green, RNG.random((5, 6)))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gamma": 0}, "gamma"),
        ({"percentile": 0}, "percentile"),
        ({"percentile": 101}, "percentile"),
    ],
)
def test_plot_rgb_invalid_params(kwargs, match):
    with pytest.raises(ValueError, match=match):
        farq.plot_rgb(*rgb_bands(), **kwargs)


def test_plot_rgb_invalid_types():
    red, green, _ = rgb_bands()
    with pytest.raises(TypeError):
        farq.plot_rgb(red, green, [[1, 2]])
    with pytest.raises(ValueError):
        farq.plot_rgb(red, green, np.array([]))


# ----------------------------------------------------------------------- compare_rgb


def test_compare_rgb_basic():
    fig = farq.compare_rgb(rgb_bands(), rgb_bands(), title1="Before", title2="After")
    assert isinstance(fig, Figure)
    assert [ax.get_title() for ax in fig.axes] == ["Before", "After"]
    for ax in fig.axes:
        assert ax.images[0].get_array().shape == (20, 30, 4)


def test_compare_rgb_matches_plot_rgb():
    bands = rgb_bands()
    single = farq.plot_rgb(*bands, gamma=1.5).axes[0].images[0].get_array()
    pair = farq.compare_rgb(bands, bands, gamma=1.5)
    for ax in pair.axes:
        np.testing.assert_allclose(ax.images[0].get_array(), single)


def test_compare_rgb_into_existing_axes():
    fig, axes = plt.subplots(1, 2)
    assert farq.compare_rgb(rgb_bands(), rgb_bands(), axes=axes) is fig
    assert all(len(ax.images) == 1 for ax in axes)


def test_compare_rgb_mismatched_images():
    with pytest.raises(ValueError, match="rgb1 and rgb2"):
        farq.compare_rgb(rgb_bands((5, 5)), rgb_bands((6, 5)))


def test_compare_rgb_wrong_band_count():
    red, green, _ = rgb_bands()
    with pytest.raises(ValueError, match="three"):
        farq.compare_rgb((red, green), rgb_bands())


def test_compare_rgb_invalid_types():
    red, green, _ = rgb_bands()
    with pytest.raises(TypeError):
        farq.compare_rgb((red, green, "blue"), rgb_bands())


# ------------------------------------------------------------------------ internals


def test_finite_sample_limits_size():
    values, weight = viz._finite_sample(np.arange(1000.0), max_samples=100)
    assert values.size == 100
    assert weight == 10
    values, weight = viz._finite_sample(np.arange(10.0), max_samples=100)
    assert values.size == 10
    assert weight == 1


def test_rgb_accepts_band_stack():
    rng = np.random.default_rng(1)
    stack = rng.integers(0, 255, (3, 20, 30), dtype=np.uint8)
    fig = farq.plot_rgb(stack)
    assert isinstance(fig, Figure)
    fig2 = farq.compare_rgb(stack, tuple(stack))
    assert len(fig2.axes) == 2


@pytest.mark.parametrize(
    "args",
    [
        (np.zeros((20, 30)),),
        (np.zeros((4, 20, 30)),),
        (np.zeros((20, 30)), np.zeros((20, 30))),
    ],
)
def test_rgb_stack_errors(args):
    with pytest.raises(ValueError):
        farq.plot_rgb(*args)
