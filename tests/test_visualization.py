"""
Tests for visualization functions.
"""
import numpy as np
import pytest
import matplotlib.pyplot as plt
import farq

@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close figures after each test."""
    yield
    plt.close('all')

def test_plot_basic():
    """Test basic plot functionality."""
    data = np.random.rand(10, 10)
    farq.plot(data, title="Test Plot")
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 1

def test_plot_with_colormap():
    """Test plot with custom colormap."""
    data = np.random.rand(10, 10)
    farq.plot(data, cmap="RdYlBu", vmin=-1, vmax=1)
    fig = plt.gcf()
    assert fig is not None
    img = fig.axes[0].get_images()[0]
    assert img.get_cmap().name == "RdYlBu"

def test_compare_plots():
    """Test comparison plot functionality."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    farq.compare(data1, data2, title1="Plot 1", title2="Plot 2")
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 2

def test_plot_invalid_data():
    """Test plotting with invalid data."""
    with pytest.raises(ValueError):
        farq.plot(np.array([]))

def test_compare_invalid_shapes():
    """Test comparison with mismatched shapes."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 15)
    with pytest.raises(ValueError):
        farq.compare(data1, data2)

def test_plot_with_title():
    """Test plot with custom title."""
    data = np.random.rand(10, 10)
    title = "Custom Title"
    farq.plot(data, title=title)
    fig = plt.gcf()
    assert fig.axes[0].get_title() == title

def test_compare_with_titles():
    """Test comparison plot with custom titles."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    title1 = "First Plot"
    title2 = "Second Plot"
    farq.compare(data1, data2, title1=title1, title2=title2)
    fig = plt.gcf()
    assert fig.axes[0].get_title() == title1
    assert fig.axes[1].get_title() == title2

def test_plot_value_range():
    """Test plot with custom value range."""
    data = np.random.rand(10, 10)
    vmin, vmax = -1, 1
    farq.plot(data, vmin=vmin, vmax=vmax)
    fig = plt.gcf()
    img = fig.axes[0].get_images()[0]
    assert img.get_clim() == (vmin, vmax)

def test_plot_colorbar():
    """Test plot with colorbar."""
    data = np.random.rand(10, 10)
    farq.plot(data)
    fig = plt.gcf()
    assert len(fig.axes) > 1  # Main axis + colorbar axis

def test_compare_colorbars():
    """Test comparison plot with colorbars."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    farq.compare(data1, data2)
    fig = plt.gcf()
    assert len(fig.axes) > 2  # Two main axes + colorbar axes

def test_plot_nan_values():
    """Test plotting with NaN values."""
    data = np.random.rand(10, 10)
    data[0, 0] = np.nan
    farq.plot(data)
    fig = plt.gcf()
    assert fig is not None

def test_plot_large_array():
    """Test plotting large arrays."""
    data = np.random.rand(1000, 1000)
    farq.plot(data)
    fig = plt.gcf()
    assert fig is not None

def test_compare_different_colormaps():
    """Test comparison with different colormaps."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    farq.compare(data1, data2, cmap="RdYlBu")
    fig = plt.gcf()
    assert fig.axes[0].get_images()[0].get_cmap().name == "RdYlBu"
    assert fig.axes[1].get_images()[0].get_cmap().name == "RdYlBu" 