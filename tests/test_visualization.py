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
    fig = farq.plot(data, title="Test Plot")
    
    # Count only non-colorbar axes
    plot_axes = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()]
    assert len(plot_axes) == 1
    assert plot_axes[0].get_title() == "Test Plot"

def test_plot_with_colormap():
    """Test plot with custom colormap."""
    data = np.random.rand(10, 10)
    fig = farq.plot(data, cmap="RdYlBu", vmin=-1, vmax=1)
    
    # Count only non-colorbar axes
    plot_axes = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()]
    assert len(plot_axes) == 1
    assert plot_axes[0].images[0].get_cmap().name == "RdYlBu"

def test_compare_plots():
    """Test comparison plot functionality."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    fig = farq.compare(data1, data2, title1="Plot 1", title2="Plot 2")
    
    # Count only non-colorbar axes
    plot_axes = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()]
    assert len(plot_axes) == 2
    assert plot_axes[0].get_title() == "Plot 1"
    assert plot_axes[1].get_title() == "Plot 2"

def test_plot_invalid_data():
    """Test plot with invalid data."""
    with pytest.raises(TypeError):
        farq.plot([1, 2, 3])  # Not a numpy array
    with pytest.raises(ValueError):
        farq.plot(np.array([]))  # Empty array
    with pytest.raises(ValueError):
        farq.plot(np.array([1, 2, 3]))  # 1D array

def test_compare_invalid_shapes():
    """Test comparison with invalid shapes."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 11)
    with pytest.raises(ValueError):
        farq.compare(data1, data2)

def test_plot_with_title():
    """Test plot with title."""
    data = np.random.rand(10, 10)
    fig = farq.plot(data, title="Test Title")
    
    # Get main plot axis (not colorbar)
    plot_ax = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()][0]
    assert plot_ax.get_title() == "Test Title"

def test_compare_with_titles():
    """Test comparison with titles."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    fig = farq.compare(data1, data2, title1="First", title2="Second")
    
    # Get main plot axes (not colorbars)
    plot_axes = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()]
    assert plot_axes[0].get_title() == "First"
    assert plot_axes[1].get_title() == "Second"

def test_plot_value_range():
    """Test plot value range."""
    data = np.array([[0, 1], [2, 3]])
    fig = farq.plot(data, vmin=0, vmax=3)
    
    # Get main plot axis (not colorbar)
    plot_ax = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()][0]
    assert plot_ax.images[0].norm.vmin == 0
    assert plot_ax.images[0].norm.vmax == 3

def test_plot_colorbar():
    """Test plot colorbar."""
    data = np.random.rand(10, 10)
    fig = farq.plot(data, colorbar_label="Values")
    
    # Get colorbar axis
    cbar_ax = [ax for ax in fig.axes if 'colorbar' in ax.get_label()][0]
    assert cbar_ax.get_ylabel() == "Values"

def test_compare_colorbars():
    """Test comparison colorbars."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    fig = farq.compare(data1, data2, colorbar_label="Values")
    
    # Get colorbar axes
    cbar_axes = [ax for ax in fig.axes if 'colorbar' in ax.get_label()]
    assert len(cbar_axes) == 2
    assert all(ax.get_ylabel() == "Values" for ax in cbar_axes)

def test_plot_nan_values():
    """Test plot with NaN values."""
    data = np.array([[1, np.nan], [3, 4]])
    fig = farq.plot(data)
    
    # Get main plot axis (not colorbar)
    plot_ax = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()][0]
    assert plot_ax is not None

def test_plot_large_array():
    """Test plot with large array."""
    data = np.random.rand(1000, 1000)
    fig = farq.plot(data)
    
    # Get main plot axis (not colorbar)
    plot_ax = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()][0]
    assert plot_ax is not None

def test_compare_different_colormaps():
    """Test comparison with different colormaps."""
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    fig = farq.compare(data1, data2, cmap="viridis")
    
    # Get main plot axes (not colorbars)
    plot_axes = [ax for ax in fig.axes if 'colorbar' not in ax.get_label()]
    assert len(plot_axes) == 2
    assert all(ax.images[0].get_cmap().name == "viridis" for ax in plot_axes)
  