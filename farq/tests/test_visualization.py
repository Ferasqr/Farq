import numpy as np
import pytest
from farq import plot, compare, changes, hist
import matplotlib.pyplot as plt

@pytest.fixture(autouse=True)
def close_figures():
    plt.close('all')
    yield
    plt.close('all')

def get_main_axes(fig):
    """Helper function to get only the main plot axes (excluding colorbars)"""
    return [ax for ax in fig.axes if len(ax.get_images()) > 0]

def test_plot():
    data = np.array([[1, 2], [3, 4]])
    fig = plot(data, title="Test Plot")
    assert isinstance(fig, plt.Figure)
    # Count only main plot axes
    main_axes = get_main_axes(fig)
    assert len(main_axes) == 1
    
    # Test with custom parameters
    fig = plot(data, cmap="RdYlBu", vmin=-1, vmax=1)
    main_axes = get_main_axes(fig)
    assert main_axes[0].get_images()[0].get_cmap().name == "RdYlBu"

def test_compare():
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[4, 3], [2, 1]])
    
    fig = compare(data1, data2, title1="First", title2="Second")
    assert isinstance(fig, plt.Figure)
    # Count only main plot axes
    main_axes = get_main_axes(fig)
    assert len(main_axes) == 2
    
    # Test with custom parameters
    fig = compare(data1, data2, cmap="Blues", vmin=0, vmax=5)
    main_axes = get_main_axes(fig)
    assert all(ax.get_images()[0].get_cmap().name == "Blues" for ax in main_axes)

def test_changes():
    data = np.array([[-1, 0], [0, 1]])
    fig = changes(data)
    assert isinstance(fig, plt.Figure)
    
    # Test symmetric scaling
    fig = changes(data)
    main_axes = get_main_axes(fig)
    img = main_axes[0].get_images()[0]
    assert img.get_clim()[0] == -1
    assert img.get_clim()[1] == 1

def test_hist():
    data = np.random.normal(0, 1, 1000)
    fig = hist(data, bins=50)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes[0].patches) == 50  # Should have 50 bars

def test_edge_cases():
    # Empty array
    data = np.array([[]])  # 2D empty array
    with pytest.raises(ValueError):
        plot(data)
    
    # NaN data
    data = np.array([[1, np.nan], [3, 4]])
    fig = plot(data)
    assert isinstance(fig, plt.Figure)
    
    # Different sized arrays for compare
    with pytest.raises(ValueError):
        compare(np.zeros((2, 2)), np.zeros((3, 3)))