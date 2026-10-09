"""Shared pytest fixtures for the Farq test suite."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def geo_metadata() -> dict:
    """Georeferenced single-band GTiff metadata (UTM 33N, 30 m pixels, 6x5 raster)."""
    return {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0,
        "width": 5,
        "height": 6,
        "count": 1,
        "crs": CRS.from_epsg(32633),
        "transform": from_origin(500000.0, 4000000.0, 30.0, 30.0),
    }


@pytest.fixture
def geotiff_path(tmp_path, geo_metadata):
    """Write a 3-band uint16 GeoTIFF with nodata=0 pixels; return (path, data)."""
    data = np.arange(1, 3 * 6 * 5 + 1, dtype=np.uint16).reshape(3, 6, 5)
    data[:, 0, 0] = 0  # nodata pixel in every band
    meta = {**geo_metadata, "count": 3}
    path = tmp_path / "multiband.tif"
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)
    return path, data


@pytest.fixture
def gcp_tiff_path(tmp_path):
    """Write a 3-band uint8 drone-style image georeferenced only by 4 GCPs.

    Returns (path, data, gcps, gcps_crs).
    """
    import warnings

    from rasterio.control import GroundControlPoint
    from rasterio.errors import NotGeoreferencedWarning

    data = np.arange(3 * 8 * 10, dtype=np.uint8).reshape(3, 8, 10)
    gcps = [
        GroundControlPoint(row=0, col=0, x=500000.0, y=4000000.0, z=0.0, id="1"),
        GroundControlPoint(row=0, col=10, x=500010.0, y=4000000.0, z=0.0, id="2"),
        GroundControlPoint(row=8, col=0, x=500000.0, y=3999992.0, z=0.0, id="3"),
        GroundControlPoint(row=8, col=10, x=500010.0, y=3999992.0, z=0.0, id="4"),
    ]
    gcps_crs = CRS.from_epsg(32633)
    path = tmp_path / "drone_gcps.tif"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path, "w", driver="GTiff", height=8, width=10, count=3, dtype="uint8"
        ) as dst:
            dst.gcps = (gcps, gcps_crs)
            dst.write(data)
    return path, data, gcps, gcps_crs
