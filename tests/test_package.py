"""Tests for the top-level package namespace."""

import subprocess
import sys

import pytest

import farq


@pytest.mark.parametrize("name", farq.__all__)
def test_every_export_resolves(name):
    assert getattr(farq, name) is not None


def test_exports_match_submodule_objects():
    assert farq.detect_changes is farq.change.detect_changes
    assert farq.read is farq.core.read
    assert farq.georeference is farq.georef.georeference


def test_import_is_lazy():
    code = (
        "import sys, farq; "
        "heavy = {'matplotlib.pyplot', 'sklearn', 'rasterio'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_legacy_conveniences():
    from rasterio.enums import Resampling

    assert farq.Resampling is Resampling
    assert hasattr(farq.plt, "figure")


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        farq.nope  # noqa: B018


def test_version():
    assert farq.__version__ == "0.3.0"
