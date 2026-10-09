# Testing and development

## Setup

```bash
git clone https://github.com/ferasqr/farq.git
cd farq
pip install -e ".[dev]"     # farq + pytest, pytest-cov, psutil, ruff, mypy, build, twine
```

The `test` extra (`pip install -e ".[test]"`) installs only the test dependencies. All
tool configuration lives in `pyproject.toml`.

## Running the tests

```bash
python -m pytest                          # whole suite; performance tests are skipped
python -m pytest tests/test_change.py     # one module
python -m pytest -k "align_pair"          # tests matching a name
python -m pytest --cov=farq --cov-report=term-missing   # with coverage
```

The tests create their own small synthetic rasters (see the fixtures in
`tests/conftest.py`, such as a georeferenced multi-band GeoTIFF and a GCP-only drone
image), so no external data is needed. There is one test module per package module
(`test_core.py`, `test_indices.py`, `test_change.py`, `test_georef.py`,
`test_analysis.py`, `test_ml.py`, `test_visualization.py`, `test_utils.py`). There are
also `test_integration.py` for end-to-end workflows (read, align, index, detect, summarize,
write), `test_package.py` for the lazy top-level namespace, and `test_performance.py`.

pytest runs with `--strict-markers`, and any `DeprecationWarning` raised from farq is an
error.

## Performance tests

Heavy speed and memory tests are marked `@pytest.mark.performance` and are **skipped by
default**. To run them, use either form:

```bash
python -m pytest -m performance tests/test_performance.py
FARQ_RUN_PERFORMANCE=1 python -m pytest tests/test_performance.py
```

To exclude them explicitly (as CI does):

```bash
python -m pytest -m "not performance"
```

Memory checks use `psutil`. If it is not installed, those tests are skipped. A few quick
scaling smoke tests in `test_performance.py` always run. They check that the vectorized
code paths do not regress to per-object Python loops.

## Linting, formatting and type checking

```bash
ruff check farq tests            # lint (rules: E, F, W, I, B, UP, SIM, RUF, NPY)
ruff format --check farq tests   # formatting; drop --check to apply
ruff check --fix farq tests      # auto-fix what can be fixed
mypy farq                        # optional static type check
```

The line length is 100, and the target version is Python 3.9.

## Building the package

```bash
python -m build                  # sdist and wheel in dist/
twine check --strict dist/*      # validate metadata and the README rendering for PyPI
```

`README.md` is the PyPI project description, so use absolute URLs for its links and
images.

## Checking the documentation examples

The code blocks in `README.md` and `docs/*.md` are written to run as they are against
real GeoTIFFs. When you change an example, run it against small synthetic rasters with
the same file names, and use the `Agg` matplotlib backend (`MPLBACKEND=Agg`) so that no
windows open.

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs on pushes to `main` and on pull
requests:

- **lint**: `ruff check` and `ruff format --check`
- **test**: `pytest -m "not performance"` with coverage on Python 3.9-3.13 (Ubuntu), plus
  Python 3.12 on macOS and Windows
- **build**: `python -m build` and `twine check --strict`

Publishing a GitHub release triggers `.github/workflows/publish.yml`, which builds the
package and uploads it to PyPI.

## Contributing tests

- Add tests next to the module that you change, and cover the edge cases: NaN and nodata,
  integer dtypes (`uint8`, `uint16`), empty or mismatched shapes, and invalid parameters.
- Use the deterministic `rng` fixture (`numpy.random.default_rng(42)`) instead of global
  random state.
- Mark anything slower than a second or so with `@pytest.mark.performance`.
- Plotting tests should close their figures (`plt.close(fig)`). The library itself never
  closes figures.
