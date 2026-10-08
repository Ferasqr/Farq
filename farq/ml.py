"""Machine-learning helpers for raster analysis.

This module provides:

- per-pixel feature extraction (bands, extra index layers, windowed mean/variance)
- supervised classification (random forest) training, prediction and persistence
- change detection between two co-registered rasters
- training-data augmentation
- unsupervised water detection with k-means / DBSCAN and analysis of the result

Conventions
-----------
* Multi-band rasters are ``(rows, cols, bands)`` ("bands last").
* Non-finite values (``NaN``/``inf``) mark invalid pixels. They are excluded
  from training and clustering, and receive a fill value in predictions.
* Every stochastic function takes ``random_state`` (an ``int``, a
  ``numpy.random.Generator`` where noted, or ``None`` for non-deterministic
  results). Results are reproducible for a fixed integer seed.

Security
--------
:func:`save_model` and :func:`load_model` use :mod:`joblib`, i.e. Python
pickle. **Loading a pickle can execute arbitrary code.** Only load model files
from sources you trust. :func:`save_model` writes a SHA-256 sidecar file and
:func:`load_model` verifies it (and/or a caller-pinned hash) *before*
unpickling, which protects against corrupted or swapped files but not against
an attacker who can replace both files - pin ``expected_sha256`` for that.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import pickle
import platform
import re
import stat
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata as _importlib_metadata
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.ndimage import uniform_filter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .analysis import PixelSize, _label, _object_metrics, _parse_pixel_size

__all__ = [
    "ModelIntegrityError",
    "analyze_water_clusters",
    "augment_training_data",
    "cluster_water_bodies",
    "detect_changes_ml",
    "extract_features",
    "load_model",
    "optimize_clustering",
    "predict_raster",
    "save_model",
    "train_classifier",
]

MODEL_FORMAT_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIDECAR_MAX_BYTES = 4096


class ModelIntegrityError(ValueError):
    """Raised when a model file does not match its expected SHA-256 hash."""


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _check_array(array: Any, name: str, ndims: tuple[int, ...]) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if array.ndim not in ndims:
        expected = " or ".join(f"{d}-D" for d in ndims)
        raise ValueError(f"{name} must be {expected}, got shape {array.shape}")
    if not (array.dtype == bool or np.issubdtype(array.dtype, np.number)):
        raise TypeError(f"{name} must have a numeric dtype, got {array.dtype}")
    return array


def _check_positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _float_dtype(*dtypes: np.dtype) -> np.dtype:
    """Smallest float dtype that holds all inputs (at least float32)."""
    return np.result_type(*dtypes, np.float32)


def _finite_rows(X: np.ndarray) -> np.ndarray | None:
    """Boolean mask of rows with only finite values, or None if all are finite."""
    if not np.issubdtype(X.dtype, np.floating):
        return None
    valid = np.isfinite(X).all(axis=1)
    return None if valid.all() else valid


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _window_mean_var(x: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware moving-window mean and (population) variance of a 2-D array.

    Computed in float64 on mean-centred data to avoid catastrophic cancellation
    in ``E[x^2] - E[x]^2``. Windows without any finite value yield NaN.
    """
    x = x.astype(np.float64, copy=False)
    finite = np.isfinite(x)
    all_finite = bool(finite.all())
    if not finite.any():
        nan = np.full(x.shape, np.nan)
        return nan, nan.copy()
    offset = float(np.mean(x[finite])) if not all_finite else float(x.mean())
    if all_finite:
        xc = x - offset
        mean = uniform_filter(xc, size=size, mode="reflect")
        sq = uniform_filter(xc * xc, size=size, mode="reflect")
        var = np.maximum(sq - mean * mean, 0.0)
        mean += offset
        return mean, var

    xc = np.where(finite, x - offset, 0.0)
    count = uniform_filter(finite.astype(np.float64), size=size, mode="reflect")
    total = uniform_filter(xc, size=size, mode="reflect")
    sq = uniform_filter(xc * xc, size=size, mode="reflect")
    empty = count < 0.5 / (size * size)
    count[empty] = 1.0
    mean = total / count
    var = np.maximum(sq / count - mean * mean, 0.0)
    mean += offset
    mean[empty] = np.nan
    var[empty] = np.nan
    return mean, var


def extract_features(
    raster_data: np.ndarray,
    indices: Mapping[str, np.ndarray] | Sequence[np.ndarray] | None = None,
    window_size: int = 3,
) -> np.ndarray:
    """Build a per-pixel feature stack for ML.

    Feature order (last axis):

    1. the raster bands,
    2. the extra ``indices`` layers (in the given order),
    3. if ``window_size > 1``, for each feature above: its moving-window mean
       followed by its moving-window variance.

    Windowed statistics are NaN-aware: invalid pixels are ignored inside each
    window (a window with no valid pixel yields NaN). Edges use mirror
    ("reflect") padding.

    Args:
        raster_data: ``(rows, cols)`` or ``(rows, cols, bands)`` array.
        indices: Optional extra 2-D layers with the raster's spatial shape, e.g.
            ``{"ndwi": farq.ndwi(green, nir)}`` or a list of arrays. Index names
            as strings are not accepted: compute the index first.
        window_size: Side length of the square window for texture features;
            ``1`` disables them.

    Returns:
        ``(rows, cols, n_features)`` float array (float32 for float32/small
        integer inputs, otherwise float64).

    Raises:
        TypeError: If inputs have the wrong type.
        ValueError: If shapes do not match or ``window_size`` < 1.
    """
    _check_array(raster_data, "raster_data", (2, 3))
    window_size = _check_positive_int(window_size, "window_size")
    rows, cols = raster_data.shape[:2]

    extra: list[np.ndarray] = []
    if indices is not None:
        layers = list(indices.values()) if isinstance(indices, Mapping) else list(indices)
        for i, layer in enumerate(layers):
            if isinstance(layer, str):
                raise TypeError(
                    f"indices must contain precomputed 2-D arrays, got the string {layer!r}; "
                    "compute the index first, e.g. indices={'ndwi': farq.ndwi(green, nir)}"
                )
            _check_array(layer, f"indices[{i}]", (2,))
            if layer.shape != (rows, cols):
                raise ValueError(f"indices[{i}] has shape {layer.shape}, expected {(rows, cols)}")
            extra.append(layer)

    n_bands = 1 if raster_data.ndim == 2 else raster_data.shape[2]
    n_base = n_bands + len(extra)
    n_features = n_base * (3 if window_size > 1 else 1)
    dtype = _float_dtype(raster_data.dtype, *(e.dtype for e in extra))
    out = np.empty((rows, cols, n_features), dtype=dtype)

    if raster_data.ndim == 2:
        out[..., 0] = raster_data
    else:
        out[..., :n_bands] = raster_data
    for i, layer in enumerate(extra):
        out[..., n_bands + i] = layer

    if window_size > 1:
        for k in range(n_base):
            if k < n_bands:
                base = raster_data if raster_data.ndim == 2 else raster_data[..., k]
            else:
                base = extra[k - n_bands]
            mean, var = _window_mean_var(base, window_size)
            out[..., n_base + 2 * k] = mean
            out[..., n_base + 2 * k + 1] = var
    return out


# --------------------------------------------------------------------------- #
# Supervised classification
# --------------------------------------------------------------------------- #
def _as_samples(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten raster-shaped features/labels to ``(n_samples, n_features)``/``(n_samples,)``."""
    _check_array(features, "features", (1, 2, 3))
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"labels must be a numpy array, got {type(labels).__name__}")
    if features.ndim == 3:
        if labels.shape != features.shape[:2]:
            raise ValueError(
                f"labels shape {labels.shape} does not match the spatial shape of "
                f"features {features.shape[:2]}"
            )
        return features.reshape(-1, features.shape[-1]), labels.reshape(-1)
    X = features.reshape(-1, 1) if features.ndim == 1 else features
    y = labels.reshape(-1) if labels.ndim == 2 and labels.shape[1] == 1 else labels
    if y.ndim != 1:
        raise ValueError(f"labels must be 1-D for 2-D features, got shape {labels.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Number of samples in features ({X.shape[0]}) and labels ({y.shape[0]}) must match"
        )
    return X, y


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    model_type: str = "rf",
    test_size: float = 0.2,
    random_state: int | None = 42,
    *,
    ignore_label: Any = None,
    stratify: bool = True,
    **model_params: Any,
) -> tuple[Any, dict[str, Any]]:
    """Train a pixel classifier.

    Samples with any non-finite feature, a ``NaN`` label or ``ignore_label`` are
    dropped before splitting. The hold-out split is stratified by class when
    every class has at least two samples.

    Note:
        A random pixel-level split of an image is spatially autocorrelated, so
        the hold-out accuracy is optimistic. For an honest estimate evaluate on
        a separate area/scene, and augment (:func:`augment_training_data`) only
        the training data, never before splitting.

    Args:
        features: ``(n_samples, n_features)``, ``(n_samples,)`` or raster-shaped
            ``(rows, cols, n_features)`` array (then ``labels`` is ``(rows, cols)``).
        labels: Class labels matching ``features``.
        model_type: ``"rf"`` (random forest) - the only supported type.
        test_size: Fraction of samples held out for evaluation, in ``[0, 1)``.
            ``0`` trains on everything and leaves the evaluation metrics ``None``.
        random_state: Seed for the split and the model.
        ignore_label: Label value marking unlabelled pixels (e.g. ``0`` or ``-1``).
        stratify: Stratify the split by class.
        **model_params: Passed to the estimator, e.g. ``n_estimators=200``.

    Returns:
        ``(model, metrics)``; ``metrics`` has ``accuracy``, ``confusion_matrix``
        (list of lists, rows/cols ordered as ``classes``),
        ``classification_report`` (str), ``classes``, ``n_train``, ``n_test`` and
        ``n_dropped``.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: On shape mismatch, unsupported ``model_type``, invalid
            ``test_size`` or fewer than two classes.
    """
    X, y = _as_samples(features, labels)
    if not isinstance(model_type, str) or model_type.lower() != "rf":
        raise ValueError(f"Unsupported model type: {model_type!r} (supported: 'rf')")
    if not 0.0 <= float(test_size) < 1.0:
        raise ValueError(f"test_size must be in [0, 1), got {test_size}")

    keep = _finite_rows(X)
    if np.issubdtype(y.dtype, np.floating):
        nan_y = np.isnan(y)
        if nan_y.any():
            keep = ~nan_y if keep is None else keep & ~nan_y
    if ignore_label is not None:
        labelled = y != ignore_label
        keep = labelled if keep is None else keep & labelled
    n_dropped = 0
    if keep is not None:
        n_dropped = int(keep.size - np.count_nonzero(keep))
        X, y = X[keep], y[keep]

    classes, class_counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError(
            f"Need at least two classes to train a classifier, got {classes.tolist()} "
            f"({n_dropped} samples were dropped as invalid/ignored)"
        )

    model = RandomForestClassifier(random_state=random_state, **model_params)

    if test_size > 0:
        strat = y if stratify and class_counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )
    else:
        X_train, y_train, X_test, y_test = X, y, None, None

    model.fit(X_train, y_train)

    metrics: dict[str, Any] = {
        "accuracy": None,
        "confusion_matrix": None,
        "classification_report": None,
        "classes": model.classes_.tolist(),
        "n_train": int(X_train.shape[0]),
        "n_test": 0 if X_test is None else int(X_test.shape[0]),
        "n_dropped": n_dropped,
    }
    if X_test is not None and X_test.shape[0] > 0:
        y_pred = model.predict(X_test)
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["confusion_matrix"] = confusion_matrix(
            y_test, y_pred, labels=model.classes_
        ).tolist()
        metrics["classification_report"] = classification_report(
            y_test, y_pred, labels=model.classes_, zero_division=0
        )
    return model, metrics


def _predict_rows(func: Any, X: np.ndarray, batch_size: int | None) -> np.ndarray:
    if batch_size is None or batch_size >= X.shape[0]:
        return np.asarray(func(X))
    parts = [np.asarray(func(X[i : i + batch_size])) for i in range(0, X.shape[0], batch_size)]
    return np.concatenate(parts)


def _apply_model(
    func: Any,
    features: np.ndarray,
    n_features_in: int | None,
    batch_size: int | None,
    fill_value: Any,
) -> np.ndarray:
    """Apply ``func`` row-wise to finite rows of a 2-D/3-D feature array."""
    _check_array(features, "features", (1, 2, 3))
    if batch_size is not None:
        batch_size = _check_positive_int(batch_size, "batch_size")
    spatial = features.shape[:-1] if features.ndim == 3 else None
    X = features.reshape(-1, 1) if features.ndim == 1 else features.reshape(-1, features.shape[-1])
    if n_features_in is not None and X.shape[1] != n_features_in:
        raise ValueError(f"Model expects {n_features_in} features, got {X.shape[1]}")

    valid = _finite_rows(X)
    if valid is None:
        pred = _predict_rows(func, X, batch_size)
    else:
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            pred = np.full(X.shape[0], fill_value)
        else:
            part = _predict_rows(func, X[valid], batch_size)
            try:
                dtype = np.result_type(part.dtype, np.asarray(fill_value).dtype)
            except TypeError:
                dtype = np.dtype(object)
            pred = np.full((X.shape[0], *part.shape[1:]), fill_value, dtype=dtype)
            pred[valid] = part
    if spatial is not None:
        pred = pred.reshape(*spatial, *pred.shape[1:])
    return pred


def predict_raster(
    model: Any,
    features: np.ndarray,
    batch_size: int | None = None,
    *,
    fill_value: Any = -1,
) -> np.ndarray:
    """Apply a trained model to a feature array.

    Args:
        model: Fitted estimator with a ``predict`` method.
        features: ``(rows, cols, n_features)`` raster features (e.g. from
            :func:`extract_features`) or ``(n_samples, n_features)``.
        batch_size: Predict in chunks of this many pixels to bound memory.
        fill_value: Output for pixels with any non-finite feature (those pixels
            are not passed to the model).

    Returns:
        Predictions with shape ``(rows, cols)`` for raster input, otherwise
        ``(n_samples,)``.

    Raises:
        TypeError: If ``model`` has no ``predict`` method or features is not an array.
        ValueError: If the number of features does not match the model.
    """
    if not hasattr(model, "predict"):
        raise TypeError("model must have a predict method")
    return _apply_model(
        model.predict, features, getattr(model, "n_features_in_", None), batch_size, fill_value
    )


# --------------------------------------------------------------------------- #
# Model persistence
# --------------------------------------------------------------------------- #
def _package_version(name: str) -> str:
    try:
        return _importlib_metadata.version(name)
    except _importlib_metadata.PackageNotFoundError:
        return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _normalize_sha256(value: str, source: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{source} must be a hex string, got {type(value).__name__}")
    value = value.strip().lower()
    if not _SHA256_RE.match(value):
        raise ValueError(f"{source} is not a valid SHA-256 hex digest: {value!r}")
    return value


def save_model(
    model: Any,
    filepath: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
    *,
    compress: int | bool = 0,
) -> str:
    """Save a trained model with metadata and an integrity hash.

    Writes ``filepath`` (a joblib pickle of ``{"model", "metadata",
    "farq_info"}``) atomically, plus ``<filepath>.sha256`` containing its
    SHA-256 in ``sha256sum`` format. ``farq_info`` records the format version,
    farq/scikit-learn/numpy/Python versions, model class and save time.

    Warning:
        The file is a Python pickle. Anyone who loads it executes code it
        contains, so only share/load model files through trusted channels. Keep
        the returned hash to verify the file later with
        ``load_model(path, expected_sha256=...)``.

    Args:
        model: Fitted model (must have ``predict``).
        filepath: Destination path; parent directories are created.
        metadata: JSON-like user metadata stored with the model
            (e.g. feature names, band order, training scene).
        compress: joblib compression level (0-9 or bool).

    Returns:
        The SHA-256 hex digest of the written file.

    Raises:
        TypeError: If ``model`` has no ``predict`` or ``metadata`` is not a mapping.
        ValueError: If ``filepath`` is an existing directory.
    """
    if not hasattr(model, "predict"):
        raise TypeError("model must have a predict method")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")
    path = Path(filepath)
    if path.is_dir():
        raise ValueError(f"filepath is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "metadata": dict(metadata or {}),
        "farq_info": {
            "format_version": MODEL_FORMAT_VERSION,
            "farq_version": _package_version("farq"),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        joblib.dump(payload, tmp, compress=compress)
        digest = _sha256_file(tmp)
        sidecar_tmp = tmp.with_suffix(".sha256tmp")
        sidecar_tmp.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
        os.replace(tmp, path)
        os.replace(sidecar_tmp, _sidecar_path(path))
    finally:
        for leftover in (tmp, tmp.with_suffix(".sha256tmp")):
            if leftover.exists():
                leftover.unlink()
    return digest


def load_model(
    filepath: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a model saved with :func:`save_model`, verifying its integrity first.

    The file's SHA-256 is checked *before* unpickling against
    ``expected_sha256`` (if given) and the ``<filepath>.sha256`` sidecar (if
    present). Files without any integrity data (e.g. saved by farq < 0.2) still
    load but emit a :class:`UserWarning`. Plain joblib dumps of an estimator are
    accepted too (with empty metadata).

    Warning:
        Loading unpickles the file, which can execute arbitrary code. A sidecar
        hash only detects accidental corruption or a swapped model file; to
        defend against a malicious file pass a hash you obtained through a
        trusted channel as ``expected_sha256``. Never load untrusted files.

    Args:
        filepath: Path to the model file.
        expected_sha256: Known-good SHA-256 hex digest to pin.

    Returns:
        ``(model, metadata)``. ``metadata`` is the user metadata plus a
        ``"_farq"`` entry with the saving environment info (when available).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a regular file, or the content is not a
            valid farq model file.
        ModelIntegrityError: If the hash does not match (a ``ValueError`` subclass).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Model path is not a regular file: {path}")

    # Read the file exactly once and unpickle the very bytes that were hashed, so the
    # file cannot be swapped between the integrity check and loading (TOCTOU).
    with open(path, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            raise ValueError(f"Model path is not a regular file: {path}")
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    if expected_sha256 is not None:
        expected = _normalize_sha256(expected_sha256, "expected_sha256")
        if not hmac.compare_digest(digest, expected):
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {digest}. "
                "The file may be corrupted or tampered with; it was not loaded."
            )
    sidecar = _sidecar_path(path)
    if sidecar.is_file():
        with open(sidecar, "rb") as fh:
            # A sha256sum line is ~70 bytes; never read an arbitrarily large file.
            raw = fh.read(_SIDECAR_MAX_BYTES + 1)
        if len(raw) > _SIDECAR_MAX_BYTES:
            raise ModelIntegrityError(f"Sidecar {sidecar} is too large to be a SHA-256 file")
        try:
            content = raw.decode("utf-8").split()
        except UnicodeDecodeError as exc:
            raise ModelIntegrityError(f"Sidecar {sidecar} is not a text file") from exc
        recorded = _normalize_sha256(content[0] if content else "", f"Sidecar {sidecar}")
        if not hmac.compare_digest(digest, recorded):
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {path}: sidecar {sidecar.name} records {recorded}, "
                f"file is {digest}. The file may be corrupted or tampered with; it was not loaded."
            )
    elif expected_sha256 is None:
        warnings.warn(
            f"No integrity data for {path} (no {sidecar.name} sidecar and no "
            "expected_sha256); loading it unverified. Only load model files you trust.",
            UserWarning,
            stacklevel=2,
        )

    try:
        obj = joblib.load(io.BytesIO(blob))
    except (EOFError, pickle.UnpicklingError, ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Could not load model from {path}: {exc}") from exc
    finally:
        del blob

    info: dict[str, Any] | None = None
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        meta = dict(obj.get("metadata") or {})
        info = obj.get("farq_info")
    elif hasattr(obj, "predict"):
        model, meta = obj, {}
    else:
        raise ValueError(f"{path} does not contain a farq model (got {type(obj).__name__})")
    if not hasattr(model, "predict"):
        raise ValueError(f"Object stored in {path} has no predict method")

    if info:
        meta["_farq"] = dict(info)
        saved_sklearn = info.get("sklearn_version")
        if saved_sklearn and saved_sklearn != sklearn.__version__:
            warnings.warn(
                f"Model was saved with scikit-learn {saved_sklearn} but "
                f"{sklearn.__version__} is installed; predictions may differ.",
                UserWarning,
                stacklevel=2,
            )
    return model, meta


# --------------------------------------------------------------------------- #
# Change detection and augmentation
# --------------------------------------------------------------------------- #
def detect_changes_ml(
    raster1: np.ndarray,
    raster2: np.ndarray,
    model: Any | None = None,
    threshold: float = 0.5,
    *,
    window_size: int = 3,
    batch_size: int | None = None,
) -> np.ndarray:
    """Detect changes between two co-registered rasters.

    The absolute difference ``|raster2 - raster1|`` is computed in floating point
    (so unsigned integer inputs cannot wrap around).

    * Without ``model``: a pixel changed if the difference magnitude exceeds
      ``threshold``. For multi-band rasters the magnitude is the Euclidean norm
      across bands (change-vector analysis).
    * With ``model``: features are extracted from the difference image with
      :func:`extract_features` (``window_size``) - the model must have been
      trained on the same features. For a binary probabilistic classifier the
      probability of the second class (``model.classes_[1]``) is compared with
      ``threshold``; otherwise the predicted label is.

    Pixels with non-finite values are reported as unchanged.

    Args:
        raster1: Earlier raster, ``(rows, cols)`` or ``(rows, cols, bands)``.
        raster2: Later raster with the same shape.
        model: Optional fitted classifier.
        threshold: Decision threshold.
        window_size: Texture window for model features.
        batch_size: Prediction chunk size for the model.

    Returns:
        Boolean ``(rows, cols)`` change mask.

    Raises:
        TypeError: If inputs are not numeric numpy arrays.
        ValueError: If shapes differ.
    """
    _check_array(raster1, "raster1", (2, 3))
    _check_array(raster2, "raster2", (2, 3))
    if raster1.shape != raster2.shape:
        raise ValueError(
            f"Rasters must have the same shape, got {raster1.shape} and {raster2.shape}"
        )
    dtype = _float_dtype(raster1.dtype, raster2.dtype)
    diff = raster2.astype(dtype, copy=False) - raster1.astype(dtype, copy=False)
    np.abs(diff, out=diff)

    if model is None:
        magnitude = diff if diff.ndim == 2 else np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        with np.errstate(invalid="ignore"):
            return np.asarray(magnitude > threshold)

    if not hasattr(model, "predict"):
        raise TypeError("model must have a predict method")
    features = extract_features(diff, window_size=window_size)
    n_in = getattr(model, "n_features_in_", None)
    classes = getattr(model, "classes_", None)
    if hasattr(model, "predict_proba") and classes is not None and len(classes) == 2:
        score = _apply_model(
            lambda X: model.predict_proba(X)[:, 1], features, n_in, batch_size, np.nan
        )
    else:
        score = _apply_model(model.predict, features, n_in, batch_size, np.nan)
    with np.errstate(invalid="ignore"):
        return np.asarray(score.astype(np.float64) > threshold)


def augment_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    augmentation_factor: int = 2,
    random_state: int | np.random.Generator | None = 42,
    *,
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment tabular training samples with Gaussian jitter.

    Returns the original samples followed by ``augmentation_factor - 1`` noisy
    copies. The noise standard deviation of each feature is ``noise_level``
    times that feature's standard deviation, so the augmentation is scale
    invariant. Labels are repeated accordingly. Use on the training split only.

    Args:
        features: ``(n_samples, n_features)`` (or ``(n_samples,)``) array.
        labels: ``(n_samples,)`` labels.
        augmentation_factor: Total size multiplier (1 = no augmentation).
        random_state: Seed or ``numpy.random.Generator``; uses a local generator
            (the global NumPy random state is not touched).
        noise_level: Relative noise standard deviation (>= 0).

    Returns:
        ``(augmented_features, augmented_labels)`` with
        ``n_samples * augmentation_factor`` rows.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If sample counts differ or parameters are invalid.
    """
    _check_array(features, "features", (1, 2))
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"labels must be a numpy array, got {type(labels).__name__}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"features ({features.shape[0]}) and labels ({labels.shape[0]}) must have the "
            "same number of samples"
        )
    factor = _check_positive_int(augmentation_factor, "augmentation_factor")
    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError(f"noise_level must be >= 0, got {noise_level}")

    rng = np.random.default_rng(random_state)
    dtype = _float_dtype(features.dtype)
    n = features.shape[0]
    out = np.empty((n * factor, *features.shape[1:]), dtype=dtype)
    out[:n] = features
    if factor > 1:
        finite = np.where(np.isfinite(features), features, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN feature columns
            scale = np.nan_to_num(np.nanstd(finite, axis=0)) * noise_level
        for k in range(1, factor):
            block = out[k * n : (k + 1) * n]
            block[...] = rng.standard_normal(features.shape)
            block *= scale
            block += features
    return out, np.concatenate([labels] * factor)


# --------------------------------------------------------------------------- #
# Unsupervised water detection
# --------------------------------------------------------------------------- #
def _cluster_features(
    raster_data: np.ndarray, water_index: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Return ``(X, valid_rows, n_bands)`` for clustering."""
    _check_array(raster_data, "raster_data", (2, 3))
    spatial = raster_data.shape[:2]
    n_bands = 1 if raster_data.ndim == 2 else raster_data.shape[2]
    dtype = _float_dtype(raster_data.dtype)
    X = raster_data.reshape(-1, n_bands).astype(dtype, copy=False)
    if water_index is not None:
        _check_array(water_index, "water_index", (2,))
        if water_index.shape != spatial:
            raise ValueError(
                f"water_index shape {water_index.shape} does not match raster shape {spatial}"
            )
        X = np.column_stack([X, water_index.reshape(-1).astype(dtype, copy=False)])
    return X, _finite_rows(X), n_bands


def cluster_water_bodies(
    raster_data: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 2,
    water_index: np.ndarray | None = None,
    *,
    random_state: int | None = 42,
    water_high: bool | None = None,
    sample_size: int | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster pixels and identify the cluster that represents water.

    Features are the raster bands (plus ``water_index`` if given), standardised.
    Pixels with non-finite values get label ``-1`` (as do DBSCAN noise points).

    The water cluster is the cluster whose mean *score* is highest (or lowest):

    * with ``water_index``: score = the index, and water is **high** by default
      (true for NDWI/MNDWI, where water > 0);
    * without it: score = mean of the raster bands. For a 2-D input (usually an
      index such as NDWI) water is assumed **high**; for multi-band reflectance
      water is assumed **low** (water is dark).

    Pass ``water_high`` to override the direction.

    Args:
        raster_data: ``(rows, cols)`` or ``(rows, cols, bands)`` array.
        method: ``"kmeans"`` or ``"dbscan"``.
        n_clusters: Number of k-means clusters.
        water_index: Optional 2-D water index (e.g. NDWI) used as an extra
            feature and to pick the water cluster.
        random_state: Seed for k-means initialisation and subsampling.
        water_high: Whether water has the highest (True) or lowest (False) score.
        sample_size: k-means only: fit on a random subset of this many valid
            pixels, then assign all pixels (much faster on large rasters).
        **kwargs: Passed to :class:`~sklearn.cluster.KMeans` or
            :class:`~sklearn.cluster.DBSCAN` (k-means ``n_init`` defaults to 10).
            Note DBSCAN memory grows quickly with pixel count.

    Returns:
        ``(labels, metadata)``: ``int32`` labels with the raster's spatial shape
        and a dict with ``water_cluster`` (int, or ``None`` if DBSCAN found no
        cluster), ``cluster_means`` (score per cluster id), ``n_clusters``,
        ``n_invalid`` and ``water_rule``; k-means adds ``cluster_centers``
        (standardised space) and ``inertia``; DBSCAN adds ``noise_points``.

    Raises:
        TypeError: If inputs have the wrong type.
        ValueError: On invalid method/parameters, shape mismatch or no valid pixels.
    """
    method_l = method.lower() if isinstance(method, str) else method
    if method_l not in ("kmeans", "dbscan"):
        raise ValueError(f"Unsupported clustering method: {method!r} (use 'kmeans' or 'dbscan')")
    X, valid, n_bands = _cluster_features(raster_data, water_index)
    spatial = raster_data.shape[:2]
    Xv = X if valid is None else X[valid]
    n_valid = Xv.shape[0]
    if n_valid == 0:
        raise ValueError("raster_data has no valid (finite) pixels to cluster")
    Xs = StandardScaler().fit_transform(Xv)

    metadata: dict[str, Any] = {}
    if method_l == "kmeans":
        n_clusters = _check_positive_int(n_clusters, "n_clusters")
        if n_clusters > n_valid:
            raise ValueError(
                f"n_clusters ({n_clusters}) exceeds the number of valid pixels ({n_valid})"
            )
        kwargs.setdefault("n_init", 10)
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
        if sample_size is not None and _check_positive_int(sample_size, "sample_size") < n_valid:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n_valid, size=max(int(sample_size), n_clusters), replace=False)
            clusterer.fit(Xs[idx])
            labels = clusterer.predict(Xs)
        else:
            labels = clusterer.fit_predict(Xs)
        cluster_ids = np.arange(n_clusters)
        metadata["cluster_centers"] = clusterer.cluster_centers_
        metadata["inertia"] = float(clusterer.inertia_)
    else:
        if sample_size is not None:
            raise ValueError("sample_size is only supported for method='kmeans'")
        labels = DBSCAN(**kwargs).fit_predict(Xs)
        cluster_ids = np.unique(labels[labels >= 0])
        metadata["noise_points"] = int(np.count_nonzero(labels == -1))

    # Score each cluster to decide which one is water.
    if water_index is not None:
        score = Xv[:, -1]
        high = True if water_high is None else bool(water_high)
        source = "water_index"
    else:
        score = Xv[:, :n_bands].mean(axis=1)
        high = (raster_data.ndim == 2) if water_high is None else bool(water_high)
        source = "band mean"
    in_cluster = labels >= 0
    size = int(cluster_ids.max()) + 1 if cluster_ids.size else 0
    counts = np.bincount(labels[in_cluster], minlength=size)
    sums = np.bincount(labels[in_cluster], weights=score[in_cluster], minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    cluster_means = [float(means[c]) for c in cluster_ids]
    if cluster_ids.size:
        pick = np.nanargmax if high else np.nanargmin
        water_cluster: int | None = int(cluster_ids[pick(cluster_means)])
    else:
        water_cluster = None

    metadata.update(
        {
            "water_cluster": water_cluster,
            "cluster_means": cluster_means,
            "n_clusters": int(cluster_ids.size),
            "n_invalid": int(X.shape[0] - n_valid),
            "water_rule": f"{'highest' if high else 'lowest'} mean {source}",
        }
    )

    full = np.full(X.shape[0], -1, dtype=np.int32)
    if valid is None:
        full[:] = labels
    else:
        full[valid] = labels
    return full.reshape(spatial), metadata


def analyze_water_clusters(
    cluster_labels: np.ndarray,
    water_cluster: int,
    pixel_size: PixelSize = 30.0,
    *,
    connectivity: int = 1,
) -> dict[str, Any]:
    """Statistics of the connected water bodies in a cluster label image.

    Note that, unlike :mod:`farq.analysis`, areas here are in **square metres**
    and perimeters in **metres** (kept for backward compatibility). Perimeter
    and compactness follow the definitions in :mod:`farq.analysis`.

    Args:
        cluster_labels: 2-D label image (e.g. from :func:`cluster_water_bodies`).
        water_cluster: Label value that represents water.
        pixel_size: Pixel size in metres, a number or ``(width, height)``.
        connectivity: 1 for 4-neighbour, 2 for 8-neighbour connected bodies.

    Returns:
        Dict with ``num_water_bodies``, ``total_water_area``,
        ``mean_water_body_area``, ``max_water_body_area``,
        ``min_water_body_area`` (m²), ``mean_perimeter`` (m),
        ``mean_compactness`` and per-body lists ``water_body_sizes`` (m²),
        ``water_body_perimeters`` (m) and ``water_body_compactness``.

    Raises:
        TypeError: If ``cluster_labels`` is not a numpy array.
        ValueError: If it is not 2-D, ``water_cluster`` is ``None`` or
            ``pixel_size`` is invalid.
    """
    _check_array(cluster_labels, "cluster_labels", (2,))
    if water_cluster is None:
        raise ValueError("water_cluster is None (no water cluster was identified)")
    dx, dy = _parse_pixel_size(pixel_size)
    labeled, n = _label(cluster_labels == water_cluster, connectivity)

    if n == 0:
        areas = perimeters = compactness = np.empty(0)
    else:
        m = _object_metrics(labeled, n, dx, dy)
        areas, perimeters, compactness = m["area"], m["perimeter"], m["compactness"]

    def _agg(values: np.ndarray, fn: Any) -> float:
        return float(fn(values)) if values.size else 0.0

    return {
        "num_water_bodies": n,
        "total_water_area": _agg(areas, np.sum),
        "mean_water_body_area": _agg(areas, np.mean),
        "max_water_body_area": _agg(areas, np.max),
        "min_water_body_area": _agg(areas, np.min),
        "mean_perimeter": _agg(perimeters, np.mean),
        "mean_compactness": _agg(compactness, np.mean),
        "water_body_sizes": areas.tolist(),
        "water_body_perimeters": perimeters.tolist(),
        "water_body_compactness": compactness.tolist(),
    }


def optimize_clustering(
    raster_data: np.ndarray,
    water_index: np.ndarray | None = None,
    method: str = "kmeans",
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    *,
    random_state: int | None = 42,
    sample_size: int = 10_000,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Grid-search clustering parameters by silhouette score.

    Each parameter combination is clustered with :func:`cluster_water_bodies`
    (with ``water_index`` and ``random_state``) and scored by the silhouette
    coefficient on a fixed random subsample of ``sample_size`` valid pixels
    (standardised features). For DBSCAN, noise points are excluded from the
    silhouette and the noise fraction is subtracted. Solutions with fewer than
    two clusters score ``-inf``.

    Args:
        raster_data: ``(rows, cols)`` or ``(rows, cols, bands)`` array.
        water_index: Optional 2-D water index (see :func:`cluster_water_bodies`).
        method: ``"kmeans"`` or ``"dbscan"``.
        param_grid: Mapping of parameter name to candidate values. Defaults:
            k-means ``{"n_clusters": [2, 3, 4, 5]}``; DBSCAN
            ``{"eps": [0.1, 0.2, 0.3, 0.4], "min_samples": [5, 10, 15, 20]}``.
        random_state: Seed for clustering and subsampling.
        sample_size: Number of pixels used for silhouette scoring.

    Returns:
        ``(best_params, info)`` where ``info`` has ``results`` (list of
        ``{"params", "score", "metadata"}``) and ``best_score``. ``best_params``
        is ``None`` if no combination produced at least two clusters.

    Raises:
        ValueError: If ``method`` is unsupported or the grid is empty.
    """
    method_l = method.lower() if isinstance(method, str) else method
    if method_l not in ("kmeans", "dbscan"):
        raise ValueError(f"Unsupported clustering method: {method!r} (use 'kmeans' or 'dbscan')")
    if param_grid is None:
        if method_l == "kmeans":
            param_grid = {"n_clusters": [2, 3, 4, 5]}
        else:
            param_grid = {"eps": [0.1, 0.2, 0.3, 0.4], "min_samples": [5, 10, 15, 20]}
    if not param_grid or any(len(v) == 0 for v in param_grid.values()):
        raise ValueError("param_grid must contain at least one value per parameter")
    sample_size = _check_positive_int(sample_size, "sample_size", minimum=2)

    X, valid, _ = _cluster_features(raster_data, water_index)
    Xv = X if valid is None else X[valid]
    if Xv.shape[0] == 0:
        raise ValueError("raster_data has no valid (finite) pixels to cluster")
    Xs = StandardScaler().fit_transform(Xv)
    rng = np.random.default_rng(random_state)
    idx = (
        rng.choice(Xs.shape[0], size=sample_size, replace=False)
        if Xs.shape[0] > sample_size
        else np.arange(Xs.shape[0])
    )
    X_eval = Xs[idx]

    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    results = []
    keys = list(param_grid.keys())
    for values in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        call = {"random_state": random_state, **params}
        labels, meta = cluster_water_bodies(
            raster_data, method=method_l, water_index=water_index, **call
        )
        flat = labels.reshape(-1)
        lab_valid = flat if valid is None else flat[valid]
        lab_eval = lab_valid[idx]
        keep = lab_eval >= 0
        n_labels = np.unique(lab_eval[keep]).size
        if 2 <= n_labels < int(keep.sum()):
            score = float(silhouette_score(X_eval[keep], lab_eval[keep]))
            if method_l == "dbscan":
                score -= float(np.count_nonzero(lab_valid < 0)) / lab_valid.size
        else:
            score = float("-inf")
        results.append({"params": params, "score": score, "metadata": meta})
        if score > best_score:
            best_score, best_params = score, params
    return best_params, {"results": results, "best_score": best_score}
