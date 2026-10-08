"""Tests for farq.ml."""

from __future__ import annotations

import hashlib
import warnings

import joblib
import numpy as np
import pytest
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

from farq.ml import (
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


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _separable(rng, n=200):
    X = np.vstack([rng.normal(0, 0.3, (n, 3)), rng.normal(3, 0.3, (n, 3))])
    y = np.repeat([0, 1], n)
    return X, y


def _water_scene(rng, shape=(30, 30)):
    """3-band reflectance with a dark square lake and its NDWI (water > 0)."""
    lake = np.zeros(shape, bool)
    lake[5:15, 5:20] = True
    bands = np.where(lake[..., None], 0.05, 0.4) + rng.normal(0, 0.01, (*shape, 3))
    ndwi = np.where(lake, 0.5, -0.4) + rng.normal(0, 0.02, shape)
    return bands, ndwi, lake


# --------------------------------------------------------------------------- #
# extract_features
# --------------------------------------------------------------------------- #
class TestExtractFeatures:
    def test_shapes_and_order(self, rng):
        img = rng.random((8, 9, 2))
        f = extract_features(img)
        assert f.shape == (8, 9, 6)
        np.testing.assert_array_equal(f[..., :2], img)
        np.testing.assert_allclose(
            f[..., 2], ndimage.uniform_filter(img[..., 0], 3, mode="reflect")
        )

    def test_2d_input(self, rng):
        assert extract_features(rng.random((5, 5))).shape == (5, 5, 3)

    def test_window_one_returns_bands(self, rng):
        img = rng.random((4, 4, 3))
        np.testing.assert_array_equal(extract_features(img, window_size=1), img)

    def test_variance_matches_bruteforce(self, rng):
        img = rng.random((12, 10)) * 1000 + 5000
        f = extract_features(img, window_size=5)
        expected = ndimage.generic_filter(img, np.var, size=5, mode="reflect")
        np.testing.assert_allclose(f[..., 2], expected, rtol=1e-6, atol=1e-6)

    def test_integer_input_promoted(self):
        img = np.array([[0, 10000], [20000, 65535]], dtype=np.uint16)
        f = extract_features(img)
        assert f.dtype == np.float32
        assert (f[..., 2] > 0).all()

    def test_nan_aware(self):
        img = np.arange(25, dtype=float).reshape(5, 5)
        img[2, 2] = np.nan
        f = extract_features(img)
        window = img[1:4, 1:4]
        assert f[2, 2, 1] == pytest.approx(np.nanmean(window))
        assert f[2, 2, 2] == pytest.approx(np.nanvar(window))
        assert np.isnan(f[2, 2, 0])
        assert np.isfinite(f[..., 1:]).all()

    def test_all_nan_window(self):
        img = np.full((5, 5), np.nan)
        img[0, 0] = 1.0
        f = extract_features(img)
        assert np.isnan(f[4, 4, 1])
        assert f[0, 0, 1] == pytest.approx(1.0)

    def test_indices(self, rng):
        img = rng.random((6, 6, 2))
        idx = rng.random((6, 6))
        f = extract_features(img, indices={"ndwi": idx})
        assert f.shape == (6, 6, 9)
        np.testing.assert_array_equal(f[..., 2], idx)
        f2 = extract_features(img, indices=[idx], window_size=1)
        assert f2.shape == (6, 6, 3)

    def test_invalid(self, rng):
        img = rng.random((6, 6))
        with pytest.raises(TypeError, match="precomputed"):
            extract_features(img, indices=["ndwi"])
        with pytest.raises(ValueError, match="shape"):
            extract_features(img, indices=[np.zeros((3, 3))])
        with pytest.raises(ValueError):
            extract_features(img, window_size=0)
        with pytest.raises(TypeError):
            extract_features(img, window_size=2.5)
        with pytest.raises(TypeError):
            extract_features(img.tolist())
        with pytest.raises(ValueError):
            extract_features(np.zeros((2, 2, 2, 2)))


# --------------------------------------------------------------------------- #
# train / predict
# --------------------------------------------------------------------------- #
class TestTrainPredict:
    def test_train_separable(self, rng):
        X, y = _separable(rng)
        _, metrics = train_classifier(X, y, n_estimators=20)
        assert metrics["accuracy"] == 1.0
        assert metrics["classes"] == [0, 1]
        assert metrics["n_train"] + metrics["n_test"] == 400
        assert np.array(metrics["confusion_matrix"]).sum() == metrics["n_test"]
        # stratified split keeps class balance
        assert np.array(metrics["confusion_matrix"]).sum(axis=1).tolist() == [40, 40]
        assert isinstance(metrics["classification_report"], str)

    def test_deterministic(self, rng):
        X, y = _separable(rng)
        X = X + rng.normal(0, 2, X.shape)  # make it noisy
        m1, r1 = train_classifier(X, y, n_estimators=10, random_state=1)
        m2, r2 = train_classifier(X, y, n_estimators=10, random_state=1)
        assert r1["accuracy"] == r2["accuracy"]
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))

    def test_raster_inputs_and_ignore_label(self, rng):
        feats = rng.random((10, 10, 2))
        labels = np.zeros((10, 10), int)
        labels[:, :5] = 1
        labels[:, 5:] = 2
        labels[0, :] = 0  # unlabelled
        feats[1, 1, 0] = np.nan
        _, metrics = train_classifier(feats, labels, ignore_label=0, n_estimators=5)
        assert metrics["n_dropped"] == 11
        assert metrics["classes"] == [1, 2]

    def test_test_size_zero(self, rng):
        X, y = _separable(rng, 20)
        _, metrics = train_classifier(X, y, test_size=0, n_estimators=5)
        assert metrics["n_train"] == 40
        assert metrics["accuracy"] is None

    def test_errors(self, rng):
        X, y = _separable(rng, 10)
        with pytest.raises(ValueError, match="match"):
            train_classifier(X, y[:-1])
        with pytest.raises(ValueError, match="two classes"):
            train_classifier(X, np.zeros(20))
        with pytest.raises(ValueError, match="Unsupported"):
            train_classifier(X, y, model_type="svm")
        with pytest.raises(ValueError, match="test_size"):
            train_classifier(X, y, test_size=1.5)
        with pytest.raises(TypeError):
            train_classifier(X.tolist(), y)

    def test_predict_raster(self, rng):
        X, y = _separable(rng)
        model, _ = train_classifier(X, y, n_estimators=10)
        feats = np.zeros((4, 5, 3))
        feats[2:] = 3.0
        pred = predict_raster(model, feats)
        assert pred.shape == (4, 5)
        assert (pred[:2] == 0).all()
        assert (pred[2:] == 1).all()
        np.testing.assert_array_equal(predict_raster(model, feats, batch_size=3), pred)

    def test_predict_nan_rows_filled(self, rng):
        X, y = _separable(rng)
        model, _ = train_classifier(X, y, n_estimators=5)
        feats = np.full((2, 2, 3), 3.0)
        feats[0, 0, 1] = np.nan
        pred = predict_raster(model, feats, fill_value=255)
        assert pred[0, 0] == 255
        assert (pred.ravel()[1:] == 1).all()
        assert (predict_raster(model, np.full((2, 3), np.nan)) == -1).all()

    def test_predict_errors(self, rng):
        X, y = _separable(rng, 10)
        model, _ = train_classifier(X, y, n_estimators=3)
        with pytest.raises(ValueError, match="expects 3 features"):
            predict_raster(model, np.zeros((2, 2, 4)))
        with pytest.raises(TypeError):
            predict_raster(object(), np.zeros((2, 3)))
        with pytest.raises(ValueError):
            predict_raster(model, np.zeros((2, 3)), batch_size=0)


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
@pytest.fixture
def model(rng):
    X, y = _separable(rng, 20)
    return RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y), X


class TestPersistence:
    def test_roundtrip(self, tmp_path, model):
        clf, X = model
        path = tmp_path / "sub" / "model.joblib"
        digest = save_model(clf, path, {"bands": ["g", "r", "nir"]})
        sidecar = tmp_path / "sub" / "model.joblib.sha256"
        assert sidecar.read_text().split() == [digest, "model.joblib"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded, meta = load_model(path)
        np.testing.assert_array_equal(loaded.predict(X), clf.predict(X))
        assert meta["bands"] == ["g", "r", "nir"]
        assert meta["_farq"]["format_version"] == 1
        assert meta["_farq"]["model_class"].endswith("RandomForestClassifier")
        assert not list(tmp_path.glob("sub/*.tmp"))

    def test_expected_sha256(self, tmp_path, model):
        path = tmp_path / "m.joblib"
        digest = save_model(model[0], path)
        load_model(path, expected_sha256=digest.upper())
        with pytest.raises(ModelIntegrityError, match="mismatch"):
            load_model(path, expected_sha256="0" * 64)
        with pytest.raises(ValueError, match="not a valid SHA-256"):
            load_model(path, expected_sha256="abc")

    def test_tampered_file_rejected_before_unpickling(self, tmp_path, model):
        path = tmp_path / "m.joblib"
        save_model(model[0], path)
        path.write_bytes(path.read_bytes() + b"x")
        with pytest.raises(ModelIntegrityError):
            load_model(path)

    def test_bad_sidecar(self, tmp_path, model):
        path = tmp_path / "m.joblib"
        save_model(model[0], path)
        (tmp_path / "m.joblib.sha256").write_text("garbage\n")
        with pytest.raises(ValueError, match="Sidecar"):
            load_model(path)

    def test_legacy_file_without_sidecar_warns(self, tmp_path, model):
        path = tmp_path / "old.joblib"
        joblib.dump({"model": model[0], "metadata": {"a": 1}}, path)
        with pytest.warns(UserWarning, match="integrity"):
            loaded, meta = load_model(path)
        assert meta == {"a": 1}
        assert hasattr(loaded, "predict")
        # pinning a hash satisfies the integrity check without a sidecar
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_model(path, expected_sha256=digest)

    def test_raw_estimator_dump(self, tmp_path, model):
        path = tmp_path / "raw.joblib"
        joblib.dump(model[0], path)
        with pytest.warns(UserWarning):
            loaded, meta = load_model(path)
        assert meta == {}
        assert hasattr(loaded, "predict")

    def test_sklearn_version_mismatch_warns(self, tmp_path, model):
        path = tmp_path / "m.joblib"
        payload = {"model": model[0], "metadata": {}, "farq_info": {"sklearn_version": "0.0"}}
        joblib.dump(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.warns(UserWarning, match="scikit-learn 0.0"):
            load_model(path, expected_sha256=digest)

    def test_errors(self, tmp_path, model):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "missing.joblib")
        with pytest.raises(ValueError, match="regular file"):
            load_model(tmp_path)
        not_model = tmp_path / "x.joblib"
        joblib.dump({"foo": 1}, not_model)
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="farq model"):
            load_model(not_model)
        corrupt = tmp_path / "c.joblib"
        corrupt.write_bytes(b"not a pickle")
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="Could not load"):
            load_model(corrupt)
        with pytest.raises(TypeError):
            save_model(object(), tmp_path / "o.joblib")
        with pytest.raises(TypeError):
            save_model(model[0], tmp_path / "o.joblib", metadata=["a"])
        with pytest.raises(ValueError, match="directory"):
            save_model(model[0], tmp_path)

    def test_compressed(self, tmp_path, model):
        path = tmp_path / "m.joblib"
        save_model(model[0], path, compress=3)
        assert hasattr(load_model(path)[0], "predict")

    def test_file_swapped_after_hashing_is_not_loaded(self, tmp_path, model, monkeypatch):
        """The bytes that are unpickled must be exactly the bytes that were verified."""
        import farq.ml as ml_module

        path = tmp_path / "m.joblib"
        digest = save_model(model[0], path)
        evil = tmp_path / "evil.joblib"
        joblib.dump({"model": "not the verified model"}, evil)
        real_load = ml_module.joblib.load

        def swap_then_load(source, *args, **kwargs):
            # An attacker replaces the file between the hash check and unpickling.
            path.write_bytes(evil.read_bytes())
            return real_load(source, *args, **kwargs)

        monkeypatch.setattr(ml_module.joblib, "load", swap_then_load)
        loaded, _ = load_model(path, expected_sha256=digest)
        assert isinstance(loaded, RandomForestClassifier)

    def test_oversized_sidecar_rejected(self, tmp_path, model):
        path = tmp_path / "m.joblib"
        save_model(model[0], path)
        (tmp_path / "m.joblib.sha256").write_bytes(b"0" * 100_000)
        with pytest.raises(ModelIntegrityError, match="too large"):
            load_model(path)


# --------------------------------------------------------------------------- #
# detect_changes_ml
# --------------------------------------------------------------------------- #
class TestDetectChanges:
    def test_threshold_unsigned_no_wraparound(self):
        a = np.array([[10, 200]], dtype=np.uint8)
        b = np.array([[20, 100]], dtype=np.uint8)
        changes = detect_changes_ml(a, b, threshold=50)
        np.testing.assert_array_equal(changes, [[False, True]])
        assert changes.dtype == bool

    def test_multiband_magnitude(self):
        a = np.zeros((2, 2, 2))
        b = np.zeros((2, 2, 2))
        b[0, 0] = [3, 4]  # magnitude 5
        b[1, 1] = [1, 1]  # magnitude ~1.41
        changes = detect_changes_ml(a, b, threshold=2)
        assert changes.shape == (2, 2)
        np.testing.assert_array_equal(changes, [[True, False], [False, False]])

    def test_nan_is_unchanged(self):
        a = np.array([[0.0, np.nan]])
        b = np.array([[1.0, 1.0]])
        np.testing.assert_array_equal(detect_changes_ml(a, b, threshold=0.5), [[True, False]])

    def test_with_model(self, rng):
        # Train on features of difference images: 0 = no change, 1 = change.
        diffs = np.concatenate([np.zeros((20, 20)), np.ones((20, 20))], axis=1)
        diffs = diffs + rng.normal(0, 0.05, diffs.shape)
        feats = extract_features(diffs)
        labels = np.concatenate([np.zeros((20, 20), int), np.ones((20, 20), int)], axis=1)
        model, _ = train_classifier(feats, labels, n_estimators=10)
        r1 = np.zeros((10, 10))
        r2 = np.zeros((10, 10))
        r2[:, 5:] = 1.0
        changes = detect_changes_ml(r1, r2, model=model)
        assert changes.shape == (10, 10)
        assert changes[:, 7:].all()
        assert not changes[:, :3].any()

    def test_errors(self):
        with pytest.raises(ValueError, match="same shape"):
            detect_changes_ml(np.zeros((2, 2)), np.zeros((3, 3)))
        with pytest.raises(TypeError):
            detect_changes_ml(np.zeros((2, 2)), [[0, 0], [0, 0]])
        with pytest.raises(TypeError):
            detect_changes_ml(np.zeros((2, 2)), np.zeros((2, 2)), model=object())


# --------------------------------------------------------------------------- #
# augment_training_data
# --------------------------------------------------------------------------- #
class TestAugment:
    def test_shapes_and_labels(self, rng):
        X, y = _separable(rng, 10)
        Xa, ya = augment_training_data(X, y, augmentation_factor=3)
        assert Xa.shape == (60, 3)
        assert ya.shape == (60,)
        np.testing.assert_array_equal(Xa[:20], X)
        np.testing.assert_array_equal(ya, np.tile(y, 3))

    def test_non_square_2d_features(self, rng):
        # Regression: np.rot90 used to scramble samples/features.
        X = rng.random((7, 3))
        Xa, ya = augment_training_data(X, np.arange(7))
        assert Xa.shape == (14, 3)
        assert ya.shape == (14,)

    def test_deterministic_and_no_global_state(self, rng):
        X, y = _separable(rng, 10)
        np.random.seed(123)  # noqa: NPY002
        before = np.random.random()  # noqa: NPY002
        np.random.seed(123)  # noqa: NPY002
        a, _ = augment_training_data(X, y, random_state=5)
        assert np.random.random() == before  # noqa: NPY002
        b, _ = augment_training_data(X, y, random_state=5)
        np.testing.assert_array_equal(a, b)
        c, _ = augment_training_data(X, y, random_state=6)
        assert not np.array_equal(a, c)

    def test_noise_is_relative(self):
        X = np.column_stack([np.linspace(0, 1, 1000), np.linspace(0, 1000, 1000)])
        Xa, _ = augment_training_data(X, np.zeros(1000), noise_level=0.1)
        noise = Xa[1000:] - X
        ratio = noise.std(axis=0) / X.std(axis=0)
        np.testing.assert_allclose(ratio, 0.1, rtol=0.1)

    def test_factor_one_and_int_input(self):
        X = np.arange(6).reshape(3, 2)
        Xa, _ = augment_training_data(X, np.arange(3), augmentation_factor=1)
        np.testing.assert_array_equal(Xa, X)
        assert Xa.dtype.kind == "f"

    def test_errors(self):
        with pytest.raises(ValueError):
            augment_training_data(np.zeros((3, 2)), np.zeros(2))
        with pytest.raises(ValueError):
            augment_training_data(np.zeros((3, 2)), np.zeros(3), augmentation_factor=0)
        with pytest.raises(ValueError):
            augment_training_data(np.zeros((3, 2)), np.zeros(3), noise_level=-1)


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
class TestClustering:
    def test_kmeans_with_water_index(self, rng):
        bands, ndwi, lake = _water_scene(rng)
        labels, meta = cluster_water_bodies(bands, water_index=ndwi)
        assert labels.shape == lake.shape
        assert labels.dtype == np.int32
        np.testing.assert_array_equal(labels == meta["water_cluster"], lake)
        assert meta["n_clusters"] == 2
        assert "highest" in meta["water_rule"]
        assert meta["cluster_means"][meta["water_cluster"]] > 0

    def test_kmeans_multiband_dark_water(self, rng):
        bands, _, lake = _water_scene(rng)
        labels, meta = cluster_water_bodies(bands)
        np.testing.assert_array_equal(labels == meta["water_cluster"], lake)

    def test_kmeans_single_band_index(self, rng):
        _, ndwi, lake = _water_scene(rng)
        labels, meta = cluster_water_bodies(ndwi)
        np.testing.assert_array_equal(labels == meta["water_cluster"], lake)
        _, meta_low = cluster_water_bodies(ndwi, water_high=False)
        assert meta_low["water_cluster"] != meta["water_cluster"]

    def test_deterministic(self, rng):
        bands, _, _ = _water_scene(rng)
        a, _ = cluster_water_bodies(bands, n_clusters=3, random_state=1)
        b, _ = cluster_water_bodies(bands, n_clusters=3, random_state=1)
        np.testing.assert_array_equal(a, b)

    def test_nan_pixels(self, rng):
        bands, ndwi, lake = _water_scene(rng)
        ndwi[0, 0] = np.nan
        bands[1, 1, 2] = np.nan
        labels, meta = cluster_water_bodies(bands, water_index=ndwi)
        assert labels[0, 0] == -1
        assert labels[1, 1] == -1
        assert meta["n_invalid"] == 2
        assert (labels[lake] == meta["water_cluster"]).all()

    def test_sample_size(self, rng):
        bands, ndwi, lake = _water_scene(rng)
        labels, meta = cluster_water_bodies(bands, water_index=ndwi, sample_size=100)
        np.testing.assert_array_equal(labels == meta["water_cluster"], lake)

    def test_dbscan(self, rng):
        _, ndwi, lake = _water_scene(rng, (12, 12))
        labels, meta = cluster_water_bodies(ndwi, method="DBSCAN", eps=0.5, min_samples=3)
        assert meta["n_clusters"] == 2
        assert meta["noise_points"] == 0
        np.testing.assert_array_equal(labels == meta["water_cluster"], lake)

    def test_dbscan_all_noise(self, rng):
        labels, meta = cluster_water_bodies(rng.random((5, 5)), method="dbscan", eps=1e-6)
        assert meta["water_cluster"] is None
        assert (labels == -1).all()

    def test_errors(self, rng):
        img = rng.random((5, 5))
        with pytest.raises(ValueError, match="Unsupported"):
            cluster_water_bodies(img, method="spectral")
        with pytest.raises(ValueError, match="shape"):
            cluster_water_bodies(img, water_index=np.zeros((4, 4)))
        with pytest.raises(ValueError, match="valid"):
            cluster_water_bodies(np.full((3, 3), np.nan))
        with pytest.raises(ValueError):
            cluster_water_bodies(img, n_clusters=0)
        with pytest.raises(ValueError, match="sample_size"):
            cluster_water_bodies(img, method="dbscan", sample_size=10)

    def test_analyze_water_clusters(self):
        labels = np.zeros((10, 10), int)
        labels[1:4, 1:4] = 1  # 9 px
        labels[6:8, 6:9] = 1  # 6 px
        labels[0, 9] = 2
        r = analyze_water_clusters(labels, water_cluster=1, pixel_size=10)
        assert r["num_water_bodies"] == 2
        assert r["water_body_sizes"] == pytest.approx([900.0, 600.0])
        assert r["water_body_perimeters"] == pytest.approx([120.0, 100.0])
        assert r["total_water_area"] == pytest.approx(1500.0)
        assert r["max_water_body_area"] == pytest.approx(900.0)
        assert r["min_water_body_area"] == pytest.approx(600.0)
        assert r["mean_perimeter"] == pytest.approx(110.0)
        assert r["water_body_compactness"][0] == pytest.approx(np.pi / 4)

    def test_analyze_no_water(self):
        r = analyze_water_clusters(np.zeros((3, 3), int), water_cluster=1)
        assert r["num_water_bodies"] == 0
        assert r["total_water_area"] == 0
        assert r["water_body_sizes"] == []

    def test_analyze_errors(self):
        with pytest.raises(ValueError):
            analyze_water_clusters(np.zeros((3, 3), int), water_cluster=None)
        with pytest.raises(ValueError):
            analyze_water_clusters(np.zeros(3, int), water_cluster=1)
        with pytest.raises(ValueError):
            analyze_water_clusters(np.zeros((3, 3), int), 1, pixel_size=-1)

    def test_pipeline_with_cluster_output(self, rng):
        bands, ndwi, lake = _water_scene(rng)
        labels, meta = cluster_water_bodies(bands, water_index=ndwi)
        r = analyze_water_clusters(labels, meta["water_cluster"], pixel_size=30)
        assert r["num_water_bodies"] == 1
        assert r["total_water_area"] == pytest.approx(lake.sum() * 900)


class TestOptimizeClustering:
    def test_default_kmeans_grid_picks_two(self, rng):
        bands, ndwi, _ = _water_scene(rng)
        best, info = optimize_clustering(bands, water_index=ndwi)
        assert best == {"n_clusters": 2}
        assert len(info["results"]) == 4
        assert info["best_score"] == max(r["score"] for r in info["results"])

    def test_random_state_in_grid(self, rng):
        # Regression: used to raise "got multiple values for keyword 'random_state'".
        bands, _, _ = _water_scene(rng, (10, 10))
        best, _ = optimize_clustering(bands, param_grid={"n_clusters": [2], "random_state": [0]})
        assert best == {"n_clusters": 2, "random_state": 0}

    def test_deterministic(self, rng):
        bands, ndwi, _ = _water_scene(rng)
        a = optimize_clustering(bands, ndwi, param_grid={"n_clusters": [2, 3]}, sample_size=50)
        b = optimize_clustering(bands, ndwi, param_grid={"n_clusters": [2, 3]}, sample_size=50)
        assert [r["score"] for r in a[1]["results"]] == [r["score"] for r in b[1]["results"]]

    def test_dbscan(self, rng):
        _, ndwi, _ = _water_scene(rng, (12, 12))
        best, info = optimize_clustering(
            ndwi, method="dbscan", param_grid={"eps": [1e-6, 0.5], "min_samples": [3]}
        )
        assert best == {"eps": 0.5, "min_samples": 3}
        assert info["results"][0]["score"] == float("-inf")

    def test_errors(self, rng):
        with pytest.raises(ValueError):
            optimize_clustering(rng.random((4, 4)), method="foo")
        with pytest.raises(ValueError):
            optimize_clustering(rng.random((4, 4)), param_grid={"n_clusters": []})
