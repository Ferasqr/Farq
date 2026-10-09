"""Tests for farq.radiometry (relative radiometric normalization and IR-MAD)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from farq import radiometry as rad

GAINS = np.array([1.3, 0.8, 1.1])
OFFSETS = np.array([0.05, -0.02, 0.10])
BLOCK = (slice(10, 30), slice(40, 70))  # 600 changed pixels


def make_scene(size=100, noise=0.003, seed=0, n_bands=3):
    """``after = gain * before + offset + noise`` except in BLOCK (real change)."""
    rng = np.random.default_rng(seed)
    latent = rng.uniform(0.05, 0.45, size=(n_bands, size, size))
    before = latent + noise * rng.standard_normal(latent.shape)
    g, o = GAINS[:n_bands, None, None], OFFSETS[:n_bands, None, None]
    after = g * latent + o + noise * rng.standard_normal(latent.shape)
    after[(slice(None), *BLOCK)] += np.array([0.3, -0.25, 0.3])[:n_bands, None, None]
    truth = np.zeros((size, size), bool)
    truth[BLOCK] = True
    return before, after, truth


@pytest.fixture
def scene():
    return make_scene()


@pytest.fixture(autouse=True)
def _no_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


# --------------------------------------------------------------------------- #
# histogram_match
# --------------------------------------------------------------------------- #
class TestHistogramMatch:
    def test_exact_same_size_reproduces_reference_values(self):
        rng = np.random.default_rng(1)
        source = rng.normal(size=(2, 30, 40))
        reference = rng.gamma(2.0, size=(2, 30, 40))
        out = rad.histogram_match(source, reference)
        assert out.shape == source.shape
        for i in range(2):
            np.testing.assert_array_equal(np.sort(out[i].ravel()), np.sort(reference[i].ravel()))
            # monotone: the rank order of the source is preserved
            order = np.argsort(source[i].ravel())
            assert np.all(np.diff(out[i].ravel()[order]) >= 0)

    def test_quantiles_match_different_sizes(self):
        rng = np.random.default_rng(2)
        source = rng.uniform(0, 255, size=(80, 90)).astype(np.uint8)
        reference = rng.normal(0.3, 0.05, size=(50, 61))
        out = rad.histogram_match(source, reference)
        assert out.dtype == np.float64
        probs = np.linspace(0.05, 0.95, 19)
        np.testing.assert_allclose(
            np.quantile(out, probs), np.quantile(reference, probs), atol=0.01
        )

    def test_n_quantiles(self):
        rng = np.random.default_rng(3)
        source = rng.exponential(size=(200, 200)).astype(np.float32)
        reference = rng.normal(5.0, 2.0, size=(150, 150))
        out = rad.histogram_match(source, reference, n_quantiles=256)
        probs = np.linspace(0.02, 0.98, 25)
        np.testing.assert_allclose(
            np.quantile(out, probs), np.quantile(reference, probs), atol=0.05
        )

    def test_ties_map_to_equal_values(self):
        source = np.array([[1.0, 1.0, 2.0, 3.0]])
        reference = np.array([[10.0, 20.0, 30.0, 40.0]])
        out = rad.histogram_match(source, reference)
        assert out[0, 0] == out[0, 1] == 15.0
        np.testing.assert_array_equal(out[0, 2:], [30.0, 40.0])

    def test_nan_preserved_per_band(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(2, 20, 20))
        reference = rng.normal(3.0, 1.0, size=(2, 20, 20))
        source[0, 0, 0] = np.nan
        reference[1, 5, 5] = np.nan  # reference NaN only shrinks the reference sample
        out = rad.histogram_match(source, reference)
        assert np.isnan(out[0, 0, 0])
        assert np.isnan(out).sum() == 1
        assert np.isfinite(out[1]).all()

    def test_nodata_and_masked_array(self):
        source = np.arange(1, 17, dtype=np.uint16).reshape(4, 4)
        source[0, 0] = 0
        reference = np.linspace(0.1, 1.0, 16).reshape(4, 4)
        out = rad.histogram_match(source, reference, nodata=0)
        assert np.isnan(out[0, 0]) and np.isfinite(out).sum() == 15
        masked = np.ma.masked_array(source.astype(float), mask=source == 0)
        np.testing.assert_array_equal(rad.histogram_match(masked, reference), out)

    def test_valid_mask_restricts_fit_but_maps_all(self):
        source = np.arange(16, dtype=float).reshape(4, 4)
        reference = 2.0 * source
        reference[3] = 1000.0  # changed row, excluded from the fit
        valid = np.ones((4, 4), bool)
        valid[3] = False
        out = rad.histogram_match(source, reference, valid=valid)
        np.testing.assert_allclose(out[:3], reference[:3])
        assert np.isfinite(out[3]).all() and out[3].max() <= reference[:3].max()

    def test_does_not_modify_input(self):
        source = np.array([[1.0, np.nan], [3.0, 4.0]])
        copy = source.copy()
        rad.histogram_match(source, np.ones((2, 2)))
        np.testing.assert_array_equal(source, copy)

    def test_errors(self):
        a = np.ones((2, 5, 5))
        with pytest.raises(ValueError, match="number of bands"):
            rad.histogram_match(a, np.ones((3, 5, 5)))
        with pytest.raises(ValueError, match="same height and width"):
            rad.histogram_match(a, np.ones((2, 4, 4)), valid=np.ones((5, 5), bool))
        with pytest.raises(ValueError, match="valid must have shape"):
            rad.histogram_match(a, a, valid=np.ones((4, 5), bool))
        with pytest.raises(ValueError, match="n_quantiles"):
            rad.histogram_match(a, a, n_quantiles=1)
        with pytest.raises(ValueError, match="2-D image or a 3-D stack"):
            rad.histogram_match(np.ones(5), np.ones(5))
        with pytest.raises(TypeError, match="numpy array"):
            rad.histogram_match([[1.0, 2.0]], a[0])
        with pytest.raises(TypeError, match="dtype"):
            rad.histogram_match(np.ones((2, 2), complex), np.ones((2, 2)))
        b = a.copy()
        b[1] = np.nan
        b[0, 0, 0] = 1.0
        with pytest.raises(ValueError, match="band 1 has no valid pixels"):
            rad.histogram_match(b, a)


# --------------------------------------------------------------------------- #
# linear_normalize
# --------------------------------------------------------------------------- #
class TestLinearNormalize:
    @pytest.mark.parametrize("method", ["ols", "orthogonal", "theil_sen", "mean_std"])
    def test_recovers_gain_offset_on_no_change_mask(self, scene, method):
        before, after, truth = scene
        res = rad.linear_normalize(before, after, mask=~truth, method=method)
        np.testing.assert_allclose(res.gains, GAINS, rtol=0.01)
        np.testing.assert_allclose(res.offsets, OFFSETS, atol=0.005)
        np.testing.assert_array_equal(res.invariant_mask, ~truth)
        assert res.n_invariant == (~truth).sum()
        assert np.all(res.r2 > 0.99)
        assert np.all(res.rmse < 0.01)
        assert res.normalized.shape == before.shape
        resid = (res.normalized - after)[:, ~truth]
        assert np.abs(resid).mean() < 0.01

    def test_normalize_after_to_before(self, scene):
        before, after, truth = scene
        res = rad.linear_normalize(after, before, mask=~truth, method="orthogonal")
        np.testing.assert_allclose(res.gains, 1.0 / GAINS, rtol=0.01)
        np.testing.assert_allclose(res.offsets, -OFFSETS / GAINS, atol=0.005)

    def test_theil_sen_is_robust_to_changed_pixels(self, scene):
        before, after, _ = scene
        robust = rad.linear_normalize(before, after, method="theil_sen")
        ols = rad.linear_normalize(before, after, method="ols")
        np.testing.assert_allclose(robust.gains, GAINS, rtol=0.02)
        assert np.abs(ols.gains - GAINS).max() > 3 * np.abs(robust.gains - GAINS).max()

    def test_theil_sen_small_sample_exact(self):
        x = np.arange(20, dtype=float).reshape(4, 5)
        y = 3.0 * x - 2.0
        y[0, 0] = 100.0  # an outlier
        res = rad.linear_normalize(x, y, method="theil_sen")
        assert res.gains[0] == pytest.approx(3.0)
        assert res.offsets[0] == pytest.approx(-2.0)

    def test_two_d_input_and_dtypes(self):
        rng = np.random.default_rng(5)
        x = rng.integers(10, 200, size=(30, 30)).astype(np.uint8)
        y = (2.0 * x + 5.0).astype(np.float32)
        res = rad.linear_normalize(x, y)
        assert res.normalized.shape == (30, 30)
        assert res.normalized.dtype == np.float32
        assert res.gains.shape == (1,) and res.gains.dtype == np.float64
        np.testing.assert_allclose(res.normalized, y, rtol=1e-5)

    def test_nan_excluded_and_preserved(self, scene):
        before, after, truth = scene
        before, after = before.copy(), after.copy()
        before[1, 0, 0] = np.nan
        after[0, 5, 5] = np.inf
        before[2, 50, 50] = -9999.0
        res = rad.linear_normalize(before, after, mask=~truth, nodata=-9999.0)
        np.testing.assert_allclose(res.gains, GAINS, rtol=0.01)
        assert not res.invariant_mask[0, 0] and not res.invariant_mask[5, 5]
        assert not res.invariant_mask[50, 50]
        assert np.isnan(res.normalized[1, 0, 0]) and np.isfinite(res.normalized[0, 0, 0])
        assert np.isnan(res.normalized[2, 50, 50])
        assert np.isfinite(res.normalized[:, 5, 5]).all()  # only the reference was invalid
        assert np.isnan(res.normalized).sum() == 2

    def test_float_mask_nan_is_false(self, scene):
        before, after, truth = scene
        mask = (~truth).astype(float)
        mask[0, 0] = np.nan
        res = rad.linear_normalize(before, after, mask=mask)
        assert not res.invariant_mask[0, 0]

    def test_apply(self, scene):
        before, after, truth = scene
        res = rad.linear_normalize(before, after, mask=~truth)
        np.testing.assert_allclose(res.apply(before), res.normalized)
        np.testing.assert_allclose(res.apply(before[:, :10, :10]), res.normalized[:, :10, :10])
        with pytest.raises(ValueError, match="band"):
            res.apply(before[:2])
        one = rad.linear_normalize(before[0], after[0])
        assert one.apply(before[0]).shape == before[0].shape

    def test_errors(self, scene):
        before, after, truth = scene
        with pytest.raises(ValueError, match="Unknown method"):
            rad.linear_normalize(before, after, method="lsq")
        with pytest.raises(ValueError, match="same shape"):
            rad.linear_normalize(before, after[:, :50])
        with pytest.raises(ValueError, match="mask must have shape"):
            rad.linear_normalize(before, after, mask=truth[:50])
        few = np.zeros_like(truth)
        few[0, :5] = True
        with pytest.raises(ValueError, match="only 5 valid pixel"):
            rad.linear_normalize(before, after, mask=few)
        const = before.copy()
        const[1] = 0.3
        with pytest.raises(ValueError, match="source band 1 is constant"):
            rad.linear_normalize(const, after)


# --------------------------------------------------------------------------- #
# pif_normalize
# --------------------------------------------------------------------------- #
class TestPIFNormalize:
    @pytest.mark.parametrize("method", ["irmad", "pca", "percentile"])
    def test_selects_invariant_pixels_and_recovers_gains(self, scene, method):
        before, after, truth = scene
        res = rad.pif_normalize(before, after, method=method)
        assert not res.invariant_mask[truth].any()
        assert res.n_invariant >= 200
        np.testing.assert_allclose(res.gains, GAINS, rtol=0.01)
        np.testing.assert_allclose(res.offsets, OFFSETS, atol=0.005)
        assert np.all(res.r2 > 0.99)

    @pytest.mark.parametrize("regression", ["ols", "orthogonal", "theil_sen", "mean_std"])
    def test_regressions(self, scene, regression):
        before, after, _ = scene
        res = rad.pif_normalize(after, before, regression=regression)
        np.testing.assert_allclose(res.gains, 1.0 / GAINS, rtol=0.01)

    def test_irmad_min_prob_controls_fraction(self, scene):
        before, after, truth = scene
        n_unchanged = (~truth).sum()
        res = rad.pif_normalize(before, after, min_prob=0.5)
        assert 0.4 < res.n_invariant / n_unchanged < 0.6
        res = rad.pif_normalize(before, after, method="percentile", percentile=10.0)
        assert res.n_invariant == pytest.approx(0.1 * truth.size, abs=2)

    def test_valid_and_nan(self, scene):
        before, after, truth = scene
        before = before.copy()
        before[:, :5, :] = np.nan
        valid = np.ones_like(truth)
        valid[:, :10] = False
        res = rad.pif_normalize(before, after, valid=valid)
        assert not res.invariant_mask[:5].any() and not res.invariant_mask[:, :10].any()
        assert np.isnan(res.normalized[:, :5]).all()
        assert np.isfinite(res.normalized[:, 5:]).all()
        np.testing.assert_allclose(res.gains, GAINS, rtol=0.01)

    def test_single_band(self, scene):
        before, after, truth = scene
        for method in ("irmad", "pca", "percentile"):
            res = rad.pif_normalize(before[0], after[0], method=method)
            assert res.normalized.shape == before[0].shape
            assert not res.invariant_mask[truth].any()
            assert res.gains[0] == pytest.approx(GAINS[0], rel=0.01)

    def test_deterministic(self, scene):
        before, after, _ = scene
        r1 = rad.pif_normalize(before, after, regression="theil_sen")
        r2 = rad.pif_normalize(before, after, regression="theil_sen")
        np.testing.assert_array_equal(r1.normalized, r2.normalized)
        np.testing.assert_array_equal(r1.invariant_mask, r2.invariant_mask)

    def test_errors(self, scene):
        before, after, _ = scene
        with pytest.raises(ValueError, match="Unknown method"):
            rad.pif_normalize(before, after, method="kmeans")
        with pytest.raises(ValueError, match="Unknown regression"):
            rad.pif_normalize(before, after, regression="lasso")
        with pytest.raises(ValueError, match="min_prob"):
            rad.pif_normalize(before, after, min_prob=1.0)
        with pytest.raises(ValueError, match="percentile"):
            rad.pif_normalize(before, after, method="percentile", percentile=0)
        with pytest.raises(ValueError, match="n_sigma"):
            rad.pif_normalize(before, after, method="pca", n_sigma=0)
        with pytest.raises(ValueError, match="min_pixels"):
            rad.pif_normalize(before, after, min_pixels=3)
        with pytest.raises(ValueError, match=r"pseudo-invariant pixel.*lower min_prob"):
            rad.pif_normalize(before, after, min_prob=0.9999)
        with pytest.raises(ValueError, match="valid pixel"):
            rad.pif_normalize(before[:, :5, :5], after[:, :5, :5])


# --------------------------------------------------------------------------- #
# IR-MAD
# --------------------------------------------------------------------------- #
class TestIRMAD:
    def test_result_structure(self, scene):
        before, after, _ = scene
        res = rad.irmad(before, after)
        assert res.mad_variates.shape == before.shape
        assert res.chi2.shape == res.no_change_prob.shape == before.shape[1:]
        assert res.converged and 1 < res.n_iter < 50
        rho = res.canonical_correlations
        assert rho.shape == (3,)
        assert np.all(np.diff(rho) >= 0) and np.all((rho > 0) & (rho < 1))
        np.testing.assert_allclose(res.no_change_prob, stats.chi2.sf(res.chi2, 3), rtol=1e-6)
        np.testing.assert_allclose(res.chi2 >= 0, True)

    def test_flags_block_with_calibrated_false_alarm_rate(self):
        before, after, truth = make_scene(size=150, seed=7)
        res = rad.irmad(before, after)
        for alpha in (0.01, 0.05, 0.2):
            mask = res.chi2 > stats.chi2.ppf(1 - alpha, 3)
            far = mask[~truth].mean()
            assert abs(far - alpha) < max(0.25 * alpha, 0.004), (alpha, far)
            assert mask[truth].mean() == 1.0
        # chi2 of unchanged pixels follows chi2(3): mean 3, variance 6
        assert res.chi2[~truth].mean() == pytest.approx(3.0, rel=0.05)
        assert res.chi2[~truth].var() == pytest.approx(6.0, rel=0.1)

    def test_single_band_calibrated(self):
        before, after, truth = make_scene(size=150, seed=8, n_bands=1)
        mask = rad.irmad_change(before, after, alpha=0.05)
        assert mask[truth].all()
        assert abs(mask[~truth].mean() - 0.05) < 0.0125

    def test_invariant_to_linear_radiometry(self, scene):
        before, after, _ = scene
        r1 = rad.irmad(before, after)
        r2 = rad.irmad(before, 3.0 * after + 7.0)
        r3 = rad.irmad(0.5 * before - 1.0, after)
        np.testing.assert_allclose(r1.chi2, r2.chi2, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(r1.chi2, r3.chi2, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(r1.mad_variates, r2.mad_variates, atol=1e-6)

    def test_deterministic_signs(self, scene):
        before, after, _ = scene
        r1 = rad.irmad(before, after)
        r2 = rad.irmad(before.copy(), after.copy())
        np.testing.assert_array_equal(r1.mad_variates, r2.mad_variates)

    def test_plain_mad_with_one_iteration(self, scene):
        before, after, _ = scene
        res = rad.irmad(before, after, max_iter=1)
        assert res.n_iter == 1 and not res.converged

    def test_chunked_statistics_match(self, scene, monkeypatch):
        before, after, _ = scene
        ref = rad.irmad(before, after)
        monkeypatch.setattr(rad, "_CHUNK", 777)
        res = rad.irmad(before, after)
        np.testing.assert_allclose(res.chi2, ref.chi2, rtol=1e-8)
        assert res.n_iter == ref.n_iter

    def test_nan_handling(self, scene):
        before, after, truth = scene
        before, after = before.copy(), after.copy().astype(np.float32)
        before[2, 0, 0] = np.nan
        after[0, 1, 1] = -1.0
        res = rad.irmad(before, after, nodata=-1.0)
        for arr in (res.chi2, res.no_change_prob, *res.mad_variates):
            assert np.isnan(arr[0, 0]) and np.isnan(arr[1, 1])
            assert np.isnan(arr).sum() == 2
        assert res.chi2.dtype == np.float64  # float64 before wins
        mask = rad.irmad_change(before, after, nodata=-1.0)
        assert not mask[0, 0] and not mask[1, 1]
        assert mask[truth].all()

    def test_float32_output(self, scene):
        before, after, _ = scene
        res = rad.irmad(before.astype(np.float32), after.astype(np.float32))
        assert res.chi2.dtype == np.float32 and res.mad_variates.dtype == np.float32

    def test_valid_restricts_statistics(self, scene):
        before, after, truth = scene
        res = rad.irmad(before, after, valid=~truth)
        assert np.isfinite(res.chi2).all()
        assert (res.chi2[truth] > stats.chi2.ppf(0.99, 3)).all()

    def test_irmad_change(self, scene):
        before, after, truth = scene
        mask = rad.irmad_change(before, after, alpha=0.001)
        assert mask.dtype == bool and mask.shape == truth.shape
        assert mask[truth].all()
        assert mask[~truth].mean() < 0.005

    def test_errors(self, scene):
        before, after, _ = scene
        with pytest.raises(ValueError, match="same shape"):
            rad.irmad(before, after[:2])
        with pytest.raises(ValueError, match="max_iter"):
            rad.irmad(before, after, max_iter=0)
        with pytest.raises(ValueError, match="tol"):
            rad.irmad(before, after, tol=0)
        with pytest.raises(ValueError, match="alpha"):
            rad.irmad_change(before, after, alpha=1.0)
        with pytest.raises(ValueError, match="at least 10 valid pixels"):
            rad.irmad(before[:, :3, :3], after[:, :3, :3])
        dup = before.copy()
        dup[2] = 2.0 * dup[0]
        with pytest.raises(ValueError, match="before_stack is singular"):
            rad.irmad(dup, after)
        with pytest.raises(ValueError, match="after_stack is singular"):
            rad.irmad(before, np.ones_like(after))
        with pytest.raises(ValueError, match="exact linear function"):
            rad.irmad(before, 2.0 * before + 1.0)
        with pytest.raises(TypeError, match="numpy array"):
            rad.irmad(before.tolist(), after)


def test_consistency_factor_matches_simulation():
    rng = np.random.default_rng(0)
    for p in (1, 2, 4):
        z = rng.chisquare(p, size=400_000)
        w = stats.chi2.sf(z, p)
        assert rad._consistency_factor(p) == pytest.approx((z * w).sum() / (p * w.sum()), rel=0.01)


def test_public_api():
    for name in rad.__all__:
        assert hasattr(rad, name)
