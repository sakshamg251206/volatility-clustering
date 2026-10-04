"""Tests for the statistical toolkit. Each test checks a property with a known answer."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import acf as sm_acf

from volclust import stats


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def simulate_garch(n, omega, alpha, beta, rng, burn=1000):
    z = rng.standard_normal(n + burn)
    r = np.empty(n + burn)
    s2 = omega / (1 - alpha - beta)
    for t in range(n + burn):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r[burn:]


def test_acf_matches_statsmodels(rng):
    x = rng.standard_normal(500)
    np.testing.assert_allclose(stats.acf(x, 10), sm_acf(x, nlags=10, fft=True), atol=1e-12)


def test_acf_handles_nan_by_dropping(rng):
    x = rng.standard_normal(300)
    x_nan = np.concatenate([[np.nan], x])
    np.testing.assert_allclose(stats.acf(x_nan, 5), stats.acf(x, 5))


def test_permutation_iid_not_rejected_and_garch_rejected(rng):
    iid = np.abs(rng.standard_t(4, 3000))
    garch = np.abs(simulate_garch(3000, 0.05, 0.1, 0.85, rng))
    s = lambda a: stats.acf(a, 20)[1:].sum()
    _, p_iid = stats.permutation_test(iid, s, 500, rng)
    _, p_garch = stats.permutation_test(garch, s, 500, rng)
    assert p_iid > 0.05
    assert p_garch < 0.01


def test_permutation_pvalue_never_zero(rng):
    x = np.abs(simulate_garch(2000, 0.05, 0.1, 0.85, rng))
    _, p = stats.permutation_test(x, lambda a: stats.acf(a, 5)[1:].sum(), 99, rng)
    assert p == pytest.approx(1 / 100)


def test_stationary_bootstrap_indices_are_valid_and_blocky(rng):
    idx = stats.stationary_bootstrap_indices(10_000, 50, rng)
    assert idx.min() >= 0 and idx.max() < 10_000
    consecutive = np.mean(np.diff(idx) == 1)
    assert 0.97 < consecutive < 0.99  # new block w.p. 1/50 per step


def test_power_law_fit_recovers_beta():
    lags = np.arange(1, 101)
    rho = 0.3 * lags ** -0.25
    fit = stats.fit_decay(rho, lags)
    assert fit["beta"] == pytest.approx(0.25, abs=1e-9)
    assert fit["sse_power"] < fit["sse_exp"]


def test_exponential_fit_wins_on_exponential_decay():
    lags = np.arange(1, 101)
    rho = 0.3 * np.exp(-lags / 10)
    fit = stats.fit_decay(rho, lags)
    assert fit["tau_exp"] == pytest.approx(10, rel=1e-6)
    assert fit["sse_exp"] < fit["sse_power"]


def test_gph_white_noise_d_near_zero(rng):
    d, se = stats.gph(rng.standard_normal(20_000))
    assert abs(d) < 3 * se


def test_gph_recovers_fractional_d(rng):
    # ARFIMA(0,d,0) via truncated MA(inf) weights psi_k = psi_{k-1} (k-1+d)/k
    d_true, n, K = 0.3, 20_000, 5_000
    psi = np.ones(K)
    for k in range(1, K):
        psi[k] = psi[k - 1] * (k - 1 + d_true) / k
    e = rng.standard_normal(n + K)
    x = np.convolve(e, psi, mode="valid")[:n]
    d, se = stats.gph(x)
    assert abs(d - d_true) < 3 * se + 0.03


def test_dm_equal_losses_not_rejected_and_clear_winner_rejected(rng):
    n = 2000
    la = rng.exponential(1, n)
    lb = la + rng.normal(0, 0.5, n)
    _, p_eq = stats.diebold_mariano(la, lb, h=1, hac_lags=5)
    _, p_win = stats.diebold_mariano(la, la + 0.2 + rng.normal(0, 0.5, n), h=1, hac_lags=5)
    assert p_eq > 0.01
    assert p_win < 1e-6


def test_dm_sign_convention(rng):
    la = rng.exponential(1, 1000)
    stat, _ = stats.diebold_mariano(la, la + 1.0 + rng.normal(0, 0.1, 1000))
    assert stat < 0  # negative => model A has lower loss


def test_holm_matches_manual():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    adj = stats.holm(p)
    # sorted: .01*4=.04, .03*3=.09, .04*2=.08->max .09, .2*1=.2
    np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.2])


def test_qlike_minimised_at_true_variance(rng):
    y = rng.chisquare(1, 200_000) * 2.0  # proxy with E[y]=2
    grid = np.linspace(1, 3, 41)
    losses = [stats.qlike(y, np.full_like(y, h)).mean() for h in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(2.0, abs=0.06)


def test_rank_acf_invariant_to_monotone_transform(rng):
    x = np.abs(simulate_garch(2000, 0.05, 0.1, 0.85, rng))
    np.testing.assert_allclose(stats.rank_acf(x, 10), stats.rank_acf(x ** 2, 10))
