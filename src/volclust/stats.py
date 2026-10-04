"""Statistical toolkit: ACFs, heavy-tail-robust inference, decay fits, long memory, forecast tests.

Design choice (Cont 2001, §5.3): classical ACF bands assume finite fourth moments, which heavy-tailed
returns may lack. So inference here relies on permutation tests (exact under an i.i.d. null),
stationary block bootstrap, and rank-based ACFs, rather than ±1.96/√n bands.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf as _sm_acf


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def acf(x, max_lag: int) -> np.ndarray:
    """Sample ACF at lags 0..max_lag (biased/standard estimator, FFT). NaNs are dropped first."""
    return _sm_acf(_clean(x), nlags=max_lag, fft=True)


def rank_acf(x, max_lag: int) -> np.ndarray:
    """ACF of ranks (Spearman-type). Bounded transform => no moment conditions needed;
    invariant to any increasing transform, so |r| and r^2 give identical rank ACFs."""
    return acf(sps.rankdata(_clean(x)), max_lag)


def power_acf(r, alphas, lags) -> np.ndarray:
    """C_alpha(tau) = corr(|r_t|^alpha, |r_{t+tau}|^alpha) (Cont eq. 16). Shape (len(alphas), len(lags))."""
    a = np.abs(_clean(r))
    m = max(lags)
    return np.array([acf(a ** al, m)[list(lags)] for al in alphas])


def permutation_test(x, stat: Callable[[np.ndarray], float], n_perm: int, rng) -> tuple[float, float]:
    """One-sided permutation test of H0: x is exchangeable (i.i.d.) against stat being large.
    p = (1 + #{perm stat >= observed}) / (1 + n_perm)  -- never exactly zero."""
    x = _clean(x)
    obs = stat(x)
    null = np.array([stat(rng.permutation(x)) for _ in range(n_perm)])
    return obs, (1 + np.sum(null >= obs)) / (1 + n_perm)


def stationary_bootstrap_indices(n: int, mean_block: float, rng) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap: blocks of geometric length (mean `mean_block`)
    starting at uniform random points, wrapping around the end. Vectorised."""
    new = rng.random(n) < 1.0 / mean_block
    new[0] = True
    block_id = np.cumsum(new) - 1
    block_start_t = np.flatnonzero(new)              # time index where each block begins
    starts = rng.integers(0, n, size=len(block_start_t))
    offset = np.arange(n) - block_start_t[block_id]
    return (starts[block_id] + offset) % n


def bootstrap(x, stat: Callable[[np.ndarray], np.ndarray], n_boot: int, mean_block: float, rng) -> np.ndarray:
    """Stationary-bootstrap distribution of stat(x). Returns array (n_boot, ...)."""
    x = _clean(x)
    return np.array([stat(x[stationary_bootstrap_indices(len(x), mean_block, rng)]) for _ in range(n_boot)])


def fit_decay(rho, lags) -> dict:
    """Fit rho(tau) ~ A tau^-beta (power law) and rho(tau) ~ A exp(-tau/tau0) (exponential).
    Both are fitted by OLS in logs on lags where rho > 0; goodness of fit is compared by SSE in levels
    on the same lags, so the two models are judged on identical data."""
    rho, lags = np.asarray(rho, float), np.asarray(lags, float)
    ok = rho > 0
    y, lt, tt = np.log(rho[ok]), np.log(lags[ok]), lags[ok]
    b_pow, a_pow = np.polyfit(lt, y, 1)
    b_exp, a_exp = np.polyfit(tt, y, 1)
    pred_pow = np.exp(a_pow) * tt ** b_pow
    pred_exp = np.exp(a_exp + b_exp * tt)
    # Same two models fitted by nonlinear least squares in levels on *all* lags, so that the
    # estimation criterion and the comparison criterion coincide (log-OLS fits are not SSE-optimal).
    from scipy.optimize import curve_fit
    f_pow = lambda t, A, b: A * t ** -b
    f_exp = lambda t, A, k: A * np.exp(-t / k)
    try:
        pp, _ = curve_fit(f_pow, lags, rho, p0=[np.exp(a_pow), -b_pow], maxfev=20_000)
        pe, _ = curve_fit(f_exp, lags, rho, p0=[np.exp(a_exp), -1 / b_exp if b_exp < 0 else 100.0], maxfev=20_000)
        sse_pow_nls = float(np.sum((rho - f_pow(lags, *pp)) ** 2))
        sse_exp_nls = float(np.sum((rho - f_exp(lags, *pe)) ** 2))
    except RuntimeError:
        sse_pow_nls = sse_exp_nls = np.nan
    return {
        "sse_power_nls": sse_pow_nls, "sse_exp_nls": sse_exp_nls,
        "beta": -b_pow, "A_power": np.exp(a_pow),
        "tau_exp": -1.0 / b_exp if b_exp < 0 else np.inf, "A_exp": np.exp(a_exp),
        "sse_power": float(np.sum((rho[ok] - pred_pow) ** 2)),
        "sse_exp": float(np.sum((rho[ok] - pred_exp) ** 2)),
        "n_pos": int(ok.sum()), "n_lags": len(lags),
    }


def gph(x, m: int | None = None) -> tuple[float, float]:
    """Geweke & Porter-Hudak (1983) log-periodogram estimate of the memory parameter d.
    Regress log I(lambda_j) on -log(4 sin^2(lambda_j/2)), j=1..m, m = floor(n^0.5).
    For a long-memory process rho(tau) ~ tau^(2d-1), i.e. beta = 1 - 2d.
    Asymptotic s.e. = pi / sqrt(24 m)."""
    x = _clean(x)
    n = len(x)
    m = m or int(np.floor(n ** 0.5))
    fx = np.fft.fft(x - x.mean())
    j = np.arange(1, m + 1)
    lam = 2 * np.pi * j / n
    I = np.abs(fx[j]) ** 2 / (2 * np.pi * n)
    reg = -np.log(4 * np.sin(lam / 2) ** 2)
    d = np.polyfit(reg, np.log(I), 1)[0]
    return float(d), float(np.pi / np.sqrt(24 * m))


def holm(pvals) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (controls family-wise error rate)."""
    return multipletests(np.asarray(pvals, float), method="holm")[1]


def qlike(y, h) -> np.ndarray:
    """QLIKE loss  log h + y/h  (Patton 2011): ranking-consistent with a noisy but conditionally
    unbiased variance proxy y. Differs from Patton's normalised form only by a term free of h,
    and stays finite when y = 0."""
    return np.log(h) + np.asarray(y) / np.asarray(h)


def diebold_mariano(loss_a, loss_b, h: int = 1, hac_lags: int | None = None) -> tuple[float, float]:
    """Diebold–Mariano test of equal expected loss with the Harvey–Leybourne–Newbold (1997)
    small-sample correction and a Bartlett (Newey–West) long-run variance.
    Returns (stat, two-sided p). stat < 0  <=>  model A has lower average loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    L = h - 1 if hac_lags is None else hac_lags
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, L + 1):
        lrv += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / n
    dm = d.mean() / np.sqrt(lrv / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * sps.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


def half_life(persistence: float) -> float:
    """Days for a shock to variance to halve when it decays as persistence^k (GARCH: alpha+beta)."""
    return float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else np.inf
