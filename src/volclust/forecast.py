"""Variance forecasting models and a leakage-free walk-forward engine.

Timing: the forecast stored at date t is the forecast of the variance of day t (or days t..t+h-1)
made at the close of day t-1, i.e. it may only use data with index < t.
Parameters are re-estimated at each refit date d using data with index < d only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.signal import lfilter

from volclust.measures import ewma_var

MODELS = ["HIST", "ROLL252", "EWMA", "GARCH", "GJR", "HAR"]


# ---------------------------------------------------------------- GARCH family
def fit_garch(r: pd.Series, gjr: bool = False, dist: str = "normal") -> dict:
    """(GJR-)GARCH(1,1), zero mean, by (Q)MLE via `arch`. Returns are scaled x100 for numerical
    stability and omega is scaled back, so parameters are in units of r.
    Normal QMLE is consistent for the variance parameters even if returns are not Gaussian."""
    am = arch_model(100 * r.dropna().values, mean="Zero", vol="GARCH", p=1, o=int(gjr), q=1, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)
    p = res.params
    out = {"omega": p["omega"] / 1e4, "alpha": p["alpha[1]"], "beta": p["beta[1]"],
           "gamma": p["gamma[1]"] if gjr else 0.0}
    if dist == "t":
        out["nu"] = p["nu"]
    out["persistence"] = out["alpha"] + out["beta"] + out["gamma"] / 2
    return out


def garch_filter(r: np.ndarray, p: dict, s2_0: float) -> np.ndarray:
    """One-step-ahead conditional variances: s2[t] = omega + (alpha + gamma*1[r_{t-1}<0]) r_{t-1}^2
    + beta s2[t-1], with s2[0] = s2_0. s2[t] uses r up to t-1 only. Linear recursion -> lfilter."""
    r = np.asarray(r, float)
    shock = (p["alpha"] + p["gamma"] * (r < 0)) * r ** 2
    u = p["omega"] + np.concatenate([[0.0], shock[:-1]])
    u[0] = s2_0
    # s2[t] - beta s2[t-1] = u[t]  (t>=1); s2[0] = s2_0
    return lfilter([1.0], [1.0, -p["beta"]], u)


def simulate_garch_t(p: dict, n: int, rng, n_paths: int = 1, burn: int = 2000) -> np.ndarray:
    """Simulate GARCH(1,1) paths with unit-variance Student-t innovations (params from
    fit_garch(dist='t')). Vectorised across paths; returns shape (n_paths, n)."""
    nu = p["nu"]
    z = rng.standard_t(nu, (n + burn, n_paths)) * np.sqrt((nu - 2) / nu)
    r = np.empty((n + burn, n_paths))
    s2 = np.full(n_paths, p["omega"] / max(1 - p["alpha"] - p["beta"], 1e-6))
    for t in range(n + burn):
        r[t] = np.sqrt(s2) * z[t]
        s2 = p["omega"] + p["alpha"] * r[t] ** 2 + p["beta"] * s2
    return r[burn:].T


def _garch_h_step(s2_next: np.ndarray, p: dict, h: int) -> np.ndarray:
    """Average of E[s2_{t+k}], k=0..h-1, given the one-step forecast (mean-reverting GARCH algebra)."""
    if h == 1:
        return s2_next
    phi = min(p["persistence"], 0.9999)
    lr = p["omega"] / (1 - phi)
    w = np.mean(phi ** np.arange(h))
    return lr + w * (s2_next - lr)


# ---------------------------------------------------------------- HAR
def har_features(y: pd.Series, lags=(1, 5, 22)) -> pd.DataFrame:
    """HAR regressors for a forecast of day t: averages of y over the previous 1, 5, 22 days (ex-ante)."""
    return pd.DataFrame({f"m{l}": y.rolling(l).mean().shift(1) for l in lags})


def forward_mean(y: pd.Series, h: int) -> pd.Series:
    """Target for horizon h at date t: mean of y_t..y_{t+h-1}."""
    return y[::-1].rolling(h).mean()[::-1] if h > 1 else y


# ---------------------------------------------------------------- walk-forward
def walk_forward(r: pd.Series, y: pd.Series, oos_start: str, refit: str = "MS", h: int = 1,
                 har_lags=(1, 5, 22), lam: float = 0.94) -> pd.DataFrame:
    """Out-of-sample variance forecasts for every date >= oos_start.

    r : daily returns (GARCH inputs); y : variance proxy (r^2 or RV) used by HIST/ROLL/EWMA/HAR
    and as the evaluation target. Returns DataFrame with the target and one column per model.
    """
    idx = r.index.intersection(y.index)
    r, y = r.loc[idx], y.loc[idx]
    target = forward_mean(y, h)
    out = pd.DataFrame(index=idx[idx >= pd.Timestamp(oos_start)])
    out["target"] = target

    # Fit-free models (already ex-ante by construction)
    out["HIST"] = y.expanding().mean().shift(1)
    out["ROLL252"] = y.rolling(252).mean().shift(1)
    out["EWMA"] = ewma_var(np.sqrt(y), lam)  # ewma_var squares its input -> EWMA of y

    X = har_features(y, har_lags)
    refit_dates = pd.date_range(out.index[0], out.index[-1], freq=refit)
    refit_dates = refit_dates.union([out.index[0]])
    floor_hits = 0
    for name in ["GARCH", "GJR", "HAR"]:
        out[name] = np.nan
    for i, d in enumerate(refit_dates):
        nxt = refit_dates[i + 1] if i + 1 < len(refit_dates) else out.index[-1] + pd.Timedelta(days=1)
        window = (out.index >= d) & (out.index < nxt)
        if not window.any():
            continue
        train = idx < d
        # GARCH / GJR: estimate on r[<d], filter whole history with frozen params
        r_tr = r[train]
        for name, gjr in [("GARCH", False), ("GJR", True)]:
            p = fit_garch(r_tr, gjr=gjr)
            s2 = pd.Series(garch_filter(r.values, p, float(r_tr.iloc[:250].var())), index=idx)
            out.loc[window, name] = _garch_h_step(s2.loc[out.index[window]].values, p, h)
        # HAR: OLS on pairs whose target window ends before d (no overlap with the future)
        tgt_end_ok = pd.Series(idx, index=idx).shift(-(h - 1)) < d
        trn = train & tgt_end_ok.values & X.notna().all(axis=1).values & target.notna().values
        A = np.column_stack([np.ones(trn.sum()), X[trn].values])
        coef, *_ = np.linalg.lstsq(A, target[trn].values, rcond=None)
        Xw = X.loc[out.index[window]].values
        pred = coef[0] + Xw @ coef[1:]
        floor = 0.01 * y[train].mean()
        floor_hits += int((pred < floor).sum())
        out.loc[window, "HAR"] = np.maximum(pred, floor)
    out.attrs["har_floor_hits"] = floor_hits
    return out.dropna(subset=["target"])
