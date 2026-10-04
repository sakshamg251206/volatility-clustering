"""Volatility measures and intraday transformations.

Timing convention used throughout: a quantity indexed t that is labelled "forecast" or "ex-ante"
uses information up to and including t-1 only. Every function below states its timing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_var(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA variance. Value at t uses returns up to t-1 (ex-ante):
    s2_t = lam * s2_{t-1} + (1-lam) * r_{t-1}^2. The first ~60 values are burn-in."""
    return (r ** 2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)


def trailing_vol(r: pd.Series, window: int = 252) -> pd.Series:
    """Std of the previous `window` returns, excluding r_t itself (ex-ante)."""
    return r.rolling(window, min_periods=window).std().shift(1)


def standardized_abs(r: pd.Series, window: int = 252) -> pd.Series:
    """|r_t| / sigma_252(t-1): removes the slowly varying level of volatility using past data only.
    Used to test whether slow ACF decay is driven by level shifts (Mikosch & Starica 2004)."""
    return (r.abs() / trailing_vol(r, window)).dropna()


# ---------------------------------------------------------------- intraday
def bar_returns(close: pd.Series, bar_minutes: int) -> pd.Series:
    """Log returns between consecutive bars; NaN where the previous bar is missing (no gap-spanning)."""
    r = np.log(close).diff()
    gap = close.index.to_series().diff() != pd.Timedelta(minutes=bar_minutes)
    r[gap] = np.nan
    return r


def aggregate(r: pd.Series, base_min: int, k_min: int, session_start: str | None = None) -> pd.Series:
    """Sum base-interval log returns into k-minute bins; bins with any missing base bar are dropped.
    `session_start` (e.g. '09:30') anchors bins to the session open for exchange-traded data."""
    if k_min == base_min:
        return r.dropna()
    offset = pd.Timedelta(session_start + ":00") if session_start else pd.Timedelta(0)
    g = r.resample(f"{k_min}min", offset=offset)
    out = g.sum(min_count=1)
    complete = g.count() == k_min // base_min
    return out[complete]


def deseasonalize(r: pd.Series) -> pd.Series:
    """Divide |r| by its average at the same time of day (Andersen & Bollerslev 1997 style filter).
    Returns the filtered absolute return series. For descriptive ACFs only: the time-of-day profile
    is estimated on the full sample, which would be look-ahead in a forecasting context."""
    a = r.abs()
    tod = a.index.hour * 60 + a.index.minute
    return a / a.groupby(tod).transform("mean")


def realized_variance(r_intraday: pd.Series, bars_per_day: int) -> pd.Series:
    """Daily RV = sum of squared intraday returns per (UTC) calendar day, rescaled by
    bars_per_day / observed_bars to correct days with a few missing bars. Days with < 90% of
    bars are dropped."""
    g = (r_intraday ** 2).groupby(r_intraday.index.floor("D"))
    rv, cnt = g.sum(), g.count()
    rv = rv * bars_per_day / cnt
    rv = rv[cnt >= 0.9 * bars_per_day]
    rv.index = rv.index.tz_localize(None)
    return rv
