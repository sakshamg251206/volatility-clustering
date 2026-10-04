"""Leakage and correctness tests for forecasting and measures."""
import numpy as np
import pandas as pd
import pytest

from volclust import forecast as fc
from volclust import measures as ms


def _garch_series(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = 1e-4
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = 2e-6 + 0.08 * r[t] ** 2 + 0.9 * s2
    return pd.Series(r, index=pd.bdate_range("2000-01-03", periods=n))


def test_garch_filter_matches_loop():
    r = _garch_series(500).values
    p = {"omega": 1e-6, "alpha": 0.05, "gamma": 0.1, "beta": 0.85}
    s2 = fc.garch_filter(r, p, 2e-4)
    ref = [2e-4]
    for t in range(1, len(r)):
        ref.append(p["omega"] + (p["alpha"] + p["gamma"] * (r[t - 1] < 0)) * r[t - 1] ** 2 + p["beta"] * ref[-1])
    np.testing.assert_allclose(s2, ref, rtol=1e-10)


def test_ewma_is_ex_ante():
    r = pd.Series([0.0] * 10 + [1.0] + [0.0] * 5, index=pd.bdate_range("2020-01-01", periods=16))
    v = ms.ewma_var(r, 0.94)
    assert v.iloc[10] == 0.0          # shock at t=10 not visible on day 10
    assert v.iloc[11] == pytest.approx(0.06)


@pytest.mark.parametrize("h", [1, 5])
def test_walk_forward_has_no_lookahead(h):
    """Changing data after date t0 must not change any forecast dated <= t0."""
    r = _garch_series()
    oos = r.index[2000].strftime("%Y-%m-%d")
    t0 = r.index[2300]
    base = fc.walk_forward(r, r ** 2, oos, h=h)
    r2 = r.copy()
    r2[r2.index > t0] *= 5.0
    pert = fc.walk_forward(r2, r2 ** 2, oos, h=h)
    models = fc.MODELS
    pd.testing.assert_frame_equal(base.loc[:t0, models], pert.loc[:t0, models])
    assert not np.allclose(base.loc[t0:, models].iloc[5:].values, pert.loc[t0:, models].iloc[5:].values)


def test_walk_forward_targets_and_garch_sane():
    r = _garch_series(4000)
    out = fc.walk_forward(r, r ** 2, r.index[3000].strftime("%Y-%m-%d"))
    assert out[fc.MODELS].notna().all().all()
    # true process is GARCH: GARCH should beat the no-clustering baseline on QLIKE
    q = lambda c: (np.log(out[c]) + out["target"] / out[c]).mean()
    assert q("GARCH") < q("HIST")


def test_aggregate_drops_incomplete_bins():
    idx = pd.date_range("2024-01-01", periods=12, freq="5min")
    r = pd.Series(1.0, index=idx)
    r.iloc[3] = np.nan
    agg = ms.aggregate(r, 5, 15)
    assert len(agg) == 3 and (agg == 3.0).all()  # bin containing the NaN is dropped


def test_standardized_abs_uses_past_only():
    r = _garch_series(600)
    s = ms.standardized_abs(r, 252)
    t = s.index[10]
    loc = r.index.get_loc(t)
    assert s[t] == pytest.approx(abs(r.iloc[loc]) / r.iloc[loc - 252:loc].std())
