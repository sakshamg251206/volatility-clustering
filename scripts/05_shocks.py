"""Step 5 (H7): how persistent is volatility after large shocks?

Shock at t: |r_t| > threshold * sigma_t, with sigma_t the ex-ante EWMA vol (data <= t-1).
Declustered: no other shock in the previous `decluster` days. Response: a_{t+k}/sigma_t, k=0..H,
i.e. post-shock |returns| in units of pre-shock volatility. Excess = response minus the same ratio
averaged over all dates (unconditional baseline). The identical pipeline is applied to paths
simulated from each asset's fitted GARCH(1,1)-t: the benchmark for "slower than GARCH implies".
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from volclust import stats
from volclust.data import load_config, load_returns
from volclust.forecast import fit_garch, simulate_garch_t
from volclust.measures import ewma_var
from volclust.plotting import save, save_table

cfg = load_config()
S = cfg["shocks"]
H = S["horizon"]
rng = np.random.default_rng(cfg["study"]["seed"] + 5)
R = load_returns(cfg)
N_SIM_PATHS = 20


def events(r: pd.Series, thr: float, decl: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Event response matrix (events x k=0..H, plus 'sign') and the unconditional baseline curve."""
    sig = np.sqrt(ewma_var(r, S["ewma_lambda"])).values
    a, rv = np.abs(r.values), r.values
    n, burn = len(a), 250
    is_shock = np.zeros(n, bool)
    is_shock[burn:] = a[burn:] > thr * sig[burn:]
    keep = [t for t in np.flatnonzero(is_shock)
            if t + H < n and not is_shock[max(0, t - decl):t].any()]
    resp = np.array([a[t:t + H + 1] / sig[t] for t in keep]).reshape(-1, H + 1)
    ts = np.arange(burn, n - H)
    base = np.array([np.mean(a[ts + k] / sig[ts]) for k in range(H + 1)])
    df = pd.DataFrame(resp)
    df["sign"] = np.sign(rv[keep])
    return df, base


def exp_half_life(excess: np.ndarray) -> float:
    """Half-life (days) of an exponential c*exp(-k/tau) fitted by NLS to excess response, k=1..H."""
    k = np.arange(1, len(excess))
    try:
        (c, tau), _ = curve_fit(lambda k, c, tau: c * np.exp(-k / tau), k, excess[1:], p0=[excess[1], 10], maxfev=10_000)
        return tau * np.log(2) if tau > 0 else np.nan
    except RuntimeError:
        return np.nan


def pooled(thr, decl, sim=False):
    """Pool events across assets: excess-response matrix of all events (each minus own-asset baseline)."""
    out = []
    for aid, r in R.items():
        if sim:
            p = fit_garch(r, dist="t")
            series = [pd.Series(x) for x in simulate_garch_t(p, len(r), rng, n_paths=N_SIM_PATHS)]
        else:
            series = [r]
        for x in series:
            ev, base = events(x, thr, decl)
            ex = ev[list(range(H + 1))].values - base
            out.append(pd.DataFrame(ex).assign(sign=ev["sign"].values, asset=aid))
    return pd.concat(out, ignore_index=True)


def summarise(ev: pd.DataFrame, label: str, n_boot=1000) -> dict:
    X = ev[list(range(H + 1))].values
    m = X.mean(0)
    boot_idx = rng.integers(0, len(X), (n_boot, len(X)))
    hl_boot = [exp_half_life(X[i].mean(0)) for i in boot_idx[:300]]
    row = {"sample": label, "n events": len(X), "excess k=1": m[1], "excess k=5": m[5],
           "excess k=20": m[20], "excess k=60": m[60],
           "excess k=20 CI lo": np.quantile(X[boot_idx, 20].mean(1), 0.025),
           "excess k=20 CI hi": np.quantile(X[boot_idx, 20].mean(1), 0.975),
           "half-life (exp fit)": exp_half_life(m),
           "half-life CI lo": np.nanquantile(hl_boot, 0.025), "half-life CI hi": np.nanquantile(hl_boot, 0.975)}
    return row, m


rows, curves = [], {}
emp = pooled(S["threshold"], S["decluster"])
sim = pooled(S["threshold"], S["decluster"], sim=True)
eq_ids = {a["id"] for a in cfg["assets"] if a["cls"] in ("Equity index", "Single stock")}
emp_eq = emp[emp.asset.isin(eq_ids)]
for label, ev in [("empirical, all", emp), ("equities, negative", emp_eq[emp_eq.sign < 0]),
                  ("equities, positive", emp_eq[emp_eq.sign > 0]), ("GARCH-t simulated, all", sim)]:
    row, m = summarise(ev, label)
    rows.append(row)
    curves[label] = m
# Per-asset-class (empirical)
cls = {a["id"]: a["cls"] for a in cfg["assets"]}
for c in sorted(set(cls.values())):
    sub = emp[emp.asset.map(cls) == c]
    if len(sub) >= 10:
        rows.append(summarise(sub, f"empirical, {c}")[0])
# Robustness: thresholds and declustering window
for thr in S["thresholds_robust"]:
    rows.append(summarise(pooled(thr, S["decluster"]), f"empirical, threshold {thr}σ")[0])
rows.append(summarise(pooled(S["threshold"], S["decluster_robust"]), f"empirical, decluster {S['decluster_robust']}d")[0])

# Asymmetry test: mean excess over k=1..20, negative minus positive shocks, bootstrap over events.
# Primary: equities only -- for FX and yields the sign of a move is a quoting convention.
eq = {a["id"] for a in cfg["assets"] if a["cls"] in ("Equity index", "Single stock")}


def asymmetry(ev):
    neg = ev[ev.sign < 0][list(range(1, 21))].values.mean(1)
    pos = ev[ev.sign > 0][list(range(1, 21))].values.mean(1)
    d = np.array([rng.choice(neg, len(neg)).mean() - rng.choice(pos, len(pos)).mean() for _ in range(5000)])
    return {"n neg": len(neg), "n pos": len(pos), "mean excess k=1..20, neg - pos": neg.mean() - pos.mean(),
            "CI lo": np.quantile(d, 0.025), "CI hi": np.quantile(d, 0.975),
            "p (two-sided, bootstrap)": max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1 / len(d))}


asym = pd.DataFrame([asymmetry(emp[emp.asset.isin(eq)]), asymmetry(emp)], index=["equities (primary)", "all assets"])
t = pd.DataFrame(rows).set_index("sample")
save_table(t, "h7_shocks")
save_table(asym, "h7_asymmetry")

# Leverage function L(tau) = corr(r_t, r_{t+tau}^2), Cont eq. (20), S&P 500
r = R["SPX"].values
taus = np.arange(-20, 21)
lev = [np.corrcoef(r[max(0, -k):len(r) - max(0, k)], r[max(0, k):len(r) - max(0, -k)] ** 2)[0, 1] for k in taus]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
k = np.arange(H + 1)
for label, ls, col in [("empirical, all", "-", "k"), ("equities, negative", "-", "C3"),
                       ("equities, positive", "-", "C0"), ("GARCH-t simulated, all", "--", "grey")]:
    ax[0].plot(k[1:], curves[label][1:], ls, color=col, label=label)
ax[0].axhline(0, c="grey", lw=0.5)
ax[0].set(xlabel="days after shock k", ylabel="excess |r_{t+k}| / σ_t (pre-shock)",
          title=f"Post-shock excess volatility (|r| > {S['threshold']}σ, pooled)")
ax[0].legend(fontsize=7)
ax[1].bar(taus, lev, color=["C3" if x > 0 else "grey" for x in taus])
ax[1].set(xlabel="τ (days)", ylabel="corr(r_t, r²_{t+τ})", title="S&P 500 leverage function L(τ)")
save(fig, "fig08_shocks")
print(t.round(3).to_string())
print(asym.round(4).to_string())
