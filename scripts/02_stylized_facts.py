"""Step 2: reproduce Cont (2001) volatility-clustering facts on daily data (H1-H4).

Outputs: results/tables/{h1_h2,h3_decay,h4_taylor}.{csv,md}, results/figures/fig01-05*.png,
results/stylized.json (numbers quoted in the report).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from volclust import stats
from volclust.data import ROOT, load_config, load_returns
from volclust.forecast import fit_garch, garch_filter, simulate_garch_t
from volclust.measures import standardized_abs
from volclust.plotting import CLASS_COLORS, save, save_table

cfg = load_config()
C = cfg["clustering"]
rng = np.random.default_rng(cfg["study"]["seed"])
R = load_returns(cfg)
meta = {a["id"]: a for a in cfg["assets"]}
L = C["s_lags"]
s20 = lambda a: stats.acf(a, L)[1:].sum()
lo, hi = C["powerlaw_lags"]
lags_pl = np.arange(lo, hi + 1)


def subsample_mean_acf(x: pd.Series, years: int, max_lag: int) -> np.ndarray:
    """Average ACF over non-overlapping `years`-long calendar blocks (blocks with < 750 obs skipped)."""
    blocks = [g.values for _, g in x.groupby(x.index.year // years) if len(g) >= 750]
    return np.mean([stats.acf(b, max_lag) for b in blocks], axis=0)


def same_pipeline(r: pd.Series) -> dict:
    """The three ACF variants of |r| used in H3c, applied identically to real and simulated data."""
    a = r.abs()
    return {
        "full": stats.acf(a, hi),
        "subsample": subsample_mean_acf(a, C["subsample_years"], hi),
        "standardized": stats.acf(standardized_abs(r, C["trailing_vol_window"]), hi),
    }


rows12, rows3, rows4, curves = [], [], [], {}
for aid, r in R.items():
    print(aid)
    a = r.abs()
    # ---- H1: linear autocorrelation of returns
    acf_r = stats.acf(r, L)
    boot_r1 = stats.bootstrap(r, lambda x: stats.acf(x, 1)[1], C["n_boot"], C["boot_block"], rng)
    # ---- H2: clustering in |r|
    S, p_perm = stats.permutation_test(a, s20, C["n_perm"], rng)
    lb_p = acorr_ljungbox(a, lags=[L])["lb_pvalue"].iloc[0]
    # conditional heavy tails (fact 7): kurtosis before/after GARCH(1,1) filtering
    pg = fit_garch(r)
    z = r / np.sqrt(garch_filter(r.values, pg, float(r.iloc[:250].var())))
    rows12.append({
        "asset": aid, "class": meta[aid]["cls"], "n": len(r),
        "rho_r(1)": acf_r[1], "rho_r(1) CI lo": np.quantile(boot_r1, 0.025), "rho_r(1) CI hi": np.quantile(boot_r1, 0.975),
        "max|rho_r(2..20)|": np.abs(acf_r[2:]).max(),
        "rho_|r|(1)": stats.acf(a, 1)[1], "S20": S, "S20 rank": stats.rank_acf(a, L)[1:].sum(),
        "p_perm": p_perm, "p_LjungBox": lb_p,
        "kurt r": float(pd.Series(r).kurt()), "kurt GARCH resid": float(pd.Series(z).kurt()),
    })

    # ---- H3: decay shape, long memory, non-stationarity
    emp = same_pipeline(r)
    pt = fit_garch(r, dist="t")
    sims = simulate_garch_t(pt, len(r), rng, n_paths=C["garch_sim_paths"])
    sim = {k: [] for k in emp}
    for s in sims:
        for k, v in same_pipeline(pd.Series(s, index=r.index)).items():
            sim[k].append(v)
    env = {k: np.quantile(np.array(v), [0.025, 0.975], axis=0) for k, v in sim.items()}
    long = slice(50, 101)
    d, d_se = stats.gph(a)
    rob_lo, rob_hi = C["powerlaw_lags_robust"]
    f_main = stats.fit_decay(emp["full"][lags_pl], lags_pl)
    f_rob = stats.fit_decay(stats.acf(a, rob_hi)[rob_lo:rob_hi + 1], np.arange(rob_lo, rob_hi + 1))
    f_sub = stats.fit_decay(emp["subsample"][lags_pl], lags_pl)
    f_std = stats.fit_decay(emp["standardized"][lags_pl], lags_pl)
    above = {k: float(np.mean(emp[k][long] > env[k][1][long])) for k in emp}
    power_wins = f_main["sse_power"] < f_main["sse_exp"]  # pre-registered (log-OLS) version
    long_memory_like = power_wins and above["subsample"] > 0.5 and above["standardized"] > 0.5
    rows3.append({
        "asset": aid, "beta[1,100]": f_main["beta"], "beta[5,250]": f_rob["beta"],
        "SSE pow/exp (log-OLS fits)": f_main["sse_power"] / f_main["sse_exp"],
        "SSE pow/exp (NLS fits)": f_main["sse_power_nls"] / f_main["sse_exp_nls"],
        "GPH d": d, "GPH se": d_se, "beta_GPH=1-2d": 1 - 2 * d,
        "beta subsample": f_sub["beta"], "beta standardized": f_std["beta"],
        "S20 subsample": emp["subsample"][1:L + 1].sum(), "S20 standardized": emp["standardized"][1:L + 1].sum(),
        "frac>GARCH env (full)": above["full"], "frac>GARCH env (sub)": above["subsample"],
        "frac>GARCH env (std)": above["standardized"],
        "emp - GARCH median, lags 50-100": float(np.mean(emp["full"][long] - np.median(np.array(sim["full"]), axis=0)[long])),
        "GARCH-t a+b": pt["alpha"] + pt["beta"], "GARCH-t nu": pt["nu"],
        "long-memory-like": long_memory_like,
    })
    curves[aid] = {"emp": emp, "env": env, "fit": f_main}

    # ---- H4: Taylor effect
    alphas, tl = C["alphas"], C["taylor_lags"]
    P = stats.power_acf(r, alphas, tl)
    boot = stats.bootstrap(r, lambda x: stats.power_acf(x, alphas, tl), C["n_boot"] // 2, C["boot_block"], rng)
    i1, i2 = alphas.index(1.0), alphas.index(2.0)
    row = {"asset": aid}
    for j, lag in enumerate(tl):
        am = np.array(alphas)[boot[:, :, j].argmax(axis=1)]
        row[f"argmax a (lag {lag})"] = alphas[int(P[:, j].argmax())]
        row[f"boot share a in [.75,1.25] (lag {lag})"] = float(np.mean((am >= 0.75) & (am <= 1.25)))
    w = lambda k: np.subtract(*np.quantile(boot[:, k, 0], [0.975, 0.025]))
    row["CI width r^2 / |r| (lag 1)"] = w(i2) / w(i1)
    rows4.append(row)
    curves[aid]["taylor"] = P

# ---------------------------------------------------------------- tables
t12 = pd.DataFrame(rows12).set_index("asset")
t12["p_perm Holm"] = stats.holm(t12["p_perm"])
t3 = pd.DataFrame(rows3).set_index("asset")
t4 = pd.DataFrame(rows4).set_index("asset")
save_table(t12, "h1_h2")
save_table(t3, "h3_decay")
save_table(t4, "h4_taylor")

# ---------------------------------------------------------------- figures
# Fig 1: Cont's Figure 1 asset (BMW) on a longer sample, plus S&P 500
fig, ax = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
for axi, k in zip(ax, ["BMW", "SPX"]):
    axi.plot(R[k].index, 100 * R[k], lw=0.4, color="k")
    axi.set_ylabel("daily log return (%)")
    axi.set_title(f"{meta[k]['name']} daily returns")
save(fig, "fig01_returns")

# Fig 2: ACF of returns vs nonlinear transforms (Cont Figs 6-8 analogue), S&P 500
r = R["SPX"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
lags = np.arange(1, 101)
perm_band = np.quantile([stats.acf(rng.permutation(r.values), 100)[1:] for _ in range(500)], [0.025, 0.975], axis=0)
ax[0].bar(lags, stats.acf(r, 100)[1:], color="k", width=0.8)
ax[0].fill_between(lags, *perm_band, color="C1", alpha=0.3, label="95% permutation band (i.i.d.)")
ax[0].axhline(1.96 / np.sqrt(len(r)), ls="--", c="C0", lw=0.8, label="±1.96/√n")
ax[0].axhline(-1.96 / np.sqrt(len(r)), ls="--", c="C0", lw=0.8)
ax[0].set(title="S&P 500: ACF of daily returns r", xlabel="lag (days)", ylim=(-0.12, 0.12))
ax[0].legend(fontsize=7)
nz = r[r != 0]
for lab, x in [("|r|", r.abs()), ("r²", r ** 2), ("rank(|r|)", None), ("log|r|", np.log(nz.abs()))]:
    v = stats.rank_acf(r.abs(), 100) if x is None else stats.acf(x, 100)
    ax[1].plot(lags, v[1:], label=lab, lw=1)
ax[1].set(title="S&P 500: ACF of nonlinear transforms", xlabel="lag (days)")
ax[1].legend()
save(fig, "fig02_acf_spx")

# Fig 3: log-log ACF of |r| with power-law fits, all assets
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for aid, c in curves.items():
    v = c["emp"]["full"][lags_pl]
    ok = v > 0
    ax.loglog(lags_pl[ok], v[ok], ".", ms=2.5, color=CLASS_COLORS[meta[aid]["cls"]], alpha=0.7)
    f = c["fit"]
    ax.loglog(lags_pl, f["A_power"] * lags_pl ** -f["beta"], lw=0.8, color=CLASS_COLORS[meta[aid]["cls"]])
for cls, col in CLASS_COLORS.items():
    ax.plot([], [], color=col, label=cls)
for b, ls in [(0.2, ":"), (0.4, "--")]:
    ax.loglog(lags_pl, 0.3 * lags_pl ** -b, ls=ls, color="grey", lw=1, label=f"slope −{b}")
ax.set(xlabel="lag τ (trading days)", ylabel="ACF of |r|", title="Decay of the ACF of absolute returns (lags 1–100)")
ax.legend(fontsize=7, ncol=2)
save(fig, "fig03_loglog")

# Fig 4: non-stationarity check vs GARCH-t simulation envelope
ids = list(curves)
fig, axs = plt.subplots(4, 4, figsize=(11, 9), sharex=True)
for axi, aid in zip(axs.flat, ids):
    c = curves[aid]
    for k, col in [("full", "k"), ("subsample", "C0"), ("standardized", "C3")]:
        axi.plot(lags_pl, c["emp"][k][lags_pl], color=col, lw=0.9, label=k)
        axi.fill_between(lags_pl, c["env"][k][0][lags_pl], c["env"][k][1][lags_pl], color=col, alpha=0.12)
    axi.set_xscale("log")
    axi.set_title(aid)
    axi.axhline(0, c="grey", lw=0.5)
for axi in axs.flat[len(ids):]:
    axi.axis("off")
axs.flat[0].legend(fontsize=6)
fig.suptitle("ACF of |r|: full sample vs 5-year sub-samples vs trailing-vol standardised\n"
             "(shaded: 95% envelope from GARCH(1,1)-t simulations processed identically)", fontsize=10)
save(fig, "fig04_nonstationarity")

# Fig 5: Taylor effect
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
P = curves["SPX"]["taylor"]
for j, lag in enumerate(C["taylor_lags"]):
    ax[0].plot(C["alphas"], P[:, j], "o-", ms=3, label=f"τ={lag}")
ax[0].set(xlabel="power α", ylabel="corr(|r_t|^α, |r_{t+τ}|^α)", title="S&P 500: Taylor effect")
ax[0].legend()
for aid, c in curves.items():
    v = c["taylor"][:, 0]
    ax[1].plot(C["alphas"], v / v.max(), color=CLASS_COLORS[meta[aid]["cls"]], lw=0.9, alpha=0.8)
ax[1].set(xlabel="power α", ylabel="relative to max", title="All assets, τ=1 (normalised)")
save(fig, "fig05_taylor")

(ROOT / "results" / "stylized.json").write_text(json.dumps({
    "n_assets": len(R), "all_reject_H2_holm": bool((t12["p_perm Holm"] < cfg["clustering"]["fwer"]).all()),
    "n_long_memory_like": int(t3["long-memory-like"].sum()),
}, indent=2))
print(t12.round(3).to_string())
print(t3.round(3).to_string())
print(t4.round(3).to_string())
