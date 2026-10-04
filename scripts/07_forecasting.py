"""Step 7 (H9): can volatility clustering improve a simple volatility forecast? (Cont's open question)

Out-of-sample walk-forward (monthly refit, expanding window, no tuned hyper-parameters).
Target: r^2 for daily assets (QLIKE/MSE are robust to this noisy proxy, Patton 2011); for BTC also
5-minute realized variance. Statistical significance: Diebold-Mariano (HLN) with Holm.
Economic significance: volatility-targeting overlay on the S&P 500, net of costs.
"""
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volclust import stats
from volclust.data import RAW, ROOT, load_config, load_returns
from volclust.forecast import MODELS, walk_forward
from volclust.measures import aggregate, bar_returns, realized_variance
from volclust.plotting import save, save_table

cfg = load_config()
F = cfg["forecast"]
R = load_returns(cfg)
CACHE = ROOT / "data" / "processed"
CACHE.mkdir(parents=True, exist_ok=True)
CLUSTER_MODELS = [m for m in MODELS if m != "HIST"]


def oos_start(aid):
    return F["btc_oos_start"] if aid.startswith("BTC") else F["oos_start"]


def get_forecasts(aid, r, y, h=1, refit=F["refit"]):
    path = CACHE / f"fc_{aid}_h{h}_{refit}.csv"
    if path.exists() and "--refresh" not in sys.argv:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    fc = walk_forward(r, y, oos_start(aid), refit=refit, h=h, har_lags=tuple(F["har_lags"]), lam=F["ewma_lambda"])
    fc.to_csv(path)
    print(f"  {aid} h={h} {refit}: {len(fc)} OOS days, HAR floor hits {fc.attrs.get('har_floor_hits')}")
    return fc


def evaluate(fc, h, label):
    """Average losses (relative to HIST) and DM tests of each model vs HIST and HAR vs GARCH/EWMA."""
    y = fc["target"]
    loss = {"QLIKE": {m: stats.qlike(y, fc[m]) for m in MODELS},
            "MSE": {m: (y - fc[m]) ** 2 for m in MODELS}}
    rows = []
    for lname, L in loss.items():
        for m in CLUSTER_MODELS:
            dm, p = stats.diebold_mariano(L[m], L["HIST"], h=h, hac_lags=max(F["dm_hac_lags"], h - 1))
            rows.append({"case": label, "loss": lname, "comparison": f"{m} vs HIST", "mean loss diff": (L[m] - L["HIST"]).mean(),
                         "rel. loss": L[m].mean() / L["HIST"].mean() if lname == "MSE" else np.nan, "DM": dm, "p": p})
        for base in ["GARCH", "EWMA"]:
            dm, p = stats.diebold_mariano(L["HAR"], L[base], h=h, hac_lags=max(F["dm_hac_lags"], h - 1))
            rows.append({"case": label, "loss": lname, "comparison": f"HAR vs {base}", "mean loss diff": (L["HAR"] - L[base]).mean(),
                         "rel. loss": L["HAR"].mean() / L[base].mean() if lname == "MSE" else np.nan, "DM": dm, "p": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- daily assets, primary spec
results, fcs = [], {}
for aid in F["assets"]:
    r = R[aid]
    fcs[aid] = get_forecasts(aid, r, r ** 2)
    results.append(evaluate(fcs[aid], 1, aid))

# ---------------------------------------------------------------- BTC realized variance
ic = cfg["intraday"]
px = pd.read_csv(RAW / f"{ic['binance_symbol']}_{ic['binance_interval']}.csv.gz", index_col=0)
px.index = pd.to_datetime(px.index, utc=True)
r5 = bar_returns(px["Close"].astype(float), 5)
rv = realized_variance(r5, 288)
r_btc = aggregate(r5, 5, 1440)
r_btc.index = r_btc.index.tz_localize(None)
fcs["BTC-RV"] = get_forecasts("BTC-RV", r_btc, rv)
results.append(evaluate(fcs["BTC-RV"], 1, "BTC-RV"))

res = pd.concat(results, ignore_index=True)
prim = res["loss"] == "QLIKE"
res["p Holm"] = np.nan
H9B = ["HAR vs GARCH", "HAR vs EWMA"]
for comp_family in [res["comparison"].str.endswith("vs HIST"), res["comparison"].isin(H9B)]:
    for lname in ["QLIKE", "MSE"]:
        mask = comp_family & (res["loss"] == lname)
        res.loc[mask, "p Holm"] = stats.holm(res.loc[mask, "p"])
save_table(res.set_index("case"), "h9_dm_all", floatfmt="{:.4g}")

# Compact summary: QLIKE difference vs HIST per model per asset (negative = better than baseline)
q = res[prim & res["comparison"].str.endswith("vs HIST")].pivot(index="case", columns="comparison", values="mean loss diff")
q.columns = [c.replace(" vs HIST", "") for c in q.columns]
sig = res[prim & res["comparison"].str.endswith("vs HIST")].pivot(index="case", columns="comparison", values="p Holm")
sig.columns = q.columns
q["best"] = q.idxmin(axis=1)
save_table(q, "h9_qlike_vs_hist")
save_table(sig, "h9_qlike_vs_hist_pholm", floatfmt="{:.2g}")
har = res[prim & res["comparison"].isin(H9B)].pivot(index="case", columns="comparison", values=["mean loss diff", "p Holm"])
save_table(har, "h9_har_vs_short_memory", floatfmt="{:.3g}")

# ---------------------------------------------------------------- robustness: h=5, yearly refit
rob = []
for aid in ["SPX", "USDJPY", "UST10", "BRENT", "AAPL"]:
    r = R[aid]
    for h, refit in [(5, F["refit"]), (1, "YS")]:
        fc = get_forecasts(aid, r, r ** 2, h=h, refit=refit)
        e = evaluate(fc, h, f"{aid} h={h} {refit}")
        rob.append(e[e["loss"] == "QLIKE"])
rob = pd.concat(rob, ignore_index=True)
rob["p Holm"] = stats.holm(rob["p"])
save_table(rob.set_index("case"), "h9_robustness", floatfmt="{:.4g}")

# ---------------------------------------------------------------- economic significance: vol targeting
V = cfg["voltarget"]
fc = fcs[V["asset"]]
r = R[V["asset"]].loc[fc.index]
simple = np.expm1(r)
tgt_d = V["target_vol"] / np.sqrt(252)


def overlay(sig2, cost_bp, delay=0):
    """Weight decided at close t-1 from the forecast for day t (already ex-ante); delay adds lag."""
    w = np.minimum(tgt_d / np.sqrt(sig2), V["max_leverage"]).shift(delay)
    turn = w.diff().abs()
    ret = (w * simple - cost_bp / 1e4 * turn).dropna()
    roll = ret.rolling(63).std() * np.sqrt(252)
    eq = (1 + ret).cumprod()
    return {"ann. vol": ret.std() * np.sqrt(252), "|vol - target|": abs(ret.std() * np.sqrt(252) - V["target_vol"]),
            "RMSE rolling-63d vol vs target": np.sqrt(np.mean((roll.dropna() - V["target_vol"]) ** 2)),
            "ann. return": ret.mean() * 252, "Sharpe (rf=0)": ret.mean() / ret.std() * np.sqrt(252),
            "max drawdown": (eq / eq.cummax() - 1).min(), "ann. turnover": turn.mean() * 252}


eco = []
bh = simple.copy()
eco.append({"model": "Buy & hold (unscaled)", "cost bp": 0, "delay": 0,
            "ann. vol": bh.std() * np.sqrt(252), "ann. return": bh.mean() * 252,
            "Sharpe (rf=0)": bh.mean() / bh.std() * np.sqrt(252),
            "max drawdown": ((1 + bh).cumprod() / (1 + bh).cumprod().cummax() - 1).min()})
for m in MODELS:
    for cost in V["cost_bp"]:
        for delay in [0, 1]:
            eco.append({"model": m, "cost bp": cost, "delay": delay, **overlay(fc[m], cost, delay)})
eco = pd.DataFrame(eco).set_index("model")
save_table(eco, "h9_voltarget")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
qq = q.drop(columns="best")
im = ax[0].imshow(qq.values, cmap="RdBu", vmin=-np.nanmax(abs(qq.values)), vmax=np.nanmax(abs(qq.values)), aspect="auto")
ax[0].set_xticks(range(qq.shape[1]), qq.columns)
ax[0].set_yticks(range(qq.shape[0]), qq.index)
for i in range(qq.shape[0]):
    for j in range(qq.shape[1]):
        star = "*" if sig.values[i, j] < 0.05 else ""
        ax[0].text(j, i, f"{qq.values[i, j]:.2f}{star}", ha="center", va="center", fontsize=6.5)
ax[0].set_title("OOS QLIKE minus HIST baseline (negative = better; * Holm p<0.05)")
plt.colorbar(im, ax=ax[0], shrink=0.8)
sub = fc.loc["2019-06":"2021-06"]
ax[1].plot(sub.index, np.sqrt(252 * sub["target"]), lw=0.3, color="lightgrey", label="|r|·√252 (proxy)")
for m, col in [("HIST", "k"), ("EWMA", "C0"), ("GARCH", "C1"), ("HAR", "C2")]:
    ax[1].plot(sub.index, np.sqrt(252 * sub[m]), lw=1, color=col, label=m)
ax[1].set(ylim=(0, 1.0), ylabel="annualised vol", title="S&P 500 one-day-ahead vol forecasts around COVID-19")
ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax[1].legend(fontsize=7)
save(fig, "fig10_forecasting")

print(q.round(4).to_string())
print(sig.round(4).to_string())
print(har.round(4).to_string())
print(rob.round(4).to_string())
print(eco.round(4).to_string())
