"""Step 3 (H5): does clustering strength differ across asset classes?

Per asset: S20 with stationary-bootstrap CI on (a) each asset's full sample and (b) a common window
in which every asset trades (added robustness: crises inside the window drive S20, so full samples
of different lengths are not directly comparable). Plus GARCH(1,1) persistence and half-life.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volclust import stats
from volclust.data import load_config, load_returns
from volclust.forecast import fit_garch
from volclust.plotting import CLASS_COLORS, save, save_table

cfg = load_config()
C = cfg["clustering"]
rng = np.random.default_rng(cfg["study"]["seed"] + 3)
R = load_returns(cfg)
meta = {a["id"]: a for a in cfg["assets"]}
common_start = max(r.index.min() for r in R.values())
s20 = lambda a: stats.acf(a, C["s_lags"])[1:].sum()

rows = []
for aid, r in R.items():
    row = {"asset": aid, "class": meta[aid]["cls"]}
    for tag, x in [("full", r), ("common", r.loc[common_start:])]:
        a = x.abs().values
        b = stats.bootstrap(a, s20, C["n_boot"], C["boot_block"], rng)
        row[f"S20 {tag}"] = s20(a)
        row[f"S20 {tag} lo"], row[f"S20 {tag} hi"] = np.quantile(b, [0.025, 0.975])
        row[f"rankS20 {tag}"] = stats.rank_acf(a, C["s_lags"])[1:].sum()
    p = fit_garch(r.loc[common_start:])
    row["GARCH a+b (common)"] = p["alpha"] + p["beta"]
    row["half-life days (common)"] = stats.half_life(p["alpha"] + p["beta"])
    row["GARCH alpha (common)"] = p["alpha"]
    rows.append(row)

t = pd.DataFrame(rows).set_index("asset")
save_table(t, "h5_cross_asset")
by_class = t.groupby("class")[["S20 full", "S20 common", "rankS20 common", "GARCH a+b (common)", "half-life days (common)"]].mean()
by_class["n assets"] = t.groupby("class").size()
save_table(by_class, "h5_by_class")

# Forest plot
fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
order = t.sort_values(["class", "S20 common"]).index
y = np.arange(len(order))
for axi, tag in zip(ax, ["full", "common"]):
    for yi, aid in zip(y, order):
        col = CLASS_COLORS[t.loc[aid, "class"]]
        axi.errorbar(t.loc[aid, f"S20 {tag}"], yi, xerr=[[t.loc[aid, f"S20 {tag}"] - t.loc[aid, f"S20 {tag} lo"]],
                     [t.loc[aid, f"S20 {tag} hi"] - t.loc[aid, f"S20 {tag}"]]], fmt="o", color=col, ms=4, capsize=2)
    axi.set_xlabel("S20 = Σ ACF of |r|, lags 1–20")
    axi.set_title("Own full sample" if tag == "full" else f"Common window {common_start.date()}–2025")
ax[0].set_yticks(y, order)
for cls, col in CLASS_COLORS.items():
    ax[1].plot([], [], "o", color=col, label=cls)
ax[1].legend(fontsize=7, loc="lower right")
fig.suptitle("Strength of volatility clustering by asset (95% stationary-bootstrap CIs)", fontsize=10)
save(fig, "fig06_cross_asset")
print(t.round(3).to_string())
print(by_class.round(3).to_string())
