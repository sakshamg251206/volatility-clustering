"""Step 6 (H8): does clustering strength change across market regimes?

Regimes are calendar windows, not volatility-sorted samples (sorting on |r| would truncate the
series and bias its ACF). Descriptive: rolling 5-year S20 and fixed sub-periods, for raw |r| and
for |r| standardised by trailing one-year vol (removes the regime's volatility level).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volclust import stats
from volclust.data import load_config, load_returns
from volclust.measures import standardized_abs
from volclust.plotting import CLASS_COLORS, save, save_table

cfg = load_config()
C = cfg["clustering"]
R = load_returns(cfg)
meta = {a["id"]: a for a in cfg["assets"]}
s20 = lambda a: stats.acf(a, C["s_lags"])[1:].sum()
WIN, STEP = 5 * 252, 21
PERIODS = [("1990-1999", "1990", "1999"), ("2000-2009", "2000", "2009"),
           ("2010-2019", "2010", "2019"), ("2020-2025", "2020", "2025")]

rows, rolling = [], {}
for aid, r in R.items():
    a, z = r.abs(), standardized_abs(r, C["trailing_vol_window"])
    ends = range(WIN, len(a), STEP)
    rolling[aid] = pd.DataFrame({
        "S20": [s20(a.values[e - WIN:e]) for e in ends],
        "ann_vol": [r.values[e - WIN:e].std() * np.sqrt(252) for e in ends],
    }, index=a.index[list(ends)])
    row = {"asset": aid}
    for lab, s, e in PERIODS:
        seg, segz = a.loc[s:e], z.loc[s:e]
        ok = len(seg) >= 750
        row[f"S20 {lab}"] = s20(seg.values) if ok else np.nan
        row[f"S20 std {lab}"] = s20(segz.values) if len(segz) >= 750 else np.nan
    roll = rolling[aid]
    row["corr(rolling S20, rolling vol)"] = roll["S20"].corr(roll["ann_vol"])
    row["rolling S20 min"], row["rolling S20 max"] = roll["S20"].min(), roll["S20"].max()
    rows.append(row)

t = pd.DataFrame(rows).set_index("asset")
save_table(t, "h8_regimes")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
for aid, roll in rolling.items():
    ax[0].plot(roll.index, roll["S20"], lw=0.9, color=CLASS_COLORS[meta[aid]["cls"]], alpha=0.8)
    ax[1].scatter(roll["ann_vol"], roll["S20"], s=2, color=CLASS_COLORS[meta[aid]["cls"]], alpha=0.5)
for cls, col in CLASS_COLORS.items():
    ax[0].plot([], [], color=col, label=cls)
ax[0].set(ylabel="S20 over trailing 5 years", title="Rolling strength of volatility clustering (window end date)")
ax[0].legend(fontsize=7, ncol=3)
ax[1].set(xlabel="annualised vol in window", ylabel="S20", title="Clustering vs volatility level", xscale="log")
save(fig, "fig09_regimes")
print(t.round(2).to_string())
