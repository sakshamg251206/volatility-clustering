"""Step 4 (H6): does volatility clustering depend on sampling frequency?

BTCUSDT 5-min bars (Binance, 2019-2025, 24/7) aggregated to 15m, 30m, 1h, 4h, 1d; SPY 5-min (60 days)
and 60-min (730 days) from Yahoo as a short equity cross-check; S&P 500 daily -> weekly -> monthly.
Intraday |r| is divided by its time-of-day mean before computing ACFs (Andersen & Bollerslev 1997).
Lags are converted to calendar hours (BTC) or trading hours (SPY) so frequencies can be overlaid.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volclust import stats
from volclust.data import RAW, load_config, load_returns
from volclust.measures import aggregate, bar_returns, deseasonalize
from volclust.plotting import save, save_table

cfg = load_config()
rng = np.random.default_rng(cfg["study"]["seed"] + 4)
ic = cfg["intraday"]
s20 = lambda a: stats.acf(a, 20)[1:].sum()
N_PERM = 500  # intraday series are long; 500 permutations bound p below at 0.002


def load_intraday(name: str, tz: str) -> pd.Series:
    df = pd.read_csv(RAW / name, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    return df["Close"].astype(float).sort_index()


def summarise(label, r, k_min, deseason, hours_per_day=24):
    """Row of statistics + ACF curve (indexed by lag in hours) for one return series."""
    a = deseasonalize(r) if deseason else r.abs()
    max_lag = int(min(10 * hours_per_day * 60 / k_min, len(a) // 4))
    rho = stats.acf(a, max_lag)
    hours = np.arange(max_lag + 1) * k_min / 60
    S, p = stats.permutation_test(a.values, s20, N_PERM, rng)
    win = lambda lo, hi: np.nanmean(np.where((hours > lo) & (hours <= hi), rho, np.nan)) if ((hours > lo) & (hours <= hi)).any() else np.nan
    row = {"series": label, "bar (min)": k_min, "n": len(a), "deseasonalised": deseason,
           "rho_r(1)": stats.acf(r, 1)[1], "rho_|r|(1 bar)": rho[1], "S20 (bars)": S, "p_perm": p,
           "mean ACF (0,1h]": win(0, 1), "mean ACF (1h,1d]": win(1, hours_per_day),
           "mean ACF (1d,5d]": win(hours_per_day, 5 * hours_per_day), "mean ACF (5d,10d]": win(5 * hours_per_day, 10 * hours_per_day)}
    return row, pd.Series(rho[1:], index=hours[1:])


rows, curves = [], {}
# ---- BTC: 24/7, calendar time
btc5 = bar_returns(load_intraday(f"{ic['binance_symbol']}_{ic['binance_interval']}.csv.gz", "UTC"), 5)
raw_curve = None
for k in ic["frequencies_min"]:
    r = aggregate(btc5, 5, k)
    for des in ([True, False] if k < 1440 else [False]):
        row, curve = summarise(f"BTC {k}m", r, k, des)
        rows.append(row)
        if des or k == 1440:
            curves[f"BTC {k}m" if k < 1440 else "BTC 1d"] = curve
        if k == 5 and not des:
            raw_curve = curve

# ---- SPY: regular session only; overnight returns excluded -> lags in trading hours (6.5 h/day)
for name, k in [("SPY_5m.csv", 5), ("SPY_60m.csv", 60)]:
    px = load_intraday(name, "America/New_York").between_time("09:30", "15:59")
    r = bar_returns(px, k).dropna()
    for des in [True, False]:
        row, curve = summarise(f"SPY {k}m", r, k, des, hours_per_day=6.5)
        rows.append(row)
        if des:
            curves[f"SPY {k}m"] = curve
        elif k == 5:
            spy_raw = curve

# ---- S&P 500: daily -> weekly -> monthly (non-overlapping sums of daily log returns)
spx = load_returns(cfg)["SPX"]
for label, rule, k_min in [("SPX 1d", None, 6.5 * 60), ("SPX 1w", "W-FRI", 5 * 6.5 * 60), ("SPX 1M", "ME", 21 * 6.5 * 60)]:
    r = spx if rule is None else spx.resample(rule).sum()
    a = r.abs()
    S, p = stats.permutation_test(a.values, s20, N_PERM, rng)
    rho = stats.acf(a, min(20, len(a) // 10))
    rows.append({"series": label, "bar (min)": k_min, "n": len(a), "deseasonalised": False,
                 "rho_r(1)": stats.acf(r, 1)[1], "rho_|r|(1 bar)": rho[1], "S20 (bars)": S, "p_perm": p})

t = pd.DataFrame(rows).set_index("series")
t["p_perm Holm"] = stats.holm(t["p_perm"])
save_table(t, "h6_frequency")

# ---- figures
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].plot(raw_curve.index, raw_curve.values, lw=0.6, color="grey", label="raw |r|")
c5 = curves["BTC 5m"]
ax[0].plot(c5.index, c5.values, lw=0.6, color="C0", label="deseasonalised |r|")
ax[0].set(xlim=(0, 96), xlabel="lag (hours)", ylabel="ACF", title="BTC 5-min: intraday periodicity in the ACF of |r|")
ax[0].set_xticks(range(0, 97, 24))
ax[0].legend()
ax[1].plot(spy_raw.index, spy_raw.values, lw=0.6, color="grey", label="raw |r|")
ax[1].plot(curves["SPY 5m"].index, curves["SPY 5m"].values, lw=0.6, color="C0", label="deseasonalised |r|")
ax[1].set(xlim=(0, 26), xlabel="lag (trading hours; 6.5 h = 1 day)", title="SPY 5-min (60 days): intraday periodicity")
ax[1].set_xticks(np.arange(0, 26.1, 6.5))
ax[1].legend()
for lab, c in curves.items():
    ls = "--" if lab.startswith("SPY") else "-"
    ax[2].plot(c.index, c.values, ls, lw=1, label=lab)
ax[2].set(xscale="log", xlabel="lag (hours; SPY: trading hours)", ylabel="ACF of (deseasonalised) |r|",
          title="Clustering across sampling frequencies")
ax[2].legend(fontsize=7, ncol=2)
save(fig, "fig07_frequency")
print(t.round(4).to_string())
