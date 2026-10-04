# Volatility Clustering — reproducing and extending Cont (2001)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23143124.svg)](https://doi.org/10.5281/zenodo.23143124)

*Cite as:* Garg, S. (2026). *Volatility Clustering Revisited: A Reproduction and Extension of Cont (2001)*.
Zenodo. https://doi.org/10.5281/zenodo.23143124

Research project testing the volatility-clustering stylized facts of
**Cont, R. (2001), "Empirical properties of asset returns: stylized facts and statistical issues",
*Quantitative Finance* 1(2), 223–236**, on modern public data. It then asks whether clustering
differs across assets, frequencies, shocks and regimes, and whether it improves volatility forecasts.

📄 **[Read the full research report →](report/report.md)**

## Results at a glance

| Question | Answer | Evidence |
|---|---|---|
| Is volatility clustered? | ✅ **Yes, universally** | i.i.d. null rejected for 14/14 assets (exact permutation test) and at every frequency from 5 min to 1 month |
| Power-law decay, β ∈ [0.2, 0.4] (Cont)? | ⚠️ **Inconclusive** | 9/14 assets in range on lags 1–100, only 1/14 on lags 5–250; exponential fits better for 12/14 |
| Taylor effect (\|r\| most predictable)? | ⚠️ Partly | Holds at lags 5–20 days, not at lag 1 |
| Differs by asset class? | ⚠️ Weak | Equities > commodities > FX, but CIs overlap |
| Depends on sampling frequency? | ✅ No, in calendar time | BTC ACFs from 5-min to daily collapse onto one curve |
| Persistence after big shocks? | ✅ Front-loaded, asymmetric | Negative equity shocks ≈ 3× the vol impact of positive ones (p ≈ 0.0002) |
| Does clustering improve forecasts? | ✅ **Yes** | Beats no-clustering baseline in 59/60 out-of-sample tests (2010–2025, QLIKE, Holm) |
| Is it economically useful? | ✅ For risk, ❌ not for Sharpe | Vol-target tracking error 4.8 → 1.2 pp, drawdown −20% → −13%; no Sharpe gain after costs |

### Volatility clustering: returns are uncorrelated, |returns| are not
![ACF of returns vs absolute returns](results/figures/fig02_acf_spx.png)

### Clustering exists at every sampling frequency
![Frequency analysis](results/figures/fig07_frequency.png)

### Is the slow decay "long memory"? The data can't tell it apart from GARCH plus level shifts
![Non-stationarity check](results/figures/fig04_nonstationarity.png)

### After large shocks: volatility jumps, then fades (faster than GARCH predicts); bad news matters more
![Shock event study](results/figures/fig08_shocks.png)

### Out-of-sample forecasting: every clustering model beats the no-clustering baseline
![Forecasting results](results/figures/fig10_forecasting.png)

<details>
<summary><b>More figures</b> (returns, power-law fits, Taylor effect, cross-asset, regimes)</summary>

![](results/figures/fig01_returns.png)
![](results/figures/fig03_loglog.png)
![](results/figures/fig05_taylor.png)
![](results/figures/fig06_cross_asset.png)
![](results/figures/fig09_regimes.png)

</details>

All tables (CSV + Markdown) are in [results/tables/](results/tables/).

## Documents

**Read in this order**
1. [docs/01_paper_review.md](docs/01_paper_review.md): what the paper claims, and what can be reproduced
2. [docs/02_research_design.md](docs/02_research_design.md): pre-registered hypotheses, tests, splits
3. [report/report.md](report/report.md): the final research report (results, robustness, limitations)

## Layout
```
config/study.toml        every parameter (assets, dates, lags, seeds, splits, costs)
src/volclust/
  data.py                download + SHA-256 manifest + cleaning (no forward-fill)
  measures.py            EWMA / trailing vol (ex-ante), intraday aggregation, deseasonalising, RV
  stats.py               ACF, rank ACF, permutation test, stationary bootstrap, decay fits, GPH, DM, Holm, QLIKE
  forecast.py            GARCH/GJR/HAR/EWMA, leakage-free walk-forward engine
  plotting.py            figure style, table export
scripts/01..07_*.py      one script per step / hypothesis block
tests/                   unit tests, including a look-ahead test for the forecaster
results/tables, figures  generated outputs (CSV + markdown tables, PNG figures)
```

## Reproduce
Python 3.13.7 (any ≥ 3.11 should work). Exact package versions are in `requirements.lock`.
```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.lock -e .
./run_all.sh          # tests, download (cached after first run), all analyses, ~8 min
```
Data sources: Yahoo Finance (equities, gold, BTC daily; SPY intraday snapshot), FRED (FX, 10Y
yield, Brent), Binance public archive (BTCUSDT 5-min). Raw data is not redistributed. The Yahoo
intraday window is rolling, so those two files cannot be re-downloaded identically; their download
date is in `data/raw/manifest.json`.

## Headline findings
* Clustering in |r| is universal: the i.i.d. null is rejected for all 14 assets (exact permutation
  test) and at every frequency from 5 minutes to 1 month.
* Cont's power-law exponent β ∈ [0.2, 0.4] reproduces only for lags 1–100. The decay shape is not
  distinguishable from a near-integrated GARCH with shifting volatility levels.
* Shocks: front-loaded decay, faster than GARCH implies; strong leverage asymmetry in equities.
* Out-of-sample forecasting: clustering-based models beat a no-clustering baseline almost everywhere
  (QLIKE). HAR beats GARCH only with intraday realized variance. Volatility targeting gains are in
  risk control, not Sharpe.
