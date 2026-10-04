# Volatility Clustering — reproducing and extending Cont (2001)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23143125.svg)](https://doi.org/10.5281/zenodo.23143125)

*Cite as:* Garg, S. (2026). *Volatility Clustering Revisited: A Reproduction and Extension of Cont (2001)*
(v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23143125

Research project testing the volatility-clustering stylized facts of
**Cont, R. (2001), "Empirical properties of asset returns: stylized facts and statistical issues",
*Quantitative Finance* 1(2), 223–236**, on modern public data. It then asks whether clustering
differs across assets, frequencies, shocks and regimes, and whether it improves volatility forecasts.

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
