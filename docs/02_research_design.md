# Phases 2–3 — Hypotheses and Empirical Design (pre-registered)

This document was written **before** any return series was analysed. The only data step done
beforehand was checking that each source is reachable. Every test, sample split, lag range and
decision rule below is fixed in advance. Any deviation is recorded in the report under
"Deviations from pre-registration". This is the main defence against data-snooping: we cannot
quietly pick the lag window or the split that makes results look best.

---

## 1. Hypotheses

Notation: r_t = daily log return (or Δt-return), a_t = |r_t|, ρ_x(τ) = lag-τ autocorrelation of series x.

### Reproduction block (Cont's claims)

**H1 — No linear autocorrelation (control).** For daily returns, |ρ_r(τ)| is economically small
(< 0.05) for τ ≥ 2. The lag-1 value may be non-zero for indices because of non-synchronous trading.
*Why a control:* volatility clustering is interesting precisely because it coexists with
unpredictable signs.

**H2 — Volatility clustering (claims C1–C3).**
* H2₀: {a_t} is i.i.d. (equivalently, under exchangeability, the time order carries no information).
* H2₁: ρ_a(τ) > 0 for τ = 1, …, 20 trading days.
* **Test statistic:** S₂₀ = Σ_{τ=1}^{20} ρ̂_a(τ). This is a single, pre-chosen statistic per asset, so
  we don't run 20 per-lag tests.
* **Inference:** one-sided **permutation test** (B = 2,000 random shuffles of a_t). Shuffling
  destroys all temporal dependence but keeps the marginal distribution, including its heavy tails.
  The test is therefore exact under H2₀ and needs **no moment conditions**. This is the direct answer
  to Cont §5.3.
* Reported alongside: Ljung–Box Q(20) on a_t (classical, but invalid if E r⁴ = ∞, shown for comparison)
  and the **Spearman (rank) ACF**, which is moment-free.
* **Multiplicity:** Holm–Bonferroni across all assets (family-wise α = 0.05).

**H3 — Slow (power-law) decay (claim C5).**
* H3a: on lags τ ∈ [1, 100] trading days, log ρ̂_a(τ) is linear in log τ with slope −β, and
  β ∈ [0.2, 0.4].
* H3b: a power law fits ρ̂_a(τ), τ ∈ [1, 100], better than an exponential (short-memory/GARCH-type)
  decay, judged by sum of squared errors in levels.
* H3c (**non-stationarity check**, Mikosch & Stărică 2004): if slow decay comes from shifts in the
  *level* of volatility rather than genuine long memory, then (i) ACFs computed **within**
  non-overlapping 5-year subsamples and (ii) ACFs of a_t / σ̂₂₅₂(t−1) (|return| divided by the
  trailing one-year volatility, using only past data) should decay much faster than the full-sample ACF.
* Second estimator of memory: GPH log-periodogram estimate of d for a_t (bandwidth m = ⌊n^0.5⌋), with
  the implied β = 1 − 2d. Two independent estimators must roughly agree before we take "β" seriously.
* Simulation benchmark: ACF of a_t simulated from the fitted GARCH(1,1) (same n, 200 paths). Is the
  empirical ACF at lags 50–100 outside the GARCH 95% simulation envelope?
* **Decision rule:** we only call the decay "long-memory-like" if H3b holds **and** the H3c subsample
  ACFs remain above the GARCH envelope at long lags. Otherwise we report "slow decay, not
  distinguishable from non-stationarity".

**H4 — Taylor effect (claim C4).** For fixed τ ∈ {1, 5, 20}, α ↦ ρ̂_{|r|^α}(τ) is maximised in
α ∈ [0.75, 1.25] (grid α ∈ {0.25, 0.5, …, 3}). Uncertainty of the arg-max is estimated by stationary
bootstrap. Extra prediction from Cont §5.3: the bootstrap CI of ρ̂_{r²}(τ) is wider than that of ρ̂_{|r|}(τ).

### Beyond-reproduction block

**H5 — Cross-asset differences.** Persistence measures (S₂₀, β, GARCH α+β and its half-life) differ
across asset classes (equity index, single stock, FX, rates, commodity, crypto).
*Caveat fixed in advance:* with 1–4 assets per class we **cannot** make population-level
claims about "asset classes". We report per-asset bootstrap CIs and treat class-level patterns as
descriptive.

**H6 — Sampling frequency.** Clustering is present at every sampling interval from 5 minutes to
1 week. Once intraday periodicity is removed and lags are expressed in **calendar time** (hours),
the ACFs from different sampling frequencies roughly line up. Without deseasonalising, intraday
ACFs show spurious periodic peaks (Andersen & Bollerslev 1997).

**H7 — Persistence after large shocks.** After a shock (|r_t| > 4·σ̂_{t−1}, σ̂ = EWMA vol using
data up to t−1 only), the **excess** volatility E[a_{t+k}/σ̂_{t−1}] decays with a half-life of
several days to weeks. The decay is slower than GARCH(1,1) implies, and slower after negative
shocks than after positive ones (leverage).
*Overlapping events:* we keep only shocks with no other shock in the previous 20 trading days
(declustering). Inference uses a bootstrap over events.

**H8 — Regimes.** The strength of clustering (S₂₀) varies across time. Measured on rolling 5-year
windows and fixed sub-periods.
*Pitfall fixed in advance:* conditioning on "high-vol regimes" defined ex post from the same a_t
creates selection bias in the ACF (truncation). So regimes are **calendar sub-periods**, not
volatility-sorted samples. Result is descriptive.

**H9 — Forecasting (Cont's open question C7).**
* H9a: models that exploit clustering (EWMA, GARCH(1,1), GJR-GARCH, HAR) have lower out-of-sample
  QLIKE loss than a no-clustering baseline (expanding-window historical variance).
* H9b: long-memory-mimicking HAR beats short-memory GARCH/EWMA. This is the forecasting
  implication of H3.
* Tests: Diebold–Mariano with Harvey–Leybourne–Newbold correction and Newey–West variance.
  Holm across the model-vs-baseline comparisons.
* Economic significance, reported separately: a volatility-targeting overlay on the S&P 500.
  Does a better forecast make realized portfolio vol closer to target, net of transaction costs?

## 2. Null-hypothesis summary

| H | Null | Test | Robust to heavy tails? |
|---|------|------|------------------------|
| H1 | ρ_r(τ)=0 | Bootstrap CI | Yes (bootstrap) |
| H2 | a_t i.i.d. | Permutation on S₂₀, Holm | **Yes (exact)** |
| H3 | Exponential decay / non-stationary level shifts | Model fit SSE, subsample ACF, GARCH simulation envelope, GPH | Partly |
| H4 | arg-max α outside [0.75, 1.25] | Stationary bootstrap | Partly |
| H5 | Equal persistence | Overlapping bootstrap CIs (descriptive) | Partly |
| H7 | No excess vol after shock / symmetric | Event bootstrap | Yes |
| H9 | Equal predictive accuracy | DM-HLN, Holm | Asymptotic |

## 3. Known pitfalls and how the design handles them

| Pitfall | Handling |
|---------|----------|
| **Heavy tails invalidate ACF bands** (Davis–Mikosch) | Permutation test, rank ACF, bootstrap; Ljung–Box shown only for contrast |
| **Non-stationarity / structural breaks** fake long memory | H3c subsample and trailing-vol-standardised ACFs; GARCH envelope |
| **Microstructure noise** (bid–ask bounce, stale prices) | Daily data for the main results. Intraday starts at 5 min (not ticks). ACF of r at intraday scale reported to show bounce. Yahoo FX avoided (stale quotes); FRED noon rates used |
| **Intraday seasonality** | Divide \|r\| by its time-of-day mean before ACFs |
| **Volatility is latent** | All clustering claims use observable \|r\|. Forecasts are evaluated against proxies (r², range or realized variance) using Patton-robust losses |
| **Look-ahead bias** | EWMA, σ̂₂₅₂ and shock thresholds use data ≤ t−1. Forecast models are refit only on data before each forecast. Vol-target weights decided at close t, applied to return t+1 (robustness: extra 1-day delay) |
| **Leakage via tuning** | No hyper-parameters are tuned. EWMA λ = 0.94 (RiskMetrics), HAR lags (1, 5, 22) as in Corsi (2009), GARCH orders fixed. So there is no validation set to leak |
| **Multiple testing** | One pre-chosen statistic per hypothesis per asset. Holm within each family. Full list of tests run is in the report |
| **Survivorship bias** | Single stocks (AAPL, JPM, XOM, BMW) are today's survivors. This affects **level** results (returns), not much the existence of clustering. Flagged as a limitation |
| **Roll / contract artifacts** | Gold futures continuous series (GC=F) has roll jumps. Brent from FRED is spot (WTI excluded: negative price on 2020-04-20 makes log returns undefined) |
| **Negative / non-price series** | 10Y Treasury: "return" := daily change in yield (bp). Documented |

## 4. Data

| Class | Series | Source | Start | Notes |
|-------|--------|--------|-------|-------|
| Equity index | S&P 500 (^GSPC) | Yahoo | 1990 | Price index; OHLC unreliable before ~1993 → range proxy only used later |
| Equity index | DAX (^GDAXI) | Yahoo | 1990 | Cont's Frankfurt market |
| Equity index | Nikkei 225 (^N225) | Yahoo | 1990 | |
| Single stock | BMW (BMW.DE) | Yahoo | 1996* | Cont Fig. 1 asset; adjusted close |
| Single stock | AAPL, JPM, XOM | Yahoo | 1990 | Adjusted close; survivors |
| FX | USD/JPY, USD/GBP, USD/EUR | FRED (DEXJPUS, DEXUSUK, DEXUSEU) | 1990 / 1999 | Noon buying rates NY |
| Rates | 10Y Treasury yield | FRED (DGS10) | 1990 | Δ yield |
| Commodity | Brent spot | FRED (DCOILBRENTEU) | 1990 | |
| Commodity | Gold futures | Yahoo (GC=F) | 2000 | Continuous contract |
| Crypto | Bitcoin | Yahoo (BTC-USD) | 2014 | 7 days/week |
| Intraday | BTCUSDT 5-min | Binance public archive | 2019 | 24/7, complete |
| Intraday | SPY 5-min / 60-min | Yahoo | last 60 / 730 days | Rolling window: snapshot cached and dated |

*Yahoo history start is whatever is available; actual start dates are recorded in the data manifest.

**End date:** 2025-12-31 for all daily series (complete calendar years, fixed for reproducibility).

**Cleaning rules (fixed):**
1. Drop non-positive and missing prices. FRED uses blanks for holidays, so dropping these is
   correct, **not** forward-filled. Forward-filling would create artificial zero returns and bias ACFs.
2. Returns are computed between consecutive *available* observations (business-day returns, no
   calendar-day interpolation).
3. Flag |r| > 25% (daily) for manual inspection. Do **not** winsorise: extreme moves are the object of study.
4. Count and report exact-zero returns (stale prices indicate data problems).
5. Intraday: keep only complete bars. Returns do not cross gaps longer than one bar.

## 5. Measures

* Return: log return r_t = ln P_t − ln P_{t−1} (Δ yield for DGS10).
* Volatility proxies: |r_t|, r_t², Garman–Klass range estimator (S&P 500, from the date the OHLC
  looks valid), 5-min realized variance RV_t = Σ r²_{t,i} (BTC).
* Persistence summaries: ρ̂_a(1), S₂₀, power-law β (lags 1–100), GPH d, GARCH(1,1) α+β and its
  half-life ln 0.5 / ln(α+β).

## 6. Samples and out-of-sample protocol (H9)

* **Daily equity/FX etc.:** in-sample (estimation) 1990-01-01 → 2009-12-31; **out-of-sample
  2010-01-01 → 2025-12-31**. Models are refit at the start of every calendar month on an **expanding**
  window that ends the day before. Forecast horizon h = 1 day (primary); h = 5 (robustness).
* **BTC realized vol:** in-sample 2019–2021, out-of-sample 2022–2025.
* The descriptive clustering analysis (H1–H8) uses the full sample. It makes no predictive claims.
  Its sub-period stability is part of H3c/H8.

## 7. Robustness checks (fixed list)

1. ACF of |r| vs r² vs ranks vs log|r| (all four of Cont's transforms).
2. Sub-period ACFs (5-year blocks) and the trailing-vol-standardised ACF.
3. Power-law fit range: lags [1, 100] (primary), [5, 250] (robustness).
4. Shock threshold 4σ (primary), 3σ and 5σ; declustering window 20 vs 60 days.
5. Forecast target: r² (primary) and Garman–Klass (S&P 500); horizon 1 vs 5 days; refit monthly vs
   yearly; QLIKE (primary) and MSE.
6. Vol-target overlay: costs 2 bp and 10 bp per unit turnover; execution delay 0 vs 1 day.
7. Intraday: with vs without deseasonalisation.

## 8. Significance vs economic relevance

Statistical significance (p-values, CIs) and economic magnitude are reported separately.
For clustering: ρ̂_a(1) of 0.2 means today's |r| explains ~4% of tomorrow's |r| variance. That is
large for finance, but tiny next to the noise in a single day's |r|. For forecasting: QLIKE gains
are translated into vol-target tracking error and net Sharpe.
