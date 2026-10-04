# Phase 1 — Reading Cont (2001)

**Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues.
*Quantitative Finance*, 1(2), 223–236.**

Page numbers below refer to the journal pagination.

---

## 1. What the paper is (and is not)

It is a **survey / position paper**, not a single-dataset empirical study. Cont collects
"stylized facts", meaning properties that many independent studies have found across markets,
instruments and periods. He then argues that the statistical properties of returns
(heavy tails, nonlinear dependence) **undermine the standard tools** people use to measure them.

* **Research question.** Which statistical properties of asset returns are universal
  (asset- and period-independent)? And how reliable are the estimators we use to establish them?
* **Methodological stance.** "Let the data speak for themselves" (p. 223): prefer **non-parametric**
  statistics (sample ACFs, kernel densities, sample moments vs sample size). Use
  **semi-parametric** ones (tail index, power-law decay exponent) only when a single number is
  needed. No parametric model is assumed to be true.
* **Consequence for us.** Most quantitative claims in the paper are *summaries of other people's
  results* (e.g. "β ∈ [0.2, 0.4]" cites refs [21, 22, 74]). The paper's own illustrative
  figures use specific datasets, but it gives no tables of volatility-ACF estimates that one could
  replicate number-for-number. So we are **reproducing stylized facts, not replicating a table.**

## 2. Notation and core definitions

* Price S(t), log price X(t) = ln S(t). Log return at scale Δt: **r(t, Δt) = X(t+Δt) − X(t)** (eq. 1).
  Cont stresses that every property depends on Δt.
* Sample autocorrelation of returns: C(τ) = corr(r(t+τ,Δt), r(t,Δt)) (eq. 14).
* **Volatility clustering measure:** autocorrelation of squared returns
  C₂(τ) = corr(|r(t+τ,Δt)|², |r(t,Δt)|²) (eq. 15).
* Generalisation: **C_α(τ) = corr(|r(t+τ,Δt)|^α, |r(t,Δt)|^α)** (eq. 16).
* Power-law decay: **C_α(τ) ~ A / τ^β** (eq. 17), with β ∈ [0.2, 0.4] for absolute or squared returns.
* Log-absolute ACF (multifractal literature): C₀(τ) = corr(ln|r(t+τ)|, ln|r(t)|) (eq. 18).
* Leverage: L(τ) = corr(|r(t+τ,Δt)|², r(t,Δt)) (eq. 20), negative for τ > 0, ≈ 0 for τ < 0.
* Multiplicative decomposition r(t,Δt) = σ(t,Δt)·ε(t) (eq. 21). Cont stresses that σ is
  **not observable**, so "volatility correlation" depends on the model, while the ACF of |r| can be
  computed directly. This is why we base the study on |r| and treat σ̂ only as a forecasting object.

### What a sample ACF is (brief refresher)
For a series x₁…x_n with mean x̄, ρ̂(τ) = Σ_{t}(x_t − x̄)(x_{t+τ} − x̄) / Σ_t (x_t − x̄)².
For an i.i.d. series with **finite fourth moment of x**, √n·ρ̂(τ) → N(0,1). That is where the
familiar ±1.96/√n bands come from. Section 5.3 of Cont is about what happens when that assumption fails.

## 3. The eleven stylized facts (p. 224) and which ones concern volatility clustering

| # | Fact | Relevance to this project |
|---|------|---------------------------|
| 1 | Absence of linear autocorrelation (except < ~20 min) | **Control**: clustering must coexist with near-zero ACF of r |
| 2 | Heavy tails, tail index in (2, 5) | **Statistical problem** for ACF inference (§5.3) |
| 3 | Gain/loss asymmetry | — |
| 4 | Aggregational Gaussianity | Relevant to the frequency question |
| 5 | Intermittency (irregular bursts) | Visual counterpart of clustering |
| **6** | **Volatility clustering: "different measures of volatility display a positive autocorrelation over several days"** | **Core claim (H2)** |
| 7 | Conditional heavy tails (GARCH residuals still heavy-tailed, less so) | Side check on GARCH fits |
| **8** | **Slow decay of ACF of absolute returns, roughly power law, β ∈ [0.2, 0.4]; "sometimes interpreted as long-range dependence"** | **Core claim (H3)** |
| 9 | Leverage effect | Asymmetric response to shocks (H7) |
| 10 | Volume/volatility correlation | Out of scope |
| 11 | Asymmetry in time scales (coarse vol predicts fine vol better than the reverse) | Motivates HAR model (H9) |

### Exact claims about volatility clustering (verbatim or near-verbatim)

1. **C1 (p. 224, fact 6):** "different measures of volatility display a positive autocorrelation over
   several days, which quantifies the fact that high-volatility events tend to cluster in time."
2. **C2 (p. 230, §5.2):** "simple nonlinear functions of returns, such as absolute or squared returns,
   exhibit significant positive autocorrelation or persistence." Absence of linear correlation
   does **not** imply independence; "log prices are therefore not random walks."
3. **C3 (p. 230):** C₂(τ) "remains positive and decays slowly, remaining significantly positive over
   several days, sometimes weeks." This is "a model-free property of returns which does not rely
   on the GARCH hypothesis."
4. **C4 (p. 230, Ding & Granger):** for a given lag, C_α(τ) is **highest for α ≈ 1**: absolute returns
   are more predictable than other powers (the "Taylor effect").
5. **C5 (p. 230, eq. 17):** decay of C_α(τ) is "well reproduced by a power law" with β ∈ [0.2, 0.4].
   Cont himself only says this is "*sometimes interpreted*" as long-range dependence.
6. **C6 (p. 231, §5.3):** with heavy tails, sample ACFs of |r|^α (especially α = 2) are unreliable.
   If E r⁴ = ∞, the sample ACF of r converges more slowly than √n, with wider bands
   (Davis & Mikosch, 1998). For squared returns, even *defining* the ACF needs E r⁴ < ∞, and
   tail indices near 4 make that borderline. Mikosch & Stărică show the ACF of squared GARCH(1,1)
   returns has non-standard sampling properties. "One should be very careful when drawing
   quantitative conclusions from the autocorrelation function of powers of the returns."
7. **C7 (p. 233, conclusion, open question):** "does the presence of volatility clustering imply
   anything interesting from a practical standpoint for volatility forecasting? … can this be put
   to use to implement a more effective risk measurement/management approach?" The paper leaves
   this **open**. It is the natural "beyond reproduction" question (H9).

### Data the paper itself shows
The illustrative figures use: daily **BMW** returns on the Frankfurt exchange, 1992–1998 (Fig. 1,
the clustering illustration); **S&P 500 index futures**, 1991–1995, at 5- and 30-minute
resolution (Figs. 2, 3, 8); **USD/Yen** tick data 1992–1994 (Fig. 6); **KLM** tick returns, NYSE
(Fig. 7); **USD/DM** and **USD/CHF** futures (Table 1). Most of these are proprietary
high-frequency datasets.

## 4. Statistical concepts the paper relies on

* **Stationarity** (§3.1): joint distributions invariant to time shifts. Cont notes calendar-time
  returns may violate it (intraday seasonality, weekends), and that one can redefine time
  ("business/tick time") to get closer to stationarity. → We must **deseasonalise intraday data.**
* **Ergodicity** (§3.2): time averages must converge to expectations. Hard to verify under long-range
  dependence. → Sample ACFs at long lags can be biased and slow to converge. We report subsample stability.
* **Finite-sample properties** (§3.3): with N ≈ 10³–10⁴ a statistic without a confidence interval is
  meaningless. Standard CIs need i.i.d./weak dependence plus finite 4th moments, which the data
  violate. → We use **permutation tests** (exact under the i.i.d. null), **block bootstrap** CIs,
  and **rank-based ACFs** (moment-free).
* **Tail index / EVT** (§4): tail index ≈ 3–4 → E r⁴ is borderline finite. This is exactly why
  §5.3 matters for us.
* **Power-law vs exponential decay**: GARCH(1,1) implies an *exponentially* decaying ACF of r²
  (rate α+β). A power law means *hyperbolic* decay, i.e. long memory. Telling them apart over a
  finite range of lags is hard, and Cont's wording ("roughly", "sometimes interpreted") reflects that.

## 5. Main findings, restated as checkable statements

| Claim | Type | Testable with public data? |
|-------|------|----------------------------|
| ACF of r ≈ 0 at daily scale | Qualitative | Yes |
| ACF of \|r\| and r² significantly > 0 for many lags (days–weeks) | Qualitative | Yes |
| ACF of \|r\|^α maximal near α = 1 | Semi-quantitative | Yes |
| Decay ≈ power law, β ∈ [0.2, 0.4] | Quantitative (cited) | Yes, but identification is weak |
| Holds across stocks, indices, FX | Universality | Partly: a handful of assets per class |
| Sample ACF of r² is noisier than that of \|r\| | Statistical | Yes (bootstrap comparison) |
| Clustering useful for forecasting / risk management | **Open question** | Yes: out-of-sample forecast study |

## 6. What can realistically be reproduced with modern public data

**Feasible**
* Daily data for decades: equity indices and stocks (Yahoo Finance), FX spot and Treasury yields
  (FRED, official noon rates), Brent crude (FRED), gold futures, Bitcoin. This covers facts 1, 6, 8,
  the Taylor effect, leverage, conditional heavy tails, and cross-asset universality at the daily scale.
* **BMW daily returns** are available (BMW.DE). We can redo Cont's Fig. 1 asset on a far longer sample.
* Intraday: Binance publishes free, complete 5-minute klines for BTCUSDT since 2017. That allows a
  clean multi-year intraday study across frequencies (24/7 trading, so no overnight gaps).

**Not feasible / must differ**
* **S&P 500 futures and FX tick data 1991–1995** are proprietary (Cont's sources: Olsen & Associates,
  CME). Free intraday equity data (Yahoo) only covers the last 60 days at 5-minute resolution and
  730 days at hourly resolution, and it is a rolling window that cannot be reproduced exactly.
  We use it only as a short cross-check and say so.
* **KLM** no longer trades on the NYSE (merged with Air France in 2004). We have no tick-level study.
* Intraday **FX** at multi-year length is not available from a free, documented, stable source.
  So the frequency question uses crypto (long sample) plus equity ETFs (short sample). This is a
  limitation: crypto microstructure differs from the 1990s futures markets in the paper.

## 7. Literature this project adds (beyond Cont)

* Mandelbrot (1963): original observation that "large changes tend to be followed by large changes".
* Engle (1982); Bollerslev (1986): ARCH/GARCH, a parametric model of clustering with *exponential* ACF decay.
* Taylor (1986); Ding, Granger & Engle (1993): the ACF of |r|^d is largest near d = 1, and decays slowly.
* Davis & Mikosch (1998); Mikosch & Stărică (2004): heavy-tailed ACF limit theory. **Non-stationarity
  (level shifts in unconditional variance) can manufacture apparent long memory.**
  Granger & Hyung (2004) and Lobato & Savin (1998) make the same point from the long-memory side.
* Andersen & Bollerslev (1997): the intraday periodicity in |r| distorts intraday ACFs and must be
  filtered out before measuring persistence.
* Andersen, Bollerslev, Diebold & Labys (2003); Corsi (2009): realized volatility and the HAR model,
  a simple cascade that mimics long memory and directly uses fact 11.
* Patton (2011): which loss functions give consistent forecast rankings when the volatility target is
  a noisy proxy (MSE and QLIKE are robust).
* Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997): tests of equal predictive accuracy.
* Hansen & Lunde (2005): "Does anything beat a GARCH(1,1)?"
* Engle & Patton (2001), *What good is a volatility model?*, in the **same journal issue** as Cont
  (*QF* 1(2), 237–245). It speaks directly to Cont's open question C7.
* Moreira & Muir (2017); Cederburg, O'Doherty, Wang & Yan (2020): economic value of volatility
  timing, with an out-of-sample critique.

Full references are in `report/report.md`.
