# Errata and refinements (Version 4 vs. Draft v3 of the paper)

Found while re-implementing the paper from scratch for this repository. **No published number changed.**
The findings below are corrections to how the numbers were *described* or *drawn*, plus the results of
an independent replication.

## 1. The bootstrap test was misdescribed (text fixed)

Draft v3 said the significance test was a *"paired, instrument-level block bootstrap ... preserving
within-instrument time order and autocorrelation"*. The code that produced every CI in the paper
(`paired_bootstrap_ci` in the original study pipeline) actually does this:

* it resamples each variant's **daily** return series **independently**, i.i.d., with replacement
  (2,000 replicates, seed 0). It is not paired, not block-based, and not instrument-level;
* it computes the difference of **per-day, non-annualized** Sharpe ratios (mean / std).

Consequences:

* Significance calls (does the CI exclude 0?) are valid for the test that was actually run. Version 4
  describes that test correctly (Sec. 4.8 and Sec. 5).
* CI bounds are on a scale **sqrt(252) ≈ 15.9× smaller** than the annualized ΔSharpe printed next to
  them. Example: Best Combined in Indian equities, holdout ΔSharpe +1.17 with CI [-0.050, 0.176],
  is roughly [-0.8, 2.8] in annualized units. Version 4 gives both scales.
* Ignoring the pairing makes the test conservative when the ML variant and its baseline are
  positively correlated. That is the usual case, since they trade the same instruments on the same days.
  Ignoring serial dependence works the other way. `mbca` provides both the paper's test (the default) and a
  date-paired variant: `run_mbca(..., paired_ci=True)`.

## 2. The forest plots mixed units (figures regenerated)

Figures 1-2 drew the **annualized** ΔSharpe as the dot and the **per-day** CI as the whiskers, so dots
could sit far outside their own intervals. Version 4's figures rescale the interval by sqrt(252), so dot
and interval share one axis (`mbca.report.forest_plot`; regenerate with `python examples/04_paper_results.py`).
Which configurations are significant is unchanged.

## 3. "CAGR" dropped from reported metrics

The daily series is realized-at-exit (each day's value is the mean return of the trades closing that day).
Compounding it as if it were a fully-invested equity curve produces meaningless CAGRs, for example millions of
percent on a one-year forward window. Version 4 no longer cites CAGR. `mbca` reports mean trade return (bps),
total return, and max drawdown of that series, and labels them as relative comparison numbers.

## 4. Independent replication (new)

`mbca` re-implements the whole protocol from scratch and re-runs all three markets
(`python examples/03_reproduce_paper.py`, results in `results/replication/`). What replicates and what
does not is reported in the paper's new Section "Reference implementation and independent
replication" and in the README. In short: the direction of the headline result replicates (no
significant ML improvement anywhere; ML Exit and every Exit-containing combination lose). Individual
effect sizes, and the forex conclusion in particular, are sensitive to data vendor, cost model, and
fold boundaries.

## 5. Stale result files in the original study repository

The per-component `ml_*_*.csv` files in the original study's working tree had been overwritten by a later
re-run and no longer matched the paper. For example, the meta-labeling Indian-equities holdout baseline
showed 3.31 there vs. 2.84 in the paper. The CSVs shipped in `results/paper/` were exported from the
**committed** versions, which do match every number in the manuscript (60 configurations, 0 significant
gains, 20 significant losses).
