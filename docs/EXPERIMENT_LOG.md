# Experiment log

Append-only. Every Study 2 run writes a machine-readable manifest to
`results/study2/runs/<run_id>/manifest.json` (git commit, config hash, data-manifest
hash, package versions, timings). This file records human decisions and deviations.
Each entry cites a commit.

## 2026-09-22 — Study 2 pre-registration

* `study.toml` committed before any Study 2 data was downloaded or any Study 2 result existed.
* Fixed in advance: universe (8 markets, ~154 Yahoo symbols), cleaning rules, 5 strategies
  and their parameter grids, component list (regime filter restricted to the `linearity` label),
  two freeze dates (2008-01-01, 2018-01-01), execution model (fill at the breaching close),
  statistics (paired circular block bootstrap, BH-FDR), regimes, and crisis windows.
* Known, accepted biases stated up front: single stocks are current large caps (survivorship);
  crypto consists of surviving coins; Yahoo continuous futures are front-month rolls without
  roll adjustment.

## 2026-09-22 — Study 2 data collected (commit: "Add Study 2 raw + clean data")

* `python -m mbca study fetch`: 153/154 Yahoo symbols + 17 FRED series downloaded; raw API bytes
  stored gzip-compressed with URL, UTC timestamp and SHA-256 in `data/study2/raw/MANIFEST.csv`.
* **Deviation from pre-registration:** `ROG.SW` (Roche) returns HTTP 404 from Yahoo's chart API
  (recorded in `data/study2/raw/FAILED.csv`); it is excluded. No substitute was added (adding one
  after the fact would be a post-hoc universe choice). Final universe: 153 series.
* Cleaning (`python -m mbca study clean`): 19 duplicate dates, 7,890 rows with missing or
  non-positive prices, 10,868 OHLC-consistency repairs, 2 bad ticks dropped; every series passes
  `min_rows`. Per-series counts in `data/study2/clean/MANIFEST.csv`.
* Cross-source validation vs FRED (`data/study2/validation/yahoo_vs_fred.csv`): median |price
  difference| 0.0-0.4% for indices/FX/WTI. 1-day FX return correlation is only ~0.4 because
  Yahoo FX bars close around London midnight while FRED uses the noon New York fix (the correlation
  splits between lag 0 and lag 1); 5-day return correlation 0.82-0.90. NG=F vs Henry Hub *spot* is a
  different instrument (5-day corr 0.64), reported rather than hidden.
* Browser spot-check (claude-in-chrome, MarketWatch historical pages): AAPL and SAP.DE, 20 sessions
  each, raw Yahoo closes match to the cent (`data/study2/validation/browser_spotcheck.csv`).
