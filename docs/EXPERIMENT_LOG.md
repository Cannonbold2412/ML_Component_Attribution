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
