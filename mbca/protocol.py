"""The MBCA protocol: freeze -> holdout -> forward, one insertion point at a time.

    result = run_mbca(frames, components, split_date="2020-07-03", forward_start="2025-07-05")
    result.table        # one row per (variant, window): delta Sharpe, DSR, CI, significant
    result.trades       # {variant name: out-of-sample trades}

1. every component is fit ONCE on data before ``split_date`` (never refit);
2. the baseline and every ML variant are traded with the same walk-forward
   harness over the whole history (same params grids, same costs);
3. trades are scored by entry date in the holdout window [split, forward_start)
   and, if given, the forward window [forward_start, end);
4. each variant is compared to the baseline *within the same window*.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from . import stats
from .components import fit_once
from .strategy import Pipeline, walk_forward

BASELINE = "Baseline (rule-based)"


@dataclass
class MBCAResult:
    table: pd.DataFrame
    trades: dict[str, pd.DataFrame]

    def matrix(self) -> pd.DataFrame:
        """ML Opportunity Matrix for this run's single market: per variant, the
        aggregate over windows plus the paper's mechanical recommendation."""
        rows = []
        for (market, variant), g in self.table[self.table["variant"] != BASELINE].groupby(["market", "variant"], sort=False):
            cell = {"sig_gain": bool((g["significant"] & (g["delta_sharpe"] > 0)).any()),
                    "sig_loss": bool((g["significant"] & (g["delta_sharpe"] < 0)).any()),
                    "mean_delta": float(g["delta_sharpe"].mean())}
            rows.append({"market": market, "variant": variant, "kind": g["kind"].iloc[0],
                         "mean_delta_sharpe": round(cell["mean_delta"], 3),
                         "recommendation": stats.recommendation({market: cell})})
        return pd.DataFrame(rows)


def _run(frames: dict[str, pd.DataFrame], pipe: Pipeline, n_splits: int, min_trades: int) -> pd.DataFrame:
    parts = [walk_forward(df, pipe, n_splits, min_trades).assign(ticker=t) for t, df in frames.items()]
    return pd.concat(parts, ignore_index=True)


def _window(trades: pd.DataFrame, lo, hi) -> pd.DataFrame:
    et = pd.to_datetime(trades["entry_time"])
    m = et >= lo
    if hi is not None:
        m &= et < hi
    return trades[m]


def run_mbca(frames: dict[str, pd.DataFrame], components: list, split_date, forward_start=None,
             base: Pipeline | None = None, market: str = "market", n_splits: int = 5,
             min_trades: int = 10, paired_ci: bool = False, verbose: bool = True) -> MBCAResult:
    """Run the full MBCA grid on one market.

    frames: {ticker: OHLC DataFrame with Date/open/high/low/close}.
    components: fitted-or-unfitted components from ``mbca.components`` (or your own
        object with ``name``, ``kind``, ``fit(frames, split_date, base)``, ``apply(pipe)``).
    split_date: freeze date; nothing at or after it is ever used for training.
    forward_start: optional start of a forward window (e.g. data fetched after the
        study was frozen). Without it, the holdout runs to the end of the data.
    n_splits / min_trades: walk-forward folds per instrument, and the minimum
        in-sample trades a parameter set needs to be selectable on a fold.
    paired_ci: use the date-paired bootstrap instead of the paper's unpaired one.
    """
    base = base or Pipeline()
    split = pd.Timestamp(split_date)
    fwd = pd.Timestamp(forward_start) if forward_start is not None else None
    log = print if verbose else (lambda *a, **k: None)

    trades = {}
    log(f"[{market}] baseline walk-forward over {len(frames)} instruments")
    trades[BASELINE] = _run(frames, base, n_splits, min_trades)
    if trades[BASELINE].empty:
        import warnings
        warnings.warn("the baseline produced no out-of-sample trades: every walk-forward fold had fewer "
                      f"than {min_trades} in-sample trades. Use a faster signal, a shorter-lived exit, longer "
                      "history, or fewer n_splits.", stacklevel=2)
    kinds = {BASELINE: "control"}
    for c in components:
        log(f"[{market}] fit + walk-forward: {c.name}")
        fit_once(c, frames, split, base)
        trades[c.name] = _run(frames, c.apply(base), n_splits, min_trades)
        kinds[c.name] = c.kind

    # deflated Sharpe corrects for how many variants of the same component type were screened
    n_trials = Counter(type(c).__name__ for c in components)
    trials = {c.name: n_trials[type(c).__name__] for c in components} | {BASELINE: 1}

    windows = [("holdout", split, fwd)] + ([("forward", fwd, None)] if fwd is not None else [])
    rows = []
    for wname, lo, hi in windows:
        base_t = _window(trades[BASELINE], lo, hi)
        base_r, base_s = stats.daily_returns(base_t), stats.summary(base_t)
        for name, t in trades.items():
            t = _window(t, lo, hi)
            r, s = stats.daily_returns(t), stats.summary(t)
            row = {"market": market, "window": wname, "variant": name, "kind": kinds[name], **s,
                   "baseline_sharpe": base_s["sharpe"], "delta_sharpe": s["sharpe"] - base_s["sharpe"],
                   "deflated_sharpe": stats.deflated_sharpe(r, trials[name])}
            if name != BASELINE and len(r) > 1 and len(base_r) > 1:
                lo_ci, hi_ci = stats.bootstrap_ci(r, base_r, paired=paired_ci)
                row.update(ci_lo=lo_ci, ci_hi=hi_ci, significant=bool(lo_ci > 0 or hi_ci < 0))
            else:
                row.update(ci_lo=float("nan"), ci_hi=float("nan"), significant=False)
            rows.append(row)
    return MBCAResult(pd.DataFrame(rows), trades)
