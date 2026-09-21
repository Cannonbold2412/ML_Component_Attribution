"""Performance and significance statistics used by MBCA.

Conventions (identical to the paper):
* a trade stream becomes a daily series by averaging the ``ret`` of all trades
  exiting on the same date (realized-at-exit, not mark-to-market);
* Sharpe is annualized (sqrt(252)) with a 6%/yr risk-free rate, population std;
* the bootstrap CI is on the difference of *per-day, non-annualized* Sharpe
  (mean/std), so CI bounds are ~sqrt(252) = 15.9x smaller than the annualized
  delta Sharpe reported next to them. "Significant" = the 95% CI excludes 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

PERIODS, RF = 252, 0.06
_EULER = 0.5772156649015329


def daily_returns(trades: pd.DataFrame) -> pd.Series:
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=float)
    return trades.groupby(pd.to_datetime(trades["exit_time"]).dt.normalize())["ret"].mean().sort_index()


def sharpe(r: pd.Series, periods: int = PERIODS, rf: float = RF) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(periods) * (r - rf / periods).mean() / r.std(ddof=0))


def summary(trades: pd.DataFrame) -> dict:
    """Trade count, annualized Sharpe, mean trade return (bps), and the total
    return / max drawdown of the compounded daily series. The daily series is
    realized-at-exit, so these are relative comparison numbers, not a
    capital-allocation equity curve."""
    r = daily_returns(trades)
    if r.empty:
        return {"n_trades": 0, "sharpe": 0.0, "mean_trade_bps": np.nan, "total_return": 0.0, "max_drawdown": 0.0}
    eq = (1 + r).cumprod()
    return {"n_trades": len(trades), "sharpe": sharpe(r), "mean_trade_bps": float(trades["ret"].mean() * 1e4),
            "total_return": float(eq.iloc[-1] - 1), "max_drawdown": float((eq / eq.cummax() - 1).min())}


def deflated_sharpe(r, n_trials: int = 1) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio: P(true SR > E[max SR
    of n_trials null strategies]). Per-period (non-annualized) returns. >= 0.95
    is the conventional 'survives multiple testing' bar."""
    r = pd.Series(r).dropna().to_numpy(float)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = st.skew(r), st.kurtosis(r, fisher=False)
    sr_std = np.sqrt(max((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1), 1e-12))
    bench = 0.0 if n_trials <= 1 else sr_std * ((1 - _EULER) * st.norm.ppf(1 - 1 / n_trials)
                                                 + _EULER * st.norm.ppf(1 - 1 / (n_trials * np.e)))
    return float(st.norm.cdf((sr - bench) / sr_std))


def _sr(x: np.ndarray) -> np.ndarray:
    """Per-period Sharpe along the last axis (ddof=1); 0 where std is 0."""
    sd = x.std(axis=-1, ddof=1)
    return np.divide(x.mean(axis=-1), sd, out=np.zeros_like(sd), where=sd > 0)


def bootstrap_ci(a: pd.Series, b: pd.Series, n: int = 2000, ci: float = 0.95, seed: int = 0,
                 paired: bool = False) -> tuple[float, float]:
    """95% bootstrap CI of per-day Sharpe(a) - Sharpe(b).

    ``paired=False`` (default, the paper's test): each daily series is resampled
    i.i.d. and independently -- ignores the cross-correlation between variant and
    baseline, so it is conservative when the two are positively correlated.
    ``paired=True``: resample the *same* calendar days for both (union of exit
    dates, 0 on days a variant has no exits) -- tighter, and the better test when
    both variants trade the same instruments over the same period."""
    rng = np.random.default_rng(seed)
    if paired:
        both = pd.concat([pd.Series(a), pd.Series(b)], axis=1).fillna(0.0).to_numpy()
        idx = rng.integers(0, len(both), size=(n, len(both)))
        diffs = _sr(both[idx, 0]) - _sr(both[idx, 1])
    else:
        a, b = pd.Series(a).dropna().to_numpy(), pd.Series(b).dropna().to_numpy()
        diffs = _sr(a[rng.integers(0, len(a), (n, len(a)))]) - _sr(b[rng.integers(0, len(b), (n, len(b)))])
    lo, hi = np.percentile(diffs, [50 * (1 - ci), 100 - 50 * (1 - ci)])
    return float(lo), float(hi)


def recommendation(cells: dict[str, dict]) -> str:
    """The paper's mechanical Opportunity-Matrix rule. ``cells`` maps market ->
    {"sig_gain": bool, "sig_loss": bool, "mean_delta": float|None} aggregated over windows."""
    gain = [m for m, c in cells.items() if c["sig_gain"]]
    loss = [m for m, c in cells.items() if c["sig_loss"]]
    pos = [m for m, c in cells.items() if c["mean_delta"] is not None and c["mean_delta"] > 0 and not c["sig_loss"]]
    if gain:
        return f"Adopt (significant gain in {', '.join(gain)})"
    if loss and not pos:
        return f"Avoid (significant loss in {', '.join(loss)})"
    if loss:
        return f"Mixed (avoid in {', '.join(loss)}; promising, unreplicated in {', '.join(pos)})"
    if pos:
        return f"Worth exploring in {', '.join(pos)} (unreplicated)"
    return "Neutral (no consistent effect)"
