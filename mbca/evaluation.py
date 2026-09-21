"""Study 2 evaluation: mark-to-market portfolios, risk metrics, and inference.

Portfolio convention (study.toml [execution] portfolio = "equal_weight"):
capital is split 1/N_t across the N_t instruments of the portfolio that have a
price bar on day t. An instrument's daily return is its position (direction x
size, held from the entry close to the exit close) times its close-to-close
return; the exit day uses the actual fill price. Half the round-trip cost is
charged on the entry day and half on the exit day, scaled by |size|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .stats import deflated_sharpe

P = 252


# ------------------------------------------------------------------ mark-to-market
def instrument_daily(trades: pd.DataFrame, df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Daily MTM return and position of one instrument. Index = the instrument's own dates."""
    dates = pd.to_datetime(df["Date"]).to_numpy()
    close = df["close"].to_numpy(float)
    ret, pos = np.zeros(len(dates)), np.zeros(len(dates))
    if trades is not None and len(trades):
        e_idx = np.searchsorted(dates, pd.to_datetime(trades["entry_time"]).to_numpy())
        x_idx = np.searchsorted(dates, pd.to_datetime(trades["exit_time"]).to_numpy())
        w = (trades["direction"].to_numpy(float) * trades["size"].to_numpy(float))
        xp = trades["exit_price"].to_numpy(float)
        half = cost_bps / 2e4
        for e, x, wi, px in zip(e_idx, x_idx, w, xp):
            if x <= e or x >= len(dates):
                continue
            seg = close[e + 1:x + 1] / close[e:x] - 1
            seg[-1] = px / close[x - 1] - 1  # exit-day fill (stop level or breaching close)
            ret[e + 1:x + 1] += wi * seg
            pos[e + 1:x + 1] = wi
            ret[e] -= half * abs(wi)
            ret[x] -= half * abs(wi)
    return pd.DataFrame({"ret": ret, "pos": pos}, index=pd.DatetimeIndex(dates))


def portfolio(daily: dict[str, pd.DataFrame], lo=None, hi=None) -> pd.DataFrame:
    """Equal-weight portfolio of instrument_daily frames -> DataFrame(ret, turnover, exposure, n)."""
    rets = pd.DataFrame({k: v["ret"] for k, v in daily.items()})
    poss = pd.DataFrame({k: v["pos"] for k, v in daily.items()})
    listed = rets.notna()
    n = listed.sum(axis=1).clip(lower=1)
    out = pd.DataFrame({
        "ret": rets.fillna(0).sum(axis=1) / n,
        "turnover": poss.ffill().fillna(0).diff().abs().sum(axis=1) / n,
        "exposure": poss.fillna(0).abs().sum(axis=1) / n,
        "n": listed.sum(axis=1),
    })
    if lo is not None:
        out = out[out.index >= pd.Timestamp(lo)]
    if hi is not None:
        out = out[out.index < pd.Timestamp(hi)]
    return out


# ------------------------------------------------------------------ metrics
def sharpe(r) -> float:
    r = np.asarray(r, float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(np.sqrt(P) * r.mean() / sd) if sd > 0 else 0.0


def metrics(pf: pd.DataFrame, trades: pd.DataFrame | None = None, n_trials: int = 1) -> dict:
    """Return / risk / tail / turnover metrics of a portfolio frame (from :func:`portfolio`)."""
    r = pf["ret"].to_numpy(float)
    if len(r) < 2 or not np.any(r):
        return {"days": len(r)}
    eq = np.cumprod(1 + r)
    years = len(r) / P
    dd = eq / np.maximum.accumulate(eq) - 1
    downside = r[r < 0]
    q05 = np.quantile(r, 0.05)
    cagr = eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1.0
    out = {
        "days": len(r), "ann_return": float(cagr), "ann_vol": float(r.std(ddof=1) * np.sqrt(P)),
        "sharpe": sharpe(r),
        "sortino": float(np.sqrt(P) * r.mean() / downside.std(ddof=1)) if len(downside) > 1 else 0.0,
        "max_drawdown": float(dd.min()), "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else 0.0,
        "var95": float(-q05), "cvar95": float(-r[r <= q05].mean()),
        "skew": float(st.skew(r)), "excess_kurtosis": float(st.kurtosis(r)),
        "turnover": float(pf["turnover"].mean() * P),  # one-way, x capital per year
        "exposure": float(pf["exposure"].mean()),
        "deflated_sharpe": deflated_sharpe(r, n_trials),
    }
    if trades is not None and len(trades):
        hold = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.days
        wins, losses = trades["ret"][trades["ret"] > 0].sum(), -trades["ret"][trades["ret"] < 0].sum()
        out.update(n_trades=len(trades), win_rate=float((trades["ret"] > 0).mean()),
                   avg_hold_days=float(hold.mean()), mean_trade_bps=float(trades["ret"].mean() * 1e4),
                   profit_factor=float(wins / losses) if losses > 0 else np.inf)
    return out


# ------------------------------------------------------------------ inference
def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, block: int = 20, n_boot: int = 2000,
                                seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired circular-block bootstrap of annualized Sharpe(a) - Sharpe(b).
    Both series are aligned on the union of dates (0 where absent) and the SAME
    blocks of days are drawn for both, preserving cross-correlation and short-range
    serial dependence. Two-sided p = 2 * min(P(d* <= 0), P(d* >= 0))."""
    j = pd.concat([a, b], axis=1).fillna(0.0).to_numpy(float)
    T = len(j)
    point = sharpe(j[:, 0]) - sharpe(j[:, 1])
    if T < 2 * block:
        return {"delta_sharpe": point, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed)
    nb = -(-T // block)
    starts = rng.integers(0, T, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)) % T).reshape(n_boot, -1)[:, :T]
    xa, xb = j[idx, 0], j[idx, 1]

    def sr(x):
        sd = x.std(axis=1, ddof=1)
        return np.divide(x.mean(axis=1), sd, out=np.zeros_like(sd), where=sd > 0) * np.sqrt(P)

    d = sr(xa) - sr(xb)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"delta_sharpe": point, "ci_lo": float(lo), "ci_hi": float(hi), "p_value": p}


def bh_fdr(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = pd.Series(p, dtype=float)
    ok = p.dropna().sort_values()
    m = len(ok)
    if not m:
        return p * np.nan
    q = (ok * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return q.reindex(p.index)


def holm(p: pd.Series) -> pd.Series:
    p = pd.Series(p, dtype=float)
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = (ok * (m - np.arange(m))).cummax().clip(upper=1)
    return adj.reindex(p.index)


# ------------------------------------------------------------------ regimes
def market_index(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Equal-weight buy-and-hold index of a set of instruments (daily returns averaged)."""
    r = pd.DataFrame({k: v.set_index("Date")["close"].pct_change() for k, v in frames.items()})
    return r.mean(axis=1).fillna(0)


def regime_labels(index_ret: pd.Series, sma_days: int = 200, vol_days: int = 20) -> pd.DataFrame:
    """Per-day regime labels known at t-1: trend (bull/bear) and volatility (high/low)."""
    level = (1 + index_ret).cumprod()
    bull = (level > level.rolling(sma_days).mean()).shift(1)
    rv = index_ret.rolling(vol_days).std()
    high = (rv > rv.expanding(min_periods=250).median()).shift(1)
    return pd.DataFrame({
        "trend": np.where(bull.isna(), None, np.where(bull.astype(bool), "bull", "bear")),
        "vol": np.where(high.isna(), None, np.where(high.astype(bool), "high_vol", "low_vol")),
    }, index=index_ret.index)
