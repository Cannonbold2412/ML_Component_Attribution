"""Technical indicators used by the baseline strategy and the shared ML feature set.

Every function is backward-only: the value at row t depends on rows <= t only.
Pure numpy/pandas -- no TA-Lib, no numba.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def ema(s: pd.Series, length: int = 20) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def jma(price: pd.Series, length: int = 7, phase: float = 0.0, power: float = 2.0) -> pd.Series:
    """Jurik Moving Average (low-lag adaptive smoother). ``phase`` in [-100, 100]."""
    p = price.astype(float).to_numpy()
    out = np.empty(len(p))
    if len(p) == 0:
        return pd.Series(out, index=price.index)
    phase_ratio = np.clip(phase / 100.0 + 1.5, 0.5, 2.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power
    ma1 = det0 = det1 = jm = p[0]
    for i, x in enumerate(p):
        ma1 = (1 - alpha) * x + alpha * ma1
        det0 = (x - ma1) * (1 - beta) + beta * det0
        ma2 = ma1 + phase_ratio * det0
        det1 = (ma2 - jm) * (1 - alpha) ** 2 + alpha ** 2 * det1
        jm = jm + det1
        out[i] = jm
    return pd.Series(out, index=price.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    prev_c = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev_c), np.abs(l - prev_c)))  # fmax skips the NaN prev close
    return pd.Series(tr, index=df.index)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range (simple rolling mean of true range)."""
    return true_range(df).rolling(length, min_periods=length).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average Directional Index (Wilder directional movement, SMA smoothing)."""
    up, down = df["high"].diff(), -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).rolling(length, min_periods=1).mean()
    plus_di = 100 * plus_dm.rolling(length, min_periods=1).mean() / tr
    minus_di = 100 * minus_dm.rolling(length, min_periods=1).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(length, min_periods=1).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, 1e-12))


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
    return line - line.ewm(span=signal).mean()


def kama(close: pd.Series, length: int = 3, fast_ema: int = 2, slow_ema: int = 50) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    p = close.to_numpy(dtype=float)
    change = np.abs(np.concatenate([[np.nan] * length, p[length:] - p[:-length]]))
    vol = pd.Series(np.abs(np.diff(p, prepend=np.nan))).rolling(length).sum().to_numpy()
    er = change / np.where(vol == 0, np.nan, vol)
    fast_sc, slow_sc = 2 / (fast_ema + 1), 2 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.empty(len(p))
    if len(p):
        out[0] = p[0]
    for i in range(1, len(p)):
        out[i] = out[i - 1] if np.isnan(sc[i]) else out[i - 1] + sc[i] * (p[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def historical_volatility(close: pd.Series, length: int = 20, periods_per_year: int = 252) -> pd.Series:
    return close.pct_change().rolling(length).std() * np.sqrt(periods_per_year)


def rolling_zscore(x: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    """z-score of the latest value within its trailing window (population std)."""
    r = x.rolling(window, min_periods=min_periods)
    return (x - r.mean()) / (r.std(ddof=0) + 1e-8)


def rolling_r2(series: pd.Series, window: int = 20) -> pd.Series:
    """R^2 of a linear time-trend fit over the ``window`` bars *before* t
    (bars t-window..t-1). Equals corr(time, price)^2; a flat window scores 1.0
    (a constant is fit perfectly), matching sklearn's ``LinearRegression.score``."""
    y = series.to_numpy(dtype=float)
    out = np.full(len(y), np.nan)
    if len(y) <= window:
        return pd.Series(out, index=series.index)
    w = sliding_window_view(y, window)[: len(y) - window]  # w[k] = y[k : k+window] -> row k+window
    x = np.arange(window) - (window - 1) / 2
    yc = w - w.mean(axis=1, keepdims=True)
    sxy, syy = yc @ x, (yc ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = sxy ** 2 / ((x ** 2).sum() * syy)
    r2[syy == 0] = 1.0
    out[window:] = r2
    return pd.Series(out, index=series.index)
