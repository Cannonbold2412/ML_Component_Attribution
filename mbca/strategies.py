"""The five historically documented strategies of Study 2, as entry-signal functions.

Every function maps an OHLC frame to a Series of +1 / -1 / 0 using data up to and
including bar t only (tests/test_study2.py checks truncation invariance). Exits are
not part of the signal: study.toml pairs each strategy with a trailing or bracket exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import rsi
from .strategy import Pipeline, bracket_exit, crossover_signal, jma_signals, trailing_exit


def sma_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Dual moving-average crossover (Brock, Lakonishok & LeBaron 1992)."""
    c = df["close"]
    return crossover_signal(c.rolling(fast).mean(), c.rolling(slow).mean())


def donchian(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Channel breakout: close above the prior ``lookback``-day high / below the prior low."""
    hi = df["high"].rolling(lookback).max().shift(1)
    lo = df["low"].rolling(lookback).min().shift(1)
    return (df["close"] > hi).astype(int) - (df["close"] < lo).astype(int)


def ts_momentum(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Time-series momentum (Moskowitz, Ooi & Pedersen 2012): signal when the sign of
    the trailing ``lookback``-day return flips."""
    sign = np.sign(df["close"].pct_change(lookback))
    flip = (sign != sign.shift(1)) & sign.shift(1).notna() & (sign != 0)
    return sign.where(flip, 0).fillna(0).astype(int)


def rsi2(df: pd.DataFrame, entry: float = 10) -> pd.Series:
    """RSI(2) mean reversion (Connors & Alvarez 2008): buy oversold dips in an uptrend
    (close > 200-day SMA), sell overbought rallies in a downtrend."""
    r, c = rsi(df["close"], 2), df["close"]
    trend = c.rolling(200).mean()
    long_ = (r < entry) & (c > trend)
    short = (r > 100 - entry) & (c < trend)
    return long_.astype(int) - short.astype(int)


SIGNALS = {"jma_cross": jma_signals, "sma_cross": sma_cross, "donchian": donchian,
           "ts_momentum": ts_momentum, "rsi2": rsi2}
EXITS = {"trailing": trailing_exit, "bracket": bracket_exit}


def build_pipeline(strategy: dict, cost_bps: float, fill: str = "close", lag: int = 0) -> Pipeline:
    """A study.toml [strategies.X] entry -> Pipeline (the rule-based control)."""
    return Pipeline(signal_fn=SIGNALS[strategy["signal"]], signal_grid=dict(strategy["signal_grid"]),
                    exit_fn=EXITS[strategy["exit"]], stop_grid=dict(strategy["stop_grid"]),
                    fee_bps=cost_bps, lag=lag, exit_kwargs={"fill": fill})
