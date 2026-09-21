"""The rule-based pipeline (MBCA's control) and the anchored walk-forward harness.

A pipeline has five insertion points, each a plain callable -- this is the seam
the ML components (``mbca.components``) and your own strategies plug into:

    signal_fn(df, **sparams) -> Series in {1, -1, 0}     entry generation
    gate_fn(df)              -> bool Series               regime filter / meta-label
    exit_fn(df, signal, **rparams) -> trades DataFrame    exit policy
    sizer(df, trades)        -> Series of multipliers     position sizing

Close-only execution: entries fill at the signal bar's close, exits at the
stop level (or the bar's close for learned exits). One position per instrument
at a time; a new entry is only considered after the previous trade exits.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from .indicators import adx, atr, jma

TRADE_COLS = ["entry_time", "exit_time", "entry_i", "exit_i", "direction",
              "entry_price", "exit_price", "reason", "ret"]


# --------------------------------------------------------------------------- signals
def crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 where ``fast`` crosses above ``slow``, -1 where it crosses below, else 0."""
    up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return pd.Series(np.select([up, dn], [1, -1], 0), index=fast.index)


def jma_signals(df: pd.DataFrame, fast: int = 7, slow: int = 21) -> pd.Series:
    """The paper's baseline entry rule: JMA(fast)/JMA(slow) crossover."""
    return crossover_signal(jma(df["close"], fast), jma(df["close"], slow))


def adx_gate(df: pd.DataFrame, threshold: float = 25.0) -> pd.Series:
    """Rule-based regime gate: trade only when ADX(14) >= threshold."""
    return (adx(df) >= threshold).fillna(False)


# --------------------------------------------------------------------------- exits
def _trades_frame(rows: list[tuple], df: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TRADE_COLS)
    t = pd.DataFrame(rows, columns=["entry_i", "exit_i", "direction", "entry_price", "exit_price", "reason"])
    dates = pd.to_datetime(df["Date"]).to_numpy()
    t["entry_time"], t["exit_time"] = dates[t["entry_i"]], dates[t["exit_i"]]
    t["ret"] = t["direction"] * (t["exit_price"] - t["entry_price"]) / t["entry_price"] - fee_bps / 1e4
    return t[TRADE_COLS]


def _scan(df: pd.DataFrame, signal: pd.Series, fee_bps: float, find_exit) -> pd.DataFrame:
    """Generic one-position-at-a-time loop. ``find_exit(i, d, a)`` returns
    ``(exit_i, exit_price, reason)`` for an entry at bar i in direction d with ATR a."""
    price = df["close"].to_numpy(dtype=float)
    sig = np.asarray(signal, dtype=int)
    atr_v = atr(df).to_numpy()
    rows, i, n = [], 0, len(df)
    while i < n - 1:
        d = sig[i]
        if d == 0 or np.isnan(atr_v[i]):  # ATR warm-up: stop level undefined, skip
            i += 1
            continue
        j, px, reason = find_exit(i, d, atr_v[i])
        rows.append((i, j, d, price[i], px, reason))
        i = j + 1
    return _trades_frame(rows, df, fee_bps)


def bracket_exit(df: pd.DataFrame, signal: pd.Series, sl_mult: float = 1.5, tp_mult: float = 3.0,
                 fee_bps: float = 7.0) -> pd.DataFrame:
    """Fixed ATR stop / ATR target, evaluated on closes."""
    price = df["close"].to_numpy(dtype=float)
    n = len(price)

    def find_exit(i, d, a):
        sl, tp = price[i] - d * sl_mult * a, price[i] + d * tp_mult * a
        fwd = price[i + 1:]
        stop_hit = fwd <= sl if d == 1 else fwd >= sl
        tgt_hit = fwd >= tp if d == 1 else fwd <= tp
        hits = np.flatnonzero(stop_hit | tgt_hit)
        if not hits.size:
            return n - 1, price[-1], "eod"
        k = hits[0]
        return (i + 1 + k, sl, "stop") if stop_hit[k] else (i + 1 + k, tp, "target")

    return _scan(df, signal, fee_bps, find_exit)


def trailing_exit(df: pd.DataFrame, signal: pd.Series, sl_mult: float = 1.5, trail_mult: float = 2.0,
                  fee_bps: float = 7.0) -> pd.DataFrame:
    """Initial ATR stop that ratchets to ``trail_mult * ATR`` behind the best close."""
    price = df["close"].to_numpy(dtype=float)
    atr_v = atr(df).to_numpy()
    n = len(price)

    def find_exit(i, d, a):
        fwd = price[i + 1:]
        fa = np.where(np.isnan(atr_v[i + 1:]), a, atr_v[i + 1:])
        init = price[i] - d * sl_mult * a
        if d == 1:
            stop = np.maximum.accumulate(np.maximum(fwd - trail_mult * fa, init))
            hits = np.flatnonzero(fwd <= stop)
        else:
            stop = np.minimum.accumulate(np.minimum(fwd + trail_mult * fa, init))
            hits = np.flatnonzero(fwd >= stop)
        if not hits.size:
            return n - 1, price[-1], "eod"
        return i + 1 + hits[0], stop[hits[0]], "stop"

    return _scan(df, signal, fee_bps, find_exit)


# --------------------------------------------------------------------------- pipeline
@dataclass(frozen=True)
class Pipeline:
    """A complete hybrid trading pipeline. ``signal_grid``/``stop_grid`` are the
    parameter grids the walk-forward harness selects from on each in-sample fold."""
    signal_fn: Callable = jma_signals
    signal_grid: dict = field(default_factory=lambda: {"fast": [5, 7, 9], "slow": [21, 28]})
    exit_fn: Callable = bracket_exit
    stop_grid: dict = field(default_factory=lambda: {"sl_mult": [1.0, 1.5, 2.0], "tp_mult": [1.5, 2.0, 3.0, 4.0]})
    gate_fn: Callable | None = None
    sizer: Callable | None = None
    fee_bps: float = 7.0

    def with_(self, **changes) -> "Pipeline":
        return replace(self, **changes)

    def signal(self, df: pd.DataFrame, gate=None, **sparams) -> pd.Series:
        """Entry signal after gating. Pass a precomputed ``gate`` mask to skip recomputing it."""
        s = pd.Series(np.asarray(self.signal_fn(df, **sparams), dtype=int), index=df.index)
        if gate is None and self.gate_fn is not None:
            gate = self.gate_fn(df)
        return s if gate is None else s.where(np.asarray(gate, dtype=bool), 0)

    def trades(self, df: pd.DataFrame, sparams: dict, rparams: dict, signal: pd.Series | None = None) -> pd.DataFrame:
        sig = self.signal(df, **sparams) if signal is None else signal
        t = self.exit_fn(df, sig, fee_bps=self.fee_bps, **rparams)
        if self.sizer is not None and len(t):
            t = t.assign(ret=t["ret"].to_numpy() * np.asarray(self.sizer(df, t), dtype=float))
        return t


def _combos(grid: dict) -> list[dict]:
    return [dict(zip(grid, v)) for v in product(*grid.values())] if grid else [{}]


def walk_forward(df: pd.DataFrame, pipe: Pipeline, n_splits: int = 5, min_trades: int = 10) -> pd.DataFrame:
    """Anchored-by-fold walk-forward on one instrument: split the history into
    ``n_splits`` equal chronological chunks; on chunk k pick the (signal, stop)
    params with the best mean trade return (>= ``min_trades`` trades), then trade
    chunk k+1 with them. Returns the concatenated out-of-sample trades.

    Sizing is not part of selection: params are chosen on the unsized trade set,
    so a sizer can never change *which* trades are taken."""
    df = df.sort_values("Date").reset_index(drop=True)
    size = len(df) // n_splits
    chunks = [df.iloc[k * size:(k + 1) * size].reset_index(drop=True) for k in range(n_splits)]
    unsized = pipe.with_(sizer=None)
    out = []
    for k in range(n_splits - 1):
        best, best_ret = None, -np.inf
        gate = pipe.gate_fn(chunks[k]) if pipe.gate_fn is not None else None
        for sp in _combos(pipe.signal_grid):
            sig = unsized.signal(chunks[k], gate, **sp)
            for rp in _combos(pipe.stop_grid):
                t = unsized.trades(chunks[k], sp, rp, signal=sig)
                if len(t) >= min_trades and t["ret"].mean() > best_ret:
                    best, best_ret = (sp, rp), t["ret"].mean()
        if best is None:
            continue
        t = pipe.trades(chunks[k + 1], *best)
        if len(t):
            out.append(t.assign(fold=k))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=TRADE_COLS + ["fold"])
