"""The five ML insertion points, each a matched-budget replacement for one piece
of the rule-based pipeline. Every component:

* uses the same 8 backward-only features (``features.FEATURE_COLS``) and the
  same LightGBM hyperparameters (``features.LGBM_PARAMS``);
* is fit exactly once on data before ``split_date``, and never on a row whose
  label looks past ``split_date`` (the freeze / leakage guard);
* exposes ``apply(pipe) -> Pipeline``, replacing exactly one insertion point,
  so components compose: ``Exit().apply(Entry().apply(base))`` is Entry+Exit.

Taxonomy (paper Sec. 4.2): independent = RegimeFilter; narrowing = MetaLabel,
PositionSizing; replacing = Entry, Exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .features import FEATURE_COLS, ProbModel, compute_features
from .indicators import atr
from .strategy import Pipeline, _scan

Frames = dict[str, pd.DataFrame]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(Date=pd.to_datetime(df["Date"])).sort_values("Date").reset_index(drop=True)


def _eligible(df: pd.DataFrame, feats: pd.DataFrame, split_date, horizon: int) -> pd.Series:
    """Rows usable for training: before the split AND whose ``horizon``-bar label
    window also ends before the split (a straddling label would leak holdout data)."""
    d = df["Date"]
    return (d < split_date) & (d.shift(-horizon) < split_date) & feats.notna().all(axis=1)


def _pool(frames: Frames, split_date, horizon: int, label_fn) -> tuple[pd.DataFrame, pd.Series]:
    Xs, ys = [], []
    for df in frames.values():
        df = _prep(df)
        feats, y = compute_features(df), label_fn(df)
        ok = _eligible(df, feats, split_date, horizon) & y.notna()
        Xs.append(feats.loc[ok])
        ys.append(y.loc[ok])
    X, y = pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)
    if X.empty:
        raise ValueError("no eligible pre-split training rows")
    return X, y.astype(int)


def _proba(model: ProbModel, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(compute_features(df)[FEATURE_COLS])


def _forward_windows(close: np.ndarray, horizon: int) -> np.ndarray:
    """(n, horizon) matrix: row t = close[t+1 .. t+horizon], NaN-padded at the end."""
    padded = np.concatenate([close, np.full(horizon, np.nan)])
    return sliding_window_view(padded[1:], horizon)[: len(close)]


# --------------------------------------------------------------------------- labels
def label_linearity(df, horizon=15, r2_thresh=0.5):
    """Trend if the next ``horizon`` closes fit a straight line with R^2 >= thresh."""
    w = _forward_windows(df["close"].to_numpy(float), horizon)
    x = np.arange(horizon) - (horizon - 1) / 2
    yc = w - w.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = (yc @ x) ** 2 / ((x ** 2).sum() * (yc ** 2).sum(axis=1))
    out = np.where(np.isnan(r2), 0.0, (r2 >= r2_thresh).astype(float))
    out[np.isnan(w).any(axis=1)] = np.nan
    return pd.Series(out, index=df.index)


def label_excursion(df, horizon=15, k=1.5):
    """Trend if, in the direction of the net forward move, the max favourable
    excursion reaches k*ATR before the adverse excursion breaches -k*ATR."""
    close = df["close"].to_numpy(float)
    a = atr(df).to_numpy()
    w = _forward_windows(close, horizon) - close[:, None]
    direction = np.sign(w[:, -1])
    direction[direction == 0] = 1.0
    fav = direction[:, None] * w
    mfe_idx = np.nanargmax(np.where(np.isnan(fav), -np.inf, fav), axis=1)
    mfe = fav[np.arange(len(fav)), mfe_idx]
    mae = np.array([fav[t, : mfe_idx[t] + 1].min() for t in range(len(fav))])
    out = ((mfe >= k * a) & (mae > -k * a)).astype(float)
    out[np.isnan(w).any(axis=1) | np.isnan(a) | (a == 0)] = np.nan
    return pd.Series(out, index=df.index)


def label_move_size(df, horizon=15, k=1.5):
    """Trend if |close[t+horizon] - close[t]| >= k*ATR[t], regardless of path."""
    close = df["close"].to_numpy(float)
    a = atr(df).to_numpy()
    fwd = pd.Series(close).shift(-horizon).to_numpy()
    out = (np.abs(fwd - close) >= k * a).astype(float)
    out[np.isnan(fwd) | np.isnan(a) | (a == 0)] = np.nan
    return pd.Series(out, index=df.index)


def label_bracket(df, direction, horizon=40, sl_mult=1.5, tp_mult=3.0):
    """1 if a ``direction`` entry at t's close hits its tp_mult*ATR target before its
    sl_mult*ATR stop within ``horizon`` closes; 0 if the stop comes first; NaN if
    neither is touched (never fabricated as a resolved outcome)."""
    close = df["close"].to_numpy(float)
    a = atr(df).to_numpy()
    w = _forward_windows(close, horizon)
    sl = (close - direction * sl_mult * a)[:, None]
    tp = (close + direction * tp_mult * a)[:, None]
    stop_hit = (w <= sl) if direction == 1 else (w >= sl)
    tgt_hit = (w >= tp) if direction == 1 else (w <= tp)
    first = lambda m: np.where(m.any(axis=1), m.argmax(axis=1), np.inf)
    s, t = first(stop_hit), first(tgt_hit)
    out = (t < s).astype(float)
    out[(s == np.inf) & (t == np.inf)] = np.nan
    out[np.isnan(a) | (a == 0)] = np.nan
    return pd.Series(out, index=df.index)


def label_continuation(df, direction, horizon=10):
    """1 if a ``direction`` position is still in profit ``horizon`` bars later."""
    close = df["close"]
    fwd = close.shift(-horizon)
    return (direction * (fwd - close) > 0).astype(float).where(fwd.notna() & close.notna())


# --------------------------------------------------------------------------- components
def fit_once(component, frames: Frames, split_date, base: Pipeline | None = None):
    """Fit unless this exact instance was already fit on these frames and split --
    lets one frozen model be shared by a single-component run and the combinations."""
    key = (id(frames), pd.Timestamp(split_date))
    if getattr(component, "_fit_key", None) != key:
        component.fit(frames, split_date, base)
        component._fit_key = key
    return component


class RegimeFilter:
    """Independent: a trend/chop classifier gates every entry, blind to the signal."""
    kind = "independent"
    LABELS = {"linearity": label_linearity, "excursion": label_excursion, "move_size": label_move_size}

    def __init__(self, label: str = "linearity", threshold: float = 0.5, horizon: int = 15):
        self.label, self.threshold, self.horizon = label, threshold, horizon
        self.name = f"ML Regime Filter ({label})"

    def fit(self, frames: Frames, split_date, base: Pipeline | None = None) -> "RegimeFilter":
        fn = self.LABELS[self.label]
        self.model = ProbModel().fit(*_pool(frames, split_date, self.horizon, lambda df: fn(df, self.horizon)))
        return self

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(_proba(self.model, df), nan=-1) >= self.threshold

    def apply(self, pipe: Pipeline) -> Pipeline:
        return pipe.with_(gate_fn=self.mask)


class MetaLabel:
    """Narrowing: P(this signal-fired trade is profitable). Rejects trades below
    ``threshold``; ``PositionSizing`` consumes the same frozen model continuously."""
    kind = "narrowing"
    name = "ML Meta-Labeling"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, frames: Frames, split_date, base: Pipeline | None = None) -> "MetaLabel":
        """Label = realized return > 0 of the base pipeline's own trades at the
        signal/exit functions' default parameters, using only trades whose entry
        AND exit both precede the split."""
        base = base or Pipeline()
        Xs, ys = [], []
        for df in frames.values():
            df = _prep(df)
            t = base.with_(gate_fn=None, sizer=None).trades(df, {}, {})
            t = t[pd.to_datetime(t["exit_time"]) < split_date]
            feats = compute_features(df).iloc[t["entry_i"].to_numpy(int)]
            ok = feats.notna().all(axis=1).to_numpy()
            Xs.append(feats[ok])
            ys.append((t["ret"].to_numpy()[ok] > 0).astype(int))
        X = pd.concat(Xs, ignore_index=True)
        if X.empty:
            raise ValueError("no resolved pre-split trades to meta-label")
        self.model = ProbModel().fit(X, pd.Series(np.concatenate(ys)))
        return self

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(_proba(self.model, df), nan=-1) >= self.threshold

    def apply(self, pipe: Pipeline) -> Pipeline:
        return pipe.with_(gate_fn=self.mask)


class PositionSizing:
    """Narrowing: never rejects a trade; scales each trade's return by
    m = min_mult + p * (max_mult - min_mult), p = the meta-label win probability.
    p = 0.5 -> 1.0x (baseline capital). Unscoreable entries get 1.0x."""
    kind = "narrowing"
    name = "ML Position Sizing"

    def __init__(self, meta: MetaLabel | None = None, min_mult: float = 0.0, max_mult: float = 2.0):
        self.meta, self.min_mult, self.max_mult = meta, min_mult, max_mult

    def fit(self, frames: Frames, split_date, base: Pipeline | None = None) -> "PositionSizing":
        self.meta = fit_once(self.meta or MetaLabel(), frames, split_date, base)
        return self

    def multipliers(self, df: pd.DataFrame, trades: pd.DataFrame) -> np.ndarray:
        p = _proba(self.meta.model, df)[trades["entry_i"].to_numpy(int)]
        return np.where(np.isnan(p), 1.0, self.min_mult + p * (self.max_mult - self.min_mult))

    def apply(self, pipe: Pipeline) -> Pipeline:
        return pipe.with_(sizer=self.multipliers)


class Entry:
    """Replacing: two classifiers (long-viable, short-viable, labelled by a 1.5/3.0
    ATR bracket over 40 bars) generate entries directly, replacing the signal rule."""
    kind = "replacing"
    name = "ML Entry"

    def __init__(self, threshold: float = 0.5, horizon: int = 40, sl_mult: float = 1.5, tp_mult: float = 3.0):
        self.threshold, self.horizon, self.sl_mult, self.tp_mult = threshold, horizon, sl_mult, tp_mult

    def fit(self, frames: Frames, split_date, base: Pipeline | None = None) -> "Entry":
        lab = lambda d: (lambda df: label_bracket(df, d, self.horizon, self.sl_mult, self.tp_mult))
        self.long = ProbModel().fit(*_pool(frames, split_date, self.horizon, lab(1)))
        self.short = ProbModel().fit(*_pool(frames, split_date, self.horizon, lab(-1)))
        return self

    def signals(self, df: pd.DataFrame, **_ignored) -> pd.Series:
        pl = np.nan_to_num(_proba(self.long, df), nan=-1)
        ps = np.nan_to_num(_proba(self.short, df), nan=-1)
        sig = np.where((pl >= self.threshold) & (pl > ps), 1, np.where((ps >= self.threshold) & (ps > pl), -1, 0))
        return pd.Series(sig, index=df.index)

    def apply(self, pipe: Pipeline) -> Pipeline:
        return pipe.with_(signal_fn=self.signals, signal_grid={})


class Exit:
    """Replacing: two continuation classifiers decide hold/exit at every bar; the
    position closes at the first bar P(still favourable in 10 bars) < threshold,
    or at a wide 3*ATR safety-net stop."""
    kind = "replacing"
    name = "ML Exit"

    def __init__(self, threshold: float = 0.5, horizon: int = 10, sl_mult: float = 3.0):
        self.threshold, self.horizon, self.sl_mult = threshold, horizon, sl_mult

    def fit(self, frames: Frames, split_date, base: Pipeline | None = None) -> "Exit":
        lab = lambda d: (lambda df: label_continuation(df, d, self.horizon))
        self.long = ProbModel().fit(*_pool(frames, split_date, self.horizon, lab(1)))
        self.short = ProbModel().fit(*_pool(frames, split_date, self.horizon, lab(-1)))
        return self

    def exit_fn(self, df: pd.DataFrame, signal: pd.Series, sl_mult: float | None = None,
                fee_bps: float = 7.0) -> pd.DataFrame:
        sl_mult = self.sl_mult if sl_mult is None else sl_mult
        price = df["close"].to_numpy(float)
        proba = {1: _proba(self.long, df), -1: _proba(self.short, df)}
        n = len(price)

        def find_exit(i, d, a):
            sl = price[i] - d * sl_mult * a
            fwd = price[i + 1:]
            stop_hit = fwd <= sl if d == 1 else fwd >= sl
            with np.errstate(invalid="ignore"):
                ml_hit = proba[d][i + 1:] < self.threshold  # NaN -> False (hold)
            hits = np.flatnonzero(stop_hit | ml_hit)
            if not hits.size:
                return n - 1, price[-1], "eod"
            k = hits[0]
            return (i + 1 + k, sl, "stop") if stop_hit[k] else (i + 1 + k, fwd[k], "ml_exit")

        return _scan(df, signal, fee_bps, find_exit)

    def apply(self, pipe: Pipeline) -> Pipeline:
        return pipe.with_(exit_fn=self.exit_fn, stop_grid={"sl_mult": [self.sl_mult]})


class RuleGate:
    """A rule-based (non-ML) gate as a comparison variant, e.g. the paper's ADX>=25
    filter: ``RuleGate(adx_gate)``. Nothing to fit."""
    kind = "independent"

    def __init__(self, gate_fn, name: str = "Rule-based ADX gate"):
        self.gate_fn, self.name = gate_fn, name

    def fit(self, frames: Frames, split_date, base: Pipeline | None = None) -> "RuleGate":
        return self

    def apply(self, pipe: Pipeline) -> Pipeline:
        return pipe.with_(gate_fn=self.gate_fn)


class Combo:
    """Compose components left to right, e.g. ``Combo(MetaLabel(), PositionSizing())``."""

    def __init__(self, *parts, name: str | None = None):
        self.parts = parts
        self.name = name or " + ".join(p.name for p in parts)
        self.kind = "combination"

    def fit(self, frames: Frames, split_date, base: Pipeline | None = None) -> "Combo":
        for p in self.parts:  # a PositionSizing reuses a MetaLabel in the same combo (one frozen model)
            if isinstance(p, PositionSizing) and p.meta is None:
                p.meta = next((q for q in self.parts if isinstance(q, MetaLabel)), None)
            fit_once(p, frames, split_date, base)
        return self

    def apply(self, pipe: Pipeline) -> Pipeline:
        for p in self.parts:
            pipe = p.apply(pipe)
        return pipe
