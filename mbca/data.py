"""Bundled daily OHLC data and the paper's per-market configuration.

Each market ships as one CSV per instrument under ``data/<market>/``. The
freeze (``split``) and forward-window (``forward``) dates are the ones the paper
used: ``split`` = 70% of the original study tree's calendar span, ``forward`` =
the last date of that original tree (everything after it was fetched later and
never seen while the study was designed).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .strategy import Pipeline, bracket_exit, trailing_exit

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAILING_GRID = {"sl_mult": [1.0, 1.5, 2.0], "trail_mult": [1.5, 2.0, 3.0]}

MARKETS = {
    "forex": dict(split="2023-01-14", forward=None,  # AUDNZD, CADJPY, AUDUSD, NZDUSD
                  pipeline=Pipeline(exit_fn=bracket_exit)),
    "commodities": dict(split="2020-07-03", forward="2025-07-05",  # WTI, Brent, NatGas, Gold, Silver
                        pipeline=Pipeline(exit_fn=trailing_exit, stop_grid=TRAILING_GRID)),
    "indian_equities": dict(split="2021-12-03", forward="2024-11-08",  # 48 NIFTY-50 constituents
                            pipeline=Pipeline(exit_fn=trailing_exit, stop_grid=TRAILING_GRID)),
}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read one OHLC CSV (columns Date, open, high, low, close[, volume])."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = [c if c == "Date" else c.lower() for c in df.columns]
    return df.sort_values("Date").reset_index(drop=True)


def load_market(market: str, tickers: list[str] | None = None, data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    files = sorted((Path(data_dir) / market).glob("*.csv*"))
    if not files:
        raise FileNotFoundError(f"no CSVs under {Path(data_dir) / market}")
    frames = {f.name.split(".")[0]: load_csv(f) for f in files}
    return {t: frames[t] for t in tickers} if tickers else frames


def synthetic_ohlc(n: int = 1500, seed: int = 0, start: str = "2010-01-01", trend_strength: float = 0.15) -> pd.DataFrame:
    """Regime-switching random walk (alternating trending / choppy 60-bar stretches)
    for tests and demos -- gives trend-following something to find without real data."""
    import numpy as np

    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 0, 1], size=n // 60 + 1), 60)[:n]
    ret = rng.normal(0, 0.01, n) + trend_strength * 0.01 * regime
    close = 100 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.006, n)) * close
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "Date": pd.bdate_range(start, periods=n), "open": open_,
        "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
        "close": close, "volume": rng.integers(1_000, 10_000, n),
    })
