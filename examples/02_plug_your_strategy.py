"""Plug in YOUR strategy and find out where ML helps it.

Edit the three marked blocks below, then:

    python examples/02_plug_your_strategy.py                       # bundled commodities data
    python examples/02_plug_your_strategy.py path/to/csv_folder    # your own daily OHLC CSVs

A CSV needs columns: Date, open, high, low, close (volume optional), one file per instrument.
"""
import sys
from pathlib import Path

import pandas as pd

import mbca
from mbca.report import forest_plot

pd.set_option("display.width", 200)


# ---- 1. YOUR ENTRY RULE: df -> Series of +1 (long) / -1 (short) / 0 (flat), one value per bar.
#         Use only data up to each bar (no .shift(-k)!). Keyword args become the tuning grid.
def donchian_breakout(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    hi = df["high"].rolling(lookback).max().shift(1)
    lo = df["low"].rolling(lookback).min().shift(1)
    return (df["close"] > hi).astype(int) - (df["close"] < lo).astype(int)


my_strategy = mbca.Pipeline(
    signal_fn=donchian_breakout,
    signal_grid={"lookback": [20, 55]},            # walk-forward picks the best per fold
    # ---- 2. YOUR EXIT: bracket_exit (ATR stop + target) or trailing_exit (ATR trailing stop),
    #         or any fn(df, signal, fee_bps=..., **params) -> trades DataFrame.
    exit_fn=mbca.trailing_exit,
    stop_grid={"sl_mult": [1.5, 2.0], "trail_mult": [2.0, 3.0]},
    fee_bps=7.0,                                   # round-trip cost per trade
)

# ---- 3. WHICH ML INSERTION POINTS TO TEST (any subset of mbca.paper_components()).
meta = mbca.MetaLabel()
components = [
    mbca.RegimeFilter("linearity"),                # independent: ML trend/chop gate
    meta,                                          # narrowing: reject low-probability trades
    mbca.PositionSizing(meta),                     # narrowing: size by win probability
    mbca.Entry(),                                  # replacing: ML generates entries
    mbca.Exit(),                                   # replacing: ML decides when to exit
]

if len(sys.argv) > 1:
    folder = Path(sys.argv[1])
    frames = {f.name.split(".")[0]: mbca.load_csv(f) for f in sorted(folder.glob("*.csv*"))}
    dates = pd.concat([d["Date"] for d in frames.values()])
    split, forward = dates.min() + 0.7 * (dates.max() - dates.min()), None   # freeze at 70% of the span
else:
    frames = mbca.load_market("commodities")
    split, forward = mbca.MARKETS["commodities"]["split"], mbca.MARKETS["commodities"]["forward"]

res = mbca.run_mbca(frames, components, split_date=split, forward_start=forward, base=my_strategy, market="my_market")
cols = ["window", "variant", "kind", "n_trades", "sharpe", "delta_sharpe", "ci_lo", "ci_hi", "significant"]
print(res.table[cols].round(3).to_string(index=False))
print()
print(res.matrix().to_string(index=False))
out = Path("results") / "my_strategy_forest.png"
out.parent.mkdir(exist_ok=True)
print("\nforest plot ->", forest_plot(res.table, out, "Where does ML help my strategy?"))
