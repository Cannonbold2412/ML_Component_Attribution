"""Quickstart: does ML meta-labeling / position sizing improve the JMA+ATR baseline
on the bundled commodities data? (~1 minute)

    python examples/01_quickstart.py
"""
import pandas as pd

import mbca

pd.set_option("display.width", 200)

cfg = mbca.MARKETS["commodities"]           # freeze 2020-07-03, forward window from 2025-07-05
frames = mbca.load_market("commodities")    # {ticker: OHLC DataFrame}

meta = mbca.MetaLabel()
sizing = mbca.PositionSizing(meta)          # same frozen model, used as a size multiplier
components = [meta, sizing, mbca.Combo(meta, sizing, name="Meta-Label + Sizing")]

res = mbca.run_mbca(frames, components, split_date=cfg["split"], forward_start=cfg["forward"],
                    base=cfg["pipeline"], market="commodities")

cols = ["window", "variant", "n_trades", "sharpe", "delta_sharpe", "deflated_sharpe", "ci_lo", "ci_hi", "significant"]
print(res.table[cols].round(3).to_string(index=False))
print()
print(res.matrix().to_string(index=False))
