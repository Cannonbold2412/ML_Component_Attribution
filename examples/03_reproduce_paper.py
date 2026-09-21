"""Re-run the paper's complete MBCA grid on the bundled data (3 markets x rule gate
+ 5 ML components (3 regime labels) + 4 combinations x holdout/forward windows).

    python examples/03_reproduce_paper.py                 # all markets (~15-25 min)
    python examples/03_reproduce_paper.py commodities     # one market (~1 min)

Writes results/replication/<market>.csv, attribution_table.csv, opportunity_matrix.csv,
and the two figures. This is an independent re-implementation of the protocol, so the
numbers are close to but not identical with the paper's (see README "Fidelity").
"""
import sys
from pathlib import Path

import pandas as pd

import mbca
from mbca.report import forest_plot, matrix_plot, opportunity_matrix

OUT = Path(__file__).resolve().parents[1] / "results" / "replication"
OUT.mkdir(parents=True, exist_ok=True)

markets = sys.argv[1:] or list(mbca.MARKETS)
tables = []
for m in markets:
    cfg = mbca.MARKETS[m]
    res = mbca.run_mbca(mbca.load_market(m), mbca.paper_components(), cfg["split"], cfg["forward"],
                        base=cfg["pipeline"], market=m)
    res.table.to_csv(OUT / f"{m}.csv", index=False)
    tables.append(res.table)

table = pd.concat([pd.read_csv(OUT / f"{m}.csv") for m in mbca.MARKETS if (OUT / f"{m}.csv").exists()])
table.to_csv(OUT / "attribution_table.csv", index=False)
matrix = opportunity_matrix(table)
matrix.to_csv(OUT / "opportunity_matrix.csv", index=False)
forest_plot(table, OUT / "forest.png", "MBCA replication: delta Sharpe vs rule-based baseline")
matrix_plot(table, OUT / "opportunity_matrix.png", "ML Opportunity Matrix (replication)")

ml = table[table["kind"] != "control"]
print(f"\n{len(ml)} configurations with a CI: "
      f"{int((ml['significant'] & (ml['delta_sharpe'] > 0)).sum())} significant gains, "
      f"{int((ml['significant'] & (ml['delta_sharpe'] < 0)).sum())} significant losses")
print(matrix.to_string(index=False))
