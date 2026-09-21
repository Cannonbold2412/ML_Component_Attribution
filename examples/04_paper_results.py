"""Explore the paper's published numbers (results/paper/, exported from the original
study pipeline) and redraw its figures with the unit-corrected forest plot. Also
regenerates the paper's Figures 1-2 (paper/figures/).

    python examples/04_paper_results.py
"""
from pathlib import Path

import pandas as pd

from mbca.report import forest_plot, matrix_plot, opportunity_matrix

ROOT = Path(__file__).resolve().parents[1]
paper = pd.read_csv(ROOT / "results" / "paper" / "component_attribution_table.csv")

KIND = {"ML Regime Filter": "independent", "ML Meta-Labeling": "narrowing", "ML Position Sizing": "narrowing",
        "ML Entry": "replacing", "ML Exit": "replacing", "Combinations": "combination"}
table = paper.rename(columns={"variant_sharpe": "sharpe", "bootstrap_ci_lo": "ci_lo", "bootstrap_ci_hi": "ci_hi"})
table["kind"] = table["component"].map(KIND)
table["variant"] = table.apply(lambda r: r["variant"] if r["component"] in ("ML Regime Filter", "Combinations")
                               else r["component"], axis=1)

ml = table
print(f"{len(ml)} configurations: {int((ml.significant & (ml.delta_sharpe > 0)).sum())} significant gains, "
      f"{int((ml.significant & (ml.delta_sharpe < 0)).sum())} significant losses\n")
print(opportunity_matrix(table).to_string(index=False))

out = ROOT / "results" / "paper"
print("\n->", forest_plot(table, out / "forest_corrected.png", "Paper results: delta Sharpe vs rule-based baseline"))
print("->", matrix_plot(table, out / "opportunity_matrix.png", "ML Opportunity Matrix (paper results)"))

figs = ROOT / "paper" / "figures"
forest_plot(table, figs / "p36_fig1_component_attribution_forest.png", "All 60 configurations: delta Sharpe vs rule-based baseline")
forest_plot(table[table["kind"] == "combination"], figs / "p36_fig2_combinations_forest.png", "Combinations: delta Sharpe vs rule-based baseline")
print("-> paper figures 1-2 regenerated in", figs)
