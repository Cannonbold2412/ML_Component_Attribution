"""Paper vs. independent replication, config by config -> paper/tex/replication_table.tex
and a one-line summary. Run after examples/03_reproduce_paper.py.

    python paper/make_replication_table.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
paper = pd.read_csv(ROOT / "results" / "paper" / "component_attribution_table.csv")
rep = pd.read_csv(ROOT / "results" / "replication" / "attribution_table.csv")

REGIME = {"Rule-based ADX gate": "Rule-based ADX gate",
          "ML regime (linearity label)": "ML Regime Filter (linearity)",
          "ML regime (excursion label)": "ML Regime Filter (excursion)",
          "ML regime (move-size label)": "ML Regime Filter (move_size)"}
paper["variant"] = [REGIME.get(v, v) if c in ("ML Regime Filter", "Combinations") else c
                    for c, v in zip(paper["component"], paper["variant"])]
paper["market"] = paper["market"].replace({"forex_4pair": "forex"})

key = ["market", "window", "variant"]
m = paper[key + ["delta_sharpe", "significant"]].merge(
    rep[key + ["delta_sharpe", "significant"]], on=key, suffixes=("_paper", "_rep"))
m["sign_agrees"] = (m["delta_sharpe_paper"] > 0) == (m["delta_sharpe_rep"] > 0)


def verdict(d, s):
    return "sig.\\ loss" if s and d < 0 else "sig.\\ gain" if s else "n.s."


def summarize(g):
    return (f"{len(g)} configs; sign agrees {int(g.sign_agrees.sum())}/{len(g)}; "
            f"paper sig. gains/losses {int((g.significant_paper & (g.delta_sharpe_paper > 0)).sum())}/"
            f"{int((g.significant_paper & (g.delta_sharpe_paper < 0)).sum())}; "
            f"replication sig. gains/losses {int((g.significant_rep & (g.delta_sharpe_rep > 0)).sum())}/"
            f"{int((g.significant_rep & (g.delta_sharpe_rep < 0)).sum())}")


lines = [r"\begin{longtable}{l l l r r l l}",
         r"\caption{Paper vs.\ independent replication (\texttt{mbca}), $\Delta$Sharpe (annualized) per configuration.}"
         r" \label{tab:replication} \\",
         r"\toprule", r"Market & Window & Variant & Paper & Replication & Paper verdict & Replication verdict \\",
         r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
for r in m.itertuples():
    lines.append(f"{r.market.replace('_', chr(92) + '_')} & {r.window} & {r.variant.replace('_', chr(92) + '_')} & "
                 f"{r.delta_sharpe_paper:+.2f} & {r.delta_sharpe_rep:+.2f} & "
                 f"{verdict(r.delta_sharpe_paper, r.significant_paper)} & {verdict(r.delta_sharpe_rep, r.significant_rep)} \\\\")
lines.append(r"\end{longtable}")
(ROOT / "paper" / "tex" / "replication_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
m.to_csv(ROOT / "results" / "replication" / "paper_vs_replication.csv", index=False)

print("ALL:", summarize(m))
for mk, g in m.groupby("market"):
    print(f"{mk}:", summarize(g))
for comp in ["ML Exit", "Entry+Exit", "Exit+Sizing", "ML Meta-Labeling", "ML Position Sizing",
             "Best Combined (Meta-Label + Sizing)"]:
    g = m[m.variant == comp]
    print(f"{comp}: paper {g.delta_sharpe_paper.round(2).tolist()} | rep {g.delta_sharpe_rep.round(2).tolist()}")
