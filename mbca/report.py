"""Figures: the forest plot and the ML Opportunity Matrix.

Both take an attribution table in the ``run_mbca`` schema (one row per market x
window x variant with delta_sharpe, ci_lo, ci_hi, significant, kind).

Units: ``ci_lo``/``ci_hi`` are per-day Sharpe differences; the forest plot scales
them by sqrt(252) so point and interval share the annualized axis of
``delta_sharpe``. (The paper's original Figure 3 drew the per-day interval
around the annualized point -- see ERRATA.md.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import PERIODS, recommendation

GAIN, LOSS, NS_POS, NS_NEG, NEUTRAL = "#1a7f37", "#cf222e", "#a6dba0", "#f4a6a6", "#e8e8e8"
KIND_COLOR = {"independent": "#6e7781", "narrowing": "#0969da", "replacing": "#bc4c00",
              "combination": "#8250df", "control": "#000000"}


def forest_plot(table: pd.DataFrame, path: str | Path, title: str = "MBCA: delta Sharpe vs rule-based baseline") -> Path:
    import matplotlib.pyplot as plt

    t = table[table["kind"] != "control"].copy()
    order = {v: i for i, v in enumerate(dict.fromkeys(t["variant"]))}
    t = t.sort_values(["market", "variant", "window"], key=lambda c: c.map(order) if c.name == "variant" else c,
                      kind="stable", ascending=[True, True, False])
    t["label"] = t["variant"] + " [" + t["window"] + "]"
    markets = list(dict.fromkeys(t["market"]))
    rows = max(len(t[t["market"] == m]) for m in markets)
    fig, axes = plt.subplots(1, len(markets), figsize=(5.5 * len(markets), 0.32 * rows + 1.8), squeeze=False)
    k = np.sqrt(PERIODS)
    for ax, m in zip(axes[0], markets):
        sub = t[t["market"] == m].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        for yi, r in zip(y, sub.itertuples()):
            c = (GAIN if r.delta_sharpe > 0 else LOSS) if r.significant else "#57606a"
            if np.isfinite(r.ci_lo):
                ax.plot([r.ci_lo * k, r.ci_hi * k], [yi, yi], color=c, lw=1.4)
            ax.scatter([r.delta_sharpe], [yi], s=34, color=c, facecolors=c if r.significant else "white", zorder=3)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y, sub["label"], fontsize=7)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("delta Sharpe, annualized (95% CI x sqrt(252))", fontsize=8)
    fig.suptitle(title + "  (filled = CI excludes 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def opportunity_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """Variant x market grid of (mean delta Sharpe, significance) + the paper's
    mechanical recommendation, aggregated over windows (and regime-label variants
    are kept separate)."""
    t = table[table["kind"] != "control"]
    out = []
    for v, g in t.groupby("variant", sort=False):
        cells, row = {}, {"variant": v, "kind": g["kind"].iloc[0]}
        for m, h in g.groupby("market", sort=False):
            cells[m] = {"sig_gain": bool((h["significant"] & (h["delta_sharpe"] > 0)).any()),
                        "sig_loss": bool((h["significant"] & (h["delta_sharpe"] < 0)).any()),
                        "mean_delta": float(h["delta_sharpe"].mean())}
            tag = "+" if cells[m]["sig_gain"] else ("-" if cells[m]["sig_loss"] else "")
            row[m] = f"{cells[m]['mean_delta']:+.2f}{'*' + tag if tag else ''}"
        row["recommendation"] = recommendation(cells)
        out.append(row)
    return pd.DataFrame(out)


def matrix_plot(table: pd.DataFrame, path: str | Path, title: str = "ML Opportunity Matrix") -> Path:
    import matplotlib.pyplot as plt

    t = table[table["kind"] != "control"]
    variants = list(dict.fromkeys(t["variant"]))
    markets = list(dict.fromkeys(t["market"]))
    rec = opportunity_matrix(table).set_index("variant")["recommendation"]
    fig, ax = plt.subplots(figsize=(3.0 + 1.6 * len(markets) + 4.2, 0.42 * len(variants) + 1.2))
    for i, v in enumerate(variants):
        yi = len(variants) - 1 - i
        for j, m in enumerate(markets):
            h = t[(t["variant"] == v) & (t["market"] == m)]
            if h.empty:
                color, txt = "white", "n/a"
            else:
                d = h["delta_sharpe"].mean()
                sl, sg = (h["significant"] & (h["delta_sharpe"] < 0)).any(), (h["significant"] & (h["delta_sharpe"] > 0)).any()
                color = LOSS if sl else GAIN if sg else NS_POS if d > 0.2 else NS_NEG if d < -0.2 else NEUTRAL
                txt = f"{d:+.2f}" + (" *" if sl or sg else "")
            ax.add_patch(plt.Rectangle((j, yi), 1, 1, color=color, ec="white"))
            ax.text(j + 0.5, yi + 0.5, txt, ha="center", va="center", fontsize=8,
                    color="white" if color in (LOSS, GAIN) else "black")
        kind = t.loc[t["variant"] == v, "kind"].iloc[0]
        ax.text(-0.1, yi + 0.5, v, ha="right", va="center", fontsize=8, color=KIND_COLOR.get(kind, "black"))
        ax.text(len(markets) + 0.15, yi + 0.5, rec[v], ha="left", va="center", fontsize=7)
    for j, m in enumerate(markets):
        ax.text(j + 0.5, len(variants) + 0.15, m, ha="center", va="bottom", fontsize=9)
    ax.set_xlim(-0.05, len(markets) + 4.5)
    ax.set_ylim(0, len(variants) + 0.8)
    ax.axis("off")
    ax.set_title(title + "  (cell = mean delta Sharpe over windows; * = CI excludes 0; "
                 "label colour = independent / narrowing / replacing / combination)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(path)
