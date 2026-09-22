"""Every figure in the paper, in one house style. Called by ``mbca.paper_assets.build``.

Figures are drawn at their exact print width -- ``COL`` = IEEEtran \\columnwidth, ``FULL`` = \\textwidth -- and saved
at 600 dpi without tight-bbox cropping, so no figure is ever rescaled: labels are 8 pt and ticks 7 pt beside the
8 pt IEEE captions (tests/test_paper.py checks the pixel widths against the manuscript).

Palette (validated with the dataviz skill's validate_palette.js, light mode, white surface):
polarity blue #2a78d6 (gain) vs red #e34948 (loss); categorical slots 1-3 blue/orange/aqua pass all-pairs CVD and
are the only hues ever overlaid in one plot; significance is always filled (significant) vs open (n.s.) markers,
so colour never carries meaning alone.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from .paper_assets import AN, COMP_LABEL, FIG, FRZ, MKT_LABEL, S1, STRAT_LABEL

COL, FULL, DPI = 3.487, 7.14, 600  # inches: IEEEtran journal column width = 21 pc, text width = 43 pc
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GAIN, LOSS, MID = "#2a78d6", "#e34948", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MARKET_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COMP_COLOR = {"meta_label": BLUE, "exit": ORANGE, "best_combined": AQUA}  # every other component: MUTED
TINT = {"independent": "#ecebe7", "narrowing": "#dbe8f9", "replacing": "#fbe1d5", "rule": "#ffffff",
        "avoid": "#f7d4d3", "conditional": "#fbe7bf", "adopt": "#dbe8f9", "neutral": "#ecebe7"}
MAIN = list(COMP_LABEL)
STRATS = list(STRAT_LABEL)
STRAT_SHORT = {"jma_trend": "JMA", "sma_cross": "Dual SMA", "donchian": "Donchian", "ts_momentum": "TS momentum",
               "rsi2_reversion": "RSI(2)"}
S1_MARKETS = {"forex_4pair": "Forex", "commodities": "Commodities", "indian_equities": "Indian equities"}

RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlepad": 3, "axes.labelpad": 2, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "lines.linewidth": 0.9,
    "lines.markersize": 3.5, "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2,
    "legend.columnspacing": 1.0, "legend.handletextpad": 0.4, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03, "figure.constrained_layout.hspace": 0.04,
    "figure.constrained_layout.wspace": 0.04,
}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("dsr", ["#9c2323", LOSS, MID, GAIN, "#104281"])


def _fig(w, h, **kw):
    return _plt().subplots(figsize=(w, h), layout="constrained", **kw)


def _save(fig, name):
    fig.savefig(FIG / name, dpi=DPI, metadata={"Software": None})
    _plt().close(fig)


def _zero(ax, axis="x"):
    (ax.axvline if axis == "x" else ax.axhline)(0, color=BASE, lw=0.6, zorder=0)


def _grid(ax, axis="x"):
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


def _sig_colour(delta, significant):
    return (GAIN if delta > 0 else LOSS) if significant else MUTED


def _dot(ax, x, y, delta, significant, marker="o", size=3.6, lo=None, hi=None):
    c = _sig_colour(delta, significant)
    if lo is not None and np.isfinite(lo):
        ax.plot([lo, hi], [y, y], color=c, lw=0.9, solid_capstyle="butt", zorder=2)
    ax.plot([x], [y], marker=marker, ms=size, mec=c, mfc=c if significant else "white", mew=0.8, ls="", zorder=3)


def _sig_legend(ax_or_fig, loc, extra=(), ncol=4, **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=3.6, mec=GAIN, mfc=GAIN, label="significant gain"),
         Line2D([], [], marker="o", ls="", ms=3.6, mec=LOSS, mfc=LOSS, label="significant loss"),
         Line2D([], [], marker="o", ls="", ms=3.6, mec=MUTED, mfc="white", label="not significant"), *extra]
    return ax_or_fig.legend(handles=h, loc=loc, ncol=ncol, **kw)


def _heat(ax, mat, vmax, fmt="{:+.2f}", stars=None, size=6.5):
    """Diverging heatmap with in-cell values; dark cells get white text (contrast on the ramp's ends)."""
    im = ax.imshow(np.asarray(mat, float), cmap=_cmap(), vmin=-vmax, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j] if hasattr(mat, "iat") else mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=size, color=MUTED)
                continue
            s = fmt.format(v)
            if float(s) == 0:
                s = s.lstrip("+-")
            s += "*" if stars is not None and bool(stars[i][j]) else ""
            ax.text(j, i, s.replace("-", "−"), ha="center", va="center", fontsize=size,
                    color="white" if abs(v) > 0.62 * vmax else INK)
    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def _repel(ys, gap, iters=200):
    """Spread sorted label positions so neighbours are >= gap apart, moving both ways (keeps labels near data)."""
    y = list(ys)
    for _ in range(iters):
        moved = False
        for k in range(len(y) - 1):
            d = y[k + 1] - y[k]
            if d < gap - 1e-12:
                y[k] -= (gap - d) / 2
                y[k + 1] += (gap - d) / 2
                moved = True
        if not moved:
            break
    return y


def _vmax(values, step=0.5):
    v = np.nanmax(np.abs(np.asarray(values, float)))
    return max(step, np.ceil(v / step) * step)


def _box(ax, x, y, w, h, text, fc, size=7.0, bold=False, ec=INK2, lw=0.6, wrap=None, italic=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=1))
    if wrap:
        text = "\n".join(textwrap.fill(t, wrap) for t in text.split("\n"))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal",
            style="italic" if italic else "normal", linespacing=1.15)


def _arrow(ax, p, q, text=None, dx=0.0, dy=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=INK2, shrinkA=0, shrinkB=0))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, fontsize=7, style="italic", color=INK2,
                ha="center", va="center")


def _canvas(w, h):
    plt = _plt()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _stest(T, family):
    return T[T.family == family].copy()


# =========================================================================== method figures
def fig_pipeline():
    """Fig. 1: the MBCA experiment -- five insertion points of one rule pipeline, each swapped for a matched ML model."""
    W, H = FULL, 2.12
    fig, ax = _canvas(W, H)
    steps = [("Regime filter", "ADX ≥ 25 gate\nor no gate", "ML regime classifier\ngates admissible bars",
              "independent"),
             ("Entry", "rule signal (JMA, SMA,\nDonchian, TSMOM, RSI(2))", "long/short classifiers\nreplace the signal",
              "replacing"),
             ("Meta-label", "take every\nsignal", "take trade iff\nP(win) ≥ 0.5", "narrowing"),
             ("Position sizing", "1× every\ntrade", "size = 2·P(win)\nin [0, 2]", "narrowing"),
             ("Exit", "ATR stop/target\nor trailing stop", "continuation classifier\n+ 3 ATR safety stop",
              "replacing")]
    n, bw, gap = len(steps), 1.18, 0.23
    x0 = (W - (n * bw + (n - 1) * gap)) / 2 + bw / 2
    ax.text(W / 2, H - 0.12, "Rule-based control: one pipeline, validated once", ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(W / 2, 0.8, "MBCA: swap exactly one insertion point for its matched-budget ML counterpart",
            ha="center", va="center", fontsize=7, style="italic", color=INK2)
    for i, (name, rule, ml, kind) in enumerate(steps):
        x = x0 + i * (bw + gap)
        _box(ax, x, 1.66, bw, 0.52, f"{name}\n", "#ffffff", size=7.5, bold=True, lw=0.8, ec=INK)
        ax.text(x, 1.57, rule, ha="center", va="center", fontsize=6.5, color=INK2, linespacing=1.1)
        _box(ax, x, 1.1, bw, 0.4, ml, TINT[kind], size=6.5)
        _arrow(ax, (x, 1.4), (x, 1.3))
        if i < n - 1:
            _arrow(ax, (x + bw / 2, 1.66), (x + bw / 2 + gap, 1.66))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=TINT[k], ec=INK2, lw=0.6, label=k) for k in ("independent", "narrowing", "replacing")],
              loc="center", bbox_to_anchor=(0.5, 0.56 / H), ncol=3, fontsize=7, handlelength=1.2, handleheight=0.8,
              title="Taxonomy (how much the component can change the trade set)", title_fontsize=7)
    invariants = ("Held fixed across every swap:   same data   ·   same 8 backward-only features   ·   "
                  "same LightGBM model and hyperparameters\nfit once before the freeze, never refit   ·   "
                  "same walk-forward protocol, execution model and statistical test")
    _box(ax, W / 2, 0.19, W - 0.5, 0.32, invariants, "#f6f6f4", size=6.8, ec=BASE)
    _save(fig, "m_pipeline.png")


def fig_timeline(cfg):
    """Fig. 2: train / out-of-sample windows of both studies, with the pre-registered crisis windows."""
    from .data import DATA_DIR, MARKETS

    plt = _plt()
    rows = []
    for key, lab in (("forex", "S1 Forex"), ("commodities", "S1 Commodities"), ("indian_equities", "S1 Indian eq.")):
        dates = [pd.read_csv(f, usecols=["Date"], parse_dates=["Date"]).Date for f in sorted((DATA_DIR / key).glob("*.csv*"))]
        start, end = min(d.min() for d in dates), max(d.max() for d in dates)
        split, fwd = pd.Timestamp(MARKETS[key]["split"]), MARKETS[key]["forward"]
        segs = [("train", start, split), ("test", split, pd.Timestamp(fwd) if fwd else end)]
        if fwd:
            segs.append(("forward", pd.Timestamp(fwd), end))
        rows.append((lab, segs))
    man = pd.read_csv(ROOT_CLEAN_MANIFEST())
    last = pd.Timestamp(man[man.status == "ok"]["last"].max())
    first = pd.Timestamp(cfg["study"]["data_start"])
    for fz in cfg["study"]["freezes"]:
        split, end = pd.Timestamp(fz["split"]), min(pd.Timestamp(fz["test_end"]), last)
        rows.append((f"S2 {fz['name']}", [("train", first, split), ("test", split, end)]))
    colour = {"train": GRID, "test": BLUE, "forward": ORANGE}
    fig, ax = _fig(COL, 1.95)
    for c in cfg["regimes"]["crises"]:
        ax.axvspan(pd.Timestamp(c["start"]), pd.Timestamp(c["end"]), color="#f7d4d3", lw=0, zorder=0)
    for i, (lab, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        for kind, a, b in segs:
            ax.barh(y, b - a, left=a, height=0.56, color=colour[kind], lw=0, zorder=2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    short = {"GFC": "GFC", "Euro debt crisis": "Euro", "2015-16 selloff": "2015–16", "COVID crash": "COVID",
             "2022 bear market": "2022"}
    for k, c in enumerate(cfg["regimes"]["crises"]):
        mid = pd.Timestamp(c["start"]) + (pd.Timestamp(c["end"]) - pd.Timestamp(c["start"])) / 2
        ax.text(mid, len(rows) - 0.35 + 0.42 * (k % 2), short.get(c["name"], c["name"]), ha="center", va="bottom",
                fontsize=6, color="#9c2323")
    ax.set_ylim(-0.5, len(rows) + 0.55)
    ax.set_xlim(pd.Timestamp("1999-06-01"), last + pd.Timedelta(days=150))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=GRID, label="training (pre-freeze)"), Patch(fc=BLUE, label="out-of-sample test"),
                       Patch(fc=ORANGE, label="forward (post-design)"), Patch(fc="#f7d4d3", label="crisis window")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, handlelength=1.0, handleheight=0.7)
    _save(fig, "m_timeline.png")


def ROOT_CLEAN_MANIFEST():
    from .datasets import CLEAN
    return CLEAN / "MANIFEST.csv"


def fig_universe(cfg):
    """Fig. 3: number of Study 2 instruments with data in each year, by market."""
    man = pd.read_csv(ROOT_CLEAN_MANIFEST())
    ok = man[man.status == "ok"].assign(first=lambda d: pd.to_datetime(d["first"]), last=lambda d: pd.to_datetime(d["last"]))
    years = np.arange(ok["first"].dt.year.min(), ok["last"].dt.year.max() + 1)
    markets = [m for m in cfg["universe"] if m in set(ok.market)]
    counts = np.array([[((g["first"].dt.year <= y) & (g["last"].dt.year >= y)).sum() for y in years]
                       for m in markets for g in [ok[ok.market == m]]])
    fig, ax = _fig(COL, 2.05)
    base = np.zeros(len(years))
    for k, (m, c) in enumerate(zip(markets, counts)):
        ax.fill_between(years, base, base + c, step="mid", color=MARKET_SLOTS[k], lw=0, label=MKT_LABEL[m])
        ax.step(years, base + c, where="mid", color="white", lw=0.5)
        base = base + c
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, base.max() * 1.04)
    ax.set_ylabel("instruments with data")
    _grid(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, handlelength=1.0, handleheight=0.7)
    _save(fig, "s2_universe.png")


# =========================================================================== Study 1
S1_ORDER = ["Rule ADX gate", "ML regime (linearity)", "ML regime (excursion)", "ML regime (move size)",
            "ML meta-labeling", "ML position sizing", "ML entry", "ML exit", "Entry + exit", "Entry + sizing",
            "Exit + sizing", "Meta-label + sizing"]
S1_KIND = ["rule", "independent", "independent", "independent", "narrowing", "narrowing", "replacing", "replacing",
           "combination", "combination", "combination", "combination"]


def _s1_table() -> pd.DataFrame:
    p = pd.read_csv(S1 / "paper" / "component_attribution_table.csv")
    name = {"Rule-based ADX gate": "Rule ADX gate", "ML regime (linearity label)": "ML regime (linearity)",
            "ML regime (excursion label)": "ML regime (excursion)", "ML regime (move-size label)": "ML regime (move size)",
            "Entry+Exit": "Entry + exit", "Entry+Sizing": "Entry + sizing", "Exit+Sizing": "Exit + sizing",
            "Best Combined (Meta-Label + Sizing)": "Meta-label + sizing"}
    comp = {"ML Meta-Labeling": "ML meta-labeling", "ML Position Sizing": "ML position sizing", "ML Entry": "ML entry",
            "ML Exit": "ML exit"}
    p["row"] = [name.get(v, comp.get(c)) for c, v in zip(p.component, p.variant)]
    assert p.row.notna().all() and set(p.row) == set(S1_ORDER), set(p.row) ^ set(S1_ORDER)
    p["mkt"] = p.market.map(S1_MARKETS)
    return p


def fig_s1_matrix():
    """Fig. 4: Study 1 opportunity matrix (mean delta Sharpe over windows) with the mechanical recommendation."""
    from .stats import recommendation

    p = _s1_table()
    mk = list(S1_MARKETS.values())
    mat = p.pivot_table(index="row", columns="mkt", values="delta_sharpe", aggfunc="mean").reindex(index=S1_ORDER, columns=mk)
    sg = p.assign(g=p.significant & (p.delta_sharpe > 0), l=p.significant & (p.delta_sharpe < 0))
    anyg = sg.pivot_table(index="row", columns="mkt", values="g", aggfunc="max").reindex(index=S1_ORDER, columns=mk)
    anyl = sg.pivot_table(index="row", columns="mkt", values="l", aggfunc="max").reindex(index=S1_ORDER, columns=mk)
    rec = [recommendation({m: {"sig_gain": bool(anyg.at[r, m]), "sig_loss": bool(anyl.at[r, m]),
                               "mean_delta": float(mat.at[r, m])} for m in mk if np.isfinite(mat.at[r, m])})
           for r in S1_ORDER]
    plt = _plt()
    H = 3.2
    fig = plt.figure(figsize=(FULL, H))
    top, bot = 0.56, 0.08
    ax = fig.add_axes((1.95 / FULL, bot / H, 2.2 / FULL, (H - top - bot) / H))
    stars = (anyg | anyl).fillna(False).to_numpy()
    im = _heat(ax, mat, _vmax(mat.to_numpy()), stars=stars, size=7)
    ax.set_xticks(range(len(mk)), mk)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(S1_ORDER)), S1_ORDER)
    for y, k in enumerate(S1_KIND):
        ax.annotate(k, (0, y), xycoords=("axes fraction", "data"), xytext=(-84, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.5, color=INK2, style="italic")
    for y, r in enumerate(rec):
        verdict, _, rest = r.partition(" (")
        txt = verdict if not rest else f"{verdict}: {rest.rstrip(')')}"
        ax.annotate(textwrap.fill(txt, 62), (1, y), xycoords=("axes fraction", "data"), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6.5, linespacing=1.05)
    ax.annotate("Recommendation (mechanical rule)", (1, -0.5), xycoords=("axes fraction", "data"), xytext=(8, 4),
                textcoords="offset points", ha="left", va="bottom", fontsize=7, fontweight="bold")
    ax.annotate("Type", (0, -0.5), xycoords=("axes fraction", "data"), xytext=(-84, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=7, fontweight="bold")
    cax = fig.add_axes((1.95 / FULL, (H - 0.24) / H, 2.2 / FULL, 0.05 / H))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cb.outline.set_visible(False)
    cax.set_title("mean ΔSharpe over windows (* = CI excludes 0)", fontsize=6.5, pad=2, color=INK2)
    _save(fig, "s1_matrix.png")


def fig_s1_forest():
    """Fig. 5: every Study 1 configuration, holdout (circle) and forward (triangle), per market."""
    from .stats import PERIODS

    p = _s1_table()
    k = np.sqrt(PERIODS)
    fig, axes = _fig(FULL, 3.35, ncols=3, sharey=True)
    ys = {r: len(S1_ORDER) - 1 - i for i, r in enumerate(S1_ORDER)}
    for ax, m in zip(axes, S1_MARKETS.values()):
        h = p[p.mkt == m]
        for r in h.itertuples():
            off = 0.17 if r.window == "holdout" else -0.17
            _dot(ax, r.delta_sharpe, ys[r.row] + off, r.delta_sharpe, r.significant,
                 marker="o" if r.window == "holdout" else "^", lo=r.bootstrap_ci_lo * k, hi=r.bootstrap_ci_hi * k)
        _zero(ax)
        _grid(ax)
        for y in range(len(S1_ORDER) - 1):
            ax.axhline(y + 0.5, color=GRID, lw=0.4, zorder=0)
        ax.set_title(m)
        ax.set_xlabel("ΔSharpe (annualized), 95% CI")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(ys.values()), list(ys.keys()))
    axes[0].set_ylim(-0.6, len(S1_ORDER) - 0.4)
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], marker="o", ls="", ms=3.6, mec=INK2, mfc="white", label="holdout"),
             Line2D([], [], marker="^", ls="", ms=3.6, mec=INK2, mfc="white", label="forward")]
    _sig_legend(fig, "outside upper center", extra=extra, ncol=5)
    _save(fig, "s1_forest.png")


def fig_s1_replication():
    """Fig. 6: original Study 1 delta Sharpe vs the independent re-implementation (replaces a 60-row table)."""
    c = pd.read_csv(S1 / "replication" / "paper_vs_replication.csv")
    fig, ax = _fig(COL, 2.75)
    lim = _vmax(c[["delta_sharpe_paper", "delta_sharpe_rep"]].to_numpy(), 2.0)
    ax.fill_between([0, lim], 0, lim, color=MID, lw=0, zorder=0)
    ax.fill_between([-lim, 0], -lim, 0, color=MID, lw=0, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color=BASE, lw=0.6, zorder=1)
    _zero(ax, "x")
    _zero(ax, "y")
    for (mk, lab), col, mkr in zip({"forex": "Forex", "commodities": "Commodities",
                                    "indian_equities": "Indian equities"}.items(), (BLUE, ORANGE, AQUA), "os^"):
        h = c[c.market == mk]
        both = h.significant_paper & h.significant_rep
        ax.plot(h.delta_sharpe_paper[~both], h.delta_sharpe_rep[~both], mkr, ms=3.4, mec=col, mfc="white", mew=0.8,
                label=lab)
        ax.plot(h.delta_sharpe_paper[both], h.delta_sharpe_rep[both], mkr, ms=3.4, mec=col, mfc=col, mew=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("original study ΔSharpe")
    ax.set_ylabel("re-implementation ΔSharpe")
    ax.text(0.97, 0.05, f"same sign: {int(c.sign_agrees.sum())}/{len(c)}\nfilled = significant in both",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.text(0.03, 0.95, "shaded = signs agree", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2)
    _save(fig, "s1_replication.png")


# =========================================================================== Study 2
def fig_pvalues(mk, alpha):
    """Fig. 7: the 528 market-level p-values against the uniform global null."""
    fig, ax = _fig(COL, 1.9)
    bins = np.linspace(0, 1, 21)
    cnt, _ = np.histogram(mk.p_value, bins)
    ax.bar(bins[:-1], cnt, width=0.05, align="edge", color=[BLUE] + [BASE] * (len(cnt) - 1), ec="white", lw=0.6)
    ax.axhline(len(mk) / 20, color=INK, lw=0.7)
    ax.text(0.99, len(mk) / 20, f"expected under the null ({len(mk) / 20:.1f} per bin)", ha="right", va="bottom",
            fontsize=6.5, color=INK2)
    g, l = int((mk.significant & (mk.delta_sharpe > 0)).sum()), int((mk.significant & (mk.delta_sharpe < 0)).sum())
    ax.text(0.065, cnt[0], f"\u2190 {cnt[0]} tests with raw p < {alpha:g};\n     after BH-FDR (q < {alpha:g}): "
            f"{g} gains, {l} losses", fontsize=6.5, va="top", ha="left", color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("p-value (paired block bootstrap)")
    ax.set_ylabel("tests")
    _grid(ax, "y")
    _save(fig, "s2_pvalues.png")


def fig_distribution(mk):
    """Fig. 8: every market-level delta Sharpe, by component, with the median."""
    rng = np.random.default_rng(0)
    fig, ax = _fig(COL, 2.55)
    lim = 2.0
    for i, c in enumerate(MAIN):
        t = mk[mk.component == c]
        y = len(MAIN) - 1 - i
        jit = rng.uniform(-0.28, 0.28, len(t))
        x = t.delta_sharpe.clip(-lim, lim)
        ns = ~t.significant
        ax.plot(x[ns], y + jit[ns], "o", ms=2.2, mec=MUTED, mfc="white", mew=0.5, alpha=0.9)
        for s, col in ((t.significant & (t.delta_sharpe > 0), GAIN), (t.significant & (t.delta_sharpe < 0), LOSS)):
            ax.plot(x[s], y + jit[s], "o", ms=3.0, mec=col, mfc=col, mew=0.5, zorder=3)
        med = t.delta_sharpe.median()
        ax.plot([med, med], [y - 0.36, y + 0.36], color=INK, lw=1.3, zorder=4, solid_capstyle="butt")
    _zero(ax)
    _grid(ax)
    ax.set_yticks(range(len(MAIN)), [COMP_LABEL[c] for c in MAIN][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-lim - 0.1, lim + 0.1)
    ax.set_xlabel(f"market-level ΔSharpe (clipped at ±{lim:g})")
    from matplotlib.lines import Line2D
    _sig_legend(fig, "outside upper center", ncol=4, columnspacing=0.7, handletextpad=0.2,
                extra=[Line2D([], [], color=INK, lw=1.3, label="median")])
    _save(fig, "s2_distribution.png")


def fig_heatmap(mk):
    """Fig. 9: mean delta Sharpe over the two freezes, strategy x component x market."""
    markets = [m for m in MKT_LABEL if m in set(mk.market)]
    mat = mk.pivot_table(index=["strategy", "component"], columns="market", values="delta_sharpe", aggfunc="mean")
    sig = mk.pivot_table(index=["strategy", "component"], columns="market", values="significant", aggfunc="max")
    idx = pd.MultiIndex.from_product([STRATS, MAIN])
    mat, sig = mat.reindex(index=idx, columns=markets), sig.reindex(index=idx, columns=markets)
    plt = _plt()
    H = 4.6
    fig = plt.figure(figsize=(FULL, H))
    left, right, top, bot = 2.05, 0.75, 0.47, 0.06
    ax = fig.add_axes((left / FULL, bot / H, (FULL - left - right) / FULL, (H - top - bot) / H))
    im = _heat(ax, mat, 1.5, fmt="{:+.2f}", stars=sig.astype("boolean").fillna(False).to_numpy(bool), size=6.2)
    ax.set_xticks(range(len(markets)), [MKT_LABEL[m].replace(" ", "\n") for m in markets])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(idx)), [COMP_LABEL[c] for _, c in idx])
    for k, s in enumerate(STRATS):
        y0, y1 = k * len(MAIN) - 0.5, (k + 1) * len(MAIN) - 0.5
        if k:
            ax.axhline(y0, color=INK, lw=0.9)
        ax.annotate(textwrap.fill(STRAT_LABEL[s], 12), (0, (y0 + y1) / 2), xycoords=("axes fraction", "data"), xytext=(-86, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7.5, fontweight="bold",
                    rotation=0)
    cax = fig.add_axes(((FULL - right + 0.15) / FULL, bot / H + 0.3, 0.08 / FULL, 0.45))
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=1.5)
    cb.set_label("mean ΔSharpe (two freezes);  * = q < 0.05 in a freeze", fontsize=6.5, color=INK2)
    _save(fig, "s2_heatmap.png")


def fig_forest_global(gl):
    """Fig. 10: all-market portfolio delta Sharpe with paired block-bootstrap CIs (replaces a 70-row table)."""
    fig, axes = _fig(FULL, 4.3, ncols=2, sharey=True)
    rows = [(s, c) for s in STRATS for c in MAIN]
    ypos = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
    for ax, fz in zip(axes, FRZ):
        h = gl[gl.freeze == fz]
        for r in h.itertuples():
            _dot(ax, r.delta_sharpe, ypos[(r.strategy, r.component)], r.delta_sharpe, r.significant, lo=r.ci_lo,
                 hi=r.ci_hi)
        _zero(ax)
        _grid(ax)
        for k in range(1, len(STRATS)):
            ax.axhline(len(rows) - k * len(MAIN) - 0.5, color=BASE, lw=0.6)
        ax.set_title(f"Freeze {fz}")
        ax.set_xlabel("ΔSharpe (annualized), 95% paired block-bootstrap CI")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[r] for r in rows], [COMP_LABEL[c] for _, c in rows])
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    for k, s in enumerate(STRATS):
        yc = len(rows) - 1 - (k * len(MAIN) + (len(MAIN) - 1) / 2)
        axes[0].annotate(STRAT_SHORT[s], (0, yc), xycoords=("axes fraction", "data"), xytext=(-80, 0),
                         textcoords="offset points", ha="right", va="center", fontsize=7.5, fontweight="bold")
    _sig_legend(fig, "outside upper center", ncol=3)
    _save(fig, "s2_forest_global.png")


def fig_equity():
    """Fig. 11: all-market equity curves of the control and three ML variants in each out-of-sample window."""
    plt = _plt()
    daily = pd.read_parquet(AN / "global_daily.parquet")
    fig, axes = _fig(FULL, 5.4, nrows=len(STRATS), ncols=2)
    series = (("baseline", INK, "rule-based control", 1.0), ("meta_label", BLUE, COMP_LABEL["meta_label"], 0.8),
              ("best_combined", AQUA, COMP_LABEL["best_combined"], 0.8), ("exit", ORANGE, COMP_LABEL["exit"], 0.8))
    for i, s in enumerate(STRATS):
        for j, fz in enumerate(FRZ):
            ax = axes[i][j]
            for v, col, lab, lw in series:
                h = daily[(daily.strategy == s) & (daily.variant == v) & (daily.freeze == fz)].sort_values("date")
                if len(h):
                    ax.plot(pd.to_datetime(h.date), (1 + h.ret).cumprod(), color=col, lw=lw, label=lab,
                            zorder=3 if v == "baseline" else 2)
            ax.axhline(1, color=BASE, lw=0.6, zorder=0)
            _grid(ax, "y")
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
            ax.set_title(f"{STRAT_LABEL[s]}  ·  {fz}", loc="left")
            if j == 0:
                ax.set_ylabel("growth of 1")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4)
    _save(fig, "s2_equity.png")


def fig_risk(gl):
    """Fig. 12: change in the portfolio's risk profile per component (replaces the ~80-row risk table)."""
    g = gl.assign(dSR=gl.v_sharpe - gl.b_sharpe, dCAGR=100 * (gl.v_ann_return - gl.b_ann_return),
                  dVol=100 * (gl.v_ann_vol - gl.b_ann_vol), dDD=100 * (gl.v_max_drawdown - gl.b_max_drawdown),
                  dTurn=gl.v_turnover - gl.b_turnover, dExp=100 * (gl.v_exposure - gl.b_exposure))
    panels = [("dSR", "ΔSharpe"), ("dCAGR", "ΔCAGR (pp)"), ("dVol", "Δvolatility (pp)"),
              ("dDD", "Δmax drawdown (pp)\n(+ = shallower)"), ("dTurn", "Δturnover (×/yr)"),
              ("dExp", "Δexposure (pp)")]
    fig, axes = _fig(FULL, 2.3, ncols=len(panels), sharey=True)
    for ax, (col, lab) in zip(axes, panels):
        for i, c in enumerate(MAIN):
            y = len(MAIN) - 1 - i
            v = g[g.component == c][col]
            ax.plot(v, np.full(len(v), y), "o", ms=2.6, mec=MUTED, mfc="white", mew=0.6, alpha=0.95)
            ax.plot([v.median()], [y], "D", ms=3.4, mec=INK, mfc=INK)
        _zero(ax)
        _grid(ax)
        ax.set_xlabel(lab)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(_plt().MaxNLocator(4))
    axes[0].set_yticks(range(len(MAIN)), [COMP_LABEL[c] for c in MAIN][::-1])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=2.6, mec=MUTED, mfc="white", label="strategy × freeze"),
                        Line2D([], [], marker="D", ls="", ms=3.4, mec=INK, mfc=INK, label="median")],
               loc="outside upper center", ncol=2)
    _save(fig, "s2_risk.png")


def fig_freeze_rep(mk):
    """Fig. 13: does a strategy x market effect keep its sign at the second, independent freeze?"""
    fig, axes = _fig(FULL, 3.55, nrows=2, ncols=4)
    axes = axes.ravel()
    piv = mk.pivot_table(index=["component", "strategy", "market"], columns="freeze", values="delta_sharpe").dropna()
    lim = 2.0
    for ax, c in zip(axes, MAIN):
        h = piv.loc[c].reset_index()
        rsi = h.strategy == "rsi2_reversion"
        ax.fill_between([0, lim], 0, lim, color=MID, lw=0, zorder=0)
        ax.fill_between([-lim, 0], -lim, 0, color=MID, lw=0, zorder=0)
        _zero(ax, "x")
        _zero(ax, "y")
        ax.plot(h.F2008[~rsi].clip(-lim, lim), h.F2018[~rsi].clip(-lim, lim), "o", ms=2.8, mec=BLUE, mfc="white",
                mew=0.7)
        ax.plot(h.F2008[rsi].clip(-lim, lim), h.F2018[rsi].clip(-lim, lim), "s", ms=2.8, mec=ORANGE, mfc=ORANGE,
                mew=0.7)
        agree = (np.sign(h.F2008) == np.sign(h.F2018)).mean()
        ax.set_title(COMP_LABEL[c])
        ax.text(0.03, 0.97, f"r = {h.F2008.corr(h.F2018):.2f}\nsame sign {100 * agree:.0f}%", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.5, color=INK2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
    for ax in axes[4:7]:
        ax.set_xlabel("ΔSharpe, F2008")
    axes[0].set_ylabel("ΔSharpe, F2018")
    axes[4].set_ylabel("ΔSharpe, F2018")
    ax = axes[7]
    ax.axis("off")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=2.8, mec=BLUE, mfc="white", label="trend-following strategy"),
                       Line2D([], [], marker="s", ls="", ms=2.8, mec=ORANGE, mfc=ORANGE, label="RSI(2) reversion")],
              loc="center", title=f"one point = strategy × market\nshaded = same sign in both freezes\n"
                                  f"(values clipped at ±{lim:g})", title_fontsize=6.5)
    _save(fig, "s2_freeze_rep.png")


def fig_robust(T, mk):
    """Fig. 14: (a) cross-market transfer, (b) budget ablations, (c) execution sensitivity."""
    fig, axes = _fig(FULL, 2.55, ncols=3, width_ratios=[1, 1.15, 1.25])
    ax = axes[0]
    tr = _stest(T, "market_transfer")
    key = ["strategy", "market", "freeze", "component"]
    m = tr.merge(mk[key + ["delta_sharpe"]], on=key, suffixes=("_tr", "_in"))
    lim = 1.5
    ax.plot([-lim, lim], [-lim, lim], color=BASE, lw=0.6)
    _zero(ax, "x")
    _zero(ax, "y")
    for c, mkr in (("meta_label", "o"), ("best_combined", "^")):
        h = m[m.component == c]
        ax.plot(h.delta_sharpe_in.clip(-lim, lim), h.delta_sharpe_tr.clip(-lim, lim), mkr, ms=2.8, mec=COMP_COLOR[c],
                mfc="white", mew=0.7, label=f"{COMP_LABEL[c]} (r = {h.delta_sharpe_in.corr(h.delta_sharpe_tr):.2f})")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("ΔSharpe, model fit in-market")
    ax.set_ylabel("ΔSharpe, model fit on other markets")
    ax.set_title("(a) Cross-market transfer", loc="left")
    ax.legend(loc="lower right", fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)

    ax = axes[1]
    ab = _stest(T, "market_ablation")
    rows = [("meta_label", mk[mk.component == "meta_label"], "ML meta-labeling (main)"),
            ("meta_label", ab[ab.component == "meta_label_thr045"], "threshold 0.45"),
            ("meta_label", ab[ab.component == "meta_label_thr055"], "threshold 0.55"),
            ("meta_label", ab[ab.component == "meta_label_small_model"], "small model"),
            ("exit", mk[mk.component == "exit"], "ML exit (main)"),
            ("exit", ab[ab.component == "exit_small_model"], "small model")]
    for i, (c, h, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        q1, med, q3 = h.delta_sharpe.quantile([0.25, 0.5, 0.75])
        ax.plot([q1, q3], [y, y], color=COMP_COLOR[c], lw=2.2, solid_capstyle="butt", alpha=0.45)
        ax.plot([med], [y], "o", ms=3.6, mec=COMP_COLOR[c], mfc=COMP_COLOR[c])
        ax.annotate(f"{med:+.2f}".replace("-", "−"), (med, y), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=6.3, color=INK2)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    _zero(ax)
    _grid(ax)
    ax.set_xlabel("market-level ΔSharpe: median, IQR")
    ax.set_title("(b) Budget ablations", loc="left")

    ax = axes[2]
    ex = _stest(T, "global_execution")
    order = [("baseline@optimistic_stop_fill", "control: stop-level fill"), ("baseline@one_day_lag", "control: 1-day lag"),
             ("best_combined@optimistic_stop_fill", "meta+sizing: stop fill"),
             ("best_combined@one_day_lag", "meta+sizing: 1-day lag"), ("exit@optimistic_stop_fill", "ML exit: stop fill"),
             ("exit@one_day_lag", "ML exit: 1-day lag")]
    for i, (v, lab) in enumerate(order):
        y = len(order) - 1 - i
        h = ex[ex.variant == v]
        for r in h.itertuples():
            _dot(ax, r.delta_sharpe, y, r.delta_sharpe, r.significant, size=3.0)
        ax.plot([h.delta_sharpe.median()] * 2, [y - 0.3, y + 0.3], color=INK, lw=1.2)
    ax.set_yticks(range(len(order)), [o[1] for o in order][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    _zero(ax)
    _grid(ax)
    ax.set_xlabel("all-market ΔSharpe (strategy × freeze)")
    ax.set_title("(c) Execution assumptions", loc="left")
    _save(fig, "s2_robust.png")


def fig_costs():
    """Fig. 15: median all-market delta Sharpe as transaction costs are scaled."""
    cs = pd.read_csv(AN / "cost_sensitivity.csv")
    fig, ax = _fig(COL, 2.2)
    ends = []
    for c in MAIN:
        h = cs[cs.variant == c].groupby("cost_multiplier").delta_sharpe.median()
        if not len(h):
            continue
        hi = c in COMP_COLOR
        ax.plot(h.index, h.values, marker="o" if hi else None, ms=3.0, color=COMP_COLOR.get(c, BASE),
                lw=1.1 if hi else 0.8, zorder=3 if hi else 2)
        ends.append([h.values[-1], COMP_LABEL[c], COMP_COLOR.get(c, MUTED)])
    ends.sort(key=lambda e: e[0])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    pos = _repel([e[0] for e in ends], 0.068 * (ymax - ymin))
    for (y, lab, col), yl in zip(ends, pos):
        ax.plot([4.04, 4.3], [y, yl], color=BASE, lw=0.5, clip_on=False)
        ax.text(4.36, yl, lab, va="center", fontsize=6.3, color=INK if col != MUTED else INK2)
    _zero(ax, "y")
    _grid(ax, "y")
    ax.set_xticks([0, 1, 2, 4], ["×0", "×1", "×2", "×4"])
    ax.set_xlim(-0.15, 6.3)
    ax.spines["bottom"].set_bounds(0, 4)
    ax.set_xlabel("transaction-cost multiplier")
    ax.set_ylabel("median all-market ΔSharpe")
    _save(fig, "s2_costs.png")


def fig_regimes_crises():
    """Fig. 16: (a) cross-regime delta Sharpe, (b) return and (c) drawdown change inside the crisis windows."""
    rg = pd.read_csv(AN / "regimes.csv")
    cr = pd.read_csv(AN / "crises.csv")
    regs = [("bull", "Bull"), ("bear", "Bear"), ("low_vol", "Low vol"), ("high_vol", "High vol")]
    a = rg.pivot_table(index="variant", columns="regime", values="delta_sharpe", aggfunc="mean").reindex(
        index=MAIN, columns=[r for r, _ in regs])
    cr = cr.assign(dret=100 * (cr.v_return - cr.b_return), ddd=100 * (cr.v_max_drawdown - cr.b_max_drawdown))
    crises = list(dict.fromkeys(cr.crisis))
    b = cr.pivot_table(index="variant", columns="crisis", values="dret", aggfunc="mean").reindex(index=MAIN, columns=crises)
    d = cr.pivot_table(index="variant", columns="crisis", values="ddd", aggfunc="mean").reindex(index=MAIN, columns=crises)
    short = {"GFC": "GFC", "Euro debt crisis": "Euro debt", "2015-16 selloff": "2015–16", "COVID crash": "COVID",
             "2022 bear market": "2022 bear"}
    fig, axes = _fig(FULL, 2.45, ncols=3, width_ratios=[4, 5, 5])
    for ax, mat, cols, title, fmt, lab in (
            (axes[0], a, [l for _, l in regs], "(a) Cross-regime ΔSharpe", "{:+.2f}", "ΔSharpe"),
            (axes[1], b, [short[c] for c in crises], "(b) Crisis Δreturn (pp)", "{:+.1f}", "pp"),
            (axes[2], d, [short[c] for c in crises], "(c) Crisis Δmax drawdown (pp, + = shallower)", "{:+.1f}",
             "pp")):
        im = _heat(ax, mat, _vmax(mat.to_numpy(), 0.5 if fmt == "{:+.2f}" else 2), fmt=fmt, size=6.3)
        ax.set_xticks(range(len(cols)), cols, rotation=0)
        ax.set_title(title, loc="left")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, aspect=30)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
        ax.set_yticks(range(len(MAIN)), [COMP_LABEL[c] for c in MAIN] if ax is axes[0] else [])
    _save(fig, "s2_regimes_crises.png")


def fig_jackknife(mk):
    """Fig. 17: leave-one-instrument-out range of every significant market-level effect."""
    jk = pd.read_csv(AN / "jackknife.csv")
    j = jk[jk.scope == "market"].merge(mk[mk.significant][["strategy", "market", "freeze", "variant", "delta_sharpe"]],
                                       on=["strategy", "market", "freeze", "variant"])
    j = j.sort_values("delta_full").reset_index(drop=True)
    fig, ax = _fig(COL, 0.13 * len(j) + 0.75)
    for i, r in j.iterrows():
        _dot(ax, r.delta_full, i, r.delta_full, True, lo=r.loo_min, hi=r.loo_max, size=3.2)
    comp = {"rule_adx": "ADX gate", "meta_label": "meta-label", "best_combined": "meta+sizing", "entry": "entry",
            "exit": "exit", "position_sizing": "sizing", "regime_linearity": "regime"}
    lab = [f"{STRAT_SHORT[r.strategy]} {comp[r.variant]}, {MKT_LABEL[r.market]}, {r.freeze}" for r in j.itertuples()]
    ax.set_yticks(range(len(j)), lab, fontsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(j) - 0.3)
    _zero(ax)
    _grid(ax)
    ax.set_xlabel("ΔSharpe; bar = leave-one-out range")
    _save(fig, "s2_jackknife.png")


def fig_decay():
    """Fig. 18: mean trade return of the walk-forward-selected controls, in-sample vs out-of-sample."""
    dec = pd.read_csv(AN / "is_oos_decay.csv")
    rows = [(s, np.average(g.is_mean_trade_bps, weights=g.n_trades), np.average(g.oos_mean_trade_bps, weights=g.n_trades))
            for s, g in dec.groupby("strategy", sort=False)]
    rows = sorted(rows, key=lambda r: STRATS.index(r[0]))
    fig, ax = _fig(COL, 1.65)
    for i, (s, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([a, b], [y, y], color=BASE, lw=1.6, solid_capstyle="butt", zorder=1)
        ax.plot([a], [y], "o", ms=4, mec=MUTED, mfc="white", mew=0.9, zorder=2)
        ax.plot([b], [y], "o", ms=4, mec=BLUE, mfc=BLUE, zorder=3)
        for v, ha, dx in ((a, "left" if a > b else "right", 5 if a > b else -5), (b, "right" if a > b else "left", -5 if a > b else 5)):
            ax.annotate(f"{v:.0f}".replace("-", "−"), (v, y), xytext=(dx, 0), textcoords="offset points", ha=ha,
                        va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(range(len(rows)), [STRAT_LABEL[r[0]] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    lo, hi = min(min(r[1:]) for r in rows), max(max(r[1:]) for r in rows)
    ax.set_xlim(lo - 60, hi + 60)
    _zero(ax)
    _grid(ax)
    ax.set_xlabel("mean trade return (bps, trade-weighted across markets)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, mec=MUTED, mfc="white", label="in-sample"),
                       Line2D([], [], marker="o", ls="", ms=4, mec=BLUE, mfc=BLUE, label="out-of-sample")],
              loc="lower right", ncol=2)
    _save(fig, "s2_decay.png")


def fig_decision(T, mk, gl):
    """Fig. 19: practitioner decision guide -- the evidence of both studies condensed per insertion point."""
    def st(c):
        t = mk[mk.component == c]
        piv = t.pivot_table(index=["strategy", "market"], columns="freeze", values="delta_sharpe").dropna()
        rsi, oth = t[t.strategy == "rsi2_reversion"], t[t.strategy != "rsi2_reversion"]
        return dict(pos=100 * (t.delta_sharpe > 0).mean(), med=t.delta_sharpe.median(), n=len(t),
                    g=int((t.significant & (t.delta_sharpe > 0)).sum()), l=int((t.significant & (t.delta_sharpe < 0)).sum()),
                    agree=100 * (np.sign(piv.F2008) == np.sign(piv.F2018)).mean(),
                    medrsi=rsi.delta_sharpe.median(), medtr=oth.delta_sharpe.median(),
                    grsi=int((rsi.significant & (rsi.delta_sharpe > 0)).sum()),
                    lrsi=int((rsi.significant & (rsi.delta_sharpe < 0)).sum()))

    def f(x):
        return f"{x:+.2f}".replace("-", "−")

    tr = _stest(T, "market_transfer")
    trm = tr[tr.component == "meta_label"].delta_sharpe.median()
    R, M, S, E, X, B = (st(c) for c in ("regime_linearity", "meta_label", "position_sizing", "entry", "exit",
                                         "best_combined"))
    W, H = FULL, 3.45
    fig, ax = _canvas(W, H)
    _box(ax, W / 2 + 0.55, 3.2, 3.5, 0.34, "Is the rule-based control stable and validated out of sample,\n"
         "under close-price fills and realistic costs?", "#f6f6f4", size=7.2, bold=True)
    _box(ax, 1.05, 2.62, 1.75, 0.42, "Validate the control first:\nMBCA attributes against a stable control",
         TINT["avoid"], size=6.5)
    _box(ax, W / 2 + 0.55, 2.62, 3.1, 0.3, "Which insertion point would ML replace or modify?", "#f6f6f4", size=7.2,
         bold=True)
    _arrow(ax, (W / 2 - 1.2, 3.03), (1.6, 2.83), "no", dy=0.07)
    _arrow(ax, (W / 2 + 0.55, 3.03), (W / 2 + 0.55, 2.77), "yes", dx=0.13)
    cards = [
        ("Regime filter", "independent", "neutral", "Skip",
         f"median {f(R['med'])}; {R['g']} gains / {R['l']} losses in {R['n']} tests; sign replicates in "
         f"{R['agree']:.0f}% of cells (chance)"),
        ("Meta-labeling", "narrowing", "adopt", "Adopt cautiously",
         f"positive in {M['pos']:.0f}% of tests, median {f(M['med'])}; never a significant loss; transfers across "
         f"markets (median {f(trm)})"),
        ("Entry", "replacing", "avoid", "Avoid",
         f"most harmful component: median {f(E['med'])}, {E['l']} significant losses ({E['lrsi']} in RSI(2))"),
        ("Exit", "replacing", "conditional", "Only if the rule's exit is weak",
         f"RSI(2): median {f(X['medrsi'])}, {X['grsi']} significant gains (fixed target caps winners); "
         f"trend-followers {f(X['medtr'])}"),
        ("Position sizing", "narrowing", "neutral", "Optional",
         f"positive in {S['pos']:.0f}% of tests, median {f(S['med'])}; no significant result; sign replicates in "
         f"{S['agree']:.0f}% of cells"),
    ]
    n, cw, gap = len(cards), 1.3, 0.1
    x0 = (W - (n * cw + (n - 1) * gap)) / 2 + cw / 2
    for i, (name, kind, tone, verdict, why) in enumerate(cards):
        x = x0 + i * (cw + gap)
        _arrow(ax, (W / 2 + 0.55, 2.47), (x, 2.2))
        _box(ax, x, 2.02, cw, 0.34, f"{name}\n", TINT[kind], size=7.2, bold=True)
        ax.text(x, 1.94, kind, ha="center", va="center", fontsize=6.3, style="italic", color=INK2)
        _arrow(ax, (x, 1.85), (x, 1.74))
        _box(ax, x, 1.31, cw, 0.84, "", TINT[tone])
        ax.text(x, 1.62, verdict, ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(x, 1.5, textwrap.fill(why, 27), ha="center", va="top", fontsize=6.2, linespacing=1.12)
        _arrow(ax, (x, 0.89), (W / 2, 0.66))
    _box(ax, W / 2, 0.5, 5.4, 0.32, f"Combining the narrowing components (meta-label + sizing): positive in "
         f"{B['pos']:.0f}% of tests, median {f(B['med'])}, significant in {B['g']}/{B['n']};\n"
         f"sign replicates across the two freezes in {B['agree']:.0f}% of cells: small, safe, and not a free lunch.",
         TINT["adopt"], size=6.6)
    _box(ax, W / 2, 0.14, 6.2, 0.2, "Default: prefer narrowing over replacing components; treat any single-window "
         "gain as exploratory until it replicates at a second, independent freeze.", "#ffffff", size=6.6, italic=True,
         ec=BASE)
    _save(fig, "s2_decision.png")


# =========================================================================== entry point
def build(cfg) -> None:
    T = pd.read_csv(AN / "tests.csv")
    mk, gl = _stest(T, "market_main"), _stest(T, "global_main")
    with _plt().rc_context(RC):
        fig_pipeline()
        fig_timeline(cfg)
        fig_universe(cfg)
        fig_s1_matrix()
        fig_s1_forest()
        fig_s1_replication()
        fig_pvalues(mk, cfg["statistics"]["alpha"])
        fig_distribution(mk)
        fig_heatmap(mk)
        fig_forest_global(gl)
        fig_equity()
        fig_risk(gl)
        fig_freeze_rep(mk)
        fig_robust(T, mk)
        fig_costs()
        fig_regimes_crises()
        fig_jackknife(mk)
        fig_decay()
        fig_decision(T, mk, gl)
