"""Study 2 part of the paper-asset generator (called by mbca.paper_assets.build)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .datasets import CLEAN, RAW, VALID
from .paper_assets import (AN, COMP, COMP_LABEL, FIG, FRZ, MKT_LABEL, STRAT, STRAT_LABEL, TEX, Macros, esc, f2,
                           longtable, pct)

MAIN = list(COMP)
ML = [c for c in MAIN if c != "rule_adx"]


def _sig(r):
    if not r.significant:
        return ""
    return "$^{+}$" if r.delta_sharpe > 0 else "$^{-}$"


def study2(M: Macros) -> None:
    cfg = load_config()
    T = pd.read_csv(AN / "tests.csv")
    mk = T[T.family == "market_main"]
    gl = T[T.family == "global_main"]
    _data(M, cfg)
    _headline(M, gl)
    _components(M, mk, gl)
    _by_strategy(mk)
    _metrics_table(gl)
    _robustness(M, T, mk)
    _regimes_crises(M)
    _figures(mk, gl)
    _bias_audit(M, T, mk)


# ------------------------------------------------------------------ data
def _data(M, cfg):
    man = pd.read_csv(CLEAN / "MANIFEST.csv")
    ok = man[man.status == "ok"]
    raw = pd.read_csv(RAW / "MANIFEST.csv")
    M["STwoInstruments"] = str(len(ok))
    M["STwoPlanned"] = str(sum(len(v["tickers"]) for v in cfg["universe"].values()))
    M["STwoMarkets"] = str(ok.market.nunique())
    M["STwoRows"] = f"{int(ok.clean_rows.sum()):,}".replace(",", "{,}")
    M["STwoFirst"], M["STwoLast"] = str(ok["first"].min()), str(ok["last"].max())
    M["STwoFetched"] = raw[raw.source == "yahoo"].fetched_at_utc.min()[:10]
    M["STwoRawFiles"] = str(len(raw))
    M["STwoDropped"] = f"{int(ok.missing_or_nonpositive.sum()):,}".replace(",", "{,}")
    M["STwoRepaired"] = f"{int(ok.ohlc_repaired.sum()):,}".replace(",", "{,}")
    M["STwoBadTicks"] = str(int(ok.bad_ticks.sum()))
    v = pd.read_csv(VALID / "yahoo_vs_fred.csv")
    close = v[~v.symbol.isin(["NG=F"])]
    M["STwoFredN"] = str(len(v))
    M["STwoFredCorrFiveMin"], M["STwoFredCorrFiveMax"] = f2(close.return_corr_5d.min()), f2(close.return_corr_5d.max())
    M["STwoFredMedDiffMax"] = pct(close.median_abs_pct_diff.max())
    M["STwoNgCorrFive"] = f2(v[v.symbol == "NG=F"].return_corr_5d.iloc[0])
    sc = pd.read_csv(VALID / "browser_spotcheck.csv")
    M["STwoSpotDays"], M["STwoSpotSymbols"] = str(len(sc)), str(sc.symbol.nunique())
    M["STwoSpotMaxDiff"] = f"{sc.abs_diff.max():.2f}"
    rows = []
    for mkt, g in ok.groupby("market", sort=False):
        u = cfg["universe"][mkt]
        rows.append([MKT_LABEL[mkt], len(g), g["first"].min(), g["last"].max(), f"{int(g.clean_rows.median()):,}",
                     u["cost_bps"], "yes" if u["survivorship_free"] else "\\textbf{no}"])
    longtable(TEX / "s2_data.tex", "Study 2 universe after cleaning (Yahoo Finance daily data, cross-checked against "
              "FRED and MarketWatch). Cost = round-trip transaction cost + slippage in bps.", "tab:s2data",
              ["Market", "N", "First", "Last", "Median rows", "Cost (bps)", "Survivorship-free"], rows, "l r l l r r l")
    rows = [[esc(s["label"]), esc(s["reference"]), esc(", ".join(f"{k}={v}" for k, v in s["signal_grid"].items())),
             s["exit"], esc(", ".join(f"{k}={v}" for k, v in s["stop_grid"].items()))]
            for s in cfg["strategies"].values()]
    longtable(TEX / "s2_strategies.tex", "The five pre-registered strategies (the rule-based controls). Each "
              "instrument's parameters are re-selected by walk-forward on every fold.", "tab:s2strat",
              ["Strategy", "Reference", "Signal grid", "Exit", "Exit grid"], rows, "p{3.2cm} p{4cm} p{2.6cm} l p{3cm}")


# ------------------------------------------------------------------ headline (global portfolio)
def _headline(M, gl):
    rows = []
    for s in STRAT_LABEL:
        for fz in FRZ:
            g = gl[(gl.strategy == s) & (gl.freeze == fz)]
            if g.empty:
                continue
            M[f"STwoG{STRAT[s]}Base{FRZ[fz]}"] = f2(g.b_sharpe.iloc[0])
            for r in g.itertuples():
                key = f"STwoG{STRAT[s]}{COMP[r.component]}{FRZ[fz]}"
                M[key + "Delta"] = f2(r.delta_sharpe, True)
                M[key + "Var"] = f2(r.v_sharpe)
                M[key + "CI"] = f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"
                M[key + "Q"] = f"{r.q_bh:.3f}"
                rows.append([STRAT_LABEL[s], fz, COMP_LABEL[r.component], f2(r.b_sharpe), f2(r.v_sharpe),
                             f"{f2(r.delta_sharpe, True)}{_sig(r)}", f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]",
                             f"{r.q_bh:.3f}", pct(r.delta_max_drawdown, True), f2(r.delta_turnover, True)])
    longtable(TEX / "s2_headline.tex", "Before vs.\\ after on the all-market portfolio (equal weight across every "
              "instrument). $\\Delta$SR = annualized Sharpe(variant) $-$ Sharpe(rule-based control); CI = paired "
              "circular-block bootstrap (20-day blocks); $q$ = Benjamini--Hochberg across all "
              f"{len(gl)} global tests; $^{{+}}/^{{-}}$ = significant gain/loss at $q<0.05$. "
              "$\\Delta$MaxDD in percentage points (positive = shallower); $\\Delta$Turnover in capital/year.",
              "tab:s2headline", ["Strategy", "Freeze", "Component", "SR before", "SR after", "$\\Delta$SR",
                                 "95\\% CI", "$q$", "$\\Delta$MaxDD", "$\\Delta$Turn."], rows,
              "p{2.4cm} l p{2.5cm} r r r l r r r")


def _components(M, mk, gl):
    M["STwoMktTests"], M["STwoGlobTests"] = str(len(mk)), str(len(gl))
    for tag, t in (("Mkt", mk), ("Glob", gl)):
        M[f"STwo{tag}SigGain"] = str(int((t.significant & (t.delta_sharpe > 0)).sum()))
        M[f"STwo{tag}SigLoss"] = str(int((t.significant & (t.delta_sharpe < 0)).sum()))
        M[f"STwo{tag}PosShare"] = pct((t.delta_sharpe > 0).mean())
        M[f"STwo{tag}RawSig"] = str(int((t.p_value < 0.05).sum()))
        M[f"STwo{tag}HolmSig"] = str(int((t.p_holm < 0.05).sum()))
    rows = []
    for c in MAIN:
        t, g = mk[mk.component == c], gl[gl.component == c]
        if t.empty:
            continue
        piv = t.pivot_table(index=["strategy", "market"], columns="freeze", values="delta_sharpe")
        agree = (np.sign(piv["F2008"]) == np.sign(piv["F2018"])).dropna() if {"F2008", "F2018"} <= set(piv) else pd.Series(dtype=bool)
        both = piv.dropna() if {"F2008", "F2018"} <= set(piv) else piv.iloc[0:0]
        k = COMP[c]
        M[f"STwoMkt{k}N"] = str(len(t))
        M[f"STwoMkt{k}Pos"] = pct((t.delta_sharpe > 0).mean())
        M[f"STwoMkt{k}Gain"] = str(int((t.significant & (t.delta_sharpe > 0)).sum()))
        M[f"STwoMkt{k}Loss"] = str(int((t.significant & (t.delta_sharpe < 0)).sum()))
        M[f"STwoMkt{k}Med"] = f2(t.delta_sharpe.median(), True)
        M[f"STwoGlob{k}Med"] = f2(g.delta_sharpe.median(), True)
        M[f"STwoGlob{k}Pos"] = f"{int((g.delta_sharpe > 0).sum())}/{len(g)}"
        M[f"STwoRep{k}Agree"] = pct(agree.mean()) if len(agree) else "n/a"
        M[f"STwoRep{k}Corr"] = f2(both["F2008"].corr(both["F2018"])) if len(both) > 2 else "n/a"
        rows.append([COMP_LABEL[c], t.kind.iloc[0], len(t), pct((t.delta_sharpe > 0).mean()),
                     f2(t.delta_sharpe.median(), True), int((t.significant & (t.delta_sharpe > 0)).sum()),
                     int((t.significant & (t.delta_sharpe < 0)).sum()), f2(g.delta_sharpe.median(), True),
                     f"{int((g.delta_sharpe > 0).sum())}/{len(g)}", pct(agree.mean()) if len(agree) else "n/a",
                     f2(t.v_turnover.median() - t.b_turnover.median(), True)])
    longtable(TEX / "s2_components.tex", "Component attribution across the universe. Market-level tests: every "
              "strategy $\\times$ market $\\times$ freeze. Sig.\\ $+$/$-$ = BH-FDR $q<0.05$ within the family. "
              "Freeze agreement = share of strategy $\\times$ market cells whose $\\Delta$SR has the same sign in "
              "both independent freezes.", "tab:s2comp",
              ["Component", "Type", "Tests", "$\\Delta$SR$>0$", "Median $\\Delta$SR", "Sig.$+$", "Sig.$-$",
               "Global median", "Global $>0$", "Freeze agree", "$\\Delta$Turn."], rows, "p{2.6cm} l r r r r r r r r r")


def _by_strategy(mk):
    rows = []
    for s in STRAT_LABEL:
        row = [STRAT_LABEL[s]]
        for c in MAIN:
            t = mk[(mk.strategy == s) & (mk.component == c)]
            sg = int((t.significant & (t.delta_sharpe > 0)).sum()) - 0
            sl = int((t.significant & (t.delta_sharpe < 0)).sum())
            row.append(f"{f2(t.delta_sharpe.median(), True)} ({sg}/{sl})" if len(t) else "n/a")
        rows.append(row)
    longtable(TEX / "s2_bystrategy.tex", "Median market-level $\\Delta$SR by strategy and component (significant "
              "gains / significant losses at $q<0.05$ in parentheses).", "tab:s2bystrat",
              ["Strategy"] + [COMP_LABEL[c] for c in MAIN], rows, "p{2.4cm} " + "r " * len(MAIN))


def _metrics_table(gl):
    rows = []
    for s in STRAT_LABEL:
        for fz in FRZ:
            g = gl[(gl.strategy == s) & (gl.freeze == fz)]
            if g.empty:
                continue
            b = g.iloc[0]
            rows.append([STRAT_LABEL[s], fz, "\\textit{Rule-based control}", pct(b.b_ann_return), pct(b.b_ann_vol),
                         f2(b.b_sharpe), f2(b.b_sortino), pct(b.b_max_drawdown), pct(b.b_cvar95), f2(b.b_turnover),
                         f2(b.b_skew), int(b.b_n_trades) if pd.notna(b.b_n_trades) else ""])
            for r in g.itertuples():
                rows.append(["", "", COMP_LABEL[r.component], pct(r.v_ann_return), pct(r.v_ann_vol), f2(r.v_sharpe),
                             f2(r.v_sortino), pct(r.v_max_drawdown), pct(r.v_cvar95), f2(r.v_turnover), f2(r.v_skew),
                             int(r.v_n_trades) if pd.notna(r.v_n_trades) else ""])
    longtable(TEX / "s2_metrics.tex", "Full before/after risk profile of the all-market portfolio in each "
              "out-of-sample window: CAGR, annualized volatility, Sharpe, Sortino, maximum drawdown, daily CVaR(95\\%), "
              "one-way turnover (capital/year), skewness, trades.", "tab:s2metrics",
              ["Strategy", "Freeze", "Variant", "CAGR", "Vol", "SR", "Sortino", "MaxDD", "CVaR", "Turn.", "Skew",
               "Trades"], rows, "p{2.2cm} l p{2.6cm} r r r r r r r r r")


# ------------------------------------------------------------------ robustness
def _robustness(M, T, mk):
    main_med = mk.groupby("component").delta_sharpe.median()
    rows = []
    # cross-market transfer
    tr = T[T.family == "market_transfer"]
    for c, g in tr.groupby("component"):
        k = COMP[c]
        M[f"STwoTransfer{k}Med"], M[f"STwoTransfer{k}Pos"] = f2(g.delta_sharpe.median(), True), pct((g.delta_sharpe > 0).mean())
        M[f"STwoTransfer{k}Loss"] = str(int((g.significant & (g.delta_sharpe < 0)).sum()))
        rows.append(["Cross-market transfer", COMP_LABEL[c] + " (fit on other markets)", len(g),
                     f2(g.delta_sharpe.median(), True), pct((g.delta_sharpe > 0).mean()),
                     int((g.significant & (g.delta_sharpe > 0)).sum()), int((g.significant & (g.delta_sharpe < 0)).sum()),
                     f"in-market {f2(main_med.get(c), True)}"])
    ab = T[T.family == "market_ablation"]
    names = {"meta_label_thr045": ("Meta-label threshold 0.45", "Thr"), "meta_label_thr055": ("Meta-label threshold 0.55", "ThrHi"),
             "meta_label_small_model": ("Meta-label, small model", "MetaSmall"), "exit_small_model": ("ML exit, small model", "ExitSmall")}
    for c, g in ab.groupby("component"):
        lab, k = names.get(c, (c, "X"))
        M[f"STwoAbl{k}Med"] = f2(g.delta_sharpe.median(), True)
        rows.append(["Budget ablation", lab, len(g), f2(g.delta_sharpe.median(), True), pct((g.delta_sharpe > 0).mean()),
                     int((g.significant & (g.delta_sharpe > 0)).sum()), int((g.significant & (g.delta_sharpe < 0)).sum()),
                     f"main {f2(main_med.get('meta_label' if c.startswith('meta') else 'exit'), True)}"])
    ex = T[T.family == "global_execution"]
    for v, g in ex.groupby("variant"):
        comp, _, setting = v.partition("@")
        lab = f"{'Rule-based control' if comp == 'baseline' else COMP_LABEL.get(comp, comp)} under {setting.replace('_', ' ')}"
        M[f"STwoExec{COMP.get(comp, 'Base')}{'Stop' if 'stop' in setting else 'Lag'}Med"] = f2(g.delta_sharpe.median(), True)
        rows.append(["Execution (global)", esc(lab), len(g), f2(g.delta_sharpe.median(), True),
                     pct((g.delta_sharpe > 0).mean()), int((g.significant & (g.delta_sharpe > 0)).sum()),
                     int((g.significant & (g.delta_sharpe < 0)).sum()),
                     "vs.\\ control, same execution" if comp != "baseline" else "vs.\\ realistic control"])
    cs = pd.read_csv(AN / "cost_sensitivity.csv")
    for (mult, var), g in cs.groupby(["cost_multiplier", "variant"]):
        if var not in COMP:
            continue
        M[f"STwoCost{COMP[var]}{['Zero', 'One', 'Two', 'Four'][[0.0, 1.0, 2.0, 4.0].index(mult)]}"] = f2(g.delta_sharpe.median(), True)
    for c in ("meta_label", "best_combined", "exit"):
        g = cs[cs.variant == c]
        if len(g):
            rows.append(["Transaction costs (global)", COMP_LABEL[c] + ", costs $\\times$0/1/2/4", len(g) // 4,
                         " / ".join(f2(g[g.cost_multiplier == m].delta_sharpe.median(), True) for m in (0.0, 1.0, 2.0, 4.0)),
                         "", "", "", "median $\\Delta$SR per multiplier"])
    sv = mk.assign(bias=np.where(mk.survivorship_free.astype(str) == "True", "survivorship-free", "survivorship-prone"))
    for (b, c), g in sv.groupby(["bias", "component"]):
        if c in ("meta_label", "best_combined", "exit"):
            M[f"STwoSurv{COMP[c]}{'Free' if b.endswith('free') else 'Prone'}"] = f2(g.delta_sharpe.median(), True)
    for b, g in sv.groupby("bias"):
        rows.append(["Survivorship", f"all components, {b} markets", len(g), f2(g.delta_sharpe.median(), True),
                     pct((g.delta_sharpe > 0).mean()), int((g.significant & (g.delta_sharpe > 0)).sum()),
                     int((g.significant & (g.delta_sharpe < 0)).sum()), ""])
    jk = pd.read_csv(AN / "jackknife.csv")
    mj = jk[jk.scope == "market"]
    M["STwoJackFlipShare"] = pct((mj.sign_flips > 0).mean())
    sigtests = mk[mk.significant][["strategy", "market", "freeze", "variant"]]
    mjs = mj.merge(sigtests, on=["strategy", "market", "freeze", "variant"])
    M["STwoJackSigFlip"] = f"{int((mjs.sign_flips > 0).sum())}/{len(mjs)}"
    gj = jk[jk.scope == "global_minus_market"]
    M["STwoJackMarketFlipShare"] = pct((np.sign(gj.delta_without_market) != np.sign(gj.delta_full)).mean())
    rows.append(["Leave-one-out", "market tests whose $\\Delta$SR sign flips when any one instrument is dropped",
                 len(mj), "", pct((mj.sign_flips > 0).mean()), "", "", ""])
    dec = pd.read_csv(AN / "is_oos_decay.csv")
    for s, g in dec.groupby("strategy"):
        w = g.n_trades
        M[f"STwoDecay{STRAT[s]}Is"] = f"{np.average(g.is_mean_trade_bps, weights=w):.0f}"
        M[f"STwoDecay{STRAT[s]}Oos"] = f"{np.average(g.oos_mean_trade_bps, weights=w):.0f}"
    longtable(TEX / "s2_robustness.tex", "Robustness, generalization, and sensitivity. Counts are tests; "
              "Sig.\\ columns use BH-FDR within each family.", "tab:s2robust",
              ["Test", "Variant", "Tests", "Median $\\Delta$SR", "$\\Delta$SR$>0$", "Sig.$+$", "Sig.$-$", "Note"],
              rows, "p{2.4cm} p{4.2cm} r l r r r p{2.4cm}")
    rows = [[STRAT_LABEL[s], f"{np.average(g.is_mean_trade_bps, weights=g.n_trades):.1f}",
             f"{np.average(g.oos_mean_trade_bps, weights=g.n_trades):.1f}", int(g.n_trades.sum())]
            for s, g in dec.groupby("strategy", sort=False)]
    longtable(TEX / "s2_decay.tex", "Overfitting check of the rule-based controls: mean trade return (bps) of the "
              "walk-forward-selected parameters in-sample vs.\\ out-of-sample (trade-weighted across markets).",
              "tab:s2decay", ["Strategy", "In-sample", "Out-of-sample", "OOS trades"], rows, "l r r r")


def _regimes_crises(M):
    rg = pd.read_csv(AN / "regimes.csv")
    rg = rg[rg.variant.isin(MAIN)]
    rows = []
    for c in MAIN:
        g = rg[rg.variant == c]
        if g.empty:
            continue
        cells = []
        for reg in ("bull", "bear", "low_vol", "high_vol"):
            h = g[g.regime == reg]
            val = h.delta_sharpe.mean() if len(h) else np.nan
            M[f"STwoReg{COMP[c]}{reg.replace('_vol', 'vol').replace('_', '').title().replace('vol', 'Vol')}"] = f2(val, True)
            cells.append(f"{f2(val, True)} ({int((h.delta_sharpe > 0).sum())}/{len(h)})")
        rows.append([COMP_LABEL[c], *cells])
    longtable(TEX / "s2_regimes.tex", "Cross-regime test: mean $\\Delta$SR of the all-market portfolio within "
              "regimes defined from information at $t-1$ (market index vs.\\ its 200-day SMA; 20-day volatility vs.\\ "
              "its expanding median). In parentheses: strategy $\\times$ freeze cells with $\\Delta$SR$>0$.",
              "tab:s2reg", ["Component", "Bull", "Bear", "Low vol", "High vol"], rows, "l r r r r")
    cr = pd.read_csv(AN / "crises.csv")
    cr = cr[cr.variant.isin(MAIN)]
    rows = []
    for (crisis, c), g in cr.groupby(["crisis", "variant"], sort=False):
        rows.append([esc(crisis), COMP_LABEL[c], len(g), pct(g.b_return.mean(), True), pct(g.v_return.mean(), True),
                     pct(g.b_max_drawdown.mean()), pct(g.v_max_drawdown.mean())])
    longtable(TEX / "s2_crises.tex", "Stress test: return and maximum drawdown of the all-market portfolio inside "
              "pre-registered crisis windows (mean over strategies).", "tab:s2crisis",
              ["Crisis", "Component", "N", "Return before", "Return after", "MaxDD before", "MaxDD after"], rows,
              "l l r r r r r")


# ------------------------------------------------------------------ figures
def _figures(mk, gl):
    import matplotlib.pyplot as plt

    from .report import GAIN, LOSS

    strategies = [s for s in STRAT_LABEL if s in set(mk.strategy)]
    markets = [m for m in MKT_LABEL if m in set(mk.market)]
    fig, axes = plt.subplots(1, len(strategies), figsize=(3.3 * len(strategies), 3.6), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], strategies):
        t = mk[mk.strategy == s]
        mat = t.pivot_table(index="component", columns="market", values="delta_sharpe", aggfunc="mean")
        mat = mat.reindex(index=MAIN, columns=markets)
        sig = t.pivot_table(index="component", columns="market", values="significant", aggfunc="max").reindex(
            index=MAIN, columns=markets)
        im = ax.imshow(mat.to_numpy(float), cmap="RdBu", vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.1f}{'*' if sig.iat[i, j] is True else ''}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(markets)), [MKT_LABEL[m] for m in markets], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(MAIN)), [COMP_LABEL[c] for c in MAIN], fontsize=7)
        ax.set_title(STRAT_LABEL[s], fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="mean delta Sharpe (2 freezes)")
    fig.suptitle("Study 2: where ML changes each strategy (* = significant at q<0.05 in at least one freeze)", fontsize=9)
    fig.savefig(FIG / "s2_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    g = gl.copy()
    g["label"] = [f"{STRAT_LABEL[s]} | {COMP_LABEL[c]}" for s, c in zip(g.strategy, g.component)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.22 * len(g) / 2 + 1.5), sharey=True)
    labels = list(dict.fromkeys(g.label))
    for ax, fz in zip(axes, FRZ):
        h = g[g.freeze == fz].set_index("label").reindex(labels)
        y = np.arange(len(labels))[::-1]
        for yi, (_, r) in zip(y, h.iterrows()):
            if pd.isna(r.delta_sharpe):
                continue
            col = (GAIN if r.delta_sharpe > 0 else LOSS) if r.significant else "#57606a"
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=1.2)
            ax.scatter([r.delta_sharpe], [yi], s=18, color=col, facecolors=col if r.significant else "white", zorder=3)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y, labels, fontsize=6)
        ax.set_title(f"Freeze {fz}", fontsize=9)
        ax.set_xlabel("delta Sharpe (annualized), 95% block-bootstrap CI", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "s2_forest_global.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    cs = pd.read_csv(AN / "cost_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for c in MAIN:
        h = cs[cs.variant == c].groupby("cost_multiplier").delta_sharpe.median()
        if len(h):
            ax.plot(h.index, h.values, marker="o", label=COMP_LABEL[c])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("transaction-cost multiplier")
    ax.set_ylabel("median delta Sharpe (global)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "s2_costs.png", dpi=200)
    plt.close(fig)

    daily = pd.read_parquet(AN / "global_daily.parquet")
    fig, axes = plt.subplots(len(strategies), 2, figsize=(10, 2.1 * len(strategies)), squeeze=False)
    for i, s in enumerate(strategies):
        for j, fz in enumerate(FRZ):
            ax = axes[i][j]
            for v, col in (("baseline", "black"), ("best_combined", "#0969da"), ("exit", "#cf222e"), ("meta_label", "#1a7f37")):
                h = daily[(daily.strategy == s) & (daily.variant == v) & (daily.freeze == fz)].sort_values("date")
                if len(h):
                    ax.plot(h.date, (1 + h.ret).cumprod(), lw=0.9, color=col,
                            label="rule-based control" if v == "baseline" else COMP_LABEL[v])
            ax.set_title(f"{STRAT_LABEL[s]} - {fz}", fontsize=8)
            ax.tick_params(labelsize=6)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "s2_equity.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ bias audit
def _bias_audit(M, T, mk):
    rows = [
        ["Look-ahead", "Truncation invariance of every feature and every strategy signal (unit tests); regimes use $t-1$ data",
         "pass (tests/test\\_core.py, tests/test\\_study2.py)"],
        ["Data leakage", "All models fit once before the freeze; scrambling all post-freeze data leaves every model "
         "bit-identical; labels straddling the freeze excluded", "pass (tests)"],
        ["Survivorship", "Indices/ETFs/futures/FX analysed separately from current-constituent stocks and crypto",
         f"median $\\Delta$SR meta-label+sizing: free \\STwoSurvBestFree{{}} vs.\\ prone \\STwoSurvBestProne{{}}"],
        ["Selection", "Universe, strategies, components, windows, statistics pre-registered in git before data "
         "existed; every variant reported", "commit 8d9f50f precedes all data/results"],
        ["Overfitting", "Walk-forward parameter selection; deflated Sharpe; IS vs.\\ OOS trade-return decay; two "
         "independent freezes", "see Table~\\ref{tab:s2decay} and freeze agreement in Table~\\ref{tab:s2comp}"],
        ["Multiple testing", f"BH-FDR and Holm within families ({len(mk)} market-level main tests)",
         f"raw $p<0.05$: \\STwoMktRawSig{{}}; BH: \\STwoMktSigGain{{}} gains / \\STwoMktSigLoss{{}} losses; "
         "Holm: \\STwoMktHolmSig{{}}"],
        ["Execution realism", "Close-fill of breached stops (default), 1-day entry lag, cost $\\times$0--4, per-market costs",
         "Table~\\ref{tab:s2robust}"],
        ["Data errors", "Two independent sources (FRED) + browser spot-check (MarketWatch)",
         "5-day return corr.\\ \\STwoFredCorrFiveMin{}--\\STwoFredCorrFiveMax{}; spot-check max diff \\STwoSpotMaxDiff{}"],
    ]
    longtable(TEX / "s2_bias.tex", "Bias audit: each threat to validity, the test applied, and its outcome.",
              "tab:s2bias", ["Bias", "Test", "Outcome"], rows, "p{2.2cm} p{7cm} p{5.2cm}")
