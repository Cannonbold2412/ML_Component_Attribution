"""Study 2 part of the paper-asset generator (called by mbca.paper_assets.build)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .datasets import CLEAN, RAW, VALID
from .paper_assets import AN, COMP, COMP_LABEL, FRZ, MKT_LABEL, STRAT, STRAT_LABEL, TEX, Macros, esc, f2, pct, table

MAIN = list(COMP)
ML = [c for c in MAIN if c != "rule_adx"]


def study2(M: Macros) -> None:
    cfg = load_config()
    T = pd.read_csv(AN / "tests.csv")
    mk = T[T.family == "market_main"]
    gl = T[T.family == "global_main"]
    _data(M, cfg)
    _headline(M, gl)
    _components(M, mk, gl)
    _by_strategy(mk)
    _robustness(M, T, mk)
    _regimes(M)
    _bias_audit(M, T, mk)


# ------------------------------------------------------------------ data
def _data(M, cfg):
    man = pd.read_csv(CLEAN / "MANIFEST.csv")
    ok = man[man.status == "ok"]
    raw = pd.read_csv(RAW / "MANIFEST.csv")
    M["STwoInstruments"] = str(len(ok))
    M["STwoPlanned"] = str(sum(len(v["tickers"]) for v in cfg["universe"].values()))
    M["STwoMarkets"] = str(ok.market.nunique())
    for mkt, n in ok.market.value_counts().items():
        M["STwoN" + mkt.replace("_", "")] = str(n)
    stocks = ok[ok.market.str.startswith("stocks")].symbol
    suffix = stocks.str.extract(r"\.([A-Z]+)$")[0].fillna("US")  # exchange suffix; none = US listing
    M["STwoExchanges"] = str(suffix.nunique())
    for mkt, u in cfg["universe"].items():
        M["STwoCost" + mkt.replace("_", "")] = f"{u['cost_bps']:g}"
    st = cfg["statistics"]
    from .study import SMALL_MODEL, make_components
    abl = make_components(["meta_label_thr045", "meta_label_thr055"])
    M["STwoAlpha"] = f"{st['alpha']:g}"
    M["STwoThrLo"], M["STwoThrHi"] = f"{abl['meta_label_thr045'].threshold:g}", f"{abl['meta_label_thr055'].threshold:g}"
    M["STwoSmallTrees"], M["STwoSmallDepth"] = str(SMALL_MODEL["n_estimators"]), str(SMALL_MODEL["max_depth"])
    M["STwoBlock"], M["STwoBoot"] = str(st["block_days"]), f"{st['n_boot']:,}".replace(",", "{,}")
    M["STwoCostMults"] = ", ".join(f"$\\times{m:g}$" for m in st["cost_multipliers"] if m != 1)
    fz = cfg["study"]["freezes"]
    M["STwoStart"] = cfg["study"]["data_start"][:4]
    M["STwoFzA"], M["STwoFzB"] = fz[0]["split"][:4], fz[1]["split"][:4]
    M["STwoFzAEnd"] = str(int(fz[0]["test_end"][:4]) - 1)
    M["STwoFzATrainEnd"], M["STwoFzBTrainEnd"] = str(int(fz[0]["split"][:4]) - 1), str(int(fz[1]["split"][:4]) - 1)
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
        rows.append([MKT_LABEL[mkt], len(g), g["first"].min(), f"{int(g.clean_rows.median()):,}", u["cost_bps"],
                     "yes" if u["survivorship_free"] else "\\textbf{no}"])
    table(TEX / "s2_data.tex", "Study 2 universe after cleaning (Yahoo Finance daily bars to \\STwoLast{}, "
          "cross-checked against FRED and MarketWatch). Cost = round-trip cost plus slippage (bps).", "tab:s2data",
          ["Market", "N", "First bar", "Median bars", "Cost", "Surv.-free"], rows, "l r l r r l")
    ref = {"jma_trend": "Study 1 control", "sma_cross": "\\cite{brock1992simple}", "donchian": "\\cite{faith2007turtle}",
           "ts_momentum": "\\cite{moskowitz2012momentum}", "rsi2_reversion": "\\cite{connors2008short}"}
    par = {"sl_mult": "stop", "trail_mult": "trail", "tp_mult": "target"}

    def grid(g):
        return "; ".join(f"{esc(par.get(k, k))} $\\in$ \\{{{', '.join(f'{x:g}' for x in v)}\\}}" for k, v in g.items())

    rows = [[STRAT_LABEL[k], ref[k], grid(s["signal_grid"]), s["exit"], grid(s["stop_grid"])]
            for k, s in cfg["strategies"].items()]
    table(TEX / "s2_strategies.tex", "The five pre-registered strategies (the rule-based controls). Each "
          "instrument's parameters are re-selected from these grids by walk-forward on every fold; exit multiples "
          "are in units of ATR.", "tab:s2strat",
          ["Strategy", "Reference", "Signal grid", "Exit", "Exit grid ($\\times$ATR)"], rows, "l l l l l", wide=True)


# ------------------------------------------------------------------ headline (global portfolio)
def _headline(M, gl):
    """Macros for every all-market before/after comparison (plotted as Fig. s2_forest_global, full CSV in analysis/)."""
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


def _components(M, mk, gl):
    M["STwoMktTests"], M["STwoGlobTests"] = str(len(mk)), str(len(gl))
    for tag, t in (("Mkt", mk), ("Glob", gl)):
        M[f"STwo{tag}SigGain"] = str(int((t.significant & (t.delta_sharpe > 0)).sum()))
        M[f"STwo{tag}SigLoss"] = str(int((t.significant & (t.delta_sharpe < 0)).sum()))
        M[f"STwo{tag}PosShare"] = pct((t.delta_sharpe > 0).mean())
        M[f"STwo{tag}RawSig"] = str(int((t.p_value < 0.05).sum()))
        M[f"STwo{tag}HolmSig"] = str(int((t.p_holm < 0.05).sum()))
    M["STwoMktNullExp"] = f"{0.05 * len(mk):.0f}"
    rows = []
    for c in MAIN:
        t, g = mk[mk.component == c], gl[gl.component == c]
        if t.empty:
            continue
        piv = t.pivot_table(index=["strategy", "market"], columns="freeze", values="delta_sharpe")
        both = piv.dropna() if {"F2008", "F2018"} <= set(piv) else piv.iloc[0:0]
        # only cells observed in both freezes can agree (crypto has no F2008 cell); see ERRATA.md
        agree = np.sign(both["F2008"]) == np.sign(both["F2018"]) if len(both) else pd.Series(dtype=bool)
        k = COMP[c]
        M[f"STwoMkt{k}N"] = str(len(t))
        M[f"STwoMkt{k}Pos"] = pct((t.delta_sharpe > 0).mean())
        M[f"STwoMkt{k}Gain"] = str(int((t.significant & (t.delta_sharpe > 0)).sum()))
        M[f"STwoMkt{k}Loss"] = str(int((t.significant & (t.delta_sharpe < 0)).sum()))
        M[f"STwoMkt{k}Med"] = f2(t.delta_sharpe.median(), True)
        rsi, other = t[t.strategy == "rsi2_reversion"], t[t.strategy != "rsi2_reversion"]
        M[f"STwoMkt{k}GainRSI"] = str(int((rsi.significant & (rsi.delta_sharpe > 0)).sum()))
        M[f"STwoMkt{k}LossRSI"] = str(int((rsi.significant & (rsi.delta_sharpe < 0)).sum()))
        M[f"STwoMkt{k}MedTrend"] = f2(other.delta_sharpe.median(), True)
        M[f"STwoMkt{k}MedRSI"] = f2(rsi.delta_sharpe.median(), True)
        M[f"STwoMkt{k}Turn"] = f2((t.v_turnover - t.b_turnover).median(), True)
        M[f"STwoGlob{k}Med"] = f2(g.delta_sharpe.median(), True)
        M[f"STwoGlob{k}Pos"] = f"{int((g.delta_sharpe > 0).sum())}/{len(g)}"
        M[f"STwoGlob{k}DD"] = f"{int((g.delta_max_drawdown > 0).sum())}/{len(g)}"
        M[f"STwoGlob{k}Exp"] = pct((g.v_exposure - g.b_exposure).median(), True)
        M[f"STwoRep{k}Agree"] = pct(agree.mean()) if len(agree) else "n/a"
        M[f"STwoRep{k}Corr"] = f2(both["F2008"].corr(both["F2018"])) if len(both) > 2 else "n/a"
        rows.append([COMP_LABEL[c], t.kind.iloc[0], len(t), pct((t.delta_sharpe > 0).mean()),
                     f2(t.delta_sharpe.median(), True), int((t.significant & (t.delta_sharpe > 0)).sum()),
                     int((t.significant & (t.delta_sharpe < 0)).sum()), f2(g.delta_sharpe.median(), True),
                     f"{int((g.delta_sharpe > 0).sum())}/{len(g)}", pct(agree.mean()) if len(agree) else "n/a",
                     f2(t.v_turnover.median() - t.b_turnover.median(), True)])
    table(TEX / "s2_components.tex", "Component attribution across the universe. Market-level tests: every "
          "strategy $\\times$ market $\\times$ freeze. Sig.\\ $+$/$-$ = BH-FDR $q<0.05$ within the family. "
          "Global = all-market portfolio (strategy $\\times$ freeze). Freeze agreement = share of strategy $\\times$ "
          "market cells whose $\\Delta$SR has the same sign in both independent freezes. $\\Delta$Turn.\\ in "
          "capital/year.", "tab:s2comp",
          ["Component", "Type", "Tests", "$\\Delta$SR$>0$", "Median $\\Delta$SR", "Sig.$+$", "Sig.$-$",
           "Global median", "Global $>0$", "Freeze agree", "$\\Delta$Turn."], rows, "l l r r r r r r r r r", wide=True)


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
    table(TEX / "s2_bystrategy.tex", "Median market-level $\\Delta$SR by strategy and component (significant "
          "gains / significant losses at $q<0.05$ in parentheses).", "tab:s2bystrat",
          ["Strategy", "ADX gate", "ML regime", "ML meta", "ML sizing", "ML entry", "ML exit", "Meta+sizing"], rows,
          "l" + " r" * len(MAIN), wide=True)


# ------------------------------------------------------------------ robustness (plotted in Figs. s2_robust, s2_costs,
# s2_jackknife, s2_decay, s2_regimes_crises; full numbers in results/study2/analysis/)
def _robustness(M, T, mk):
    for c, g in T[T.family == "market_transfer"].groupby("component"):
        k = COMP[c]
        M[f"STwoTransfer{k}Med"], M[f"STwoTransfer{k}Pos"] = f2(g.delta_sharpe.median(), True), pct((g.delta_sharpe > 0).mean())
        M[f"STwoTransfer{k}Loss"] = str(int((g.significant & (g.delta_sharpe < 0)).sum()))
    names = {"meta_label_thr045": "Thr", "meta_label_thr055": "ThrHi", "meta_label_small_model": "MetaSmall",
             "exit_small_model": "ExitSmall"}
    for c, g in T[T.family == "market_ablation"].groupby("component"):
        M[f"STwoAbl{names.get(c, 'X')}Med"] = f2(g.delta_sharpe.median(), True)
    for v, g in T[T.family == "global_execution"].groupby("variant"):
        comp, _, setting = v.partition("@")
        tag = f"STwoExec{COMP.get(comp, 'Base')}{'Stop' if 'stop' in setting else 'Lag'}"
        M[tag + "Med"] = f2(g.delta_sharpe.median(), True)
        M[tag + "Sig"] = f"{int(g.significant.sum())}/{len(g)}"
        M[tag + "Min"], M[tag + "Max"] = f2(g.delta_sharpe.min(), True), f2(g.delta_sharpe.max(), True)
    cs = pd.read_csv(AN / "cost_sensitivity.csv")
    for (mult, var), g in cs.groupby(["cost_multiplier", "variant"]):
        if var in COMP:
            k = ["Zero", "One", "Two", "Four"][[0.0, 1.0, 2.0, 4.0].index(mult)]
            M[f"STwoCost{COMP[var]}{k}"] = f2(g.delta_sharpe.median(), True)
    sv = mk.assign(bias=np.where(mk.survivorship_free.astype(str) == "True", "free", "prone"))
    for (b, c), g in sv.groupby(["bias", "component"]):
        if c in ("meta_label", "best_combined", "exit"):
            M[f"STwoSurv{COMP[c]}{b.title()}"] = f2(g.delta_sharpe.median(), True)
    jk = pd.read_csv(AN / "jackknife.csv")
    mj = jk[jk.scope == "market"]
    M["STwoJackFlipShare"] = pct((mj.sign_flips > 0).mean())
    mjs = mj.merge(mk[mk.significant][["strategy", "market", "freeze", "variant"]],
                   on=["strategy", "market", "freeze", "variant"])
    M["STwoJackSigFlip"] = f"{int((mjs.sign_flips > 0).sum())}/{len(mjs)}"
    gj = jk[jk.scope == "global_minus_market"]
    M["STwoJackMarketFlipShare"] = pct((np.sign(gj.delta_without_market) != np.sign(gj.delta_full)).mean())
    dec = pd.read_csv(AN / "is_oos_decay.csv")
    for s, g in dec.groupby("strategy"):
        w = g.n_trades
        M[f"STwoDecay{STRAT[s]}Is"] = f"{np.average(g.is_mean_trade_bps, weights=w):.0f}"
        M[f"STwoDecay{STRAT[s]}Oos"] = f"{np.average(g.oos_mean_trade_bps, weights=w):.0f}"


def _regimes(M):
    rg = pd.read_csv(AN / "regimes.csv")
    for c in MAIN:
        g = rg[rg.variant == c]
        for reg, tag in (("bull", "Bull"), ("bear", "Bear"), ("low_vol", "LowVol"), ("high_vol", "HighVol")):
            h = g[g.regime == reg]
            M[f"STwoReg{COMP[c]}{tag}"] = f2(h.delta_sharpe.mean() if len(h) else np.nan, True)


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
         "existed; every variant reported", f"study.toml first committed in {_prereg_commit()}, before all data/results"],
        ["Overfitting", "Walk-forward parameter selection; deflated Sharpe; IS vs.\\ OOS trade-return decay; two "
         "independent freezes", "Fig.~\\ref{fig:s2decay}; freeze agreement in Table~\\ref{tab:s2comp}"],
        ["Multiple testing", f"BH-FDR and Holm within families ({len(mk)} market-level main tests)",
         "raw $p<0.05$: \\STwoMktRawSig{}; BH: \\STwoMktSigGain{} gains / \\STwoMktSigLoss{} losses; "
         "Holm: \\STwoMktHolmSig{}"],
        ["Execution realism", "Close-fill of breached stops (default), 1-day entry lag, cost $\\times$0--4, per-market costs",
         "Fig.~\\ref{fig:s2robust}"],
        ["Data errors", "Two independent sources (FRED) + browser spot-check (MarketWatch)",
         "5-day return corr.\\ \\STwoFredCorrFiveMin{}--\\STwoFredCorrFiveMax{}; spot-check max diff \\STwoSpotMaxDiff{}"],
    ]
    table(TEX / "s2_bias.tex", "Bias audit: each threat to validity, the test applied, and its outcome.",
          "tab:s2bias", ["Threat", "Test applied", "Outcome"], rows, "p{2.0cm} p{8.2cm} p{6.5cm}", wide=True)


def _prereg_commit() -> str:
    """The commit that first added study.toml (read from git, not typed)."""
    import subprocess

    from .config import ROOT
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", "study.toml"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return out[-1] if out else "unknown"
