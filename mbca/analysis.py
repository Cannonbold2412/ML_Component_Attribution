"""Study 2 analysis: trades -> every statistic the paper reports.

    python -m mbca study analyze

Reads results/study2/trades/*.parquet (+ clean data, study.toml) and writes
results/study2/analysis/*.csv. Nothing here is hand-edited; the paper's tables,
figures and in-text numbers are generated from these files by mbca.paper_assets.

Families of hypothesis tests (BH-FDR and Holm are applied within each family):
  market_main   rule gate + 6 ML components  x strategy x market x freeze
  global_main   same components on the all-market portfolio x strategy x freeze
  transfer      models fit on every other market (cross-market generalization)
  ablation      threshold and model-capacity variants of the ML budget
  execution     ML variants vs baseline under optimistic-fill / one-day-lag execution
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluation as E
from .config import ROOT, load_config
from .datasets import load_study2

OUT = ROOT / "results" / "study2"
AN = OUT / "analysis"
MAIN = ["rule_adx", "regime_linearity", "meta_label", "position_sizing", "entry", "exit", "best_combined"]
KIND = {"rule_adx": "rule", "regime_linearity": "independent", "meta_label": "narrowing",
        "position_sizing": "narrowing", "entry": "replacing", "exit": "replacing", "best_combined": "combination"}


class Acc:
    """Accumulates instrument-level daily MTM frames into portfolio sums, so market and
    global portfolios (and leave-one-out variants) never need all series in memory."""

    def __init__(self):
        self.ret, self.turn, self.exp, self.n = {}, {}, {}, {}

    def add(self, key, d: pd.DataFrame):
        turn = d["pos"].diff().abs().fillna(d["pos"].abs())
        for store, s in ((self.ret, d["ret"]), (self.turn, turn), (self.exp, d["pos"].abs()),
                         (self.n, pd.Series(1.0, index=d.index))):
            store[key] = s if key not in store else store[key].add(s, fill_value=0)

    def frame(self, key, lo=None, hi=None, minus=None) -> pd.DataFrame:
        parts = [self.ret[key], self.turn[key], self.exp[key], self.n[key]]
        if minus is not None:  # leave-one-out: subtract another accumulator's sums
            parts = [p.sub(q, fill_value=0) for p, q in zip(parts, [minus.ret[key], minus.turn[key],
                                                                     minus.exp[key], minus.n[key]])]
        r, t, e, n = parts
        n = n.clip(lower=1)
        pf = pd.DataFrame({"ret": r / n, "turnover": t / n, "exposure": e / n, "n": n}).sort_index()
        if lo is not None:
            pf = pf[pf.index >= pd.Timestamp(lo)]
        if hi is not None:
            pf = pf[pf.index < pd.Timestamp(hi)]
        return pf


def _windows(cfg):
    return {f["name"]: (pd.Timestamp(f["split"]), pd.Timestamp(f["test_end"])) for f in cfg["study"]["freezes"]}


def _compare(var_pf, base_pf, st, n_trials, var_trades=None, base_trades=None) -> dict:
    """Metrics of both portfolios on their common window + paired block-bootstrap test."""
    idx = var_pf.index.union(base_pf.index)
    v, b = var_pf.reindex(idx).fillna(0), base_pf.reindex(idx).fillna(0)
    mv, mb = E.metrics(v, var_trades, n_trials), E.metrics(b, base_trades, 1)
    test = E.block_bootstrap_sharpe_diff(v["ret"], b["ret"], st["block_days"], st["n_boot"], 0, st["alpha"])
    row = {f"v_{k}": x for k, x in mv.items()} | {f"b_{k}": x for k, x in mb.items()} | test
    for k in ("ann_return", "ann_vol", "sortino", "max_drawdown", "cvar95", "turnover"):
        if f"v_{k}" in row and f"b_{k}" in row:
            row[f"delta_{k}"] = row[f"v_{k}"] - row[f"b_{k}"]
    return row


def _in(trades, lo, hi):
    if trades is None or trades.empty:
        return trades
    et = pd.to_datetime(trades["entry_time"])
    return trades[(et >= lo) & (et < hi)]


def analyze() -> None:
    cfg = load_config()
    st, wins = cfg["statistics"], _windows(cfg)
    data = load_study2()
    AN.mkdir(parents=True, exist_ok=True)
    rows, cost_rows, loo_rows, decay_rows, bh_rows = [], [], [], [], []
    glob = {}   # strategy -> Acc over every instrument (global portfolio)
    per_mkt = {}  # (strategy, market) -> Acc, for leave-one-market-out
    glob_trades = {}
    cost_glob = {}  # (strategy, mult) -> Acc for cost sensitivity of main variants
    for s_key in cfg["strategies"]:
        glob[s_key] = Acc()
        for mult in st["cost_multipliers"]:
            cost_glob[(s_key, mult)] = Acc()
        for market, frames in data.items():
            path = OUT / "trades" / f"{s_key}__{market}.parquet"
            if not path.exists():
                continue
            tr = pd.read_parquet(path)
            cost = cfg["universe"][market]["cost_bps"]
            acc, inst = Acc(), {}
            for (variant, freeze, sym), t in tr.groupby(["variant", "freeze", "symbol"], sort=False):
                d = E.instrument_daily(t, frames[sym], cost)
                acc.add((variant, freeze), d)
                inst[(variant, freeze, sym)] = d
                glob_trades.setdefault((s_key, variant, freeze), []).append(t)
                base_v = variant.split("@")[0]
                if base_v in MAIN + ["baseline"] and "@" not in variant:
                    for mult in st["cost_multipliers"]:
                        dm = d if mult == 1.0 else E.instrument_daily(t, frames[sym], cost * mult)
                        cost_glob[(s_key, mult)].add((variant, freeze), dm)
            per_mkt[(s_key, market)] = acc
            _market_tests(rows, loo_rows, s_key, market, acc, inst, tr, wins, st, cfg)
            # in-sample vs out-of-sample decay of the walk-forward parameter selection (baseline)
            b = tr[tr["variant"] == "baseline"]
            if len(b):
                decay_rows.append({"strategy": s_key, "market": market,
                                   "is_mean_trade_bps": float(b["is_mean_ret"].mean() * 1e4),
                                   "oos_mean_trade_bps": float(b["ret"].mean() * 1e4), "n_trades": len(b)})
            # buy and hold benchmark per market and window
            idx = E.market_index(frames)
            for fz, (lo, hi) in wins.items():
                r = idx[(idx.index >= lo) & (idx.index < hi)]
                m = E.metrics(pd.DataFrame({"ret": r, "turnover": 0.0, "exposure": 1.0}))
                bh_rows.append({"strategy": s_key, "market": market, "freeze": fz, **m})
        # global accumulators: rebuild properly from per-market sums (exact, incl. turnover/exposure)
        g = Acc()
        for (sk, _m), a in per_mkt.items():
            if sk != s_key:
                continue
            for k in a.ret:
                for store, src in ((g.ret, a.ret), (g.turn, a.turn), (g.exp, a.exp), (g.n, a.n)):
                    store[k] = src[k] if k not in store else store[k].add(src[k], fill_value=0)
        glob[s_key] = g
    _global_tests(rows, cost_rows, loo_rows, glob, per_mkt, cost_glob, glob_trades, wins, st, cfg, data)

    res = pd.DataFrame(rows)
    for fam, g in res.groupby("family"):
        res.loc[g.index, "q_bh"] = E.bh_fdr(g["p_value"]).to_numpy()
        res.loc[g.index, "p_holm"] = E.holm(g["p_value"]).to_numpy()
    res["significant"] = res["q_bh"] < st["alpha"]
    res.to_csv(AN / "tests.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(AN / "cost_sensitivity.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(AN / "jackknife.csv", index=False)
    pd.DataFrame(decay_rows).to_csv(AN / "is_oos_decay.csv", index=False)
    pd.DataFrame(bh_rows).drop_duplicates(["market", "freeze"]).drop(columns="strategy").to_csv(
        AN / "buy_and_hold.csv", index=False)
    _data_summary()
    print(f"analysis written to {AN}: {len(res)} tests")


def _families(variant: str) -> tuple[str, str, str]:
    """variant -> (family, component, comparison baseline variant)."""
    comp, _, setting = variant.partition("@")
    if setting == "transfer":
        return "transfer", comp, "baseline"
    if setting:
        return "execution", comp, f"baseline@{setting}"
    if comp in MAIN:
        return "main", comp, "baseline"
    return "ablation", comp, "baseline"


def _market_tests(rows, loo_rows, s_key, market, acc, inst, tr, wins, st, cfg):
    n_trials = len(MAIN)
    for (variant, freeze) in list(acc.ret):
        if variant.startswith("baseline"):
            continue
        fam, comp, base_v = _families(variant)
        fz_list = list(wins) if freeze == "all" else [freeze]
        for fz in fz_list:
            lo, hi = wins[fz]
            if (base_v, "all") not in acc.ret:
                continue
            vpf, bpf = acc.frame((variant, freeze), lo, hi), acc.frame((base_v, "all"), lo, hi)
            if vpf.empty or bpf.empty or len(bpf) < 250:
                continue
            vt = _in(tr[(tr.variant == variant) & (tr.freeze == freeze)], lo, hi)
            bt = _in(tr[(tr.variant == base_v) & (tr.freeze == "all")], lo, hi)
            row = _compare(vpf, bpf, st, n_trials, vt, bt)
            rows.append({"family": f"market_{fam}", "scope": "market", "strategy": s_key, "market": market,
                         "freeze": fz, "variant": variant, "component": comp, "kind": KIND.get(comp, "ablation"),
                         "survivorship_free": cfg["universe"][market]["survivorship_free"], **row})
            if fam == "main":  # leave-one-instrument-out stability of the sign of delta Sharpe
                syms = sorted({k[2] for k in inst if k[0] == variant and k[1] == freeze})
                deltas = []
                for sym in syms:
                    one_v, one_b = Acc(), Acc()
                    one_v.add((variant, freeze), inst[(variant, freeze, sym)])
                    if (base_v, "all", sym) in inst:
                        one_b.add((base_v, "all"), inst[(base_v, "all", sym)])
                    v2 = acc.frame((variant, freeze), lo, hi, minus=one_v)
                    b2 = (acc.frame((base_v, "all"), lo, hi, minus=one_b) if (base_v, "all") in one_b.ret
                          else acc.frame((base_v, "all"), lo, hi))
                    j = v2.index.union(b2.index)
                    deltas.append(E.sharpe(v2["ret"].reindex(j).fillna(0)) - E.sharpe(b2["ret"].reindex(j).fillna(0)))
                if deltas:
                    d = np.array(deltas)
                    loo_rows.append({"scope": "market", "strategy": s_key, "market": market, "freeze": fz,
                                     "variant": variant, "delta_full": row["delta_sharpe"], "loo_min": d.min(),
                                     "loo_max": d.max(), "n_left_out": len(d),
                                     "sign_flips": int((np.sign(d) != np.sign(row["delta_sharpe"])).sum())})


def _global_tests(rows, cost_rows, loo_rows, glob, per_mkt, cost_glob, glob_trades, wins, st, cfg, data):
    regimes_rows, crisis_rows, daily_rows = [], [], []
    all_frames = {s: d for fr in data.values() for s, d in fr.items()}
    gidx = E.market_index(all_frames)
    labels = E.regime_labels(gidx, cfg["regimes"]["trend_sma_days"], cfg["regimes"]["vol_window_days"])
    for s_key, g in glob.items():
        for (variant, freeze) in list(g.ret):
            if variant.startswith("baseline") and "@" not in variant:
                continue
            fam, comp, base_v = _families(variant) if not variant.startswith("baseline@") else \
                ("execution", "baseline", "baseline")
            for fz in (list(wins) if freeze == "all" else [freeze]):
                lo, hi = wins[fz]
                if (base_v, "all") not in g.ret:
                    continue
                vpf, bpf = g.frame((variant, freeze), lo, hi), g.frame((base_v, "all"), lo, hi)
                if vpf.empty or len(bpf) < 250:
                    continue
                vt = _in(pd.concat(glob_trades[(s_key, variant, freeze)]), lo, hi)
                bt = _in(pd.concat(glob_trades[(s_key, base_v, "all")]), lo, hi)
                row = _compare(vpf, bpf, st, len(MAIN), vt, bt)
                if fam == "main":
                    daily_rows.append(vpf["ret"].rename((s_key, variant, fz)))
                    if variant == "rule_adx" or not any(k.name == (s_key, "baseline", fz) for k in daily_rows):
                        daily_rows.append(bpf["ret"].rename((s_key, "baseline", fz)))
                rows.append({"family": f"global_{fam}", "scope": "global", "strategy": s_key, "market": "ALL",
                             "freeze": fz, "variant": variant, "component": comp,
                             "kind": KIND.get(comp, "ablation"), "survivorship_free": None, **row})
                if fam != "main":
                    continue
                # leave-one-market-out
                for (sk, m), a in per_mkt.items():
                    if sk != s_key or (variant, freeze) not in a.ret or (base_v, "all") not in a.ret:
                        continue
                    v2, b2 = g.frame((variant, freeze), lo, hi, minus=a), g.frame((base_v, "all"), lo, hi, minus=a)
                    j = v2.index.union(b2.index)
                    loo_rows.append({"scope": "global_minus_market", "strategy": s_key, "market": m, "freeze": fz,
                                     "variant": variant, "delta_full": row["delta_sharpe"],
                                     "delta_without_market": E.sharpe(v2["ret"].reindex(j).fillna(0))
                                     - E.sharpe(b2["ret"].reindex(j).fillna(0))})
                # cost sensitivity
                for mult in st["cost_multipliers"]:
                    cg = cost_glob[(s_key, mult)]
                    if (variant, freeze) in cg.ret:
                        v3, b3 = cg.frame((variant, freeze), lo, hi), cg.frame(("baseline", "all"), lo, hi)
                        j = v3.index.union(b3.index)
                        cost_rows.append({"strategy": s_key, "freeze": fz, "variant": variant, "cost_multiplier": mult,
                                          "v_sharpe": E.sharpe(v3["ret"].reindex(j).fillna(0)),
                                          "b_sharpe": E.sharpe(b3["ret"].reindex(j).fillna(0))})
                # regimes and crises (global portfolio)
                idx = vpf.index.union(bpf.index)
                v, b = vpf["ret"].reindex(idx).fillna(0), bpf["ret"].reindex(idx).fillna(0)
                lab = labels.reindex(idx)
                for dim in ("trend", "vol"):
                    for reg, sel in lab.groupby(dim).groups.items():
                        if len(sel) < 60:
                            continue
                        regimes_rows.append({"strategy": s_key, "freeze": fz, "variant": variant, "dimension": dim,
                                             "regime": reg, "days": len(sel), "v_sharpe": E.sharpe(v[sel]),
                                             "b_sharpe": E.sharpe(b[sel])})
                for c in cfg["regimes"]["crises"]:
                    sel = (idx >= pd.Timestamp(c["start"])) & (idx <= pd.Timestamp(c["end"]))
                    if sel.sum() < 20:
                        continue
                    crisis_rows.append({"strategy": s_key, "freeze": fz, "variant": variant, "crisis": c["name"],
                                        "v_return": float(np.prod(1 + v[sel]) - 1),
                                        "b_return": float(np.prod(1 + b[sel]) - 1),
                                        "v_max_drawdown": _mdd(v[sel]), "b_max_drawdown": _mdd(b[sel])})
    for r in cost_rows:
        r["delta_sharpe"] = r["v_sharpe"] - r["b_sharpe"]
    pd.DataFrame(regimes_rows).assign(delta_sharpe=lambda x: x.v_sharpe - x.b_sharpe).to_csv(
        AN / "regimes.csv", index=False)
    pd.DataFrame(crisis_rows).to_csv(AN / "crises.csv", index=False)
    long = pd.concat([s.rename("ret").to_frame().assign(strategy=s.name[0], variant=s.name[1], freeze=s.name[2])
                      for s in daily_rows]).rename_axis("date").reset_index()
    long.drop_duplicates(["strategy", "variant", "freeze", "date"]).to_parquet(AN / "global_daily.parquet", index=False)


def _mdd(r: pd.Series) -> float:
    eq = np.cumprod(1 + r.to_numpy())
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def _data_summary() -> None:
    from .datasets import CLEAN, VALID
    man = pd.read_csv(CLEAN / "MANIFEST.csv")
    ok = man[man["status"] == "ok"]
    s = ok.groupby("market").agg(instruments=("symbol", "size"), first=("first", "min"), last=("last", "max"),
                                 median_rows=("clean_rows", "median"), rows_repaired=("ohlc_repaired", "sum"),
                                 rows_dropped=("missing_or_nonpositive", "sum")).reset_index()
    s.to_csv(AN / "data_summary.csv", index=False)
    pd.read_csv(VALID / "yahoo_vs_fred.csv").to_csv(AN / "validation_fred.csv", index=False)
