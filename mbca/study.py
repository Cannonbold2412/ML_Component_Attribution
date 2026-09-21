"""Study 2 runner: every pre-registered variant, for every strategy x market.

    python -m mbca study run [--workers 11] [--only jma_trend:crypto]

Output (all derived from study.toml + data/study2/clean + the code at one commit):
    results/study2/trades/<strategy>__<market>.parquet   every out-of-sample trade, tagged
                                                         with variant / freeze / setting
    results/study2/runs/<run_id>/manifest.json           provenance of the run
    results/study2/runs/<run_id>/tasks.csv               per-task timing and skips

Variant naming: "<component>" for the main grid, "<component>@transfer" for models fit on
every OTHER market, "<component>@<setting>" for execution-assumption re-runs.
Freeze-independent variants (baseline, rule_adx) carry freeze="all".
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from . import components as C
from .config import ROOT, config_hash, load_config
from .datasets import CLEAN, load_study2, sha256
from .strategies import build_pipeline
from .strategy import adx_gate, walk_forward

OUT = ROOT / "results" / "study2"
SMALL_MODEL = dict(n_estimators=50, num_leaves=7, max_depth=3)


def make_components(keys: list[str]) -> dict:
    """study.toml component keys -> fresh component objects (one shared meta-label model)."""
    meta = C.MetaLabel()
    sizing = C.PositionSizing(meta)
    exit_ = C.Exit()
    factory = {
        "rule_adx": lambda: C.RuleGate(adx_gate),
        "regime_linearity": lambda: C.RegimeFilter("linearity"),
        "meta_label": lambda: meta,
        "position_sizing": lambda: sizing,
        "entry": C.Entry,
        "exit": lambda: exit_,
        "best_combined": lambda: C.Combo(meta, sizing, name="best_combined"),
        "meta_label_thr045": lambda: C.MetaLabel(threshold=0.45),
        "meta_label_thr055": lambda: C.MetaLabel(threshold=0.55),
        "meta_label_small_model": lambda: C.MetaLabel(model_params=SMALL_MODEL),
        "exit_small_model": lambda: C.Exit(model_params=SMALL_MODEL),
    }
    return {k: factory[k]() for k in keys}


def _run_all(frames: dict, pipe, min_trades: int, n_splits: int) -> pd.DataFrame:
    parts = []
    for sym, df in frames.items():
        t = walk_forward(df, pipe, n_splits, min_trades)
        if len(t):
            parts.append(t.assign(symbol=sym))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run_task(strategy_key: str, market: str) -> dict:
    """One strategy x one market: baseline, rule gate, every ML variant for both freezes."""
    t0 = time.time()
    cfg = load_config()
    strat, mcfg, ex = cfg["strategies"][strategy_key], cfg["universe"][market], cfg["execution"]
    data = load_study2()
    frames = data[market]
    others = {s: d for m, fr in data.items() if m != market for s, d in fr.items()}
    base = build_pipeline(strat, mcfg["cost_bps"], fill=ex["fill"], lag=ex["lag"])
    sens = {s["name"]: (build_pipeline(strat, mcfg["cost_bps"], fill=s["fill"], lag=s["lag"]), s["variants"])
            for s in ex["sensitivity"]}
    run = lambda pipe: _run_all(frames, pipe, strat["min_trades"], ex["n_splits"])  # noqa: E731
    out, skipped = [], []

    def keep(trades, variant, freeze, lo=None, hi=None):
        if trades is None or trades.empty:
            skipped.append(f"{variant}/{freeze}: no trades")
            return
        et = pd.to_datetime(trades["entry_time"])
        m = pd.Series(True, index=trades.index)
        if lo is not None:
            m &= (et >= pd.Timestamp(lo)) & (et < pd.Timestamp(hi))
        out.append(trades[m].assign(variant=variant, freeze=freeze))

    keep(run(base), "baseline", "all")
    keep(run(C.RuleGate(adx_gate).apply(base)), "rule_adx", "all")
    for name, (pipe, variants) in sens.items():
        if "baseline" in variants:
            keep(run(pipe), f"baseline@{name}", "all")

    comp_cfg = cfg["components"]
    for fz in cfg["study"]["freezes"]:
        split, hi = pd.Timestamp(fz["split"]), fz["test_end"]
        ml_keys = [k for k in comp_cfg["main"] + comp_cfg["ablations"] if k != "rule_adx"]
        comps = make_components(ml_keys)
        for key, comp in comps.items():
            try:
                C.fit_once(comp, frames, split, base)
            except ValueError as e:  # e.g. crypto has no pre-2008 history
                skipped.append(f"{key}/{fz['name']}: {e}")
                continue
            keep(run(comp.apply(base)), key, fz["name"], split, hi)
            for name, (pipe, variants) in sens.items():
                if key in variants:
                    keep(run(comp.apply(pipe)), f"{key}@{name}", fz["name"], split, hi)
        for key, comp in make_components(comp_cfg["transfer"]).items():
            try:
                C.fit_once(comp, others, split, base)  # fit on every OTHER market only
            except ValueError as e:
                skipped.append(f"{key}@transfer/{fz['name']}: {e}")
                continue
            keep(run(comp.apply(base)), f"{key}@transfer", fz["name"], split, hi)

    trades = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    path = OUT / "trades" / f"{strategy_key}__{market}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(path, index=False)
    return {"strategy": strategy_key, "market": market, "seconds": round(time.time() - t0, 1),
            "n_trades": len(trades), "n_variants": trades["variant"].nunique() if len(trades) else 0,
            "skipped": "; ".join(skipped)}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run(workers: int = 11, only: list[str] | None = None, allow_dirty: bool = False) -> str:
    dirty = _git("status", "--porcelain", "--", "mbca", "study.toml", "data/study2")
    if dirty and not allow_dirty:
        raise SystemExit(f"refusing to run with uncommitted code/config/data (commit first):\n{dirty}")
    # one BLAS/OpenMP thread per worker: LightGBM and numpy otherwise start one thread per core in
    # every worker, and 11 workers x 12 threads thrash (observed: ~10% useful CPU). Spawned workers
    # inherit this environment. Results do not depend on it; it is recorded in the manifest.
    threads = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update(threads)
    cfg = load_config()
    tasks = [(s, m) for s in cfg["strategies"] for m in cfg["universe"]]
    if only:
        tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in only]
    size = {m: len(v["tickers"]) for m, v in cfg["universe"].items()}
    tasks.sort(key=lambda t: -size[t[1]])  # biggest first for load balance
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rdir = OUT / "runs" / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    import lightgbm, numpy, sklearn, scipy  # noqa: E401

    manifest = {
        "run_id": run_id, "started_utc": run_id, "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(dirty),
        "config_sha256_16": config_hash(), "clean_data_manifest_sha256": sha256((CLEAN / "MANIFEST.csv").read_bytes()),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {"numpy": numpy.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "scipy": scipy.__version__},
        "workers": workers, "thread_env": threads, "tasks": [f"{s}:{m}" for s, m in tasks],
    }
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_task, s, m): (s, m) for s, m in tasks}
        for f in as_completed(futs):
            s, m = futs[f]
            try:
                rows.append(f.result())
            except Exception:  # noqa: BLE001 -- a failed task is recorded, not silently dropped
                rows.append({"strategy": s, "market": m, "error": traceback.format_exc()[-800:]})
            r = rows[-1]
            print(f"[{len(rows)}/{len(tasks)}] {s} x {m}: {r.get('seconds', 'ERROR')}s {r.get('n_trades', '')} trades",
                  flush=True)
    pd.DataFrame(rows).to_csv(rdir / "tasks.csv", index=False)
    manifest["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest["n_failed"] = sum("error" in r and isinstance(r.get("error"), str) for r in rows)
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "LATEST_RUN").write_text(run_id)
    return run_id
