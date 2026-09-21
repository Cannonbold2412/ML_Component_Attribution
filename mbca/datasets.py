"""Study 2 data layer: download -> raw (immutable) -> clean -> validate.

    data/study2/raw/yahoo/<SYMBOL>.json.gz     exact API response bytes (gzip)
    data/study2/raw/fred/<SERIES>.csv.gz       exact CSV bytes (gzip)
    data/study2/raw/MANIFEST.csv               url, fetched_at_utc, sha256, bytes per file
    data/study2/clean/<market>/<SYMBOL>.csv.gz cleaned daily OHLC (the ONLY input to experiments)
    data/study2/clean/MANIFEST.csv             rows, dates, rows dropped per rule, sha256
    data/study2/validation/*.csv               cross-source checks (Yahoo vs FRED)

Raw files are never modified; cleaning is a pure function of raw bytes + the
[cleaning] rules in study.toml, so the clean tree can be rebuilt bit-for-bit.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config

DATA = ROOT / "data" / "study2"
RAW, CLEAN, VALID = DATA / "raw", DATA / "clean", DATA / "validation"
YAHOO_URL = ("https://query2.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}"
             "&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
UA = {"User-Agent": "Mozilla/5.0 (research data collection; mbca)"}


def safe_name(symbol: str) -> str:
    """Filesystem-safe, reversible-enough file stem for a Yahoo symbol."""
    return symbol.replace("^", "IDX_").replace("=", "_EQ_").replace("/", "_")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _append_manifest(rows: list[dict]) -> None:
    path = RAW / "MANIFEST.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["file"].isin(new["file"])], new], ignore_index=True)
    new.sort_values("file").to_csv(path, index=False)


def _get(url: str, tries: int = 4) -> bytes:
    import requests

    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                raise FileNotFoundError(f"404 {url}")
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def fetch(force: bool = False) -> pd.DataFrame:
    """Download every universe symbol (Yahoo) and validation series (FRED) not already on disk."""
    cfg = load_config()
    p1 = int(pd.Timestamp(cfg["study"]["data_start"]).timestamp())
    p2 = int(time.time())
    jobs = [("yahoo", s, YAHOO_URL.format(sym=urllib.parse.quote(s, safe=""), p1=p1, p2=p2))
            for m in cfg["universe"].values() for s in m["tickers"]]
    jobs += [("fred", sid, FRED_URL.format(sid=sid)) for sid in cfg["validation"]["fred"].values()]
    rows, failed = [], []
    for source, sym, url in jobs:
        ext = ".json.gz" if source == "yahoo" else ".csv.gz"
        path = RAW / source / f"{safe_name(sym)}{ext}"
        if path.exists() and not force:
            continue
        try:
            body = _get(url)
        except Exception as e:  # noqa: BLE001 -- record and continue; a missing symbol is data, not a crash
            failed.append({"source": source, "symbol": sym, "error": str(e)[:200]})
            print(f"  FAILED {source}:{sym}: {e}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(body, mtime=0))
        rows.append({"file": str(path.relative_to(DATA)).replace("\\", "/"), "source": source, "symbol": sym,
                     "url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "sha256_raw_bytes": sha256(body), "bytes": len(body)})
        print(f"  ok {source}:{sym} ({len(body):,} bytes)")
        time.sleep(0.4)
    if rows:
        _append_manifest(rows)
    if failed:
        pd.DataFrame(failed).to_csv(RAW / "FAILED.csv", index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ parsing + cleaning
def parse_yahoo(raw: bytes) -> pd.DataFrame:
    """Yahoo chart JSON -> DataFrame(Date, open, high, low, close, adjclose, volume)."""
    res = json.loads(raw)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    dates = ts.tz_convert(tz).tz_localize(None).normalize()
    cols = {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
            "adjclose": adj, "volume": q.get("volume", [None] * len(dates))}
    df = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce") for k, v in cols.items()})
    return df.assign(Date=dates.to_numpy())[["Date", *cols]]


def clean_frame(raw: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the study.toml [cleaning] rules. Returns (clean df, counts of rows dropped per rule)."""
    df = raw.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    log = {"raw_rows": len(raw), "duplicate_dates": len(raw) - len(df)}
    px = ["open", "high", "low", "close"]
    bad = df["close"].isna() | (rules["drop_nonpositive"] and (df[px] <= 0).any(axis=1))
    log["missing_or_nonpositive"] = int(bad.sum())
    df = df[~bad].copy()
    for c in ("open", "high", "low"):  # missing intraday fields -> close (flat bar), never invented range
        df[c] = df[c].fillna(df["close"])
    if rules["use_adjusted"]:
        f = (df["adjclose"] / df["close"]).where(df["adjclose"].notna() & (df["adjclose"] > 0), 1.0)
        df[px] = df[px].mul(f, axis=0)
    if rules["repair_ohlc"]:
        lo, hi = df[["open", "close"]].min(axis=1), df[["open", "close"]].max(axis=1)
        log["ohlc_repaired"] = int(((df["low"] > lo) | (df["high"] < hi)).sum())
        df["low"], df["high"] = np.minimum(df["low"], lo), np.maximum(df["high"], hi)
    # bad ticks: drop rows whose close-to-close move is implausible AND reverses next day
    r = df["close"].pct_change()
    spike = (r.abs() > rules["max_abs_daily_return"]) & (r.shift(-1).abs() > rules["max_abs_daily_return"] / 2)
    log["bad_ticks"] = int(spike.sum())
    df = df[~spike]
    log["clean_rows"] = len(df)
    out = df[["Date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return out, log


def clean() -> pd.DataFrame:
    cfg = load_config()
    rules, rows = cfg["cleaning"], []
    for market, m in cfg["universe"].items():
        for sym in m["tickers"]:
            src = RAW / "yahoo" / f"{safe_name(sym)}.json.gz"
            row = {"market": market, "symbol": sym}
            if not src.exists():
                rows.append({**row, "status": "no_raw_file"})
                continue
            try:
                df, log = clean_frame(parse_yahoo(gzip.decompress(src.read_bytes())), rules)
            except (KeyError, TypeError, IndexError, ValueError) as e:
                rows.append({**row, "status": f"parse_error: {e}"[:120]})
                continue
            row.update(log)
            if len(df) < rules["min_rows"]:
                rows.append({**row, "status": "excluded_min_rows"})
                continue
            dst = CLEAN / market / f"{safe_name(sym)}.csv.gz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            body = df.to_csv(index=False, float_format="%.8g", date_format="%Y-%m-%d").encode()
            dst.write_bytes(gzip.compress(body, mtime=0))
            rows.append({**row, "status": "ok", "first": df["Date"].min().date(), "last": df["Date"].max().date(),
                         "file": str(dst.relative_to(DATA)).replace("\\", "/"), "sha256_csv": sha256(body)})
    man = pd.DataFrame(rows)
    CLEAN.mkdir(parents=True, exist_ok=True)
    man.to_csv(CLEAN / "MANIFEST.csv", index=False)
    return man


def load_study2(markets: list[str] | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    """{market: {symbol: clean OHLC}} for every 'ok' row of the clean manifest."""
    man = pd.read_csv(CLEAN / "MANIFEST.csv")
    man = man[man["status"] == "ok"]
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for r in man.itertuples():
        if markets and r.market not in markets:
            continue
        df = pd.read_csv(DATA / r.file, parse_dates=["Date"])
        out.setdefault(r.market, {})[r.symbol] = df
    return out


# ------------------------------------------------------------------ cross-source validation
def validate() -> pd.DataFrame:
    """Yahoo vs FRED: overlap, correlation of daily returns, median |price diff| (bias test B1).
    FRED FX series are noon New York fixes and some are quoted inverted (e.g. DEXUSEU = USD per EUR);
    the orientation is chosen by which of (x, 1/x) matches Yahoo better, and both are reported."""
    cfg = load_config()
    rows = []
    frames = {s: d for m in load_study2().values() for s, d in m.items()}
    for sym, sid in cfg["validation"]["fred"].items():
        src = RAW / "fred" / f"{safe_name(sid)}.csv.gz"
        if sym not in frames or not src.exists():
            rows.append({"symbol": sym, "fred": sid, "status": "missing"})
            continue
        f = pd.read_csv(io.BytesIO(gzip.decompress(src.read_bytes())))
        f.columns = ["Date", "v"]
        f["Date"] = pd.to_datetime(f["Date"])
        f["v"] = pd.to_numeric(f["v"], errors="coerce")
        f = f.dropna().set_index("Date")["v"]
        f = f[f > 0]
        y = frames[sym].set_index("Date")["close"]
        best = None
        for orient, fv in (("as_is", f), ("inverted", 1 / f)):
            j = pd.concat([y, fv], axis=1, join="inner").dropna()
            if len(j) < 250:
                continue
            ry, rf = j.iloc[:, 0].pct_change(), j.iloc[:, 1].pct_change()
            r5 = j.iloc[::5].pct_change()  # 5-day returns: insensitive to intraday fixing-time differences
            cand = {"orientation": orient, "overlap_days": len(j), "first": j.index.min().date(),
                    "last": j.index.max().date(), "return_corr": float(ry.corr(rf)),
                    "return_corr_5d": float(r5.iloc[:, 0].corr(r5.iloc[:, 1])),
                    "median_abs_pct_diff": float((j.iloc[:, 0] / j.iloc[:, 1] - 1).abs().median())}
            if best is None or cand["median_abs_pct_diff"] < best["median_abs_pct_diff"]:
                best = cand
        rows.append({"symbol": sym, "fred": sid, "status": "ok" if best else "too_little_overlap", **(best or {})})
    out = pd.DataFrame(rows)
    VALID.mkdir(parents=True, exist_ok=True)
    out.to_csv(VALID / "yahoo_vs_fred.csv", index=False)
    return out
