"""Core invariants of MBCA. Run: pytest -q   (~1 minute, synthetic data only)"""
import numpy as np
import pandas as pd
import pytest

import mbca
from mbca import components as C
from mbca import indicators as ind
from mbca import stats
from mbca.data import synthetic_ohlc
from mbca.features import FEATURE_COLS, compute_features

SPLIT = pd.Timestamp("2013-06-01")


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlc(900, seed=1)


@pytest.fixture(scope="module")
def frames():
    return {f"S{i}": synthetic_ohlc(1400, seed=10 + i) for i in range(3)}


# ------------------------------------------------------------------ features: no lookahead
def test_features_are_backward_only(df):
    full = compute_features(df)
    for t in (80, 300, 650, len(df) - 1):
        part = compute_features(df.iloc[: t + 1])
        np.testing.assert_allclose(part.iloc[-1].to_numpy(), full.iloc[t].to_numpy(), rtol=1e-9, atol=1e-12)


def test_vectorized_indicators_match_loop_reference(df):
    close, w = df["close"], 20
    ref_r2 = [np.nan] * w + [np.corrcoef(np.arange(w), close.to_numpy()[i - w:i])[0, 1] ** 2
                             for i in range(w, len(df))]
    np.testing.assert_allclose(ind.rolling_r2(close, w), ref_r2, rtol=1e-9)
    x = close.pct_change()
    ref_z = x.rolling(w, min_periods=10).apply(lambda v: (v[-1] - v.mean()) / (v.std() + 1e-8), raw=True)
    np.testing.assert_allclose(ind.rolling_zscore(x, w)[w:], ref_z[w:], rtol=1e-9)


def test_feature_set_is_the_papers_eight(df):
    assert list(compute_features(df).columns) == FEATURE_COLS and len(FEATURE_COLS) == 8


# ------------------------------------------------------------------ exit engines
def _frame(closes):
    c = np.asarray(closes, float)
    return pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=len(c)),
                         "open": c, "high": c + 0.5, "low": c - 0.5, "close": c})


def test_bracket_exit_hits_target_then_next_trade():
    df = _frame([100.0] * 15 + [100, 101, 102, 104, 100, 99, 97, 94])  # ATR(14) = 1.0 on the flat run
    sig = pd.Series(0, index=df.index)
    sig[15] = 1   # long @100: stop 98.5, target 103 -> hit at bar 18 (close 104), filled at 103
    sig[17] = -1  # ignored: a position is already open
    sig[19] = -1  # short @100 once flat again
    t = mbca.bracket_exit(df, sig, sl_mult=1.5, tp_mult=3.0, fee_bps=0)
    assert t["entry_i"].tolist() == [15, 19] and t["exit_i"].iloc[0] == 18
    assert t["reason"].iloc[0] == "target" and t["ret"].iloc[0] == pytest.approx(0.03)


def test_trailing_stop_ratchets_and_locks_gain():
    df = _frame([100.0] * 15 + [100, 102, 104, 106, 108, 104, 100])
    sig = pd.Series(0, index=df.index)
    sig[15] = 1
    t = mbca.trailing_exit(df, sig, sl_mult=1.5, trail_mult=2.0, fee_bps=0)
    assert t["reason"].iloc[0] == "stop" and t["ret"].iloc[0] > 0  # stopped out in profit after the peak


def test_one_position_at_a_time(df):
    t = mbca.bracket_exit(df, mbca.jma_signals(df))
    assert len(t) > 5 and (t["entry_i"].to_numpy()[1:] > t["exit_i"].to_numpy()[:-1]).all()


# ------------------------------------------------------------------ freeze discipline
def _scramble_after(frames, split):
    """Replace every bar at/after ``split`` with unrelated noise."""
    out = {}
    for k, d in frames.items():
        d = d.copy()
        post = (d["Date"] >= split).to_numpy()
        noise = synthetic_ohlc(len(d), seed=999)
        cols = ["open", "high", "low", "close"]
        d.loc[post, cols] = noise.loc[post, cols].to_numpy()
        out[k] = d
    return out


@pytest.mark.parametrize("make", [lambda: C.RegimeFilter("excursion"), C.MetaLabel, C.Entry, C.Exit])
def test_models_never_see_post_split_data(frames, make):
    """Scrambling everything after the split must not change the fitted model at all."""
    a, b = make().fit(frames, SPLIT), make().fit(_scramble_after(frames, SPLIT), SPLIT)
    probe = compute_features(frames["S0"])[FEATURE_COLS].iloc[100:600]
    fitted = [m for m in ("model", "long", "short") if hasattr(a, m)]
    assert fitted
    for m in fitted:
        np.testing.assert_array_equal(getattr(a, m).predict_proba(probe), getattr(b, m).predict_proba(probe))


def test_labels_straddling_split_are_excluded(frames):
    d = C._prep(frames["S0"])
    ok = C._eligible(d, compute_features(d), SPLIT, horizon=40)
    last = ok[ok].index.max()
    assert d.loc[last + 40, "Date"] < SPLIT


# ------------------------------------------------------------------ components compose
def test_components_replace_exactly_one_insertion_point(frames):
    base = mbca.Pipeline()
    meta, entry, exit_ = C.MetaLabel().fit(frames, SPLIT), C.Entry().fit(frames, SPLIT), C.Exit().fit(frames, SPLIT)
    sizing = C.PositionSizing(meta).fit(frames, SPLIT)
    assert sizing.meta is meta  # shares the frozen meta-label model
    assert meta.apply(base).signal_fn is base.signal_fn and meta.apply(base).gate_fn is not None
    assert entry.apply(base).exit_fn is base.exit_fn and entry.apply(base).signal_grid == {}
    assert exit_.apply(base).signal_fn is base.signal_fn and exit_.apply(base).gate_fn is None
    d = frames["S1"]
    unsized, sized = base.trades(d, {}, {}), sizing.apply(base).trades(d, {}, {})
    assert unsized["entry_i"].tolist() == sized["entry_i"].tolist()  # sizing never adds/removes a trade
    m = sizing.multipliers(d, unsized)
    assert ((m >= 0) & (m <= 2)).all()


# ------------------------------------------------------------------ stats
def test_bootstrap_ci_identical_series():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 400), index=pd.bdate_range("2020-01-01", periods=400))
    lo, hi = stats.bootstrap_ci(r, r)
    assert lo < 0 < hi  # unpaired: independent resamples of the same series straddle 0
    assert stats.bootstrap_ci(r, r, paired=True) == (0.0, 0.0)  # paired: zero variance


def test_bootstrap_ci_detects_a_large_difference():
    idx = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(3)
    good, bad = pd.Series(rng.normal(0.004, 0.01, 600), idx), pd.Series(rng.normal(-0.002, 0.01, 600), idx)
    assert stats.bootstrap_ci(good, bad)[0] > 0


def test_deflated_sharpe_penalizes_more_trials():
    r = np.random.default_rng(1).normal(0.0005, 0.01, 500)
    assert stats.deflated_sharpe(r, 1) > stats.deflated_sharpe(r, 10) > stats.deflated_sharpe(r, 100)


def test_recommendation_rule():
    cell = lambda g=False, l=False, d=0.0: {"sig_gain": g, "sig_loss": l, "mean_delta": d}
    assert stats.recommendation({"a": cell(g=True, d=1)}).startswith("Adopt")
    assert stats.recommendation({"a": cell(l=True, d=-1)}).startswith("Avoid")
    assert stats.recommendation({"a": cell(l=True, d=-1), "b": cell(d=0.5)}).startswith("Mixed")
    assert stats.recommendation({"a": cell(d=0.5)}).startswith("Worth exploring")
    assert stats.recommendation({"a": cell(d=-0.5)}).startswith("Neutral")


# ------------------------------------------------------------------ end to end
def test_run_mbca_end_to_end(frames):
    meta = C.MetaLabel()
    sizing = C.PositionSizing(meta)
    res = mbca.run_mbca(frames, [meta, sizing, C.Combo(C.Exit(), sizing, name="Exit+Sizing")],
                        split_date=SPLIT, forward_start="2014-06-01", verbose=False)
    t = res.table
    assert set(t["window"]) == {"holdout", "forward"} and len(t) == 2 * 4
    ml = t[t["variant"] != mbca.BASELINE]
    assert ml["ci_lo"].notna().all() and (ml["ci_lo"] <= ml["ci_hi"]).all()
    for _, g in t.groupby("window"):  # sizing keeps every baseline trade; meta-labeling only removes
        n = g.set_index("variant")["n_trades"]
        assert n["ML Position Sizing"] == n[mbca.BASELINE] >= n["ML Meta-Labeling"]
    assert len(res.matrix()) == 3


def test_custom_strategy_plugs_in(frames):
    def ema_cross(df, fast=5, slow=20):
        return mbca.crossover_signal(df["close"].ewm(span=fast).mean(), df["close"].ewm(span=slow).mean())

    pipe = mbca.Pipeline(signal_fn=ema_cross, signal_grid={"fast": [5], "slow": [20]},
                         exit_fn=mbca.trailing_exit, stop_grid={"sl_mult": [1.5], "trail_mult": [2.0]})
    res = mbca.run_mbca(frames, [C.MetaLabel(), C.RuleGate(mbca.adx_gate)], split_date=SPLIT, base=pipe, min_trades=3, verbose=False)
    assert (res.table["n_trades"] > 0).all()


def test_warns_when_strategy_never_trades(frames):
    never = mbca.Pipeline(signal_fn=lambda df: pd.Series(0, index=df.index), signal_grid={})
    with pytest.warns(UserWarning, match="no out-of-sample trades"):
        mbca.run_mbca(frames, [], split_date=SPLIT, base=never, verbose=False)
