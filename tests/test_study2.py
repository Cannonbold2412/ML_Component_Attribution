"""Study 2 invariants: strategy look-ahead, execution realism, MTM accounting, inference."""
import numpy as np
import pandas as pd
import pytest

from mbca import evaluation as E
from mbca import strategies as S
from mbca.config import load_config
from mbca.data import synthetic_ohlc
from mbca.strategy import bracket_exit, trailing_exit


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlc(900, seed=4)


@pytest.mark.parametrize("name", list(S.SIGNALS))
def test_strategy_signals_have_no_lookahead(df, name):
    """Bias test B2: every signal at bar t is unchanged when the future is deleted."""
    fn = S.SIGNALS[name]
    full = np.asarray(fn(df))
    for t in (260, 500, 899):
        assert np.asarray(fn(df.iloc[: t + 1]))[-1] == full[t]


@pytest.mark.parametrize("name", list(S.SIGNALS))
def test_strategies_trade(df, name):
    assert (np.asarray(S.SIGNALS[name](df)) != 0).sum() > 3


def test_every_configured_strategy_builds():
    cfg = load_config()
    for k, s in cfg["strategies"].items():
        p = S.build_pipeline(s, cost_bps=5)
        assert p.exit_kwargs == {"fill": "close"}


def _frame(closes):
    c = np.asarray(closes, float)
    return pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=len(c)),
                         "open": c, "high": c + 0.5, "low": c - 0.5, "close": c})


def test_close_fill_is_never_better_than_stop_fill():
    """Bias test B6: a gap through the stop fills at the (worse) close, not at the stop."""
    df = _frame([100.0] * 15 + [100, 101, 95, 96])  # gaps from 101 to 95 through the 98.5 stop
    sig = pd.Series(0, index=df.index)
    sig[15] = 1
    at_stop = bracket_exit(df, sig, fill="stop", fee_bps=0)["ret"].iloc[0]
    at_close = bracket_exit(df, sig, fill="close", fee_bps=0)["ret"].iloc[0]
    assert at_close == pytest.approx(-0.05) and at_stop == pytest.approx(-0.015) and at_close < at_stop
    t = trailing_exit(df, sig, fill="close", fee_bps=0)
    assert t["exit_price"].iloc[0] == 95


def test_mtm_accounting_matches_trade_return():
    """The MTM daily returns of one trade compound to its fill-to-fill return, minus costs."""
    df = _frame([100.0] * 15 + [100, 102, 101, 104, 99, 98])
    sig = pd.Series(0, index=df.index)
    sig[15] = 1
    t = bracket_exit(df, sig, fill="close", fee_bps=0, sl_mult=1.5, tp_mult=10)
    d = E.instrument_daily(t, df, cost_bps=0)
    assert np.prod(1 + d["ret"]) - 1 == pytest.approx(t["gross"].iloc[0])
    d10 = E.instrument_daily(t, df, cost_bps=10)
    assert d["ret"].sum() - d10["ret"].sum() == pytest.approx(10 / 1e4)
    assert d["pos"].max() == 1 and d["pos"].iloc[15] == 0  # no exposure on the entry bar itself


def test_block_bootstrap_detects_difference_and_not_noise():
    idx = pd.bdate_range("2010-01-01", periods=1500)
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0.0002, 0.01, 1500), idx)
    better = base + 0.002
    # size under the null: independent pairs with identical distributions -> ~5% rejections
    rejections = sum(
        E.block_bootstrap_sharpe_diff(pd.Series(rng.normal(0.0002, 0.01, 1500), idx),
                                      pd.Series(rng.normal(0.0002, 0.01, 1500), idx),
                                      n_boot=400, seed=k)["p_value"] < 0.05
        for k in range(60))
    assert rejections <= 9  # 5% of 60 = 3; generous binomial upper tail
    r = E.block_bootstrap_sharpe_diff(better, base)
    assert r["p_value"] < 0.01 and r["ci_lo"] > 0


def test_fdr_and_holm():
    p = pd.Series([0.001, 0.01, 0.03, 0.04, 0.5])
    q = E.bh_fdr(p)
    assert (q >= p).all() and q.iloc[0] == pytest.approx(0.005)
    assert (E.holm(p) >= q).all()


def test_regimes_use_only_past_information():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 600), pd.bdate_range("2015-01-01", periods=600))
    full = E.regime_labels(r)
    part = E.regime_labels(r.iloc[:400])
    assert (full.iloc[:400].fillna("na") == part.fillna("na")).all().all()
