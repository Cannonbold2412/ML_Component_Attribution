"""The shared 8-feature, backward-only feature set and the one model every ML
component uses. Holding both fixed across components is the core MBCA constraint:
only the insertion point varies between experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

FEATURE_COLS = [
    "adx", "atr_norm_mom", "hist_vol", "ret_zscore", "backward_r2", "rsi", "macd_hist", "kama_slope",
]

# Identical for every component -- no per-component tuning (MBCA's matched budget).
LGBM_PARAMS = dict(
    n_estimators=200, num_leaves=31, max_depth=6, learning_rate=0.05,
    min_child_samples=30, class_weight="balanced", random_state=42, verbosity=-1,
)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame aligned to ``df.index``; row t uses rows <= t only."""
    close = df["close"]
    atr_v = ind.atr(df)
    return pd.DataFrame({
        "adx": ind.adx(df),
        "atr_norm_mom": (close - ind.ema(close, 20)) / atr_v.replace(0, np.nan),
        "hist_vol": ind.historical_volatility(close),
        "ret_zscore": ind.rolling_zscore(close.pct_change(), window=20),
        "backward_r2": ind.rolling_r2(close, window=20),
        "rsi": ind.rsi(close),
        "macd_hist": ind.macd_hist(close),
        "kama_slope": ind.kama(close).diff(),
    }, index=df.index)


class ProbModel:
    """StandardScaler + LightGBM binary classifier with the shared hyperparameters.
    ``predict_proba`` returns NaN for rows with any missing feature (warm-up)."""

    def __init__(self, **overrides):
        self.params = {**LGBM_PARAMS, **overrides}
        self.mu = self.sd = self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbModel":
        import lightgbm as lgb

        y = pd.Series(y).astype(int)
        if y.nunique() < 2:
            raise ValueError("training labels contain a single class -- cannot fit a classifier")
        self.mu, self.sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
        self.model = lgb.LGBMClassifier(**self.params).fit((X - self.mu) / self.sd, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() before predict_proba()")
        out = np.full(len(X), np.nan)
        valid = X.notna().all(axis=1).to_numpy()
        if valid.any():
            out[valid] = self.model.predict_proba((X.loc[valid] - self.mu) / self.sd)[:, 1]
        return out
