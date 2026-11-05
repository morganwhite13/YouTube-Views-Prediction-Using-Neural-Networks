"""Reference models. A neural network is only interesting if it beats these.

v1 reported a single number (MAE 1.73 in log space) with nothing to compare it against, so it was
impossible to tell whether the network had learned anything beyond "big channels get more views".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from .features import NUMERIC_FEATURES, FeatureBuilder, numeric_frame


class Baseline:
    name = "baseline"

    def fit(self, df: pd.DataFrame, inputs: dict[str, np.ndarray], y: np.ndarray) -> Baseline:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame, inputs: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class GlobalMedian(Baseline):
    name = "global median"

    def fit(self, df, inputs, y):
        self.value_ = float(np.median(y))
        return self

    def predict(self, df, inputs):
        return np.full(len(df), self.value_)


class ChannelHeuristic(Baseline):
    """Assume a video does what the channel's recent videos did (adjusted for age).

    Falls back to a log-linear fit on subscribers + age when no history is available.
    """

    name = "channel heuristic"
    _cols = ["log_subscribers", "log_age_days", "subscribers_missing"]

    def _x(self, df):
        return numeric_frame(df)[self._cols].fillna(0.0).to_numpy()

    def fit(self, df, inputs, y):
        num = numeric_frame(df)
        self.fallback_ = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(self._x(df), y)
        has = num["hist_median"].notna().to_numpy()
        if has.any():
            # Correct for the difference between this video's age and its history's ages.
            age_gap = (num["log_age_days"] - num["hist_log_age"]).fillna(0.0).to_numpy()[has]
            resid = y[has] - num["hist_median"].to_numpy()[has]
            self.age_slope_ = float(np.polyfit(age_gap, resid, 1)[0]) if has.sum() > 2 else 0.0
            self.bias_ = float(np.median(resid - self.age_slope_ * age_gap))
        else:
            self.age_slope_, self.bias_ = 0.0, 0.0
        return self

    def predict(self, df, inputs):
        num = numeric_frame(df)
        pred = self.fallback_.predict(self._x(df))
        hist = num["hist_median"].to_numpy()
        has = ~np.isnan(hist)
        age_gap = (num["log_age_days"] - num["hist_log_age"]).fillna(0.0).to_numpy()
        pred[has] = hist[has] + self.bias_ + self.age_slope_ * age_gap[has]
        return pred


class RidgeBaseline(Baseline):
    name = "ridge (all features)"

    def __init__(self, features: FeatureBuilder) -> None:
        self.features = features

    def fit(self, df, inputs, y):
        self.model_ = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(self.features.flat(inputs), y)
        return self

    def predict(self, df, inputs):
        return self.model_.predict(self.features.flat(inputs))


class LightGBMBaseline(Baseline):
    name = "lightgbm (all features)"

    def __init__(self, features: FeatureBuilder, seed: int = 0) -> None:
        self.features = features
        self.seed = seed

    def fit(self, df, inputs, y, sample_weight=None):
        import lightgbm as lgb

        x = self.features.flat(inputs, one_hot=False)
        # Cumulative views can only grow with age: constrain the trees to agree, so blending
        # with the network (whose growth curve guarantees it) keeps that property.
        monotone = [0] * x.shape[1]
        monotone[NUMERIC_FEATURES.index("log_age_days")] = 1  # flat() starts with numerics
        self.model_ = lgb.LGBMRegressor(
            monotone_constraints=monotone,
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            # Robust like L1 (which LightGBM can't combine with monotone constraints); a
            # threshold of 0.5 in log views beat L1 on the synthetic data (0.698 vs 0.703).
            objective="huber",
            alpha=0.5,
            random_state=self.seed,
            verbose=-1,
        )
        self.model_.fit(x, y, sample_weight=sample_weight, categorical_feature=[x.shape[1] - 1])
        return self

    def predict(self, df, inputs):
        return self.model_.predict(self.features.flat(inputs, one_hot=False))


def default_baselines(features: FeatureBuilder) -> list[Baseline]:
    models: list[Baseline] = [GlobalMedian(), ChannelHeuristic(), RidgeBaseline(features)]
    try:
        import lightgbm  # noqa: F401

        models.append(LightGBMBaseline(features))
    except ImportError:
        pass
    return models
