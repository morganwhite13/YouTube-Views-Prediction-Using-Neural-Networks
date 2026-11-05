"""Rolling-origin backtest: evaluate at several points in time instead of one split.

A single time split answers "how well would this have worked at one moment". The backtest
repeats it: train on everything published before cutoff k, test on the slice up to cutoff k+1,
for several cutoffs. Stable errors across folds mean the model isn't relying on a lucky split;
errors that grow over time point to drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd

from .baselines import ChannelHeuristic
from .evaluate import regression_metrics
from .features import FeatureBuilder, add_channel_history, log_views
from .pipeline import _fit_gbm, _fit_regressor, _lightgbm_available
from .train import TrainConfig


def backtest(
    df: pd.DataFrame,
    *,
    folds: int = 4,
    start: float = 0.5,
    text_encoder: str = "auto",
    image_encoder: str = "auto",
    config: TrainConfig | None = None,
    blend_weight: float = 0.5,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """One row per (fold, model) with metrics; fold k trains on the first
    ``start + k * step`` fraction of videos (by publish date) and tests on the next ``step``."""
    config = replace(config or TrainConfig(), ensemble=(config.ensemble if config else 2))
    df = add_channel_history(df).reset_index(drop=True)
    published = df.groupby("video_id")["published_at"].transform("min")
    edges = published.quantile(np.linspace(start, 1.0, folds + 1)).to_list()
    gbm_ok = _lightgbm_available()
    rows = []
    for k in range(folds):
        lo, hi = edges[k], edges[k + 1]
        train = df[published <= lo].reset_index(drop=True)
        test = df[(published > lo) & (published <= hi)].reset_index(drop=True)
        if len(test) < 10 or train["video_id"].nunique() < 30:
            log(f"fold {k + 1}: skipped (too little data)")
            continue
        log(f"fold {k + 1}/{folds}: train up to {lo.date()} ({len(train)} rows), "
            f"test to {hi.date()} ({len(test)} rows)")  # fmt: skip
        features = FeatureBuilder(text_encoder=text_encoder, image_encoder=image_encoder)
        features.fit(train)
        x_test, y_test = features.transform(test), log_views(test["views"].to_numpy())
        nn = _fit_regressor(train, features, config, None, 0.3)
        preds = {"neural net": nn.predict(x_test)}
        if gbm_ok:
            gbm = _fit_gbm(train, features, config.seed)
            preds["lightgbm"] = gbm.predict(features.flat(x_test, one_hot=False))
            preds[f"blend ({blend_weight:.0%} net)"] = (
                blend_weight * preds["neural net"] + (1 - blend_weight) * preds["lightgbm"]
            )
        heuristic = ChannelHeuristic().fit(train, None, log_views(train["views"].to_numpy()))
        preds["channel heuristic"] = heuristic.predict(test, None)
        for name, pred in preds.items():
            m = regression_metrics(y_test, pred)
            rows.append({"fold": k + 1, "test_until": str(hi.date()), "model": name,
                         "n_test": len(test)} | m)  # fmt: skip
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of the key metrics per model across folds."""
    g = results.groupby("model")
    out = pd.DataFrame(
        {
            "folds": g["fold"].nunique(),
            "mae_log_mean": g["mae_log"].mean(),
            "mae_log_std": g["mae_log"].std(),
            "within_2x_mean": g["within_2x"].mean(),
            "median_factor_error": g["median_factor_error"].mean(),
        }
    )
    return out.sort_values("mae_log_mean")
