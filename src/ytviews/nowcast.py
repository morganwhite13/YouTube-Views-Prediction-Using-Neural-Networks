"""Forecast a live video's final views from its early views ("nowcasting").

After a video is published its early numbers say a lot: a video that is 2x ahead of the model's
expectation after one day usually stays ahead. How much of that lead persists is *measured*,
not assumed: for held-out videos fetched at two different ages (``ytviews refresh`` builds
these snapshot pairs) we regress the later surprise on the earlier one,

    surprise_late = a + b * surprise_early + c * surprise_early * log1p(age_early)

where ``surprise = log1p(actual views) - log1p(predicted views)``. The ``c`` term lets very early
surprises (an hour in) count for less than surprises after a week. The spread of the residuals
gives the forecast range.

Without enough snapshot pairs the fallback assumes the surprise persists unchanged (``b = 1``)
and keeps the model's own range; ``calibrated`` is then False and callers should say so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .features import age_days, log_views

if TYPE_CHECKING:  # pragma: no cover
    from .features import FeatureBuilder
    from .predictor import Predictor, VideoInput
    from .train import NeuralRegressor

MIN_PAIRS = 20
FALLBACK: dict[str, Any] = {
    "a": 0.0, "b": 1.0, "c": 0.0, "resid_q": None, "n_pairs": 0, "calibrated": False,
}  # fmt: skip


def snapshot_pairs(df: pd.DataFrame, min_ratio: float = 1.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(earliest, latest) snapshot of each video whose later age is >= ``min_ratio`` x earlier."""
    df = df.assign(_age=age_days(df)).sort_values(["video_id", "fetched_at"])
    first = df.groupby("video_id").head(1).set_index("video_id")
    last = df.groupby("video_id").tail(1).set_index("video_id")
    ok = last.index[last["_age"] >= min_ratio * first.loc[last.index, "_age"]]
    return (
        first.loc[ok].reset_index().drop(columns="_age"),
        last.loc[ok].reset_index().drop(columns="_age"),
    )


def fit_nowcast(features: FeatureBuilder, reg: NeuralRegressor, df: pd.DataFrame) -> dict:
    """Calibrate the persistence of early surprises on (ideally held-out) snapshot pairs."""
    early, late = snapshot_pairs(df)
    if len(early) < MIN_PAIRS:
        return dict(FALLBACK, n_pairs=len(early))
    e1 = log_views(early["views"].to_numpy()) - reg.predict(features.transform(early))
    e2 = log_views(late["views"].to_numpy()) - reg.predict(features.transform(late))
    la = np.log1p(age_days(early).to_numpy())
    x = np.column_stack([np.ones_like(e1), e1, e1 * la])
    if len(e1) < 2.5 * MIN_PAIRS:  # too few pairs for the age interaction
        x = x[:, :2]
    coef, *_ = np.linalg.lstsq(x, e2, rcond=None)
    resid = e2 - x @ coef
    coef = list(coef) + [0.0] * (3 - len(coef))
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": float(coef[2]),
        "resid_q": [float(np.quantile(resid, 0.1)), float(np.quantile(resid, 0.9))],
        "n_pairs": int(len(e1)),
        "calibrated": True,
    }


@dataclass
class Nowcast:
    observed_views: float
    observed_age_days: float
    expected_views: float  # what the model expected by now, before seeing any views
    calibrated: bool
    forecast: pd.DataFrame  # days, low, median, high

    @property
    def vs_expected(self) -> float:
        return (self.observed_views + 1) / (self.expected_views + 1)


def track(
    predictor: Predictor,
    video: VideoInput,
    observed_views: float,
    observed_age_days: float,
    days: tuple[float, ...] = (1, 3, 7, 14, 30, 90, 180, 365),
) -> Nowcast:
    if observed_age_days <= 0:
        raise ValueError("observed_age_days must be positive")
    nc = predictor.meta.get("nowcast") or FALLBACK
    reg, feats = predictor.regressor, predictor.features
    now = replace(video, age_days=float(observed_age_days))
    m1 = float(reg.predict(feats.transform(pd.DataFrame([now.to_row()])))[0])
    e1 = float(log_views(observed_views)) - m1
    shift = nc["a"] + e1 * (nc["b"] + nc["c"] * np.log1p(observed_age_days))

    future = tuple(d for d in days if d > observed_age_days) or (observed_age_days * 2,)
    base = predictor.trajectory(video, future)
    median = np.log1p(base["median"].to_numpy()) + shift
    if nc.get("resid_q"):
        low, high = median + nc["resid_q"][0], median + nc["resid_q"][1]
    else:  # uncalibrated: keep the model's own (wider) range around the shifted median
        low = np.log1p(base["low"].to_numpy()) + shift
        high = np.log1p(base["high"].to_numpy()) + shift
    floor = float(observed_views)  # cumulative views never go down
    out = pd.DataFrame(
        {
            "days": list(future),
            "low": np.maximum.accumulate(np.maximum(np.expm1(low), floor)),
            "median": np.maximum.accumulate(np.maximum(np.expm1(median), floor)),
            "high": np.maximum.accumulate(np.maximum(np.expm1(high), floor)),
        }
    )
    return Nowcast(
        observed_views=float(observed_views),
        observed_age_days=float(observed_age_days),
        expected_views=float(np.expm1(m1)),
        calibrated=bool(nc.get("calibrated")),
        forecast=out,
    )
