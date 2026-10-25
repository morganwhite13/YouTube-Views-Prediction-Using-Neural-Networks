"""How a saved model holds up on videos published after it was trained.

This is the only fully out-of-sample check there is: neither the model nor any choice made while
building it (split, tuning, blend weight) could have seen these videos. Keep collecting
(``ytviews refresh`` or the scheduled workflow) and run ``ytviews monitor`` now and then; when
the live error drifts well above the error measured at training time, retrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import OURS_NAME, regression_metrics
from .features import add_channel_history, log_views
from .predictor import Predictor

DEGRADED_RATIO = 1.25  # live MAE this much above the training-time test MAE -> retrain


@dataclass
class MonitorResult:
    since: pd.Timestamp
    n_videos: int
    live: dict[str, float] | None
    reference: dict[str, float] | None
    weekly: pd.DataFrame  # week, n, mae_log
    in_sample: bool = False  # some scored videos were in the model's training data

    @property
    def verdict(self) -> str:
        if not self.live:
            return "no new videos yet"
        if not self.reference:
            return "no training-time reference to compare with"
        if self.in_sample:
            return "not a fair test: the model was trained on some of these videos"
        ratio = self.live["mae_log"] / self.reference["mae_log"]
        if self.n_videos < 30:
            return f"too few new videos ({self.n_videos}) to judge; keep collecting"
        if ratio > DEGRADED_RATIO:
            return f"degraded: live error is {ratio:.2f}x the training-time error; retrain"
        return f"healthy: live error is {ratio:.2f}x the training-time error"


def monitor(
    predictor: Predictor, df: pd.DataFrame, since: pd.Timestamp | str | None = None
) -> MonitorResult:
    """Score the model on videos published after ``since`` (default: when it was trained)."""
    if predictor.target != "views":
        raise ValueError(f"this needs a views model; this model predicts {predictor.target}")
    since = pd.Timestamp(since or predictor.meta["created_at"])
    if since.tzinfo is None:
        since = since.tz_localize("UTC")
    df = add_channel_history(df)  # history may use any earlier video, old or new
    latest = df.sort_values("fetched_at").drop_duplicates("video_id", keep="last")
    new = latest[latest["published_at"] > since].reset_index(drop=True)
    reference = predictor.meta.get("metrics", {}).get(OURS_NAME)
    if new.empty:
        return MonitorResult(
            since, 0, None, reference, pd.DataFrame(columns=["week", "n", "mae_log"])
        )
    q = np.log1p(predictor.predict_frame(new))
    mid = predictor.regressor.quantiles.index(0.5)
    y = log_views(new["views"].to_numpy())
    live = regression_metrics(y, q[:, mid], (q[:, 0], q[:, -1]))
    err = pd.DataFrame(
        {"week": new["published_at"].dt.tz_convert(None).dt.to_period("W").dt.start_time.dt.date,
         "abs_err": np.abs(q[:, mid] - y)}
    )  # fmt: skip
    weekly = (
        err.groupby("week")["abs_err"]
        .agg(n="size", mae_log="mean")
        .reset_index()
        .sort_values("week")
    )
    trained_at = pd.Timestamp(predictor.meta["created_at"])
    if trained_at.tzinfo is None:
        trained_at = trained_at.tz_localize("UTC")
    in_sample = bool(since < trained_at and new["fetched_at"].min() <= trained_at)
    return MonitorResult(since, len(new), live, reference, weekly, in_sample)


def summary_dict(result: MonitorResult) -> dict[str, Any]:
    return {
        "since": str(result.since),
        "new_videos": result.n_videos,
        "live": result.live,
        "training_time": result.reference,
        "verdict": result.verdict,
    }
