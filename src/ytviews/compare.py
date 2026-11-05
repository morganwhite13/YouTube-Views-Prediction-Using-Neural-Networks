"""Compare two models on the same videos, fairly, and decide whether a retrain is better.

Two uses:

* ``ytviews compare-models A B`` scores two saved models on videos published after both were
  trained, so neither has seen them.
* ``ytviews update`` trains a candidate when the current model drifts, and promotes it only if
  it is not worse. That check uses the candidate's held-out test videos, restricted to the ones
  the current model never saw either.

Either way the verdict comes from a paired bootstrap over videos: both models predict the same
videos, so the difference in their errors is far less noisy than either error alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_channel_history, log_views
from .predictor import Predictor

MIN_VIDEOS = 30  # fewer than this and the comparison says too little to act on


@dataclass
class Comparison:
    name_a: str
    name_b: str
    n: int
    mae_a: float  # mean absolute error of log views
    mae_b: float
    ci_low: float  # 95% interval of mae_b - mae_a (negative = B is better)
    ci_high: float
    note: str = ""  # e.g. why the comparison isn't a fair one

    @property
    def diff(self) -> float:
        return self.mae_b - self.mae_a

    @property
    def verdict(self) -> str:
        if self.n < MIN_VIDEOS:
            return f"too few shared videos ({self.n}) to tell them apart"
        if self.ci_high < 0:
            return f"{self.name_b} is better"
        if self.ci_low > 0:
            return f"{self.name_a} is better"
        return "no significant difference"

    def as_dict(self) -> dict:
        return {
            "a": self.name_a, "b": self.name_b, "videos": self.n, "mae_log_a": self.mae_a,
            "mae_log_b": self.mae_b, "difference": self.diff,
            "ci95": [self.ci_low, self.ci_high], "verdict": self.verdict, "note": self.note,
        }  # fmt: skip


def paired_comparison(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
    n_boot: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Compare log-scale predictions of two models on the same targets ``y`` (log views)."""
    err_a, err_b = np.abs(np.asarray(pred_a) - y), np.abs(np.asarray(pred_b) - y)
    n = len(y)
    if n == 0:
        return Comparison(name_a, name_b, 0, np.nan, np.nan, np.nan, np.nan)
    d = err_b - err_a
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return Comparison(
        name_a, name_b, n, float(err_a.mean()), float(err_b.mean()), float(low), float(high)
    )


def _created_at(p: Predictor) -> pd.Timestamp:
    t = pd.Timestamp(p.meta["created_at"])
    return t if t.tzinfo else t.tz_localize("UTC")


def _unseen(df: pd.DataFrame, after: pd.Timestamp) -> pd.Series:
    """Rows of videos published, and first fetched, after ``after``: no model trained by then
    can have seen any snapshot of them."""
    first_seen = df.groupby("video_id")["fetched_at"].transform("min")
    return (df["published_at"] > after) & (first_seen > after)


def compare_models(
    a: Predictor,
    b: Predictor,
    df: pd.DataFrame,
    since: pd.Timestamp | str | None = None,
    names: tuple[str, str] = ("A", "B"),
) -> Comparison:
    """Score both models on the latest snapshot of videos neither could have trained on."""
    for p in (a, b):
        if p.target != "views":
            raise ValueError(f"this needs views models; one predicts {p.target}")
    both_trained = max(_created_at(a), _created_at(b))
    df = add_channel_history(df)
    keep = _unseen(df, both_trained)
    note = ""
    if since is not None:
        since = pd.Timestamp(since)
        since = since if since.tzinfo else since.tz_localize("UTC")
        keep = df["published_at"] > since
        if since < both_trained:
            note = "includes videos one of the models may have trained on"
    latest = df[keep].sort_values("fetched_at").drop_duplicates("video_id", keep="last")
    latest = latest.reset_index(drop=True)
    if latest.empty:
        return Comparison(*names, 0, np.nan, np.nan, np.nan, np.nan, note)
    mid_a, mid_b = a.regressor.quantiles.index(0.5), b.regressor.quantiles.index(0.5)
    pred_a = np.log1p(a.predict_frame(latest))[:, mid_a]
    pred_b = np.log1p(b.predict_frame(latest))[:, mid_b]
    out = paired_comparison(log_views(latest["views"].to_numpy()), pred_a, pred_b, *names)
    out.note = note
    return out


def gate_candidate(current: Predictor, candidate_dir: str | Path, df: pd.DataFrame) -> Comparison:
    """Compare the current model with a freshly trained candidate before replacing it.

    The candidate's saved model has been refit on everything, so it is judged by its held-out
    test predictions (``test_predictions.csv``, made before the refit), on the test videos the
    current model never saw. Name order: ("current", "candidate").
    """
    names = ("current", "candidate")
    held_out = pd.read_csv(Path(candidate_dir) / "test_predictions.csv")
    if "fetched_at" not in held_out:
        return Comparison(*names, 0, np.nan, np.nan, np.nan, np.nan, "no held-out predictions")
    held_out["fetched_at"] = pd.to_datetime(held_out["fetched_at"], utc=True, format="ISO8601")
    df = add_channel_history(df)
    df = df[_unseen(df, _created_at(current))]
    rows = df.merge(
        held_out[["video_id", "fetched_at", "pred_median"]], on=["video_id", "fetched_at"]
    )
    rows = rows.sort_values("fetched_at").drop_duplicates("video_id", keep="last")
    rows = rows.reset_index(drop=True)
    if rows.empty:
        return Comparison(*names, 0, np.nan, np.nan, np.nan, np.nan)
    mid = current.regressor.quantiles.index(0.5)
    pred_current = np.log1p(current.predict_frame(rows))[:, mid]
    pred_candidate = np.log1p(rows["pred_median"].to_numpy())
    y = log_views(rows["views"].to_numpy())
    return paired_comparison(y, pred_current, pred_candidate, *names)


def should_promote(result: Comparison) -> bool:
    """Replace the current model unless the candidate is clearly worse on shared videos."""
    return not (result.n >= MIN_VIDEOS and result.ci_low > 0)
