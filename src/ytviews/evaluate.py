"""Train/test splits, metrics and permutation importance."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

from .features import FEATURE_GROUPS, NUMERIC_FEATURES

SPLITS = ("time", "channel", "random")


def split_frame(
    df: pd.DataFrame, strategy: str = "time", test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) positional indices.

    * ``time``    - train on older uploads, test on the newest ones (closest to real use).
    * ``channel`` - test channels are never seen in training (hardest; new creators).
    * ``random``  - random videos (v1's setup; optimistic because channels leak across splits).

    All strategies keep every snapshot of a video on the same side of the split.
    """
    if strategy == "time":
        published = df.groupby("video_id")["published_at"].transform("min")
        cutoff = published.quantile(1 - test_size)
        test = (published > cutoff).to_numpy()
        return np.flatnonzero(~test), np.flatnonzero(test)
    if strategy in ("channel", "random"):
        groups = df["channel_id"] if strategy == "channel" else df["video_id"]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(df, groups=groups))
        return np.asarray(tr), np.asarray(te)
    raise ValueError(f"unknown split {strategy!r}; choose from {SPLITS}")


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, interval: tuple[np.ndarray, np.ndarray] | None = None
) -> dict[str, float]:
    """Metrics on the log1p(views) scale plus a few human-readable ones."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) or 1.0
    ratio = np.exp(np.abs(err))  # e.g. 2.0 = off by a factor of two
    out = {
        "mae_log": float(np.abs(err).mean()),
        "rmse_log": float(np.sqrt((err**2).mean())),
        "r2_log": 1.0 - ss_res / ss_tot,
        "spearman": float(spearmanr(y_true, y_pred).statistic)
        if len(y_true) > 2 and np.ptp(y_pred) > 0
        else float("nan"),
        "median_factor_error": float(np.median(ratio)),
        "within_2x": float((ratio <= 2.0).mean()),
        "within_10x": float((ratio <= 10.0).mean()),
    }
    if interval is not None:
        lo, hi = interval
        out["interval_coverage"] = float(((y_true >= lo) & (y_true <= hi)).mean())
        out["interval_width_factor"] = float(np.median(np.exp(hi - lo)))
    return out


def permutation_importance(
    predict: Callable[[dict[str, np.ndarray]], np.ndarray],
    inputs: dict[str, np.ndarray],
    y: np.ndarray,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Increase in log-scale MAE when a feature group is shuffled across rows."""
    rng = np.random.default_rng(seed)
    base = float(np.abs(predict(inputs) - y).mean())
    groups: dict[str, tuple[str, list[int] | None]] = {
        "title text": ("title", None),
        "description text": ("description", None),
        "category": ("category", None),
    }
    for name, cols in FEATURE_GROUPS.items():
        groups[name] = ("numeric", [NUMERIC_FEATURES.index(c) for c in cols])

    scores: dict[str, float] = {}
    for name, (key, cols) in groups.items():
        deltas = []
        for _ in range(repeats):
            perm = rng.permutation(len(y))
            shuffled = dict(inputs)
            if cols is None:
                shuffled[key] = inputs[key][perm]
            else:
                arr = inputs[key].copy()
                arr[:, cols] = arr[perm][:, cols]
                shuffled[key] = arr
            deltas.append(float(np.abs(predict(shuffled) - y).mean()) - base)
        scores[name] = float(np.mean(deltas))
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))
