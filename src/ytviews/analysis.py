"""Analyses built on a trained model: channel outliers and batch predictions from a CSV."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd

from .features import add_channel_history
from .predictor import Predictor, VideoInput

_LIST_FIELDS = {"tags", "recent_video_views", "recent_video_age_days"}
_FLOAT_LIST_FIELDS = {"recent_video_views", "recent_video_age_days"}


def outliers(predictor: Predictor, df: pd.DataFrame, channel_id: str | None = None) -> pd.DataFrame:
    """Each video's actual views against what the model expected for it.

    ``df`` is a dataset as returned by ``VideoStore.load_dataset()``. ``factor`` > 1 means the
    video beat expectations. If the model was trained on these same videos the expectations are
    in-sample and somewhat flattering, so treat small factors as noise.
    """
    if predictor.target != "views":
        raise ValueError(f"this needs a views model; this model predicts {predictor.target}")
    df = add_channel_history(df)
    if channel_id:
        df = df[df["channel_id"] == channel_id]
        if df.empty:
            raise ValueError(f"no stored videos for channel {channel_id}")
    df = df.reset_index(drop=True)
    q = predictor.predict_frame(df)
    out = pd.DataFrame(
        {
            "video_id": df["video_id"],
            "channel_title": df["channel_title"],
            "title": df["title"],
            "published_at": df["published_at"].dt.date,
            "views": df["views"].astype(np.int64),
            "expected": q[:, predictor.regressor.quantiles.index(0.5)],
            "low": q[:, 0],
            "high": q[:, -1],
        }
    )
    out["factor"] = (out["views"] + 1) / (out["expected"] + 1)
    out["verdict"] = np.select(
        [out["views"] > out["high"], out["views"] < out["low"]],
        ["beat expectations", "below expectations"],
        "as expected",
    )
    return out.sort_values("factor", ascending=False).reset_index(drop=True)


# fmt: off
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "my", "your", "our",
    "is", "are", "was", "i", "you", "we", "it", "this", "that", "at", "by", "from", "how", "what",
    "why", "vs", "not", "be", "me", "new", "part", "ep", "episode", "video",
}
# fmt: on


def title_terms(table: pd.DataFrame, min_count: int = 3, top: int = 5) -> pd.DataFrame:
    """Title words linked to beating (or missing) the model's expectation on one channel.

    ``table`` is the output of :func:`outliers`. For each word used in at least ``min_count``
    titles (and not in nearly all of them), ``lift`` compares the typical ``factor`` of videos
    with the word against those without it. The model already accounts for the words it
    understands, so a strong lift points at something specific to this audience. With few
    videos these are hints, not findings: ``n`` says how many videos back each one.
    """
    if len(table) < 2 * min_count:
        return pd.DataFrame(columns=["term", "n", "lift"])
    words = table["title"].str.lower().str.findall(r"#?[a-z0-9][a-z0-9']+")
    words = words.map(lambda ws: {w for w in ws if len(w) > 2 and w not in _STOPWORDS})
    log_factor = np.log(table["factor"].to_numpy())
    counts = pd.Series([w for ws in words for w in ws]).value_counts()
    rows = []
    for term, n in counts.items():
        if n < min_count or n > len(table) - min_count:
            continue
        has = words.map(lambda ws, t=term: t in ws).to_numpy()
        lift = np.median(log_factor[has]) - np.median(log_factor[~has])
        rows.append((term, int(n), float(np.exp(lift))))
    out = pd.DataFrame(rows, columns=["term", "n", "lift"])
    if out.empty:
        return out
    out = out.sort_values("lift", ascending=False)
    best = out[out["lift"] > 1].head(top)
    worst = out[out["lift"] < 1].tail(top)
    return pd.concat([best, worst]).drop_duplicates("term").reset_index(drop=True)


def _parse_cell(name: str, value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return None
    if name in _LIST_FIELDS:
        items = [v.strip() for v in str(value).replace(";", "|").split("|") if v.strip()]
        return [float(v) for v in items] if name in _FLOAT_LIST_FIELDS else items
    if name in ("caption", "hd", "is_short"):
        return str(value).strip().lower() in ("1", "true", "yes", "y")
    if name == "published_at":
        return pd.Timestamp(value).to_pydatetime()
    if name in ("title", "description", "category_id", "thumbnail_path", "language"):
        text = str(value)
        return text[:-2] if name == "category_id" and text.endswith(".0") else text
    return float(value)


def videos_from_frame(df: pd.DataFrame) -> list[VideoInput]:
    """Build ``VideoInput``s from a table whose columns are ``VideoInput`` field names.

    List columns (``tags``, ``recent_video_views``) use ``|`` or ``;`` as separators.
    """
    if "title" not in df:
        raise ValueError("the CSV needs at least a 'title' column")
    names = {f.name for f in fields(VideoInput)}
    unknown = sorted(set(df.columns) - names)
    if unknown:
        expected = ", ".join(sorted(names))
        raise ValueError(f"unknown columns: {', '.join(unknown)} (expected some of: {expected})")
    videos = []
    for record in df.to_dict(orient="records"):
        kwargs = {k: _parse_cell(k, v) for k, v in record.items()}
        videos.append(VideoInput(**{k: v for k, v in kwargs.items() if v is not None}))
    return videos


def predict_table(predictor: Predictor, df: pd.DataFrame) -> pd.DataFrame:
    preds = predictor.predict(videos_from_frame(df))
    return df.assign(
        pred_low=[round(p.low) for p in preds],
        pred_median=[round(p.median) for p in preds],
        pred_high=[round(p.high) for p in preds],
    )
