"""Compare a channel with a few others: size, pace, performance against the model, and style.

``ytviews compare-channels MINE OTHER1 OTHER2`` works from the stored data (collect the other
channels first with ``ytviews collect --channel ...``). Raw views favour big channels, so the
headline is *performance against the model*: the typical ratio of a channel's actual views to
what the model expected for its videos, which already accounts for size, age and topic. Style
rows (titles, length, timing, thumbnails) then show what the channels that beat the model do
differently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import outliers
from .features import numeric_frame
from .predictor import Predictor

RECENT_UPLOADS = 30  # compare each channel's latest uploads, not its whole history


@dataclass
class ChannelComparison:
    table: pd.DataFrame  # one row per metric, one column per channel (display strings)
    raw: pd.DataFrame  # the same, as numbers
    names: list[str]
    hints: list[str]
    in_sample: bool = False  # the model was trained on these videos


def _thumbnail_stats(paths: pd.Series) -> tuple[float, float]:
    """Median brightness and saturation (0-1) of the thumbnails that exist on disk."""
    from PIL import Image

    bright, sat = [], []
    for path in paths.dropna():
        try:
            with Image.open(path) as im:
                hsv = np.asarray(im.convert("RGB").resize((64, 36)).convert("HSV"), float) / 255
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
        sat.append(hsv[..., 1].mean())
        bright.append(hsv[..., 2].mean())
    return (float(np.median(bright)), float(np.median(sat))) if bright else (np.nan, np.nan)


def _channel_metrics(predictor: Predictor, latest: pd.DataFrame, channel_id: str) -> dict:
    rows = latest[latest["channel_id"] == channel_id].sort_values("published_at")
    if rows.empty:
        raise ValueError(f"no stored videos for channel {channel_id}; collect it first")
    recent = rows.tail(RECENT_UPLOADS)
    num = numeric_frame(recent.reset_index(drop=True))
    span_weeks = (rows["published_at"].iloc[-1] - rows["published_at"].iloc[0]).days / 7
    # scored with the channel's full history (as in training), then narrowed to recent uploads
    vs_model = outliers(predictor, latest, channel_id)
    vs_model = vs_model[vs_model["video_id"].isin(set(recent["video_id"]))]
    subs = recent["subscribers"].iloc[-1]
    published = recent["published_at"]
    bright, sat = (
        _thumbnail_stats(recent["thumbnail_path"]) if "thumbnail_path" in recent else (np.nan,) * 2
    )
    return {
        "subscribers": subs,
        "uploads per week": (len(rows) - 1) / span_weeks if span_weeks > 0 else np.nan,
        "median views": recent["views"].median(),
        "views per subscriber": recent["views"].median() / subs if subs and subs > 0 else np.nan,
        "vs model (median)": float(np.exp(np.log(vs_model["factor"]).median())),
        "beat the model's range": float((vs_model["verdict"] == "beat expectations").mean()),
        "median length (min)": recent["duration_s"].median() / 60,
        "Shorts share": float(num["is_short"].mean()),
        "title length (chars)": float(num["title_chars"].median()),
        "titles with a number": float(num["title_has_digit"].mean()),
        "titles with a question": float(num["title_has_question"].mean()),
        "title caps ratio": float(num["title_caps_ratio"].median()),
        "most common weekday": published.dt.day_name().mode().iloc[0],
        "most common hour (UTC)": int(published.dt.hour.mode().iloc[0]),
        "thumbnail brightness": bright,
        "thumbnail saturation": sat,
        "_name": str(rows["channel_title"].iloc[-1] or channel_id),
    }


PERCENT = {"beat the model's range", "Shorts share", "titles with a number",
           "titles with a question", "title caps ratio", "thumbnail brightness",
           "thumbnail saturation"}  # fmt: skip
STYLE = ["median length (min)", "Shorts share", "title length (chars)", "titles with a number",
         "titles with a question", "title caps ratio", "thumbnail brightness",
         "thumbnail saturation", "uploads per week"]  # fmt: skip


def _display(metric: str, value) -> str:
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return "-"
    if metric in PERCENT:
        return f"{value:.0%}"
    if metric == "vs model (median)":
        return f"{value:.2f}x"
    if metric in ("subscribers", "median views"):
        for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if value >= div:
                return f"{value / div:.1f}{unit}"
        return f"{value:.0f}"
    if metric == "most common hour (UTC)":
        return f"{int(value):02d}:00"
    return f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}"


def style_hints(raw: pd.DataFrame, mine: str, others: list[str]) -> list[str]:
    """Where the channels that beat the model clearly more than ``mine`` differ from it in
    style (``raw``: metrics x channels, as numbers)."""
    vs = raw.loc["vs model (median)"].astype(float)
    leaders = [n for n in others if vs[n] > vs[mine] * 1.1]
    if not leaders:
        return [f"{mine} does as well as or better than the others against the model."]
    hints = []
    for metric in STYLE:
        ours = raw.at[metric, mine]
        theirs = pd.to_numeric(raw.loc[metric, leaders], errors="coerce").dropna()
        if pd.isna(ours) or theirs.empty:
            continue
        typical = float(theirs.median())
        scale = max(abs(typical), abs(float(ours)), 1e-9)
        if abs(typical - float(ours)) / scale >= 0.25:  # a clear difference, not noise
            direction = "more" if typical > ours else "less"
            hints.append(
                f"Channels beating the model more than {mine} ({', '.join(leaders)}) have "
                f"{direction} {metric}: {_display(metric, typical)} vs your "
                f"{_display(metric, ours)}."
            )
    return hints


def compare_channels(
    predictor: Predictor, df: pd.DataFrame, channel_ids: list[str]
) -> ChannelComparison:
    """Metrics for each channel (the first is "yours") and hints where the channels that beat
    the model most differ from it. ``df``: one row per video (``load_dataset()``)."""
    if predictor.target != "views":
        raise ValueError(f"this needs a views model; this model predicts {predictor.target}")
    if len(channel_ids) < 2:
        raise ValueError("give your channel and at least one other")
    if len(set(channel_ids)) < len(channel_ids):
        raise ValueError("each channel may be listed once")
    latest = df.sort_values("fetched_at").drop_duplicates("video_id", keep="last")
    metrics = {cid: _channel_metrics(predictor, latest, cid) for cid in channel_ids}
    names = [metrics[c].pop("_name") for c in channel_ids]
    # Columns are keyed by name: channels sharing a display name get their id added.
    names = [f"{n} ({c})" if names.count(n) > 1 else n
             for n, c in zip(names, channel_ids, strict=True)]  # fmt: skip
    raw = pd.DataFrame({n: metrics[c] for n, c in zip(names, channel_ids, strict=True)})
    table = raw.apply(lambda col: [_display(m, v) for m, v in col.items()], result_type="expand")
    table.index = raw.index

    hints = style_hints(raw, names[0], names[1:])
    in_sample = predictor.meta.get("trained_on") == "all data"
    return ChannelComparison(table, raw, names, hints, in_sample)


def to_markdown(result: ChannelComparison) -> str:
    cols = ["metric", *result.names]
    lines = ["| " + " | ".join(c.replace("|", "\\|") for c in cols) + " |",
             "|" + "---|" * len(cols)]  # fmt: skip
    for metric, row in result.table.iterrows():
        lines.append("| " + " | ".join([metric, *map(str, row.tolist())]) + " |")
    out = ["\n".join(lines), ""]
    if result.hints:
        out += ["**What stands out**", ""] + [f"- {h}" for h in result.hints]
    out += ["", "_Against the model, 1.00x is what it expects for a channel of that size and "
            "topic; style differences are hints, not causes._"]  # fmt: skip
    if result.in_sample:
        out += ["", "_The model was trained on these videos, so its expectations are in-sample "
                "and differences against it are muted. Compare with a model trained before them "
                "(`ytviews train --no-refit`) for a sharper picture._"]  # fmt: skip
    return "\n".join(out)
