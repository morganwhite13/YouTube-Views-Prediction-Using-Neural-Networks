"""A short weekly summary of a channel, in Markdown (easy to email, paste or commit).

``ytviews digest`` answers "what happened this week and what should I try next": subscriber and
view growth from the stored snapshots, how each new upload is pacing against the model, the
biggest surprises among recent uploads, and a few concrete things to try. The full picture is in
``ytviews channel-report``; this is the version you read in a minute.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .analysis import outliers, title_terms
from .channel_report import live_uploads, next_video
from .predictor import Predictor

RECENT_DAYS = 90  # surprises are picked from uploads published this recently


def _fmt(n: float) -> str:
    if n is None or pd.isna(n):
        return "n/a"
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{sign}{n / div:.1f}{unit}"
    return f"{sign}{n:.0f}"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def subscriber_change(df: pd.DataFrame, channel_id: str, start: pd.Timestamp) -> tuple:
    """(subscribers now, change since ``start``) from the channel snapshots in ``df``."""
    if "channel_fetched_at" not in df or "subscribers_at_fetch" not in df:
        return np.nan, np.nan
    snaps = (
        df.loc[df["channel_id"] == channel_id, ["channel_fetched_at", "subscribers_at_fetch"]]
        .dropna()
        .drop_duplicates("channel_fetched_at")
        .sort_values("channel_fetched_at")
    )
    if snaps.empty:
        return np.nan, np.nan
    now = float(snaps["subscribers_at_fetch"].iloc[-1])
    before = snaps[snaps["channel_fetched_at"] <= start]
    if before.empty:
        return now, np.nan
    return now, now - float(before["subscribers_at_fetch"].iloc[-1])


def views_gained(df: pd.DataFrame, channel_id: str, start: pd.Timestamp) -> tuple[float, int]:
    """Views the channel's tracked videos gained since ``start``, and how many videos that
    covers. A video counts if it has a snapshot from before ``start`` or was published after
    it (then all its views are new)."""
    rows = df[df["channel_id"] == channel_id].sort_values("fetched_at")
    gained, n = 0.0, 0
    for _, video in rows.groupby("video_id"):
        last = float(video["views"].iloc[-1])
        if video["published_at"].iloc[-1] >= start:
            gained, n = gained + last, n + 1
            continue
        before = video[video["fetched_at"] <= start]
        if not before.empty and video["fetched_at"].iloc[-1] > start:
            gained, n = gained + last - float(before["views"].iloc[-1]), n + 1
    return gained, n


def weekly_digest(
    predictor: Predictor,
    df: pd.DataFrame,
    channel_id: str,
    *,
    days: float = 7,
    end: pd.Timestamp | None = None,
    timezone_name: str = "UTC",
    top: int = 3,
) -> str:
    """The digest for ``channel_id`` over the ``days`` before ``end`` (default: the newest
    snapshot in ``df``, so an old database gives a digest of its last week, not an empty one).

    ``df`` is the full dataset with every snapshot (``load_dataset(all_snapshots=True)``).
    """
    if predictor.target != "views":
        raise ValueError(f"this needs a views model; this model predicts {predictor.target}")
    rows = df[df["channel_id"] == channel_id]
    if rows.empty:
        raise ValueError(f"no stored videos for channel {channel_id}")
    end = pd.Timestamp(end) if end is not None else rows["fetched_at"].max()
    end = end if end.tzinfo else end.tz_localize("UTC")
    start = end - timedelta(days=days)
    rows = rows[rows["fetched_at"] <= end]  # nothing measured after the period counts
    if rows.empty:
        raise ValueError(f"no snapshots of channel {channel_id} on or before {end:%Y-%m-%d}")
    latest = rows.sort_values("fetched_at").drop_duplicates("video_id", keep="last")
    latest = latest.sort_values("published_at")
    name = str(latest["channel_title"].iloc[-1] or channel_id)
    # Scored against the channel's full history, as in training; then narrowed below.
    scored = outliers(predictor, latest, channel_id)

    out = [f"# {name}: {start:%d %b} - {end:%d %b %Y}", ""]
    subs, subs_delta = subscriber_change(rows, channel_id, start)
    gained, n_gained = views_gained(rows, channel_id, start)
    facts = []
    if not pd.isna(subs):
        delta = "" if pd.isna(subs_delta) else f" ({'+' if subs_delta >= 0 else ''}"
        delta += "" if pd.isna(subs_delta) else f"{_fmt(subs_delta)} this period)"
        facts.append(f"**{_fmt(subs)}** subscribers{delta}")
    if n_gained:
        facts.append(f"**{_fmt(gained)}** views gained across {n_gained} tracked videos")
    new = latest[latest["published_at"] >= start]
    facts.append(f"**{len(new)}** new upload{'s' if len(new) != 1 else ''}")
    out += [f"- {f}" for f in facts] + [""]

    out += ["## New uploads", ""]
    if new.empty:
        gaps = latest["published_at"].diff().dt.total_seconds().dropna() / 86400
        typical = (
            f" The channel usually uploads every {gaps.median():.0f} days." if len(gaps) else ""
        )
        out += [f"Nothing published this period.{typical}", ""]
    else:
        live = live_uploads(predictor, rows, channel_id,
                            max_age_days=days + 1, horizon_days=30)  # fmt: skip
        live = live[live["title"].isin(set(new["title"]))]
        if live.empty:
            out += ["New uploads were published, but none has a snapshot yet to judge pace.", ""]
        else:
            out += ["| upload | age | views | vs expected | 30-day forecast |",
                    "|---|---|---|---|---|"]  # fmt: skip
            for r in live.itertuples():
                out.append(
                    f"| {_cell(r.title)} | {r.age_days:.1f} d | {_fmt(r.views)} | "
                    f"{r.pace:.2f}x | {_fmt(r.median)} ({_fmt(r.low)}-{_fmt(r.high)}) |"
                )
            out.append("")

    recent_ids = set(latest.loc[latest["published_at"] >= end - timedelta(days=RECENT_DAYS),
                                "video_id"])  # fmt: skip
    table = scored[scored["video_id"].isin(recent_ids)].reset_index(drop=True)
    if len(table) >= 2:
        out += [f"## Surprises (uploads from the last {RECENT_DAYS} days)", ""]
        # only uploads outside the model's range: within it, a miss is just noise
        beat = table[table["verdict"] == "beat expectations"].head(top)
        short = table[table["verdict"] == "below expectations"].iloc[::-1].head(top)
        for label, part in (("Beat expectations", beat), ("Fell short", short)):
            for r in part.itertuples():
                out.append(f"- {label}: **{_cell(r.title)}**, {_fmt(r.views)} views vs "
                           f"~{_fmt(r.expected)} expected ({r.factor:.2f}x)")  # fmt: skip
        if beat.empty and short.empty:
            out.append(f"Every one of the {len(table)} recent uploads landed within the model's "
                       "expected range.")  # fmt: skip
        out.append("")

    out += ["## Try next", ""]
    grid = predictor.publish_times(next_video(latest), timezone_name)
    slots = grid.sort_values("median", ascending=False).head(3)
    when = ", ".join(f"{r.weekday} {r.hour:02d}:00" for r in slots.itertuples())
    best = _fmt(slots["median"].iloc[0])
    out.append(f"- Publish around **{when}** ({timezone_name}): the model's best slots for a "
               f"typical upload, about {best} views in 7 days.")  # fmt: skip
    terms = title_terms(scored)
    good = terms[terms["lift"] > 1].head(top)
    if not good.empty:
        words = ", ".join(f'"{r.term}" ({r.lift:.1f}x, {r.n} videos)' for r in good.itertuples())
        out.append(f"- Title words that have beaten expectations on this channel: {words}.")
    out += ["", "_Expectations come from the model; with few videos, treat these as hints._", ""]
    return "\n".join(out)
