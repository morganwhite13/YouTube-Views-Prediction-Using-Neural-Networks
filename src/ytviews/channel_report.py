"""A shareable one-page report for a single channel: how its uploads did against expectations
and when its next upload should go out."""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import outliers, title_terms
from .categories import CATEGORY_NAMES
from .predictor import Predictor, VideoInput
from .report_html import CSS, FONTS, _fmt_views, _sup

EXTRA_CSS = """
.pills { display: flex; flex-wrap: wrap; gap: 6px; }
.pill { font: 12px/1 var(--mono); padding: 5px 8px; border-radius: 999px;
  border: 1px solid var(--rule); color: var(--muted); }
.up { color: var(--good); } .down { color: var(--warn); }
td.t { white-space: normal; min-width: 220px; text-align: left; }
.heat { display: grid; grid-template-columns: 44px repeat(24, minmax(10px, 1fr)); gap: 2px;
  font: 10px/1 var(--mono); color: var(--muted); min-width: 520px; }
.heat div { height: 18px; border-radius: 2px; }
.heat .lab { background: none; display: flex; align-items: center; height: 18px; }
.heat .hr { background: none; height: auto; text-align: center; padding-top: 4px; }
"""


def _timeline_svg(table: pd.DataFrame) -> str:
    """Actual (dots) vs expected (line) views per upload, by publish date, log scale."""
    t = table.sort_values("published_at")
    dates = pd.to_datetime(t["published_at"])
    x0, x1 = dates.min().value, max(dates.max().value, dates.min().value + 1)
    act = np.log10(t["views"].clip(lower=1).to_numpy())
    exp = np.log10(t["expected"].clip(lower=1).to_numpy())
    k0, k1 = math.floor(min(act.min(), exp.min())), math.ceil(max(act.max(), exp.max()))
    k1 = max(k1, k0 + 1)
    w, h, left, bottom, top, right = 640, 260, 52, 30, 10, 12

    def sx(v):
        return left + (v - x0) / (x1 - x0) * (w - left - right)

    def sy(v):
        return h - bottom - (v - k0) / (k1 - k0) * (h - bottom - top)

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Views per upload '
           'against expectation" style="max-width:100%;height:auto;display:block">']  # fmt: skip
    for k in range(k0, k1 + 1):
        out.append(
            f'<line x1="{left}" y1="{sy(k):.1f}" x2="{w - right}" y2="{sy(k):.1f}" '
            f'stroke="var(--rule)"/><text x="{left - 8}" y="{sy(k) + 4:.1f}" '
            f'text-anchor="end">{_sup(k)}</text>'
        )
    for d in (dates.min(), dates.max()):
        anchor = "start" if d == dates.min() else "end"
        out.append(f'<text x="{sx(d.value):.1f}" y="{h - 8}" text-anchor="{anchor}">'
                   f"{d:%d %b %Y}</text>")  # fmt: skip
    pts = " ".join(f"{sx(d.value):.1f},{sy(e):.1f}" for d, e in zip(dates, exp, strict=True))
    out.append(f'<polyline points="{pts}" fill="none" stroke="var(--band)" stroke-width="2"/>')
    for d, a, e in zip(dates, act, exp, strict=True):
        colour = "var(--good)" if a > e else "var(--warn)"
        out.append(f'<circle cx="{sx(d.value):.1f}" cy="{sy(a):.1f}" r="3" fill="{colour}"/>')
    out.append("</svg>")
    return "".join(out)


def _video_table(rows: pd.DataFrame) -> str:
    body = "".join(
        f'<tr><td class="t">{html.escape(r.title)}</td><td>{r.published_at}</td>'
        f"<td>{_fmt_views(r.views)}</td><td>{_fmt_views(r.expected)}</td>"
        f'<td class="{"up" if r.factor >= 1 else "down"}">{r.factor:.2f}×</td></tr>'
        for r in rows.itertuples()
    )
    return (
        '<div class="scroll"><table><thead><tr><th>Video</th><th>Published</th><th>Views</th>'
        f"<th>Expected</th><th>vs expected</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def _heatmap(grid: pd.DataFrame) -> str:
    lo, hi = grid["median"].min(), grid["median"].max()
    hours = [f'<div class="hr">{h if h % 3 == 0 else ""}</div>' for h in range(24)]
    cells = ['<div class="lab"></div>', *hours]
    for _, day in grid.groupby("weekday_index", sort=True):
        cells.append(f'<div class="lab">{day["weekday"].iloc[0][:3]}</div>')
        for r in day.sort_values("hour").itertuples():
            share = 0.12 + 0.88 * (r.median - lo) / (hi - lo or 1)
            cells.append(
                f'<div title="{r.weekday} {r.hour:02d}:00 ~{_fmt_views(r.median)}" '
                f'style="background: color-mix(in srgb, var(--accent) {share * 100:.0f}%, '
                f'var(--surface))"></div>'
            )
    return f'<div class="scroll"><div class="heat">{"".join(cells)}</div></div>'


def _video_from_row(row: pd.Series, earlier: pd.DataFrame) -> VideoInput:
    """A ``VideoInput`` for a stored video, with the channel's *earlier* uploads as context."""
    tags = row.get("tags")
    thumb = row.get("thumbnail_path")
    recent = earlier.sort_values("published_at").tail(10)[::-1]
    ages = (recent["fetched_at"] - recent["published_at"]).dt.total_seconds() / 86400
    return VideoInput(
        title=str(row["title"]),
        description=str(row.get("description") or ""),
        tags=[t for t in str(tags).split("|") if t] if isinstance(tags, str) else [],
        category_id=str(row["category_id"]),
        duration_s=None if pd.isna(row.get("duration_s")) else float(row["duration_s"]),
        published_at=row["published_at"].to_pydatetime(),
        subscribers=None if pd.isna(row.get("subscribers")) else float(row["subscribers"]),
        channel_video_count=None
        if pd.isna(row.get("channel_video_count"))
        else float(row["channel_video_count"]),
        recent_video_views=recent["views"].astype(float).tolist(),
        recent_video_age_days=ages.tolist(),
        thumbnail_path=thumb if isinstance(thumb, str) and thumb else None,
        caption=bool(row.get("caption")),
        is_short=None if pd.isna(row.get("is_short")) else bool(row["is_short"]),
    )  # fmt: skip


def live_uploads(
    predictor: Predictor,
    df: pd.DataFrame,
    channel_id: str,
    *,
    max_age_days: float = 14,
    horizon_days: float = 30,
) -> pd.DataFrame:
    """Recent uploads still growing: their pace so far and where they are heading.

    Each video's latest snapshot goes through ``Predictor.track`` (the early-views nowcast),
    with the channel's uploads published *before* it as context. Empty if the model does not
    predict views or the channel has no uploads younger than ``max_age_days``.
    """
    cols = ["title", "age_days", "views", "expected_now", "pace", "low", "median", "high",
            "calibrated"]  # fmt: skip
    if predictor.target != "views":
        return pd.DataFrame(columns=cols)
    videos = df[df["channel_id"] == channel_id].sort_values("fetched_at")
    latest = videos.drop_duplicates("video_id", keep="last")
    age = (latest["fetched_at"] - latest["published_at"]).dt.total_seconds() / 86400
    rows = []
    for (_, row), a in zip(latest.iterrows(), age, strict=True):
        if not 0 < a < min(max_age_days, horizon_days):
            continue
        earlier = latest[latest["published_at"] < row["published_at"]]
        nc = predictor.track(_video_from_row(row, earlier), float(row["views"]), float(a),
                             days=(horizon_days,))  # fmt: skip
        f = nc.forecast.iloc[-1]
        rows.append([row["title"], a, float(row["views"]), nc.expected_views, nc.vs_expected,
                     f["low"], f["median"], f["high"], nc.calibrated])  # fmt: skip
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("age_days").reset_index(drop=True)


def pulse(
    predictor: Predictor,
    client: Any,
    channel_id: str,
    *,
    n: int = 5,
    history: int = 15,
    horizon_days: float = 30,
) -> pd.DataFrame:
    """Live version of :func:`live_uploads`: fetch a channel's latest uploads from the API now.

    Costs a few quota units (one channel call, one or two playlist pages, one videos call). The
    ``history`` older uploads are fetched too, as the channel context for each forecast.
    """
    from .collect import keep_video

    ch = client.channels([channel_id])
    if not ch or not ch[0].get("uploads_playlist"):
        raise ValueError(f"channel {channel_id} not found or has no uploads")
    ch = ch[0]
    ids = client.upload_ids(ch["uploads_playlist"], n + history)
    recs = [r for r in client.videos(ids) if keep_video(r, language=None)]
    if not recs:
        return live_uploads(predictor, pd.DataFrame(columns=["channel_id"]), channel_id)
    df = pd.DataFrame(recs).assign(
        channel_id=channel_id,
        subscribers=ch.get("subscribers"),
        channel_video_count=ch.get("channel_video_count"),
    )
    for col in ("published_at", "fetched_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    live = live_uploads(predictor, df, channel_id, max_age_days=horizon_days,
                        horizon_days=horizon_days)  # fmt: skip
    return live.head(n)


def _live_section(live: pd.DataFrame, horizon_days: float) -> str:
    if live.empty:
        return ""
    body = "".join(
        f'<tr><td class="t">{html.escape(str(r.title))}</td><td>{r.age_days:.1f} d</td>'
        f"<td>{_fmt_views(r.views)}</td>"
        f'<td class="{"up" if r.pace >= 1 else "down"}">{r.pace:.2f}×</td>'
        f"<td>{_fmt_views(r.median)}</td><td>{_fmt_views(r.low)}–{_fmt_views(r.high)}</td></tr>"
        for r in live.itertuples()
    )
    note = (
        ""
        if live["calibrated"].all()
        else '<p class="note">No snapshot pairs to calibrate against yet, so these forecasts '
        "assume each video keeps its current pace. Run <code>ytviews refresh</code> regularly "
        "and retrain.</p>"
    )
    return f"""<section>
    <h2>Still growing</h2>
    <p>Uploads from the last few days: how far ahead of or behind the model's expectation each
    one is so far, and where that puts it at day {horizon_days:g}.</p>
    {note}
    <div class="scroll"><table><thead><tr><th>Video</th><th>Age</th><th>Views now</th>
    <th>Pace</th><th>Day {horizon_days:g}</th><th>Range</th></tr></thead>
    <tbody>{body}</tbody></table></div>
  </section>"""


def _terms_section(terms: pd.DataFrame) -> str:
    if terms.empty:
        return ""
    pills = "".join(
        f'<span class="pill {"up" if r.lift >= 1 else "down"}">{html.escape(r.term)} · '
        f"{r.lift:.2f}× · {r.n} videos</span>"
        for r in terms.itertuples()
    )
    return f"""<section>
    <h2>What's working</h2>
    <p>Title words whose videos did better (green) or worse (amber) than the model expected,
    compared with this channel's other uploads. The model already counts what it knows about
    words in general, so these point at what is specific to this audience. With a handful of
    videos each, treat them as ideas to test, not rules.</p>
    <div class="pills">{pills}</div>
  </section>"""


def channel_report_html(
    predictor: Predictor,
    df: pd.DataFrame,
    channel_id: str,
    *,
    timezone_name: str = "UTC",
    top: int = 5,
    live_horizon_days: float = 30,
) -> str:
    table = outliers(predictor, df, channel_id)
    live = live_uploads(predictor, df, channel_id, horizon_days=live_horizon_days)
    videos = df[df["channel_id"] == channel_id].sort_values("published_at")
    latest = videos.drop_duplicates("video_id", keep="last")
    name = str(latest["channel_title"].iloc[-1] or channel_id)
    subs = latest["subscribers"].iloc[-1]
    cat = latest["category_id"].mode().iloc[0]
    recent = latest.tail(10)
    ages = ((recent["fetched_at"] - recent["published_at"]).dt.total_seconds() / 86400).tolist()
    next_video = VideoInput(
        title=str(recent["title"].iloc[-1]),
        subscribers=None if pd.isna(subs) else float(subs),
        category_id=str(cat),
        duration_s=float(latest["duration_s"].median()) if latest["duration_s"].notna().any()
        else None,
        recent_video_views=recent["views"].tolist()[::-1],
        recent_video_age_days=ages[::-1],
    )  # fmt: skip
    grid = predictor.publish_times(next_video, timezone_name)
    best = grid.sort_values("median", ascending=False).head(3)
    beat = int((table["verdict"] == "beat expectations").sum())
    below = int((table["verdict"] == "below expectations").sum())
    in_sample = predictor.meta.get("trained_on") == "all data"

    best_pills = "".join(
        f'<span class="pill">{r.weekday} {r.hour:02d}:00 · ~{_fmt_views(r.median)}</span>'
        for r in best.itertuples()
    )
    caveat = (
        '<p class="note">The model was trained on these videos, so "expected" is in-sample and '
        "somewhat flattering. Treat small differences as noise.</p>"
        if in_sample
        else ""
    )
    created = datetime.now(timezone.utc).strftime("%d %B %Y")
    subs_text = "hidden" if pd.isna(subs) else _fmt_views(subs)
    return f"""<title>{html.escape(name)} Channel Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{FONTS}
<style>{CSS}{EXTRA_CSS}</style>
<main>
  <header>
    <span class="eyebrow">ytviews · channel report</span>
    <h1>{html.escape(name)}</h1>
    <ul class="facts">
      <li><b>{subs_text}</b> subscribers</li>
      <li><b>{len(table)}</b> uploads analysed</li>
      <li>median <b>{_fmt_views(table["views"].median())}</b> views</li>
      <li>category <b>{html.escape(CATEGORY_NAMES.get(str(cat), str(cat)))}</b></li>
    </ul>
    <p class="lead">{beat} uploads beat the model's expectation by more than its usual range;
    {below} fell short of it.</p>
    {caveat}
  </header>
  <section>
    <h2>Every upload against expectations</h2>
    <figure>{_timeline_svg(table)}
      <figcaption>Dots are actual views (green above expectation, amber below); the line is
      what the model expected for each video at the age it was measured.</figcaption>
    </figure>
  </section>
  {_live_section(live, live_horizon_days)}
  {_terms_section(title_terms(table))}
  <section>
    <h2>Biggest wins</h2>
    {_video_table(table.head(top))}
  </section>
  <section>
    <h2>Biggest misses</h2>
    {_video_table(table.tail(top).iloc[::-1])}
  </section>
  <section>
    <h2>When to publish next</h2>
    <p>Predicted views after 7 days for the next upload, by weekday and hour
    ({html.escape(timezone_name)}). Darker is better.</p>
    <div class="pills">{best_pills}</div>
    {_heatmap(grid)}
  </section>
  <footer>Generated {created} by <code>ytviews channel-report</code></footer>
</main>
"""


def write_channel_report(
    predictor: Predictor, df: pd.DataFrame, channel_id: str, out: str | Path, **kwargs
) -> Path:
    out = Path(out)
    out.write_text("<!doctype html>\n" + channel_report_html(predictor, df, channel_id, **kwargs))
    return out
