"""SQLite storage for collected videos, channels and view-count snapshots.

Every time a video's statistics are fetched a new row is appended to ``video_stats``. Collecting
the same videos again later therefore builds up *growth curves*, which the model can learn from
because the exact age of the video at each observation is known.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    channel_id TEXT,
    channel_title TEXT,
    title TEXT,
    description TEXT,
    tags TEXT,
    category_id TEXT,
    published_at TEXT,
    default_language TEXT,
    live_broadcast TEXT,
    duration_s REAL,
    definition TEXT,
    caption INTEGER,
    thumbnail_url TEXT,
    source TEXT
);
CREATE TABLE IF NOT EXISTS video_stats (
    video_id TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    views INTEGER,
    likes INTEGER,
    comments INTEGER,
    PRIMARY KEY (video_id, fetched_at)
);
CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY,
    channel_title TEXT,
    country TEXT,
    channel_published_at TEXT,
    subscribers INTEGER,
    channel_views INTEGER,
    channel_video_count INTEGER,
    uploads_playlist TEXT,
    fetched_at TEXT
);
-- Channel statistics at every fetch, so each video snapshot can use the channel's size at
-- that time rather than its latest.
CREATE TABLE IF NOT EXISTS channel_stats (
    channel_id TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    subscribers INTEGER,
    channel_views INTEGER,
    channel_video_count INTEGER,
    PRIMARY KEY (channel_id, fetched_at)
);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
-- When a channel's uploads were last collected (channel stats alone don't count).
CREATE TABLE IF NOT EXISTS channel_uploads (
    channel_id TEXT PRIMARY KEY,
    collected_at TEXT NOT NULL
);
-- Plans recorded with `ytviews plan --record`, linked to the video once it is collected.
CREATE TABLE IF NOT EXISTS plans (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    channel_id TEXT,
    draft_title TEXT NOT NULL,
    video_json TEXT NOT NULL,          -- the recommended VideoInput (VideoInput.to_json)
    trajectory_json TEXT,              -- its predicted range by age when it was planned
    age_days REAL NOT NULL,
    predicted_low REAL, predicted_median REAL, predicted_high REAL,
    model TEXT,
    video_id TEXT                      -- the published video, once linked
);
"""

VIDEO_COLUMNS = [
    "video_id", "channel_id", "channel_title", "title", "description", "tags", "category_id",
    "published_at", "default_language", "live_broadcast", "duration_s", "definition", "caption",
    "thumbnail_url", "source",
]  # fmt: skip
CHANNEL_COLUMNS = [
    "channel_id", "channel_title", "country", "channel_published_at", "subscribers",
    "channel_views", "channel_video_count", "uploads_playlist", "fetched_at",
]  # fmt: skip


class VideoStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(videos)")}
        if "is_short" not in cols:  # 1 = Short, 0 = not, NULL = not checked yet
            self.conn.execute("ALTER TABLE videos ADD COLUMN is_short INTEGER")
            self.conn.commit()
        plan_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(plans)")}
        if plan_cols and "trajectory_json" not in plan_cols:
            self.conn.execute("ALTER TABLE plans ADD COLUMN trajectory_json TEXT")
            self.conn.commit()
        # Databases from before channel_stats existed: their one stored row per channel is its
        # only known snapshot.
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO channel_stats SELECT channel_id, fetched_at, subscribers, "
                "channel_views, channel_video_count FROM channels WHERE fetched_at IS NOT NULL "
                "AND channel_id NOT IN (SELECT DISTINCT channel_id FROM channel_stats)"
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> VideoStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ writes
    def upsert_videos(
        self, records: Iterable[dict[str, Any]], source: str = "", *, keep_existing: bool = False
    ) -> int:
        """Store videos and add a statistics snapshot for each.

        ``keep_existing`` leaves the metadata of videos already stored alone and only adds the
        snapshot; used for sparse sources (v1 caches) that would otherwise erase richer data.
        """
        n = 0
        conflict = (
            "ON CONFLICT(video_id) DO NOTHING"
            if keep_existing
            else "ON CONFLICT(video_id) DO UPDATE SET "
            + ",".join(f"{c}=excluded.{c}" for c in VIDEO_COLUMNS if c != "source")
        )
        with self.conn:
            for rec in records:
                if rec.get("views") is None:  # views hidden / unavailable: useless as a target
                    continue
                row = {c: rec.get(c) for c in VIDEO_COLUMNS}
                row["source"] = rec.get("source") or source
                self.conn.execute(
                    f"INSERT INTO videos ({','.join(VIDEO_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(VIDEO_COLUMNS))}) " + conflict,
                    [row[c] for c in VIDEO_COLUMNS],
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO video_stats VALUES (?,?,?,?,?)",
                    (
                        rec["video_id"],
                        _utc_iso(rec["fetched_at"]),
                        rec["views"],
                        rec.get("likes"),
                        rec.get("comments"),
                    ),
                )
                n += 1
        return n

    def upsert_channels(self, records: Iterable[dict[str, Any]]) -> int:
        n = 0
        with self.conn:
            for rec in records:
                self.conn.execute(
                    f"INSERT OR REPLACE INTO channels ({','.join(CHANNEL_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(CHANNEL_COLUMNS))})",
                    [rec.get(c) for c in CHANNEL_COLUMNS],
                )
                if rec.get("fetched_at"):
                    self.conn.execute(
                        "INSERT OR REPLACE INTO channel_stats VALUES (?,?,?,?,?)",
                        (
                            rec["channel_id"],
                            _utc_iso(rec["fetched_at"]),
                            rec.get("subscribers"),
                            rec.get("channel_views"),
                            rec.get("channel_video_count"),
                        ),
                    )
                n += 1
        return n

    # ------------------------------------------------------------------ reads
    def video_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT video_id FROM videos")]

    def thumbnail_urls(self) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT video_id, thumbnail_url FROM videos WHERE thumbnail_url IS NOT NULL"
        )
        return dict(rows.fetchall())

    def mark_uploads_collected(self, channel_id: str, collected_at: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO channel_uploads VALUES (?, ?)", (channel_id, collected_at)
            )

    def uploads_collected_at(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT channel_id, collected_at FROM channel_uploads"))

    def add_plan(self, plan: dict) -> int:
        """Store a plan (keys as the ``plans`` table, without plan_id); returns its id."""
        cols = [
            "created_at", "channel_id", "draft_title", "video_json", "trajectory_json", "age_days",
            "predicted_low", "predicted_median", "predicted_high", "model", "video_id",
        ]  # fmt: skip
        with self.conn:
            cur = self.conn.execute(
                f"INSERT INTO plans ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [plan.get(c) for c in cols],
            )
        return int(cur.lastrowid)

    def plans(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM plans ORDER BY plan_id", self.conn)

    def link_plan(self, plan_id: int, video_id: str | None) -> None:
        """Link a plan to a stored video (None unlinks). The video must be stored and not
        already linked to another plan, or the review would count it twice."""
        if video_id is not None:
            if not self.conn.execute(
                "SELECT 1 FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone():
                raise ValueError(f"video {video_id} is not in the database; collect it first")
            other = self.conn.execute(
                "SELECT plan_id FROM plans WHERE video_id = ? AND plan_id != ?", (video_id, plan_id)
            ).fetchone()
            if other:
                raise ValueError(f"video {video_id} is already linked to plan {other[0]}")
        with self.conn:
            cur = self.conn.execute(
                "UPDATE plans SET video_id = ? WHERE plan_id = ?", (video_id, plan_id)
            )
        if cur.rowcount == 0:
            raise KeyError(f"no plan {plan_id}")

    def channel_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT channel_id FROM channels")]

    def video_ids_without_short_flag(self) -> list[str]:
        return [
            r[0] for r in self.conn.execute("SELECT video_id FROM videos WHERE is_short IS NULL")
        ]

    def set_short_flags(self, flags: dict[str, bool | None]) -> int:
        rows = [(None if v is None else int(v), k) for k, v in flags.items()]
        with self.conn:
            self.conn.executemany("UPDATE videos SET is_short = ? WHERE video_id = ?", rows)
        return len(rows)

    def known_channel_ids_without_stats(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT v.channel_id FROM videos v "
            "LEFT JOIN channels c ON c.channel_id = v.channel_id WHERE c.channel_id IS NULL"
        )
        return [r[0] for r in rows if r[0]]

    def counts(self) -> dict[str, int]:
        q = {
            "videos": "SELECT COUNT(*) FROM videos",
            "snapshots": "SELECT COUNT(*) FROM video_stats",
            "channels": "SELECT COUNT(*) FROM channels",
        }
        return {k: self.conn.execute(v).fetchone()[0] for k, v in q.items()}

    def load_dataset(self, all_snapshots: bool = False) -> pd.DataFrame:
        """Return one row per video (latest snapshot) or one row per snapshot.

        Channel statistics come from the channel snapshot nearest in time to each video
        snapshot (``channel_fetched_at`` says which), so an old snapshot of a video is paired
        with the channel as it was then, not with its size today. ``subscribers`` is then
        estimated at the video's publish time from the channel's growth (see
        :func:`subscribers_at_publish`); ``subscribers_at_fetch`` keeps the matched count.
        """
        snapshot_filter = (
            ""
            if all_snapshots
            else "WHERE s.fetched_at = (SELECT MAX(fetched_at) FROM video_stats "
            "WHERE video_id = s.video_id)"
        )
        sql = f"""
            SELECT v.*, s.fetched_at, s.views, s.likes, s.comments,
                   c.subscribers, c.channel_views, c.channel_video_count, c.country
            FROM video_stats s
            JOIN videos v ON v.video_id = s.video_id
            LEFT JOIN channels c ON c.channel_id = v.channel_id
            {snapshot_filter}
        """
        df = normalise_frame(pd.read_sql_query(sql, self.conn))
        return self._channel_stats_as_of(df)

    def _channel_stats_as_of(self, df: pd.DataFrame) -> pd.DataFrame:
        snaps = pd.read_sql_query("SELECT * FROM channel_stats", self.conn)
        df["channel_fetched_at"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        df["subscribers_at_fetch"] = df["subscribers"]
        if snaps.empty or df.empty:
            return df
        snaps["channel_fetched_at"] = pd.to_datetime(
            snaps.pop("fetched_at"), utc=True, errors="coerce", format="ISO8601"
        ).astype("datetime64[ns, UTC]")
        snaps = snaps.dropna(subset=["channel_fetched_at"]).sort_values("channel_fetched_at")
        cols = ["subscribers", "channel_views", "channel_video_count"]
        for c in cols:
            snaps[c] = pd.to_numeric(snaps[c], errors="coerce").astype(float)
        left = df[["channel_id", "fetched_at"]].copy()
        left["fetched_at"] = left["fetched_at"].astype("datetime64[ns, UTC]")
        left["_row"] = np.arange(len(df))

        # The channel snapshot nearest in time. If it is later than the video snapshot, its
        # total includes views the video had not reached yet; add_channel_history subtracts the
        # video's own views from the same time, so they don't leak into the channel average.
        matched = pd.merge_asof(
            left.sort_values("fetched_at"), snaps, left_on="fetched_at",
            right_on="channel_fetched_at", by="channel_id", direction="nearest",
        ).set_index("_row").sort_index()  # fmt: skip
        found = matched["channel_fetched_at"].notna().to_numpy()
        for c in cols:  # channels with no snapshot keep the channels-table values
            df.loc[found, c] = matched.loc[found, c].to_numpy()
        df.loc[found, "channel_fetched_at"] = matched.loc[found, "channel_fetched_at"].to_numpy()
        df["subscribers_at_fetch"] = df["subscribers"]
        df["subscribers"] = subscribers_at_publish(df, snaps)
        return df

    # ------------------------------------------------------------------ v1 compatibility
    def import_legacy_json(self, path: str | Path, fetched_at: str | None = None) -> int:
        """Import a JSON cache written by the v1 ``projectFinal.py`` data collectors.

        v1 did not record when statistics were fetched; by default the file's modification time
        is used as the best available estimate.
        """
        path = Path(path)
        if fetched_at is None:
            fetched_at = (
                datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
        records = json.loads(path.read_text())
        videos, channels = [], {}
        for r in records:
            videos.append(
                {
                    "video_id": r["videoId"],
                    "channel_id": r.get("channelId"),
                    "channel_title": r.get("channelTitle", ""),
                    "title": r.get("title", ""),
                    "description": r.get("description", ""),
                    "tags": "",
                    "category_id": str(r.get("categoryId", "")),
                    "published_at": r.get("publishedAt"),
                    "views": r.get("views"),
                    "fetched_at": fetched_at,
                }
            )
            if r.get("channelId"):
                channels[r["channelId"]] = {
                    "channel_id": r["channelId"],
                    "channel_title": r.get("channelTitle", ""),
                    "subscribers": r.get("subscriberCount") or None,
                    "fetched_at": fetched_at,
                }
        # Never overwrite richer channel rows that were collected with the v2 client.
        existing = set(self.channel_ids())
        self.upsert_channels(c for cid, c in channels.items() if cid not in existing)
        return self.upsert_videos(videos, source=f"legacy:{path.name}", keep_existing=True)


# Estimating a channel's size when an older video went up, from its snapshots' growth.
MIN_GROWTH_SPAN_DAYS = 14  # snapshots closer together than this say too little about growth
MAX_GROWTH_PER_DAY = np.log(2) / 30  # at most doubling (or halving) per month
MAX_EXTRAPOLATION_DAYS = 365  # beyond this before/after the snapshots, hold the estimate
MAX_EXTRAPOLATED_FACTOR = 10.0  # and never move more than 10x from the nearest snapshot


def subscribers_at_publish(df: pd.DataFrame, snaps: pd.DataFrame) -> pd.Series:
    """Each row's subscriber count at its video's publish time.

    A prediction is made before publishing, with the channel's size then, so training rows
    should use the size at publish time too, not at the (later) fetch. Between two channel
    snapshots this interpolates log subscribers; before the first or after the last it
    extrapolates the channel's average growth rate, bounded (see the constants above).
    Channels with a single snapshot, or snapshots too close together, keep ``subscribers``.
    """
    out = df["subscribers"].astype(float).copy()
    snaps = snaps[snaps["subscribers"] > 0]
    day = np.timedelta64(1, "D")
    epoch = pd.Timestamp(0, tz="UTC")
    for channel, group in snaps.groupby("channel_id"):
        t = ((group["channel_fetched_at"] - epoch) / day).to_numpy(float)
        y = np.log(group["subscribers"].to_numpy(float))
        span = t[-1] - t[0]
        if not channel or span < MIN_GROWTH_SPAN_DAYS:
            continue
        rows = (df["channel_id"] == channel).to_numpy()
        if not rows.any():
            continue
        tp = ((df.loc[rows, "published_at"] - epoch) / day).to_numpy(float)
        rate = np.clip((y[-1] - y[0]) / span, -MAX_GROWTH_PER_DAY, MAX_GROWTH_PER_DAY)
        before = np.clip(tp - t[0], -MAX_EXTRAPOLATION_DAYS, 0)
        after = np.clip(tp - t[-1], 0, MAX_EXTRAPOLATION_DAYS)
        cap = np.log(MAX_EXTRAPOLATED_FACTOR)
        est = np.interp(tp, t, y)  # flat outside the snapshots ...
        est = est + np.clip(rate * (before + after), -cap, cap)  # ... plus bounded growth
        out[rows] = np.exp(est).round()
    return out


def _utc_iso(value: Any) -> str:
    """A timestamp as UTC ISO text. Snapshots are picked by MAX(fetched_at) on the text, which
    only orders correctly when every value has the same offset (a user-typed
    ``import-legacy --fetched-at`` may not)."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes so that downstream code can rely on them."""
    df = df.copy()
    for col in ("published_at", "fetched_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="ISO8601")
    for col in (
        "title", "description", "tags", "channel_title", "category_id", "channel_id",
        "default_language",
    ):  # fmt: skip
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    for col in (
        "views",
        "subscribers",
        "channel_views",
        "channel_video_count",
        "duration_s",
        "caption",
        "is_short",
        "likes",
        "comments",
    ):
        if col not in df:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    if "definition" not in df:
        df["definition"] = None
    df = df.dropna(subset=["views", "published_at", "fetched_at"])
    return df.reset_index(drop=True)
