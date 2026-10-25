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
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
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

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> VideoStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ writes
    def upsert_videos(self, records: Iterable[dict[str, Any]], source: str = "") -> int:
        n = 0
        with self.conn:
            for rec in records:
                if rec.get("views") is None:  # views hidden / unavailable: useless as a target
                    continue
                row = {c: rec.get(c) for c in VIDEO_COLUMNS}
                row["source"] = rec.get("source") or source
                self.conn.execute(
                    f"INSERT INTO videos ({','.join(VIDEO_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(VIDEO_COLUMNS))}) "
                    "ON CONFLICT(video_id) DO UPDATE SET "
                    + ",".join(f"{c}=excluded.{c}" for c in VIDEO_COLUMNS if c != "source"),
                    [row[c] for c in VIDEO_COLUMNS],
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO video_stats VALUES (?,?,?,?,?)",
                    (
                        rec["video_id"],
                        rec["fetched_at"],
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

    def channel_fetched_at(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT channel_id, fetched_at FROM channels")
        return {cid: ts for cid, ts in rows if ts}

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
        """Return one row per video (latest snapshot) or one row per snapshot."""
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
        df = pd.read_sql_query(sql, self.conn)
        return normalise_frame(df)

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
        return self.upsert_videos(videos, source=f"legacy:{path.name}")


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
