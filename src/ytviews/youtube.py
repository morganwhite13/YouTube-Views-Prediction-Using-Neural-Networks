"""Thin, quota-aware wrapper around the YouTube Data API v3.

Compared to the v1 script this client:

* batches ``videos.list`` / ``channels.list`` calls 50 ids at a time (v1 issued one request per
  video *and* one per video's channel),
* walks a channel's uploads playlist (1 quota unit per 50 videos) instead of ``search.list``
  (100 units per page),
* records *when* statistics were fetched, so a video's age at observation time is known exactly,
* retries transient errors and stops cleanly when the daily quota is exhausted.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

# Documented quota costs: https://developers.google.com/youtube/v3/determine_quota_cost
COST_LIST = 1
COST_SEARCH = 100
MAX_IDS_PER_CALL = 50

_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


class QuotaExceeded(RuntimeError):
    """Raised when the API reports that the daily quota (or our local budget) is used up."""


def parse_duration(value: str | None) -> float | None:
    """Convert an ISO-8601 duration such as ``PT1H2M3S`` into seconds."""
    if not value:
        return None
    match = _DURATION_RE.match(value)
    if not match:
        return None
    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    return float(
        parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]
    )


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_video(item: dict[str, Any], fetched_at: str) -> dict[str, Any]:
    """Flatten a ``videos.list`` item into the record format stored in the database."""
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    details = item.get("contentDetails", {})
    thumbs = snippet.get("thumbnails", {})
    thumb = (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url")
    return {
        "video_id": item["id"],
        "channel_id": snippet.get("channelId"),
        "channel_title": snippet.get("channelTitle", ""),
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "tags": "|".join(snippet.get("tags", []) or []),
        "category_id": str(snippet.get("categoryId", "")),
        "published_at": snippet.get("publishedAt"),
        "default_language": snippet.get("defaultLanguage") or snippet.get("defaultAudioLanguage"),
        "live_broadcast": snippet.get("liveBroadcastContent", "none"),
        "duration_s": parse_duration(details.get("duration")),
        "definition": details.get("definition"),
        "caption": 1 if details.get("caption") == "true" else 0,
        "thumbnail_url": thumb,
        "views": _int_or_none(stats.get("viewCount")),
        "likes": _int_or_none(stats.get("likeCount")),
        "comments": _int_or_none(stats.get("commentCount")),
        "fetched_at": fetched_at,
    }


def parse_channel(item: dict[str, Any], fetched_at: str) -> dict[str, Any]:
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    playlists = item.get("contentDetails", {}).get("relatedPlaylists", {})
    hidden = bool(stats.get("hiddenSubscriberCount", False))
    return {
        "channel_id": item["id"],
        "channel_title": snippet.get("title", ""),
        "country": snippet.get("country"),
        "channel_published_at": snippet.get("publishedAt"),
        "subscribers": None if hidden else _int_or_none(stats.get("subscriberCount")),
        "channel_views": _int_or_none(stats.get("viewCount")),
        "channel_video_count": _int_or_none(stats.get("videoCount")),
        "uploads_playlist": playlists.get("uploads"),
        "fetched_at": fetched_at,
    }


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class YouTubeClient:
    """Minimal YouTube Data API client.

    ``service`` can be injected (e.g. a fake in tests); otherwise one is built from ``api_key``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        service: Any = None,
        quota_budget: int = 9_000,
        max_retries: int = 4,
    ) -> None:
        if service is None:
            if not api_key:
                raise ValueError(
                    "A YouTube Data API key is required. Set YOUTUBE_API_KEY in your environment "
                    "or .env file (see .env.example)."
                )
            try:
                from googleapiclient.discovery import build
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "google-api-python-client is not installed; run `pip install ytviews[youtube]`"
                ) from exc
            service = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        self._service = service
        self.quota_budget = quota_budget
        self.quota_used = 0
        self.max_retries = max_retries

    # ------------------------------------------------------------------ low level
    def _execute(self, request: Any, cost: int) -> dict[str, Any]:
        if self.quota_used + cost > self.quota_budget:
            raise QuotaExceeded(
                f"Local quota budget of {self.quota_budget} units reached "
                f"({self.quota_used} used). Raise --quota-budget or try again tomorrow."
            )
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                response = request.execute()
                self.quota_used += cost
                return response
            except Exception as exc:  # googleapiclient.errors.HttpError, socket errors, ...
                status = getattr(getattr(exc, "resp", None), "status", None)
                text = str(getattr(exc, "content", b"") or exc)
                if status == 403 and ("quotaExceeded" in text or "dailyLimitExceeded" in text):
                    raise QuotaExceeded("YouTube API daily quota exhausted.") from exc
                retriable = status is None or status in (429, 500, 502, 503, 504)
                if not retriable or attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ resources
    def videos(self, video_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Fetch full metadata + statistics for any number of videos (50 per request)."""
        ids = list(dict.fromkeys(v for v in video_ids if v))
        out: list[dict[str, Any]] = []
        for chunk in _chunks(ids, MAX_IDS_PER_CALL):
            fetched_at = utcnow_iso()
            resp = self._execute(
                self._service.videos().list(
                    part="snippet,statistics,contentDetails", id=",".join(chunk), maxResults=50
                ),
                COST_LIST,
            )
            out.extend(parse_video(item, fetched_at) for item in resp.get("items", []))
        return out

    def channels(self, channel_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(c for c in channel_ids if c))
        out: list[dict[str, Any]] = []
        for chunk in _chunks(ids, MAX_IDS_PER_CALL):
            fetched_at = utcnow_iso()
            resp = self._execute(
                self._service.channels().list(
                    part="snippet,statistics,contentDetails", id=",".join(chunk), maxResults=50
                ),
                COST_LIST,
            )
            out.extend(parse_channel(item, fetched_at) for item in resp.get("items", []))
        return out

    def upload_ids(self, uploads_playlist: str, max_videos: int = 50) -> list[str]:
        """Most recent video ids from a channel's uploads playlist."""
        ids: list[str] = []
        token = None
        while len(ids) < max_videos:
            resp = self._execute(
                self._service.playlistItems().list(
                    part="contentDetails",
                    playlistId=uploads_playlist,
                    maxResults=min(50, max_videos - len(ids)),
                    pageToken=token,
                ),
                COST_LIST,
            )
            ids.extend(i["contentDetails"]["videoId"] for i in resp.get("items", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return ids[:max_videos]

    def trending(
        self, region: str = "US", category_id: str | None = None, max_videos: int = 50
    ) -> list[dict[str, Any]]:
        """Videos on the "most popular" chart (already include full metadata)."""
        out: list[dict[str, Any]] = []
        token = None
        while len(out) < max_videos:
            kwargs: dict[str, Any] = {
                "part": "snippet,statistics,contentDetails",
                "chart": "mostPopular",
                "regionCode": region,
                "maxResults": min(50, max_videos - len(out)),
                "pageToken": token,
            }
            if category_id:
                kwargs["videoCategoryId"] = category_id
            fetched_at = utcnow_iso()
            resp = self._execute(self._service.videos().list(**kwargs), COST_LIST)
            out.extend(parse_video(item, fetched_at) for item in resp.get("items", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return out[:max_videos]

    def search_channel_ids(
        self,
        query: str | None = None,
        *,
        category_id: str | None = None,
        max_results: int = 50,
        language: str = "en",
        region: str | None = None,
    ) -> list[str]:
        """Discover channels via search (expensive: 100 units per page of 50)."""
        kwargs: dict[str, Any] = {
            "part": "snippet",
            "type": "video",
            "maxResults": min(50, max_results),
            "relevanceLanguage": language,
            "safeSearch": "moderate",
        }
        if query:
            kwargs["q"] = query
        if category_id:
            kwargs["videoCategoryId"] = category_id
        if region:
            kwargs["regionCode"] = region
        resp = self._execute(self._service.search().list(**kwargs), COST_SEARCH)
        return list(dict.fromkeys(i["snippet"]["channelId"] for i in resp.get("items", [])))
