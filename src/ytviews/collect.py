"""Data-collection strategies built on :class:`~ytviews.youtube.YouTubeClient`.

All strategies end up in :func:`collect_channels`, which grabs the most recent uploads of each
channel. Sampling *many videos per channel* is deliberate: it lets the model learn how a given
video performs relative to the channel's typical video, which is the most useful question for a
creator ("will this upload beat my usual numbers?").
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .store import VideoStore
from .youtube import QuotaExceeded, YouTubeClient

log = logging.getLogger(__name__)

# Channels hand-picked for the original (v1) project; kept as a sensible default seed list.
DEFAULT_SEED_CHANNELS = [
    "UCgRQHK8Ttr1j9xCEpCAlgbQ", "UCHnyfMqiRRG1u-2MsSQLbXA", "UCGfUuxBzB8E30XjCjOvji2w",
    "UCBJycsmduvYEL83R_U4JriQ", "UCvK4bOhULCpmLabd2pDMtnA", "UCc9CjaAjsMMvaSghZB7-Kog",
    "UCX6OQ3DkcsbYNE6H8uQQuVA", "UCQJWtTnAHhEG5w4uN0udnUQ", "UCL6JmiMXKoXS6bpP1D3bk8g",
    "UCZB6V9fUov0Mx_us3MWWILg", "UClyGlKOhDUooPJFy4v_mqPg", "UCw1SQ6QRRtfAhrN_cjkrOgA",
    "UCTkXRDQl0luXxVQrRQvWS6w", "UCLXo7UDZvByw2ixzpQCufnA", "UCAuk798iHprjTtwlClkFxMA",
    "UCimiUgDLbi6P17BdaCZpVbg", "UCDogdKl7t7NHzQ95aEwkdMw", "UC2C_jShtL725hvbm1arSV9w",
    "UCftwRNsjfRo08xYE31tkiyw", "UCyps-v4WNjWDnYRKmZ4BUGw", "UCsEukrAd64fqA7FjwkmZ_Dw",
    "UCq6VFHwMzcMXbuKyG7SQYIg", "UCTSRIY3GLFYIpkR2QwyeklA",
]  # fmt: skip


def keep_video(rec: dict[str, Any], language: str | None = "en", min_views: int = 0) -> bool:
    """Filter out live/upcoming streams, non-matching languages and hidden view counts."""
    if rec.get("views") is None or rec["views"] < min_views:
        return False
    if rec.get("live_broadcast") not in (None, "none"):
        return False
    lang = rec.get("default_language")
    return not (language and lang and not str(lang).lower().startswith(language.lower()))


def _filtered(records: Iterable[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [r for r in records if keep_video(r, **filters)]


def collect_channels(
    client: YouTubeClient,
    store: VideoStore,
    channel_ids: Iterable[str],
    *,
    max_videos_per_channel: int = 50,
    source: str = "channels",
    progress: Callable[[str], None] | None = None,
    **filters: Any,
) -> int:
    """Store channel stats and the latest ``max_videos_per_channel`` uploads of each channel."""
    channels = client.channels(channel_ids)
    store.upsert_channels(channels)
    total = 0
    for i, ch in enumerate(channels, 1):
        if not ch.get("uploads_playlist"):
            continue
        try:
            ids = client.upload_ids(ch["uploads_playlist"], max_videos_per_channel)
            videos = _filtered(client.videos(ids), **filters)
        except QuotaExceeded:
            raise
        except Exception as exc:  # a deleted/private playlist should not abort the whole run
            log.warning("Skipping channel %s: %s", ch["channel_id"], exc)
            continue
        total += store.upsert_videos(videos, source=source)
        if progress:
            progress(
                f"[{i}/{len(channels)}] {ch['channel_title'][:40]:<40} +{len(videos):>3} videos "
                f"(quota used: {client.quota_used})"
            )
    return total


def collect_trending(
    client: YouTubeClient,
    store: VideoStore,
    *,
    regions: Iterable[str] = ("US",),
    category_ids: Iterable[str | None] = (None,),
    max_videos: int = 50,
    expand_channels: bool = True,
    max_videos_per_channel: int = 30,
    progress: Callable[[str], None] | None = None,
    **filters: Any,
) -> int:
    """Store trending videos and (optionally) the recent back-catalogue of their channels.

    Expanding to the channel's back-catalogue matters: trending videos alone are a heavily
    biased sample (they are all hits), which is one reason v1 struggled.
    """
    total = 0
    channel_ids: list[str] = []
    for region in regions:
        for cat in category_ids:
            vids = _filtered(
                client.trending(region=region, category_id=cat, max_videos=max_videos), **filters
            )
            total += store.upsert_videos(vids, source=f"trending:{region}:{cat or 'all'}")
            channel_ids += [v["channel_id"] for v in vids]
    channel_ids = list(dict.fromkeys(channel_ids))
    if expand_channels:
        total += collect_channels(
            client,
            store,
            channel_ids,
            max_videos_per_channel=max_videos_per_channel,
            source="trending-channel",
            progress=progress,
            **filters,
        )
    else:
        store.upsert_channels(client.channels(channel_ids))
    return total


def collect_search(
    client: YouTubeClient,
    store: VideoStore,
    queries: Iterable[str | None],
    *,
    category_id: str | None = None,
    channels_per_query: int = 25,
    max_videos_per_channel: int = 30,
    progress: Callable[[str], None] | None = None,
    **filters: Any,
) -> int:
    """Discover channels through search queries, then sample their uploads."""
    channel_ids: list[str] = []
    for q in queries:
        channel_ids += client.search_channel_ids(
            q, category_id=category_id, max_results=channels_per_query
        )
    return collect_channels(
        client, store, dict.fromkeys(channel_ids), max_videos_per_channel=max_videos_per_channel,
        source="search", progress=progress, **filters,
    )  # fmt: skip


def refresh(client: YouTubeClient, store: VideoStore, include_channels: bool = True) -> int:
    """Re-fetch statistics for every stored video, adding a new snapshot for each one."""
    n = store.upsert_videos(client.videos(store.video_ids()), source="refresh")
    if include_channels:
        store.upsert_channels(client.channels(store.channel_ids()))
    missing = store.known_channel_ids_without_stats()
    if missing:
        store.upsert_channels(client.channels(missing))
    return n


def download_thumbnails(
    store: VideoStore, folder: str | Path, *, workers: int = 8, overwrite: bool = False
) -> tuple[int, int]:
    """Download thumbnails (no API quota needed) to ``<folder>/<video_id>.jpg``.

    Returns ``(downloaded, failed)``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    todo = {
        vid: url
        for vid, url in store.thumbnail_urls().items()
        if overwrite or not (folder / f"{vid}.jpg").exists()
    }

    def fetch(item: tuple[str, str]) -> bool:
        vid, url = item
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = resp.read()
            (folder / f"{vid}.jpg").write_bytes(data)
            return True
        except Exception as exc:
            log.debug("thumbnail %s failed: %s", vid, exc)
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fetch, todo.items()))
    return sum(results), len(results) - sum(results)


SHORTS_URL = "https://www.youtube.com/shorts/{}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - urllib hook
        return None


def is_short(video_id: str, opener: Any = None, timeout: float = 10) -> bool | None:
    """Ask youtube.com whether a video is a Short (no API quota).

    ``/shorts/<id>`` answers 200 for a Short and redirects to ``/watch`` for anything else.
    Returns None when the answer is unclear (network error, unexpected status).
    """
    opener = opener or urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(SHORTS_URL.format(video_id), method="HEAD")
    try:
        with opener.open(request, timeout=timeout) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:  # redirects surface as HTTPError without a handler
        status = exc.code
    except Exception as exc:  # noqa: BLE001
        log.debug("shorts check for %s failed: %s", video_id, exc)
        return None
    if status == 200:
        return True
    if 300 <= status < 400:
        return False
    return None


def detect_shorts(
    store: VideoStore, *, workers: int = 8, recheck: bool = False, opener: Any = None
) -> dict[str, int]:
    """Fill the ``is_short`` flag for stored videos. Returns counts per outcome."""
    ids = store.video_ids() if recheck else store.video_ids_without_short_flag()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        flags = dict(zip(ids, pool.map(lambda v: is_short(v, opener), ids), strict=True))
    store.set_short_flags({k: v for k, v in flags.items() if v is not None})
    values = list(flags.values())
    return {"shorts": values.count(True), "not_shorts": values.count(False),
            "unknown": values.count(None)}  # fmt: skip
