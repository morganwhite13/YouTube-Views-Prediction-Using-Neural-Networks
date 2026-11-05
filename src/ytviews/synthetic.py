"""Synthetic YouTube-like data with a known generating process.

Used for the offline demo and the test-suite: it exercises the exact same storage / feature /
training code paths as real data, and because the ground truth is known we can check that the
model actually learns the effects we planted (channel size, channel quality, title wording,
format, thumbnail design, publish time, age-driven growth that differs per video).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .images import thumbnail_dir_for
from .store import VideoStore

CATEGORIES = {  # id: (name, effect on log views, topic words)
    "20": ("Gaming", 0.2, ["minecraft", "fortnite", "speedrun", "boss", "build", "survival"]),
    "24": ("Entertainment", 0.3, ["prank", "reacting", "mystery", "box", "giant", "world"]),
    "28": ("Science & Technology", 0.0, ["iphone", "robot", "laptop", "ai", "battery", "chip"]),
    "26": ("Howto & Style", -0.2, ["recipe", "makeup", "garden", "room", "budget", "diy"]),
    "27": ("Education", -0.3, ["history", "math", "physics", "space", "brain", "ocean"]),
    "10": ("Music", 0.1, ["song", "cover", "remix", "guitar", "piano", "beat"]),
    "17": ("Sports", 0.0, ["goal", "dunk", "workout", "match", "trickshot", "race"]),
    "22": ("People & Blogs", -0.4, ["day", "morning", "routine", "family", "life", "trip"]),
}
HOOKS = {  # wording patterns and their true effect on log views
    "I Survived 100 Days in {t}": 0.9,
    "$1 vs $10,000 {t}": 0.8,
    "I Built the World's Biggest {t}": 0.7,
    "{t} Challenge (GONE WRONG)": 0.5,
    "Testing Viral {t} Hacks": 0.4,
    "The Truth About {t}": 0.3,
    "Why {t} Is Insane": 0.3,
    "{t} Tier List": 0.1,
    "How to {t} for Beginners": 0.0,
    "My Honest {t} Review": -0.1,
    "{t} Tutorial - Part 3": -0.4,
    "Weekly {t} Update": -0.6,
    "{t} Livestream Highlights": -0.7,
    "Q&A + {t} Vlog": -0.8,
}
FILLER = "in this video we take a look at {t} and more. thanks for watching, subscribe for more!"


def _timing_effect(published: datetime) -> float:
    """Weak publish-time effect: weekends and US afternoons/evenings (UTC 14-22h) do better."""
    effect = 0.15 if published.weekday() >= 5 else 0.0
    if 14 <= published.hour <= 22:
        effect += 0.2
    elif published.hour <= 6:
        effect -= 0.15
    return effect


def generate_records(
    n_channels: int = 120,
    videos_per_channel: tuple[int, int] = (12, 40),
    snapshots: int = 2,
    seed: int = 0,
    now: datetime | None = None,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    # Engagement draws use their own generator, so adding them left the views data unchanged.
    erng = np.random.default_rng([seed, 1])
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    cat_ids = list(CATEGORIES)
    hook_items = list(HOOKS.items())
    videos: list[dict] = []
    channels: list[dict] = []

    for c in range(n_channels):
        cid = f"UCsynthetic{c:05d}"
        cat = rng.choice(cat_ids)
        _, cat_effect, topics = CATEGORIES[cat]
        subs = float(np.clip(rng.lognormal(np.log(60_000), 2.0), 100, 2e8))
        quality = rng.normal(0, 0.6)
        style = rng.normal(0, 1.0, size=len(hook_items))  # each channel favours some hooks
        shorts_share = rng.uniform(0, 0.5)
        thumb_skill = rng.normal(0, 0.7)  # some channels make better thumbnails
        like_base = float(erng.lognormal(np.log(0.03), 0.4))
        comment_base = float(erng.lognormal(np.log(0.002), 0.6))
        language = rng.choice(["en"] * 8 + ["es", "pt", "hi-IN", "de"])
        n_videos = int(rng.integers(*videos_per_channel))
        cadence = rng.uniform(2, 21)  # days between uploads
        start = now - timedelta(days=float(rng.uniform(30, 1000)))
        total_views = 0.0
        for v in range(n_videos):
            published = start + timedelta(days=v * cadence * rng.uniform(0.6, 1.4))
            if published >= now - timedelta(hours=6):
                break
            published = published.replace(
                hour=int(rng.integers(0, 24)), minute=int(rng.integers(0, 60)), second=0
            )
            probs = np.exp(style - style.max())
            hook, hook_effect = hook_items[rng.choice(len(hook_items), p=probs / probs.sum())]
            topic = rng.choice(topics)
            title = hook.format(t=topic.title())
            if rng.random() < 0.3:
                title = title.upper()
            is_short = rng.random() < shorts_share
            duration = rng.uniform(15, 60) if is_short else float(rng.lognormal(np.log(600), 0.6))
            if is_short:
                title += " #shorts"
            thumb_quality = float(thumb_skill + rng.normal(0, 0.7))
            potential = (
                1.6
                + 0.8 * np.log(subs)
                + quality
                + cat_effect
                + hook_effect
                + (0.3 if is_short else 0.0)
                + 0.35 * thumb_quality
                + _timing_effect(published)
                + rng.normal(0, 0.55)
            )
            drift = rng.normal(0, 0.3)  # evergreen (>0) vs fading (<0) videos
            # Engagement: likes/comments per view vary by channel and by hook style.
            like_rate = like_base * np.exp(0.3 * hook_effect + erng.normal(0, 0.3))
            comment_rate = comment_base * np.exp(erng.normal(0, 0.4))
            likes_hidden = erng.random() < 0.05
            tags = [topic, CATEGORIES[cat][0].lower(), f"{topic} {rng.choice(['tips', 'fun'])}"]
            base = {
                "video_id": f"vid{c:05d}{v:04d}",
                "channel_id": cid,
                "channel_title": f"Synthetic Channel {c}",
                "title": title,
                "description": FILLER.format(t=topic) + f" https://example.com/{topic}",
                "tags": "|".join(tags),
                "category_id": str(cat),
                "published_at": published.isoformat(),
                "default_language": str(language),
                "live_broadcast": "none",
                "duration_s": round(duration),
                "definition": "hd" if rng.random() < 0.9 else "sd",
                "caption": int(rng.random() < 0.3),
                "_thumb": (thumb_quality, float(rng.random()), int(rng.integers(1 << 30))),
                "_short": bool(is_short),
            }
            for s in range(snapshots):
                fetched = now - timedelta(days=45 * s)
                age = (fetched - published).total_seconds() / 86400
                if age < 1 / 24:
                    continue
                growth = (1 - np.exp(-age / 4.0)) * (1 + 0.25 * np.log1p(age / 30))
                growth *= np.exp(drift * np.log1p(age / 30))
                views = int(np.exp(potential) * growth)
                engagement = {
                    "likes": None if likes_hidden else int(views * like_rate),
                    "comments": int(views * comment_rate),
                }
                videos.append(
                    base | engagement | {"views": views, "fetched_at": fetched.isoformat()}
                )
                if s == 0:
                    total_views += views
        channels.append(
            {
                "channel_id": cid,
                "channel_title": f"Synthetic Channel {c}",
                "subscribers": int(subs),
                "channel_views": int(total_views * 1.5),
                "channel_video_count": n_videos + int(rng.integers(0, 50)),
                "fetched_at": now.isoformat(),
            }
        )
    return videos, channels


def render_thumbnail(quality: float, hue: float, seed: int, path: Path) -> None:
    """Draw a 128x72 thumbnail whose design gets "better" (brighter, more saturated, higher
    contrast, a clear central subject) as ``quality`` increases."""
    import colorsys

    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    q = 1 / (1 + np.exp(-1.5 * quality))  # 0..1
    bg = colorsys.hsv_to_rgb(hue, 0.15 + 0.75 * q, 0.25 + 0.6 * q)
    img = Image.new("RGB", (128, 72), tuple(int(255 * c) for c in bg))
    draw = ImageDraw.Draw(img)
    for _ in range(int(12 * (1 - q)) + 2):  # clutter for weak thumbnails
        x, y = rng.integers(0, 128), rng.integers(0, 72)
        grey = int(rng.integers(60, 160))
        draw.rectangle([x, y, x + 14, y + 8], fill=(grey, grey, grey))
    subject = colorsys.hsv_to_rgb((hue + 0.5) % 1.0, 0.3 + 0.7 * q, 0.4 + 0.6 * q)
    r = int(10 + 16 * q)
    draw.ellipse([64 - r, 36 - r, 64 + r, 36 + r], fill=tuple(int(255 * c) for c in subject))
    if q > 0.5:
        draw.rectangle([6, 54, 6 + int(70 * q), 66], fill=(255, 255, 255))  # bold caption
    img.save(path, quality=85)


def write_synthetic_db(path: str | Path, thumbnails: bool = True, **kwargs) -> dict[str, int]:
    videos, channels = generate_records(**kwargs)
    if thumbnails:
        folder = thumbnail_dir_for(path)
        folder.mkdir(parents=True, exist_ok=True)
        done = set()
        for v in videos:
            if v["video_id"] not in done:
                render_thumbnail(*v["_thumb"], folder / f"{v['video_id']}.jpg")
                done.add(v["video_id"])
    with VideoStore(path) as store:
        store.upsert_channels(channels)
        # Insert oldest snapshots first so "latest" matches the most recent fetch.
        store.upsert_videos(sorted(videos, key=lambda r: r["fetched_at"]), source="synthetic")
        store.set_short_flags({v["video_id"]: v["_short"] for v in videos})
        return store.counts()
