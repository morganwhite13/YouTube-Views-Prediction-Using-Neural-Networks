"""Feature engineering.

Design rules (each fixes a v1 problem):

* **Only use information available when the video is published** (plus its age). Likes and
  comments are outcomes, not inputs, so they are never features.
* **Age is measured at the time views were fetched**, not at training time. v1 computed
  ``today - publishedAt`` when the model was trained, so cached data silently got older.
* **Fit every statistic on training rows only** (scalers, vocabularies, text encoders). v1 fitted
  tokenizers and scalers on the full dataset before splitting.
* **Channel history is past-only**: a video's "channel prior" is computed from that channel's
  *earlier* uploads, which is exactly what a creator knows before publishing.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .text import clean_text, make_encoder

HISTORY_WINDOW = 10
LANGUAGES = ("en", "es", "pt", "hi", "de", "fr", "ja", "ko", "ru", "id")
# Channel-level view features that are estimated from subscribers when missing.
SIZE_IMPUTED = ("hist_median", "hist_mean", "hist_max", "log_channel_avg_views")
MIN_AGE_DAYS = 1.0 / 24.0  # one hour
# Fixed-date holidays observed across most of YouTube's audience (month, day). Moving and
# country-specific holidays are left to the time-of-year features.
HOLIDAYS = ((1, 1), (2, 14), (10, 31), (12, 24), (12, 25), (12, 31))
HOLIDAY_WINDOW_DAYS = 3  # "near" = published within this many days of one

_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff]")
_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|i've|my|we|our)\b", re.IGNORECASE)
_VS_RE = re.compile(r"\bvs\.?\b", re.IGNORECASE)

NUMERIC_FEATURES = [
    # timing
    "log_age_days", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # season: time of year and the days around widely observed holidays
    "doy_sin", "doy_cos", "near_holiday",
    # channel
    "log_subscribers", "subscribers_missing", "log_channel_videos", "channel_videos_missing",
    "log_channel_avg_views", "channel_avg_missing",
    # channel history (past uploads)
    "hist_median", "hist_mean", "hist_std", "hist_max", "hist_log_age", "log_hist_count",
    "has_history", "log_gap_days",
    # format
    "log_duration", "duration_missing", "is_short", "short_known", "is_long", "caption", "is_hd",
    # language
    "lang_en", "lang_es", "lang_pt", "lang_hi", "lang_de", "lang_fr", "lang_ja", "lang_ko",
    "lang_ru", "lang_id", "lang_other", "lang_missing",
    # title style
    "title_chars", "title_words", "title_caps_ratio", "title_has_digit", "title_has_question",
    "title_has_exclaim", "title_emoji", "title_has_brackets", "title_has_separator",
    "title_has_vs", "title_has_money", "title_first_person", "title_has_hashtag",
    # description / tags
    "desc_log_chars", "desc_links", "desc_hashtags", "n_tags",
]  # fmt: skip

# Groups used for permutation importance (name -> numeric columns); text/category are added
# separately because they are separate model inputs.
FEATURE_GROUPS: dict[str, list[str]] = {
    "video age": ["log_age_days"],
    "publish time": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "near_holiday",
    ],
    "channel size": [
        "log_subscribers", "subscribers_missing", "log_channel_videos", "channel_videos_missing",
        "log_channel_avg_views", "channel_avg_missing",
    ],
    "channel history": [
        "hist_median", "hist_mean", "hist_std", "hist_max", "hist_log_age", "log_hist_count",
        "has_history", "log_gap_days",
    ],
    "format": [
        "log_duration", "duration_missing", "is_short", "short_known", "is_long", "caption",
        "is_hd",
    ],
    "language": [c for c in NUMERIC_FEATURES if c.startswith("lang_")],
    "title style": [c for c in NUMERIC_FEATURES if c.startswith("title_")],
    "description stats": ["desc_log_chars", "desc_links", "desc_hashtags", "n_tags"],
}  # fmt: skip


def _fmt_count(log_value: float) -> str:
    n = float(np.expm1(log_value))
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def _fmt_days(log_value: float) -> str:
    d = float(np.expm1(log_value))
    return f"{d * 24:.0f} hours" if d < 1 else f"{d:.0f} days"


# Inputs checked against the training range (see FeatureBuilder.range_notes):
# column -> (label, formatter of the log1p value).
RANGE_CHECKS: dict[str, tuple[str, Any]] = {
    "log_age_days": ("video age", _fmt_days),
    "log_subscribers": ("subscriber count", _fmt_count),
    "hist_median": ("typical views of recent uploads", _fmt_count),
    "log_duration": ("length in seconds", _fmt_count),
    "log_channel_videos": ("channel video count", _fmt_count),
}


def log_views(views: Any) -> Any:
    return np.log1p(np.clip(views, 0, None))


def age_days(df: pd.DataFrame) -> pd.Series:
    age = (df["fetched_at"] - df["published_at"]).dt.total_seconds() / 86400.0
    return age.clip(lower=MIN_AGE_DAYS)


def _history_as_of_publish(df: pd.DataFrame, latest: pd.DataFrame, window: int) -> pd.DataFrame:
    """``hist_*`` for each video from the channel's previous ``window`` uploads, each counted
    *as it was when this video went up*.

    At prediction time a creator knows their recent uploads' views today, i.e. at publish time.
    So for training, each earlier upload contributes its latest snapshot taken on or before the
    video's publish time (and the age it had then); only when none is that old does the earliest
    snapshot stand in, the least "future" count available. With one snapshot per video this is
    simply that snapshot.
    """
    snaps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ordered = df.dropna(subset=["fetched_at"]).sort_values("fetched_at")
    for vid, g in ordered.groupby("video_id", sort=False):
        fetched = g["fetched_at"].to_numpy("datetime64[ns]")
        snaps[vid] = (fetched, log_views(g["views"].to_numpy(float)),
                      g["published_at"].iloc[0].to_datetime64())  # fmt: skip
    day = np.timedelta64(1, "D")
    cols = {k: [] for k in ("video_id", "hist_median", "hist_mean", "hist_std", "hist_max",
                            "hist_log_age")}  # fmt: skip
    for _, g in latest.groupby("channel_id", sort=False):
        ids = g["video_id"].tolist()
        pubs = g["published_at"].to_numpy("datetime64[ns]")
        for i, vid in enumerate(ids):
            lv, la = [], []
            for u in ids[max(0, i - window) : i]:
                fetched, views, published = snaps[u]
                k = np.searchsorted(fetched, pubs[i], side="right") - 1
                k = max(k, 0)  # nothing that old: the earliest snapshot
                lv.append(views[k])
                la.append(np.log1p(max((fetched[k] - published) / day, MIN_AGE_DAYS)))
            cols["video_id"].append(vid)
            if lv:
                a = np.asarray(lv)
                cols["hist_median"].append(float(np.median(a)))
                cols["hist_mean"].append(float(a.mean()))
                cols["hist_std"].append(float(a.std(ddof=1)) if len(a) > 1 else np.nan)
                cols["hist_max"].append(float(a.max()))
                cols["hist_log_age"].append(float(np.median(la)))
            else:
                for k in ("hist_median", "hist_mean", "hist_std", "hist_max", "hist_log_age"):
                    cols[k].append(np.nan)
    return pd.DataFrame(cols)


def add_channel_history(df: pd.DataFrame, window: int = HISTORY_WINDOW) -> pd.DataFrame:
    """Add ``hist_*`` columns describing each channel's earlier uploads (training data only).

    Works with multiple snapshots per video: each earlier upload is counted as it was when the
    video was published (see :func:`_history_as_of_publish`), merged back by ``video_id``.
    """
    df = df.copy()
    latest = (
        df.sort_values("fetched_at")
        .drop_duplicates("video_id", keep="last")
        .sort_values(["channel_id", "published_at"])
        .copy()
    )
    hist = _history_as_of_publish(df, latest, window)
    grouped = latest.groupby("channel_id", sort=False)
    # days since the channel's previous upload: uploading again soon may split the audience,
    # a long break may lose it
    hist["gap_days"] = (grouped["published_at"].diff().dt.total_seconds() / 86400).to_numpy()
    hist["hist_count"] = grouped.cumcount().clip(upper=window).astype(float).to_numpy()
    df = df.drop(columns=[c for c in hist.columns if c != "video_id" and c in df], errors="ignore")
    df = df.merge(hist, on="video_id", how="left")

    # Channel average views *excluding this video* (the channel total includes it, which would
    # otherwise leak the target for small channels). Subtract the video's own views *at the time
    # of the channel total*: its snapshot nearest to ``channel_fetched_at`` when the store
    # recorded which channel snapshot the row uses, otherwise its latest (the total is then the
    # channel's latest). Subtracting an earlier count would leave the video's later growth in
    # "the other videos", i.e. its own future views in a feature.
    vids = df["channel_video_count"]
    newest = df.sort_values("fetched_at").drop_duplicates("video_id", keep="last")
    own = df["video_id"].map(newest.set_index("video_id")["views"])
    if "channel_fetched_at" in df and df["channel_fetched_at"].notna().any():
        own = own.copy()
        rows = df["channel_fetched_at"].notna().to_numpy()
        query = pd.DataFrame({
            "video_id": df.loc[rows, "video_id"].to_numpy(),
            "at": df.loc[rows, "channel_fetched_at"].astype("datetime64[ns, UTC]").to_numpy(),
            "_row": np.flatnonzero(rows),
        }).sort_values("at")  # fmt: skip
        snaps = pd.DataFrame({
            "video_id": df["video_id"].to_numpy(),
            "at": df["fetched_at"].astype("datetime64[ns, UTC]").to_numpy(),
            "_views": df["views"].to_numpy(float),
        }).dropna(subset=["at"]).sort_values("at")  # fmt: skip
        nearest = pd.merge_asof(query, snaps, on="at", by="video_id", direction="nearest")
        own.iloc[nearest["_row"].to_numpy()] = nearest["_views"].to_numpy()
    others = (df["channel_views"] - own).clip(lower=0)
    df["channel_avg_views"] = np.where(vids > 1, others / (vids - 1).clip(lower=1), np.nan)
    df["channel_video_count"] = _video_count_at_publish(df, latest)
    return df


def _video_count_at_publish(df: pd.DataFrame, latest: pd.DataFrame) -> pd.Series:
    """The channel's video count when each video went up (counting the video itself).

    The stored count is from a channel fetch, usually later, so it includes uploads made after
    the video; at prediction time the count is today's, i.e. at publishing. Subtract the
    channel's stored uploads published after the video and up to that fetch. Collection keeps
    each channel's latest uploads, so the recent ones (most training rows) are all there.
    """
    count = df["channel_video_count"].astype(float)
    if count.isna().all():
        return count
    when = df["channel_fetched_at"] if "channel_fetched_at" in df else df["fetched_at"]
    when = when.fillna(df["fetched_at"]).to_numpy("datetime64[ns]")
    published = df["published_at"].to_numpy("datetime64[ns]")
    later = np.zeros(len(df))
    pubs = {c: np.sort(g.to_numpy("datetime64[ns]"))
            for c, g in latest.groupby("channel_id")["published_at"]}  # fmt: skip
    channel = df["channel_id"].to_numpy()
    for c, times in pubs.items():
        rows = np.flatnonzero(channel == c)
        if not len(rows):
            continue
        after = np.searchsorted(times, published[rows], side="right")
        upto = np.searchsorted(times, when[rows], side="right")
        later[rows] = np.clip(upto - after, 0, None)
    return (count - later).clip(lower=1).where(count.notna())


HISTORY_COLUMNS = [
    "hist_median", "hist_mean", "hist_std", "hist_max", "hist_log_age", "gap_days",
]  # fmt: skip


def drop_channel_context(df: pd.DataFrame, fraction: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Append copies of a random ``fraction`` of rows with the optional channel context removed.

    At prediction time users often know only their subscriber count. Without this augmentation
    nearly every training row has channel history, so such inputs are out of distribution and
    the model under-reacts to subscribers.
    """
    if fraction <= 0:
        return df
    rng = np.random.default_rng(seed)
    videos = df["video_id"].unique()
    chosen = set(rng.choice(videos, size=int(len(videos) * fraction), replace=False))
    extra = df[df["video_id"].isin(chosen)].copy()
    extra[HISTORY_COLUMNS] = np.nan
    extra["hist_count"] = 0.0
    extra["channel_avg_views"] = np.nan
    extra["channel_video_count"] = np.nan
    extra["_augmented"] = True
    return pd.concat([df, extra], ignore_index=True)


def drop_thumbnails(
    df: pd.DataFrame, fraction: float = 0.15, seed: int = 0, enabled: bool = True
) -> pd.DataFrame:
    """Append copies of a random ``fraction`` of rows without their thumbnail.

    Teaches the model what to do when a thumbnail isn't supplied at prediction time.
    """
    if not enabled or fraction <= 0 or not has_thumbnails(df):
        return df
    rng = np.random.default_rng(seed + 1)
    videos = df["video_id"].unique()
    chosen = set(rng.choice(videos, size=int(len(videos) * fraction), replace=False))
    extra = df[df["video_id"].isin(chosen)].copy()
    extra["thumbnail_path"] = None
    extra["_augmented"] = True
    return pd.concat([df, extra], ignore_index=True)


def history_from_recent(
    recent_views: list[float] | None, recent_age_days: list[float] | None = None
) -> dict[str, float]:
    """Compute ``hist_*`` values at prediction time from a channel's most recent uploads."""
    views = np.asarray(recent_views or [], dtype=float)[:HISTORY_WINDOW]
    if views.size == 0:
        return dict.fromkeys(HISTORY_COLUMNS, np.nan) | {"hist_count": 0.0}
    lv = log_views(views)
    ages = np.asarray(recent_age_days or [], dtype=float)[:HISTORY_WINDOW]
    return {
        "hist_median": float(np.median(lv)),
        "hist_mean": float(lv.mean()),
        "hist_std": float(lv.std(ddof=1)) if lv.size > 1 else np.nan,
        "hist_max": float(lv.max()),
        "hist_log_age": float(np.median(np.log1p(ages))) if ages.size else np.nan,
        "hist_count": float(lv.size),
        "gap_days": np.nan,  # set by VideoInput.to_row, which knows the publish time
    }


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(float)


def _days_to_holiday(published: pd.Series) -> pd.Series:
    """Days from each (UTC) publish date to the nearest holiday in :data:`HOLIDAYS`."""
    day = published.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    best = pd.Series(np.inf, index=published.index)
    for offset in (-1, 0, 1):  # the nearest one may fall in the previous or next year
        for month, dom in HOLIDAYS:
            when = pd.to_datetime(
                {"year": day.dt.year + offset, "month": month, "day": dom}, errors="coerce"
            )
            best = np.minimum(best, (when - day).dt.days.abs().astype(float))
    return best.where(published.notna())


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the raw (unscaled, possibly NaN) numeric feature matrix."""
    out = pd.DataFrame(index=df.index)
    out["log_age_days"] = np.log1p(age_days(df))
    published = df["published_at"]
    hour = published.dt.hour + published.dt.minute / 60.0
    dow = published.dt.day_of_week
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    doy = (published.dt.day_of_year - 1 + hour / 24.0) / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)
    out["near_holiday"] = _bool(_days_to_holiday(published) <= HOLIDAY_WINDOW_DAYS)

    subs = df["subscribers"].where(df["subscribers"] > 0)
    out["log_subscribers"] = np.log1p(subs)
    out["subscribers_missing"] = _bool(subs.isna())
    out["log_channel_videos"] = np.log1p(df["channel_video_count"])
    out["channel_videos_missing"] = _bool(df["channel_video_count"].isna())
    avg = df.get("channel_avg_views", pd.Series(np.nan, index=df.index))
    out["log_channel_avg_views"] = np.log1p(avg)
    out["channel_avg_missing"] = _bool(avg.isna())

    for col in ("hist_median", "hist_mean", "hist_std", "hist_max", "hist_log_age"):
        out[col] = df.get(col, np.nan)
    count = df.get("hist_count", pd.Series(0.0, index=df.index)).fillna(0.0)
    out["log_hist_count"] = np.log1p(count)
    out["has_history"] = _bool(count > 0)
    gap = df.get("gap_days", pd.Series(np.nan, index=df.index)).astype(float)
    out["log_gap_days"] = np.log1p(gap.clip(lower=0))

    dur = df["duration_s"]
    out["log_duration"] = np.log1p(dur)
    out["duration_missing"] = _bool(dur.isna())
    # A detected Shorts flag (``ytviews shorts``) beats the length rule when available.
    detected = df["is_short"] if "is_short" in df else pd.Series(np.nan, index=df.index)
    out["is_short"] = np.where(detected.notna(), detected, (dur <= 180).astype(float))
    out["short_known"] = _bool(detected.notna())
    out["is_long"] = _bool(dur >= 20 * 60)
    out["caption"] = df["caption"].fillna(0.0)
    out["is_hd"] = _bool(df["definition"].fillna("hd") == "hd")

    title = df["title"].fillna("")
    letters = title.str.count(r"[A-Za-z]").clip(lower=1)
    out["title_chars"] = title.str.len()
    out["title_words"] = title.str.split().str.len().fillna(0)
    out["title_caps_ratio"] = title.str.count(r"[A-Z]") / letters
    out["title_has_digit"] = _bool(title.str.contains(r"\d"))
    out["title_has_question"] = _bool(title.str.contains(r"\?"))
    out["title_has_exclaim"] = _bool(title.str.contains("!"))
    out["title_emoji"] = title.str.count(_EMOJI_RE)
    out["title_has_brackets"] = _bool(title.str.contains(r"[\(\[]"))
    out["title_has_separator"] = _bool(title.str.contains(r"[|:\-–]"))
    out["title_has_vs"] = _bool(title.str.contains(_VS_RE))
    out["title_has_money"] = _bool(title.str.contains(r"[$€£]"))
    out["title_first_person"] = _bool(title.str.contains(_FIRST_PERSON_RE))
    out["title_has_hashtag"] = _bool(title.str.contains("#"))

    lang = (
        df["default_language"].fillna("").astype(str).str.lower().str.split("-").str[0]
        if "default_language" in df
        else pd.Series("", index=df.index)
    )
    for code in LANGUAGES:
        out[f"lang_{code}"] = _bool(lang == code)
    out["lang_other"] = _bool((lang != "") & ~lang.isin(LANGUAGES))
    out["lang_missing"] = _bool(lang == "")

    desc = df["description"].fillna("")
    out["desc_log_chars"] = np.log1p(desc.str.len())
    out["desc_links"] = desc.str.count(r"https?://")
    out["desc_hashtags"] = desc.str.count(r"#\w+")
    tags = df["tags"].fillna("")
    out["n_tags"] = np.where(tags.str.len() > 0, tags.str.count(r"\|") + 1, 0)
    return out[NUMERIC_FEATURES].astype(float)


def description_text(df: pd.DataFrame) -> list[str]:
    tags = df["tags"].fillna("").str.replace("|", ", ", regex=False)
    return [
        clean_text(f"{d} {t}", max_chars=1000)
        for d, t in zip(df["description"].fillna(""), tags, strict=True)
    ]


def title_text(df: pd.DataFrame) -> list[str]:
    return [clean_text(t) for t in df["title"].fillna("")]


def _safe_std(x: np.ndarray) -> np.ndarray:
    """Column std for standardising, with near-constant columns left unscaled.

    A column that never varied in training (e.g. face count when no training thumbnail had a
    face) would otherwise blow any new value up to the clip limit.
    """
    std = x.std(axis=0)
    return np.where(std < 1e-3, 1.0, std)


def _image_features(encoder: Any, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    paths = df["thumbnail_path"].tolist() if "thumbnail_path" in df else [None] * len(df)
    if getattr(encoder, "reads_text", False):  # OCR features compare thumbnail text to titles
        return encoder.transform(paths, df["title"].fillna("").astype(str).tolist())
    return encoder.transform(paths)


def has_thumbnails(df: pd.DataFrame) -> bool:
    return "thumbnail_path" in df and df["thumbnail_path"].notna().any()


class FeatureBuilder:
    """Fit-on-train / transform-anywhere feature pipeline producing the model's inputs.

    Inputs produced: ``numeric`` (standardised), ``category`` (int ids) and one dense embedding
    per field: ``title``, ``description`` and, when thumbnails are available, ``thumbnail``
    (image features plus a final "thumbnail missing" flag column).
    """

    def __init__(
        self,
        text_encoder: str = "auto",
        text_dim: int = 128,
        image_encoder: str = "auto",
        target: str = "views",
    ) -> None:
        self.text_encoder = text_encoder
        self.text_dim = text_dim
        self.image_encoder = image_encoder  # auto | clip | stats | none (+ocr)
        self.target = target  # what a fine-tuned title encoder learns to explain
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        num = numeric_frame(df)
        self.medians_ = num.median().fillna(0.0)
        # Missing channel-level views are estimated from subscribers rather than filled with
        # the global median: "no history" must not read as "a typical-sized channel", or a
        # subscribers-only prediction barely reacts to channel size.
        self.size_imputers_: dict[str, tuple[float, float]] = {}
        subs = num["log_subscribers"]
        for col in SIZE_IMPUTED:
            both = num[col].notna() & subs.notna()
            if both.sum() >= 20 and subs[both].std() > 0:
                slope, intercept = np.polyfit(subs[both], num.loc[both, col], 1)
                self.size_imputers_[col] = (float(intercept), float(slope))
        # Where the training data actually was, for the inputs people most often push past it.
        self.ranges_ = {
            col: (float(num[col].quantile(0.005)), float(num[col].quantile(0.995)))
            for col in RANGE_CHECKS
            if num[col].notna().sum() >= 20
        }
        filled = self._impute(num)
        self.means_ = filled.mean()
        self.stds_ = filled.std().replace(0, 1.0).fillna(1.0)
        cats = sorted(c for c in df["category_id"].astype(str).unique() if c)
        self.categories_ = {c: i + 1 for i, c in enumerate(cats)}  # 0 = unknown
        self.feature_names_ = list(NUMERIC_FEATURES)
        self.category_counts_ = df["category_id"].astype(str).value_counts().to_dict()
        self.title_encoder_ = make_encoder(self.text_encoder, self.text_dim)
        if getattr(self.title_encoder_, "needs_targets", False):
            # Fine-tune on what the title has to explain: log views beyond a linear model of
            # everything else (channel, timing, format), so it learns titles, not channel size.
            self.title_encoder_.fit(title_text(df), self._residual_target(df, filled))
            desc_kind = "sbert"  # descriptions use the frozen base model
        else:
            self.title_encoder_.fit(title_text(df))
            desc_kind = self.text_encoder
        self.desc_encoder_ = make_encoder(desc_kind, self.text_dim).fit(description_text(df))
        self.embed_stats_ = {}
        for key, emb in self._encode_text(df).items():
            self.embed_stats_[key] = (emb.mean(axis=0), _safe_std(emb))

        self.image_encoder_ = None
        if self.image_encoder != "none" and has_thumbnails(df):
            from .images import make_image_encoder

            enc = make_image_encoder(self.image_encoder)
            emb, missing = _image_features(enc, df)
            if (~missing).sum() >= 2:
                self.image_encoder_ = enc
                present = emb[~missing]
                self.embed_stats_["thumbnail"] = (present.mean(axis=0), _safe_std(present))
        self.fitted = True
        return self

    @property
    def age_stats(self) -> tuple[float, float]:
        """(mean, std) used to standardise the log1p(age) column."""
        return float(self.means_["log_age_days"]), float(self.stds_["log_age_days"])

    @property
    def age_floor(self) -> float | None:
        """The youngest training age (0.5th percentile), standardised like the model input."""
        lo = getattr(self, "ranges_", {}).get("log_age_days")
        if lo is None:
            return None
        mean, std = self.age_stats
        return float((lo[0] - mean) / std)

    @property
    def n_numeric(self) -> int:
        return len(NUMERIC_FEATURES)

    @property
    def n_categories(self) -> int:
        return len(self.categories_) + 1

    @property
    def uses_thumbnails(self) -> bool:
        return getattr(self, "image_encoder_", None) is not None

    @property
    def embed_dims(self) -> dict[str, int]:
        dims = {
            "title": self.title_encoder_.output_dim,
            "description": self.desc_encoder_.output_dim,
        }
        if self.uses_thumbnails:
            dims["thumbnail"] = self.image_encoder_.output_dim + 1  # + missing flag
        return dims

    @property
    def encoder_kind(self) -> str:
        return self.title_encoder_.kind

    @property
    def image_kind(self) -> str:
        return self.image_encoder_.kind if self.uses_thumbnails else "none"

    def range_notes(self, df: pd.DataFrame) -> list[list[str]]:
        """Per row, a note for each key input outside the range the model was trained on.

        The model still answers, but beyond its training data it can only extrapolate, and the
        number deserves less trust than its range suggests. Models saved before this existed
        give no notes.
        """
        ranges = getattr(self, "ranges_", {})
        notes: list[list[str]] = [[] for _ in range(len(df))]
        if not ranges:
            return notes
        num = numeric_frame(df)
        for col, (lo, hi) in ranges.items():
            label, fmt = RANGE_CHECKS[col]
            for i, value in enumerate(num[col].to_numpy()):
                if np.isnan(value) or lo <= value <= hi:
                    continue
                side, bound = ("below", lo) if value < lo else ("above", hi)
                notes[i].append(
                    f"{label} ({fmt(value)}) is {side} anything in the training data "
                    f"({'min' if side == 'below' else 'max'} about {fmt(bound)}), so this "
                    "prediction is an extrapolation"
                )
        return notes

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("FeatureBuilder must be fitted first")
        if getattr(self, "feature_names_", None) != NUMERIC_FEATURES:
            raise RuntimeError(
                "This model was trained with an older version of ytviews that used different "
                "features. Retrain it with `ytviews train`."
            )
        num = self._impute(numeric_frame(df))
        num = ((num - self.means_) / self.stds_).clip(-8, 8)
        cats = df["category_id"].astype(str).map(self.categories_).fillna(0).astype(np.int64)
        out = {"numeric": num.to_numpy(np.float32), "category": cats.to_numpy()}
        for key, emb in self._encode_text(df).items():
            out[key] = self._standardise(key, emb)
        if self.uses_thumbnails:
            emb, missing = _image_features(self.image_encoder_, df)
            emb = self._standardise("thumbnail", emb)
            emb[missing] = 0.0  # missing thumbnail -> "average thumbnail" + flag
            out["thumbnail"] = np.concatenate([emb, missing[:, None].astype(np.float32)], axis=1)
        return out

    def _residual_target(self, df: pd.DataFrame, filled: pd.DataFrame) -> np.ndarray:
        from sklearn.linear_model import Ridge

        y = log_views(df[getattr(self, "target", "views")].to_numpy(float))
        known = np.isfinite(y)  # hidden like/comment counts
        x = ((filled - self.means_) / self.stds_).to_numpy()
        residual = np.zeros_like(y)
        residual[known] = y[known] - Ridge(alpha=1.0).fit(x[known], y[known]).predict(x[known])
        return residual

    def _impute(self, num: pd.DataFrame) -> pd.DataFrame:
        num = num.copy()
        subs = num["log_subscribers"]
        for col, (intercept, slope) in getattr(self, "size_imputers_", {}).items():
            missing = num[col].isna() & subs.notna()
            num.loc[missing, col] = intercept + slope * subs[missing]
        return num.fillna(self.medians_)

    def _standardise(self, key: str, emb: np.ndarray) -> np.ndarray:
        mean, std = self.embed_stats_[key]
        return ((emb - mean) / std).clip(-8, 8).astype(np.float32)

    def _encode_text(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "title": self.title_encoder_.transform(title_text(df)),
            "description": self.desc_encoder_.transform(description_text(df)),
        }

    def flat(self, inputs: dict[str, np.ndarray], one_hot: bool = True) -> np.ndarray:
        """Concatenate model inputs into one matrix (for the classical baselines)."""
        parts = [inputs["numeric"]] + [inputs[k] for k in sorted(self.embed_dims)]
        if one_hot:
            parts.append(np.eye(self.n_categories, dtype=np.float32)[inputs["category"]])
        else:
            parts.append(inputs["category"][:, None].astype(np.float32))
        return np.concatenate(parts, axis=1)
