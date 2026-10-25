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

_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff]")
_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|i've|my|we|our)\b", re.IGNORECASE)
_VS_RE = re.compile(r"\bvs\.?\b", re.IGNORECASE)

NUMERIC_FEATURES = [
    # timing
    "log_age_days", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # channel
    "log_subscribers", "subscribers_missing", "log_channel_videos", "channel_videos_missing",
    "log_channel_avg_views", "channel_avg_missing",
    # channel history (past uploads)
    "hist_median", "hist_mean", "hist_std", "hist_max", "hist_log_age", "log_hist_count",
    "has_history",
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
    "publish time": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    "channel size": [
        "log_subscribers", "subscribers_missing", "log_channel_videos", "channel_videos_missing",
        "log_channel_avg_views", "channel_avg_missing",
    ],
    "channel history": [
        "hist_median", "hist_mean", "hist_std", "hist_max", "hist_log_age", "log_hist_count",
        "has_history",
    ],
    "format": [
        "log_duration", "duration_missing", "is_short", "short_known", "is_long", "caption",
        "is_hd",
    ],
    "language": [c for c in NUMERIC_FEATURES if c.startswith("lang_")],
    "title style": [c for c in NUMERIC_FEATURES if c.startswith("title_")],
    "description stats": ["desc_log_chars", "desc_links", "desc_hashtags", "n_tags"],
}  # fmt: skip


def log_views(views: Any) -> Any:
    return np.log1p(np.clip(views, 0, None))


def age_days(df: pd.DataFrame) -> pd.Series:
    age = (df["fetched_at"] - df["published_at"]).dt.total_seconds() / 86400.0
    return age.clip(lower=MIN_AGE_DAYS)


def add_channel_history(df: pd.DataFrame, window: int = HISTORY_WINDOW) -> pd.DataFrame:
    """Add ``hist_*`` columns describing each channel's earlier uploads (training data only).

    Works with multiple snapshots per video: history is computed from the latest snapshot of
    each video and merged back by ``video_id``.
    """
    df = df.copy()
    latest = (
        df.sort_values("fetched_at")
        .drop_duplicates("video_id", keep="last")
        .sort_values(["channel_id", "published_at"])
        .copy()
    )
    latest["_lv"] = log_views(latest["views"])
    latest["_la"] = np.log1p(age_days(latest))
    grouped = latest.groupby("channel_id", sort=False)

    def past(col: str, fn: str) -> pd.Series:
        return grouped[col].transform(
            lambda s: getattr(s.shift(1).rolling(window, min_periods=1), fn)()
        )

    hist = pd.DataFrame(
        {
            "video_id": latest["video_id"],
            "hist_median": past("_lv", "median"),
            "hist_mean": past("_lv", "mean"),
            "hist_std": past("_lv", "std"),
            "hist_max": past("_lv", "max"),
            "hist_log_age": past("_la", "median"),
            "hist_count": grouped.cumcount().clip(upper=window).astype(float),
        }
    )
    df = df.drop(columns=[c for c in hist.columns if c != "video_id" and c in df], errors="ignore")
    df = df.merge(hist, on="video_id", how="left")

    # Channel average views *excluding this video* (the channel total includes it, which would
    # otherwise leak the target for small channels).
    vids = df["channel_video_count"]
    others = (df["channel_views"] - df["views"]).clip(lower=0)
    df["channel_avg_views"] = np.where(vids > 1, others / (vids - 1).clip(lower=1), np.nan)
    return df


HISTORY_COLUMNS = ["hist_median", "hist_mean", "hist_std", "hist_max", "hist_log_age"]


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


def drop_thumbnails(df: pd.DataFrame, fraction: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Append copies of a random ``fraction`` of rows without their thumbnail.

    Teaches the model what to do when a thumbnail isn't supplied at prediction time.
    """
    if fraction <= 0 or not has_thumbnails(df):
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
    }


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(float)


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the raw (unscaled, possibly NaN) numeric feature matrix."""
    out = pd.DataFrame(index=df.index)
    out["log_age_days"] = np.log1p(age_days(df))
    published = df["published_at"]
    hour = published.dt.hour + published.dt.minute / 60.0
    dow = published.dt.dayofweek
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

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


def has_thumbnails(df: pd.DataFrame) -> bool:
    return "thumbnail_path" in df and df["thumbnail_path"].notna().any()


class FeatureBuilder:
    """Fit-on-train / transform-anywhere feature pipeline producing the model's inputs.

    Inputs produced: ``numeric`` (standardised), ``category`` (int ids) and one dense embedding
    per field: ``title``, ``description`` and, when thumbnails are available, ``thumbnail``
    (image features plus a final "thumbnail missing" flag column).
    """

    def __init__(
        self, text_encoder: str = "auto", text_dim: int = 128, image_encoder: str = "auto"
    ) -> None:
        self.text_encoder = text_encoder
        self.text_dim = text_dim
        self.image_encoder = image_encoder  # auto | clip | stats | none
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
        filled = self._impute(num)
        self.means_ = filled.mean()
        self.stds_ = filled.std().replace(0, 1.0).fillna(1.0)
        cats = sorted(c for c in df["category_id"].astype(str).unique() if c)
        self.categories_ = {c: i + 1 for i, c in enumerate(cats)}  # 0 = unknown
        self.feature_names_ = list(NUMERIC_FEATURES)
        self.category_counts_ = df["category_id"].astype(str).value_counts().to_dict()
        self.title_encoder_ = make_encoder(self.text_encoder, self.text_dim).fit(title_text(df))
        self.desc_encoder_ = make_encoder(self.text_encoder, self.text_dim).fit(
            description_text(df)
        )
        self.embed_stats_ = {}
        for key, emb in self._encode_text(df).items():
            self.embed_stats_[key] = (emb.mean(axis=0), emb.std(axis=0) + 1e-6)

        self.image_encoder_ = None
        if self.image_encoder != "none" and has_thumbnails(df):
            from .images import make_image_encoder

            enc = make_image_encoder(self.image_encoder)
            emb, missing = enc.transform(df["thumbnail_path"].tolist())
            if (~missing).sum() >= 2:
                self.image_encoder_ = enc
                present = emb[~missing]
                self.embed_stats_["thumbnail"] = (present.mean(axis=0), present.std(axis=0) + 1e-6)
        self.fitted = True
        return self

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
            paths = df["thumbnail_path"].tolist() if "thumbnail_path" in df else [None] * len(df)
            emb, missing = self.image_encoder_.transform(paths)
            emb = self._standardise("thumbnail", emb)
            emb[missing] = 0.0  # missing thumbnail -> "average thumbnail" + flag
            out["thumbnail"] = np.concatenate([emb, missing[:, None].astype(np.float32)], axis=1)
        return out

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
