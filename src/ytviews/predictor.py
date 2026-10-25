"""Load a trained model and answer creator-facing questions about a (planned) video."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .features import FeatureBuilder, history_from_recent
from .train import BlendedRegressor, NeuralRegressor

MODEL_FILE = "model.pt"
FEATURES_FILE = "features.joblib"
META_FILE = "meta.json"
GBM_FILE = "gbm.joblib"


@dataclass
class VideoInput:
    """Everything the model needs to know about a video (none of it requires it to be live)."""

    title: str
    subscribers: float | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category_id: str = "24"  # Entertainment
    duration_s: float | None = None
    published_at: datetime | None = None  # defaults to "now"
    age_days: float = 7.0  # predict the view count this many days after publishing
    channel_video_count: float | None = None
    channel_total_views: float | None = None
    recent_video_views: list[float] = field(default_factory=list)  # newest first
    recent_video_age_days: list[float] = field(default_factory=list)
    caption: bool = False
    hd: bool = True
    thumbnail_path: str | None = None
    language: str | None = None  # e.g. "en", "es-419"; None = not specified
    is_short: bool | None = None  # None = guess from duration

    def to_row(self) -> dict[str, Any]:
        published = self.published_at or datetime.now(timezone.utc)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        # Training data is in UTC, so hour/weekday features must be computed in UTC too.
        published = published.astimezone(timezone.utc)
        avg = (
            self.channel_total_views / self.channel_video_count
            if self.channel_total_views and self.channel_video_count
            else np.nan
        )
        row = {
            "title": self.title,
            "description": self.description,
            "tags": "|".join(self.tags),
            "category_id": str(self.category_id),
            "duration_s": np.nan if self.duration_s is None else float(self.duration_s),
            "published_at": pd.Timestamp(published),
            "fetched_at": pd.Timestamp(published + timedelta(days=self.age_days)),
            "subscribers": np.nan if self.subscribers is None else float(self.subscribers),
            "channel_video_count": np.nan
            if self.channel_video_count is None
            else float(self.channel_video_count),
            "channel_avg_views": avg,
            "caption": float(self.caption),
            "definition": "hd" if self.hd else "sd",
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "default_language": self.language or "",
            "is_short": np.nan if self.is_short is None else float(self.is_short),
        }
        row.update(history_from_recent(self.recent_video_views, self.recent_video_age_days))
        return row


@dataclass
class Prediction:
    low: float
    median: float
    high: float
    coverage: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class Predictor:
    def __init__(
        self,
        features: FeatureBuilder,
        regressor: NeuralRegressor | BlendedRegressor,
        meta: dict[str, Any],
    ) -> None:
        self.features = features
        self.regressor = regressor
        self.meta = meta

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        reg = self.regressor
        if isinstance(reg, BlendedRegressor):
            joblib.dump(
                {"model": reg.gbm, "weight": reg.weight, "interval_adjust": reg.interval_adjust},
                path / GBM_FILE,
            )
            reg = reg.nn
        elif (path / GBM_FILE).exists():
            (path / GBM_FILE).unlink()
        torch.save(reg.state(), path / MODEL_FILE)
        joblib.dump(self.features, path / FEATURES_FILE)
        (path / META_FILE).write_text(json.dumps(self.meta, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Predictor:
        path = Path(path)
        if not (path / MODEL_FILE).exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Train one first with `ytviews train` "
                "(or `ytviews demo` for a synthetic-data model)."
            )
        state = torch.load(path / MODEL_FILE, map_location="cpu", weights_only=False)
        features = joblib.load(path / FEATURES_FILE)
        meta = json.loads((path / META_FILE).read_text())
        reg: NeuralRegressor | BlendedRegressor = NeuralRegressor.from_state(state)
        if (path / GBM_FILE).exists():
            gbm = joblib.load(path / GBM_FILE)
            reg = BlendedRegressor(
                reg, gbm["model"], features, gbm["weight"], gbm.get("interval_adjust", 0.0)
            )
        return cls(features, reg, meta)

    # ------------------------------------------------------------------ prediction
    @property
    def target(self) -> str:
        """What the model predicts: "views" (default), "likes" or "comments"."""
        return self.meta.get("target", "views")

    @property
    def coverage(self) -> float:
        q = self.regressor.quantiles
        return q[-1] - q[0]

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Quantiles of *views* (not log) for each row of a prepared frame."""
        q = self.regressor.predict_quantiles(self.features.transform(df))
        return np.expm1(q).clip(min=0)

    def predict(self, videos: VideoInput | list[VideoInput]) -> list[Prediction]:
        videos = [videos] if isinstance(videos, VideoInput) else videos
        df = pd.DataFrame([v.to_row() for v in videos])
        q = self.predict_frame(df)
        return [Prediction(float(r[0]), float(r[1]), float(r[-1]), self.coverage) for r in q]

    def explain(self, video: VideoInput):
        """Which feature groups push this prediction up or down (see :mod:`ytviews.explain`)."""
        from .explain import explain

        return explain(self, video)

    def track(
        self,
        video: VideoInput,
        observed_views: float,
        observed_age_days: float,
        days: tuple[float, ...] = (1, 3, 7, 14, 30, 90, 180, 365),
    ):
        """Forecast a live video's future views from its views so far."""
        from .nowcast import track

        return track(self, video, observed_views, observed_age_days, days)

    def publish_times(
        self, video: VideoInput, timezone_name: str = "UTC", start: datetime | None = None
    ) -> pd.DataFrame:
        """Predicted views for every weekday x hour slot in the coming week (local time)."""
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(timezone_name)
        start = (start or datetime.now(tz)).astimezone(tz)
        day0 = start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        slots = [day0 + timedelta(days=d, hours=h) for d in range(7) for h in range(24)]
        preds = self.predict([replace(video, published_at=t) for t in slots])
        out = pd.DataFrame(
            {
                "weekday": [t.strftime("%A") for t in slots],
                "weekday_index": [t.weekday() for t in slots],
                "hour": [t.hour for t in slots],
                "median": [p.median for p in preds],
            }
        )
        out["vs_best"] = out["median"] / out["median"].max()
        return out.sort_values(["weekday_index", "hour"]).reset_index(drop=True)

    def compare_thumbnails(self, base: VideoInput, paths: list[str]) -> pd.DataFrame:
        """Rank alternative thumbnails for the same video."""
        if not self.features.uses_thumbnails:
            raise ValueError("this model was trained without thumbnails")
        preds = self.predict([replace(base, thumbnail_path=str(p)) for p in paths])
        df = pd.DataFrame(
            {
                "thumbnail": [str(p) for p in paths],
                "low": [p.low for p in preds],
                "median": [p.median for p in preds],
                "high": [p.high for p in preds],
            }
        )
        df["vs_best"] = df["median"] / df["median"].max()
        return df.sort_values("median", ascending=False).reset_index(drop=True)

    def compare_titles(self, base: VideoInput, titles: list[str]) -> pd.DataFrame:
        """Rank alternative titles for the same video (an offline A/B test)."""
        preds = self.predict([replace(base, title=t) for t in titles])
        df = pd.DataFrame(
            {
                "title": titles,
                "low": [p.low for p in preds],
                "median": [p.median for p in preds],
                "high": [p.high for p in preds],
            }
        )
        df["vs_best"] = df["median"] / df["median"].max()
        return df.sort_values("median", ascending=False).reset_index(drop=True)

    def trajectory(
        self, base: VideoInput, days: tuple[float, ...] = (1, 3, 7, 14, 30, 90, 180, 365)
    ) -> pd.DataFrame:
        days = tuple(days)
        """Predicted cumulative views at several ages after publishing."""
        preds = self.predict([replace(base, age_days=float(d)) for d in days])
        out = pd.DataFrame(
            {
                "days": list(days),
                "low": [p.low for p in preds],
                "median": [p.median for p in preds],
                "high": [p.high for p in preds],
            }
        )
        # Cumulative views cannot go down; enforce that on the (independently predicted) points.
        for col in ("low", "median", "high"):
            out[col] = np.maximum.accumulate(out[col].to_numpy())
        return out
