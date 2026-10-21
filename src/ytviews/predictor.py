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
from .train import NeuralRegressor

MODEL_FILE = "model.pt"
FEATURES_FILE = "features.joblib"
META_FILE = "meta.json"


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

    def to_row(self) -> dict[str, Any]:
        published = self.published_at or datetime.now(timezone.utc)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
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
        self, features: FeatureBuilder, regressor: NeuralRegressor, meta: dict[str, Any]
    ) -> None:
        self.features = features
        self.regressor = regressor
        self.meta = meta

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.regressor.state(), path / MODEL_FILE)
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
        return cls(features, NeuralRegressor.from_state(state), meta)

    # ------------------------------------------------------------------ prediction
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
