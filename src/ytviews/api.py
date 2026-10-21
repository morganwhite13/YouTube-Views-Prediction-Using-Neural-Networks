"""HTTP API.  Run with ``ytviews serve`` (or ``uvicorn ytviews.api:app``).

Endpoints (JSON in, JSON out; interactive docs at ``/docs``):

* ``GET  /health``   - liveness
* ``GET  /model``    - what the loaded model was trained on and how well it scored
* ``POST /predict``  - prediction range + growth trajectory for one video
* ``POST /compare``  - rank alternative titles for one video
* ``POST /explain``  - per-feature-group effects behind a prediction
* ``POST /suggest``  - generate and rank title ideas
"""

from __future__ import annotations

import base64
import binascii
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .predictor import Predictor, VideoInput


class Video(BaseModel):
    title: str = Field(..., examples=["I Survived 100 Days in Minecraft"])
    subscribers: float | None = Field(None, examples=[250_000])
    description: str = ""
    tags: list[str] = []
    category_id: str = Field("24", description="YouTube category id, e.g. 20 = Gaming")
    duration_s: float | None = None
    published_at: datetime | None = None
    age_days: float = Field(7.0, gt=0, description="Predict views this many days after upload")
    channel_video_count: float | None = None
    channel_total_views: float | None = None
    recent_video_views: list[float] = Field([], description="Views of recent uploads")
    recent_video_age_days: list[float] = []
    caption: bool = False
    hd: bool = True
    thumbnail_base64: str | None = Field(None, description="JPEG/PNG thumbnail, base64")

    def to_input(self, thumbnail_path: str | None = None) -> VideoInput:
        fields = self.model_dump(exclude={"thumbnail_base64"})
        return VideoInput(**fields, thumbnail_path=thumbnail_path)


class CompareRequest(BaseModel):
    video: Video
    titles: list[str] = Field(..., min_length=1, max_length=50)


class SuggestRequest(BaseModel):
    video: Video
    use_llm: bool = Field(False, description="Also ask Claude for ideas (paid API call)")
    extra: list[str] = []


@contextmanager
def _materialise(video: Video):
    """Write an uploaded base64 thumbnail to a temp file for the duration of a request."""
    if not video.thumbnail_base64:
        yield video.to_input()
        return
    try:
        data = base64.b64decode(video.thumbnail_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, "thumbnail_base64 is not valid base64") from exc
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "thumbnail"
        path.write_bytes(data)
        yield video.to_input(str(path))


def _records(df) -> list[dict]:
    return df.to_dict(orient="records")


def create_app(model_path: str | Path | None = None) -> FastAPI:
    model_path = Path(model_path or os.environ.get("YTVIEWS_MODEL", "models/latest"))
    api = FastAPI(
        title="ytviews",
        version=__version__,
        description="Predict YouTube views (with uncertainty) before you publish.",
    )
    state: dict[str, Predictor] = {}

    def predictor() -> Predictor:
        if "p" not in state:
            try:
                state["p"] = Predictor.load(model_path)
            except FileNotFoundError as exc:
                raise HTTPException(503, str(exc)) from exc
        return state["p"]

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @api.get("/model")
    def model_info() -> dict:
        meta = predictor().meta
        keys = ("version", "created_at", "split", "text_encoder", "image_encoder", "dataset",
                "metrics", "importance")  # fmt: skip
        return {k: meta.get(k) for k in keys}

    @api.post("/predict")
    def predict(video: Video) -> dict:
        p = predictor()
        with _materialise(video) as v:
            pred = p.predict(v)[0]
            traj = p.trajectory(v)
        return {"prediction": pred.as_dict(), "trajectory": _records(traj)}

    @api.post("/compare")
    def compare(req: CompareRequest) -> dict:
        p = predictor()
        with _materialise(req.video) as v:
            return {"ranking": _records(p.compare_titles(v, req.titles))}

    @api.post("/explain")
    def explain(video: Video) -> dict:
        p = predictor()
        with _materialise(video) as v:
            ex = p.explain(v)
        return {
            "baseline_views": ex.baseline_views,
            "predicted_views": ex.predicted_views,
            "effects": _records(ex.effects),
        }

    @api.post("/suggest")
    def suggest(req: SuggestRequest) -> dict:
        from .suggest import suggest_titles

        p = predictor()
        with _materialise(req.video) as v:
            ranked = suggest_titles(p, v, use_llm=req.use_llm, extra=req.extra)
        return {"ranking": _records(ranked)}

    return api


app = create_app()
