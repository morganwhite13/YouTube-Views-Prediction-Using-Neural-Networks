"""HTTP API.  Run with ``ytviews serve`` (or ``uvicorn ytviews.api:app``).

Endpoints (JSON in, JSON out; interactive docs at ``/docs``):

* ``GET  /health``   - liveness
* ``GET  /model``    - what the loaded model was trained on and how well it scored
* ``POST /predict``  - prediction range + growth trajectory for one video
* ``POST /compare``  - rank alternative titles for one video
* ``POST /explain``  - per-feature-group effects behind a prediction
* ``POST /suggest``  - generate and rank title ideas
* ``POST /track``    - forecast a live video's views from its views so far
* ``POST /publish-times`` - predicted views for each weekday x hour slot

Configuration (environment variables, or ``create_app`` arguments):

* ``YTVIEWS_API_KEY``       - if set, every endpoint except ``/health`` requires the header
  ``X-API-Key: <value>``. Set it whenever the service is reachable by anyone but you.
* ``YTVIEWS_CORS_ORIGINS``  - comma-separated origins allowed to call the API from a browser.
* ``YTVIEWS_ALLOW_LLM=1``   - allow ``/suggest`` with ``use_llm`` (each call is a paid Claude
  request on the server's Anthropic account); refused otherwise.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import io
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import __version__
from .predictor import Predictor, VideoInput

MAX_THUMBNAIL_BYTES = 2_000_000


class Video(BaseModel):
    title: str = Field(..., min_length=1, max_length=300,
                       examples=["I Survived 100 Days in Minecraft"])  # fmt: skip
    subscribers: float | None = Field(None, ge=0, examples=[250_000])
    description: str = Field("", max_length=10_000)
    tags: list[str] = Field([], max_length=100)
    category_id: str = Field("24", max_length=10, description="YouTube category id, e.g. 20")
    duration_s: float | None = Field(None, ge=0, le=86_400 * 2)
    published_at: datetime | None = None
    age_days: float = Field(7.0, gt=0, le=3650, description="Predict views N days after upload")
    channel_video_count: float | None = Field(None, ge=0)
    channel_total_views: float | None = Field(None, ge=0)
    recent_video_views: list[float] = Field([], max_length=50, description="Recent uploads' views")
    recent_video_age_days: list[float] = Field([], max_length=50)
    caption: bool = False
    hd: bool = True
    language: str | None = Field(None, max_length=16, examples=["en"])
    is_short: bool | None = Field(None, description="Omit to guess from duration")
    thumbnail_base64: str | None = Field(
        None,
        max_length=MAX_THUMBNAIL_BYTES * 4 // 3 + 4,
        description="JPEG/PNG/WebP thumbnail, base64, at most 2 MB",
    )

    def to_input(self, thumbnail_path: str | None = None) -> VideoInput:
        fields = self.model_dump(exclude={"thumbnail_base64"})
        return VideoInput(**fields, thumbnail_path=thumbnail_path)


class CompareRequest(BaseModel):
    video: Video
    titles: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        ..., min_length=1, max_length=50
    )


class TrackRequest(BaseModel):
    video: Video
    observed_views: float = Field(..., ge=0, examples=[12000])
    observed_hours: float = Field(..., gt=0, examples=[24])


class PublishTimesRequest(BaseModel):
    video: Video
    timezone: str = Field("UTC", max_length=64, examples=["America/New_York"])


class SuggestRequest(BaseModel):
    video: Video
    use_llm: bool = Field(False, description="Also ask Claude for ideas (paid API call)")
    extra: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field([], max_length=20)


class PlanRequest(BaseModel):
    video: Video
    alt_titles: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field([], max_length=20)
    alt_thumbnails_base64: list[
        Annotated[str, Field(max_length=MAX_THUMBNAIL_BYTES * 4 // 3 + 4)]
    ] = Field([], max_length=5, description="Alternative thumbnails to rank, base64")
    timezone: str = Field("UTC", max_length=64, examples=["America/New_York"])
    use_llm: bool = Field(False, description="Also ask Claude for title ideas (paid API call)")


def _decode_image(b64: str) -> bytes:
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, "thumbnail_base64 is not valid base64") from exc
    if len(data) > MAX_THUMBNAIL_BYTES:
        raise HTTPException(413, "thumbnail is larger than 2 MB")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001 - any decoder error means "not an image"
        raise HTTPException(422, "thumbnail_base64 is not a readable image") from exc
    return data


@contextmanager
def _materialise(video: Video, extra_images: list[str] | None = None):
    """Write uploaded base64 thumbnails to temp files for the duration of a request.

    Yields the ``VideoInput`` and the paths of ``extra_images`` (alternative thumbnails).
    """
    images = [video.thumbnail_base64, *(extra_images or [])]
    decoded = [_decode_image(b) if b else None for b in images]
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, data in enumerate(decoded):
            if data is None:
                paths.append(None)
                continue
            path = Path(tmp) / f"thumbnail-{i}"
            path.write_bytes(data)
            paths.append(str(path))
        yield video.to_input(paths[0]), paths[1:]


def _records(df) -> list[dict]:
    return df.to_dict(orient="records")


def create_app(
    model_path: str | Path | None = None,
    *,
    api_key: str | None = None,
    cors_origins: list[str] | None = None,
    allow_llm: bool | None = None,
) -> FastAPI:
    model_path = Path(model_path or os.environ.get("YTVIEWS_MODEL", "models/latest"))
    api_key = api_key if api_key is not None else os.environ.get("YTVIEWS_API_KEY") or None
    if cors_origins is None:
        raw = os.environ.get("YTVIEWS_CORS_ORIGINS", "")
        cors_origins = [o.strip() for o in raw.split(",") if o.strip()]
    if allow_llm is None:
        allow_llm = os.environ.get("YTVIEWS_ALLOW_LLM", "").lower() in ("1", "true", "yes")

    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(given: str | None = Security(header)) -> None:
        if api_key and not (given and hmac.compare_digest(given, api_key)):
            raise HTTPException(401, "missing or wrong X-API-Key header")

    api = FastAPI(
        title="ytviews",
        version=__version__,
        description="Predict YouTube views (with uncertainty) before you publish.",
    )
    protected = [Depends(require_key)]
    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        api.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
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

    @api.get("/model", dependencies=protected)
    def model_info() -> dict:
        meta = predictor().meta
        keys = ("version", "created_at", "target", "split", "text_encoder", "image_encoder",
                "dataset",
                "metrics", "importance")  # fmt: skip
        return {k: meta.get(k, "views" if k == "target" else None) for k in keys}

    @api.post("/predict", dependencies=protected)
    def predict(video: Video) -> dict:
        p = predictor()
        with _materialise(video) as (v, _):
            pred = p.predict(v)[0]
            traj = p.trajectory(v)
        return {"prediction": pred.as_dict(), "trajectory": _records(traj)}

    @api.post("/compare", dependencies=protected)
    def compare(req: CompareRequest) -> dict:
        p = predictor()
        with _materialise(req.video) as (v, _):
            return {"ranking": _records(p.compare_titles(v, req.titles))}

    @api.post("/explain", dependencies=protected)
    def explain(video: Video) -> dict:
        p = predictor()
        with _materialise(video) as (v, _):
            ex = p.explain(v)
        return {
            "baseline_views": ex.baseline_views,
            "predicted_views": ex.predicted_views,
            "effects": _records(ex.effects),
        }

    @api.post("/suggest", dependencies=protected)
    def suggest(req: SuggestRequest) -> dict:
        from .suggest import suggest_titles

        if req.use_llm and not allow_llm:
            raise HTTPException(
                403, "Claude suggestions are disabled on this server (set YTVIEWS_ALLOW_LLM=1)"
            )
        p = predictor()
        with _materialise(req.video) as (v, _):
            ranked = suggest_titles(p, v, use_llm=req.use_llm, extra=req.extra)
        return {"ranking": _records(ranked)}

    @api.post("/track", dependencies=protected)
    def track(req: TrackRequest) -> dict:
        p = predictor()
        if p.target != "views":
            raise HTTPException(422, f"/track needs a views model; this model predicts {p.target}")
        with _materialise(req.video) as (v, _):
            nc = p.track(v, req.observed_views, req.observed_hours / 24)
        return {
            "expected_views_so_far": nc.expected_views,
            "vs_expected": nc.vs_expected,
            "calibrated": nc.calibrated,
            "forecast": _records(nc.forecast),
        }

    @api.post("/publish-times", dependencies=protected)
    def publish_times(req: PublishTimesRequest) -> dict:
        from zoneinfo import ZoneInfoNotFoundError

        p = predictor()
        with _materialise(req.video) as (v, _):
            try:
                grid = p.publish_times(v, req.timezone)
            except ZoneInfoNotFoundError as exc:
                raise HTTPException(422, f"unknown timezone {req.timezone!r}") from exc
        return {"slots": _records(grid)}

    @api.post("/plan", dependencies=protected)
    def plan(req: PlanRequest) -> dict:
        from zoneinfo import ZoneInfoNotFoundError

        from .planner import make_plan

        if req.use_llm and not allow_llm:
            raise HTTPException(
                403, "Claude suggestions are disabled on this server (set YTVIEWS_ALLOW_LLM=1)"
            )
        p = predictor()
        with _materialise(req.video, req.alt_thumbnails_base64) as (v, alts):
            try:
                result = make_plan(
                    p, v, extra_titles=req.alt_titles, thumbnails=alts,
                    timezone_name=req.timezone, use_llm=req.use_llm,
                )  # fmt: skip
            except ZoneInfoNotFoundError as exc:
                raise HTTPException(422, f"unknown timezone {req.timezone!r}") from exc
            # Temp file paths mean nothing to the caller: name thumbnails by their position.
            names = {v.thumbnail_path: "draft"} | {a: f"alt {i}" for i, a in enumerate(alts, 1)}
            out = result.to_dict()
        for key in ("draft", "best"):
            out[key]["thumbnail"] = names.get(out[key].pop("thumbnail_path"))
        for row in out["thumbnails"] or []:
            row["thumbnail"] = names.get(row["thumbnail"], row["thumbnail"])
        out["changes"] = [_rename(c, names) for c in out["changes"]]
        return out

    return api


def _rename(text: str, names: dict) -> str:
    for path, name in names.items():
        if path:
            text = text.replace(Path(path).name, name)
    return text


app = create_app()
