"""Thumbnail encoders: turn thumbnail images into fixed-size vectors.

* ``clip``  - CLIP image embeddings (``clip-ViT-B-32`` via sentence-transformers), which capture
  *what* is in the thumbnail (faces, text, objects, style).
* ``stats`` - hand-crafted, fully offline image statistics: brightness, contrast, saturation,
  colourfulness, edge density, a coarse luminance layout and a hue histogram. These are the
  properties thumbnail-design advice usually talks about ("bright, high-contrast, saturated").

Every encoder returns ``(features, missing)``: rows whose image is absent or unreadable get zero
features and ``missing=True``, so a model can be used with or without a thumbnail.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "clip-ViT-B-32"
_SIZE = (64, 36)  # 16:9, small enough to be fast, large enough for layout/colour statistics
_MODEL_CACHE: dict[str, Any] = {}


def thumbnail_dir_for(db_path: str | Path) -> Path:
    """Folder where thumbnails for a given database live (``data/ytviews_thumbnails/``)."""
    db_path = Path(db_path)
    return db_path.parent / f"{db_path.stem}_thumbnails"


def _open(path: Any):
    from PIL import Image

    if path is None or (isinstance(path, float) and np.isnan(path)) or not str(path):
        return None
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (OSError, ValueError):
        return None


def image_stats(img) -> np.ndarray:
    """~40 interpretable statistics of an RGB PIL image."""
    from PIL import Image

    rgb = np.asarray(img.resize(_SIZE, Image.BILINEAR), dtype=np.float32) / 255.0
    hsv = np.asarray(img.resize(_SIZE, Image.BILINEAR).convert("HSV"), dtype=np.float32) / 255.0
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colourfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2
    )
    gy, gx = np.gradient(lum)
    edges = np.hypot(gx, gy)
    h, w = lum.shape
    centre = lum[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    grid = lum.reshape(4, h // 4, 4, w // 4).mean(axis=(1, 3)).ravel()
    hue_hist = np.histogram(hsv[..., 0], bins=8, range=(0, 1), weights=hsv[..., 1])[0]
    hue_hist = hue_hist / (hue_hist.sum() + 1e-6)
    return np.concatenate(
        [
            rgb.mean(axis=(0, 1)),
            rgb.std(axis=(0, 1)),
            [
                lum.mean(),
                lum.std(),  # contrast
                np.percentile(lum, 95) - np.percentile(lum, 5),
                hsv[..., 1].mean(),  # saturation
                hsv[..., 2].mean(),
                colourfulness,
                edges.mean(),  # busy vs clean
                (edges > 0.1).mean(),
                centre.mean() - lum.mean(),  # bright subject in the middle?
                (lum > 0.9).mean(),  # blown-out highlights / white text
                (lum < 0.1).mean(),
            ],
            grid,
            hue_hist,
        ]
    ).astype(np.float32)


class ImageStatsEncoder:
    kind = "stats"
    output_dim = 3 + 3 + 11 + 16 + 8

    def fit(self, paths: Sequence[Any]) -> ImageStatsEncoder:
        return self

    def transform(self, paths: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros((len(paths), self.output_dim), dtype=np.float32)
        missing = np.ones(len(paths), dtype=bool)
        for i, p in enumerate(paths):
            img = _open(p)
            if img is not None:
                out[i] = image_stats(img)
                missing[i] = False
        return out, missing


class ClipImageEncoder:
    kind = "clip"

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL) -> None:
        self.model_name = model_name
        self._dim: int | None = None

    def _model(self) -> Any:
        if self.model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name, device="cpu")
        return _MODEL_CACHE[self.model_name]

    @property
    def output_dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._model().get_sentence_embedding_dimension())
        return self._dim

    def fit(self, paths: Sequence[Any]) -> ClipImageEncoder:
        _ = self.output_dim
        return self

    def transform(self, paths: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        from .cache import cached_encode, file_key

        out = np.zeros((len(paths), self.output_dim), dtype=np.float32)
        keys = [file_key(p) if _open(p) is not None else None for p in paths]
        idx = [i for i, k in enumerate(keys) if k is not None]

        def encode(batch: list[Any]) -> np.ndarray:
            images = [_open(p) for p in batch]
            return self._model().encode(
                images, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )

        if idx:
            out[idx] = cached_encode(
                self.model_name, [keys[i] for i in idx], [paths[i] for i in idx], encode,
                self.output_dim,
            )  # fmt: skip
        missing = np.ones(len(paths), dtype=bool)
        missing[idx] = False
        return out, missing


def make_image_encoder(kind: str = "auto", model_name: str = DEFAULT_CLIP_MODEL):
    """``auto`` prefers CLIP and falls back to image statistics when it can't be loaded."""
    if kind == "stats":
        return ImageStatsEncoder()
    if kind in ("clip", "auto"):
        try:
            return ClipImageEncoder(model_name).fit([])
        except Exception as exc:
            if kind == "clip":
                raise
            log.warning("CLIP unavailable (%s); using offline image statistics", exc)
            return ImageStatsEncoder()
    raise ValueError(f"unknown image encoder {kind!r}; expected auto, clip or stats")


def attach_thumbnails(df, directory: str | Path):
    """Add a ``thumbnail_path`` column pointing at ``<directory>/<video_id>.jpg`` when present."""
    directory = Path(directory)
    df = df.copy()
    paths = [directory / f"{vid}.jpg" for vid in df["video_id"]]
    df["thumbnail_path"] = [str(p) if p.exists() else None for p in paths]
    return df
