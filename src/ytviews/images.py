"""Thumbnail encoders: turn thumbnail images into fixed-size vectors.

* ``clip``  - CLIP image embeddings (``clip-ViT-B-32`` via sentence-transformers), which capture
  *what* is in the thumbnail (faces, text, objects, style).
* ``stats`` - hand-crafted, fully offline image statistics: brightness, contrast, saturation,
  colourfulness, edge density, a coarse luminance layout and a hue histogram. These are the
  properties thumbnail-design advice usually talks about ("bright, high-contrast, saturated").
* ``stats+faces`` - the same plus face count, the largest face's size and its position (needs
  ``pip install 'ytviews[faces]'``).

Every encoder returns ``(features, missing)``: rows whose image is absent or unreadable get zero
features and ``missing=True``, so a model can be used with or without a thumbnail.
"""

from __future__ import annotations

import logging
import threading
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


STATS_DIM = 3 + 3 + 11 + 16 + 8
FACE_DIM = 4
FACE_CACHE_NAMESPACE = "faces-haar-v1"
_FACES_OK: bool | None = None
_LOCAL = threading.local()  # OpenCV detectors are not safe to share between threads


def _cascade_path() -> str:
    import cv2

    return cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def faces_available() -> bool:
    """Whether OpenCV with its bundled face detector is installed and loads."""
    global _FACES_OK
    if _FACES_OK is None:
        try:
            import cv2

            _FACES_OK = not cv2.CascadeClassifier(_cascade_path()).empty()
        except Exception:  # noqa: BLE001 - missing module, no cv2.data, OpenCV 5, ...
            _FACES_OK = False
    return _FACES_OK


def _detector():
    if getattr(_LOCAL, "detector", None) is None:
        import cv2

        _LOCAL.detector = cv2.CascadeClassifier(_cascade_path())
    return _LOCAL.detector


def face_stats(img) -> np.ndarray:
    """Faces in a thumbnail: count (capped at 3), largest face's share of the frame and its
    centre (x, y in 0..1; 0.5 when there is no face). Uses OpenCV's bundled detector."""
    import cv2

    rgb = np.asarray(img)
    scale = 320 / max(rgb.shape[1], 1)
    if scale < 1:
        rgb = cv2.resize(rgb, (320, max(1, int(rgb.shape[0] * scale))))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    side = max(12, int(min(gray.shape) * 0.08))
    found = _detector().detectMultiScale(gray, 1.1, 5, minSize=(side, side))
    if len(found) == 0:
        return np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32)
    x, y, w, h = max(found, key=lambda f: f[2] * f[3])
    area = w * h / (gray.shape[0] * gray.shape[1])
    cx, cy = (x + w / 2) / gray.shape[1], (y + h / 2) / gray.shape[0]
    return np.array([min(len(found), 3), area, cx, cy], dtype=np.float32)


class ImageStatsEncoder:
    """Offline image statistics; with ``faces=True`` also face count, size and position.

    Face features are an explicit choice (``--image-encoder stats+faces``) and are recorded in
    ``kind``, so a model's inputs never depend on what happened to be installed.
    """

    def __init__(self, faces: bool = False) -> None:
        if faces and not faces_available():
            raise RuntimeError(
                "face features need OpenCV 4.x with its bundled face detector: "
                "pip install 'ytviews[faces]'"
            )
        self.faces = faces

    @property
    def kind(self) -> str:
        return "stats+faces" if getattr(self, "faces", False) else "stats"

    @property
    def output_dim(self) -> int:
        return STATS_DIM + (FACE_DIM if getattr(self, "faces", False) else 0)

    def fit(self, paths: Sequence[Any]) -> ImageStatsEncoder:
        return self

    def transform(self, paths: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros((len(paths), self.output_dim), dtype=np.float32)
        missing = np.ones(len(paths), dtype=bool)
        present: list[int] = []
        for i, p in enumerate(paths):
            img = _open(p)
            if img is not None:
                out[i, :STATS_DIM] = image_stats(img)
                missing[i] = False
                present.append(i)
        if getattr(self, "faces", False) and present:
            if not faces_available():
                raise RuntimeError(
                    "this model uses thumbnail face features; install them with "
                    "`pip install 'ytviews[faces]'`"
                )
            from .cache import cached_encode, file_key

            keys = [file_key(paths[i]) or f"unreadable:{i}" for i in present]
            out[present, STATS_DIM:] = cached_encode(
                FACE_CACHE_NAMESPACE, keys, [paths[i] for i in present],
                lambda batch: np.stack([face_stats(_open(p)) for p in batch]), FACE_DIM,
            )  # fmt: skip
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
    """``auto`` prefers CLIP, then image statistics with face features if OpenCV is installed,
    then plain image statistics. The choice is recorded in the encoder's ``kind``."""
    if kind == "stats":
        return ImageStatsEncoder()
    if kind == "stats+faces":
        return ImageStatsEncoder(faces=True)
    if kind in ("clip", "auto"):
        try:
            return ClipImageEncoder(model_name).fit([])
        except Exception as exc:
            if kind == "clip":
                raise
            fallback = ImageStatsEncoder(faces=faces_available())
            log.warning("CLIP unavailable (%s); using offline %s", exc, fallback.kind)
            return fallback
    raise ValueError(f"unknown image encoder {kind!r}; expected auto, clip, stats or stats+faces")


def attach_thumbnails(df, directory: str | Path):
    """Add a ``thumbnail_path`` column pointing at ``<directory>/<video_id>.jpg`` when present."""
    directory = Path(directory)
    df = df.copy()
    paths = [directory / f"{vid}.jpg" for vid in df["video_id"]]
    df["thumbnail_path"] = [str(p) if p.exists() else None for p in paths]
    return df
