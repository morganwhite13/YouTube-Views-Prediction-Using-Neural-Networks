"""Text on thumbnails (OCR), as extra thumbnail features.

``--image-encoder stats+ocr`` (or ``clip+ocr``) reads the words on each thumbnail with RapidOCR
(``pip install 'ytviews[ocr]'``; its models ship inside the package, so it works offline) and
adds four columns to the thumbnail inputs:

* ``log1p(words)``: how many words the thumbnail shows,
* ``text_area``: the share of the image covered by text boxes,
* ``has_text``: whether there is any readable text at all,
* ``title_overlap``: the share of the thumbnail's words that are also in the title (a thumbnail
  that just repeats the title is common advice to avoid; the model learns whether it matters).

Reading text is slow (about half a second a thumbnail), so results are cached on disk by file
content under the embedding cache (``$YTVIEWS_CACHE_DIR/ocr``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cache import cache_dir

OCR_DIM = 4  # log1p(words), text area, has text, title overlap
MAX_WIDTH = 640  # thumbnails are read at this width: plenty for thumbnail-sized lettering
MIN_CONFIDENCE = 0.5
_WORD_RE = re.compile(r"[a-z0-9']+")
_ENGINE: Any = None


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _engine() -> Any:
    global _ENGINE
    if _ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise ImportError(
                "thumbnail text features need RapidOCR: pip install 'ytviews[ocr]'"
            ) from exc
        _ENGINE = RapidOCR()
    return _ENGINE


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def read_thumbnail(path: str | Path) -> dict[str, Any] | None:
    """``{"text", "area"}`` for an image file (cached by content), or None if unreadable."""
    try:
        data = Path(path).read_bytes()
    except (OSError, TypeError):
        return None
    cache = cache_dir() / "ocr" / f"{hashlib.sha1(data).hexdigest()}.json"
    try:
        return json.loads(cache.read_text())
    except (OSError, ValueError):
        pass
    from PIL import Image

    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):  # as images._open: unreadable
        return None
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)))
    boxes, _ = _engine()(np.asarray(img))
    texts, area = [], 0.0
    for box, text, score in boxes or []:
        if float(score) < MIN_CONFIDENCE:
            continue
        pts = np.asarray(box, dtype=float)
        area += (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        texts.append(str(text))
    result = {"text": " ".join(texts), "area": min(area / (img.width * img.height), 1.0)}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result))
    except OSError:
        pass
    return result


def ocr_features(paths: Sequence[Any], titles: Sequence[str]) -> np.ndarray:
    """``(n, OCR_DIM)`` features; rows without a readable thumbnail are all zero."""
    out = np.zeros((len(paths), OCR_DIM), dtype=np.float32)
    for i, (path, title) in enumerate(zip(paths, titles, strict=True)):
        read = read_thumbnail(path) if path else None
        if read is None:
            continue
        found = words(read["text"])
        in_title = set(words(title or ""))
        out[i] = [
            np.log1p(len(found)),
            read["area"],
            float(bool(found)),
            sum(w in in_title for w in found) / len(found) if found else 0.0,
        ]
    return out


class OcrImageEncoder:
    """Any image encoder plus the OCR columns (titles are passed in by the feature pipeline)."""

    reads_text = True

    def __init__(self, base: Any) -> None:
        self.base = base
        _engine()  # fail now, not halfway through training, if RapidOCR is missing

    @property
    def kind(self) -> str:
        return f"{self.base.kind}+ocr"

    @property
    def output_dim(self) -> int:
        return self.base.output_dim + OCR_DIM

    def fit(self, paths: Sequence[Any]) -> OcrImageEncoder:
        self.base.fit(paths)
        return self

    def transform(
        self, paths: Sequence[Any], titles: Sequence[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        emb, missing = self.base.transform(paths)
        extra = ocr_features(paths, titles if titles is not None else [""] * len(paths))
        extra[missing] = 0.0
        return np.concatenate([emb, extra], axis=1), missing
