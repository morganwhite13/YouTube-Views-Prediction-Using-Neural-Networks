"""Text encoders that turn titles / descriptions into fixed-size dense vectors.

* ``sbert``  - a pretrained sentence-transformer (default ``all-MiniLM-L6-v2``). This replaces
  v1's per-field ``Embedding(100000, 300)`` layers, which had ~30M trainable parameters per field
  but only a few hundred training titles to learn them from.
* ``tfidf``  - word + character n-gram TF-IDF compressed with truncated SVD. Fully offline and
  fast; used automatically when sentence-transformers is unavailable.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union

log = logging.getLogger(__name__)

DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_URL_RE = re.compile(r"https?://\S+")
_MODEL_CACHE: dict[str, Any] = {}


def clean_text(text: str, max_chars: int | None = None) -> str:
    text = _URL_RE.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] if max_chars else text


class TfidfSvdEncoder:
    kind = "tfidf"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._vectorizer: Any = None
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> TfidfSvdEncoder:
        texts = [t.lower() for t in texts]
        min_df = 2 if len(texts) >= 50 else 1
        self._vectorizer = make_union(
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, max_features=50_000
            ),
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=min_df,
                sublinear_tf=True,
                max_features=50_000,
            ),
        )
        matrix = self._vectorizer.fit_transform(texts)
        n_components = max(1, min(self.dim, matrix.shape[1] - 1, matrix.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=0).fit(matrix)
        return self

    @property
    def output_dim(self) -> int:
        return self._svd.n_components if self._svd is not None else self.dim

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("encoder must be fitted first")
        out = self._svd.transform(self._vectorizer.transform([t.lower() for t in texts]))
        return out.astype(np.float32)


class SentenceTransformerEncoder:
    kind = "sbert"

    def __init__(self, model_name: str = DEFAULT_SBERT_MODEL) -> None:
        self.model_name = model_name
        self._dim: int | None = None

    def _model(self) -> Any:
        if self.model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name, device="cpu")
        return _MODEL_CACHE[self.model_name]

    def fit(self, texts: list[str]) -> SentenceTransformerEncoder:
        self._dim = int(self._model().get_sentence_embedding_dimension())
        return self

    @property
    def output_dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._model().get_sentence_embedding_dimension())
        return self._dim

    def transform(self, texts: list[str]) -> np.ndarray:
        emb = self._model().encode(
            list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(emb, dtype=np.float32)


def make_encoder(kind: str = "auto", dim: int = 128, model_name: str = DEFAULT_SBERT_MODEL):
    """Create a text encoder. ``auto`` prefers sentence-transformers and falls back to TF-IDF."""
    if kind == "tfidf":
        return TfidfSvdEncoder(dim)
    if kind in ("sbert", "auto"):
        try:
            enc = SentenceTransformerEncoder(model_name)
            enc.fit([])  # loads (and if needed downloads) the model now, so failures surface early
            return enc
        except Exception as exc:
            if kind == "sbert":
                raise
            log.warning("sentence-transformers unavailable (%s); falling back to TF-IDF+SVD", exc)
            return TfidfSvdEncoder(dim)
    raise ValueError(f"unknown text encoder {kind!r}; expected auto, sbert or tfidf")
