"""Text encoders that turn titles / descriptions into fixed-size dense vectors.

* ``sbert``  - a pretrained sentence-transformer (default ``all-MiniLM-L6-v2``). This replaces
  v1's per-field ``Embedding(100000, 300)`` layers, which had ~30M trainable parameters per field
  but only a few hundred training titles to learn them from.
* ``tfidf``  - word + character n-gram TF-IDF compressed with truncated SVD. Fully offline and
  fast; used automatically when sentence-transformers is unavailable.
* ``sbert-ft`` - the sentence-transformer *fine-tuned on your training titles* to predict how
  much better or worse a video did than its channel, timing and format alone suggest. Slower
  to train, and only worth it with a few thousand videos; descriptions stay frozen.
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
# For channels that publish in several languages (titles in 50+ languages share one space).
MULTILINGUAL_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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
        from .cache import cached_encode, text_key

        texts = list(texts)

        def encode(batch: list[str]) -> np.ndarray:
            return self._model().encode(
                batch, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )

        keys = [text_key(t) for t in texts]
        return cached_encode(self.model_name, keys, texts, encode, self.output_dim)


class FineTunedSentenceEncoder(SentenceTransformerEncoder):
    """A sentence-transformer fine-tuned to put titles that beat expectations apart.

    ``fit(texts, targets)`` trains the transformer plus a linear head (mean-squared error on
    the standardised ``targets``) for a few epochs on the training titles only, then encodes
    with the tuned transformer (the head is dropped). The tuned weights travel inside the saved
    model; loading it needs the base model too (downloaded once, then cached).
    """

    kind = "sbert-ft"
    needs_targets = True

    def __init__(
        self,
        model_name: str = DEFAULT_SBERT_MODEL,
        epochs: int = 2,
        lr: float = 2e-5,
        batch_size: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__(model_name)
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed
        self._tuned: Any = None
        self._state: dict | None = None
        self.history: list[float] = []

    def _base(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device="cpu")  # a private copy to tune

    def fit(self, texts: list[str], targets: np.ndarray | None = None) -> FineTunedSentenceEncoder:
        import torch

        model = self._base()
        self._dim = int(model.get_sentence_embedding_dimension())
        if targets is None or len(texts) < 2:
            self._tuned = model.eval()
            return self
        y = np.asarray(targets, dtype=np.float32)
        y = (y - y.mean()) / (y.std() or 1.0)
        torch.manual_seed(self.seed)
        head = torch.nn.Linear(self._dim, 1)
        params = list(model.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        rng = np.random.default_rng(self.seed)
        model.train()
        for _ in range(self.epochs):
            order, losses = rng.permutation(len(texts)), []
            for start in range(0, len(texts), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch = model.tokenize([texts[i] for i in idx])
                emb = model(batch)["sentence_embedding"]
                emb = torch.nn.functional.normalize(emb, dim=-1)
                loss = torch.nn.functional.mse_loss(head(emb).squeeze(-1), torch.tensor(y[idx]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
            self.history.append(float(np.mean(losses)))
        self._tuned = model.eval()
        return self

    def _model(self) -> Any:
        if self._tuned is None:  # loaded from disk: rebuild from the base + tuned weights
            model = self._base()
            if self._state is not None:
                model.load_state_dict(self._state)
            self._tuned = model.eval()
        return self._tuned

    def transform(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self._model().encode(
                list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
        return np.asarray(out, dtype=np.float32)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self._tuned is not None:
            state["_state"] = {k: v.detach().cpu() for k, v in self._tuned.state_dict().items()}
        state["_tuned"] = None
        return state


def make_encoder(kind: str = "auto", dim: int = 128, model_name: str = DEFAULT_SBERT_MODEL):
    """Create a text encoder. ``auto`` prefers sentence-transformers and falls back to TF-IDF."""
    if kind == "tfidf":
        return TfidfSvdEncoder(dim)
    if kind == "sbert-ft":
        return FineTunedSentenceEncoder(model_name)
    if kind == "sbert-multi":
        return SentenceTransformerEncoder(MULTILINGUAL_SBERT_MODEL).fit([])
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
    raise ValueError(
        f"unknown text encoder {kind!r}; expected auto, sbert, sbert-ft, sbert-multi or tfidf"
    )
