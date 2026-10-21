import sys
import types

import numpy as np

from ytviews import text


def test_tfidf_encoder_shapes():
    enc = text.TfidfSvdEncoder(dim=8).fit(["a cat video", "a dog video", "minecraft build"] * 5)
    out = enc.transform(["cat build"])
    assert out.shape == (1, enc.output_dim) and out.dtype == np.float32


def test_auto_falls_back_to_tfidf(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import fails
    text._MODEL_CACHE.clear()
    assert isinstance(text.make_encoder("auto"), text.TfidfSvdEncoder)


def test_sbert_encoder_with_fake_model(monkeypatch):
    class FakeST:
        def __init__(self, name, device):
            self.name = name

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, **kw):
            return np.ones((len(texts), 4))

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST)
    )
    text._MODEL_CACHE.clear()
    enc = text.make_encoder("sbert")
    assert enc.kind == "sbert" and enc.output_dim == 4
    assert enc.transform(["x", "y"]).shape == (2, 4)
    text._MODEL_CACHE.clear()
