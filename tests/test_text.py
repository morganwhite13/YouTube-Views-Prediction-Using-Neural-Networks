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


def test_sbert_embeddings_are_cached_and_deduplicated(monkeypatch, tmp_path):
    calls = []

    class CountingST:
        def __init__(self, name, device):
            pass

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kw):
            calls.append(list(texts))
            return np.array([[len(t), 1.0, 2.0] for t in texts])

    monkeypatch.setenv("YTVIEWS_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=CountingST)
    )
    text._MODEL_CACHE.clear()
    enc = text.make_encoder("sbert")
    first = enc.transform(["a", "bb", "a"])
    assert calls == [["a", "bb"]]  # duplicate encoded once
    second = enc.transform(["bb", "ccc"])
    assert calls[-1] == ["ccc"]  # "bb" came from the cache
    assert np.allclose(first[1], second[0]) and first[0][0] == 1
    monkeypatch.setenv("YTVIEWS_CACHE", "0")
    enc.transform(["bb"])
    assert calls[-1] == ["bb"]  # cache disabled
    text._MODEL_CACHE.clear()


def test_multilingual_encoder_uses_the_multilingual_model(monkeypatch, tmp_path):
    names = []

    class FakeST:
        def __init__(self, name, device):
            names.append(name)

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kw):
            return np.ones((len(texts), 2))

    monkeypatch.setenv("YTVIEWS_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST)
    )
    text._MODEL_CACHE.clear()
    enc = text.make_encoder("sbert-multi")
    assert enc.model_name == text.MULTILINGUAL_SBERT_MODEL and names == [enc.model_name]
    assert enc.transform(["hola", "bonjour"]).shape == (2, 2)
    text._MODEL_CACHE.clear()
