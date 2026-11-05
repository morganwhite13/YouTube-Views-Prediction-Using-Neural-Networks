"""Fine-tuned title encoder, with a tiny trainable stand-in for sentence-transformers."""

import pickle
import sys
import types

import numpy as np
import torch

from ytviews import text

VOCAB = "abcdefghijklmnopqrstuvwxyz!?0123456789 "


class TinyST(torch.nn.Module):
    """Bag-of-characters -> linear embedding; same calls the encoder makes on the real thing."""

    def __init__(self, name, device="cpu"):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(len(VOCAB), 8)

    def get_sentence_embedding_dimension(self):
        return 8

    def tokenize(self, texts):
        x = torch.zeros(len(texts), len(VOCAB))
        for i, t in enumerate(texts):
            for c in t.lower():
                if c in VOCAB:
                    x[i, VOCAB.index(c)] += 1
        return {"bag": x}

    def forward(self, features):
        return {"sentence_embedding": self.proj(features["bag"])}

    def encode(self, texts, normalize_embeddings=True, **kw):
        emb = self.forward(self.tokenize(list(texts)))["sentence_embedding"]
        if normalize_embeddings:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.detach().numpy()


def _fake(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=TinyST)
    )
    text._MODEL_CACHE.clear()


def test_fine_tuning_learns_and_survives_a_save(monkeypatch):
    _fake(monkeypatch)
    rng = np.random.default_rng(0)
    titles = [("WOW!!! " if i % 2 else "a quiet update ") + str(i) for i in range(200)]
    y = np.array([2.0 if "!" in t else -2.0 for t in titles]) + rng.normal(0, 0.1, 200)
    base = text.FineTunedSentenceEncoder("tiny").fit(titles)  # no targets: the base model
    enc = text.FineTunedSentenceEncoder("tiny", epochs=8, lr=0.05).fit(titles, y)
    assert enc.kind == "sbert-ft" and enc.output_dim == 8
    assert enc.history[-1] < enc.history[0]  # the regression loss went down

    # tuned embeddings separate the two kinds of title more than the base model's did
    def gap(e):
        out = e.transform(["WOW!!! big news", "a quiet update"])
        return float(np.linalg.norm(out[0] - out[1]))

    assert gap(enc) > gap(base)
    restored = pickle.loads(pickle.dumps(enc))
    np.testing.assert_allclose(restored.transform(titles[:5]), enc.transform(titles[:5]), atol=1e-6)
    text._MODEL_CACHE.clear()


def test_feature_builder_fine_tunes_titles_only(monkeypatch, synthetic_df):
    _fake(monkeypatch)
    from ytviews.features import FeatureBuilder, add_channel_history

    df = add_channel_history(synthetic_df).head(300)
    seen = {}
    real_fit = text.FineTunedSentenceEncoder.fit

    def spy(self, texts, targets=None):
        seen["targets"] = targets
        return real_fit(self, texts, targets)

    monkeypatch.setattr(text.FineTunedSentenceEncoder, "fit", spy)
    fb = FeatureBuilder(text_encoder="sbert-ft", image_encoder="none").fit(df)
    assert fb.title_encoder_.kind == "sbert-ft" and fb.desc_encoder_.kind == "sbert"
    targets = seen["targets"]
    assert len(targets) == len(df) and abs(float(np.mean(targets))) < 1e-6  # ridge residuals
    out = fb.transform(df.head(3))
    assert out["title"].shape == (3, 8) and np.isfinite(out["title"]).all()
    text._MODEL_CACHE.clear()
