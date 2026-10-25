"""Regression tests for issues found in code review."""

import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

from ytviews.collect import collect_channels
from ytviews.predictor import Predictor, VideoInput
from ytviews.store import VideoStore
from ytviews.youtube import QuotaExceeded, YouTubeClient


class FiveQuantiles:
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)

    def predict_quantiles(self, inputs, calibrated=True):
        n = len(inputs["numeric"])
        return np.tile(np.log1p([100.0, 200.0, 300.0, 400.0, 500.0]), (n, 1))


def test_median_comes_from_the_0_5_quantile(trained):
    model, _ = trained
    p = Predictor.load(model)
    p.regressor = FiveQuantiles()
    pred = p.predict(VideoInput(title="x"))[0]
    assert (pred.low, pred.median, pred.high) == pytest.approx((100, 300, 500))
    assert pred.coverage == pytest.approx(0.9)


def test_trajectory_sorts_days(trained):
    p = Predictor.load(trained[0])
    traj = p.trajectory(VideoInput(title="x", subscribers=1e5), days=(30, 1, 7))
    assert traj["days"].tolist() == [1, 7, 30]


def test_collection_resumes_after_running_out_of_quota(fake_youtube, tmp_path):
    with VideoStore(tmp_path / "db.sqlite") as store:
        # 3 units: channel stats (1) + UC1's playlist (1) + UC1's videos (1); UC2 then fails.
        client = YouTubeClient(service=fake_youtube, quota_budget=3)
        with pytest.raises(QuotaExceeded):
            collect_channels(client, store, ["UC1", "UC2"], max_videos_per_channel=5)
        assert set(store.uploads_collected_at()) == {"UC1"}
        # Rerunning the same day skips UC1 (fresh) but still collects UC2.
        retry = YouTubeClient(service=fake_youtube)
        collect_channels(retry, store, ["UC1", "UC2"], max_videos_per_channel=5,
                         skip_fresh_days=1)  # fmt: skip
        assert set(store.uploads_collected_at()) == {"UC1", "UC2"}


def test_batch_parses_language_and_is_short():
    from ytviews.analysis import videos_from_frame

    (v,) = videos_from_frame(
        pd.DataFrame({"title": ["a"], "language": ["en"], "is_short": ["yes"]})
    )
    assert v.language == "en" and v.is_short is True


def test_channel_context_skips_unusable_uploads(fake_youtube, monkeypatch):
    import ytviews.cli as cli

    item = fake_youtube.video_item

    def video_item(vid):
        rec = item(vid)
        if vid == "v000":
            rec["snippet"]["liveBroadcastContent"] = "upcoming"
        if vid == "v001":
            rec["statistics"].pop("viewCount")
        return rec

    fake_youtube.video_item = video_item
    monkeypatch.setattr(cli, "_client", lambda quota_budget: YouTubeClient(service=fake_youtube))
    ctx = cli._channel_context("UC1", n_recent=4)
    assert len(ctx["recent_video_views"]) == 4
    assert all(v is not None for v in ctx["recent_video_views"])
    assert 1000 not in ctx["recent_video_views"] and 1001 not in ctx["recent_video_views"]


def test_drop_thumbnails_can_be_disabled():
    from ytviews.features import drop_thumbnails

    df = pd.DataFrame({"video_id": list("abcd"), "thumbnail_path": ["x"] * 4})
    assert len(drop_thumbnails(df, enabled=False)) == 4


def test_nowcast_recentre_keeps_persistence(trained, synthetic_df):
    from ytviews.features import add_channel_history
    from ytviews.nowcast import recentre

    out, meta = trained
    p = Predictor.load(out)
    nc = meta["nowcast"]
    df = add_channel_history(synthetic_df)
    moved = recentre(nc, p.features, p.regressor, df)
    assert {k: moved[k] for k in ("b", "c", "resid_q")} == {k: nc[k] for k in ("b", "c", "resid_q")}
    assert recentre({"calibrated": False}, p.features, p.regressor, df) == {"calibrated": False}


def test_cache_failures_do_not_break_encoding(monkeypatch):
    import ytviews.cache as cache

    def broken(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cache, "EmbeddingCache", broken)
    out = cache.cached_encode("ns", ["k1", "k2"], ["a", "b"], lambda b: np.ones((len(b), 2)), 2)
    assert out.shape == (2, 2) and (out == 1).all()


def test_clip_marks_unreadable_images_missing_and_does_not_cache_them(monkeypatch, tmp_path):
    from ytviews import images
    from ytviews.synthetic import render_thumbnail

    class FakeST:
        def __init__(self, name, device):
            pass

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, imgs, **kw):
            return np.ones((len(imgs), 3))

    monkeypatch.setenv("YTVIEWS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST)
    )
    images._MODEL_CACHE.clear()
    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    render_thumbnail(0.0, 0.5, 0, good)
    bad.write_bytes(b"not an image")
    enc = images.ClipImageEncoder("fake")
    feats, missing = enc.transform([str(good), str(bad), None])
    assert missing.tolist() == [False, True, True]
    assert (feats[0] == 1).all() and (feats[1] == 0).all()
    from ytviews.cache import EmbeddingCache

    with EmbeddingCache("fake") as c:
        assert c.size() == 1  # only the readable image was cached
    images._MODEL_CACHE.clear()
