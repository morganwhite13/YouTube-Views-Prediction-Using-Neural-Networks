import sqlite3
import urllib.error

import numpy as np
import pandas as pd
import pytest

from ytviews.collect import detect_shorts, is_short
from ytviews.features import FeatureBuilder, add_channel_history, numeric_frame
from ytviews.store import SCHEMA, VideoStore


class FakeResponse:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Short ids answer 200, 'w' ids redirect (as HTTPError without a redirect handler)."""

    def open(self, request, timeout):
        vid = request.full_url.rsplit("/", 1)[1]
        assert request.get_method() == "HEAD"
        if vid.startswith("s"):
            return FakeResponse(200)
        if vid.startswith("w"):
            raise urllib.error.HTTPError(request.full_url, 303, "See Other", {}, None)
        raise OSError("network down")


def test_is_short_statuses():
    opener = FakeOpener()
    assert is_short("s1", opener) is True
    assert is_short("w1", opener) is False
    assert is_short("x1", opener) is None


def test_old_database_is_migrated_and_flags_stored(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)  # the schema as it was before the is_short column existed
    conn.executemany(
        "INSERT INTO videos (video_id, title) VALUES (?, ?)",
        [("s1", "a"), ("w1", "b"), ("x1", "c")],
    )
    conn.commit()
    conn.close()
    with VideoStore(path) as store:
        assert sorted(store.video_ids_without_short_flag()) == ["s1", "w1", "x1"]
        counts = detect_shorts(store, opener=FakeOpener(), workers=2)
        assert counts == {"shorts": 1, "not_shorts": 1, "unknown": 1}
        assert store.video_ids_without_short_flag() == ["x1"]  # unknown stays unchecked
        flags = dict(store.conn.execute("SELECT video_id, is_short FROM videos"))
    assert flags == {"s1": 1, "w1": 0, "x1": None}


def _frame(**cols):
    n = len(next(iter(cols.values())))
    base = {
        "video_id": [f"v{i}" for i in range(n)], "channel_id": "UC1", "title": "t",
        "description": "", "tags": "", "category_id": "20",
        "published_at": pd.Timestamp("2024-01-01", tz="UTC"),
        "fetched_at": pd.Timestamp("2024-02-01", tz="UTC"), "views": 100.0,
        "subscribers": 1000.0, "channel_views": 1e5, "channel_video_count": 10.0,
        "duration_s": 60.0, "caption": 0.0, "definition": "hd",
    }  # fmt: skip
    return pd.DataFrame(base | cols)


def test_detected_shorts_flag_beats_the_length_rule():
    df = _frame(duration_s=[60.0, 60.0, 600.0], is_short=[0.0, np.nan, 1.0])
    num = numeric_frame(add_channel_history(df))
    assert num["is_short"].tolist() == [0.0, 1.0, 1.0]
    assert num["short_known"].tolist() == [1.0, 0.0, 1.0]


def test_language_one_hot():
    df = _frame(default_language=["en-US", "es-419", "sw", "", "PT"])
    num = numeric_frame(add_channel_history(df))
    assert num.loc[0, "lang_en"] == 1 and num.loc[1, "lang_es"] == 1
    assert num.loc[2, "lang_other"] == 1 and num.loc[3, "lang_missing"] == 1
    assert num.loc[4, "lang_pt"] == 1
    lang_cols = [c for c in num.columns if c.startswith("lang_")]
    assert (num[lang_cols].sum(axis=1) == 1).all()


def test_models_from_an_older_feature_set_ask_for_retraining(synthetic_df):
    df = add_channel_history(synthetic_df).head(50)
    fb = FeatureBuilder(text_encoder="tfidf", text_dim=8, image_encoder="none").fit(df)
    del fb.feature_names_  # what an older pickle looks like
    with pytest.raises(RuntimeError, match="Retrain"):
        fb.transform(df)


def test_synthetic_data_has_languages_and_short_flags(synthetic_df):
    assert synthetic_df["default_language"].nunique() > 1
    assert synthetic_df["is_short"].notna().all()


def test_export_csv(synthetic_db, tmp_path):
    from typer.testing import CliRunner

    from ytviews.cli import app

    out = tmp_path / "data.csv"
    r = CliRunner().invoke(app, ["export", str(out), "--db", str(synthetic_db), "--features"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(out)
    assert {"title", "views", "f_log_subscribers", "f_lang_en"} <= set(df.columns)
    bad = CliRunner().invoke(app, ["export", str(tmp_path / "x.txt"), "--db", str(synthetic_db)])
    assert bad.exit_code != 0
