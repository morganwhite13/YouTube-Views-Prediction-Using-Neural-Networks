import numpy as np
import pandas as pd

from ytviews.features import (
    NUMERIC_FEATURES,
    FeatureBuilder,
    add_channel_history,
    history_from_recent,
    numeric_frame,
)


def _frame(views):
    n = len(views)
    return pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(n)],
            "channel_id": "UC1",
            "title": [f"Title {i}" for i in range(n)],
            "description": "",
            "tags": "",
            "category_id": "20",
            "published_at": pd.date_range("2024-01-01", periods=n, freq="7D", tz="UTC"),
            "fetched_at": pd.Timestamp("2024-06-01", tz="UTC"),
            "views": views,
            "subscribers": 1000.0,
            "channel_views": float(sum(views)),
            "channel_video_count": float(n),
            "duration_s": 300.0,
            "caption": 0.0,
            "definition": "hd",
        }
    )


def test_channel_history_uses_only_earlier_uploads():
    df = add_channel_history(_frame([100, 1000, 10_000, 100_000]))
    df = df.sort_values("published_at")
    assert np.isnan(df["hist_median"].iloc[0])
    assert df["hist_count"].tolist() == [0, 1, 2, 3]
    assert np.isclose(df["hist_median"].iloc[1], np.log1p(100))
    assert np.isclose(df["hist_max"].iloc[3], np.log1p(10_000))  # never sees its own views


def test_channel_average_excludes_the_video_itself():
    df = add_channel_history(_frame([100, 300]))
    assert df.set_index("video_id").loc["v0", "channel_avg_views"] == 300
    assert df.set_index("video_id").loc["v1", "channel_avg_views"] == 100


def test_history_from_recent_matches_training_definition():
    h = history_from_recent([100, 1000], [10, 20])
    assert np.isclose(h["hist_median"], np.median(np.log1p([100, 1000])))
    assert h["hist_count"] == 2
    empty = history_from_recent([])
    assert empty["hist_count"] == 0 and np.isnan(empty["hist_median"])


def test_numeric_frame_shape_and_title_flags():
    df = add_channel_history(_frame([10, 20]))
    df.loc[0, "title"] = "$1 VS $1,000 WATER?!"
    num = numeric_frame(df)
    assert list(num.columns) == NUMERIC_FEATURES
    row = num.iloc[0]
    assert row["title_has_money"] == 1 and row["title_has_vs"] == 1
    assert row["title_has_question"] == 1 and row["title_has_exclaim"] == 1
    assert row["title_caps_ratio"] == 1.0


def test_feature_builder_is_fit_on_train_only(synthetic_df):
    df = add_channel_history(synthetic_df)
    train, test = df.iloc[:400], df.iloc[400:500].copy()
    fb = FeatureBuilder(text_encoder="tfidf", text_dim=16).fit(train)
    test.loc[:, "category_id"] = "999"  # unseen category -> index 0
    x = fb.transform(test)
    assert x["numeric"].shape == (100, len(NUMERIC_FEATURES))
    assert x["title"].shape == (100, fb.embed_dims["title"])
    assert (x["category"] == 0).all()
    assert np.isfinite(x["numeric"]).all()
