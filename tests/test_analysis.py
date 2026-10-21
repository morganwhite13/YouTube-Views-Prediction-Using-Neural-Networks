import pandas as pd
import pytest

from ytviews.analysis import outliers, predict_table, videos_from_frame
from ytviews.predictor import Predictor, VideoInput


def test_outliers_for_one_channel(trained, synthetic_df):
    model, _ = trained
    p = Predictor.load(model)
    latest = synthetic_df.sort_values("fetched_at").drop_duplicates("video_id", keep="last")
    channel = latest["channel_id"].iloc[0]
    table = outliers(p, latest, channel)
    assert set(table["video_id"]) <= set(latest.loc[latest["channel_id"] == channel, "video_id"])
    assert table["factor"].is_monotonic_decreasing
    assert set(table["verdict"]) <= {"beat expectations", "below expectations", "as expected"}
    with pytest.raises(ValueError):
        outliers(p, latest, "UCnope")


def test_videos_from_frame_parses_lists_and_types():
    df = pd.DataFrame(
        {
            "title": ["A", "B"],
            "subscribers": [1000, None],
            "tags": ["x|y", ""],
            "recent_video_views": ["100;200", None],
            "category_id": ["20", "24"],
            "caption": ["yes", "0"],
        }
    )
    a, b = videos_from_frame(df)
    assert a.tags == ["x", "y"] and a.recent_video_views == [100.0, 200.0] and a.caption
    assert b.subscribers is None and b.tags == [] and not b.caption
    assert isinstance(a, VideoInput)
    with pytest.raises(ValueError, match="unknown columns"):
        videos_from_frame(df.assign(colour="red"))


def test_predict_table(trained):
    model, _ = trained
    out = predict_table(Predictor.load(model), pd.DataFrame({"title": ["A", "B"]}))
    assert {"pred_low", "pred_median", "pred_high"} <= set(out.columns)
    assert (out["pred_low"] <= out["pred_median"]).all()


def test_publish_times_and_thumbnail_compare(trained, synthetic_db, tmp_path):
    from ytviews.synthetic import render_thumbnail

    model, _ = trained
    p = Predictor.load(model)
    video = VideoInput(title="I Survived 100 Days", subscribers=100_000, category_id="20")
    grid = p.publish_times(video, "America/New_York")
    assert len(grid) == 168 and grid["hour"].between(0, 23).all()
    assert grid["weekday"].nunique() == 7 and grid["vs_best"].max() == 1.0

    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    render_thumbnail(2.5, 0.2, 0, good)
    render_thumbnail(-2.5, 0.2, 0, bad)
    ranked = p.compare_thumbnails(video, [str(bad), str(good)])
    assert ranked.iloc[0]["thumbnail"] == str(good)
