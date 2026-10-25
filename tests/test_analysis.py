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

    # compare_thumbnails must rank by each thumbnail's predicted median. (Whether the model
    # learned that better thumbnails get more views is tested in test_end_to_end.)
    paths = []
    for i, quality in enumerate((-2.5, 0.0, 2.5)):
        path = tmp_path / f"t{i}.jpg"
        render_thumbnail(quality, 0.2, i, path)
        paths.append(str(path))
    ranked = p.compare_thumbnails(video, paths)
    expected = {
        path: p.predict(VideoInput(**{**video.__dict__, "thumbnail_path": path}))[0].median
        for path in paths
    }
    assert ranked["thumbnail"].tolist() == sorted(paths, key=expected.get, reverse=True)
    assert ranked["median"].tolist() == pytest.approx(sorted(expected.values(), reverse=True))
    assert ranked["vs_best"].iloc[0] == 1.0


def test_local_publish_times_are_converted_to_utc(trained):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from ytviews.features import numeric_frame

    model, _ = trained
    p = Predictor.load(model)
    local = VideoInput(
        title="x", published_at=datetime(2025, 3, 3, 9, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    )
    utc = VideoInput(title="x", published_at=datetime(2025, 3, 3, 0, 0, tzinfo=ZoneInfo("UTC")))
    rows = pd.DataFrame([local.to_row(), utc.to_row()])
    num = numeric_frame(rows)
    assert (
        num.iloc[0][["hour_sin", "hour_cos", "dow_sin"]].tolist()
        == num.iloc[1][["hour_sin", "hour_cos", "dow_sin"]].tolist()
    )
    a, b = p.predict([local, utc])
    assert abs(a.median - b.median) < 1e-6 * b.median + 1e-6


def test_channel_report(trained, synthetic_df, tmp_path):
    from ytviews.channel_report import write_channel_report

    model, _ = trained
    latest = synthetic_df.sort_values("fetched_at").drop_duplicates("video_id", keep="last")
    channel = latest["channel_id"].iloc[0]
    out = write_channel_report(
        Predictor.load(model), latest, channel, tmp_path / "c.html", timezone_name="Europe/Paris"
    )
    page = out.read_text()
    assert page.startswith("<!doctype html>") and "Channel Report</title>" in page
    assert "Biggest wins" in page and page.count('class="heat"') == 1
    assert page.count("<circle") == latest["channel_id"].eq(channel).sum()
    with pytest.raises(ValueError):
        write_channel_report(Predictor.load(model), latest, "UCnope", tmp_path / "x.html")
