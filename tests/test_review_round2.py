"""Regression tests for the second whole-package review."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ytviews.predictor import Predictor, VideoInput

VIDEO = VideoInput(title="I Survived 100 Days in Minecraft", subscribers=100_000, category_id="20")


def test_csv_flag_columns_read_as_float_are_parsed():
    from ytviews.analysis import videos_from_frame

    df = pd.DataFrame({"title": ["a", "b"], "caption": [1.0, np.nan], "hd": [0.0, 1.0]})
    a, b = videos_from_frame(df)
    assert a.caption is True and a.hd is False
    assert b.hd is True  # a blank caption cell keeps the default


def test_stored_video_keeps_its_definition_language_and_missing_caption():
    from ytviews.channel_report import _video_from_row

    row = pd.Series({
        "title": "t", "description": "", "tags": "", "category_id": "20", "duration_s": 600.0,
        "published_at": pd.Timestamp("2026-01-01", tz="UTC"), "subscribers": 1000.0,
        "channel_video_count": 10.0, "caption": np.nan, "definition": "sd",
        "default_language": "de", "is_short": np.nan,
    })  # fmt: skip
    when = pd.Series(dtype="datetime64[ns, UTC]")
    empty = pd.DataFrame(
        {"published_at": when, "fetched_at": when, "views": pd.Series(dtype=float)}
    )
    v = _video_from_row(row, empty)
    assert v.caption is False and v.hd is False and v.language == "de"


def test_track_labels_forecast_days_in_order(trained):
    p = Predictor.load(trained[0])
    shuffled = p.track(VIDEO, 5000, 1.0, days=(30, 7)).forecast
    ordered = p.track(VIDEO, 5000, 1.0, days=(7, 30)).forecast
    assert shuffled["days"].tolist() == [7, 30]
    assert np.allclose(shuffled["median"], ordered["median"])


def test_monitor_sees_videos_trained_on_through_earlier_snapshots(trained, synthetic_df):
    from ytviews.monitor import monitor

    p = Predictor.load(trained[0])
    trained_at = pd.Timestamp(p.meta["created_at"])
    df = synthetic_df.copy()
    # every video's newest snapshot moves past training time; older ones stay before it
    newest = df.groupby("video_id")["fetched_at"].transform("max") == df["fetched_at"]
    shift = trained_at + pd.Timedelta(days=30) - df["fetched_at"].max()
    df.loc[newest, "fetched_at"] += shift
    since = df["published_at"].quantile(0.8)
    assert monitor(p, df, since=since).in_sample


def test_publish_times_skip_the_hour_lost_to_daylight_saving(trained):
    p = Predictor.load(trained[0])
    start = datetime(2027, 3, 12, 12, tzinfo=timezone.utc)  # NY springs forward on 14 March
    grid = p.publish_times(VIDEO, "America/New_York", start=start)
    instants = grid["published_at"].map(lambda t: t.astimezone(timezone.utc))
    assert instants.is_unique  # no hour evaluated twice under two labels
    sunday = grid[grid["published_at"].map(lambda t: t.date().isoformat()) == "2027-03-14"]
    assert 2 not in set(sunday["hour"]) and len(grid) == 7 * 24 - 1


def test_schedule_handles_a_day_with_23_hours(trained):
    from ytviews.planner import schedule

    p = Predictor.load(trained[0])
    start = datetime(2027, 3, 12, 12, tzinfo=timezone.utc)
    plan = schedule(p, [VIDEO, VideoInput(title="Weekly Update", subscribers=100_000)],
                    timezone_name="America/New_York", days=4, start=start)  # fmt: skip
    assert plan["published_at"].map(lambda t: t.date()).is_unique
