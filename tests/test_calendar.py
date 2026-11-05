from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ytviews.cli import app
from ytviews.planner import MAX_CALENDAR_VIDEOS, calendar
from ytviews.predictor import Prediction, Predictor, VideoInput

START = datetime(2030, 1, 7, 6, tzinfo=timezone.utc)


class GapModel:
    """A stand-in model: every hour is equal and views grow with the gap until 7 days."""

    def publish_times(self, video, timezone_name="UTC", start=None):
        day0 = start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        slots = [day0 + timedelta(hours=h) for h in range(7 * 24)]
        return pd.DataFrame(
            {
                "weekday": [t.strftime("%A") for t in slots],
                "weekday_index": [t.weekday() for t in slots],
                "hour": [t.hour for t in slots],
                "published_at": slots,
                "median": [1000.0 + (t.hour == 15) for t in slots],
                "vs_best": 1.0,
            }
        )

    def predict(self, videos):
        out = []
        for v in videos:
            gap = 7.0 if v.days_since_last_upload is None else v.days_since_last_upload
            m = float(v.subscribers) * min(gap, 7.0) / 7.0
            out.append(Prediction(m / 2, m, m * 2, 0.8))
        return out


def test_calendar_spaces_uploads_when_gaps_matter():
    videos = [
        VideoInput(title=t, subscribers=s) for t, s in (("a", 1000), ("b", 5000), ("c", 2000))
    ]
    last = START - timedelta(days=1)
    plan = calendar(GapModel(), videos, days=21, last_upload=last, start=START)
    assert sorted(plan["title"]) == ["a", "b", "c"]
    assert plan["published_at"].is_monotonic_increasing
    assert (plan["gap_days"] >= 7 - 1e-9).all()  # waiting a week each time is worth it here
    assert (plan["hour"] == 15).all()  # each video's best hour


def test_calendar_respects_the_minimum_gap_and_widens_the_horizon():
    videos = [VideoInput(title=str(i), subscribers=1000) for i in range(4)]
    plan = calendar(GapModel(), videos, days=3, min_gap_days=2, start=START)
    dates = plan["published_at"].map(lambda t: t.date())
    assert len(plan) == 4 and (np.diff([d.toordinal() for d in dates]) >= 2).all()


def test_calendar_rejects_bad_requests():
    model = GapModel()
    with pytest.raises(ValueError, match="no videos"):
        calendar(model, [])
    with pytest.raises(ValueError, match="at most"):
        calendar(model, [VideoInput(title="x")] * (MAX_CALENDAR_VIDEOS + 1))
    with pytest.raises(ValueError, match="min_gap"):
        calendar(model, [VideoInput(title="x")], min_gap_days=0)


def test_calendar_with_a_real_model_and_cli(trained, tmp_path):
    p = Predictor.load(trained[0])
    videos = [VideoInput(title=t, subscribers=20_000, recent_video_views=[5_000, 7_000],
                         recent_video_age_days=[3, 10]) for t in ("One", "Two")]  # fmt: skip
    plan = calendar(p, videos, days=7, start=START, timezone_name="Europe/Paris")
    assert len(plan) == 2 and plan["median"].gt(0).all()
    src = tmp_path / "planned.csv"
    src.write_text("title,subscribers\nA video,1000\nAnother video,1000\n")
    out = tmp_path / "calendar.csv"
    args = ["calendar", str(src), "--min-gap", "2", "--last-upload", "2030-01-01",
            "-m", str(trained[0]), "--out", str(out)]  # fmt: skip
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert "Total:" in r.output and len(pd.read_csv(out)) == 2
    r = CliRunner().invoke(app, ["calendar", str(src), "--last-upload", "soon",
                                 "-m", str(trained[0])])  # fmt: skip
    assert r.exit_code == 2


def test_gap_feature_uses_only_earlier_uploads():
    from ytviews.features import add_channel_history

    t = pd.to_datetime
    df = pd.DataFrame({
        "video_id": ["a", "b", "c"], "channel_id": ["c"] * 3, "views": [1.0, 2.0, 3.0],
        "published_at": t(["2030-01-01", "2030-01-04", "2030-01-14"], utc=True),
        "fetched_at": t(["2030-02-01"] * 3, utc=True),
        "channel_views": [10.0] * 3, "channel_video_count": [3.0] * 3,
    })  # fmt: skip
    gaps = add_channel_history(df).set_index("video_id")["gap_days"]
    assert np.isnan(gaps["a"]) and gaps["b"] == 3 and gaps["c"] == 10
    later = df.copy()
    later.loc[2, "published_at"] = t("2030-01-05", utc=True)  # moving the newest upload ...
    moved = add_channel_history(later).set_index("video_id")["gap_days"]
    assert moved["b"] == gaps["b"]  # ... changes nothing for the ones before it
    future = VideoInput(title="x", recent_video_views=[10], recent_video_age_days=[2],
                        published_at=datetime.now(timezone.utc) + timedelta(days=3))  # fmt: skip
    assert future.to_row()["gap_days"] == pytest.approx(5, abs=0.01)
    assert VideoInput(title="x", days_since_last_upload=4).to_row()["gap_days"] == 4
