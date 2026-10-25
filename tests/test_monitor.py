import pandas as pd

from ytviews.monitor import monitor
from ytviews.predictor import Predictor


def test_monitor_scores_only_newer_videos(trained, synthetic_df):
    model, _ = trained
    p = Predictor.load(model)
    # The fixture model was trained "now", after every synthetic video: nothing to score yet.
    assert monitor(p, synthetic_df).verdict == "no new videos yet"

    cutoff = synthetic_df["published_at"].quantile(0.8)
    result = monitor(p, synthetic_df, since=cutoff)
    newer = synthetic_df.loc[synthetic_df["published_at"] > cutoff, "video_id"].nunique()
    assert result.n_videos == newer > 0
    assert 0 < result.live["mae_log"] < 3
    assert result.reference is not None
    assert result.weekly["n"].sum() == newer
    assert result.in_sample  # those videos were fetched before the model was trained
    assert result.verdict.startswith("not a fair test")


def test_monitor_accepts_naive_dates(trained, synthetic_df):
    model, _ = trained
    result = monitor(Predictor.load(model), synthetic_df, since="2000-01-01")
    assert result.since.tzinfo is not None
    assert result.n_videos == synthetic_df["video_id"].nunique()
    assert isinstance(result.weekly, pd.DataFrame)
