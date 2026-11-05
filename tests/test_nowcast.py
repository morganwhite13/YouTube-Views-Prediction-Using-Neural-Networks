import numpy as np
import pandas as pd

from ytviews.nowcast import FALLBACK, snapshot_pairs
from ytviews.predictor import Predictor, VideoInput

VIDEO = VideoInput(title="I Survived 100 Days in Minecraft", subscribers=100_000, category_id="20")


def test_snapshot_pairs_pick_earliest_and_latest():
    t = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame(
        {
            "video_id": ["a", "a", "a", "b"],
            "published_at": [t] * 4,
            "fetched_at": [t + pd.Timedelta(days=d) for d in (2, 5, 30, 10)],
            "views": [10, 20, 30, 5],
        }
    )
    early, late = snapshot_pairs(df)
    assert early["video_id"].tolist() == ["a"]  # b has a single snapshot
    assert early["views"].tolist() == [10] and late["views"].tolist() == [30]


def test_model_ships_a_calibrated_nowcast(trained):
    _, meta = trained
    nc = meta["nowcast"]
    assert nc["calibrated"] and nc["n_pairs"] >= 20
    assert 0.3 < nc["b"] < 1.5  # early surprises largely persist in the synthetic data
    assert nc["resid_q"][0] < 0 < nc["resid_q"][1]


def test_track_forecast(trained):
    model, _ = trained
    p = Predictor.load(model)
    expected_now = p.predict(VideoInput(**{**VIDEO.__dict__, "age_days": 1.0}))[0].median
    ahead = p.track(VIDEO, observed_views=expected_now * 4, observed_age_days=1.0)
    behind = p.track(VIDEO, observed_views=expected_now / 4, observed_age_days=1.0)
    assert np.isclose(ahead.vs_expected, 4, rtol=0.05)
    f = ahead.forecast
    assert f["days"].min() > 1 and (f["low"] >= ahead.observed_views).all()
    assert (np.diff(f["median"]) >= 0).all() and (f["low"] <= f["median"]).all()
    assert ahead.forecast["median"].iloc[-1] > behind.forecast["median"].iloc[-1]


def test_track_without_calibration_falls_back(trained):
    model, _ = trained
    p = Predictor.load(model)
    p.meta["nowcast"] = dict(FALLBACK)
    nc = p.track(VIDEO, observed_views=1000, observed_age_days=2)
    assert not nc.calibrated and len(nc.forecast) > 0


def test_early_views_beat_the_model_alone_on_held_out_pairs(synthetic_df, trained):
    import zlib

    from ytviews.features import add_channel_history, log_views
    from ytviews.nowcast import fit_nowcast, snapshot_pairs

    p = Predictor.load(trained[0])
    df = add_channel_history(synthetic_df).reset_index(drop=True)
    half = df["video_id"].map(lambda v: zlib.crc32(v.encode()) % 2)
    nc = fit_nowcast(p.features, p.regressor, df[half == 0])
    early, late = snapshot_pairs(df[half == 1])  # pairs the calibration never saw
    assert len(early) >= 20
    pe = p.regressor.predict(p.features.transform(early))
    pl = p.regressor.predict(p.features.transform(late))
    yl = log_views(late["views"].to_numpy())
    e1 = log_views(early["views"].to_numpy()) - pe
    ages = (early["fetched_at"] - early["published_at"]).dt.total_seconds().to_numpy() / 86400
    nowcast = pl + nc["a"] + e1 * (nc["b"] + nc["c"] * np.log1p(ages))
    assert np.abs(nowcast - yl).mean() < 0.6 * np.abs(pl - yl).mean()
