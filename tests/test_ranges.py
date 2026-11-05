from dataclasses import replace

import numpy as np
import pandas as pd

from ytviews.predictor import Predictor, VideoInput
from ytviews.synthetic import generate_records

BASE = VideoInput(title="I Survived 100 Days in Minecraft", subscribers=100_000, category_id="20")


def test_synthetic_data_includes_young_videos():
    videos, _ = generate_records(n_channels=60, seed=3)
    df = pd.DataFrame(videos)
    age = (pd.to_datetime(df["fetched_at"]) - pd.to_datetime(df["published_at"])).dt.days
    # A real collection of recent uploads, refreshed twice a week, has plenty of these; the
    # default "views after 7 days" question needs them in the training data.
    assert (age < 7).mean() > 0.04
    assert (age < 30).mean() > 0.15
    # extra snapshots only for young videos: a handful per video, not one per refresh for all
    assert len(df) < 3 * df["video_id"].nunique()


def test_inputs_outside_the_training_data_are_flagged(trained):
    p = Predictor.load(trained[0])
    assert p.predict(BASE)[0].notes == ()
    huge = p.predict(replace(BASE, subscribers=1e10))[0]
    assert any("subscriber count" in n and "above" in n for n in huge.notes)
    old = p.predict(replace(BASE, age_days=20_000))[0]
    assert any("video age" in n for n in old.notes)
    assert "notes" in huge.as_dict()


def test_models_saved_without_ranges_give_no_notes(trained):
    p = Predictor.load(trained[0])
    del p.features.ranges_  # as a FeatureBuilder pickled before this existed
    assert p.predict(replace(BASE, subscribers=1e10))[0].notes == ()


def test_lightgbm_follows_the_growth_curve_below_its_youngest_training_age(trained):
    from ytviews.train import BlendedRegressor

    p = Predictor.load(trained[0])
    reg = p.regressor
    assert reg.age_floor is not None
    ages = [1 / 24, 0.25, 1, 3, 7, 30]
    young = [replace(BASE, age_days=a) for a in ages]
    x = p.features.transform(pd.DataFrame([v.to_row() for v in young]))
    extended = reg.predict(x)
    assert (np.diff(extended) >= -1e-6).all()  # still rises with age
    flat = BlendedRegressor(reg.nn, reg.gbm, p.features, reg.weight, reg.interval_adjust,
                            reg.interval_scale, age_floor=None)  # fmt: skip
    plain = flat.predict(x)
    below = x["numeric"][:, 0] < reg.age_floor
    if reg.weight < 1 and below.any():
        assert (extended[below] <= plain[below] + 1e-9).all()  # young videos no longer lifted
        assert np.allclose(extended[~below], plain[~below])  # unchanged inside the range
