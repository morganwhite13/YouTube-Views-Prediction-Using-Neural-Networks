from dataclasses import replace

import numpy as np

from ytviews.predictor import Predictor, VideoInput


def test_explanation_adds_up(trained):
    model, _ = trained
    p = Predictor.load(model)
    video = VideoInput(
        title="I Survived 100 Days in Minecraft",
        subscribers=2_000_000,
        category_id="20",
        recent_video_views=[500_000, 800_000],
    )
    ex = p.explain(video)
    # Shapley efficiency: the effects exactly account for prediction vs baseline
    total = ex.effects["log_effect"].sum()
    assert np.isclose(np.log1p(ex.baseline_views) + total, np.log1p(ex.predicted_views), atol=1e-4)
    assert np.isclose(ex.predicted_views, p.predict(video)[0].median, rtol=1e-4)
    assert "video age" not in set(ex.effects["group"])
    assert len(ex.summary(top=3)) == 3

    # A bigger channel gets a bigger channel-size effect. (The absolute factor for 2M subscribers
    # varies from about 1.45x to 2.6x between training seeds of this small test model, so the
    # test checks the ordering and the size of the gap rather than one fixed value.)
    def channel_size(subs):
        e = p.explain(replace(video, subscribers=subs)).effects
        return e.set_index("group").loc["channel size", "factor"]

    small, mid, big = (channel_size(s) for s in (20_000, 200_000, 2_000_000))
    assert small < mid < big
    assert big > 1.2  # a 2M-subscriber channel is above average
    assert big / small > 2


def test_weak_title_gets_negative_effect(trained):
    model, _ = trained
    p = Predictor.load(model)
    base = dict(subscribers=100_000, category_id="20")
    good = p.explain(VideoInput(title="I Survived 100 Days in Minecraft", **base)).effects
    bad = p.explain(VideoInput(title="Weekly Minecraft Update", **base)).effects
    get = lambda e: e.set_index("group").loc["title text", "log_effect"]  # noqa: E731
    assert get(good) > get(bad)
