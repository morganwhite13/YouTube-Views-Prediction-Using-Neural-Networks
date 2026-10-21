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
    effects = dict(zip(ex.effects["group"], ex.effects["factor"], strict=True))
    assert effects["channel size"] > 1.5  # a 2M-subscriber channel is well above average
    assert len(ex.summary(top=3)) == 3


def test_weak_title_gets_negative_effect(trained):
    model, _ = trained
    p = Predictor.load(model)
    base = dict(subscribers=100_000, category_id="20")
    good = p.explain(VideoInput(title="I Survived 100 Days in Minecraft", **base)).effects
    bad = p.explain(VideoInput(title="Weekly Minecraft Update", **base)).effects
    get = lambda e: e.set_index("group").loc["title text", "log_effect"]  # noqa: E731
    assert get(good) > get(bad)
