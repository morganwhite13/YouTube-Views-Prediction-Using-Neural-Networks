"""Train on synthetic data and check that the model learned the planted effects."""

import numpy as np
import pytest

from ytviews.evaluate import OURS_NAME as NEURAL_NAME
from ytviews.predictor import Predictor, VideoInput


def test_beats_naive_baseline(trained):
    _, meta = trained
    ours = meta["metrics"][NEURAL_NAME]
    assert ours["mae_log"] < 0.7 * meta["metrics"]["global median"]["mae_log"]
    assert ours["spearman"] > 0.7
    assert 0.5 < ours["interval_coverage"] <= 1.0


def test_report_and_artifacts_written(trained):
    out, meta = trained
    for name in ("model.pt", "features.joblib", "meta.json", "report.md", "MODEL_CARD.md",
                 "test_predictions.csv"):  # fmt: skip
        assert (out / name).exists(), name
    assert "channel size" in meta["importance"]


def test_save_load_roundtrip_and_learned_effects(trained):
    out, _ = trained
    p = Predictor.load(out)
    small = VideoInput(
        title="I Survived 100 Days in Minecraft", subscribers=1_000, category_id="20"
    )
    big = VideoInput(
        title="I Survived 100 Days in Minecraft", subscribers=5_000_000, category_id="20"
    )
    pred_small, pred_big = p.predict([small, big])
    assert pred_big.median > 20 * pred_small.median  # channel size matters a lot
    assert pred_small.low <= pred_small.median <= pred_small.high

    ranked = p.compare_titles(
        VideoInput(title="", subscribers=200_000, category_id="20"),
        ["Weekly Minecraft Update", "I Survived 100 Days in Minecraft"],
    )
    assert ranked.iloc[0]["title"] == "I Survived 100 Days in Minecraft"

    traj = p.trajectory(small)
    assert np.all(np.diff(traj["median"]) >= 0)


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ytviews train"):
        Predictor.load(tmp_path)


def test_thumbnail_changes_prediction(trained, synthetic_db, tmp_path):
    from ytviews.synthetic import render_thumbnail

    out, meta = trained
    assert meta["image_encoder"] == "stats"
    p = Predictor.load(out)
    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    render_thumbnail(2.5, 0.6, 1, good)
    render_thumbnail(-2.5, 0.6, 1, bad)
    base = dict(title="I Survived 100 Days in Minecraft", subscribers=100_000, category_id="20")
    with_good, with_bad, without = p.predict(
        [
            VideoInput(**base, thumbnail_path=str(good)),
            VideoInput(**base, thumbnail_path=str(bad)),
            VideoInput(**base),
        ]
    )
    assert with_good.median > with_bad.median
    assert with_bad.median < without.median < with_good.median * 1.5


def test_blend_is_saved_and_reloaded(trained, synthetic_df):
    from ytviews.features import add_channel_history
    from ytviews.train import BlendedRegressor

    out, meta = trained
    assert 0.0 <= meta["blend_weight"] <= 1.0
    assert "neural net only" in meta["metrics"]
    p = Predictor.load(out)
    assert isinstance(p.regressor, BlendedRegressor) and (out / "gbm.joblib").exists()

    x = p.features.transform(add_channel_history(synthetic_df).head(20))
    pure = BlendedRegressor(p.regressor.nn, p.regressor.gbm, p.features, 1.0)
    assert np.allclose(pure.predict_quantiles(x), p.regressor.nn.predict_quantiles(x))
    q = p.regressor.predict_quantiles(x)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()


def test_report_plot_leaves_the_matplotlib_backend_alone(tmp_path):
    """Writing report.png used to force the Agg backend, which broke notebook plots."""
    matplotlib = pytest.importorskip("matplotlib")
    import pandas as pd

    from ytviews.pipeline import _maybe_plot

    previous = matplotlib.get_backend()
    matplotlib.use("svg")
    try:
        preds = pd.DataFrame(
            {"views": [10, 100], "pred_median": [12, 90], "pred_low": [5, 50],
             "pred_high": [20, 200]}
        )  # fmt: skip
        _maybe_plot(tmp_path, preds, {"title text": 0.1})
        assert (tmp_path / "report.png").exists()
        assert matplotlib.get_backend() == "svg"
    finally:
        matplotlib.use(previous)


def test_blend_range_recalibration_survives_save_and_load(trained):
    out, meta = trained
    p = Predictor.load(out)
    assert np.isfinite(p.regressor.interval_adjust)
    ours = meta["metrics"]["ytviews (ours)"]
    assert 0.6 <= ours["interval_coverage"] <= 0.97  # close to the nominal 80%
