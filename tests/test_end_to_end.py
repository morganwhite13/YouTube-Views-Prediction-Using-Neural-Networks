"""Train on synthetic data and check that the model learned the planted effects."""

import numpy as np
import pytest

from ytviews.pipeline import NEURAL_NAME
from ytviews.predictor import Predictor, VideoInput


def test_beats_naive_baseline(trained):
    _, meta = trained
    ours = meta["metrics"][NEURAL_NAME]
    assert ours["mae_log"] < 0.7 * meta["metrics"]["global median"]["mae_log"]
    assert ours["spearman"] > 0.7
    assert 0.5 < ours["interval_coverage"] <= 1.0


def test_report_and_artifacts_written(trained):
    out, meta = trained
    for name in ("model.pt", "features.joblib", "meta.json", "report.md", "test_predictions.csv"):
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
    assert pred_big.median > 10 * pred_small.median  # channel size matters a lot
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
