import shutil

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ytviews.cli import app
from ytviews.compare import (
    MIN_VIDEOS,
    compare_models,
    gate_candidate,
    paired_comparison,
    should_promote,
)
from ytviews.predictor import Predictor


def test_paired_comparison_verdicts():
    rng = np.random.default_rng(0)
    y = rng.normal(8, 2, 200)
    good, bad = y + rng.normal(0, 0.3, 200), y + rng.normal(0, 1.0, 200)
    assert paired_comparison(y, bad, good).verdict == "B is better"
    assert paired_comparison(y, good, bad).verdict == "A is better"
    assert paired_comparison(y, good, good).verdict == "no significant difference"
    few = paired_comparison(y[:5], bad[:5], good[:5])
    assert few.n == 5 and "too few" in few.verdict and should_promote(few)


def _future(df: pd.DataFrame, p: Predictor) -> pd.DataFrame:
    """The synthetic videos, moved to after the model was trained (so it hasn't seen them)."""
    shift = pd.Timestamp(p.meta["created_at"]) - df["published_at"].min() + pd.Timedelta(days=1)
    return df.assign(published_at=df["published_at"] + shift, fetched_at=df["fetched_at"] + shift)


def test_a_model_compared_with_itself_is_a_tie(trained, synthetic_df):
    p = Predictor.load(trained[0])
    r = compare_models(p, p, _future(synthetic_df, p))
    assert r.n >= MIN_VIDEOS and r.diff == 0 and r.verdict == "no significant difference"
    # videos the models trained on are excluded unless asked for, with a warning
    assert compare_models(p, p, synthetic_df).n == 0
    assert "may have trained" in compare_models(p, p, synthetic_df, since="2000-01-01").note


@pytest.mark.parametrize(("factor", "promote"), [(1.0, True), (300.0, False)])
def test_the_gate_keeps_the_current_model_when_the_candidate_is_worse(
    trained, synthetic_df, tmp_path, factor, promote
):
    p = Predictor.load(trained[0])
    df = _future(synthetic_df, p)
    held_out = df.sample(frac=0.3, random_state=0)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    pd.DataFrame({  # perfect held-out predictions, or ones 300x too high
        "video_id": held_out["video_id"], "fetched_at": held_out["fetched_at"],
        "views": held_out["views"], "pred_median": held_out["views"] * factor,
    }).to_csv(candidate / "test_predictions.csv", index=False)  # fmt: skip
    r = gate_candidate(p, candidate, df)
    assert r.n >= MIN_VIDEOS
    assert should_promote(r) is promote


def test_compare_models_cli(trained, synthetic_db, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(trained[0], other)
    r = CliRunner().invoke(
        app, ["compare-models", str(trained[0]), str(other), "--db", str(synthetic_db),
              "--since", "2000-01-01"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "Verdict: no significant difference" in r.output and "may have trained" in r.output
