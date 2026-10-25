import json

import pytest
from typer.testing import CliRunner

from ytviews.cli import app
from ytviews.collect import collect_channels, recently_fetched
from ytviews.images import attach_thumbnails, thumbnail_dir_for
from ytviews.pipeline import EXPERIMENTS_FILE, run_experiment
from ytviews.predictor import Predictor, VideoInput
from ytviews.store import VideoStore
from ytviews.train import TrainConfig
from ytviews.youtube import YouTubeClient


@pytest.fixture(scope="module")
def likes_model(synthetic_db, synthetic_df, tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "likes"
    df = attach_thumbnails(synthetic_df, thumbnail_dir_for(synthetic_db))
    meta = run_experiment(
        df, out, text_encoder="tfidf", image_encoder="none", target="likes",
        config=TrainConfig(ensemble=1, epochs=30, patience=5), log=lambda _: None,
    )  # fmt: skip
    return out, meta


def test_likes_model_trains_and_predicts(likes_model, synthetic_df):
    out, meta = likes_model
    assert meta["target"] == "likes" and meta["nowcast"] is None
    hidden = synthetic_df["likes"].isna().sum()
    assert hidden > 0  # some synthetic videos hide their likes; those rows are dropped
    p = Predictor.load(out)
    assert p.target == "likes"
    small, big = p.predict(
        [VideoInput(title="x", subscribers=s, category_id="20") for s in (5_000, 5_000_000)]
    )
    assert big.median > 5 * small.median
    # Likes are a few percent of views, so a likes model predicts far smaller numbers.
    views_scale = synthetic_df["views"].median()
    assert small.median < views_scale


def test_views_only_tools_refuse_other_targets(likes_model, synthetic_df):
    from ytviews.analysis import outliers
    from ytviews.monitor import monitor

    p = Predictor.load(likes_model[0])
    with pytest.raises(ValueError, match="views model"):
        p.track(VideoInput(title="x"), 100, 1)
    with pytest.raises(ValueError, match="views model"):
        outliers(p, synthetic_df)
    with pytest.raises(ValueError, match="views model"):
        monitor(p, synthetic_df)


def test_cli_wording_and_experiment_log(likes_model):
    out, _ = likes_model
    r = CliRunner().invoke(app, ["predict", "--title", "x", "--model", str(out)])
    assert r.exit_code == 0 and " likes" in r.output, r.output
    log = out.parent / EXPERIMENTS_FILE
    record = json.loads(log.read_text().splitlines()[-1])
    assert record["target"] == "likes" and record["model_dir"] == str(out)
    r = CliRunner().invoke(app, ["experiments", str(out.parent)])
    assert r.exit_code == 0 and "likes" in r.output, r.output


def test_unknown_target_is_rejected(synthetic_df, tmp_path):
    with pytest.raises(ValueError, match="unknown target"):
        run_experiment(synthetic_df, tmp_path, target="dislikes", log=lambda _: None)


def test_recently_collected_channels_are_skipped(fake_youtube, tmp_path):
    client = YouTubeClient(service=fake_youtube)
    with VideoStore(tmp_path / "db.sqlite") as store:
        assert collect_channels(client, store, ["UC1"], max_videos_per_channel=5) > 0
        used = client.quota_used
        assert "UC1" in recently_fetched(store, 1)
        assert collect_channels(client, store, ["UC1"], skip_fresh_days=1) == 0
        assert client.quota_used == used  # nothing fetched
        assert collect_channels(client, store, ["UC1"], max_videos_per_channel=5) > 0


def test_reports_name_the_target(likes_model, tmp_path):
    from ytviews.report_html import write_html_report

    out, _ = likes_model
    page = write_html_report(out, tmp_path / "r.html").read_text()
    assert "How many likes will a video get?" in page and "actual likes" in page
    assert "Target: `likes`" in (out / "report.md").read_text()


def test_api_with_a_likes_model(likes_model):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ytviews.api import create_app

    c = TestClient(create_app(likes_model[0]))
    assert c.get("/model").json()["target"] == "likes"
    r = c.post("/track", json={"video": {"title": "x"}, "observed_views": 5, "observed_hours": 2})
    assert r.status_code == 422 and "views model" in r.text
