"""Regression tests for the second whole-package review."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ytviews.predictor import Predictor, VideoInput

VIDEO = VideoInput(title="I Survived 100 Days in Minecraft", subscribers=100_000, category_id="20")


def test_csv_flag_columns_read_as_float_are_parsed():
    from ytviews.analysis import videos_from_frame

    df = pd.DataFrame({"title": ["a", "b"], "caption": [1.0, np.nan], "hd": [0.0, 1.0]})
    a, b = videos_from_frame(df)
    assert a.caption is True and a.hd is False
    assert b.hd is True  # a blank caption cell keeps the default


def test_stored_video_keeps_its_definition_language_and_missing_caption():
    from ytviews.channel_report import _video_from_row

    row = pd.Series({
        "title": "t", "description": "", "tags": "", "category_id": "20", "duration_s": 600.0,
        "published_at": pd.Timestamp("2026-01-01", tz="UTC"), "subscribers": 1000.0,
        "channel_video_count": 10.0, "caption": np.nan, "definition": "sd",
        "default_language": "de", "is_short": np.nan,
    })  # fmt: skip
    when = pd.Series(dtype="datetime64[ns, UTC]")
    empty = pd.DataFrame(
        {"published_at": when, "fetched_at": when, "views": pd.Series(dtype=float)}
    )
    v = _video_from_row(row, empty)
    assert v.caption is False and v.hd is False and v.language == "de"


def test_track_labels_forecast_days_in_order(trained):
    p = Predictor.load(trained[0])
    shuffled = p.track(VIDEO, 5000, 1.0, days=(30, 7)).forecast
    ordered = p.track(VIDEO, 5000, 1.0, days=(7, 30)).forecast
    assert shuffled["days"].tolist() == [7, 30]
    assert np.allclose(shuffled["median"], ordered["median"])


def test_monitor_sees_videos_trained_on_through_earlier_snapshots(trained, synthetic_df):
    from ytviews.monitor import monitor

    p = Predictor.load(trained[0])
    trained_at = pd.Timestamp(p.meta["created_at"])
    df = synthetic_df.copy()
    # every video's newest snapshot moves past training time; older ones stay before it
    newest = df.groupby("video_id")["fetched_at"].transform("max") == df["fetched_at"]
    shift = trained_at + pd.Timedelta(days=30) - df["fetched_at"].max()
    df.loc[newest, "fetched_at"] += shift
    since = df["published_at"].quantile(0.8)
    assert monitor(p, df, since=since).in_sample


def test_publish_times_skip_the_hour_lost_to_daylight_saving(trained):
    p = Predictor.load(trained[0])
    start = datetime(2027, 3, 12, 12, tzinfo=timezone.utc)  # NY springs forward on 14 March
    grid = p.publish_times(VIDEO, "America/New_York", start=start)
    instants = grid["published_at"].map(lambda t: t.astimezone(timezone.utc))
    assert instants.is_unique  # no hour evaluated twice under two labels
    sunday = grid[grid["published_at"].map(lambda t: t.date().isoformat()) == "2027-03-14"]
    assert 2 not in set(sunday["hour"]) and len(grid) == 7 * 24 - 1


def test_schedule_handles_a_day_with_23_hours(trained):
    from ytviews.planner import schedule

    p = Predictor.load(trained[0])
    start = datetime(2027, 3, 12, 12, tzinfo=timezone.utc)
    plan = schedule(p, [VIDEO, VideoInput(title="Weekly Update", subscribers=100_000)],
                    timezone_name="America/New_York", days=4, start=start)  # fmt: skip
    assert plan["published_at"].map(lambda t: t.date()).is_unique


def test_channel_average_excludes_the_videos_latest_views_not_the_snapshots():
    from ytviews.features import add_channel_history

    t = pd.Timestamp("2026-01-01", tz="UTC")
    df = pd.DataFrame({
        "video_id": ["a", "b", "b"], "channel_id": "c", "title": "x",
        "published_at": [t, t, t],
        "fetched_at": [t + pd.Timedelta(days=d) for d in (30, 1, 30)],
        "views": [1000, 100, 10_000], "channel_views": 11_000, "channel_video_count": 2,
    })  # fmt: skip
    out = add_channel_history(df)
    # b's day-1 row must not count b's own later 9,900 views as "the other videos"
    assert out["channel_avg_views"].tolist() == [10_000, 1000, 1000]


def test_unknown_category_embedding_stays_neutral():
    import torch

    from ytviews.model import ViewsNet

    net = ViewsNet(10, {"title": 6}, 4, hidden=16)
    assert torch.all(net.category.weight[0] == 0) and net.category.padding_idx == 0
    out = net({"numeric": torch.randn(3, 10), "title": torch.randn(3, 6),
               "category": torch.tensor([0, 1, 2])})  # fmt: skip
    out.sum().backward()
    assert torch.all(net.category.weight.grad[0] == 0)  # never trained away from zero


def test_blend_choice_fits_its_own_features_on_the_inner_rows(synthetic_df, monkeypatch):
    from ytviews import pipeline
    from ytviews.features import FeatureBuilder, add_channel_history
    from ytviews.train import TrainConfig

    df = add_channel_history(synthetic_df).reset_index(drop=True)
    outer = FeatureBuilder(text_encoder="tfidf", image_encoder="none").fit(df)
    fitted = []
    real_fit = FeatureBuilder.fit
    monkeypatch.setattr(
        FeatureBuilder, "fit", lambda self, d: fitted.append(len(d)) or real_fit(self, d)
    )
    cfg = TrainConfig(ensemble=1, epochs=5, patience=2)
    pipeline._choose_blend_weight(df, "time", outer, cfg, lambda _: None, 0.3)
    assert fitted and fitted[0] < len(df)  # fitted on the inner rows, not all training rows


def test_tuned_config_records_its_split_and_still_loads(tmp_path):
    import json

    from ytviews.train import TrainConfig
    from ytviews.tune import load_config, save_config

    path = tmp_path / "best.json"
    save_config(TrainConfig(hidden=32), path, split="time", test_size=0.3)
    assert json.loads(path.read_text())["tuned_with"] == {"split": "time", "test_size": 0.3}
    assert load_config(path).hidden == 32


def _likes_model(trained, tmp_path):
    import json
    import shutil

    out = tmp_path / "likes"
    shutil.copytree(trained[0], out)
    meta = json.loads((out / "meta.json").read_text())
    (out / "meta.json").write_text(json.dumps(meta | {"target": "likes"}))
    return out


def test_api_rejects_oversized_or_out_of_range_inputs(trained):
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ytviews.api import create_app

    client = TestClient(create_app(trained[0]))
    video = {"title": "x", "subscribers": 1000}
    assert client.post("/predict", json=video | {"tags": ["t" * 101]}).status_code == 422
    assert (
        client.post("/predict", json=video | {"published_at": "9999-12-30T00:00:00"}).status_code
        == 422
    )
    assert client.post("/predict", json=video | {"recent_video_age_days": [-1]}).status_code == 422
    r = client.post("/track", json={"video": video, "observed_views": 10, "observed_hours": 1e12})
    assert r.status_code == 422
    for tz in ("", "../../etc/passwd", "Foo/Bar"):
        assert (
            client.post("/publish-times", json={"video": video, "timezone": tz}).status_code == 422
        )
        assert client.post("/plan", json={"video": video, "timezone": tz}).status_code == 422


def test_api_plan_without_a_thumbnail_names_none(trained, tmp_path):
    import base64

    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ytviews.api import create_app
    from ytviews.synthetic import render_thumbnail

    thumb = tmp_path / "alt.jpg"
    render_thumbnail(0.5, 0.3, 0, thumb)
    body = {"video": {"title": "Weekly Update", "subscribers": 1000},
            "alt_thumbnails_base64": [base64.b64encode(thumb.read_bytes()).decode()]}  # fmt: skip
    out = TestClient(create_app(trained[0])).post("/plan", json=body).json()
    assert out["draft"]["thumbnail"] is None  # the caller uploaded no draft thumbnail
    assert out["best"]["thumbnail"] in (None, "alt 1")


def test_api_missing_model_error_does_not_reveal_the_path(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ytviews.api import create_app

    r = TestClient(create_app(tmp_path / "secret-dir")).post("/predict", json={"title": "x"})
    assert r.status_code == 503 and "secret-dir" not in r.text


def test_cli_errors_are_clean_messages_not_tracebacks(trained, tmp_path):
    from typer.testing import CliRunner

    from ytviews.cli import app

    runner = CliRunner()
    likes = _likes_model(trained, tmp_path)
    r = runner.invoke(app, ["track", "--title", "x", "--views", "100", "--hours", "24",
                            "-m", str(likes)])  # fmt: skip
    assert r.exit_code == 2 and "views model" in r.output, r.output
    for tz in ("Foo/Bar", ""):
        r = runner.invoke(app, ["when", "--title", "x", "--timezone", tz, "-m", str(trained[0])])
        assert r.exit_code == 2 and "unknown timezone" in r.output, r.output
    r = runner.invoke(app, ["update", "--db", str(tmp_path / "e.db"), "--model",
                            str(tmp_path / "none"), "--retrain", "never", "--no-fetch",
                            "--no-enrich", "--reports", str(tmp_path / "r")])  # fmt: skip
    assert r.exit_code == 0 and "no model at" in r.output, r.output


def test_stats_on_an_empty_database_is_not_synthetic():
    from ytviews.pipeline import dataset_summary

    cols = ["video_id", "channel_id", "category_id", "views", "published_at"]
    assert dataset_summary(pd.DataFrame(columns=cols))["synthetic"] is False


def test_api_docs_can_be_turned_off(trained):
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ytviews.api import create_app

    assert TestClient(create_app(trained[0])).get("/openapi.json").status_code == 200
    hidden = TestClient(create_app(trained[0], docs=False))
    assert [hidden.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404] * 3
    assert hidden.get("/health").status_code == 200
