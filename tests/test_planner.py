import base64
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ytviews.channel_report import channel_report_html, live_uploads
from ytviews.cli import app
from ytviews.images import attach_thumbnails, thumbnail_dir_for
from ytviews.planner import make_plan, plan_html
from ytviews.predictor import Predictor, VideoInput
from ytviews.store import VideoStore
from ytviews.synthetic import render_thumbnail

runner = CliRunner()
VIDEO = VideoInput(title="minecraft hardcore day 12", subscribers=50_000, category_id="20",
                   duration_s=900, recent_video_views=[12_000, 8_000, 20_000])  # fmt: skip


@pytest.fixture(scope="module")
def predictor(trained):
    return Predictor.load(trained[0])


@pytest.fixture
def thumbs(tmp_path):
    paths = []
    for i, (bright, sat) in enumerate([(0.2, 0.1), (1.0, 0.9), (0.6, 0.5)]):
        path = tmp_path / f"t{i}.jpg"
        render_thumbnail(bright, sat, i % 2, path)
        paths.append(str(path))
    return paths


def test_plan_combines_every_tool_and_never_recommends_worse(predictor, thumbs):
    video = VideoInput(**{**VIDEO.__dict__, "thumbnail_path": thumbs[0]})
    plan = make_plan(predictor, video, extra_titles=["My own idea"], thumbnails=thumbs[1:],
                     timezone_name="Asia/Tokyo")  # fmt: skip
    assert plan.best_prediction.median >= plan.draft_prediction.median
    assert plan.uplift >= 1
    assert "My own idea" in set(plan.titles["title"])
    assert len(plan.slots) == 7 * 24
    if predictor.features.uses_thumbnails:
        assert set(plan.thumbnails["thumbnail"]) == set(thumbs)
    if plan.best is not video:
        # the combined pick is predicted jointly, not as a product of separate gains
        alone = predictor.predict(plan.best)[0]
        assert plan.best_prediction.median == pytest.approx(alone.median, rel=1e-4)
        assert plan.best.published_at.tzinfo is not None
    page = plan_html(plan)
    assert "<title>Video Plan</title>" in page and "What to change" in page
    json.dumps(plan.to_dict(), default=str)


def test_plan_keeps_draft_when_nothing_beats_it(predictor, monkeypatch):
    real = Predictor.predict

    def every_change_loses(self, videos):
        preds = real(self, videos)
        if isinstance(videos, list) and len(videos) <= 8:  # the final search over changes
            return [p if v == VIDEO else type(p)(p.low, 0.0, p.high, p.coverage)
                    for v, p in zip(videos, preds, strict=True)]  # fmt: skip
        return preds

    monkeypatch.setattr(Predictor, "predict", every_change_loses)
    plan = make_plan(predictor, VIDEO)
    assert plan.best is VIDEO and plan.uplift == 1
    assert plan.changes()[0].startswith("Keep the draft")
    assert plan.uplift_by_lever().empty
    assert "already the best" in plan_html(plan)


def test_uplift_is_split_exactly_between_the_changes(predictor):
    plan = make_plan(predictor, VIDEO, timezone_name="Europe/Berlin")
    if plan.best is VIDEO:
        pytest.skip("nothing beat the draft on this model")
    levers = plan.uplift_by_lever()
    # Shapley shares multiply to the whole uplift (in log space they add up exactly)
    ex = plan.uplift_explanation
    assert ex.baseline_views == pytest.approx(plan.draft_prediction.median, rel=1e-4)
    assert ex.predicted_views == pytest.approx(plan.best_prediction.median, rel=1e-4)
    assert levers["factor"].prod() == pytest.approx(
        (ex.predicted_views + 1) / (ex.baseline_views + 1), rel=1e-3
    )
    # the adopted set of changes is the best subset, so dropping any one can only lose views
    assert set(plan.skip_cost) <= {"title", "thumbnail", "publish time"}
    assert all(c >= 1 for c in plan.skip_cost.values())


def test_explain_against_a_reference_video(predictor):
    other = VideoInput(**{**VIDEO.__dict__, "title": "A completely different title"})
    ex = predictor.explain(other, reference=VIDEO)
    assert ex.baseline_views == pytest.approx(predictor.predict(VIDEO)[0].median, rel=1e-4)
    changed = ex.effects.set_index("group")["log_effect"]
    assert (changed.drop(["title text", "title style"]).abs() < 1e-6).all()


def test_plan_cli(trained, tmp_path):
    model, _ = trained
    out = tmp_path / "plan.html"
    args = ["plan", "--title", "Weekly Update", "--subscribers", "20000", "--alt-title",
            "Weekly Update #12", "--timezone", "Europe/London", "--model", str(model),
            "--html", str(out)]  # fmt: skip
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    assert "What to change" in r.output and out.exists()
    r = runner.invoke(app, [*args[:-2], "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["best_prediction"]["median"] >= data["draft_prediction"]["median"]
    assert len(data["top_slots"]) == 5


def test_plan_api(trained, thumbs):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ytviews.api import create_app

    client = TestClient(create_app(trained[0]))
    b64 = [base64.b64encode(Path(p).read_bytes()).decode() for p in thumbs]
    video = {"title": "Weekly Update", "subscribers": 20000, "thumbnail_base64": b64[0]}
    body = {"video": video,
            "alt_titles": ["Weekly Update #12"], "alt_thumbnails_base64": b64[1:],
            "timezone": "America/New_York"}  # fmt: skip
    r = client.post("/plan", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["draft"]["thumbnail"] == "draft"
    assert out["best"]["thumbnail"] in {"draft", "alt 1", "alt 2"}
    assert "/tmp" not in r.text and "thumbnail-" not in r.text  # no server temp paths leak
    if out["thumbnails"]:
        assert {t["thumbnail"] for t in out["thumbnails"]} == {"draft", "alt 1", "alt 2"}
    bad = client.post("/plan", json=body | {"timezone": "Mars/Base"})
    assert bad.status_code == 422
    llm = client.post("/plan", json=body | {"use_llm": True})
    assert llm.status_code == 403


def test_live_uploads_in_channel_report(predictor, synthetic_db):
    with VideoStore(synthetic_db) as store:
        df = store.load_dataset(all_snapshots=False)
    df = attach_thumbnails(df, thumbnail_dir_for(synthetic_db))
    age = (df["fetched_at"] - df["published_at"]).dt.total_seconds() / 86400
    channel = df[age < 14]["channel_id"].value_counts().index[0]
    live = live_uploads(predictor, df, channel)
    assert len(live) >= 1 and (live["age_days"] < 14).all()
    assert (live["median"] >= live["views"]).all()  # cumulative views never go down
    assert (live["low"] <= live["median"]).all() and (live["median"] <= live["high"]).all()
    assert "Still growing" in channel_report_html(predictor, df, channel)
    old = df.assign(fetched_at=df["published_at"] + pd.Timedelta(days=400))
    assert live_uploads(predictor, old, channel).empty


def test_update_retrains_and_writes_reports(trained, synthetic_db, tmp_path):
    model = tmp_path / "model"
    shutil.copytree(trained[0], model)
    with VideoStore(synthetic_db) as store:
        channel = store.load_dataset(all_snapshots=False)["channel_id"].iloc[0]
    reports = tmp_path / "reports"
    args = ["update", "--db", str(synthetic_db), "--model", str(model), "--no-fetch",
            "--no-enrich", "--reports", str(reports), "--channel", channel]  # fmt: skip
    r = runner.invoke(app, [*args, "--retrain", "never"])
    assert r.exit_code == 0, r.output
    assert "no retraining needed" in r.output
    assert (reports / "model-report.html").exists()
    assert (reports / f"channel-{channel}.html").exists()

    before = json.loads((model / "meta.json").read_text())["created_at"]
    r = runner.invoke(app, [*args, "--retrain", "always"])
    assert r.exit_code == 0, r.output
    meta = json.loads((model / "meta.json").read_text())
    assert meta["created_at"] != before
    assert (
        meta["train_config"] == json.loads((trained[0] / "meta.json").read_text())["train_config"]
    )  # the previous model's settings are reused

    r = runner.invoke(app, [*args, "--retrain", "sometimes"])
    assert r.exit_code != 0


def test_pulse_fetches_and_forecasts_latest_uploads(predictor):
    from datetime import datetime, timedelta, timezone

    from conftest import FakeYouTube

    from ytviews.channel_report import pulse
    from ytviews.youtube import YouTubeClient

    class Recent(FakeYouTube):
        def video_item(self, vid):
            item = FakeYouTube.video_item(vid)
            age = timedelta(days=0.5 + 2 * int(vid[1:]))  # v000 is the newest
            item["snippet"]["publishedAt"] = (datetime.now(timezone.utc) - age).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            return item

    client = YouTubeClient(service=Recent(n_uploads=30))
    live = pulse(predictor, client, "UC1", n=4, horizon_days=30)
    assert list(live["title"]) == ["Video v000", "Video v001", "Video v002", "Video v003"]
    assert (live["median"] >= live["views"]).all()
    assert client.quota_used <= 5


def test_title_terms_find_words_that_beat_expectations():
    from ytviews.analysis import title_terms

    titles = [f"Honest review {i}" for i in range(5)] + [f"Weekly vlog {i}" for i in range(7)]
    factor = [3.0] * 5 + [0.8] * 7
    terms = title_terms(pd.DataFrame({"title": titles, "factor": factor}))
    by_term = terms.set_index("term")["lift"]
    assert by_term["honest"] > 1 > by_term["weekly"]
    assert "the" not in by_term.index
    assert title_terms(pd.DataFrame({"title": ["a"], "factor": [1.0]})).empty
