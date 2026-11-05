import dataclasses

from typer.testing import CliRunner

from ytviews import cli
from ytviews.benchmark import collect_dataset, render_results, run_benchmark
from ytviews.store import VideoStore
from ytviews.train import TrainConfig
from ytviews.youtube import YouTubeClient

METRICS = {"mae_log": 0.9, "r2_log": 0.6, "median_factor_error": 2.1, "within_2x": 0.45,
           "interval_coverage": 0.81}  # fmt: skip


def test_collect_dataset_uses_seed_channels_and_trending(fake_youtube, tmp_path):
    with VideoStore(tmp_path / "r.db") as store:
        n = collect_dataset(YouTubeClient(service=fake_youtube), store, regions=("US",),
                            log=lambda _: None)  # fmt: skip
        assert n > 0 and store.counts()["videos"] > 0
        # out of quota part-way: keeps what it has instead of failing
        more = collect_dataset(YouTubeClient(service=fake_youtube, quota_budget=2), store,
                               log=lambda _: None)  # fmt: skip
        assert more >= 0


def test_results_page_has_both_splits_and_a_readme_table():
    published = ["2019-01-01 00:00:00+00:00", "2030-01-01 00:00:00+00:00"]
    summary = {"rows": 3000, "videos": 2000, "channels": 150, "categories": 12,
               "median_views": 15000.0, "published_range": published}  # fmt: skip
    tables = {"ytviews (ours)": METRICS, "global median": METRICS | {"mae_log": 1.7}}
    metas = {s: {"metrics": tables} for s in ("time", "channel")}
    text = render_results(summary, metas)
    assert "2,000 videos" in text and "150 channels" in text
    assert "## Time split" in text and "## Channel split" in text
    assert "## For the README" in text and "| time | 0.900 | 45% | 81% |" in text


def test_run_benchmark_trains_and_writes_results(synthetic_df, tmp_path):
    cfg = TrainConfig(ensemble=1, epochs=20, patience=5)
    path = run_benchmark(synthetic_df, tmp_path / "bench", text_encoder="tfidf",
                         image_encoder="none", config=cfg, splits=("time",),
                         log=lambda _: None)  # fmt: skip
    assert path.name == "REAL_RESULTS.md" and (tmp_path / "bench" / "time" / "model.pt").exists()
    assert "ytviews (ours)" in path.read_text()


def test_benchmark_cli_guards(synthetic_db, fake_youtube, tmp_path, monkeypatch):
    runner = CliRunner()
    keyless = dataclasses.replace(cli.settings, api_key=None)
    monkeypatch.setattr(cli, "settings", keyless)
    r = runner.invoke(cli.app, ["benchmark", "--db", str(tmp_path / "empty.db")])
    assert r.exit_code == 2 and "YOUTUBE_API_KEY" in r.output
    r = runner.invoke(cli.app, ["benchmark", "--db", str(synthetic_db), "--no-collect"])
    assert r.exit_code == 2 and "synthetic" in r.output
    # with a key: one collection pass, then a note to come back tomorrow
    monkeypatch.setattr(cli, "settings", dataclasses.replace(keyless, api_key="fake"))
    monkeypatch.setattr(cli, "_client", lambda quota_budget: YouTubeClient(service=fake_youtube))
    r = runner.invoke(cli.app, ["benchmark", "--db", str(tmp_path / "real.db"),
                                "--min-videos", "100000", "--region", "US"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "run this again tomorrow" in r.output
