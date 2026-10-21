import json

from typer.testing import CliRunner

from ytviews.cli import app

runner = CliRunner()


def test_synth_and_stats(tmp_path):
    db = tmp_path / "s.db"
    result = runner.invoke(app, ["synth", "--db", str(db), "--channels", "5"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["stats", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert '"channels": 5' in result.output


def test_predict_json(trained):
    model, _ = trained
    args = ["predict", "--title", "I Survived 100 Days", "--subscribers", "50000",
            "--recent-views", "1000,2000,3000", "--model", str(model), "--json"]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    p = out["prediction"]
    assert p["low"] <= p["median"] <= p["high"]
    assert len(out["trajectory"]) == 8


def test_compare(trained):
    model, _ = trained
    result = runner.invoke(
        app, ["compare", "Weekly Update", "I Survived 100 Days", "--model", str(model)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("1.")


def test_collect_requires_a_source(tmp_path):
    result = runner.invoke(app, ["collect", "--db", str(tmp_path / "x.db")])
    assert result.exit_code != 0


def test_explain_and_suggest(trained):
    model, _ = trained
    common = ["--subscribers", "100000", "--category", "20", "--model", str(model)]
    result = runner.invoke(app, ["explain", "--title", "Weekly Update", *common])
    assert result.exit_code == 0, result.output
    assert "An average video" in result.output and "channel size" in result.output
    result = runner.invoke(app, ["suggest", "--title", "Minecraft Hardcore Ep. 12", *common])
    assert result.exit_code == 0, result.output
    assert "[rewrite]" in result.output


def test_predict_with_missing_thumbnail_errors(trained, tmp_path):
    model, _ = trained
    result = runner.invoke(
        app,
        ["predict", "--title", "x", "--thumbnail", str(tmp_path / "no.jpg"), "--model", str(model)],
    )
    assert result.exit_code != 0


def test_report_html(trained, tmp_path):
    model, _ = trained
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", "--model", str(model), "--html", str(out)])
    assert result.exit_code == 0, result.output
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert "<title>ytviews Model Report</title>" in page
    assert "synthetic dataset" in page  # trained on synthetic data -> flagged
    assert page.count("<circle") > 10  # scatter points rendered


def test_track_when_outliers_batch(trained, synthetic_db, tmp_path):
    model, _ = trained
    m = ["--model", str(model)]
    r = runner.invoke(app, ["track", "--title", "x", "--views", "5000", "--hours", "24", *m])
    assert r.exit_code == 0 and "Forecast:" in r.output, r.output
    r = runner.invoke(app, ["when", "--title", "x", "--timezone", "Europe/London", *m])
    assert r.exit_code == 0 and "Best times" in r.output, r.output
    r = runner.invoke(app, ["outliers", "--db", str(synthetic_db), "--top", "3", *m])
    assert r.exit_code == 0 and "Beat expectations" in r.output, r.output
    src, out = tmp_path / "in.csv", tmp_path / "out.csv"
    src.write_text("title,subscribers,category_id\nA,1000,20\nB,500000,24\n")
    r = runner.invoke(app, ["batch", str(src), "--out", str(out), *m])
    assert r.exit_code == 0, r.output
    assert "pred_median" in out.read_text()
