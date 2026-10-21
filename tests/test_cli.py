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
