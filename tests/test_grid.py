import re

import numpy as np
from typer.testing import CliRunner

from ytviews.cli import app
from ytviews.predictor import Predictor, VideoInput
from ytviews.synthetic import render_thumbnail


def _thumbs(tmp_path):
    paths = []
    for i, (b, c) in enumerate(((0.9, 0.8), (0.2, 0.1))):
        path = tmp_path / f"t{i}.jpg"
        render_thumbnail(b, c, i, path)
        paths.append(str(path))
    return paths


def test_grid_scores_every_pair_consistently_with_single_predictions(trained, tmp_path):
    p = Predictor.load(trained[0])
    base = VideoInput(title="x", subscribers=50_000)
    titles = ["I Tried Every Fast Food Burger", "weekly update #12", "Building a House in 24h"]
    thumbs = _thumbs(tmp_path)
    grid = p.compare_grid(base, titles, thumbs)
    assert len(grid) == 6 and grid["median"].is_monotonic_decreasing
    assert grid["vs_best"].iloc[0] == 1.0
    row = grid[(grid["title"] == titles[1]) & (grid["thumbnail"] == thumbs[1])].iloc[0]
    single = p.predict(VideoInput(title=titles[1], subscribers=50_000, thumbnail_path=thumbs[1]))
    assert np.isclose(row["median"], single[0].median, rtol=1e-5)
    # interactions are relative to an additive model: their log mean is zero
    assert abs(np.log(grid["interaction"]).mean()) < 1e-6


def test_grid_cli_prints_the_table_and_best_pairs(trained, tmp_path):
    thumbs = _thumbs(tmp_path)
    args = ["grid", "-t", "First title", "-t", "Second title", "--subscribers", "1000",
            "-m", str(trained[0])]  # fmt: skip
    for t in thumbs:
        args += ["--thumbnail", t]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert "T1" in r.output and "T2" in r.output and "Best pairs:" in r.output
    r = CliRunner().invoke(app, [*args[:-1], str(tmp_path / "missing.jpg")])
    assert r.exit_code == 2 and "not found" in r.output


def test_grid_cli_ignores_repeated_options(trained, tmp_path):
    thumbs = _thumbs(tmp_path)
    args = ["grid", "-t", "Same", "-t", "Same", "-t", "Other", "--thumbnail", thumbs[0],
            "--thumbnail", thumbs[0], "--thumbnail", thumbs[1], "-m", str(trained[0])]  # fmt: skip
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert len(re.findall(r"^T\d = ", r.output, flags=re.MULTILINE)) == 2  # one per thumbnail
