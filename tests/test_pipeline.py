import pandas as pd


def test_report_shows_coverage_by_segment(trained):
    from ytviews.pipeline import render_report

    _, meta = trained
    segments = meta["coverage_by_segment"]
    assert set(segments) <= {"channel size", "video age"} and "video age" in segments
    total = sum(g["n"] for g in segments["video age"].values())
    assert total == len(pd.read_csv(trained[0] / "test_predictions.csv"))
    assert all(0 <= g["coverage"] <= 1 for d in segments.values() for g in d.values())
    assert "## Is the range right for every kind of video?" in render_report(meta)


def test_html_report_has_the_segment_table(trained, tmp_path):
    from ytviews.report_html import write_html_report

    html = write_html_report(trained[0], tmp_path / "r.html").read_text()
    assert "Is the range right for every kind of video?" in html
