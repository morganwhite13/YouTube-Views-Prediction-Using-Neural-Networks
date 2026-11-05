import pandas as pd
import pytest
from typer.testing import CliRunner

from ytviews.cli import app
from ytviews.competitors import compare_channels, to_markdown
from ytviews.predictor import Predictor


@pytest.fixture(scope="module")
def latest(synthetic_df):
    return synthetic_df.sort_values("fetched_at").drop_duplicates("video_id", keep="last")


def _channels(df: pd.DataFrame, n: int = 3) -> list[str]:
    return df["channel_id"].value_counts().index[:n].tolist()


def test_compare_channels_table_and_hints(trained, latest):
    p = Predictor.load(trained[0])
    ids = _channels(latest)
    result = compare_channels(p, latest, ids)
    assert list(result.table.columns) == result.names and len(result.names) == 3
    assert "vs model (median)" in result.table.index
    first = latest[latest["channel_id"] == ids[0]]
    assert (
        result.raw.at["subscribers", result.names[0]]
        == first.sort_values("published_at")["subscribers"].iloc[-1]
    )
    assert result.hints  # either "does as well" or concrete differences
    text = to_markdown(result)
    assert text.startswith("| metric |") and "What stands out" in text


def test_hints_name_what_leading_channels_do_differently(trained, latest):
    from ytviews.competitors import style_hints

    p = Predictor.load(trained[0])
    result = compare_channels(p, latest, _channels(latest, 2))
    raw, (mine, other) = result.raw.copy(), result.names
    raw.loc["vs model (median)", [mine, other]] = [1.0, 2.0]  # the other clearly beats the model
    raw.loc["title length (chars)", [mine, other]] = [20.0, 60.0]
    raw.loc["Shorts share", [mine, other]] = [0.5, 0.5]
    hints = style_hints(raw, mine, [other])
    assert any("more title length (chars): 60 vs your 20" in h for h in hints), hints
    assert not any("Shorts share" in h for h in hints)
    raw.loc["vs model (median)", other] = 1.05  # no clear leader: nothing to copy
    assert style_hints(raw, mine, [other]) == [
        f"{mine} does as well as or better than the others against the model."
    ]


def test_compare_channels_errors_and_cli(trained, synthetic_db, latest, tmp_path):
    p = Predictor.load(trained[0])
    with pytest.raises(ValueError, match="at least one other"):
        compare_channels(p, latest, _channels(latest, 1))
    with pytest.raises(ValueError, match="collect it first"):
        compare_channels(p, latest, [*_channels(latest, 1), "UC-missing"])
    out = tmp_path / "c.md"
    args = ["compare-channels", *_channels(latest, 2), "--db", str(synthetic_db),
            "-m", str(trained[0]), "--out", str(out)]  # fmt: skip
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert out.read_text().startswith("| metric |")
