import pandas as pd
import pytest
from typer.testing import CliRunner

from ytviews.cli import app
from ytviews.digest import subscriber_change, views_gained, weekly_digest
from ytviews.predictor import Predictor


def _busiest_channel(df: pd.DataFrame) -> str:
    return df.groupby("channel_id")["video_id"].nunique().idxmax()


def test_digest_has_every_section(trained, synthetic_df):
    p = Predictor.load(trained[0])
    channel = _busiest_channel(synthetic_df)
    text = weekly_digest(p, synthetic_df, channel, days=30)
    assert text.startswith("# ")
    for heading in ("## New uploads", "## Try next"):
        assert heading in text
    assert "Publish around" in text
    with pytest.raises(ValueError, match="no stored videos"):
        weekly_digest(p, synthetic_df, "nope")


def test_views_gained_and_subscriber_change_from_snapshots():
    t = pd.to_datetime
    df = pd.DataFrame({
        "channel_id": ["c"] * 5, "video_id": ["old", "old", "new", "stale", "stale"],
        "published_at": t(["2030-01-01", "2030-01-01", "2030-03-05", "2030-01-01", "2030-01-01"],
                          utc=True),
        "fetched_at": t(["2030-02-28", "2030-03-07", "2030-03-07", "2030-01-02", "2030-02-01"],
                        utc=True),
        "views": [1_000.0, 1_500.0, 300.0, 10.0, 20.0],
        "channel_fetched_at": t(["2030-02-28", "2030-03-07", "2030-03-07", "2030-01-02",
                                 "2030-02-01"], utc=True),
        "subscribers_at_fetch": [100.0, 130.0, 130.0, 80.0, 90.0],
    })  # fmt: skip
    start = pd.Timestamp("2030-03-01", tz="UTC")
    # +500 on the old video, all 300 of the new one; the stale one has nothing after start
    assert views_gained(df, "c", start) == (800.0, 2)
    assert subscriber_change(df, "c", start) == (130.0, 30.0)


def test_digest_cli_writes_markdown(trained, synthetic_db, synthetic_df, tmp_path):
    channel = _busiest_channel(synthetic_df)
    out = tmp_path / "d" / "digest.md"
    r = CliRunner().invoke(app, ["digest", "--channel-id", channel, "--db", str(synthetic_db),
                                 "-m", str(trained[0]), "--out", str(out)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert out.read_text().startswith("# ")
    r = CliRunner().invoke(app, ["digest", "--channel-id", channel, "--db", str(synthetic_db),
                                 "-m", str(trained[0]), "--timezone", "Mars/Base"])  # fmt: skip
    assert r.exit_code == 2
