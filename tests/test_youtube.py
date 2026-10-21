import pytest

from ytviews.collect import collect_channels, keep_video
from ytviews.store import VideoStore
from ytviews.youtube import QuotaExceeded, YouTubeClient, parse_duration, parse_video


@pytest.mark.parametrize(
    "value, seconds",
    [("PT4M13S", 253), ("PT1H", 3600), ("P1DT2H", 93600), ("PT45S", 45), ("P0D", 0)],
)
def test_parse_duration(value, seconds):
    assert parse_duration(value) == seconds


def test_parse_duration_invalid():
    assert parse_duration(None) is None
    assert parse_duration("garbage") is None


def test_parse_video_flattens_record(fake_youtube):
    rec = parse_video(fake_youtube.video_item("v005"), "2024-02-01T00:00:00+00:00")
    assert rec["video_id"] == "v005"
    assert rec["views"] == 1005
    assert rec["tags"] == "a|b"
    assert rec["duration_s"] == 253
    assert rec["comments"] is None


def test_videos_are_batched_50_per_call(fake_youtube):
    client = YouTubeClient(service=fake_youtube)
    out = client.videos(fake_youtube.uploads)  # 120 ids
    assert len(out) == 120
    assert len(fake_youtube._videos.calls) == 3
    assert client.quota_used == 3


def test_upload_ids_paginates(fake_youtube):
    client = YouTubeClient(service=fake_youtube)
    ids = client.upload_ids("UU1", max_videos=75)
    assert ids == fake_youtube.uploads[:75]
    assert len(fake_youtube._playlist.calls) == 2


def test_quota_budget_is_enforced(fake_youtube):
    client = YouTubeClient(service=fake_youtube, quota_budget=150)
    client.search_channel_ids("x")
    with pytest.raises(QuotaExceeded):
        client.search_channel_ids("y")


def test_keep_video_filters():
    assert keep_video({"views": 10, "default_language": "en-GB"})
    assert not keep_video({"views": 10, "default_language": "fr"})
    assert keep_video({"views": 10, "default_language": "fr"}, language=None)
    assert not keep_video({"views": None})
    assert not keep_video({"views": 10, "live_broadcast": "upcoming"})


def test_collect_channels_stores_videos(fake_youtube, tmp_path):
    client = YouTubeClient(service=fake_youtube)
    with VideoStore(tmp_path / "db.sqlite") as store:
        n = collect_channels(client, store, ["UC1"], max_videos_per_channel=20)
        assert n == 19  # v007 is French and filtered out
        df = store.load_dataset()
    assert len(df) == 19
    assert df["subscribers"].eq(1000).all()
    # 1 channels call + 1 playlist page + 1 videos batch
    assert client.quota_used == 3
