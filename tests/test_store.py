import json

from ytviews.store import VideoStore


def _rec(vid, views, fetched_at):
    return {
        "video_id": vid,
        "channel_id": "UC1",
        "title": "t",
        "description": "",
        "tags": "",
        "category_id": "20",
        "published_at": "2024-01-01T00:00:00+00:00",
        "views": views,
        "fetched_at": fetched_at,
    }


def test_snapshots_accumulate_and_latest_is_used(tmp_path):
    with VideoStore(tmp_path / "db.sqlite") as store:
        store.upsert_videos([_rec("a", 100, "2024-01-02T00:00:00+00:00")])
        store.upsert_videos([_rec("a", 500, "2024-02-01T00:00:00+00:00")])
        assert store.counts() == {"videos": 1, "snapshots": 2, "channels": 0}
        latest = store.load_dataset()
        everything = store.load_dataset(all_snapshots=True)
    assert latest["views"].tolist() == [500]
    assert sorted(everything["views"].tolist()) == [100, 500]


def test_hidden_views_are_skipped(tmp_path):
    with VideoStore(tmp_path / "db.sqlite") as store:
        assert store.upsert_videos([_rec("a", None, "2024-01-02T00:00:00+00:00")]) == 0


def test_import_legacy_json(tmp_path):
    legacy = [
        {
            "videoId": "x1",
            "title": "I Survived 100 Days",
            "description": "d",
            "channelTitle": "C",
            "channelId": "UCx",
            "subscriberCount": 1000,
            "categoryId": "20",
            "publishedAt": "2023-06-01T00:00:00Z",
            "views": 12345,
        },
    ]
    path = tmp_path / "randomChannelsVideos8.json"
    path.write_text(json.dumps(legacy))
    with VideoStore(tmp_path / "db.sqlite") as store:
        assert store.import_legacy_json(path, fetched_at="2023-12-01T00:00:00+00:00") == 1
        df = store.load_dataset()
    row = df.iloc[0]
    assert row["views"] == 12345
    assert row["subscribers"] == 1000
    assert str(row["fetched_at"].date()) == "2023-12-01"
