import json

import pytest

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


def test_old_plans_table_gains_the_trajectory_column(tmp_path):
    import sqlite3

    from ytviews.store import VideoStore

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE plans (plan_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT "
                "NOT NULL, channel_id TEXT, draft_title TEXT NOT NULL, video_json TEXT NOT NULL, "
                "age_days REAL NOT NULL, predicted_low REAL, predicted_median REAL, "
                "predicted_high REAL, model TEXT, video_id TEXT)")  # fmt: skip
    con.commit()
    con.close()
    with VideoStore(db) as store:
        assert "trajectory_json" in store.plans().columns
        store.add_plan({"created_at": "2026-01-01", "draft_title": "t", "video_json": "{}",
                        "age_days": 7})  # fmt: skip
        assert len(store.plans()) == 1


def test_legacy_import_keeps_richer_metadata(tmp_path, fake_youtube):
    import json

    from ytviews.collect import collect_channels
    from ytviews.store import VideoStore
    from ytviews.youtube import YouTubeClient

    with VideoStore(tmp_path / "a.db") as store:
        collect_channels(YouTubeClient(service=fake_youtube), store, ["UC1"],
                         max_videos_per_channel=5)  # fmt: skip
        query = "SELECT duration_s, tags, default_language, thumbnail_url FROM videos WHERE video_id='v001'"  # noqa: E501
        before = store.conn.execute(query).fetchone()
        legacy = tmp_path / "v1.json"
        v1 = {"videoId": "v001", "channelId": "UC1", "title": "Video v001", "views": 900,
              "publishedAt": "2024-01-01T12:00:00Z"}  # fmt: skip
        legacy.write_text(json.dumps([v1]))
        store.import_legacy_json(legacy, fetched_at="2023-06-01T00:00:00+00:00")
        assert store.conn.execute(query).fetchone() == before  # metadata untouched
        assert store.counts()["snapshots"] == 6  # but the v1 snapshot is added


def test_refresh_keeps_batches_fetched_before_quota_runs_out(tmp_path, monkeypatch):
    from conftest import FakeYouTube

    from ytviews.collect import collect_channels, refresh
    from ytviews.store import VideoStore
    from ytviews.youtube import QuotaExceeded, YouTubeClient

    with VideoStore(tmp_path / "b.db") as store:
        collect_channels(YouTubeClient(service=FakeYouTube(n_uploads=500), quota_budget=1000),
                         store, ["UC1"], max_videos_per_channel=500)  # fmt: skip
        before = store.counts()["snapshots"]
        # a later fetch time, so refreshed snapshots add rows instead of replacing them
        monkeypatch.setattr("ytviews.youtube.utcnow_iso", lambda: "2030-01-01T00:00:00+00:00")
        client = YouTubeClient(service=FakeYouTube(), quota_budget=7)
        with pytest.raises(QuotaExceeded):
            refresh(client, store)
        assert store.counts()["snapshots"] - before == 7 * 50  # 7 calls' worth kept


def test_latest_snapshot_is_chosen_by_time_not_text(tmp_path):
    from conftest import FakeYouTube

    from ytviews.store import VideoStore
    from ytviews.youtube import parse_video

    with VideoStore(tmp_path / "c.db") as store:
        item = FakeYouTube.video_item("v001")
        later = parse_video(item, "2024-05-01T09:00:00+00:00") | {"views": 5000}
        earlier = parse_video(item, "2024-05-01T12:00:00+05:00") | {"views": 100}  # 07:00 UTC
        store.upsert_videos([later, earlier])
        assert store.load_dataset()["views"].tolist() == [5000]


def test_old_video_snapshots_use_the_channel_size_of_their_time(tmp_path):
    from ytviews.store import VideoStore

    with VideoStore(tmp_path / "s.db") as store:
        for t, subs, total in (
            ("2030-01-01T00:00:00Z", 100, 1_000),
            ("2030-06-01T00:00:00Z", 10_000, 90_000),
        ):
            store.upsert_channels(
                [
                    {
                        "channel_id": "c",
                        "subscribers": subs,
                        "channel_views": total,
                        "channel_video_count": 10,
                        "fetched_at": t,
                    }
                ]
            )
        for t, views in (("2030-01-02T00:00:00Z", 50), ("2030-05-30T00:00:00Z", 5_000)):
            store.upsert_videos(
                [
                    {
                        "video_id": "v",
                        "channel_id": "c",
                        "title": "x",
                        "views": views,
                        "published_at": "2029-12-31T00:00:00Z",
                        "fetched_at": t,
                    }
                ]
            )
        df = store.load_dataset(all_snapshots=True).sort_values("fetched_at")
        assert df["subscribers"].tolist() == [100, 10_000]
        assert df["channel_views"].tolist() == [1_000, 90_000]
        latest = store.load_dataset()
        assert latest["subscribers"].tolist() == [10_000]


def test_channel_average_subtracts_the_videos_views_at_that_time():
    import pandas as pd

    from ytviews.features import add_channel_history

    df = pd.DataFrame(
        {
            "video_id": ["v", "v"],
            "channel_id": ["c", "c"],
            "views": [50.0, 5_000.0],
            "published_at": pd.to_datetime(["2029-12-31"] * 2, utc=True),
            "fetched_at": pd.to_datetime(["2030-01-02", "2030-05-30"], utc=True),
            "channel_fetched_at": pd.to_datetime(["2030-01-01", "2030-06-01"], utc=True),
            "channel_views": [1_050.0, 95_000.0],
            "channel_video_count": [11.0, 11.0],
        }
    )
    out = add_channel_history(df).sort_values("fetched_at")
    assert out["channel_avg_views"].tolist() == [100.0, 9_000.0]


def test_existing_channels_are_backfilled_as_snapshots(tmp_path):
    import sqlite3

    from ytviews.store import VideoStore

    path = tmp_path / "old.db"
    with VideoStore(path) as store:
        store.upsert_channels(
            [{"channel_id": "c", "subscribers": 7, "fetched_at": "2030-01-01T00:00:00Z"}]
        )
    con = sqlite3.connect(path)
    con.execute("DELETE FROM channel_stats")
    con.commit()
    con.close()
    with VideoStore(path) as store:
        assert store.conn.execute("SELECT subscribers FROM channel_stats").fetchall() == [(7,)]
