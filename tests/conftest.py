from __future__ import annotations

import pytest

from ytviews.images import attach_thumbnails, thumbnail_dir_for
from ytviews.pipeline import run_experiment
from ytviews.store import VideoStore
from ytviews.synthetic import write_synthetic_db
from ytviews.train import TrainConfig


@pytest.fixture(autouse=True, scope="session")
def _isolated_cache(tmp_path_factory):
    """Keep the embedding cache out of the user's ~/.cache during tests."""
    mp = pytest.MonkeyPatch()
    mp.setenv("YTVIEWS_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    yield
    mp.undo()


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synthetic.db"
    write_synthetic_db(path, n_channels=40, seed=1)
    return path


@pytest.fixture(scope="session")
def synthetic_df(synthetic_db):
    with VideoStore(synthetic_db) as store:
        return store.load_dataset(all_snapshots=True)


@pytest.fixture(scope="session")
def trained(synthetic_db, synthetic_df, tmp_path_factory):
    """A small model trained on synthetic data: (model_dir, meta)."""
    out = tmp_path_factory.mktemp("model")
    cfg = TrainConfig(ensemble=2, epochs=60, patience=10)
    df = attach_thumbnails(synthetic_df, thumbnail_dir_for(synthetic_db))
    meta = run_experiment(
        df, out, text_encoder="tfidf", image_encoder="stats", config=cfg, log=lambda _: None
    )
    return out, meta


class _Request:
    def __init__(self, response):
        self._response = response

    def execute(self):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeResource:
    def __init__(self, handler):
        self._handler = handler
        self.calls: list[dict] = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return _Request(self._handler(**kwargs))


class FakeYouTube:
    """Stands in for googleapiclient's service object."""

    def __init__(self, n_uploads: int = 120):
        self.uploads = [f"v{i:03d}" for i in range(n_uploads)]
        self._videos = FakeResource(self._video_list)
        self._channels = FakeResource(self._channel_list)
        self._playlist = FakeResource(self._playlist_items)
        self._search = FakeResource(lambda **kw: {"items": [{"snippet": {"channelId": "UC1"}}] * 3})

    def videos(self):
        return self._videos

    def channels(self):
        return self._channels

    def playlistItems(self):  # noqa: N802 - mirrors the Google API name
        return self._playlist

    def search(self):
        return self._search

    @staticmethod
    def video_item(vid: str) -> dict:
        n = int(vid[1:]) if vid[1:].isdigit() else 0
        return {
            "id": vid,
            "snippet": {
                "channelId": "UC1",
                "channelTitle": "Chan",
                "title": f"Video {vid}",
                "description": "desc",
                "tags": ["a", "b"],
                "categoryId": "20",
                "publishedAt": "2024-01-01T12:00:00Z",
                "liveBroadcastContent": "none",
                "defaultAudioLanguage": "fr" if n == 7 else "en-US",
            },
            "statistics": {"viewCount": str(1000 + n), "likeCount": "5"},
            "contentDetails": {"duration": "PT4M13S", "definition": "hd", "caption": "false"},
        }

    def _video_list(self, id=None, chart=None, **kw):
        if chart:
            return {"items": [self.video_item("v001"), self.video_item("v002")]}
        return {"items": [self.video_item(v) for v in id.split(",")]}

    def _channel_list(self, id, **kw):
        return {
            "items": [
                {
                    "id": cid,
                    "snippet": {"title": f"Channel {cid}"},
                    "statistics": {
                        "subscriberCount": "1000",
                        "viewCount": "50000",
                        "videoCount": "120",
                    },
                    "contentDetails": {"relatedPlaylists": {"uploads": f"UU{cid}"}},
                }
                for cid in id.split(",")
            ]
        }

    def _playlist_items(self, playlistId, maxResults, pageToken=None, **kw):
        start = int(pageToken or 0)
        chunk = self.uploads[start : start + maxResults]
        nxt = start + maxResults
        resp = {"items": [{"contentDetails": {"videoId": v}} for v in chunk]}
        if nxt < len(self.uploads):
            resp["nextPageToken"] = str(nxt)
        return resp


@pytest.fixture
def fake_youtube():
    return FakeYouTube()
