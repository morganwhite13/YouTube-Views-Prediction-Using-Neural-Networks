"""Property-based tests: random but valid inputs must never crash or give nonsense."""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from ytviews.predictor import Predictor, VideoInput  # noqa: E402

# More examples locally with YTVIEWS_PROPERTY_EXAMPLES=500 (CI keeps it quick).
EXAMPLES = int(os.environ.get("YTVIEWS_PROPERTY_EXAMPLES", "40"))
SETTINGS = settings(
    max_examples=EXAMPLES, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)  # fmt: skip
text = st.text(st.characters(codec="utf-8", exclude_categories=("Cs",)), max_size=300)
counts = st.one_of(st.none(), st.floats(0, 1e10, allow_nan=False))
moments = st.datetimes(
    min_value=datetime(2005, 4, 24), max_value=datetime(2099, 12, 31),
    timezones=st.one_of(st.none(), st.just(timezone.utc),
                        st.sampled_from(["Asia/Tokyo", "America/St_Johns", "Asia/Kathmandu",
                                         "Pacific/Chatham", "Europe/London"]).map(ZoneInfo)),
)  # fmt: skip
videos = st.builds(
    VideoInput,
    title=text,
    subscribers=counts,
    description=text,
    tags=st.lists(st.text(max_size=30), max_size=20),
    category_id=st.sampled_from(["1", "10", "20", "22", "24", "27", "28", "999", ""]),
    duration_s=st.one_of(st.none(), st.floats(0, 12 * 3600, allow_nan=False)),
    published_at=st.one_of(st.none(), moments),
    age_days=st.floats(1 / 24, 3650, allow_nan=False),
    channel_video_count=counts,
    channel_total_views=counts,
    recent_video_views=st.lists(st.floats(0, 1e9, allow_nan=False), max_size=10),
    caption=st.booleans(),
    hd=st.booleans(),
    language=st.one_of(st.none(), st.sampled_from(["en", "es-419", "ja", "xx", ""])),
    is_short=st.one_of(st.none(), st.booleans()),
)


@pytest.fixture(scope="module")
def predictor(trained):
    return Predictor.load(trained[0])


@SETTINGS
@given(video=videos)
def test_any_valid_video_gets_an_ordered_finite_range(predictor, video):
    (p,) = predictor.predict(video)
    assert all(math.isfinite(v) and v >= 0 for v in (p.low, p.median, p.high))
    assert p.low <= p.median * (1 + 1e-6) and p.median <= p.high * (1 + 1e-6)


@SETTINGS
@given(video=videos, ages=st.lists(st.floats(1 / 24, 3650), min_size=2, max_size=6))
def test_predicted_views_never_fall_as_a_video_ages(predictor, video, ages):
    ages = sorted(ages)
    medians = [p.median for p in predictor.predict([
        VideoInput(**{**video.__dict__, "age_days": a}) for a in ages
    ])]  # fmt: skip
    assert all(b >= a * (1 - 1e-4) - 1e-6 for a, b in zip(medians, medians[1:], strict=False))


@SETTINGS
@given(video=videos)
def test_video_json_round_trips(video):
    assert VideoInput.from_json(video.to_json()) == video


@settings(max_examples=max(EXAMPLES // 3, 5), deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])  # fmt: skip
@given(
    name=st.sampled_from(sorted(available_timezones())),
    start=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2040, 1, 1)),
)
def test_publish_times_work_in_every_timezone(predictor, name, start):
    grid = predictor.publish_times(VideoInput(title="x"), name, start.replace(tzinfo=timezone.utc))
    assert 7 * 24 - 2 <= len(grid) <= 7 * 24  # a skipped DST hour (or two) is left out
    assert not grid.duplicated(["weekday_index", "hour"]).any()
    assert grid["median"].gt(0).all() and grid["vs_best"].max() == 1.0
    times = grid["published_at"].sort_values()
    assert times.iloc[-1] - times.iloc[0] <= timedelta(days=7)


scalars = (
    st.none()
    | st.booleans()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.integers()
    | st.text(max_size=50)
)
json_values = st.recursive(
    scalars,
    lambda inner: (
        st.lists(inner, max_size=5) | st.dictionaries(st.text(max_size=10), inner, max_size=5)
    ),
    max_leaves=20,
)
payloads = st.dictionaries(
    st.sampled_from(["title", "subscribers", "tags", "published_at", "duration_s", "age_days",
                     "recent_video_views", "category_id", "language", "is_short", "extra"]),
    json_values, max_size=8,
)  # fmt: skip


@SETTINGS
@given(body=payloads)
def test_the_api_answers_any_json_without_a_server_error(trained, body):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ytviews.api import create_app

    client = _client(trained[0], TestClient, create_app)
    r = client.post("/predict", json=body)
    assert r.status_code in (200, 422), (body, r.text)


_clients: dict = {}


def _client(path, test_client, create_app):
    if path not in _clients:
        _clients[path] = test_client(create_app(path))
    return _clients[path]
