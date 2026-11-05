import base64

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ytviews.api import create_app  # noqa: E402
from ytviews.synthetic import render_thumbnail  # noqa: E402

VIDEO = {"title": "I Survived 100 Days in Minecraft", "subscribers": 100000, "category_id": "20"}


@pytest.fixture(scope="module")
def client(trained):
    model, _ = trained
    return TestClient(create_app(model))


def test_health_and_model(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model").json()
    assert "ytviews (ours)" in info["metrics"]


def test_predict(client, tmp_path):
    thumb = tmp_path / "t.jpg"
    render_thumbnail(1.0, 0.3, 0, thumb)
    body = VIDEO | {"thumbnail_base64": base64.b64encode(thumb.read_bytes()).decode()}
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    pred = r.json()["prediction"]
    assert pred["low"] <= pred["median"] <= pred["high"]
    assert len(r.json()["trajectory"]) == 8


def test_bad_thumbnail_is_422(client):
    r = client.post("/predict", json=VIDEO | {"thumbnail_base64": "not base64!!"})
    assert r.status_code == 422


def test_compare_explain_suggest(client):
    r = client.post("/compare", json={"video": VIDEO, "titles": ["A", "B title"]})
    assert r.status_code == 200 and len(r.json()["ranking"]) == 2
    r = client.post("/explain", json=VIDEO)
    assert r.status_code == 200 and r.json()["effects"]
    r = client.post("/suggest", json={"video": VIDEO})
    assert r.status_code == 200 and len(r.json()["ranking"]) > 5


def test_missing_model_is_503(tmp_path):
    r = TestClient(create_app(tmp_path)).post("/predict", json=VIDEO)
    assert r.status_code == 503


def test_track_and_publish_times(client):
    r = client.post("/track", json={"video": VIDEO, "observed_views": 5000, "observed_hours": 12})
    assert r.status_code == 200, r.text
    assert r.json()["forecast"]
    r = client.post("/publish-times", json={"video": VIDEO, "timezone": "Asia/Tokyo"})
    assert r.status_code == 200 and len(r.json()["slots"]) == 168
    r = client.post("/publish-times", json={"video": VIDEO, "timezone": "Mars/Base"})
    assert r.status_code == 422


def test_api_key_is_required_when_configured(trained):
    model, _ = trained
    c = TestClient(create_app(model, api_key="s3cret"))
    assert c.get("/health").status_code == 200  # liveness stays open
    assert c.post("/predict", json=VIDEO).status_code == 401
    assert c.post("/predict", json=VIDEO, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/model", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.post("/predict", json=VIDEO, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_input_limits_and_image_validation(client):
    assert client.post("/predict", json=VIDEO | {"title": "x" * 301}).status_code == 422
    assert client.post("/predict", json=VIDEO | {"age_days": 10_000}).status_code == 422
    not_image = base64.b64encode(b"definitely not a jpeg").decode()
    r = client.post("/predict", json=VIDEO | {"thumbnail_base64": not_image})
    assert r.status_code == 422 and "image" in r.text
    r = client.post("/compare", json={"video": VIDEO, "titles": ["ok"] * 51})
    assert r.status_code == 422


def test_paid_llm_suggestions_are_opt_in(trained):
    model, _ = trained
    off = TestClient(create_app(model, allow_llm=False))
    r = off.post("/suggest", json={"video": VIDEO, "use_llm": True})
    assert r.status_code == 403
    assert off.post("/suggest", json={"video": VIDEO}).status_code == 200


def test_cors_origins(trained):
    model, _ = trained
    c = TestClient(create_app(model, cors_origins=["https://example.com"]))
    r = c.options(
        "/predict",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "https://example.com"
