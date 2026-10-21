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
    assert "neural net (ours)" in info["metrics"]


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
