from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_renders_prediction(trained, monkeypatch):
    model, _ = trained
    monkeypatch.setenv("YTVIEWS_MODEL", str(model))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Expected views" in m.label for m in at.metric)


def test_app_plan_tab(trained, monkeypatch):
    model, _ = trained
    monkeypatch.setenv("YTVIEWS_MODEL", str(model))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    next(b for b in at.button if b.label == "Make a plan").click().run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Your draft", "Best found", "Uplift"} <= labels
