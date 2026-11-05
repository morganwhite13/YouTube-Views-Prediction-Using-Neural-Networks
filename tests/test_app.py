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


def test_app_explains_the_range_and_saves_and_loads_drafts(trained, monkeypatch, tmp_path):
    model, _ = trained
    monkeypatch.setenv("YTVIEWS_MODEL", str(model))
    monkeypatch.setenv("YTVIEWS_DRAFTS", str(tmp_path / "drafts.json"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert any("in 10) end up between" in m.value for m in at.markdown)
    at.text_input(key="title").set_value("My saved title").run()
    next(b for b in at.button if b.label == "Save draft").click().run()
    assert not at.exception
    at.text_input(key="title").set_value("Something else").run()
    next(b for b in at.button if b.label == "Load").click().run()
    assert not at.exception
    assert at.text_input(key="title").value == "My saved title"


def test_app_plan_flow_shows_every_step(trained, monkeypatch, tmp_path):
    model, _ = trained
    monkeypatch.setenv("YTVIEWS_MODEL", str(model))
    monkeypatch.setenv("YTVIEWS_DRAFTS", str(tmp_path / "drafts.json"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    next(b for b in at.button if b.label == "Make a plan").click().run()
    steps = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert steps == ["#### 1. The result", "#### 2. What to change", "#### 3. What to expect",
                     "#### 4. Keep it"]  # fmt: skip
    next(b for b in at.button if b.label == "Save the best version as a draft").click().run()
    assert not at.exception
    from ytviews.drafts import load_drafts

    assert any(name.endswith("(plan)") for name in load_drafts(tmp_path / "drafts.json"))
