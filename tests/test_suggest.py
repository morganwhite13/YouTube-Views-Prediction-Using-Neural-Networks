import json
from types import SimpleNamespace

from ytviews.predictor import Predictor, VideoInput
from ytviews.suggest import _core, claude_variants, rule_based_variants, suggest_titles


def test_core_strips_episode_noise():
    assert _core("Minecraft Hardcore Ep. 12 (FULL)") == "Minecraft Hardcore"
    assert _core("Weekly Update |") == "Weekly Update"


def test_rule_based_variants_are_unique_and_reasonable():
    out = rule_based_variants("Minecraft Hardcore Ep. 12")
    assert len(out) == len(set(out)) >= 8
    assert "The Truth About Minecraft Hardcore" in out
    assert "Minecraft Hardcore" in out
    assert all(len(t) <= 100 for t in out)


class FakeClient:
    """Mimics anthropic.Anthropic().beta.messages.create for one canned response."""

    def __init__(self, titles, stop_reason="end_turn"):
        self.calls = []
        text = json.dumps({"titles": titles})
        self._response = SimpleNamespace(
            stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_claude_variants_request_shape():
    client = FakeClient(["A", " B ", ""])
    out = claude_variants("Draft", category_id="20", client=client)
    assert out == ["A", "B"]
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "Gaming" in call["messages"][0]["content"]


def test_claude_refusal_returns_nothing():
    assert claude_variants("Draft", client=FakeClient(["A"], stop_reason="refusal")) == []


def test_suggest_titles_ranks_all_sources(trained):
    model, _ = trained
    p = Predictor.load(model)
    video = VideoInput(title="Minecraft Hardcore Ep. 12", subscribers=100_000, category_id="20")
    client = FakeClient(["I Survived 100 Days in Minecraft Hardcore"])
    ranked = suggest_titles(p, video, use_llm=True, client=client)
    assert set(ranked["source"]) == {"yours", "rewrite", "claude"}
    assert ranked["median"].is_monotonic_decreasing
    assert ranked.iloc[0]["vs_best"] == 1.0
