"""Title suggestions: generate candidate titles, then let the view model rank them.

Candidates come from two places:

* **rules** - deterministic rewrites of your draft using common YouTube title patterns
  (curiosity gap, numbers, challenge framing, ...). Free and offline.
* **Claude** (optional, ``use_llm=True``) - asks Claude for fresh alternatives that keep the
  video's actual topic. Needs ``pip install ytviews[llm]`` and Anthropic credentials
  (e.g. ``ANTHROPIC_API_KEY``); each call is a paid API request.

The model only ranks what it is given. It can't tell whether a title is *honest* about the
video, so treat its ranking as one input, not a verdict.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd

from .categories import CATEGORY_NAMES

if TYPE_CHECKING:  # pragma: no cover
    from .predictor import Predictor, VideoInput

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-5-5"

_TEMPLATES = [
    "I Tried {core} for 30 Days",
    "{core} (I Didn't Expect This)",
    "The Truth About {core}",
    "Why {core} Changes Everything",
    "{core} - Everything You Need to Know",
    "I Spent $1,000 on {core}",
    "Is {core} Actually Worth It?",
    "{core} Challenge",
    "Testing Viral {core} Hacks",
    "Beginner vs Pro: {core}",
]
_FILLER = re.compile(r"\b(?:episode|ep\.?|part|pt\.?)\s*\d+\b|\(.*?\)|\[.*?\]", re.IGNORECASE)


def _core(title: str) -> str:
    core = _FILLER.sub("", title)
    core = re.sub(r"[|:\-–]+\s*$", "", core)
    return re.sub(r"\s+", " ", core).strip(" !?.|-–:") or title.strip()


def rule_based_variants(title: str) -> list[str]:
    """Deterministic rewrites of a draft title."""
    core = _core(title)
    words = core.split()
    out = [t.format(core=core) for t in _TEMPLATES]
    if words:
        longest = max(words, key=len)
        out.append(core.replace(longest, longest.upper(), 1))  # one emphasised word
    if core != title.strip():
        out.append(core)  # draft without episode numbers / brackets
    return [t for t in dict.fromkeys(out) if t and len(t) <= 100]


def claude_variants(
    title: str,
    *,
    description: str = "",
    category_id: str | None = None,
    n: int = 8,
    client=None,
) -> list[str]:
    """Ask Claude for ``n`` alternative titles. Returns [] (with a warning) on any failure."""
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic is not installed; run `pip install ytviews[llm]`")
        return []
    client = client or anthropic.Anthropic()
    category = CATEGORY_NAMES.get(str(category_id), "")
    prompt = (
        f"Suggest {n} alternative YouTube titles for this video.\n\n"
        f"Draft title: {title}\n"
        + (f"Category: {category}\n" if category else "")
        + (f"Description: {description[:1500]}\n" if description else "")
        + "\nKeep each title accurate to what the video is about (no bait the video can't "
        "deliver), under 70 characters, and varied in style: some curiosity-driven, some "
        "specific/benefit-driven, some short and punchy."
    )
    schema = {
        "type": "object",
        "properties": {"titles": {"type": "array", "items": {"type": "string"}}},
        "required": ["titles"],
        "additionalProperties": False,
    }
    try:
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIConnectionError as exc:
        log.warning("Could not reach the Claude API: %s", exc)
        return []
    except anthropic.APIStatusError as exc:
        log.warning("Claude API error %s: %s", exc.status_code, exc.message)
        return []
    if response.stop_reason == "refusal":
        log.warning("Claude declined to suggest titles for this video")
        return []
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        titles = json.loads(text)["titles"]
    except (json.JSONDecodeError, KeyError, TypeError):
        log.warning("Unexpected response from Claude: %r", text[:200])
        return []
    return [t.strip() for t in titles if isinstance(t, str) and t.strip()][:n]


def suggest_titles(
    predictor: Predictor,
    video: VideoInput,
    *,
    use_llm: bool = False,
    n_llm: int = 8,
    extra: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    """Rank the draft, rule-based variants, optional Claude variants and ``extra`` titles."""
    sources: dict[str, str] = {video.title: "yours"}
    for t in extra or []:
        sources.setdefault(t, "yours")
    for t in rule_based_variants(video.title):
        sources.setdefault(t, "rewrite")
    if use_llm:
        for t in claude_variants(
            video.title, description=video.description, category_id=video.category_id,
            n=n_llm, client=client,
        ):  # fmt: skip
            sources.setdefault(t, "claude")
    ranked = predictor.compare_titles(replace(video), list(sources))
    ranked.insert(1, "source", ranked["title"].map(sources))
    return ranked
