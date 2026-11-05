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

# Templates that take a topic ("Minecraft Hardcore"), and ones that can wrap any title,
# including a whole sentence ("I Survived 100 Days in Minecraft", "Why I Quit My Job").
_TOPIC_TEMPLATES = [
    "I Tried {core} for 30 Days",
    "The Truth About {core}",
    "Why {core} Changes Everything",
    "{core} - Everything You Need to Know",
    "I Spent $1,000 on {core}",
    "Is {core} Actually Worth It?",
    "{core} Challenge",
    "Testing Viral {core} Hacks",
    "Beginner vs Pro: {core}",
]
_CLAUSE_TEMPLATES = [
    "{core} (I Didn't Expect This)",
    "{core} - Here's What Happened",
]
_FILLER = re.compile(r"\b(?:episode|ep\.?|part|pt\.?)\s*\d+\b|\(.*?\)|\[.*?\]", re.IGNORECASE)
# A draft that is already a sentence or a question, not a topic.
_SENTENCE = re.compile(
    r"^(?:i|i'm|i've|we|we're|my|our|you|your|this|these|why|how|what|when|where|who|which|"
    r"is|are|can|should|does|do|did|will|would)\b",
    re.IGNORECASE,
)
# "I tried / built / spent ... X": X is a topic the topic templates can use.
_FIRST_PERSON = re.compile(
    r"^(?:i|we)\s+(?:tried|tested|played|built|made|bought|spent \S+ on|learned|cooked|"
    r"reviewed|ranked|attempted)\s+(?P<topic>.+)$",
    re.IGNORECASE,
)
# A draft that already has a format: "X vs Y" or "$1 vs ..." comparisons get no second
# format; a trailing format word ("iPhone 17 Review") is dropped to find the topic.
_COMPARISON = re.compile(r"\bvs\.?\b|\$\d", re.IGNORECASE)
_FORMAT_SUFFIX = re.compile(
    r"\s+(?:review|tutorial|guide|explained|unboxing|tier list|ranked|reaction)$", re.IGNORECASE
)
_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the",
                "to", "vs", "with"}  # fmt: skip


def _core(title: str) -> str:
    core = _FILLER.sub("", title)
    core = re.sub(r"[|:\-–]+\s*$", "", core)
    return re.sub(r"\s+", " ", core).strip(" !?.|-–:") or title.strip()


def _title_case(text: str) -> str:
    words = text.split()
    out = []
    for i, w in enumerate(words):
        if any(c.isupper() for c in w) or not w[:1].isalpha():
            out.append(w)  # keep acronyms, "iPhone", "$1,000", "#shorts"
        elif 0 < i < len(words) - 1 and w.lower() in _SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def rule_based_variants(title: str) -> list[str]:
    """Deterministic rewrites of a draft title.

    Topic templates are only applied to topics: a draft that is already a sentence or a
    question gets the templates that can wrap any title, plus the topic templates on what
    follows "I tried / built / spent ..." when it has that shape. Lower-case drafts also get
    a title-cased version.
    """
    core = _title_case(_core(title))
    out = [t.format(core=core) for t in _CLAUSE_TEMPLATES]
    templates = _TOPIC_TEMPLATES
    topic = None
    if _SENTENCE.match(core):
        match = _FIRST_PERSON.match(core)
        if match:
            topic = _title_case(match["topic"])
            templates = [t for t in _TOPIC_TEMPLATES if not t.startswith("I ")]
    elif not _COMPARISON.search(core):
        topic = _FORMAT_SUFFIX.sub("", core) or core
    if topic and not _COMPARISON.search(topic):
        out += [t.format(core=topic) for t in templates]
    plain = [w for w in core.split() if w.isalpha() and w == w[:1].upper() + w[1:].lower()]
    if plain:
        longest = max(plain, key=len)
        out.append(core.replace(longest, longest.upper(), 1))  # one emphasised word
    if core != title.strip():
        out.append(core)  # draft without episode numbers / brackets, or title-cased
    return [t for t in dict.fromkeys(out) if t and len(t) <= 100 and t != title.strip()]


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
