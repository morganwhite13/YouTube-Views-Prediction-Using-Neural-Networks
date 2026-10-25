"""Plan a video: every tool in one pass.

Starting from a draft (title, description, optional thumbnails, channel context) the planner

1. predicts the draft and explains the number,
2. generates and ranks title ideas (rules, optionally Claude, plus any you supply),
3. ranks the thumbnails you supply (if the model uses thumbnails),
4. finds the best publish slot in the coming week for the improved video,
5. predicts the *combined* result - the real joint effect, not a product of separate gains -
   and keeps the draft if the combination doesn't beat it.
"""

from __future__ import annotations

import base64
import html
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .predictor import Prediction, Predictor, VideoInput
from .report_html import CSS, FONTS, _fmt_views
from .suggest import suggest_titles

OPTION_LEVER = {"title": "title", "thumbnail_path": "thumbnail", "published_at": "publish time"}
LEVERS = {"title text": "title", "title style": "title", "thumbnail": "thumbnail",
          "publish time": "publish time"}  # fmt: skip


@dataclass
class Plan:
    draft: VideoInput
    draft_prediction: Prediction
    explanation: Any  # explain.Explanation
    titles: pd.DataFrame
    thumbnails: pd.DataFrame | None
    slots: pd.DataFrame  # the whole weekday x hour grid
    best: VideoInput
    best_prediction: Prediction
    timezone: str
    target: str = "views"
    uplift_explanation: Any = None  # explain.Explanation of best relative to the draft
    skip_cost: dict[str, float] | None = None  # lever -> best / (best without that change)

    def uplift_by_lever(self) -> pd.DataFrame:
        """How much of the uplift each change accounts for (exact Shapley, multiplicative)."""
        if self.uplift_explanation is None:
            return pd.DataFrame(columns=["lever", "factor"])
        eff = self.uplift_explanation.effects
        lever = eff["group"].map(LEVERS).fillna("other")
        out = eff.groupby(lever)["log_effect"].sum()
        out = out[out.abs() > 1e-6].sort_values(ascending=False)
        skip = self.skip_cost or {}
        return pd.DataFrame(
            {
                "lever": out.index,
                "factor": np.exp(out.to_numpy()),
                "skip_cost": [skip.get(k, np.nan) for k in out.index],
            }
        )

    @property
    def uplift(self) -> float:
        return (self.best_prediction.median + 1) / (self.draft_prediction.median + 1)

    @property
    def top_slots(self) -> pd.DataFrame:
        return self.slots.sort_values("median", ascending=False).head(5)

    def changes(self) -> list[str]:
        out = []
        if self.best.title != self.draft.title:
            out.append(f'Use the title "{self.best.title}"')
        if self.best.thumbnail_path and self.best.thumbnail_path != self.draft.thumbnail_path:
            out.append(f"Use the thumbnail {Path(self.best.thumbnail_path).name}")
        if self.best.published_at is not None and self.best.published_at != self.draft.published_at:
            local = self.best.published_at
            out.append(f"Publish on {local:%A %d %b} at {local:%H:%M} ({self.timezone})")
        return out or ["Keep the draft as it is: none of the alternatives beat it"]

    def to_dict(self) -> dict[str, Any]:
        def video(v: VideoInput) -> dict[str, Any]:
            d = asdict(v)
            d["published_at"] = v.published_at.isoformat() if v.published_at else None
            return d

        records = self.slots.assign(published_at=self.slots["published_at"].astype(str))
        return {
            "target": self.target,
            "draft": video(self.draft),
            "draft_prediction": self.draft_prediction.as_dict(),
            "best": video(self.best),
            "best_prediction": self.best_prediction.as_dict(),
            "uplift": self.uplift,
            "changes": self.changes(),
            "uplift_by_lever": self.uplift_by_lever().to_dict(orient="records"),
            "explanation": self.explanation.effects.to_dict(orient="records"),
            "titles": self.titles.to_dict(orient="records"),
            "thumbnails": None
            if self.thumbnails is None
            else self.thumbnails.to_dict(orient="records"),
            "top_slots": records.sort_values("median", ascending=False)
            .head(5)
            .to_dict(orient="records"),
        }


def make_plan(
    predictor: Predictor,
    video: VideoInput,
    *,
    extra_titles: list[str] | None = None,
    thumbnails: list[str] | None = None,
    timezone_name: str = "UTC",
    use_llm: bool = False,
    client: Any = None,
) -> Plan:
    draft_pred = predictor.predict(video)[0]
    explanation = predictor.explain(video)

    titles = suggest_titles(predictor, video, use_llm=use_llm, extra=extra_titles, client=client)
    options: dict[str, Any] = {"title": str(titles.iloc[0]["title"])}
    improved = replace(video, **options)

    ranked_thumbs = None
    candidates = list(dict.fromkeys([t for t in [video.thumbnail_path, *(thumbnails or [])] if t]))
    if candidates and predictor.features.uses_thumbnails:
        ranked_thumbs = predictor.compare_thumbnails(improved, candidates)
        options["thumbnail_path"] = str(ranked_thumbs.iloc[0]["thumbnail"])
        improved = replace(improved, thumbnail_path=options["thumbnail_path"])

    slots = predictor.publish_times(improved, timezone_name)
    best_slot = slots.loc[slots["median"].idxmax()]
    options["published_at"] = best_slot["published_at"].to_pydatetime()

    # Each pick was made with the others partly fixed, and changes interact, so adopt the
    # subset of changes that predicts best. The empty subset is the draft: never worse.
    options = {k: v for k, v in options.items() if v != getattr(video, k)}
    subsets = [dict(c) for r in range(len(options) + 1) for c in combinations(options.items(), r)]
    versions = [replace(video, **sub) for sub in subsets]
    preds = predictor.predict(versions)
    i = max(range(len(preds)), key=lambda j: (preds[j].median, -len(subsets[j])))
    best, best_pred = (video, draft_pred) if not subsets[i] else (versions[i], preds[i])
    skip_cost = {}
    for key in subsets[i]:
        without = {k: v for k, v in subsets[i].items() if k != key}
        j = subsets.index(without)
        skip_cost[OPTION_LEVER[key]] = (best_pred.median + 1) / (preds[j].median + 1)
    uplift = predictor.explain(best, reference=video) if best is not video else None
    return Plan(
        video, draft_pred, explanation, titles, ranked_thumbs, slots, best, best_pred,
        timezone_name, predictor.target, uplift, skip_cost,
    )  # fmt: skip


# --------------------------------------------------------------------------- HTML


def _thumb_tag(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return ""
    if len(data) > 1_500_000:
        return ""
    mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    return (
        f'<img alt="{html.escape(p.name)}" src="data:{mime};base64,'
        f'{base64.b64encode(data).decode()}" style="width:100%;max-width:320px;'
        'border-radius:6px;display:block">'
    )


PLAN_CSS = """
.compare { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px; }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px;
  padding: 16px; display: grid; gap: 8px; align-content: start; min-width: 0; }
.card.best { border-color: var(--accent); }
.big { font: 600 34px/1.1 var(--display); font-variant-numeric: tabular-nums; }
.sub { color: var(--muted); font-size: 13px; }
ol.changes { margin: 0; padding-left: 22px; display: grid; gap: 6px; }
td.t { white-space: normal; text-align: left; min-width: 200px; }
"""


def plan_html(plan: Plan) -> str:
    unit = plan.target
    d, b = plan.draft_prediction, plan.best_prediction

    def card(label: str, video: VideoInput, pred: Prediction, cls: str = "") -> str:
        return (
            f'<div class="card {cls}"><span class="eyebrow">{label}</span>'
            f"{_thumb_tag(video.thumbnail_path)}"
            f"<strong>{html.escape(video.title)}</strong>"
            f'<span class="big">~{_fmt_views(pred.median)}</span>'
            f'<span class="sub">{unit} after {video.age_days:g} days · range '
            f"{_fmt_views(pred.low)}–{_fmt_views(pred.high)}</span></div>"
        )

    effects = "".join(
        f"<tr><td>{html.escape(r.group)}</td><td>×{r.factor:.2f}</td></tr>"
        for r in plan.explanation.effects.itertuples()
    )
    titles = "".join(
        f'<tr><td class="t">{html.escape(r.title)}</td><td>{html.escape(r.source)}</td>'
        f"<td>{_fmt_views(r.median)}</td><td>{r.vs_best:.0%}</td></tr>"
        for r in plan.titles.head(8).itertuples()
    )
    thumbs = ""
    if plan.thumbnails is not None:
        rows = "".join(
            f'<tr><td class="t">{html.escape(Path(r.thumbnail).name)}</td>'
            f"<td>{_fmt_views(r.median)}</td><td>{r.vs_best:.0%}</td></tr>"
            for r in plan.thumbnails.itertuples()
        )
        thumbs = (
            '<section><h2>Thumbnails</h2><div class="scroll"><table><thead><tr><th>File</th>'
            f"<th>Expected</th><th>vs best</th></tr></thead><tbody>{rows}</tbody></table></div>"
            "</section>"
        )
    slots = "".join(
        f"<tr><td>{r.weekday}</td><td>{r.hour:02d}:00</td><td>{_fmt_views(r.median)}</td></tr>"
        for r in plan.top_slots.itertuples()
    )
    changes = "".join(f"<li>{html.escape(c)}</li>" for c in plan.changes())
    levers = plan.uplift_by_lever()
    lever_rows = "".join(
        f"<tr><td>{html.escape(r.lever)}</td><td>×{r.factor:.2f}</td><td>"
        f"{'–' if np.isnan(r.skip_cost) else f'−{1 - 1 / r.skip_cost:.0%}'}</td></tr>"
        for r in levers.itertuples()
    )
    uplift_section = (
        "<section><h2>Where the uplift comes from</h2><p>Each change's share of the gain, "
        "with interactions between changes split fairly (Shapley values). The factors "
        f"multiply to the {plan.uplift:.2f}× total. The last column is what skipping only that "
        'change would cost, with the others kept.</p><div class="scroll"><table><thead><tr>'
        "<th>Change</th><th>Share of uplift</th><th>If you skip it</th></tr></thead>"
        f"<tbody>{lever_rows}</tbody></table></div>"
        "</section>"
        if len(levers)
        else ""
    )
    n = len(plan.changes()) if plan.best is not plan.draft else 0
    headline = (
        f"{plan.uplift:.2f}× the {unit} with {n} change{'s' if n != 1 else ''}"
        if n
        else "Your draft is already the best version found"
    )
    created = datetime.now(timezone.utc).strftime("%d %B %Y")
    return f"""<!doctype html>
<title>Video Plan</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{FONTS}
<style>{CSS}{PLAN_CSS}</style>
<main>
  <header>
    <span class="eyebrow">ytviews · video plan</span>
    <h1>{headline}</h1>
    <p class="lead">The draft is expected to reach about {_fmt_views(d.median)} {unit}. The best
    combination of title, thumbnail and publish time found below reaches about
    {_fmt_views(b.median)}. Both are model estimates with wide ranges.</p>
  </header>
  <div class="compare">{card("Your draft", plan.draft, d)}{
        card("Best found", plan.best, b, "best")
    }</div>
  <section><h2>What to change</h2><ol class="changes">{changes}</ol></section>
  {uplift_section}
  <section>
    <h2>Why the draft scores what it does</h2>
    <p>Each factor multiplies what an average video of the same age gets.</p>
    <div class="scroll"><table><thead><tr><th>Feature group</th><th>Effect</th></tr></thead>
    <tbody>{effects}</tbody></table></div>
  </section>
  <section>
    <h2>Title ideas</h2>
    <p>Check that the winner is still an honest description of the video.</p>
    <div class="scroll"><table><thead><tr><th>Title</th><th>Source</th><th>Expected</th>
    <th>vs best</th></tr></thead><tbody>{titles}</tbody></table></div>
  </section>
  {thumbs}
  <section>
    <h2>Best publish slots ({html.escape(plan.timezone)})</h2>
    <div class="scroll"><table><thead><tr><th>Day</th><th>Time</th><th>Expected</th></tr>
    </thead><tbody>{slots}</tbody></table></div>
  </section>
  <footer>Generated {created} by <code>ytviews plan</code></footer>
</main>
"""
