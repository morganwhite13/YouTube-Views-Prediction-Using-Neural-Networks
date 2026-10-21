"""Render a trained model's evaluation as a self-contained, shareable web page.

Everything is inline (CSS, SVG charts), so the file can be opened locally, attached to an
email, or published as-is. Only the fonts load from Google Fonts, with system fallbacks.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NEURAL = "neural net (ours)"

CSS = """
/* Layout: a single reading column; tables and charts sit in full-width figure bands. */
:root {
  --bg: #f5f6fa; --surface: #ffffff; --fg: #161a24; --muted: #5a6275;
  --rule: #dce0ea; --accent: #2b59c3; --band: #a9bde9; --good: #1f7a4d; --warn: #9a5b00;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f121a; --surface: #161a24; --fg: #e5e8f1; --muted: #9aa2b6;
  --rule: #272d3b; --accent: #82a4ff; --band: #3a4d7a; --good: #5cc48e; --warn: #e0a548;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f121a; --surface: #161a24; --fg: #e5e8f1; --muted: #9aa2b6;
  --rule: #272d3b; --accent: #82a4ff; --band: #3a4d7a; --good: #5cc48e; --warn: #e0a548;
  color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font: 16px/1.6 var(--body); margin: 0; }
main { max-width: 860px; margin: 0 auto; padding-block: 48px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 40px; }
header { display: grid; gap: 12px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); }
h1 { font: 600 clamp(32px, 6vw, 48px)/1.05 var(--display); margin: 0; text-wrap: balance;
  letter-spacing: -.01em; }
h2 { font: 600 24px/1.2 var(--display); margin: 0 0 4px; text-wrap: balance; }
p { margin: 0; max-width: 65ch; }
.lead { font-size: 18px; color: var(--muted); }
.note { border: 1px solid var(--warn); color: var(--warn); border-radius: 6px;
  padding: 10px 14px; font-size: 14px; max-width: 65ch; }
.facts { display: flex; flex-wrap: wrap; gap: 8px 24px; margin: 4px 0 0; padding: 0;
  list-style: none; font: 14px/1.4 var(--mono); color: var(--muted); }
.facts b { color: var(--fg); font-weight: 600; }
.headline { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 8px;
  overflow: hidden; }
.headline div { background: var(--surface); padding: 16px 18px; display: grid; gap: 2px; }
.headline .v { font: 600 30px/1.1 var(--display); font-variant-numeric: tabular-nums; }
.headline .k { font-size: 13px; color: var(--muted); }
section { display: grid; gap: 14px; min-width: 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px;
  font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--rule);
  white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { font: 600 12px/1.3 var(--mono); letter-spacing: .04em; text-transform: uppercase;
  color: var(--muted); }
tr.ours td { font-weight: 600; }
tr.ours td:first-child::before { content: ""; display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; background: var(--accent); margin-right: 8px; vertical-align: 1px; }
.bar { display: inline-block; height: 6px; border-radius: 3px; background: var(--band);
  vertical-align: middle; margin-left: 8px; }
tr.ours .bar { background: var(--accent); }
figure { margin: 0; display: grid; gap: 8px; }
figcaption { font-size: 13px; color: var(--muted); max-width: 65ch; }
svg text { fill: var(--muted); font: 11px var(--mono); }
.imp { display: grid; gap: 8px; font-size: 14px; }
.imp .row { display: grid; grid-template-columns: minmax(110px, 160px) 1fr 56px; gap: 4px 12px;
  align-items: center; }
.imp .track { height: 10px; background: var(--rule); border-radius: 5px; min-width: 0; }
.imp .fill { height: 100%; background: var(--accent); border-radius: 5px; }
.imp .num { font: 13px var(--mono); color: var(--muted); font-variant-numeric: tabular-nums;
  text-align: right; }
ul.method { margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 65ch; }
footer { font: 12px/1.5 var(--mono); color: var(--muted); border-top: 1px solid var(--rule);
  padding-top: 16px; }
@media (max-width: 520px) { .imp .row { grid-template-columns: 1fr auto; }
  .imp .track { grid-column: 1 / -1; grid-row: 2; } }
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;'
    "600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600"
    '&display=swap">'
)


def _fmt_views(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def _sup(k: int) -> str:
    return "10" + "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] if d.isdigit() else "⁻" for d in str(k))


def scatter_svg(preds: pd.DataFrame, max_points: int = 500, seed: int = 0) -> str:
    """Predicted (p50, with p10-p90 whiskers) vs actual views on log-log axes, drawn to scale."""
    if len(preds) > max_points:
        preds = preds.sample(max_points, random_state=seed)
    a = np.log10(preds["views"].clip(lower=1).to_numpy())
    m = np.log10(preds["pred_median"].clip(lower=1).to_numpy())
    lo = np.log10(preds["pred_low"].clip(lower=1).to_numpy())
    hi = np.log10(preds["pred_high"].clip(lower=1).to_numpy())
    k0 = math.floor(min(a.min(), lo.min()))
    k1 = math.ceil(max(a.max(), hi.max()))
    w, h, left, bottom, top, right = 640, 440, 52, 40, 12, 12

    def sx(v):
        return left + (v - k0) / (k1 - k0) * (w - left - right)

    def sy(v):
        return h - bottom - (v - k0) / (k1 - k0) * (h - bottom - top)

    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
        'aria-label="Predicted versus actual views on the test set" '
        'style="max-width:100%;height:auto;display:block">'
    ]
    for k in range(k0, k1 + 1):
        x, y = sx(k), sy(k)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom}" '
            'stroke="var(--rule)" stroke-width="1"/>'
            f'<line x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" '
            'stroke="var(--rule)" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{h - bottom + 16}" text-anchor="middle">{_sup(k)}</text>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_sup(k)}</text>'
        )
    parts.append(
        f'<line x1="{sx(k0):.1f}" y1="{sy(k0):.1f}" x2="{sx(k1):.1f}" y2="{sy(k1):.1f}" '
        'stroke="var(--muted)" stroke-dasharray="4 4" stroke-width="1"/>'
    )
    for ai, li, hi_ in zip(a, lo, hi, strict=True):
        parts.append(
            f'<line x1="{sx(ai):.1f}" y1="{sy(li):.1f}" x2="{sx(ai):.1f}" y2="{sy(hi_):.1f}" '
            'stroke="var(--band)" stroke-width="1"/>'
        )
    for ai, mi in zip(a, m, strict=True):
        parts.append(
            f'<circle cx="{sx(ai):.1f}" cy="{sy(mi):.1f}" r="2.4" fill="var(--accent)" '
            'fill-opacity=".75"/>'
        )
    parts.append(
        f'<text x="{(left + w - right) / 2:.0f}" y="{h - 6}" text-anchor="middle">'
        "actual views</text>"
        f'<text transform="translate(12 {(top + h - bottom) / 2:.0f}) rotate(-90)" '
        'text-anchor="middle">predicted views</text></svg>'
    )
    return "".join(parts)


def _metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    rows = sorted(metrics.items(), key=lambda kv: kv[1]["mae_log"])
    worst = max(m["mae_log"] for _, m in rows)
    out = [
        '<div class="scroll"><table><thead><tr><th>Model</th><th>Typical error</th>'
        "<th>Within 2×</th><th>Within 10×</th><th>Rank corr.</th><th>MAE (log)</th>"
        "</tr></thead><tbody>"
    ]
    for name, m in rows:
        cls = ' class="ours"' if name == NEURAL else ""
        width = 60 * m["mae_log"] / worst
        rho = "–" if not np.isfinite(m.get("spearman", np.nan)) else f"{m['spearman']:.2f}"
        out.append(
            f"<tr{cls}><td>{html.escape(name)}</td>"
            f"<td>{m['median_factor_error']:.2f}×</td><td>{m['within_2x']:.0%}</td>"
            f"<td>{m['within_10x']:.0%}</td><td>{rho}</td>"
            f'<td>{m["mae_log"]:.3f}<span class="bar" style="width:{width:.0f}px"></span></td>'
            "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _importance(importance: dict[str, float]) -> str:
    items = [(k, v) for k, v in importance.items() if v > 0.0005]
    if not items:
        return "<p>No feature group changed the error noticeably when shuffled.</p>"
    top = max(v for _, v in items)
    rows = "".join(
        f'<div class="row"><div>{html.escape(k)}</div><div class="track"><div class="fill" '
        f'style="width:{100 * v / top:.1f}%"></div></div><div class="num">+{v:.3f}</div></div>'
        for k, v in items
    )
    return f'<div class="imp">{rows}</div>'


def render_html(
    meta: dict[str, Any], preds: pd.DataFrame | None = None, standalone: bool = True
) -> str:
    """Build the page. ``standalone=False`` omits the doctype (for hosts that add their own)."""
    ds = meta["dataset"]
    ours = meta["metrics"].get(NEURAL, {})
    best_baseline = min(
        (m["mae_log"] for k, m in meta["metrics"].items() if k != NEURAL), default=None
    )
    created = datetime.fromisoformat(meta["created_at"]).strftime("%d %B %Y")
    start, end = (str(d)[:10] for d in ds.get("published_range", ["?", "?"]))

    note = ""
    if ds.get("synthetic"):
        note = (
            '<p class="note">These results come from the built-in synthetic dataset. They show '
            "the pipeline working end to end; they are not a measure of accuracy on real "
            "YouTube videos.</p>"
        )
    verdict = ""
    if ours and best_baseline is not None:
        diff = best_baseline - ours["mae_log"]
        verdict = (
            f"The network's log-scale error is {abs(diff):.3f} "
            f"{'lower' if diff >= 0 else 'higher'} than the best baseline."
        )

    headline = ""
    if ours:
        cov = ours.get("interval_coverage")
        cells = [
            (f"{ours['median_factor_error']:.2f}×", "typical error (median factor)"),
            (f"{ours['within_2x']:.0%}", "of videos predicted within 2×"),
            (f"{cov:.0%}" if cov is not None else "–", "landed inside the p10–p90 range"),
            (f"{ours['spearman']:.2f}", "rank correlation with actual views"),
        ]
        headline = '<div class="headline">' + "".join(
            f'<div><span class="v">{v}</span><span class="k">{k}</span></div>' for v, k in cells
        ) + "</div>"  # fmt: skip

    scatter = ""
    if preds is not None and len(preds):
        scatter = (
            "<section><h2>Predictions against reality</h2><figure>"
            + scatter_svg(preds)
            + "<figcaption>Each dot is a test-set video: its predicted median against the views "
            "it actually had, on log scales. Vertical lines show the p10–p90 range; the dashed "
            "diagonal is a perfect prediction.</figcaption></figure></section>"
        )

    split_text = {
        "time": "trained on older uploads and tested on the newest",
        "channel": "tested on channels never seen in training",
        "random": "tested on randomly held-out videos",
    }.get(meta["split"], meta["split"])
    doctype = "<!doctype html>\n" if standalone else ""
    return f"""{doctype}<title>ytviews Model Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{FONTS}
<style>{CSS}</style>
<main>
  <header>
    <span class="eyebrow">ytviews v{html.escape(meta.get("version", ""))} · model report</span>
    <h1>How many views will a video get?</h1>
    <p class="lead">A neural network that predicts a YouTube video's view count from what is
    known before it goes live: title, description, thumbnail, category, length, upload time
    and the channel's size and recent results. It gives a range, not a single number.
    {verdict}</p>
    <ul class="facts">
      <li><b>{ds["videos"]:,}</b> videos</li><li><b>{ds["channels"]:,}</b> channels</li>
      <li>published <b>{start}</b> to <b>{end}</b></li>
      <li>median <b>{_fmt_views(ds["median_views"])}</b> views</li>
    </ul>
    {note}
  </header>
  {headline}
  <section>
    <h2>Against simpler models</h2>
    <p>All models were {split_text} ({meta["test_size"]:.0%} of videos held out). Typical
    error is the median factor between prediction and truth: 1.5× means half the predictions
    were within 50% of the real count.</p>
    {_metrics_table(meta["metrics"])}
  </section>
  {scatter}
  <section>
    <h2>What the network relies on</h2>
    <p>How much the test error grows when one group of inputs is shuffled between videos.
    Longer bars mean the model leans on that information more.</p>
    {_importance(meta.get("importance", {}))}
  </section>
  <section>
    <h2>How it was built</h2>
    <ul class="method">
      <li>Target: log of views at the moment they were fetched; the video's age at that moment
      is an input, so the model can predict views after 1 day, 1 week or 1 year.</li>
      <li>Text encoder: <code>{html.escape(meta.get("text_encoder", "?"))}</code>; thumbnails:
      <code>{html.escape(str(meta.get("image_encoder", "none")))}</code>.</li>
      <li>Wide-and-deep network with one tower per input; the channel's context gates the
      text, and a quantile head outputs p10, p50 and p90.</li>
      <li>Ensemble of {meta["train_config"].get("ensemble", "?")} networks, early stopping,
      and conformal calibration so the p10–p90 range covers about 80% of outcomes.</li>
      <li>No leakage: likes and comments are never inputs, the channel's history only uses
      earlier uploads, and every scaler and encoder is fitted on training videos only.</li>
    </ul>
  </section>
  <footer>Generated {created} by <code>ytviews report --html</code> ·
  trained on {html.escape(meta.get("trained_on", "?"))} in {meta.get("elapsed_s", "?")} s</footer>
</main>
"""


def write_html_report(model_dir: str | Path, out: str | Path, standalone: bool = True) -> Path:
    model_dir, out = Path(model_dir), Path(out)
    meta = json.loads((model_dir / "meta.json").read_text())
    preds_path = model_dir / "test_predictions.csv"
    preds = pd.read_csv(preds_path) if preds_path.exists() else None
    out.write_text(render_html(meta, preds, standalone=standalone))
    return out
