"""Real-data benchmark: collect a dataset with your API key, train, and report honest numbers.

``ytviews benchmark`` replaces the synthetic numbers in the README with real ones in one go:

1. **Collect** (quota-aware, resumable): the v1 project's seed channels plus trending videos
   (and their channels' uploads) in a few regions. A free API key allows 10,000 units a day;
   one run uses ``--quota-budget`` of them. Below ``--min-videos`` it stops and says to run
   again tomorrow; channels collected recently are skipped, so later runs add new data.
2. **Enrich** (no quota): Shorts flags and thumbnails from youtube.com.
3. **Train** twice: a *time* split (train on older uploads, test on the newest) and a
   *channel* split (test channels never seen in training, the harder case of a brand-new
   creator). Both are reported against the baselines.
4. **Write** ``REAL_RESULTS.md``: the dataset, both metric tables, and a short table to paste
   into the README.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate import OURS_NAME
from .pipeline import dataset_summary, run_experiment
from .train import TrainConfig

SPLITS = ("time", "channel")
DEFAULT_REGIONS = ("US", "GB", "IN", "CA", "AU")


def collect_dataset(
    client: Any,
    store: Any,
    *,
    regions: tuple[str, ...] = DEFAULT_REGIONS,
    videos_per_channel: int = 50,
    language: str | None = "en",
    log: Callable[[str], None] = print,
) -> int:
    """One quota-bounded collection pass: seed channels, then trending in each region.
    Channels collected in the last day are skipped, so repeated runs add new data."""
    from .collect import DEFAULT_SEED_CHANNELS, collect_channels, collect_trending
    from .youtube import QuotaExceeded

    total = 0
    filters = {"language": language or None, "progress": log}
    try:
        total += collect_channels(client, store, DEFAULT_SEED_CHANNELS, skip_fresh_days=1.0,
                                  max_videos_per_channel=videos_per_channel, **filters)  # fmt: skip
        total += collect_trending(client, store, regions=list(regions),
                                  max_videos_per_channel=videos_per_channel, **filters)  # fmt: skip
    except QuotaExceeded as exc:
        log(f"Stopped collecting: {exc}")
    return total


def enrich(store: Any, db: Path, log: Callable[[str], None] = print) -> None:
    """Shorts flags and thumbnails; skipped (with a note) when youtube.com is unreachable."""
    from .collect import detect_shorts, download_thumbnails
    from .images import thumbnail_dir_for

    try:
        log(f"Shorts check: {detect_shorts(store)}")
        ok, failed = download_thumbnails(store, thumbnail_dir_for(db))
        log(f"Thumbnails: {ok} new, {failed} failed")
    except OSError as exc:
        log(f"Skipped Shorts/thumbnails ({exc}); the model works without them")


def _table(metrics: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| model | MAE (log) | R² (log) | median factor error | within 2x | p10-p90 coverage |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in sorted(metrics.items(), key=lambda kv: kv[1]["mae_log"]):
        cov = f"{m['interval_coverage']:.0%}" if "interval_coverage" in m else "-"
        bold = "**" if name == OURS_NAME else ""
        lines.append(
            f"| {bold}{name}{bold} | {m['mae_log']:.3f} | {m['r2_log']:.3f} | "
            f"{m['median_factor_error']:.2f}x | {m['within_2x']:.0%} | {cov} |"
        )
    return lines


def render_results(summary: dict[str, Any], metas: dict[str, dict[str, Any]]) -> str:
    """The ``REAL_RESULTS.md`` page for the dataset ``summary`` and one meta per split."""
    first, last = (d[:10] for d in summary["published_range"])
    out = [
        "# ytviews on real YouTube data",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d} by `ytviews benchmark`.",
        "",
        f"* **{summary['videos']:,} videos** from **{summary['channels']:,} channels** in "
        f"{summary['categories']} categories ({summary['rows']:,} rows with snapshots)",
        f"* Published {first} to {last}; median {summary['median_views']:,.0f} views",
        "",
        "Errors are on the log scale; *median factor error* 1.8 means the typical prediction is "
        "within 1.8x of the true count. Coverage is how often the true count fell inside the "
        "predicted p10-p90 range (target 80%).",
    ]
    titles = {"time": "Time split: predicting the newest uploads",
              "channel": "Channel split: channels never seen in training"}  # fmt: skip
    for split, meta in metas.items():
        out += ["", f"## {titles.get(split, split)}", "", *_table(meta["metrics"])]
    ours = {s: m["metrics"][OURS_NAME] for s, m in metas.items() if OURS_NAME in m["metrics"]}
    if ours:
        out += ["", "## For the README", "", "| split | MAE (log) | within 2x | coverage |",
                "|---|---|---|---|"]  # fmt: skip
        for split, m in ours.items():
            out.append(f"| {split} | {m['mae_log']:.3f} | {m['within_2x']:.0%} | "
                       f"{m.get('interval_coverage', float('nan')):.0%} |")  # fmt: skip
    out += ["", "Sampling caveat: the API over-represents active, discoverable channels, so "
            "these numbers are for channels like the ones collected.", ""]  # fmt: skip
    return "\n".join(out)


def run_benchmark(
    df: Any,
    out_dir: str | Path,
    *,
    text_encoder: str = "auto",
    image_encoder: str = "auto",
    config: TrainConfig | None = None,
    splits: tuple[str, ...] = SPLITS,
    log: Callable[[str], None] = print,
) -> Path:
    """Train one model per split into ``out_dir/<split>`` and write ``REAL_RESULTS.md``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = {}
    for split in splits:
        log(f"\n== Training with the {split} split")
        metas[split] = run_experiment(
            df, out_dir / split, split=split, text_encoder=text_encoder,
            image_encoder=image_encoder, config=config, log=log,
        )  # fmt: skip
    path = out_dir / "REAL_RESULTS.md"
    path.write_text(render_results(dataset_summary(df), metas))
    return path
