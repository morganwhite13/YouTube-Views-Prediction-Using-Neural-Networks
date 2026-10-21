"""Command-line interface: ``ytviews --help``."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from .config import load_settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Collect YouTube data, train the view-count model and make predictions.",
)
settings = load_settings()

DbOpt = Annotated[Path, typer.Option("--db", help="SQLite database path.")]
ModelOpt = Annotated[Path, typer.Option("--model", "-m", help="Trained model directory.")]


def _client(quota_budget: int):
    from .youtube import YouTubeClient

    try:
        return YouTubeClient(settings.api_key, quota_budget=quota_budget)
    except (ValueError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _fmt(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


# ---------------------------------------------------------------------------- data
@app.command()
def collect(
    channel: Annotated[list[str] | None, typer.Option(help="Channel id (repeatable).")] = None,
    seed_channels: Annotated[bool, typer.Option(help="Use the v1 project's channel list.")] = False,
    trending: Annotated[
        bool, typer.Option(help="Sample trending videos + their channels.")
    ] = False,
    region: Annotated[list[str] | None, typer.Option(help="Trending region (repeatable).")] = None,
    query: Annotated[list[str] | None, typer.Option(help="Search query to find channels.")] = None,
    category: Annotated[
        str | None, typer.Option(help="Restrict trending/search to a category.")
    ] = None,
    videos_per_channel: int = 50,
    language: Annotated[str, typer.Option(help="Keep videos in this language ('' = any).")] = "en",
    quota_budget: Annotated[int, typer.Option(help="Stop before using this many units.")] = 9000,
    db: DbOpt = settings.db_path,
) -> None:
    """Collect videos from the YouTube Data API into the local database."""
    from .collect import (
        DEFAULT_SEED_CHANNELS,
        collect_channels,
        collect_search,
        collect_trending,
    )
    from .store import VideoStore
    from .youtube import QuotaExceeded

    channels = list(channel or []) + (DEFAULT_SEED_CHANNELS if seed_channels else [])
    if not (channels or trending or query):
        raise typer.BadParameter(
            "choose at least one of --channel, --seed-channels, --trending, --query"
        )
    client = _client(quota_budget)
    filters = {"language": language or None, "progress": typer.echo}
    total = 0
    with VideoStore(db) as store:
        try:
            if channels:
                total += collect_channels(
                    client, store, channels, max_videos_per_channel=videos_per_channel, **filters
                )
            if trending:
                total += collect_trending(
                    client,
                    store,
                    regions=region or ["US"],
                    category_ids=[category],
                    max_videos_per_channel=videos_per_channel,
                    **filters,
                )
            if query:
                total += collect_search(
                    client,
                    store,
                    query,
                    category_id=category,
                    max_videos_per_channel=videos_per_channel,
                    **filters,
                )
        except QuotaExceeded as exc:
            typer.secho(f"Stopped early: {exc}", fg="yellow")
        counts = store.counts()
    typer.echo(
        f"Stored {total} video snapshots using {client.quota_used} quota units. DB: {counts}"
    )


@app.command()
def refresh(
    db: DbOpt = settings.db_path,
    quota_budget: int = 9000,
) -> None:
    """Re-fetch stats for all stored videos (adds a snapshot; builds growth curves)."""
    from .collect import refresh as do_refresh
    from .store import VideoStore

    client = _client(quota_budget)
    with VideoStore(db) as store:
        n = do_refresh(client, store)
        typer.echo(f"Refreshed {n} videos using {client.quota_used} quota units. {store.counts()}")


@app.command("import-legacy")
def import_legacy(
    files: Annotated[list[Path], typer.Argument(help="v1 JSON cache files.")],
    db: DbOpt = settings.db_path,
    fetched_at: Annotated[
        str | None,
        typer.Option(help="When the stats were fetched (ISO date). Default: file mtime."),
    ] = None,
) -> None:
    """Import JSON datasets produced by the original projectFinal.py."""
    from .store import VideoStore

    with VideoStore(db) as store:
        for f in files:
            n = store.import_legacy_json(f, fetched_at=fetched_at)
            typer.echo(f"{f}: imported {n} videos")
        missing = store.known_channel_ids_without_stats()
        typer.echo(f"DB now holds {store.counts()}")
        if missing:
            typer.echo(
                f"{len(missing)} channels have no stats; run `ytviews refresh` to fill them."
            )


@app.command()
def synth(
    db: DbOpt = settings.data_dir / "synthetic.db",
    channels: int = 120,
    snapshots: int = 2,
    seed: int = 0,
) -> None:
    """Generate a synthetic dataset (no API key needed)."""
    from .synthetic import write_synthetic_db

    if db.exists():
        db.unlink()
    counts = write_synthetic_db(db, n_channels=channels, snapshots=snapshots, seed=seed)
    typer.echo(f"Wrote synthetic dataset to {db}: {counts}")


@app.command()
def stats(db: DbOpt = settings.db_path) -> None:
    """Summarise the collected dataset."""
    from .pipeline import dataset_summary
    from .store import VideoStore

    with VideoStore(db) as store:
        df = store.load_dataset()
        typer.echo(json.dumps(store.counts() | dataset_summary(df), indent=2))
        top = df.groupby("category_id")["views"].agg(["count", "median"]).sort_values("count")
        typer.echo(top.iloc[::-1].head(10).to_string())


# ---------------------------------------------------------------------------- modelling
@app.command()
def train(
    db: DbOpt = settings.db_path,
    out: Annotated[
        Path, typer.Option(help="Where to save the model.")
    ] = settings.default_model_path,
    split: Annotated[str, typer.Option(help="time | channel | random")] = "time",
    test_size: float = 0.2,
    encoder: Annotated[str, typer.Option(help="auto | sbert | tfidf")] = "auto",
    all_snapshots: Annotated[bool, typer.Option(help="Train on every stored snapshot.")] = True,
    ensemble: int = 5,
    epochs: int = 300,
    hidden: int = 128,
    seed: int = 42,
    baselines: bool = True,
    refit: Annotated[bool, typer.Option(help="Refit on all data before saving.")] = True,
) -> None:
    """Train + evaluate against baselines, then save the model and a report."""
    from .pipeline import run_experiment
    from .store import VideoStore
    from .train import TrainConfig

    logging.basicConfig(level=logging.WARNING)
    with VideoStore(db) as store:
        df = store.load_dataset(all_snapshots=all_snapshots)
    if len(df) < 30:
        raise typer.BadParameter(f"only {len(df)} rows in {db}; collect more data first")
    cfg = TrainConfig(ensemble=ensemble, epochs=epochs, hidden=hidden, seed=seed)
    meta = run_experiment(
        df,
        out,
        split=split,
        test_size=test_size,
        text_encoder=encoder,
        config=cfg,
        baselines=baselines,
        refit=refit,
        log=typer.echo,
    )
    typer.echo("")
    typer.echo((out / "report.md").read_text())
    typer.echo(f"Model saved to {out} ({meta['elapsed_s']}s)")


def _channel_context(channel_id: str, n_recent: int = 10) -> dict:
    client = _client(quota_budget=50)
    ch = client.channels([channel_id])
    if not ch:
        raise typer.BadParameter(f"channel {channel_id} not found")
    ch = ch[0]
    recent = client.videos(client.upload_ids(ch["uploads_playlist"], n_recent))
    now = datetime.now(timezone.utc)
    ages = [
        (now - datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))).total_seconds()
        / 86400
        for v in recent
    ]
    return {
        "subscribers": ch["subscribers"],
        "channel_video_count": ch["channel_video_count"],
        "channel_total_views": ch["channel_views"],
        "recent_video_views": [v["views"] for v in recent],
        "recent_video_age_days": ages,
    }


def _video_input(
    title, subscribers, description, tags, category, duration, days, channel_id, recent_views
):
    from .predictor import VideoInput

    ctx = _channel_context(channel_id) if channel_id else {}
    if subscribers is not None:
        ctx["subscribers"] = subscribers
    if recent_views:
        ctx["recent_video_views"] = [float(x) for x in recent_views.split(",") if x.strip()]
    return VideoInput(
        title=title,
        description=description,
        category_id=category,
        duration_s=duration,
        age_days=days,
        tags=[t.strip() for t in (tags or "").split(",") if t.strip()],
        **ctx,
    )


TitleOpt = Annotated[str, typer.Option(help="Video title.")]
SubsOpt = Annotated[float | None, typer.Option(help="Channel subscriber count.")]
ChannelOpt = Annotated[
    str | None, typer.Option(help="Fetch channel stats + recent uploads via the API.")
]
RecentOpt = Annotated[
    str | None, typer.Option(help="Comma-separated views of your recent uploads.")
]


@app.command()
def predict(
    title: TitleOpt,
    subscribers: SubsOpt = None,
    description: str = "",
    tags: Annotated[str | None, typer.Option(help="Comma-separated tags.")] = None,
    category: Annotated[str, typer.Option(help="YouTube category id, e.g. 20=Gaming.")] = "24",
    duration: Annotated[float | None, typer.Option(help="Duration in seconds.")] = None,
    days: Annotated[float, typer.Option(help="Predict views this many days after upload.")] = 7,
    channel_id: ChannelOpt = None,
    recent_views: RecentOpt = None,
    model: ModelOpt = settings.default_model_path,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Predict views for a video, with an uncertainty band and growth trajectory."""
    from .predictor import Predictor

    p = Predictor.load(model)
    video = _video_input(
        title, subscribers, description, tags, category, duration, days, channel_id, recent_views
    )
    pred = p.predict(video)[0]
    traj = p.trajectory(video)
    if as_json:
        typer.echo(
            json.dumps(
                {"prediction": pred.as_dict(), "trajectory": traj.to_dict(orient="records")},
                indent=2,
            )
        )
        return
    typer.echo(f'"{title}" after {days:g} days:')
    typer.secho(f"  ~{_fmt(pred.median)} views", bold=True)
    typer.echo(f"  {pred.coverage:.0%} range: {_fmt(pred.low)} - {_fmt(pred.high)}")
    typer.echo("\nExpected trajectory:")
    for r in traj.itertuples():
        typer.echo(f"  day {r.days:>4g}: {_fmt(r.median):>7}  ({_fmt(r.low)} - {_fmt(r.high)})")


@app.command()
def compare(
    titles: Annotated[list[str], typer.Argument(help="Candidate titles to compare.")],
    subscribers: SubsOpt = None,
    description: str = "",
    category: str = "24",
    duration: float | None = None,
    days: float = 7,
    channel_id: ChannelOpt = None,
    recent_views: RecentOpt = None,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Rank alternative titles for the same video."""
    from .predictor import Predictor

    p = Predictor.load(model)
    base = _video_input(
        titles[0],
        subscribers,
        description,
        None,
        category,
        duration,
        days,
        channel_id,
        recent_views,
    )
    ranked = p.compare_titles(base, titles)
    for i, r in enumerate(ranked.itertuples(), 1):
        typer.echo(f"{i}. {_fmt(r.median):>7}  ({r.vs_best:>4.0%} of best)  {r.title}")


@app.command()
def report(model: ModelOpt = settings.default_model_path) -> None:
    """Print the evaluation report saved with a model."""
    path = model / "report.md"
    if not path.exists():
        raise typer.BadParameter(f"no report at {path}")
    typer.echo(path.read_text())


@app.command()
def demo(
    out: Path = settings.model_dir / "demo",
    channels: int = 120,
    encoder: str = "tfidf",
) -> None:
    """Offline end-to-end run on synthetic data: generate -> train -> predict."""
    from .pipeline import run_experiment
    from .predictor import Predictor, VideoInput
    from .store import VideoStore
    from .synthetic import write_synthetic_db
    from .train import TrainConfig

    db = settings.data_dir / "synthetic.db"
    if db.exists():
        db.unlink()
    typer.echo(f"Synthetic data: {write_synthetic_db(db, n_channels=channels)}")
    with VideoStore(db) as store:
        df = store.load_dataset(all_snapshots=True)
    run_experiment(df, out, text_encoder=encoder, config=TrainConfig(ensemble=2), log=typer.echo)
    typer.echo((out / "report.md").read_text())
    p = Predictor.load(out)
    base = VideoInput(title="", subscribers=250_000, category_id="20", duration_s=900)
    typer.echo("Title comparison for a 250K-subscriber gaming channel (7 days after upload):")
    ranked = p.compare_titles(
        base,
        [
            "I Survived 100 Days in Minecraft",
            "Minecraft Tutorial - Part 3",
            "Weekly Minecraft Update",
            "$1 vs $10,000 Minecraft Build",
        ],
    )
    for r in ranked.itertuples():
        typer.echo(f"  {_fmt(r.median):>7}  {r.title}")


if __name__ == "__main__":  # pragma: no cover
    app()
