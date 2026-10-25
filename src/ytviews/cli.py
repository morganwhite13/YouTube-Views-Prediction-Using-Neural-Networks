"""Command-line interface: ``ytviews --help``."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
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
    skip_fresh_days: Annotated[
        float, typer.Option(help="Skip channels collected less than this many days ago.")
    ] = 1.0,
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
                    client,
                    store,
                    channels,
                    max_videos_per_channel=videos_per_channel,
                    skip_fresh_days=skip_fresh_days,
                    **filters,
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


@app.command()
def thumbnails(
    db: DbOpt = settings.db_path,
    workers: int = 8,
    overwrite: bool = False,
) -> None:
    """Download thumbnails for stored videos (uses no API quota)."""
    from .collect import download_thumbnails
    from .images import thumbnail_dir_for
    from .store import VideoStore

    folder = thumbnail_dir_for(db)
    with VideoStore(db) as store:
        ok, failed = download_thumbnails(store, folder, workers=workers, overwrite=overwrite)
    typer.echo(f"Downloaded {ok} thumbnails to {folder} ({failed} failed)")


@app.command()
def shorts(
    db: DbOpt = settings.db_path,
    workers: int = 8,
    recheck: Annotated[bool, typer.Option(help="Re-check videos already classified.")] = False,
) -> None:
    """Detect which stored videos are Shorts (asks youtube.com; no API quota)."""
    from .collect import detect_shorts
    from .store import VideoStore

    with VideoStore(db) as store:
        counts = detect_shorts(store, workers=workers, recheck=recheck)
    typer.echo(
        f"{counts['shorts']} Shorts, {counts['not_shorts']} regular videos, "
        f"{counts['unknown']} could not be checked (those fall back to the length rule)"
    )


@app.command()
def export(
    out: Annotated[Path, typer.Argument(help="Output file: .csv or .parquet")],
    db: DbOpt = settings.db_path,
    all_snapshots: Annotated[bool, typer.Option(help="One row per snapshot.")] = False,
    features: Annotated[
        bool, typer.Option(help="Add the model's engineered numeric features as columns.")
    ] = False,
) -> None:
    """Export the collected dataset for analysis in other tools."""
    from .features import add_channel_history, numeric_frame

    df = _load_training_frame(db, all_snapshots=all_snapshots)
    if features:
        df = add_channel_history(df)
        df = df.join(numeric_frame(df).add_prefix("f_"))
    if out.suffix == ".parquet":
        try:
            df.to_parquet(out, index=False)
        except ImportError as exc:
            raise typer.BadParameter("writing .parquet needs pyarrow: pip install pyarrow") from exc
    elif out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        raise typer.BadParameter("use a .csv or .parquet file name")
    typer.echo(f"Wrote {len(df)} rows x {df.shape[1]} columns to {out}")


# ---------------------------------------------------------------------------- modelling
def _load_training_frame(db: Path, all_snapshots: bool = True):
    """Load the dataset and attach thumbnails if a thumbnail folder exists for this DB."""
    from .images import attach_thumbnails, thumbnail_dir_for
    from .store import VideoStore

    with VideoStore(db) as store:
        df = store.load_dataset(all_snapshots=all_snapshots)
    folder = thumbnail_dir_for(db)
    return attach_thumbnails(df, folder) if folder.exists() else df


@app.command()
def train(
    db: DbOpt = settings.db_path,
    out: Annotated[
        Path, typer.Option(help="Where to save the model.")
    ] = settings.default_model_path,
    split: Annotated[str, typer.Option(help="time | channel | random")] = "time",
    test_size: float = 0.2,
    target: Annotated[str, typer.Option(help="What to predict: views | likes | comments")] = (
        "views"
    ),
    encoder: Annotated[str, typer.Option(help="auto | sbert | sbert-multi | tfidf")] = "auto",
    image_encoder: Annotated[
        str, typer.Option(help="auto | clip | stats | stats+faces | none")
    ] = "auto",
    all_snapshots: Annotated[bool, typer.Option(help="Train on every stored snapshot.")] = True,
    ensemble: int = 5,
    epochs: int = 300,
    hidden: int = 64,
    seed: int = 42,
    baselines: bool = True,
    refit: Annotated[bool, typer.Option(help="Refit on all data before saving.")] = True,
    config: Annotated[
        Path | None, typer.Option(help="TrainConfig JSON (e.g. from `ytviews tune`).")
    ] = None,
) -> None:
    """Train + evaluate against baselines, then save the model and a report."""
    from .pipeline import run_experiment
    from .train import TrainConfig

    logging.basicConfig(level=logging.WARNING)
    df = _load_training_frame(db, all_snapshots)
    if len(df) < 30:
        raise typer.BadParameter(f"only {len(df)} rows in {db}; collect more data first")
    if config is not None:
        from .tune import load_config

        cfg = load_config(config)
    else:
        cfg = TrainConfig(ensemble=ensemble, epochs=epochs, hidden=hidden, seed=seed)
    meta = run_experiment(
        df,
        out,
        split=split,
        test_size=test_size,
        text_encoder=encoder,
        image_encoder=image_encoder,
        config=cfg,
        baselines=baselines,
        refit=refit,
        target=target,
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
    title: str,
    *,
    subscribers: float | None = None,
    description: str = "",
    tags: str | None = None,
    category: str = "24",
    duration: float | None = None,
    days: float = 7,
    channel_id: str | None = None,
    recent_views: str | None = None,
    thumbnail: Path | None = None,
):
    from .predictor import VideoInput

    ctx = _channel_context(channel_id) if channel_id else {}
    if subscribers is not None:
        ctx["subscribers"] = subscribers
    if recent_views:
        ctx["recent_video_views"] = [float(x) for x in recent_views.split(",") if x.strip()]
    if thumbnail is not None and not thumbnail.exists():
        raise typer.BadParameter(f"thumbnail {thumbnail} does not exist")
    return VideoInput(
        title=title,
        description=description,
        category_id=category,
        duration_s=duration,
        age_days=days,
        tags=[t.strip() for t in (tags or "").split(",") if t.strip()],
        thumbnail_path=str(thumbnail) if thumbnail else None,
        **ctx,
    )


TitleOpt = Annotated[str, typer.Option(help="Video title.")]
SubsOpt = Annotated[float | None, typer.Option(help="Channel subscriber count.")]
DescOpt = Annotated[str, typer.Option(help="Video description.")]
TagsOpt = Annotated[str | None, typer.Option(help="Comma-separated tags.")]
CategoryOpt = Annotated[str, typer.Option(help="YouTube category id, e.g. 20=Gaming.")]
DurationOpt = Annotated[float | None, typer.Option(help="Duration in seconds.")]
DaysOpt = Annotated[float, typer.Option(help="Predict views this many days after upload.")]
ChannelOpt = Annotated[
    str | None, typer.Option(help="Fetch channel stats + recent uploads via the API.")
]
RecentOpt = Annotated[
    str | None, typer.Option(help="Comma-separated views of your recent uploads.")
]
ThumbOpt = Annotated[Path | None, typer.Option(help="Thumbnail image file.")]


@app.command()
def predict(
    title: TitleOpt,
    subscribers: SubsOpt = None,
    description: DescOpt = "",
    tags: TagsOpt = None,
    category: CategoryOpt = "24",
    duration: DurationOpt = None,
    days: DaysOpt = 7,
    channel_id: ChannelOpt = None,
    recent_views: RecentOpt = None,
    thumbnail: ThumbOpt = None,
    model: ModelOpt = settings.default_model_path,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Predict views for a video, with an uncertainty band and growth trajectory."""
    from .predictor import Predictor

    p = Predictor.load(model)
    video = _video_input(
        title, subscribers=subscribers, description=description, tags=tags, category=category,
        duration=duration, days=days, channel_id=channel_id, recent_views=recent_views,
        thumbnail=thumbnail,
    )  # fmt: skip
    pred = p.predict(video)[0]
    traj = p.trajectory(video)
    if as_json:
        out = {"prediction": pred.as_dict(), "trajectory": traj.to_dict(orient="records")}
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(f'"{title}" after {days:g} days:')
    typer.secho(f"  ~{_fmt(pred.median)} {p.target}", bold=True)
    typer.echo(f"  {pred.coverage:.0%} range: {_fmt(pred.low)} - {_fmt(pred.high)}")
    typer.echo("\nExpected trajectory:")
    for r in traj.itertuples():
        typer.echo(f"  day {r.days:>4g}: {_fmt(r.median):>7}  ({_fmt(r.low)} - {_fmt(r.high)})")


@app.command()
def compare(
    titles: Annotated[list[str], typer.Argument(help="Candidate titles to compare.")],
    subscribers: SubsOpt = None,
    description: DescOpt = "",
    category: CategoryOpt = "24",
    duration: DurationOpt = None,
    days: DaysOpt = 7,
    channel_id: ChannelOpt = None,
    recent_views: RecentOpt = None,
    thumbnail: ThumbOpt = None,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Rank alternative titles for the same video."""
    from .predictor import Predictor

    p = Predictor.load(model)
    base = _video_input(
        titles[0], subscribers=subscribers, description=description, category=category,
        duration=duration, days=days, channel_id=channel_id, recent_views=recent_views,
        thumbnail=thumbnail,
    )  # fmt: skip
    ranked = p.compare_titles(base, titles)
    for i, r in enumerate(ranked.itertuples(), 1):
        typer.echo(f"{i}. {_fmt(r.median):>7}  ({r.vs_best:>4.0%} of best)  {r.title}")


@app.command()
def explain(
    title: TitleOpt,
    subscribers: SubsOpt = None,
    description: DescOpt = "",
    tags: TagsOpt = None,
    category: CategoryOpt = "24",
    duration: DurationOpt = None,
    days: DaysOpt = 7,
    channel_id: ChannelOpt = None,
    recent_views: RecentOpt = None,
    thumbnail: ThumbOpt = None,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Explain a prediction: which feature groups push it up or down."""
    from .predictor import Predictor

    p = Predictor.load(model)
    video = _video_input(
        title, subscribers=subscribers, description=description, tags=tags, category=category,
        duration=duration, days=days, channel_id=channel_id, recent_views=recent_views,
        thumbnail=thumbnail,
    )  # fmt: skip
    ex = p.explain(video)
    typer.echo(f"An average video after {days:g} days: ~{_fmt(ex.baseline_views)} {p.target}")
    for r in ex.effects.itertuples():
        bar = ("+" if r.log_effect > 0 else "-") * min(20, max(1, round(abs(r.log_effect) * 8)))
        typer.echo(f"  {r.group:<18} x{r.factor:<6.2f} {bar}")
    typer.secho(f"This video: ~{_fmt(ex.predicted_views)} {p.target}", bold=True)


@app.command()
def suggest(
    title: TitleOpt,
    subscribers: SubsOpt = None,
    description: DescOpt = "",
    category: CategoryOpt = "24",
    duration: DurationOpt = None,
    days: DaysOpt = 7,
    channel_id: ChannelOpt = None,
    recent_views: RecentOpt = None,
    thumbnail: ThumbOpt = None,
    llm: Annotated[
        bool, typer.Option(help="Also ask Claude for ideas (needs ANTHROPIC_API_KEY; paid).")
    ] = False,
    top: int = 10,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Generate title ideas from your draft and rank them with the model."""
    from .predictor import Predictor
    from .suggest import suggest_titles

    p = Predictor.load(model)
    video = _video_input(
        title, subscribers=subscribers, description=description, category=category,
        duration=duration, days=days, channel_id=channel_id, recent_views=recent_views,
        thumbnail=thumbnail,
    )  # fmt: skip
    ranked = suggest_titles(p, video, use_llm=llm)
    for i, r in enumerate(ranked.head(top).itertuples(), 1):
        mark = " <- your draft" if r.source == "yours" else ""
        typer.echo(f"{i:>2}. {_fmt(r.median):>7}  [{r.source:<7}] {r.title}{mark}")
    if title not in set(ranked.head(top)["title"]):
        rank = int(ranked.index[ranked["title"] == title][0]) + 1
        typer.echo(f"    (your draft ranks #{rank} of {len(ranked)})")


@app.command()
def track(
    title: TitleOpt,
    views: Annotated[float, typer.Option(help="Views the video has right now.")],
    hours: Annotated[float, typer.Option(help="Hours since it was published.")],
    subscribers: SubsOpt = None,
    description: DescOpt = "",
    category: CategoryOpt = "24",
    duration: DurationOpt = None,
    channel_id: ChannelOpt = None,
    recent_views: RecentOpt = None,
    thumbnail: ThumbOpt = None,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Forecast where a live video will end up, from its views so far."""
    from .predictor import Predictor

    p = Predictor.load(model)
    video = _video_input(
        title, subscribers=subscribers, description=description, category=category,
        duration=duration, channel_id=channel_id, recent_views=recent_views, thumbnail=thumbnail,
    )  # fmt: skip
    nc = p.track(video, views, hours / 24)
    pace = nc.vs_expected
    typer.echo(
        f"After {hours:g} hours the model expected ~{_fmt(nc.expected_views)} views; "
        f"you have {_fmt(views)} ({pace:.2f}x {'ahead of' if pace >= 1 else 'behind'} pace)."
    )
    if not nc.calibrated:
        typer.secho(
            "Note: this model has no snapshot pairs to calibrate against, so the forecast assumes "
            "the current pace holds. Run `ytviews refresh` over time and retrain.",
            fg="yellow",
        )
    typer.echo("\nForecast:")
    for r in nc.forecast.itertuples():
        typer.echo(f"  day {r.days:>4g}: {_fmt(r.median):>7}  ({_fmt(r.low)} - {_fmt(r.high)})")


@app.command()
def when(
    title: TitleOpt,
    subscribers: SubsOpt = None,
    category: CategoryOpt = "24",
    duration: DurationOpt = None,
    days: DaysOpt = 7,
    recent_views: RecentOpt = None,
    timezone: Annotated[str, typer.Option(help="IANA timezone, e.g. America/New_York.")] = "UTC",
    top: int = 5,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Find the best weekday and hour to publish."""
    from .predictor import Predictor

    p = Predictor.load(model)
    video = _video_input(
        title, subscribers=subscribers, category=category, duration=duration, days=days,
        recent_views=recent_views,
    )  # fmt: skip
    grid = p.publish_times(video, timezone)
    best = grid.sort_values("median", ascending=False).head(top)
    worst = grid["median"].min()
    typer.echo(f"Best times to publish ({timezone}), predicted views after {days:g} days:")
    for r in best.itertuples():
        typer.echo(f"  {r.weekday:<9} {r.hour:02d}:00  {_fmt(r.median):>7}")
    spread = grid["median"].max() / worst
    typer.echo(f"Best vs worst slot: {spread:.2f}x. Small differences are within the noise.")


@app.command("compare-thumbnails")
def compare_thumbnails(
    thumbnails: Annotated[list[Path], typer.Argument(help="Thumbnail image files.")],
    title: TitleOpt,
    subscribers: SubsOpt = None,
    category: CategoryOpt = "24",
    days: DaysOpt = 7,
    recent_views: RecentOpt = None,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Rank alternative thumbnails for the same video."""
    from .predictor import Predictor

    missing = [str(t) for t in thumbnails if not t.exists()]
    if missing:
        raise typer.BadParameter(f"not found: {', '.join(missing)}")
    p = Predictor.load(model)
    base = _video_input(
        title, subscribers=subscribers, category=category, days=days, recent_views=recent_views
    )
    try:
        ranked = p.compare_thumbnails(base, [str(t) for t in thumbnails])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for i, r in enumerate(ranked.itertuples(), 1):
        typer.echo(f"{i}. {_fmt(r.median):>7}  ({r.vs_best:>4.0%} of best)  {r.thumbnail}")


@app.command()
def outliers(
    db: DbOpt = settings.db_path,
    channel_id: Annotated[str | None, typer.Option(help="Only this channel.")] = None,
    top: int = 10,
    model: ModelOpt = settings.default_model_path,
    csv: Annotated[Path | None, typer.Option(help="Also write the full table here.")] = None,
) -> None:
    """Which stored videos beat or missed what the model expected."""
    from .analysis import outliers as find_outliers
    from .predictor import Predictor

    p = Predictor.load(model)
    df = _load_training_frame(db, all_snapshots=False)
    try:
        table = find_outliers(p, df, channel_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if csv:
        table.to_csv(csv, index=False)

    def show(rows):
        for r in rows.itertuples():
            typer.echo(
                f"  {r.factor:>5.2f}x  {_fmt(r.views):>7} vs {_fmt(r.expected):>7} expected  "
                f"{r.title[:60]}"
            )

    typer.echo("Beat expectations the most:")
    show(table.head(top))
    typer.echo("\nFell furthest below expectations:")
    show(table.tail(top).iloc[::-1])
    if p.meta.get("trained_on") == "all data":
        typer.echo("\n(The model was trained on these videos, so expectations are in-sample.)")


@app.command()
def batch(
    input_csv: Annotated[Path, typer.Argument(help="CSV with a 'title' column and options.")],
    out: Annotated[Path, typer.Option(help="Where to write predictions.")] = Path(
        "predictions.csv"
    ),
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Predict views for every row of a CSV (columns = VideoInput fields)."""
    import pandas as pd

    from .analysis import predict_table
    from .predictor import Predictor

    p = Predictor.load(model)
    try:
        result = predict_table(p, pd.read_csv(input_csv, dtype={"category_id": str}))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    result.to_csv(out, index=False)
    typer.echo(f"Wrote {len(result)} predictions to {out}")


@app.command()
def doctor(
    db: DbOpt = settings.db_path,
    model: ModelOpt = settings.default_model_path,
    online: Annotated[
        bool, typer.Option(help="Also check the API key with one request (1 quota unit).")
    ] = False,
) -> None:
    """Check your setup: API key, optional libraries, data and model."""
    import importlib.util
    import os
    import platform

    from . import __version__

    problems = 0

    def line(ok: bool | None, text: str) -> None:
        nonlocal problems
        mark = {True: "ok  ", False: "FAIL", None: "note"}[ok]
        problems += ok is False
        typer.secho(f"[{mark}] {text}", fg={True: "green", False: "red", None: "yellow"}[ok])

    line(True, f"ytviews {__version__} on Python {platform.python_version()}")
    key_msg = (
        "YOUTUBE_API_KEY is set"
        if settings.api_key
        else "YOUTUBE_API_KEY not set: collecting real data needs it (see .env.example)"
    )
    line(bool(settings.api_key) or None, key_msg)
    if online and settings.api_key:
        try:
            _client(quota_budget=5).channels(["UCX6OQ3DkcsbYNE6H8uQQuVA"])
            line(True, "YouTube API key works")
        except Exception as exc:  # noqa: BLE001 - report whatever the API says
            line(False, f"YouTube API request failed: {exc}")
    extras = {
        "googleapiclient": ("youtube", "data collection"),
        "sentence_transformers": ("text", "pretrained text and CLIP thumbnail encoders"),
        "lightgbm": ("gbm", "LightGBM baseline and blend"),
        "matplotlib": ("plots", "report.png"),
        "streamlit": ("app", "web app"),
        "fastapi": ("api", "HTTP API"),
        "anthropic": ("llm", "Claude title ideas"),
    }
    for module, (extra, purpose) in extras.items():
        found = importlib.util.find_spec(module) is not None
        line(True if found else None,
             f"{module}: {'installed' if found else 'missing'} ({purpose}"
             + ("" if found else f"; pip install 'ytviews[{extra}]'") + ")")  # fmt: skip
    has_llm_key = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"))
    llm_msg = (
        "Anthropic credentials found"
        if has_llm_key
        else "no ANTHROPIC_API_KEY: `suggest --llm` will fall back to rule-based ideas"
    )
    line(True if has_llm_key else None, llm_msg)

    if db.exists():
        from .images import thumbnail_dir_for
        from .nowcast import snapshot_pairs
        from .store import VideoStore

        with VideoStore(db) as store:
            counts = store.counts()
            pairs = len(snapshot_pairs(store.load_dataset(all_snapshots=True))[0])
        line(counts["videos"] >= 300 or None,
             f"database {db}: {counts['videos']} videos, {counts['channels']} channels"
             + ("" if counts["videos"] >= 300 else " (aim for 300+ before training)"))  # fmt: skip
        line(pairs >= 20 or None,
             f"{pairs} videos with repeat snapshots"
             + ("" if pairs >= 20 else ": run `ytviews refresh` in a few weeks to calibrate "
                "`ytviews track`"))  # fmt: skip
        folder = thumbnail_dir_for(db)
        n_thumbs = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        line(n_thumbs > 0 or None,
             f"{n_thumbs} thumbnails downloaded"
             + ("" if n_thumbs else ": run `ytviews thumbnails`"))  # fmt: skip
    else:
        line(None, f"no database at {db} yet: run `ytviews collect` (or `ytviews demo`)")

    if (model / "meta.json").exists():
        meta = json.loads((model / "meta.json").read_text())
        synthetic = meta.get("dataset", {}).get("synthetic")
        line(not synthetic or None,
             f"model {model}: trained {meta['created_at'][:10]} on "
             f"{meta['dataset']['videos']} videos, text={meta.get('text_encoder')}, "
             f"thumbnails={meta.get('image_encoder')}"
             + (" (synthetic data!)" if synthetic else ""))  # fmt: skip
    else:
        line(None, f"no model at {model}: run `ytviews train`")
    if problems:
        raise typer.Exit(1)


@app.command()
def models(
    directory: Annotated[Path, typer.Argument(help="Folder holding model folders.")] = (
        settings.model_dir
    ),
) -> None:
    """Compare trained models side by side."""
    from .evaluate import OURS_NAME

    rows = []
    for meta_path in sorted(directory.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        ours = meta.get("metrics", {}).get(OURS_NAME) or meta.get("metrics", {}).get(
            "neural net (ours)", {}
        )
        gbm = meta.get("metrics", {}).get("lightgbm (all features)", {})
        rows.append(
            {
                "model": meta_path.parent.name,
                "created": meta.get("created_at", "")[:10],
                "videos": meta.get("dataset", {}).get("videos"),
                "data": "synthetic" if meta.get("dataset", {}).get("synthetic") else "real",
                "split": meta.get("split"),
                "text": meta.get("text_encoder"),
                "MAE": ours.get("mae_log"),
                "vs LightGBM": (ours.get("mae_log", np.nan) - gbm["mae_log"]) if gbm else None,
                "within 2x": ours.get("within_2x"),
                "coverage": ours.get("interval_coverage"),
            }
        )
    if not rows:
        typer.echo(f"No trained models in {directory}")
        return
    import pandas as pd

    table = pd.DataFrame(rows).sort_values("MAE")
    typer.echo(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    typer.echo("\nMAE is on the log scale (lower is better); compare models only on the same data.")


@app.command()
def experiments(
    directory: Annotated[Path, typer.Argument(help="Folder holding model folders.")] = (
        settings.model_dir
    ),
    last: int = 20,
) -> None:
    """Every training run so far (newest last), from the experiment log."""
    import pandas as pd

    from .pipeline import EXPERIMENTS_FILE

    path = directory / EXPERIMENTS_FILE
    if not path.exists():
        typer.echo(f"No experiment log at {path} yet; it is written by `ytviews train`.")
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    table = pd.DataFrame(rows).tail(last)
    table["created_at"] = table["created_at"].str[:16].str.replace("T", " ")
    table["data"] = np.where(table["synthetic"], "synthetic", "real")
    cols = ["created_at", "model_dir", "target", "data", "videos", "split", "text_encoder",
            "mae_log", "lightgbm_mae_log", "within_2x", "coverage"]  # fmt: skip
    typer.echo(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@app.command("backtest")
def backtest_cmd(
    db: DbOpt = settings.db_path,
    folds: int = 4,
    encoder: Annotated[str, typer.Option(help="auto | sbert | sbert-multi | tfidf")] = "auto",
    image_encoder: Annotated[
        str, typer.Option(help="auto | clip | stats | stats+faces | none")
    ] = "auto",
    ensemble: int = 2,
    config: Annotated[Path | None, typer.Option(help="TrainConfig JSON.")] = None,
    csv: Annotated[Path | None, typer.Option(help="Write per-fold results here.")] = None,
) -> None:
    """Evaluate at several points in time (rolling origin) instead of one split."""
    from dataclasses import replace

    from .backtest import backtest, summarise
    from .train import TrainConfig
    from .tune import load_config

    cfg = load_config(config) if config else TrainConfig()
    cfg = replace(cfg, ensemble=ensemble)
    results = backtest(
        _load_training_frame(db), folds=folds, text_encoder=encoder,
        image_encoder=image_encoder, config=cfg, log=typer.echo,
    )  # fmt: skip
    if results.empty:
        raise typer.BadParameter("not enough data for any fold")
    if csv:
        results.to_csv(csv, index=False)
    pivot = results.pivot(index="fold", columns="model", values="mae_log")
    typer.echo("\nMAE (log) per fold:")
    typer.echo(pivot.to_string(float_format=lambda v: f"{v:.3f}"))
    typer.echo("\nAcross folds:")
    typer.echo(summarise(results).to_string(float_format=lambda v: f"{v:.3f}"))


@app.command("monitor")
def monitor_cmd(
    db: DbOpt = settings.db_path,
    model: ModelOpt = settings.default_model_path,
    since: Annotated[
        str | None,
        typer.Option(
            help="Only videos published after this date (default: when the model was trained)."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """How the saved model holds up on videos published after it was trained."""
    from .monitor import monitor, summary_dict
    from .predictor import Predictor

    p = Predictor.load(model)
    result = monitor(p, _load_training_frame(db), since)
    if as_json:
        typer.echo(json.dumps(summary_dict(result), indent=2, default=str))
        return
    typer.echo(f"Videos published after {result.since:%Y-%m-%d}: {result.n_videos}")
    if result.live:
        live, ref = result.live, result.reference or {}

        def row(label: str, m: dict) -> str:
            cov = m.get("interval_coverage", float("nan"))
            return (
                f"  {label:<22} MAE {m['mae_log']:.3f}   within 2x: {m['within_2x']:.0%}   "
                f"p10-p90 coverage: {cov:.0%}"
            )

        typer.echo(row("live:", live))
        if ref:
            typer.echo(row("at training time:", ref))
        if len(result.weekly) > 1:
            typer.echo("\nBy publish week:")
            for r in result.weekly.itertuples():
                typer.echo(f"  {r.week}  n={r.n:<4} MAE {r.mae_log:.3f}")
    colour = "red" if result.verdict.startswith("degraded") else None
    typer.secho(f"\nVerdict: {result.verdict}", fg=colour, bold=True)


@app.command("channel-report")
def channel_report(
    channel_id: Annotated[str, typer.Option(help="Channel to report on (must be collected).")],
    out: Annotated[Path, typer.Option(help="HTML file to write.")] = Path("channel-report.html"),
    timezone: Annotated[str, typer.Option(help="IANA timezone for publish times.")] = "UTC",
    db: DbOpt = settings.db_path,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Write a shareable HTML report for one channel."""
    from .channel_report import write_channel_report
    from .predictor import Predictor

    p = Predictor.load(model)
    df = _load_training_frame(db, all_snapshots=False)
    try:
        path = write_channel_report(p, df, channel_id, out, timezone_name=timezone)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {path}")


@app.command()
def tune(
    db: DbOpt = settings.db_path,
    trials: int = 20,
    split: Annotated[str, typer.Option(help="time | channel | random")] = "time",
    encoder: Annotated[str, typer.Option(help="auto | sbert | sbert-multi | tfidf")] = "auto",
    image_encoder: Annotated[
        str, typer.Option(help="auto | clip | stats | stats+faces | none")
    ] = "auto",
    out: Annotated[Path, typer.Option(help="Where to write the best config.")] = Path(
        "best_config.json"
    ),
    seed: int = 0,
) -> None:
    """Search hyperparameters on an inner validation split (the test set is never used)."""
    from .tune import save_config
    from .tune import tune as run_tune

    df = _load_training_frame(db)
    best, table = run_tune(
        df, n_trials=trials, split=split, text_encoder=encoder, image_encoder=image_encoder,
        seed=seed, log=typer.echo,
    )  # fmt: skip
    save_config(best, out)
    table.to_csv(out.with_suffix(".trials.csv"), index=False)
    typer.echo(table.head(5).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    typer.echo(f"\nBest config written to {out}; use it with `ytviews train --config {out}`")


@app.command()
def serve(
    model: ModelOpt = settings.default_model_path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the HTTP API (docs at http://HOST:PORT/docs)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("install the API extra: pip install 'ytviews[api]'") from exc
    from .api import create_app

    uvicorn.run(create_app(model), host=host, port=port)


@app.command()
def report(
    model: ModelOpt = settings.default_model_path,
    html: Annotated[
        Path | None, typer.Option(help="Write a self-contained HTML report to this file.")
    ] = None,
) -> None:
    """Print the evaluation report saved with a model (or export it as a web page)."""
    if html is not None:
        from .report_html import write_html_report

        typer.echo(f"Wrote {write_html_report(model, html)}")
        return
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
    from .synthetic import write_synthetic_db
    from .train import TrainConfig

    db = settings.data_dir / "synthetic.db"
    if db.exists():
        db.unlink()
    typer.echo(f"Synthetic data: {write_synthetic_db(db, n_channels=channels)}")
    df = _load_training_frame(db)
    cfg = TrainConfig(ensemble=2)
    run_experiment(df, out, text_encoder=encoder, image_encoder="stats", config=cfg, log=typer.echo)
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
