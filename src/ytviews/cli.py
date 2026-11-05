"""Command-line interface: ``ytviews --help``."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
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


def _check_timezone(name: str) -> None:
    """A clean usage error for a name zoneinfo can't use (unknown, empty or a path)."""
    from zoneinfo import ZoneInfo

    try:
        ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001 - ZoneInfoNotFoundError or ValueError
        raise typer.BadParameter(f"unknown timezone {name!r}; use e.g. America/New_York") from exc


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
    ensemble: Annotated[int | None, typer.Option(help="Networks in the ensemble [5].")] = None,
    epochs: Annotated[int | None, typer.Option(help="Maximum epochs [300].")] = None,
    hidden: Annotated[int | None, typer.Option(help="Hidden width [64].")] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed [42].")] = None,
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
    cfg = TrainConfig()
    if config is not None:
        from .tune import load_config

        cfg = load_config(config)
        tuned = json.loads(config.read_text()).get("tuned_with", {})
        if tuned and (tuned.get("split") != split or tuned.get("test_size") != test_size):
            typer.secho(
                f"Warning: {config} was tuned with split={tuned.get('split')}, test_size="
                f"{tuned.get('test_size')}; training with split={split}, test_size={test_size} "
                "may test on rows the tuning was scored on.",
                fg="yellow",
            )
    # flags given on the command line win over the config file; unset ones keep its values
    overrides = {"ensemble": ensemble, "epochs": epochs, "hidden": hidden, "seed": seed}
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
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
    from .collect import keep_video

    ch = ch[0]
    uploads = (
        client.upload_ids(ch["uploads_playlist"], n_recent + 5)
        if ch.get("uploads_playlist")
        else []
    )
    # Skip premieres, live streams and hidden view counts: they would poison the history.
    recent = [v for v in client.videos(uploads) if keep_video(v, language=None)][:n_recent]
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
    for note in pred.notes:
        typer.secho(f"  Note: {note}.", fg="yellow")
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


def _stored_video(db: Path, video_id: str):
    """(VideoInput, views, hours) from a stored video's latest snapshot."""
    from .channel_report import _video_from_row

    df = _load_training_frame(db)
    rows = df[df["video_id"] == video_id].sort_values("fetched_at")
    if rows.empty:
        raise typer.BadParameter(f"video {video_id} is not in {db}")
    row = rows.iloc[-1]
    channel = df[df["channel_id"] == row["channel_id"]].sort_values("fetched_at")
    earlier = channel.drop_duplicates("video_id", keep="last")
    earlier = earlier[earlier["published_at"] < row["published_at"]]
    hours = (row["fetched_at"] - row["published_at"]).total_seconds() / 3600
    return _video_from_row(row, earlier), float(row["views"]), hours


@app.command()
def track(
    title: Annotated[str | None, typer.Option(help="Video title.")] = None,
    views: Annotated[float | None, typer.Option(help="Views the video has right now.")] = None,
    hours: Annotated[float | None, typer.Option(help="Hours since it was published.")] = None,
    video_id: Annotated[
        str | None,
        typer.Option(help="A stored video: use its latest snapshot instead of the options."),
    ] = None,
    db: DbOpt = settings.db_path,
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
    if p.target != "views":
        raise typer.BadParameter(f"track needs a views model; this model predicts {p.target}")
    if video_id:
        video, views, hours = _stored_video(db, video_id)
        title = video.title
        typer.echo(f'"{title}": {_fmt(views)} views after {hours / 24:.1f} days (stored)')
    elif title is None or views is None or hours is None:
        raise typer.BadParameter("give --video-id, or all of --title, --views and --hours")
    else:
        video = _video_input(
            title, subscribers=subscribers, description=description, category=category,
            duration=duration, channel_id=channel_id, recent_views=recent_views,
            thumbnail=thumbnail,
        )  # fmt: skip
    if hours <= 0:
        raise typer.BadParameter("the video must have been published before it was fetched")
    nc = p.track(video, views, hours / 24)
    pace = nc.vs_expected
    typer.echo(
        f"After {hours:.3g} hours the model expected ~{_fmt(nc.expected_views)} views; "
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

    _check_timezone(timezone)
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
def grid(
    title: Annotated[list[str], typer.Option("--title", "-t", help="A candidate title (repeat).")],
    thumbnail: Annotated[
        list[Path], typer.Option("--thumbnail", help="A candidate thumbnail image (repeat).")
    ],
    subscribers: SubsOpt = None,
    description: DescOpt = "",
    category: CategoryOpt = "24",
    duration: DurationOpt = None,
    days: DaysOpt = 7,
    channel_id: ChannelOpt = None,
    recent_views: RecentOpt = None,
    top: Annotated[int, typer.Option(help="How many pairs to list.")] = 5,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Score every title x thumbnail combination and show the best pairs."""
    from .predictor import Predictor

    missing = [str(t) for t in thumbnail if not t.exists()]
    if missing:
        raise typer.BadParameter(f"not found: {', '.join(missing)}")
    p = Predictor.load(model)
    base = _video_input(
        title[0], subscribers=subscribers, description=description, category=category,
        duration=duration, days=days, channel_id=channel_id, recent_views=recent_views,
    )  # fmt: skip
    try:
        ranked = p.compare_grid(base, list(title), [str(t) for t in thumbnail])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    names = {str(t): f"T{i}" for i, t in enumerate(thumbnail, 1)}
    table = ranked.pivot(index="title", columns="thumbnail", values="median")
    table = table.reindex(index=list(dict.fromkeys(title)), columns=list(names))
    width = max(len(t) for t in table.index)
    typer.echo(" " * width + "".join(f"{names[c]:>9}" for c in table.columns))
    for t, row in table.iterrows():
        typer.echo(f"{t:<{width}}" + "".join(f"{_fmt(v):>9}" for v in row))
    typer.echo("")
    for name, path in names.items():
        typer.echo(f"{path} = {name}")
    typer.echo("\nBest pairs:")
    for i, r in enumerate(ranked.head(top).itertuples(), 1):
        note = ""
        if r.interaction >= 1.1:
            note = f"  (works {r.interaction:.2f}x better together than apart)"
        elif r.interaction <= 0.9:
            note = f"  (clash: {r.interaction:.2f}x of what each suggests)"
        typer.echo(f"{i}. {_fmt(r.median):>7}  {names[r.thumbnail]}  {r.title}{note}")


@app.command()
def plan(
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
    alt_title: Annotated[
        list[str] | None, typer.Option(help="Your own alternative title (repeatable).")
    ] = None,
    alt_thumbnail: Annotated[
        list[Path] | None, typer.Option(help="Alternative thumbnail file (repeatable).")
    ] = None,
    timezone: Annotated[str, typer.Option(help="IANA timezone, e.g. America/New_York.")] = "UTC",
    llm: Annotated[
        bool, typer.Option(help="Also ask Claude for title ideas (needs ANTHROPIC_API_KEY).")
    ] = False,
    html_out: Annotated[
        Path | None, typer.Option("--html", help="Also write the plan as an HTML page.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    record: Annotated[
        bool,
        typer.Option(help="Save the plan, to compare with the real views later (`plans`)."),
    ] = False,
    db: DbOpt = settings.db_path,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Plan a video before publishing: best title, thumbnail and time, in one pass."""
    from .planner import make_plan, plan_html
    from .predictor import Predictor

    missing = [str(t) for t in alt_thumbnail or [] if not t.exists()]
    if missing:
        raise typer.BadParameter(f"not found: {', '.join(missing)}")
    _check_timezone(timezone)
    p = Predictor.load(model)
    video = _video_input(
        title, subscribers=subscribers, description=description, tags=tags, category=category,
        duration=duration, days=days, channel_id=channel_id, recent_views=recent_views,
        thumbnail=thumbnail,
    )  # fmt: skip
    result = make_plan(
        p, video, extra_titles=alt_title, thumbnails=[str(t) for t in alt_thumbnail or []],
        timezone_name=timezone, use_llm=llm,
    )  # fmt: skip
    if html_out:
        html_out.write_text(plan_html(result))
    if record:
        from .planner import record_plan
        from .store import VideoStore

        with VideoStore(db) as store:
            plan_id = record_plan(store, result, predictor=p, channel_id=channel_id,
                                  model=str(model))  # fmt: skip
        if not as_json:
            typer.echo(f"Recorded as plan {plan_id}; `ytviews plans --review` once it is out.")
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, default=str))
        return
    d, b = result.draft_prediction, result.best_prediction
    typer.echo(f"Draft:  ~{_fmt(d.median)} {p.target} after {days:g} days")
    typer.secho(
        f"Best:   ~{_fmt(b.median)} {p.target}  ({result.uplift:.2f}x the draft)", bold=True
    )
    typer.echo("\nWhat to change:")
    for c in result.changes():
        typer.echo(f"  - {c}")
    levers = result.uplift_by_lever()
    if len(levers):
        typer.echo("\nWhere the uplift comes from (share, and what skipping it alone costs):")
        for r in levers.itertuples():
            skip = "" if r.skip_cost != r.skip_cost else f"   skip it: -{1 - 1 / r.skip_cost:.0%}"
            typer.echo(f"  {r.lever:<13} x{r.factor:.2f}{skip}")
    typer.echo("\nWhat drives the draft's number (x an average video of the same age):")
    for r in result.explanation.effects.head(5).itertuples():
        typer.echo(f"  {r.group:<16} x{r.factor:.2f}")
    if html_out:
        typer.echo(f"\nWrote {html_out}")


@app.command("schedule")
def schedule_cmd(
    csv: Annotated[Path, typer.Argument(help="Planned videos; columns = VideoInput fields.")],
    timezone: Annotated[str, typer.Option(help="IANA timezone, e.g. America/New_York.")] = "UTC",
    days: Annotated[int, typer.Option(help="Spread over at least this many days.")] = 7,
    out: Annotated[Path | None, typer.Option(help="Also write the schedule as CSV.")] = None,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Spread planned videos over the coming days (one per day) for the most total views."""
    import pandas as pd

    from .analysis import videos_from_frame
    from .planner import schedule
    from .predictor import Predictor

    _check_timezone(timezone)
    p = Predictor.load(model)
    try:
        videos = videos_from_frame(pd.read_csv(csv))
        plan = schedule(p, videos, timezone_name=timezone, days=days)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for r in plan.itertuples():
        typer.echo(
            f"{r.published_at:%a %d %b %H:%M}  ~{_fmt(r.median):>7}  "
            f"({r.vs_own_best:>4.0%} of its own best slot)  {r.title[:60]}"
        )
    typer.echo(f"Total: ~{_fmt(plan['median'].sum())} {p.target} ({timezone})")
    if out:
        plan.to_csv(out, index=False)
        typer.echo(f"Wrote {out}")


@app.command("calendar")
def calendar_cmd(
    csv: Annotated[Path, typer.Argument(help="Planned videos; columns = VideoInput fields.")],
    timezone: Annotated[str, typer.Option(help="IANA timezone, e.g. America/New_York.")] = "UTC",
    days: Annotated[int, typer.Option(help="Plan over this many days.")] = 14,
    min_gap: Annotated[int, typer.Option(help="At least this many days between uploads.")] = 1,
    last_upload: Annotated[
        str | None,
        typer.Option(help="When the channel last published (ISO date/time), for the first gap."),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Also write the calendar as CSV.")] = None,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Plan the order, spacing and hour of several uploads for the most total views."""
    import pandas as pd

    from .analysis import videos_from_frame
    from .planner import calendar
    from .predictor import Predictor

    _check_timezone(timezone)
    last = None
    if last_upload:
        try:
            last = datetime.fromisoformat(last_upload)
        except ValueError as exc:
            raise typer.BadParameter(f"--last-upload: {exc}") from exc
    p = Predictor.load(model)
    try:
        videos = videos_from_frame(pd.read_csv(csv))
        plan = calendar(p, videos, timezone_name=timezone, days=days, min_gap_days=min_gap,
                        last_upload=last)  # fmt: skip
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for r in plan.itertuples():
        gap = "" if pd.isna(r.gap_days) else f"{r.gap_days:4.1f}d gap  "
        typer.echo(f"{r.published_at:%a %d %b %H:%M}  ~{_fmt(r.median):>7}  {gap}{r.title[:60]}")
    typer.echo(
        f"Total: ~{_fmt(plan['median'].sum())} {p.target} after {videos[0].age_days:g} days "
        f"each ({timezone})"
    )
    if out:
        plan.to_csv(out, index=False)
        typer.echo(f"Wrote {out}")


@app.command()
def pulse(
    channel_id: Annotated[str, typer.Option(help="Channel whose latest uploads to check.")],
    n: Annotated[int, typer.Option(help="How many recent uploads to show.")] = 5,
    horizon: Annotated[float, typer.Option(help="Forecast views at this age (days).")] = 30,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Live check of a channel's newest uploads: pace so far and where each is heading."""
    from .channel_report import pulse as do_pulse
    from .predictor import Predictor

    p = Predictor.load(model)
    if p.target != "views":
        raise typer.BadParameter(f"pulse needs a views model; this model predicts {p.target}")
    client = _client(quota_budget=50)
    try:
        live = do_pulse(p, client, channel_id, n=n, horizon_days=horizon)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if live.empty:
        typer.echo(f"No uploads younger than {horizon:g} days.")
        return
    typer.echo(f"{'age':>6}  {'views':>7}  {'pace':>6}  {f'day {horizon:g}':>8}  title")
    for r in live.itertuples():
        typer.echo(
            f"{r.age_days:>5.1f}d  {_fmt(r.views):>7}  {r.pace:>5.2f}x  {_fmt(r.median):>8}  "
            f"{r.title[:60]}"
        )
    if not live["calibrated"].all():
        typer.secho("Uncalibrated: forecasts assume each video keeps its current pace.",
                    fg="yellow")  # fmt: skip
    typer.echo(f"({client.quota_used} quota units)")


@app.command("thumbnail-tips")
def thumbnail_tips_cmd(
    thumbnail: Annotated[Path, typer.Argument(help="Your thumbnail image.")],
    title: TitleOpt,
    subscribers: SubsOpt = None,
    category: CategoryOpt = "24",
    days: DaysOpt = 7,
    recent_views: RecentOpt = None,
    save: Annotated[
        Path | None, typer.Option(help="Folder to keep the edited versions in.")
    ] = None,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Try simple edits of a thumbnail (brighter, more contrast, ...) and rank them."""
    import tempfile

    from .planner import thumbnail_tips
    from .predictor import Predictor

    if not thumbnail.exists():
        raise typer.BadParameter(f"not found: {thumbnail}")
    p = Predictor.load(model)
    video = _video_input(
        title, subscribers=subscribers, category=category, days=days, recent_views=recent_views
    )
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ranked = thumbnail_tips(p, video, thumbnail, save or tmp)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    for r in ranked.itertuples():
        mark = "  <- yours" if r.edit == "original" else ""
        typer.echo(f"  {r.vs_original:>5.0%}  {_fmt(r.median):>7}  {r.edit}{mark}")
    best = ranked.iloc[0]
    if best["edit"] != "original" and best["vs_original"] > 1.02:
        where = f" (saved in {save})" if save else " (use --save DIR to keep the edits)"
        typer.echo(f"Best: {best['edit']}, about {best['vs_original'] - 1:.0%} more{where}.")
    else:
        typer.echo("None of these simple edits beats your thumbnail by more than 2%.")
    typer.echo("This is what the model associates with more views, not a guarantee.")


@app.command("plans")
def plans_cmd(
    review: Annotated[
        bool, typer.Option(help="Compare published plans with the views they really got.")
    ] = False,
    link: Annotated[
        tuple[int, str] | None,
        typer.Option(help="Link a plan to a video by hand: --link PLAN_ID VIDEO_ID."),
    ] = None,
    db: DbOpt = settings.db_path,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """Recorded plans (`plan --record`) and, with --review, how they turned out."""
    import pandas as pd

    from .store import VideoStore

    with VideoStore(db) as store:
        if link:
            try:
                store.link_plan(link[0], link[1])
            except (KeyError, ValueError) as exc:
                raise typer.BadParameter(str(exc)) from exc
            typer.echo(f"Linked plan {link[0]} to {link[1]}.")
        plans = store.plans()
        if not review:
            for r in plans.itertuples():
                status = "not linked" if pd.isna(r.video_id) else f"-> {r.video_id}"
                typer.echo(f"{r.plan_id:>4}  {r.created_at[:10]}  ~{_fmt(r.predicted_median):>7}"
                           f"  {r.draft_title[:50]}  ({status})")  # fmt: skip
            return
    if plans.empty:
        typer.echo("No recorded plans yet: use `ytviews plan ... --record`.")
        return
    table = _review_plans(db, model)
    for r in table.itertuples():
        if pd.isna(r.predicted):
            typer.echo(f"{r.plan_id:>4}  {r.title[:50]}  ({r.status})")
            continue
        followed = "" if r.followed else ", published with a different title"
        typer.echo(
            f"{r.plan_id:>4}  {r.title[:50]}  day {r.age_days:.0f}: {_fmt(r.views)} views, "
            f"predicted {_fmt(r.predicted)} ({_fmt(r.low)}-{_fmt(r.high)}, {r.source}), "
            f"{r.vs_predicted:.2f}x, {r.status}{followed}"
        )
    _echo_review_summary(table)


def _review_plans(db: Path, model: Path):
    """Link and review recorded plans; the model is only needed for plans recorded without
    their trajectory (and is skipped if there is none)."""
    from .planner import review_plans
    from .predictor import Predictor
    from .store import VideoStore

    try:
        predictor = Predictor.load(model)
    except FileNotFoundError:
        predictor = None
    try:
        with VideoStore(db) as store:
            return review_plans(predictor, store, _load_training_frame(db, all_snapshots=False))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _echo_review_summary(table) -> None:
    from .planner import summarise_review

    s = summarise_review(table)
    typer.echo(f"\n{s['plans']} plans, {s['reviewed']} published and collected.")
    if s["reviewed"]:
        typer.echo(
            f"{s['within_range']:.0%} within their predicted range; typical outcome "
            f"{s['typical_ratio']:.2f}x the prediction."
        )


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
    ensemble: Annotated[
        int | None,
        typer.Option(help="Networks per fold [2, or the config's when --config is given]."),
    ] = None,
    config: Annotated[Path | None, typer.Option(help="TrainConfig JSON.")] = None,
    csv: Annotated[Path | None, typer.Option(help="Write per-fold results here.")] = None,
) -> None:
    """Evaluate at several points in time (rolling origin) instead of one split."""
    from dataclasses import replace

    from .backtest import backtest, summarise
    from .train import TrainConfig
    from .tune import load_config

    cfg = load_config(config) if config else TrainConfig(ensemble=2)
    if ensemble is not None:  # only when given: don't override a tuned config's ensemble
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


@app.command("compare-models")
def compare_models_cmd(
    model_a: Annotated[Path, typer.Argument(help="First model folder.")],
    model_b: Annotated[Path, typer.Argument(help="Second model folder.")],
    db: DbOpt = settings.db_path,
    since: Annotated[
        str | None,
        typer.Option(
            help="Score videos published after this date (default: after both were trained, "
            "so neither has seen them)."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Which of two models predicts better, on the same unseen videos (paired bootstrap)."""
    from .compare import compare_models
    from .predictor import Predictor

    a, b = Predictor.load(model_a), Predictor.load(model_b)
    try:
        r = compare_models(a, b, _load_training_frame(db), since, (str(model_a), str(model_b)))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        typer.echo(json.dumps(r.as_dict(), indent=2, default=str))
        return
    typer.echo(f"Videos scored: {r.n}")
    if r.n:
        typer.echo(f"  {r.name_a:<30} MAE {r.mae_a:.3f}")
        typer.echo(f"  {r.name_b:<30} MAE {r.mae_b:.3f}")
        typer.echo(
            f"  difference (B - A): {r.diff:+.3f}, 95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
        )
    if r.note:
        typer.secho(f"Note: {r.note}", fg="yellow")
    typer.secho(f"\nVerdict: {r.verdict}", bold=True)


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

    _check_timezone(timezone)
    p = Predictor.load(model)
    df = _load_training_frame(db, all_snapshots=False)
    try:
        path = write_channel_report(p, df, channel_id, out, timezone_name=timezone)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {path}")


@app.command()
def digest(
    channel_id: Annotated[str, typer.Option(help="Channel to summarise (must be collected).")],
    days: Annotated[
        float, typer.Option(help="Length of the period, ending at the newest data.")
    ] = 7,
    out: Annotated[
        Path | None, typer.Option(help="Markdown file to write (default: print it).")
    ] = None,
    timezone: Annotated[str, typer.Option(help="IANA timezone for publish times.")] = "UTC",
    db: DbOpt = settings.db_path,
    model: ModelOpt = settings.default_model_path,
) -> None:
    """A one-minute weekly summary of a channel: growth, new uploads, surprises, what to try."""
    from .digest import weekly_digest
    from .predictor import Predictor

    _check_timezone(timezone)
    p = Predictor.load(model)
    try:
        text = weekly_digest(p, _load_training_frame(db), channel_id, days=days,
                             timezone_name=timezone)  # fmt: skip
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if out is None:
        typer.echo(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        typer.echo(f"Wrote {out}")


@app.command()
def tune(
    db: DbOpt = settings.db_path,
    trials: int = 20,
    split: Annotated[str, typer.Option(help="time | channel | random")] = "time",
    test_size: Annotated[
        float, typer.Option(help="The test share `train` will use; tuning never touches it.")
    ] = 0.2,
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
        df, n_trials=trials, split=split, test_size=test_size, text_encoder=encoder,
        image_encoder=image_encoder, seed=seed, log=typer.echo,
    )  # fmt: skip
    save_config(best, out, split=split, test_size=test_size)
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
def update(
    db: DbOpt = settings.db_path,
    model: ModelOpt = settings.default_model_path,
    fetch: Annotated[
        bool, typer.Option(help="Refresh stats via the API (needs YOUTUBE_API_KEY).")
    ] = True,
    enrich: Annotated[
        bool, typer.Option(help="Detect Shorts and download thumbnails (asks youtube.com).")
    ] = True,
    retrain: Annotated[
        str, typer.Option(help="auto (if missing or degraded) | always | never")
    ] = "auto",
    channel: Annotated[
        list[str] | None, typer.Option(help="Also write a channel report (repeatable).")
    ] = None,
    reports: Annotated[Path, typer.Option(help="Folder for the HTML reports.")] = Path("reports"),
    timezone: Annotated[str, typer.Option(help="IANA timezone for channel reports.")] = "UTC",
    quota_budget: int = 9000,
    keep_better: Annotated[
        bool,
        typer.Option(
            help="Replace the model only if the retrained one isn't worse on videos neither "
            "has seen."
        ),
    ] = True,
) -> None:
    """Routine upkeep in one go: refresh, enrich, check the model, retrain if needed, report."""
    _check_timezone(timezone)
    from .monitor import monitor
    from .predictor import Predictor
    from .store import VideoStore

    if retrain not in ("auto", "always", "never"):
        raise typer.BadParameter("--retrain must be auto, always or never")

    def step(text: str) -> None:
        typer.secho(f"\n== {text}", bold=True)

    step("Refreshing stats")
    if not fetch:
        typer.echo("skipped (--no-fetch)")
    elif not settings.api_key:
        typer.secho("skipped: no YOUTUBE_API_KEY set", fg="yellow")
    else:
        from .collect import refresh as do_refresh
        from .youtube import QuotaExceeded

        client = _client(quota_budget)
        with VideoStore(db) as store:
            try:
                n = do_refresh(client, store)
                typer.echo(f"refreshed {n} videos ({client.quota_used} quota units)")
            except QuotaExceeded as exc:
                typer.secho(f"stopped early: {exc} Kept what was fetched.", fg="yellow")

    step("Shorts and thumbnails (no quota)")
    from .collect import detect_shorts, download_thumbnails
    from .images import thumbnail_dir_for

    with VideoStore(db) as store:
        try:
            if not enrich:
                raise OSError("--no-enrich")
            counts = detect_shorts(store)
            typer.echo(f"Shorts check: {counts}")
            ok, failed = download_thumbnails(store, thumbnail_dir_for(db))
            typer.echo(f"thumbnails: {ok} new, {failed} failed")
        except OSError as exc:  # offline: the model falls back gracefully without these
            typer.secho(f"skipped: {exc}", fg="yellow")

    step("Checking the model")
    df_all = _load_training_frame(db)
    existing = None
    reason = "--retrain always" if retrain == "always" else None
    try:
        existing = Predictor.load(model)
    except FileNotFoundError:
        reason = reason or f"no model at {model}"
    if existing is not None:
        if existing.target == "views":
            verdict = monitor(existing, df_all).verdict
            typer.echo(f"monitor: {verdict}")
            if verdict.startswith("degraded"):
                reason = reason or verdict
        else:
            typer.echo(f"monitor skipped: the model predicts {existing.target}")
    if retrain == "never":
        reason = None

    if reason:
        step(f"Retraining ({reason})")
        if len(df_all) < 30:
            raise typer.BadParameter(f"only {len(df_all)} rows in {db}; collect more data first")
        from .pipeline import run_experiment
        from .train import TrainConfig

        old = existing.meta if existing is not None else {}
        cfg = TrainConfig()
        if old.get("train_config"):
            from .tune import config_from_dict

            cfg = config_from_dict(old["train_config"])
        # With a model to replace, train next to it and swap only if the new one isn't worse.
        gated = existing is not None and existing.target == "views" and keep_better
        target_dir = model.with_name(f".{model.name}-candidate") if gated else model
        if target_dir != model and target_dir.exists():
            shutil.rmtree(target_dir)
        meta = run_experiment(
            df_all, target_dir, split=old.get("split", "time"),
            test_size=old.get("test_size", 0.2),
            text_encoder=old.get("text_encoder", "auto"),
            image_encoder=old.get("image_encoder", "auto"),
            config=cfg, target=old.get("target", "views"), log=typer.echo,
        )  # fmt: skip
        if gated:
            from .compare import gate_candidate, should_promote

            check = gate_candidate(existing, target_dir, df_all)
            if check.n:
                typer.echo(
                    f"on {check.n} videos neither has seen: current MAE {check.mae_a:.3f}, "
                    f"candidate {check.mae_b:.3f} ({check.verdict})"
                )
            else:
                typer.echo("no held-out videos the current model hasn't seen; can't compare")
            if should_promote(check):
                shutil.rmtree(model)
                target_dir.rename(model)
                typer.echo(f"saved to {model} ({meta['elapsed_s']}s)")
            else:
                shutil.rmtree(target_dir)
                typer.secho(
                    f"kept the current model: the retrained one did worse ({check.verdict})",
                    fg="yellow",
                )
        else:
            typer.echo(f"saved to {model} ({meta['elapsed_s']}s)")
    else:
        typer.echo("no retraining needed")

    step("Writing reports")
    from .channel_report import write_channel_report
    from .report_html import write_html_report

    reports.mkdir(parents=True, exist_ok=True)
    if (model / "meta.json").exists():
        typer.echo(f"wrote {write_html_report(model, reports / 'model-report.html')}")
    else:  # --retrain never with no model yet
        typer.secho(f"no model at {model}, so no model report; run `ytviews train`", fg="yellow")
    if channel and (model / "meta.json").exists():
        p = Predictor.load(model)
        latest = _load_training_frame(db, all_snapshots=False)
        from .digest import weekly_digest

        everything = _load_training_frame(db)
        for ch in channel:
            try:
                out = write_channel_report(
                    p, latest, ch, reports / f"channel-{ch}.html", timezone_name=timezone
                )
                typer.echo(f"wrote {out}")
                if p.target == "views":
                    text = weekly_digest(p, everything, ch, timezone_name=timezone)
                    (reports / f"digest-{ch}.md").write_text(text)
                    typer.echo(f"wrote {reports / f'digest-{ch}.md'}")
            except ValueError as exc:
                typer.secho(f"{ch}: {exc}", fg="yellow")

    with VideoStore(db) as store:
        has_plans = not store.plans().empty
    if has_plans:
        step("Recorded plans")
        _echo_review_summary(_review_plans(db, model))
        typer.echo("Details: `ytviews plans --review`.")


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
