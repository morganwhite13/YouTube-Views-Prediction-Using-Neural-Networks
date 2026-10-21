"""End-to-end experiment: split -> fit features -> train NN + baselines -> report -> save."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .baselines import default_baselines
from .evaluate import OURS_NAME, permutation_importance, regression_metrics, split_frame
from .features import (
    FeatureBuilder,
    add_channel_history,
    drop_channel_context,
    drop_thumbnails,
    log_views,
)
from .nowcast import fit_nowcast
from .predictor import Predictor
from .train import BlendedRegressor, NeuralRegressor, TrainConfig

NN_ONLY_NAME = "neural net only"
BLEND_WEIGHTS = np.round(np.linspace(0, 1, 11), 2)


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "videos": int(df["video_id"].nunique()),
        "channels": int(df["channel_id"].nunique()),
        "categories": int(df["category_id"].nunique()),
        "median_views": float(df["views"].median()),
        "published_range": [str(df["published_at"].min()), str(df["published_at"].max())],
        "synthetic": bool(df["channel_id"].astype(str).str.startswith("UCsynthetic").all()),
    }


def _fit_regressor(df, features, config, log, context_dropout):
    df = drop_channel_context(df, context_dropout, seed=config.seed)
    df = drop_thumbnails(df, seed=config.seed)
    inputs = features.transform(df)
    reg = NeuralRegressor(config, features.n_numeric, features.embed_dims, features.n_categories)
    augmented = df["_augmented"].eq(True).to_numpy() if "_augmented" in df else None
    reg.fit(
        inputs,
        log_views(df["views"].to_numpy()),
        groups=df["video_id"].to_numpy(),
        log=log,
        augmented=augmented,
    )
    return reg


def _fit_gbm(df, features, seed):
    from .baselines import LightGBMBaseline

    return (
        LightGBMBaseline(features, seed)
        .fit(df, features.transform(df), log_views(df["views"].to_numpy()))
        .model_
    )


def _choose_blend_weight(train, split, features, config, log, context_dropout) -> float:
    """Pick the network/LightGBM mix on an inner split of the training data."""
    itr, iva = split_frame(train, split, test_size=0.2, seed=config.seed)
    inner, val = train.iloc[itr].reset_index(drop=True), train.iloc[iva].reset_index(drop=True)
    small = TrainConfig(**{**config.to_dict(), "quantiles": config.quantiles, "ensemble": 2})
    nn = _fit_regressor(inner, features, small, None, context_dropout)
    gbm = _fit_gbm(inner, features, config.seed)
    x_val, y_val = features.transform(val), log_views(val["views"].to_numpy())
    p_nn, p_gbm = nn.predict(x_val), gbm.predict(features.flat(x_val, one_hot=False))
    errors = {w: float(np.abs(w * p_nn + (1 - w) * p_gbm - y_val).mean()) for w in BLEND_WEIGHTS}
    best = min(errors, key=errors.get)
    log(
        f"Blend weight for the network: {best:.1f} (inner-val MAE {errors[best]:.3f}; "
        f"network alone {errors[1.0]:.3f}, LightGBM alone {errors[0.0]:.3f})"
    )
    return float(best)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def run_experiment(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    split: str = "time",
    test_size: float = 0.2,
    text_encoder: str = "auto",
    image_encoder: str = "auto",
    config: TrainConfig | None = None,
    baselines: bool = True,
    refit: bool = True,
    blend: bool = True,
    context_dropout: float = 0.3,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train and evaluate; save the final model and a report into ``out_dir``."""
    config = config or TrainConfig()
    out_dir = Path(out_dir)
    t0 = time.time()
    df = add_channel_history(df).reset_index(drop=True)
    tr_idx, te_idx = split_frame(df, split, test_size, seed=config.seed)
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    log(f"Split '{split}': {len(train)} train rows / {len(test)} test rows")

    features = FeatureBuilder(text_encoder=text_encoder, image_encoder=image_encoder).fit(train)
    log(
        f"Text encoder: {features.encoder_kind} ({features.embed_dims['title']} dims); "
        f"thumbnails: {features.image_kind}"
    )
    y_train, y_test = log_views(train["views"].to_numpy()), log_views(test["views"].to_numpy())

    log("Training neural network ensemble...")
    nn = _fit_regressor(train, features, config, log, context_dropout)
    x_train, x_test = features.transform(train), features.transform(test)
    mid = nn.quantiles.index(0.5)
    reg: NeuralRegressor | BlendedRegressor = nn
    weight = 1.0
    if blend and _lightgbm_available():
        log("Choosing the network/LightGBM blend on an inner split...")
        weight = _choose_blend_weight(train, split, features, config, log, context_dropout)
        reg = BlendedRegressor(nn, _fit_gbm(train, features, config.seed), features, weight)
    q = reg.predict_quantiles(x_test)
    results = {OURS_NAME: regression_metrics(y_test, q[:, mid], (q[:, 0], q[:, -1]))}
    if isinstance(reg, BlendedRegressor):
        q_nn = nn.predict_quantiles(x_test)
        results[NN_ONLY_NAME] = regression_metrics(y_test, q_nn[:, mid], (q_nn[:, 0], q_nn[:, -1]))
    test_pred = pd.DataFrame(
        {
            "video_id": test["video_id"],
            "title": test["title"],
            "views": test["views"],
            "pred_low": np.expm1(q[:, 0]),
            "pred_median": np.expm1(q[:, mid]),
            "pred_high": np.expm1(q[:, -1]),
        }
    )

    if baselines:
        for model in default_baselines(features):
            log(f"Fitting baseline: {model.name}")
            model.fit(train, x_train, y_train)
            results[model.name] = regression_metrics(y_test, model.predict(test, x_test))

    log("Computing permutation importance...")
    importance = permutation_importance(reg.predict, x_test, y_test)
    nowcast = fit_nowcast(features, reg, test)  # held-out videos only
    log(
        f"Early-views forecast: calibrated on {nowcast['n_pairs']} held-out snapshot pairs"
        if nowcast["calibrated"]
        else "Early-views forecast: not enough snapshot pairs (run `ytviews refresh` later)"
    )

    final_reg, final_features, trained_on = reg, features, "train split"
    if refit:
        log("Refitting on all data for the saved model...")
        final_features = FeatureBuilder(text_encoder=text_encoder, image_encoder=image_encoder)
        final_features.fit(df)
        final_reg = _fit_regressor(df, final_features, config, log, context_dropout)
        if isinstance(reg, BlendedRegressor):
            gbm = _fit_gbm(df, final_features, config.seed)
            final_reg = BlendedRegressor(final_reg, gbm, final_features, weight)
        trained_on = "all data"

    meta = {
        "version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "split": split,
        "test_size": test_size,
        "text_encoder": features.encoder_kind,
        "image_encoder": features.image_kind,
        "trained_on": trained_on,
        "context_dropout": context_dropout,
        "blend_weight": weight,
        "train_config": config.to_dict(),
        "dataset": dataset_summary(df),
        "metrics": results,
        "importance": importance,
        "nowcast": nowcast,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Predictor(final_features, final_reg, meta).save(out_dir)
    test_pred.to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "report.md").write_text(render_report(meta))
    _maybe_plot(out_dir, test_pred, importance)
    return meta


def render_report(meta: dict[str, Any]) -> str:
    ds = meta["dataset"]
    lines = [
        "# ytviews training report",
        "",
        f"* Created: {meta['created_at']}",
        f"* Data: {ds['videos']} videos from {ds['channels']} channels ({ds['rows']} rows)",
        f"* Split: `{meta['split']}` (test fraction {meta['test_size']})",
        f"* Text encoder: `{meta['text_encoder']}`; thumbnails: `{meta.get('image_encoder')}`",
        f"* Saved model trained on: {meta['trained_on']}",
        "",
        "## Test-set metrics",
        "",
        "Errors are on the log scale. *Median factor error* 1.8 means the typical prediction is "
        "within 1.8x of the true view count.",
        "",
        "| model | MAE (log) | R² (log) | Spearman | median factor error | within 2x "
        "| within 10x | p10-p90 coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in sorted(meta["metrics"].items(), key=lambda kv: kv[1]["mae_log"]):
        cov = f"{m['interval_coverage']:.0%}" if "interval_coverage" in m else "-"
        lines.append(
            f"| {name} | {m['mae_log']:.3f} | {m['r2_log']:.3f} | {m['spearman']:.3f} | "
            f"{m['median_factor_error']:.2f}x | {m['within_2x']:.0%} | {m['within_10x']:.0%} "
            f"| {cov} |"
        )
    lines += [
        "",
        "## What the network relies on",
        "",
        "Permutation importance: increase in test MAE (log) when a feature group is shuffled.",
        "",
        "| feature group | MAE increase |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:+.3f} |" for k, v in meta["importance"].items()]
    return "\n".join(lines) + "\n"


def _maybe_plot(out_dir: Path, test_pred: pd.DataFrame, importance: dict[str, float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    yerr = np.vstack(
        [
            test_pred["pred_median"] - test_pred["pred_low"],
            test_pred["pred_high"] - test_pred["pred_median"],
        ]
    ).clip(min=0)
    ax.errorbar(
        test_pred["views"],
        test_pred["pred_median"],
        yerr=yerr,
        fmt="o",
        ms=3,
        alpha=0.35,
        elinewidth=0.6,
        color="#2f6db5",
    )
    lo = max(1, min(test_pred["views"].min(), test_pred["pred_median"].min()))
    hi = max(test_pred["views"].max(), test_pred["pred_median"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="actual views",
        ylabel="predicted views (p10-p90)",
        title="Test set: predicted vs actual",
    )
    ax = axes[1]
    names = list(importance)[::-1]
    ax.barh(names, [importance[n] for n in names], color="#2f6db5")
    ax.set(xlabel="MAE increase when shuffled (log views)", title="Feature-group importance")
    fig.tight_layout()
    fig.savefig(out_dir / "report.png", dpi=120)
    plt.close(fig)


def load_report(model_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "meta.json").read_text())
