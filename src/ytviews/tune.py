"""Hyperparameter search for the neural network.

Random search, scored on an *inner* validation split carved out of the training portion: the
outer test set that ``ytviews train`` reports on is never looked at, so tuning cannot leak
into the headline metrics. Trial 0 is always the current defaults, so the result can only
improve on them (on the inner split).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import split_frame
from .features import (
    FeatureBuilder,
    add_channel_history,
    drop_channel_context,
    drop_thumbnails,
    log_views,
)
from .train import NeuralRegressor, TrainConfig

SEARCH_SPACE: dict[str, Callable[[np.random.Generator], Any]] = {
    "lr": lambda r: float(10 ** r.uniform(-3.7, -2.5)),
    "weight_decay": lambda r: float(10 ** r.uniform(-3.5, -1.0)),
    "hidden": lambda r: int(r.choice([32, 64, 96, 128])),
    "dropout": lambda r: float(r.uniform(0.05, 0.4)),
    "embed_dropout": lambda r: float(r.uniform(0.2, 0.7)),
    "augmented_weight": lambda r: float(r.uniform(0.1, 0.6)),
    "batch_size": lambda r: int(r.choice([64, 128, 256])),
}


def sample_config(base: TrainConfig, rng: np.random.Generator) -> TrainConfig:
    return replace(base, **{k: f(rng) for k, f in SEARCH_SPACE.items()})


def tune(
    df: pd.DataFrame,
    *,
    n_trials: int = 20,
    split: str = "time",
    text_encoder: str = "auto",
    image_encoder: str = "auto",
    base: TrainConfig | None = None,
    ensemble: int = 2,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[TrainConfig, pd.DataFrame]:
    base = replace(base or TrainConfig(), ensemble=ensemble)
    df = add_channel_history(df).reset_index(drop=True)
    outer_train, _ = split_frame(df, split, seed=base.seed)  # outer test set is never used
    train = df.iloc[outer_train].reset_index(drop=True)
    itr, iva = split_frame(train, split, test_size=0.2, seed=base.seed)
    inner, val = train.iloc[itr].reset_index(drop=True), train.iloc[iva].reset_index(drop=True)
    log(f"Tuning on {len(inner)} rows, scoring on {len(val)} inner-validation rows")

    features = FeatureBuilder(text_encoder=text_encoder, image_encoder=image_encoder).fit(inner)
    aug = drop_thumbnails(drop_channel_context(inner, 0.3, seed=base.seed), seed=base.seed)
    x_aug, y_aug = features.transform(aug), log_views(aug["views"].to_numpy())
    augmented = aug["_augmented"].eq(True).to_numpy() if "_augmented" in aug else None
    x_val, y_val = features.transform(val), log_views(val["views"].to_numpy())

    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        cfg = base if trial == 0 else sample_config(base, rng)
        reg = NeuralRegressor(cfg, features.n_numeric, features.embed_dims, features.n_categories)
        reg.fit(x_aug, y_aug, groups=aug["video_id"].to_numpy(), augmented=augmented)
        q = reg.predict_quantiles(x_val)
        mae = float(np.abs(q[:, reg.quantiles.index(0.5)] - y_val).mean())
        coverage = float(((y_val >= q[:, 0]) & (y_val <= q[:, -1])).mean())
        rows.append({"trial": trial, "val_mae": mae, "val_coverage": coverage}
                    | {k: getattr(cfg, k) for k in SEARCH_SPACE})  # fmt: skip
        best = min(r["val_mae"] for r in rows)
        log(f"  trial {trial:>2}: val MAE {mae:.3f}  (best {best:.3f})")

    trials = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    best_params = {k: trials.iloc[0][k] for k in SEARCH_SPACE}
    best_cfg = replace(
        base,
        **{k: type(getattr(base, k))(v) for k, v in best_params.items()},
        ensemble=TrainConfig().ensemble,
    )
    return best_cfg, trials


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2))


def load_config(path: str | Path) -> TrainConfig:
    data = json.loads(Path(path).read_text())
    known = {f.name for f in fields(TrainConfig)}
    data = {k: v for k, v in data.items() if k in known}
    if "quantiles" in data:
        data["quantiles"] = tuple(data["quantiles"])
    return TrainConfig(**data)
