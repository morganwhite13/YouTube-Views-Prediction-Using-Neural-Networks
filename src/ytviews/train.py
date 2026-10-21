"""Training loop for :class:`~ytviews.model.ViewsNet`: ensembles, early stopping, calibration."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from .model import ViewsNet, pinball_loss


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 25
    val_fraction: float = 0.15
    hidden: int = 64
    dropout: float = 0.2
    text_dropout: float = 0.5
    ensemble: int = 5
    seed: int = 42
    quantiles: tuple[float, ...] = field(default=(0.1, 0.5, 0.9))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quantiles"] = list(self.quantiles)
        return d


def _to_tensors(inputs: dict[str, np.ndarray], idx: np.ndarray | None = None):
    out = {}
    for k, v in inputs.items():
        arr = v if idx is None else v[idx]
        out[k] = torch.tensor(arr, dtype=torch.long if k == "category" else torch.float32)
    return out


class NeuralRegressor:
    """An ensemble of :class:`ViewsNet` models predicting quantiles of log1p(views)."""

    def __init__(
        self,
        config: TrainConfig,
        n_numeric: int,
        text_dims: dict[str, int],
        n_categories: int,
    ) -> None:
        self.config = config
        self.model_kwargs = {
            "n_numeric": n_numeric,
            "text_dims": dict(text_dims),
            "n_categories": n_categories,
            "hidden": config.hidden,
            "dropout": config.dropout,
            "text_dropout": config.text_dropout,
            "quantiles": tuple(config.quantiles),
        }
        self.members: list[ViewsNet] = []
        self.y_mean = 0.0
        self.y_std = 1.0
        self.interval_offset = 0.0  # conformal widening of the outer quantiles (log space)
        self.history: list[dict[str, Any]] = []

    @property
    def quantiles(self) -> tuple[float, ...]:
        return tuple(sorted(self.config.quantiles))

    # ------------------------------------------------------------------ training
    def fit(
        self,
        inputs: dict[str, np.ndarray],
        y: np.ndarray,
        groups: np.ndarray | None = None,
        log: Callable[[str], None] | None = None,
    ) -> NeuralRegressor:
        cfg = self.config
        y = np.asarray(y, dtype=np.float32)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)
        y_std = (y - self.y_mean) / self.y_std
        groups = np.arange(len(y)) if groups is None else np.asarray(groups)

        self.members, self.history = [], []
        conformity: list[np.ndarray] = []
        for k in range(cfg.ensemble):
            seed = cfg.seed + k
            splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.val_fraction, random_state=seed)
            tr, va = next(splitter.split(y, groups=groups))
            member, info = self._fit_member(inputs, y_std, tr, va, seed)
            self.members.append(member)
            self.history.append(info)
            if log:
                log(
                    f"  member {k + 1}/{cfg.ensemble}: best epoch {info['best_epoch']}, "
                    f"val MAE {info['val_mae']:.3f} (log views)"
                )
            q = self._predict_member(member, _to_tensors(inputs, va))
            conformity.append(np.maximum(q[:, 0] - y[va], y[va] - q[:, -1]))

        scores = np.concatenate(conformity)
        n = len(scores)
        coverage = self.quantiles[-1] - self.quantiles[0]
        level = min(1.0, math.ceil((n + 1) * coverage) / n)
        self.interval_offset = float(np.quantile(scores, level))
        return self

    def _fit_member(self, inputs, y_std, tr, va, seed):
        cfg = self.config
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ViewsNet(**self.model_kwargs)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
        x_va, y_va = _to_tensors(inputs, va), torch.as_tensor(y_std[va])
        best = (float("inf"), 0, None)
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            order = rng.permutation(tr)
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                if len(idx) < 2:  # skip a trailing batch of one
                    continue
                pred = model(_to_tensors(inputs, idx))
                loss = pinball_loss(pred, torch.as_tensor(y_std[idx]), model.quantiles)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(pinball_loss(model(x_va), y_va, model.quantiles))
            sched.step(val_loss)
            if val_loss < best[0] - 1e-5:
                best = (val_loss, epoch, copy.deepcopy(model.state_dict()))
            elif epoch - best[1] >= cfg.patience:
                break
        model.load_state_dict(best[2])
        model.eval()
        with torch.no_grad():
            med = model(x_va)[:, model.median_index].numpy()
        val_mae = float(np.abs(med - y_std[va]).mean() * self.y_std)
        return model, {"best_epoch": best[1], "val_loss": best[0], "val_mae": val_mae}

    # ------------------------------------------------------------------ inference
    def _predict_member(self, model: ViewsNet, x: dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            return model(x).numpy() * self.y_std + self.y_mean

    def predict_quantiles(self, inputs: dict[str, np.ndarray], calibrated: bool = True):
        """Return an ``(n, n_quantiles)`` array of log1p(views) quantiles."""
        x = _to_tensors(inputs)
        q = np.mean([self._predict_member(m, x) for m in self.members], axis=0)
        if calibrated:
            m = self.quantiles.index(0.5)
            q[:, 0] = np.minimum(q[:, 0] - self.interval_offset, q[:, m])
            q[:, -1] = np.maximum(q[:, -1] + self.interval_offset, q[:, m])
        return q

    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return self.predict_quantiles(inputs)[:, self.quantiles.index(0.5)]

    # ------------------------------------------------------------------ persistence
    def state(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "model_kwargs": self.model_kwargs,
            "members": [m.state_dict() for m in self.members],
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "interval_offset": self.interval_offset,
            "history": self.history,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> NeuralRegressor:
        cfg = dict(state["config"])
        cfg["quantiles"] = tuple(cfg["quantiles"])
        kw = state["model_kwargs"]
        obj = cls(TrainConfig(**cfg), kw["n_numeric"], kw["text_dims"], kw["n_categories"])
        obj.model_kwargs = kw
        for sd in state["members"]:
            m = ViewsNet(**kw)
            m.load_state_dict(sd)
            m.eval()
            obj.members.append(m)
        obj.y_mean, obj.y_std = state["y_mean"], state["y_std"]
        obj.interval_offset = state["interval_offset"]
        obj.history = state.get("history", [])
        return obj
