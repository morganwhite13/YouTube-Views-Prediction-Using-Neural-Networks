"""The network/LightGBM blend used at prediction time.

Kept free of PyTorch so a model exported to ONNX (:mod:`ytviews.onnx_backend`) can be served
without it: the blend only needs something with ``quantiles`` and ``predict_quantiles``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .features import NUMERIC_FEATURES

if TYPE_CHECKING:
    from .train import NeuralRegressor, TrainConfig


class BlendedRegressor:
    """Blend of the network's median with a LightGBM model, keeping the network's ranges.

    The blended median is ``weight * nn + (1 - weight) * gbm``; the network's quantiles are
    shifted by the same amount, so the range keeps its calibrated width. Exposes the same
    interface as :class:`NeuralRegressor`, so explanations, forecasts and the API work unchanged.
    """

    def __init__(
        self,
        nn: NeuralRegressor,
        gbm: Any,
        features: Any,
        weight: float,
        interval_adjust: float = 0.0,
        interval_scale: tuple[float, float] | None = None,
        age_floor: float | None = None,
    ) -> None:
        self.nn = nn
        self.gbm = gbm  # a fitted lightgbm.LGBMRegressor
        self.features = features  # FeatureBuilder producing the inputs (for the flat matrix)
        self.weight = float(weight)
        self.interval_adjust = float(interval_adjust)  # re-calibration of the blended range
        # Newer models: separate stretch factors for the lower and upper half of the range,
        # applied to the network's own half-widths (see pipeline._calibrate_blend). Models
        # saved before this use the single additive ``interval_adjust``.
        self.interval_scale = None if interval_scale is None else tuple(map(float, interval_scale))
        # Standardised log-age below which LightGBM saw (almost) no training rows. Trees are
        # flat beyond their data, so there LightGBM is extended along the network's growth
        # curve: gbm(x) + nn(x) - nn(x at the floor). That is continuous at the floor and still
        # rises with age. None (models saved before this): LightGBM is used as it is.
        self.age_floor = age_floor

    @property
    def quantiles(self) -> tuple[float, ...]:
        return self.nn.quantiles

    @property
    def config(self) -> TrainConfig:
        return self.nn.config

    def predict_quantiles(self, inputs: dict[str, np.ndarray], calibrated: bool = True):
        q = self.nn.predict_quantiles(inputs, calibrated)
        m = self.quantiles.index(0.5)
        if self.weight < 1.0:
            gbm = self.gbm.predict(self.features.flat(inputs, one_hot=False))
            gbm = gbm + self._below_floor_growth(inputs, q[:, m])
            blended = self.weight * q[:, m] + (1 - self.weight) * gbm
        else:
            blended = q[:, m]
        if calibrated and self.interval_scale is not None:
            s_lo, s_hi = self.interval_scale
            out = np.repeat(blended[:, None], q.shape[1], axis=1)
            out[:, :m] -= s_lo * (q[:, m : m + 1] - q[:, :m])
            out[:, m + 1 :] += s_hi * (q[:, m + 1 :] - q[:, m : m + 1])
            return out
        q = q + (blended - q[:, m])[:, None]
        if calibrated and self.interval_adjust:
            q[:, 0] = np.minimum(q[:, 0] - self.interval_adjust, q[:, m])
            q[:, -1] = np.maximum(q[:, -1] + self.interval_adjust, q[:, m])
        return q

    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return self.predict_quantiles(inputs)[:, self.quantiles.index(0.5)]

    def _below_floor_growth(self, inputs: dict[str, np.ndarray], nn_median: np.ndarray):
        """The network's growth between each row's age and the floor, for rows below it."""
        out = np.zeros(len(nn_median))
        floor = getattr(self, "age_floor", None)
        if floor is None:
            return out
        col = NUMERIC_FEATURES.index("log_age_days")
        below = inputs["numeric"][:, col] < floor
        if not below.any():
            return out
        at_floor = {k: v[below].copy() for k, v in inputs.items()}
        at_floor["numeric"][:, col] = floor
        m = self.quantiles.index(0.5)
        out[below] = nn_median[below] - self.nn.predict_quantiles(at_floor)[:, m]
        return out
