"""Per-prediction explanations: exact Shapley values over feature groups.

The question answered is *"why does the model expect this many views rather than what an average
video of the same age gets?"*. Each feature group (title text, thumbnail, channel size, ...) is
either "as given" or replaced by its training average; with ~10 groups every one of the 2^10
combinations fits in a single forward pass, so the Shapley values are exact rather than sampled.

Effects are additive in log space, i.e. multiplicative on views:
``baseline_views x factor_1 x factor_2 x ... = predicted_views``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .evaluate import input_groups

if TYPE_CHECKING:  # pragma: no cover
    from .predictor import Predictor, VideoInput

# Age is held at the requested value: "views after 7 days" should be explained relative to an
# average video *after 7 days*, not relative to an average-aged video.
FIXED_GROUPS = ("video age",)


@dataclass
class Explanation:
    baseline_views: float
    predicted_views: float
    effects: pd.DataFrame  # columns: group, log_effect, factor

    def summary(self, top: int | None = None) -> list[str]:
        rows = self.effects if top is None else self.effects.head(top)
        return [f"{r.group:<18} x{r.factor:.2f}" for r in rows.itertuples()]


def _baseline_inputs(predictor: Predictor, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The 'average video': standardised features at 0 (= training mean), commonest category."""
    base = {k: np.zeros_like(v) for k, v in x.items()}
    known = predictor.features.categories_
    counts = {
        c: n for c, n in getattr(predictor.features, "category_counts_", {}).items() if c in known
    }
    if counts:
        base["category"][:] = known[max(counts, key=counts.get)]
    return base


def explain(predictor: Predictor, video: VideoInput) -> Explanation:
    x = predictor.features.transform(pd.DataFrame([video.to_row()]))
    base = _baseline_inputs(predictor, x)
    groups = {n: g for n, g in input_groups(x).items() if n not in FIXED_GROUPS}
    for name in FIXED_GROUPS:  # fixed groups always take the actual value
        key, cols = input_groups(x)[name]
        base[key][:, cols] = x[key][:, cols]
    names = list(groups)
    n = len(names)

    # Build one row per coalition (bitmask over groups): actual values where the bit is set.
    masks = np.array([[(m >> i) & 1 for i in range(n)] for m in range(2**n)], dtype=bool)
    batch = {k: np.repeat(v, len(masks), axis=0) for k, v in base.items()}
    for i, name in enumerate(names):
        key, cols = groups[name]
        on = masks[:, i]
        if cols is None:
            batch[key][on] = x[key][0]
        else:
            rows = np.flatnonzero(on)
            batch[key][np.ix_(rows, cols)] = x[key][0, cols]
    value = predictor.regressor.predict(batch)  # median log1p(views) per coalition

    index = {tuple(m): j for j, m in enumerate(masks)}
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            w = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in combinations(others, size):
                m = np.zeros(n, dtype=bool)
                m[list(subset)] = True
                without = value[index[tuple(m)]]
                m[i] = True
                phi[i] += w * (value[index[tuple(m)]] - without)

    effects = pd.DataFrame({"group": names, "log_effect": phi, "factor": np.exp(phi)})
    effects = effects.reindex(effects["log_effect"].abs().sort_values(ascending=False).index)
    return Explanation(
        baseline_views=float(np.expm1(value[0])),
        predicted_views=float(np.expm1(value[-1])),
        effects=effects.reset_index(drop=True),
    )
