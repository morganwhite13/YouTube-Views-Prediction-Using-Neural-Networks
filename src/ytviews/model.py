"""The neural network.

Architecture (wide & deep: one tower per input, a gated fusion, plus a linear "wide" path):

    title emb ──► MLP ─┐
    desc emb  ──► MLP ─┤   context (numeric + category) gates the text representation:
    category  ──► Emb ─┤   "does this title work *for this channel*?"
    numeric   ──► MLP ─┘
                        └─► concat ─► 2 residual blocks ─► quantile head (p10, p50, p90)

The head predicts log1p(views) quantiles. Upper/lower quantiles are built as the median
plus/minus a softplus, so they can never cross.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _tower(in_dim: int, out_dim: int, input_dropout: float) -> nn.Sequential:
    # Inputs arrive standardised per column, so no per-sample normalisation here: LayerNorm
    # across heterogeneous features (e.g. subscribers next to hour-of-day) destroys magnitudes.
    # Input dropout is the main defence against memorising individual videos from their text.
    return nn.Sequential(nn.Dropout(input_dropout), nn.Linear(in_dim, out_dim), nn.GELU())


class ViewsNet(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        text_dims: dict[str, int],
        n_categories: int,
        *,
        hidden: int = 64,
        cat_dim: int = 8,
        dropout: float = 0.2,
        text_dropout: float = 0.5,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include the median (0.5)")
        self.quantiles = tuple(sorted(quantiles))
        self.median_index = self.quantiles.index(0.5)
        self.text_fields = tuple(sorted(text_dims))

        self.text_towers = nn.ModuleDict(
            {k: _tower(d, hidden, text_dropout) for k, d in text_dims.items()}
        )
        self.category = nn.Embedding(n_categories, cat_dim)
        self.numeric = _tower(n_numeric, hidden, dropout / 2)
        context_dim = hidden + cat_dim
        self.text_gate = nn.Linear(context_dim, hidden * len(self.text_fields))

        fused = hidden * len(self.text_fields) + context_dim
        self.fuse = nn.Sequential(nn.Linear(fused, hidden), nn.GELU())
        self.blocks = nn.Sequential(ResidualBlock(hidden, dropout), ResidualBlock(hidden, dropout))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, len(self.quantiles)))
        # "Wide" path: a linear model on the raw inputs, so the deep part only has to learn
        # interactions and non-linearities on top of it.
        wide_in = n_numeric + cat_dim + sum(text_dims.values())
        self.wide = nn.Linear(wide_in, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        cat = self.category(inputs["category"])
        context = torch.cat([self.numeric(inputs["numeric"]), cat], dim=-1)
        text = torch.cat([self.text_towers[k](inputs[k]) for k in self.text_fields], dim=-1)
        text = text * torch.sigmoid(self.text_gate(context)) * 2.0  # gate centred on identity
        h = self.blocks(self.fuse(torch.cat([text, context], dim=-1)))
        wide = self.wide(
            torch.cat([inputs["numeric"], cat, *(inputs[k] for k in self.text_fields)], dim=-1)
        )
        raw = self.head(h)
        m = self.median_index
        raw = raw + nn.functional.pad(wide, (m, len(self.quantiles) - 1 - m))  # wide -> median
        return self._monotone(raw)

    def _monotone(self, raw: torch.Tensor) -> torch.Tensor:
        m = self.median_index
        median = raw[:, m : m + 1]
        below = torch.nn.functional.softplus(raw[:, :m]).flip(-1).cumsum(-1).flip(-1)
        above = torch.nn.functional.softplus(raw[:, m + 1 :]).cumsum(-1)
        return torch.cat([median - below, median, median + above], dim=-1)


def pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]
) -> torch.Tensor:
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()
