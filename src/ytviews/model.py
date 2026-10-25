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

import math
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


class PeriodicEmbedding(nn.Module):
    """Per-feature periodic embeddings of numeric inputs ("PLR", Gorishniy et al. 2022).

    Each standardised feature x becomes ``ReLU(W_i [sin(2πc·x), cos(2πc·x)])`` with learned
    frequencies ``c`` per feature, so an MLP can learn sharp non-linear responses to a single
    feature (thresholds in subscriber count, the shape of a channel-history effect) that a
    plain linear layer over all features at once struggles with.
    """

    def __init__(self, n_features: int, n_frequencies: int = 8, dim: int = 8,
                 sigma: float = 0.3) -> None:  # fmt: skip
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        self.weight = nn.Parameter(
            torch.randn(n_features, 2 * n_frequencies, dim) / math.sqrt(2 * n_frequencies)
        )
        self.bias = nn.Parameter(torch.zeros(n_features, dim))
        self.out_dim = n_features * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 2 * math.pi * self.freq[None] * x[..., None]  # (batch, features, frequencies)
        z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        z = torch.einsum("bfk,fkd->bfd", z, self.weight) + self.bias
        return torch.relu(z).flatten(1)


class GrowthCurve(nn.Module):
    """A monotone growth curve in video age: ``sum_k w_k(x) * sigmoid((t - c_k) / s_k)``.

    ``t`` is the (standardised) log age; the weights ``w_k >= 0`` depend on the video, the
    centres and widths are shared. Predicted views can therefore only rise with age, at a
    video-specific pace, instead of the network learning (and sometimes breaking) that.
    """

    def __init__(self, in_dim: int, n_basis: int = 8) -> None:
        super().__init__()
        self.centres = nn.Parameter(torch.linspace(-2.0, 2.0, n_basis))
        self.log_width = nn.Parameter(torch.zeros(n_basis))
        self.weights = nn.Linear(in_dim, n_basis)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        w = nn.functional.softplus(self.weights(h))
        basis = torch.sigmoid((t[:, None] - self.centres) / self.log_width.exp())
        return (w * basis).sum(-1, keepdim=True)


def _tower(in_dim: int, out_dim: int, input_dropout: float) -> nn.Sequential:
    # Inputs arrive standardised per column, so no per-sample normalisation here: LayerNorm
    # across heterogeneous features (e.g. subscribers next to hour-of-day) destroys magnitudes.
    # Input dropout is the main defence against memorising individual videos from their text.
    return nn.Sequential(nn.Dropout(input_dropout), nn.Linear(in_dim, out_dim), nn.GELU())


class ViewsNet(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        embed_dims: dict[str, int],
        n_categories: int,
        *,
        hidden: int = 64,
        cat_dim: int = 8,
        dropout: float = 0.2,
        embed_dropout: float = 0.5,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        numeric_embedding: str = "linear",
        age_index: int | None = None,
    ) -> None:
        super().__init__()
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include the median (0.5)")
        self.quantiles = tuple(sorted(quantiles))
        self.median_index = self.quantiles.index(0.5)
        self.embed_fields = tuple(sorted(embed_dims))

        self.embed_towers = nn.ModuleDict(
            {k: _tower(d, hidden, embed_dropout) for k, d in embed_dims.items()}
        )
        self.category = nn.Embedding(n_categories, cat_dim)
        # Optional architecture choices (defaults reproduce the original network, so models
        # saved before they existed load unchanged).
        if numeric_embedding not in ("linear", "periodic"):
            raise ValueError(f"unknown numeric_embedding {numeric_embedding!r}")
        self.periodic = PeriodicEmbedding(n_numeric) if numeric_embedding == "periodic" else None
        numeric_in = self.periodic.out_dim if self.periodic is not None else n_numeric
        self.numeric = _tower(numeric_in, hidden, dropout / 2)
        self.age_index = age_index  # set -> age enters only through a monotone growth curve
        context_dim = hidden + cat_dim
        self.embed_gate = nn.Linear(context_dim, hidden * len(self.embed_fields))

        fused = hidden * len(self.embed_fields) + context_dim
        self.fuse = nn.Sequential(nn.Linear(fused, hidden), nn.GELU())
        self.blocks = nn.Sequential(ResidualBlock(hidden, dropout), ResidualBlock(hidden, dropout))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, len(self.quantiles)))
        # "Wide" path: a linear model on the raw inputs, so the deep part only has to learn
        # interactions and non-linearities on top of it.
        wide_in = n_numeric + cat_dim + sum(embed_dims.values())
        self.wide = nn.Linear(wide_in, 1)
        self.growth = GrowthCurve(hidden) if age_index is not None else None

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        numeric = inputs["numeric"]
        if self.age_index is not None:  # hide age from everything but the growth curve
            age = numeric[:, self.age_index]
            numeric = numeric.clone()
            numeric[:, self.age_index] = 0.0
        cat = self.category(inputs["category"])
        num = self.periodic(numeric) if self.periodic is not None else numeric
        context = torch.cat([self.numeric(num), cat], dim=-1)
        text = torch.cat([self.embed_towers[k](inputs[k]) for k in self.embed_fields], dim=-1)
        text = text * torch.sigmoid(self.embed_gate(context)) * 2.0  # gate centred on identity
        h = self.blocks(self.fuse(torch.cat([text, context], dim=-1)))
        wide = self.wide(torch.cat([numeric, cat, *(inputs[k] for k in self.embed_fields)], dim=-1))
        raw = self.head(h)
        m = self.median_index
        raw = raw + nn.functional.pad(wide, (m, len(self.quantiles) - 1 - m))  # wide -> median
        out = self._monotone(raw)
        if self.growth is not None:  # the same rise for every quantile keeps them ordered
            out = out + self.growth(h, age)
        return out

    def _monotone(self, raw: torch.Tensor) -> torch.Tensor:
        m = self.median_index
        median = raw[:, m : m + 1]
        below = torch.nn.functional.softplus(raw[:, :m]).flip(-1).cumsum(-1).flip(-1)
        above = torch.nn.functional.softplus(raw[:, m + 1 :]).cumsum(-1)
        return torch.cat([median - below, median, median + above], dim=-1)


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: Sequence[float],
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    diff = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * diff, (q - 1) * diff).mean(dim=-1)
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum()
