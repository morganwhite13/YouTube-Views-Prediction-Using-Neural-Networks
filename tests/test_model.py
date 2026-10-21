import numpy as np
import pytest
import torch

from ytviews.evaluate import regression_metrics, split_frame
from ytviews.model import ViewsNet, pinball_loss


def _inputs(n=8):
    g = torch.Generator().manual_seed(0)
    return {
        "numeric": torch.randn(n, 5, generator=g),
        "category": torch.randint(0, 3, (n,), generator=g),
        "title": torch.randn(n, 7, generator=g),
        "description": torch.randn(n, 7, generator=g),
    }


def test_quantiles_never_cross():
    net = ViewsNet(5, {"title": 7, "description": 7}, 3, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 3)  # wild weights: monotonicity must hold by construction
        out = net(_inputs())
    assert out.shape == (8, 5)
    assert (out.diff(dim=1) >= 0).all()


def test_quantiles_must_include_median():
    with pytest.raises(ValueError):
        ViewsNet(5, {"title": 7}, 3, quantiles=(0.1, 0.9))


def test_pinball_loss_matches_mae_at_median():
    pred = torch.tensor([[1.0], [3.0]])
    target = torch.tensor([2.0, 2.0])
    assert torch.isclose(pinball_loss(pred, target, (0.5,)), torch.tensor(0.5))


def test_metrics_on_perfect_prediction():
    y = np.log1p(np.array([10, 100, 1000, 10000.0]))
    m = regression_metrics(y, y, (y - 1, y + 1))
    assert m["mae_log"] == 0 and m["within_2x"] == 1 and m["interval_coverage"] == 1


def test_splits_keep_snapshots_and_channels_together(synthetic_df):
    for strategy in ("time", "channel", "random"):
        tr, te = split_frame(synthetic_df, strategy)
        assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(synthetic_df)
        tr_v = set(synthetic_df.iloc[tr]["video_id"])
        te_v = set(synthetic_df.iloc[te]["video_id"])
        assert not tr_v & te_v
    tr, te = split_frame(synthetic_df, "channel")
    assert not set(synthetic_df.iloc[tr]["channel_id"]) & set(synthetic_df.iloc[te]["channel_id"])
    tr, te = split_frame(synthetic_df, "time")
    assert (
        synthetic_df.iloc[tr]["published_at"].max() <= synthetic_df.iloc[te]["published_at"].min()
    )
