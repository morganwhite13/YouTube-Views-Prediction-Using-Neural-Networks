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


def test_architecture_options_are_monotone_in_age_and_load_back():
    import torch

    from ytviews.model import ViewsNet

    torch.manual_seed(0)
    net = ViewsNet(10, {"title": 6}, 3, hidden=16, numeric_embedding="periodic", age_index=0)
    net.eval()
    x = {"numeric": torch.randn(50, 10), "title": torch.randn(50, 6),
         "category": torch.randint(0, 3, (50,))}  # fmt: skip
    outs = []
    for t in torch.linspace(-3, 3, 9):
        xt = dict(x, numeric=x["numeric"].clone())
        xt["numeric"][:, 0] = t
        outs.append(net(xt))
    outs = torch.stack(outs)  # (ages, batch, quantiles)
    assert (outs[1:] >= outs[:-1] - 1e-6).all()  # every quantile rises with age
    assert (outs[..., 0] <= outs[..., 1]).all() and (outs[..., 1] <= outs[..., 2]).all()
    again = ViewsNet(10, {"title": 6}, 3, hidden=16, numeric_embedding="periodic", age_index=0)
    again.load_state_dict(net.state_dict())
    again.eval()
    assert torch.allclose(again(x), net(x))
    import pytest

    with pytest.raises(ValueError):
        ViewsNet(10, {"title": 6}, 3, numeric_embedding="bogus")


def test_trained_model_rises_with_age_without_post_processing(trained):
    from dataclasses import replace

    from ytviews.predictor import Predictor, VideoInput

    p = Predictor.load(trained[0])
    base = VideoInput(title="Weekly Update", subscribers=20_000, category_id="20")
    for title in ("Weekly Update", "I Survived 100 Days", "minecraft #shorts"):
        ages = [0.05, 0.5, 1, 3, 7, 30, 90, 365, 1500]
        preds = p.predict([replace(base, title=title, age_days=a) for a in ages])
        for q in ("low", "median", "high"):
            values = [getattr(x, q) for x in preds]
            assert (np.diff(values) >= -1e-6).all(), (title, q, values)
