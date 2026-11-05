"""Run the network with ONNX Runtime instead of PyTorch.

``ytviews export-onnx models/latest`` writes ``network.onnx`` (the whole ensemble as one graph)
and ``network.json`` (its output scaling and calibration) next to the model. Then
``Predictor.load(path, backend="onnx")`` (or ``YTVIEWS_BACKEND=onnx``) predicts without
importing PyTorch: a much smaller install for serving (``pip install onnxruntime`` instead of
torch), and usually faster on CPU. LightGBM, the feature pipeline and the calibration are
unchanged, so predictions match the PyTorch ones to float32 precision.

Serving without PyTorch needs a model trained with the offline encoders (``--text-encoder
tfidf``, ``--image-encoder stats``): sentence-transformers and CLIP are PyTorch models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ONNX_FILE = "network.onnx"
ONNX_META_FILE = "network.json"


def export_onnx(model_dir: str | Path, opset: int = 18) -> Path:
    """Export a saved model's network ensemble to ONNX (needs torch and onnx)."""
    import torch

    from .predictor import Predictor, VideoInput

    model_dir = Path(model_dir)
    p = Predictor.load(model_dir, backend="torch")
    nn_reg = getattr(p.regressor, "nn", p.regressor)
    example = p.features.transform(pd.DataFrame([VideoInput(title="example").to_row()] * 2))
    keys = sorted(example)

    class Ensemble(torch.nn.Module):
        def __init__(self, members: list[torch.nn.Module]) -> None:
            super().__init__()
            self.members = torch.nn.ModuleList(members)

        def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
            x = dict(zip(keys, tensors, strict=True))
            return torch.stack([m(x) for m in self.members]).mean(0)

    model = Ensemble(nn_reg.members).eval()
    args = tuple(
        torch.tensor(example[k], dtype=torch.long if k == "category" else torch.float32)
        for k in keys
    )
    out = model_dir / ONNX_FILE
    with torch.no_grad():
        batch = torch.export.Dim("batch", min=1)
        torch.onnx.export(
            model, args, str(out), input_names=keys, output_names=["quantiles"],
            dynamic_shapes=(tuple({0: batch} for _ in keys),),  # *tensors is one argument
            opset_version=opset, dynamo=True,
            verbose=False,
        )  # fmt: skip
    (model_dir / ONNX_META_FILE).write_text(
        json.dumps(
            {
                "inputs": keys,
                "quantiles": list(nn_reg.quantiles),
                "y_mean": float(nn_reg.y_mean),
                "y_std": float(nn_reg.y_std),
                "interval_offset": float(nn_reg.interval_offset),
            },
            indent=2,
        )
    )
    return out


class OnnxNetwork:
    """Drop-in for :class:`~ytviews.train.NeuralRegressor` at prediction time."""

    def __init__(self, session: Any, meta: dict[str, Any]) -> None:
        self.session = session
        self.inputs = list(meta["inputs"])
        self.quantiles = tuple(meta["quantiles"])
        self.y_mean = float(meta["y_mean"])
        self.y_std = float(meta["y_std"])
        self.interval_offset = float(meta["interval_offset"])

    @classmethod
    def load(cls, model_dir: str | Path) -> OnnxNetwork:
        model_dir = Path(model_dir)
        if not (model_dir / ONNX_FILE).exists():
            raise FileNotFoundError(
                f"No ONNX export in {model_dir}; run `ytviews export-onnx {model_dir}` first "
                "(or install PyTorch)."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError("the ONNX backend needs `pip install onnxruntime`") from exc
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(model_dir / ONNX_FILE), options, providers=["CPUExecutionProvider"]
        )
        return cls(session, json.loads((model_dir / ONNX_META_FILE).read_text()))

    def predict_quantiles(self, inputs: dict[str, np.ndarray], calibrated: bool = True):
        feed = {
            k: inputs[k].astype(np.int64 if k == "category" else np.float32) for k in self.inputs
        }
        q = self.session.run(None, feed)[0].astype(np.float64) * self.y_std + self.y_mean
        if calibrated:
            m = self.quantiles.index(0.5)
            q[:, 0] = np.minimum(q[:, 0] - self.interval_offset, q[:, m])
            q[:, -1] = np.maximum(q[:, -1] + self.interval_offset, q[:, m])
        return q

    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return self.predict_quantiles(inputs)[:, self.quantiles.index(0.5)]
