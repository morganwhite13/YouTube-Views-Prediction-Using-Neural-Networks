import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from ytviews.cli import app  # noqa: E402
from ytviews.onnx_backend import ONNX_FILE, export_onnx  # noqa: E402
from ytviews.predictor import Predictor, VideoInput  # noqa: E402


@pytest.fixture(scope="module")
def exported(trained, tmp_path_factory):
    model = tmp_path_factory.mktemp("onnx") / "model"
    shutil.copytree(trained[0], model)
    export_onnx(model)
    return model


VIDEOS = [
    VideoInput(title=f"Video {i}: a test", subscribers=10.0 ** (2 + i % 5), age_days=0.1 + 3 * i,
               recent_video_views=[1_000.0 * (i + 1)], recent_video_age_days=[5.0])
    for i in range(40)
]  # fmt: skip


def test_onnx_predictions_match_pytorch(exported):
    a = Predictor.load(exported, backend="torch").predict(VIDEOS)
    b = Predictor.load(exported, backend="onnx").predict(VIDEOS)
    for x, y in zip(a, b, strict=True):
        for field in ("low", "median", "high"):
            assert getattr(y, field) == pytest.approx(getattr(x, field), rel=1e-4)


def test_onnx_backend_runs_without_pytorch(exported):
    script = f"""
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not installed")
sys.meta_path.insert(0, NoTorch())
from ytviews.predictor import Predictor, VideoInput
p = Predictor.load({str(exported)!r})
r = p.predict(VideoInput(title="hello", subscribers=5000))[0]
p.publish_times(VideoInput(title="x"), "Asia/Tokyo")
p.explain(VideoInput(title="x", subscribers=1000))
assert not any(m == "torch" or m.startswith("torch.") for m in sys.modules)
print(type(p.regressor.nn).__name__, r.median)
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=300)  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    name, median = out.stdout.split()
    assert name == "OnnxNetwork" and float(median) > 0


def test_resaving_drops_a_stale_export_and_onnx_models_cannot_be_saved(exported, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(exported, copy)
    Predictor.load(copy, backend="torch").save(copy)
    assert not (copy / ONNX_FILE).exists()
    with pytest.raises(ValueError, match="can't be saved"):
        Predictor.load(exported, backend="onnx").save(tmp_path / "other")
    with pytest.raises(ValueError, match="unknown backend"):
        Predictor.load(exported, backend="tpu")
    with pytest.raises(FileNotFoundError, match="export-onnx"):
        Predictor.load(copy, backend="onnx")


def test_export_onnx_cli(trained, tmp_path):
    model = tmp_path / "m"
    shutil.copytree(trained[0], model)
    r = CliRunner().invoke(app, ["export-onnx", "-m", str(model)])
    assert r.exit_code == 0, r.output
    assert "max difference from PyTorch" in r.output and (model / ONNX_FILE).exists()
    meta = json.loads((model / "network.json").read_text())
    assert meta["quantiles"] == [0.1, 0.5, 0.9] and np.isfinite(meta["y_std"])
