import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

pytest.importorskip("rapidocr_onnxruntime")

from ytviews.features import FeatureBuilder  # noqa: E402
from ytviews.images import make_image_encoder  # noqa: E402
from ytviews.ocr import OCR_DIM, ocr_features, read_thumbnail  # noqa: E402


def _thumb(path, text):
    img = Image.new("RGB", (1280, 720), (30, 30, 60))
    if text:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 150)
        except OSError:
            font = ImageFont.load_default(size=150)
        ImageDraw.Draw(img).text((80, 250), text, fill=(255, 230, 0), font=font)
    img.save(path)
    return str(path)


def test_reads_thumbnail_text_and_caches_it(tmp_path, monkeypatch):
    monkeypatch.setenv("YTVIEWS_CACHE_DIR", str(tmp_path / "cache"))
    path = _thumb(tmp_path / "a.jpg", "I QUIT")
    read = read_thumbnail(path)
    assert "QUIT" in read["text"].upper() and 0.02 < read["area"] < 0.5
    assert len(list((tmp_path / "cache" / "ocr").glob("*.json"))) == 1
    assert read_thumbnail(tmp_path / "missing.jpg") is None


def test_ocr_features(tmp_path, monkeypatch):
    monkeypatch.setenv("YTVIEWS_CACHE_DIR", str(tmp_path / "cache"))
    text = _thumb(tmp_path / "a.jpg", "I QUIT")
    blank = _thumb(tmp_path / "b.jpg", "")
    out = ocr_features([text, text, blank, None],
                       ["Why I quit my job", "A cooking video", "x", "x"])  # fmt: skip
    assert out.shape == (4, OCR_DIM)
    assert out[0, 2] == 1.0 and out[0, 3] == 1.0  # has text, all of it in the title
    assert out[1, 3] == 0.0  # none of it in this title
    assert (out[2] == 0).all() and (out[3] == 0).all()  # no text / no thumbnail


def test_feature_builder_appends_ocr_columns(synthetic_df, tmp_path, monkeypatch):
    monkeypatch.setenv("YTVIEWS_CACHE_DIR", str(tmp_path / "cache"))
    df = synthetic_df.drop_duplicates("video_id").head(40).copy()
    df["thumbnail_path"] = [_thumb(tmp_path / f"{i}.jpg", "WOW" if i % 2 else "")
                            for i in range(len(df))]  # fmt: skip
    enc = make_image_encoder("stats+ocr")
    assert enc.kind == "stats+ocr"
    fb = FeatureBuilder(text_encoder="tfidf", image_encoder="stats+ocr").fit(df)
    out = fb.transform(df.head(4))["thumbnail"]
    assert out.shape[1] == make_image_encoder("stats").output_dim + OCR_DIM + 1  # + missing flag
    assert np.isfinite(out).all()


def test_an_unreadable_image_is_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("YTVIEWS_CACHE_DIR", str(tmp_path / "cache"))
    from PIL import PngImagePlugin

    path = tmp_path / "bomb.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("x", "a" * 5_000_000, zip=True)  # inflates past PIL's text-chunk limit
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    assert read_thumbnail(path) is None
    emb, missing = make_image_encoder("stats+ocr").transform([str(path)], ["t"])
    assert missing.tolist() == [True] and (emb == 0).all()
