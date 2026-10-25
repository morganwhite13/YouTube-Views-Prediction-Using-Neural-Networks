import numpy as np
import pandas as pd
import pytest

from ytviews.features import FeatureBuilder, add_channel_history, drop_thumbnails
from ytviews.images import ImageStatsEncoder, attach_thumbnails, thumbnail_dir_for
from ytviews.synthetic import render_thumbnail


def test_image_stats_reflect_design(tmp_path):
    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    render_thumbnail(2.0, 0.1, 0, good)
    render_thumbnail(-2.0, 0.1, 0, bad)
    enc = ImageStatsEncoder()
    feats, missing = enc.transform([str(good), str(bad), None, str(tmp_path / "nope.jpg")])
    assert feats.shape == (4, enc.output_dim)
    assert missing.tolist() == [False, False, True, True]
    sat = 9  # index of mean saturation in image_stats
    assert feats[0, sat] > feats[1, sat]
    assert np.all(feats[2] == 0)


def test_attach_thumbnails(synthetic_db, synthetic_df):
    df = attach_thumbnails(synthetic_df, thumbnail_dir_for(synthetic_db))
    assert df["thumbnail_path"].notna().all()


def test_feature_builder_adds_thumbnail_input(synthetic_db, synthetic_df):
    df = add_channel_history(attach_thumbnails(synthetic_df, thumbnail_dir_for(synthetic_db)))
    train = df.iloc[:300]
    fb = FeatureBuilder(text_encoder="tfidf", text_dim=16, image_encoder="stats").fit(train)
    assert fb.uses_thumbnails and "thumbnail" in fb.embed_dims
    test = df.iloc[300:310].copy()
    test["thumbnail_path"] = [None] * 5 + list(test["thumbnail_path"].iloc[5:])
    x = fb.transform(test)["thumbnail"]
    assert x.shape == (10, fb.embed_dims["thumbnail"])
    assert (x[:5, -1] == 1).all() and (x[5:, -1] == 0).all()  # missing flag column
    assert np.allclose(x[:5, :-1], 0)
    # without a thumbnail column at all, the input is still produced (all missing)
    assert fb.transform(test.drop(columns="thumbnail_path"))["thumbnail"][:, -1].all()


def test_drop_thumbnails_marks_augmented_rows():
    df = pd.DataFrame({"video_id": list("abcd"), "thumbnail_path": ["x"] * 4})
    out = drop_thumbnails(df, fraction=0.5)
    assert len(out) == 6
    assert out["thumbnail_path"].isna().sum() == 2
    assert out["_augmented"].eq(True).sum() == 2


def test_face_features_find_a_real_face(tmp_path):
    pytest.importorskip("cv2")
    skimage_data = pytest.importorskip("skimage.data")
    from PIL import Image

    from ytviews.images import STATS_DIM, faces_available

    if not faces_available():
        pytest.skip("OpenCV without the bundled face detector")
    face = tmp_path / "astronaut.jpg"
    Image.fromarray(skimage_data.astronaut()).save(face)
    blank = tmp_path / "blank.jpg"
    render_thumbnail(0.0, 0.5, 0, blank)
    enc = ImageStatsEncoder(faces=True)
    feats, missing = enc.transform([str(face), str(blank)])
    assert enc.output_dim == STATS_DIM + 4 and not missing.any()
    n_faces, area, cx, cy = feats[0, STATS_DIM:]
    assert n_faces >= 1 and 0.01 < area < 0.5 and 0 < cx < 1 and 0 < cy < 1
    assert feats[1, STATS_DIM] == 0  # no face in the synthetic thumbnail


def test_face_features_are_optional():
    enc = ImageStatsEncoder(faces=False)
    from ytviews.images import STATS_DIM

    assert enc.output_dim == STATS_DIM


def test_faces_are_explicit_and_recorded_in_kind(monkeypatch):
    import ytviews.images as images

    assert images.make_image_encoder("stats").kind == "stats"
    monkeypatch.setattr(images, "_FACES_OK", False)
    with pytest.raises(RuntimeError, match="faces"):
        images.make_image_encoder("stats+faces")


def test_faces_model_without_opencv_still_predicts_without_thumbnails(monkeypatch, tmp_path):
    import ytviews.images as images

    if not images.faces_available():
        pytest.skip("needs OpenCV to build the encoder")
    enc = images.ImageStatsEncoder(faces=True)
    monkeypatch.setattr(images, "_FACES_OK", False)  # e.g. served where OpenCV is missing
    feats, missing = enc.transform([None, None])
    assert missing.all() and feats.shape == (2, enc.output_dim)
    thumb = tmp_path / "t.jpg"
    render_thumbnail(0.0, 0.5, 0, thumb)
    with pytest.raises(RuntimeError, match="ytviews\\[faces\\]"):
        enc.transform([str(thumb)])


def test_face_detection_is_cached(monkeypatch, tmp_path):
    import ytviews.images as images

    if not images.faces_available():
        pytest.skip("needs OpenCV")
    monkeypatch.setenv("YTVIEWS_CACHE_DIR", str(tmp_path / "cache"))
    calls = []
    real = images.face_stats
    monkeypatch.setattr(images, "face_stats", lambda img: calls.append(1) or real(img))
    thumb = tmp_path / "t.jpg"
    render_thumbnail(1.0, 0.5, 0, thumb)
    enc = images.ImageStatsEncoder(faces=True)
    first, _ = enc.transform([str(thumb), str(thumb)])
    second, _ = enc.transform([str(thumb)])
    assert len(calls) == 1  # duplicate in the batch and the second call hit the cache
    assert np.allclose(first[0], second[0])


def test_constant_columns_are_not_blown_up():
    from ytviews.features import _safe_std

    x = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]])
    std = _safe_std(x)
    assert std[0] == 1.0 and np.isclose(std[1], x[:, 1].std())
