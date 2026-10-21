import numpy as np
import pandas as pd

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
