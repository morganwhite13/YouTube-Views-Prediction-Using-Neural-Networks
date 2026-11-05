from ytviews.train import TrainConfig
from ytviews.tune import SEARCH_SPACE, load_config, save_config, tune


def test_tune_returns_best_trial_and_never_uses_test_rows(synthetic_df, monkeypatch):
    seen_rows = []
    import ytviews.tune as tune_mod

    original = tune_mod.FeatureBuilder.fit

    def spy(self, df):
        seen_rows.append(set(df["video_id"]))
        return original(self, df)

    monkeypatch.setattr(tune_mod.FeatureBuilder, "fit", spy)
    base = TrainConfig(epochs=8, patience=3)
    best, trials = tune(
        synthetic_df, n_trials=3, text_encoder="tfidf", image_encoder="none", base=base,
        ensemble=1, log=lambda _: None,
    )  # fmt: skip
    assert len(trials) == 3 and trials["val_mae"].is_monotonic_increasing
    for k in SEARCH_SPACE:
        assert getattr(best, k) == trials.iloc[0][k]
    assert best.ensemble == TrainConfig().ensemble

    from ytviews.evaluate import split_frame
    from ytviews.features import add_channel_history

    df = add_channel_history(synthetic_df).reset_index(drop=True)
    _, test_idx = split_frame(df, "time", seed=base.seed)
    assert not seen_rows[0] & set(df.iloc[test_idx]["video_id"])


def test_config_roundtrip(tmp_path):
    cfg = TrainConfig(lr=0.005, hidden=32, quantiles=(0.05, 0.5, 0.95))
    save_config(cfg, tmp_path / "c.json")
    assert load_config(tmp_path / "c.json") == cfg
