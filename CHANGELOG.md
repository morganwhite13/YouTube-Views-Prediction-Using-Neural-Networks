# Changelog

## 2.0.0 (unreleased)

A rewrite of the project as the `ytviews` package. The original is kept in `legacy/`.

**Modelling**
- Predicts `log1p(views)` (or likes/comments with `--target`) as calibrated p10/p50/p90 ranges.
- Wide-and-deep PyTorch network with gated text/context fusion and a monotone quantile head,
  ensembled and blended with LightGBM when an inner split shows that helps.
- Inputs: sentence-transformer or TF-IDF text embeddings (multilingual option), CLIP or
  offline image statistics for thumbnails (with optional face features), channel size and
  past-only channel history, publish time, length, Shorts flag and language.
- Video age enters through a monotone growth curve, so predicted views never fall as a video
  ages (8% of predicted steps did before); lower backtest error too. Periodic numeric
  embeddings are available as an option. LightGBM is constrained the same way (Huber loss,
  since L1 can't take monotone constraints), so the blend keeps the guarantee.
- Missing channel history is estimated from subscribers; training augments rows without channel
  context or thumbnails so partial inputs still predict well.
- Leakage-safe evaluation: time / unseen-channel / random splits, four baselines, rolling
  backtests, permutation importance, monitoring on videos newer than the model.

**Data**
- Quota-aware YouTube client (50 ids per call, uploads playlists, retries, budgets); SQLite
  store with a snapshot per fetch; Shorts detection and thumbnail download without quota;
  import of v1 JSON caches; CSV/Parquet export; scheduled collection workflow.

**Tools**
- CLI: predict, compare titles and thumbnails, explain (exact group-Shapley), suggest titles
  (rules, optionally Claude), forecast live videos from early views, best publish time,
  outliers, channel report, batch predictions, tuning, experiment log, doctor.
- `plan` ties the tools together for a draft: explanation, title ideas, thumbnail ranking
  and publish slot, with the winning combination predicted jointly (CLI with HTML output,
  `/plan` endpoint, app tab). `update` chains refresh, enrichment, monitoring, retraining
  when degraded, and reports. Channel reports forecast recent uploads from their early views
  and list title words that beat expectations; `pulse` does the forecast live from the API.
- `schedule` assigns several planned videos to days (one per day) to maximise total predicted
  views; `track --video-id` forecasts a stored video from its latest snapshot.
- Explanations can be relative to another video; `plan` uses this to split its uplift between
  title, thumbnail and publish time, and adopts the best subset of the changes.
- Streamlit app, FastAPI service (optional API key, CORS, input limits), Dockerfile, HTML
  reports, model cards, walkthrough notebook.

## 1.0 (legacy)

The original course project: BiLSTM models over title, description, channel name and category
in a single script. See `legacy/README.md`.
