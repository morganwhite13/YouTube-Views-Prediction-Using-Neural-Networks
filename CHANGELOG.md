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
- The blend's range is calibrated separately below and above the median, as a stretch of the
  network's half-widths (held-out coverage 77.7% -> 79.9% at the same seed; target 80%).
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

- Young videos: the synthetic data now has recent uploads with twice-weekly refresh snapshots,
  the growth curve has a launch term so predictions keep falling below the youngest training
  age, and the blend extends LightGBM along the network's growth curve there. A 1-hour-old
  video no longer gets the same prediction as a 7-day-old one; for held-out videos under a day
  the bias fell from about 7x to 1.2x.
- Predictions note when an input is outside the training range (CLI, app and API).

- Title rewrites read as titles: sentence-style drafts ("Why I Quit My Job") only get
  rewrites that wrap a sentence, "I tried X" drafts reuse X as the topic, drafts that already
  have a format ("$1 vs $10,000 ...", "... review") get no second one, and lower-case drafts
  get a title-cased version.

- `thumbnail-tips` tries simple edits of a thumbnail (brighter, more contrast, more
  saturated, sharper, and the opposites) and ranks them against the original; also as the
  `/thumbnail-tips` API endpoint and in the app's Compare tab.
- The API refuses thumbnails above 4096x4096, read from the header before decoding.
- Cleared pandas 3.1 and GitHub Actions Node 20 deprecation warnings.

- `plan --record` saves a plan; `plans --review` (and `update`) links it to the published
  video by title and compares it with the views it really got. A plan stores its predicted
  range by age, so it is scored on what it predicted, not on a model since retrained on the
  video itself.

**Robustness** (from a review of the whole package)
- Data: a legacy import no longer overwrites fresher rows; `refresh` saves after each batch so
  a quota error keeps the work done; snapshot times are stored in UTC; interrupted thumbnail
  downloads leave no partial file; channel averages leave out the video's own views.
- Channel statistics are kept per fetch (`channel_stats`), and each video snapshot is paired
  with the channel snapshot nearest in time, so old snapshots no longer train on today's
  subscriber count. Existing databases are backfilled from their stored channel row.
- Subscribers are estimated at each video's publish time from the channel's growth between
  snapshots (bounded extrapolation outside them), matching what a creator enters before
  publishing; `subscribers_at_fetch` keeps the measured count.
- Season features: time of year (cyclic) and a flag for publishing within 3 days of a
  widely observed fixed-date holiday (New Year, Valentine's, Halloween, Christmas). Part of
  the "publish time" group in explanations. Models saved before this need retraining.
- Tools: publish-time grids skip hours a daylight-saving jump removes; plans recorded by
  another version still load; booleans from the database parse correctly.
- Interfaces: the API bounds tag lengths, dates, counts and hours, and answers an unknown
  timezone with a 422; CLI flags override `--config`; bad timezones and a non-views model in
  `track` give clean errors; `update` works before a model exists; the app reloads a retrained
  model and labels likes/comments models correctly.

**Legacy**
- `legacy/` keeps v1 unchanged, plus a corrected copy (`projectFinal_fixed.py`, from PR #2):
  no validation on the test set, the right category search parameter, a separate numeric
  subscriber input, log1p targets, baselines and a split by channel. `README_fixed.md`
  describes the code accurately; the original docs claim a Transformer with GloVe embeddings.

## 1.0 (legacy)

The original course project: BiLSTM models over title, description, channel name and category
in a single script. See `legacy/README.md`.
