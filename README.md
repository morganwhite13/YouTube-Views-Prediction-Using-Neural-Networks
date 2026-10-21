# YouTube Views Prediction Using Neural Networks — v2

[![CI](https://github.com/morganwhite13/YouTube-Views-Prediction-Using-Neural-Networks/actions/workflows/ci.yml/badge.svg)](https://github.com/morganwhite13/YouTube-Views-Prediction-Using-Neural-Networks/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Predict how many views a YouTube video will get from things a creator knows **before
publishing**: title, description, tags, category, length, upload time and the channel's size and
recent track record. Instead of one number, you get a **range** (p10 / p50 / p90). You also get
a **growth curve** (views after 1 day, 1 week, 1 year) and an **offline A/B test** for
alternative titles.

> **Looking for the original project?** v1 (the COMP 3106 course project) is preserved unchanged
> in [`legacy/`](legacy/), including its README, docs and `projectFinal.py`. If you have the
> JSON datasets v1 produced, you can import them into v2 with `ytviews import-legacy`.

![Streamlit app](docs/images/app_predict.png)

## What's new in v2

| | v1 (`legacy/projectFinal.py`) | v2 (`src/ytviews/`) |
|---|---|---|
| **Target** | raw view count, MSE loss | `log1p(views)`, quantile (pinball) loss → p10/p50/p90 |
| **Uncertainty** | none | quantile heads that can't cross + conformal calibration (≈80% of outcomes land in the p10–p90 band) |
| **Text** | `Embedding(100000, 300)` trained from scratch per field (~30M params each) on a few hundred videos | pretrained sentence-transformer embeddings (offline TF-IDF+SVD fallback) |
| **Video age** | `today − publishedAt` measured at *training* time, so cached data silently aged | age measured at the moment views were fetched; repeat fetches build growth curves |
| **Channel signal** | subscriber count | subscribers, channel size, and a *past-only* prior from the channel's earlier uploads (the strongest signal for creators) |
| **Network** | separate BiLSTMs over title / description / channel name / category id | wide & deep: per-input towers, channel context gates the text representation, residual fusion, monotone quantile head, seed ensemble |
| **Evaluation** | random split, tokenizer/scaler fit before splitting, no baselines | time / unseen-channel / random splits, everything fit on train only, compared against 4 baselines, permutation importance |
| **Data collection** | 2 API calls per video, `search.list` for uploads (100 units) | 50 videos per call, uploads playlist (1 unit), retries, quota budget, SQLite with snapshots |
| **Usability** | edit the script, results printed | `ytviews` CLI, Streamlit app, saved models, tests + CI |

There's also an offline `demo` command and a synthetic data generator, so the whole pipeline
runs without an API key.

## Quick start

```bash
git clone https://github.com/morganwhite13/YouTube-Views-Prediction-Using-Neural-Networks.git
cd YouTube-Views-Prediction-Using-Neural-Networks
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"        # or just `pip install -e .` for the core

# 1. See everything work offline on synthetic data (≈1 min on a laptop CPU)
ytviews demo

# 2. Explore it in the browser
YTVIEWS_MODEL=models/demo streamlit run app/streamlit_app.py
```

### With real YouTube data

```bash
cp .env.example .env               # then paste your YouTube Data API v3 key into .env

ytviews collect --seed-channels                     # v1's 23 channels, ~50 uploads each
ytviews collect --trending --region US --region GB  # trending videos + their back-catalogues
ytviews collect --query "minecraft" --query "cooking" --videos-per-channel 30
ytviews stats

ytviews train                      # trains, evaluates vs baselines, saves models/latest + report
ytviews refresh                    # later: re-fetch stats → growth-curve snapshots → retrain
```

The default daily API quota is 10,000 units. With batching, `collect --seed-channels` costs
roughly 50 units, and a `search` query costs 100. `--quota-budget` stops collection before you
hit the limit.

### Making predictions

```bash
ytviews predict --title "I Survived 100 Days in Minecraft" --category 20 \
    --subscribers 250000 --recent-views 180000,95000,240000,130000 --duration 900

# or let it look up your channel's stats and recent uploads:
ytviews predict --title "I Survived 100 Days in Minecraft" --channel-id UCxxxxxxxxxxxx

# Which title should I use?
ytviews compare "I Survived 100 Days in Minecraft" "Minecraft Hardcore Ep. 12" \
    "$1 vs $10,000 Minecraft Build" --subscribers 250000 --category 20
```

Output from the synthetic demo model:

```text
"I Survived 100 Days in Minecraft" after 7 days:
  ~166.3K views
  80% range: 74.0K - 390.5K

Expected trajectory:
  day    1:  131.2K  (58.5K - 304.9K)
  day    7:  166.3K  (74.0K - 390.5K)
  day   30:  204.7K  (90.8K - 485.6K)
  day  365:  273.8K  (121.3K - 658.0K)

1.  177.4K  (100% of best)  I Survived 100 Days in Minecraft
2.  115.4K  ( 65% of best)  $1 vs $10,000 Minecraft Build
3.   88.2K  ( 50% of best)  Minecraft Hardcore Ep. 12
```

From Python:

```python
from ytviews.predictor import Predictor, VideoInput

p = Predictor.load("models/latest")
video = VideoInput(
    title="I Survived 100 Days in Minecraft",
    category_id="20",
    subscribers=250_000,
    recent_video_views=[180_000, 95_000, 240_000],
)
p.predict(video)  # [Prediction(low=..., median=..., high=..., coverage=0.8)]
p.trajectory(video)  # DataFrame: days → low / median / high
p.compare_titles(video, ["Title A", "Title B"])
```

## How it works

```
                 ┌──────────────── features (fit on training rows only) ────────────────┐
 title ─────────►│ sentence embedding (MiniLM, 384-d)  │ or TF-IDF word+char n-grams → SVD │
 description+tags│ sentence embedding                  │                                    │
 category ──────►│ learned embedding                                                       │
 channel ───────►│ log subscribers, channel size, avg views/video (excluding this video)    │
 past uploads ──►│ median / mean / max / spread of log views of the channel's last 10      │
 timing ────────►│ log age at fetch time, hour & weekday (cyclic)                          │
 format/style ──►│ duration, Shorts, captions, HD, title length/caps/emoji/"vs"/"$"/"?"... │
                 └───────────────────────────────────────────────────────────────────────────┘
                                                   │
          ┌────────── deep ──────────┐             │            ┌─── wide ───┐
          │ text towers ─► × gate(channel context)│            │  linear on │
          │ numeric+category towers ─► concat ─► 2 residual blocks          │  all inputs│
          └───────────────────┬───────────────────┘            └─────┬──────┘
                              └──────────────► + ◄──────────────────┘
                                               ▼
                    monotone quantile head: p10 ≤ p50 ≤ p90  (of log1p views)
```

* **Gating:** the channel context scales the text representation, so the model can learn
  "this kind of title works for *this kind of channel*". This is a lighter version of v1's
  planned feature-interaction layer.
* **Missing-context augmentation:** during training, 30% of videos also appear with their
  channel history removed. That way the model still predicts well when you only know a
  subscriber count.
* **Ensemble + early stopping:** 5 networks trained on different validation splits are
  averaged. Their held-out errors set the conformal widening of the p10–p90 band.
* **Leakage guards:** likes and comments are never inputs, because they're outcomes. The
  channel prior uses only *earlier* uploads, and the channel's average excludes the video being
  predicted.

## Results

`ytviews train` writes `report.md`, `report.png`, `test_predictions.csv` and `meta.json` next to
the model. The table below is from **`ytviews demo` (synthetic data, time-based split)**, so it
shows the machinery working, not real-world accuracy. Run `ytviews train` on collected data for
real numbers.

| model | MAE (log) | R² (log) | median factor error | within 2× | p10–p90 coverage |
|---|---|---|---|---|---|
| **neural net (ours)** | **0.551** | 0.834 | **1.56×** | **70%** | 80% |
| LightGBM (all features) | 0.561 | 0.836 | 1.60× | 68% | – |
| ridge (all features) | 0.582 | 0.821 | 1.65× | 66% | – |
| channel heuristic | 0.659 | 0.775 | 1.79× | 61% | – |
| global median | 1.376 | −0.016 | 3.30× | 32% | – |

*Median factor error 1.56× = the typical prediction is within ~1.6× of the true view count.*

![Report](docs/images/report.png)

On the synthetic data the network recovers what was planted: channel size first, then channel
history, then title wording. For comparison, v1 reported a log-scale MAE of 1.73 on 552
videos, with a random split and no baseline. That number isn't comparable to the table above
(different data and split), but `ytviews train --split random` reproduces v1's setup if you
want a like-for-like run.

## Commands

| command | what it does |
|---|---|
| `ytviews collect` | `--channel ID`, `--seed-channels`, `--trending [--region]`, `--query TEXT` |
| `ytviews refresh` | re-fetch stats for every stored video (adds snapshots) |
| `ytviews import-legacy FILE...` | import v1 JSON caches |
| `ytviews stats` | dataset summary |
| `ytviews train` | `--split time\|channel\|random`, `--encoder auto\|sbert\|tfidf`, `--ensemble N` |
| `ytviews predict` / `compare` | single prediction + trajectory / rank candidate titles |
| `ytviews report` | print the saved evaluation report |
| `ytviews synth` / `demo` | synthetic dataset / full offline run |

## Project layout

```
src/ytviews/
  youtube.py     quota-aware YouTube Data API client (batching, retries, parsing)
  collect.py     collection strategies (channels, trending, search, refresh)
  store.py       SQLite storage with per-fetch snapshots; v1 JSON import
  features.py    feature engineering, past-only channel history, FeatureBuilder
  text.py        sentence-transformer / TF-IDF+SVD text encoders
  model.py       ViewsNet (wide & deep, gated fusion, monotone quantile head)
  train.py       ensemble training, early stopping, conformal calibration
  baselines.py   global median, channel heuristic, ridge, LightGBM
  evaluate.py    splits, metrics, permutation importance
  pipeline.py    train → evaluate → report → save
  predictor.py   load a model; predict / compare titles / trajectory
  synthetic.py   synthetic data with a known generating process
  cli.py         `ytviews` command
app/streamlit_app.py   web UI
tests/                 offline test-suite (fake API, synthetic data)
legacy/                the original v1 project, untouched
```

## Development

```bash
pip install -e ".[dev,gbm,app]"
pytest            # ~20 s, fully offline
ruff check . && ruff format --check .
```

## Limitations and next steps

* **Selection bias:** API sampling over-represents active, discoverable channels. Use
  `--split channel` for a more pessimistic estimate of performance on new creators.
* **Snapshot timing:** subscriber counts and the channel prior are measured at fetch time,
  not at the moment each historical video was published.
* **Ideas:** thumbnail embeddings (CLIP), fine-tuning the text encoder end to end,
  per-category growth curves from repeated `refresh` snapshots, SHAP explanations for single
  predictions.

## License & acknowledgements

MIT — see [LICENSE](LICENSE). Built on the YouTube Data API v3, PyTorch, scikit-learn,
sentence-transformers, LightGBM and Streamlit. Original v1 by
[Morgan White](https://github.com/morganwhite13).
