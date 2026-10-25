# YouTube Views Prediction Using Neural Networks — v2

[![CI](https://github.com/morganwhite13/YouTube-Views-Prediction-Using-Neural-Networks/actions/workflows/ci.yml/badge.svg)](https://github.com/morganwhite13/YouTube-Views-Prediction-Using-Neural-Networks/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Predict how many views a YouTube video will get from things a creator knows **before
publishing**: title, description, tags, thumbnail, category, length, upload time and the
channel's size and recent track record. Instead of one number, you get a **range** (p10 / p50 /
p90). You also get:

* a **growth curve** (views after 1 day, 1 week, 1 year),
* an **explanation** of what pushes the prediction up or down ("your title ×0.4, your channel
  history ×1.8"),
* **title ideas** generated from your draft (optionally by Claude) and ranked by the model,
* **thumbnail A/B comparison** and the **best weekday and hour** to publish,
* for videos already live, a **forecast from early views** ("60K after 24 hours → ~280K after
  a year"),
* a **channel audit** of which past videos beat or missed expectations,
* a CLI, a web app, an HTTP API, batch CSV predictions and a shareable HTML report.

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
| **Network** | separate BiLSTMs over title / description / channel name / category id | wide & deep: per-input towers, channel context gates the text representation, residual fusion, monotone quantile head, seed ensemble; blended with LightGBM when that helps on an inner split |
| **Evaluation** | random split, tokenizer/scaler fit before splitting, no baselines | time / unseen-channel / random splits, everything fit on train only, compared against 4 baselines, permutation importance |
| **Data collection** | 2 API calls per video, `search.list` for uploads (100 units) | 50 videos per call, uploads playlist (1 unit), retries, quota budget, SQLite with snapshots |
| **Thumbnails** | not used | CLIP image embeddings, or offline image statistics (brightness, contrast, colourfulness, layout) |
| **Explanations** | none (README listed "SHAP/LIME" as future work) | exact Shapley values over feature groups for every prediction |
| **Tuning** | hand-picked constants | `ytviews tune`: random search on an inner validation split |
| **Live videos** | not supported | early-views forecast, calibrated on held-out videos fetched at two ages |
| **Usability** | edit the script, results printed | `ytviews` CLI, Streamlit app, FastAPI service + Dockerfile, HTML report, tests + CI |

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

ytviews doctor --online                             # check key, libraries, data, model
ytviews collect --seed-channels                     # v1's 23 channels, ~50 uploads each
ytviews collect --trending --region US --region GB  # trending videos + their back-catalogues
ytviews collect --query "minecraft" --query "cooking" --videos-per-channel 30
ytviews thumbnails                 # download thumbnails (no API quota)
ytviews stats

ytviews tune --trials 20           # optional: find good hyperparameters → best_config.json
ytviews train --config best_config.json   # train, evaluate vs baselines, save models/latest
ytviews backtest --folds 4         # optional: is accuracy stable over time?
ytviews report --html report.html  # shareable results page
ytviews models                     # compare everything you've trained
ytviews refresh                    # later: re-fetch stats → growth-curve snapshots → retrain
```

The default daily API quota is 10,000 units. With batching, `collect --seed-channels` costs
roughly 50 units, and a `search` query costs 100. `--quota-budget` stops collection before you
hit the limit.

### Making predictions

```bash
ytviews predict --title "I Survived 100 Days in Minecraft" --category 20 \
    --subscribers 250000 --recent-views 180000,95000,240000,130000 --duration 900 \
    --thumbnail thumb.jpg

# or let it look up your channel's stats and recent uploads:
ytviews predict --title "I Survived 100 Days in Minecraft" --channel-id UCxxxxxxxxxxxx

ytviews compare "I Survived 100 Days in Minecraft" "Minecraft Hardcore Ep. 12" \
    "$1 vs $10,000 Minecraft Build" --subscribers 250000 --category 20   # which title?
ytviews explain --title "Minecraft Hardcore Ep. 12" --subscribers 250000 --category 20
ytviews suggest --title "Minecraft Hardcore Ep. 12" --subscribers 250000 --category 20 [--llm]
```

For a video that's already live, and for planning:

```bash
ytviews track --title "I Survived 100 Days in Minecraft" --views 60000 --hours 24 \
    --subscribers 250000 --category 20                   # where will it end up?
ytviews when --title "..." --subscribers 250000 --timezone America/New_York   # best slot
ytviews compare-thumbnails a.jpg b.jpg c.jpg --title "..." --subscribers 250000
ytviews outliers --channel-id UCxxxxxxxxxxxx             # your over/under-performers
ytviews batch planned_videos.csv --out predictions.csv   # columns = VideoInput fields
```

Output from the synthetic demo model (`ytviews demo`):

```text
"I Survived 100 Days in Minecraft" after 7 days:
  ~267.5K views
  80% range: 54.8K - 1.1M

Expected trajectory:
  day    1:  191.8K  (40.7K - 749.8K)
  day   14:  296.2K  (59.7K - 1.3M)
  day  180:  379.3K  (71.7K - 1.8M)

1.  251.6K  (100% of best)  I Survived 100 Days in Minecraft
2.  250.3K  ( 99% of best)  $1 vs $10,000 Minecraft Build
3.  142.6K  ( 57% of best)  Minecraft Hardcore Ep. 12

An average video after 7 days: ~78.8K views
  channel size       x1.51   +++
  description stats  x1.30   ++
  ...

After 24 hours the model expected ~178.6K views; you have 60.0K (0.34x behind pace).
Forecast:
  day    7:  176.7K  (138.4K - 223.6K)
  day   90:  243.6K  (190.8K - 308.3K)
  day  365:  281.9K  (220.9K - 356.8K)
```

`suggest` rewrites your draft with common title patterns and, with `--llm`, also asks Claude
(`claude-opus-5-5`, needs `ANTHROPIC_API_KEY`) for ideas that stay true to the video. The model
then ranks every candidate. It can't judge whether a title is honest, so check the winner.

From Python:

```python
from ytviews.predictor import Predictor, VideoInput

p = Predictor.load("models/latest")
video = VideoInput(
    title="I Survived 100 Days in Minecraft",
    category_id="20",
    subscribers=250_000,
    recent_video_views=[180_000, 95_000, 240_000],
    thumbnail_path="thumb.jpg",
)
p.predict(video)  # [Prediction(low=..., median=..., high=..., coverage=0.8)]
p.trajectory(video)  # DataFrame: days → low / median / high
p.compare_titles(video, ["Title A", "Title B"])
p.explain(video).effects  # group, log_effect, factor
```

### HTTP API and Docker

```bash
ytviews serve --model models/latest          # http://127.0.0.1:8000/docs
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"title": "I Survived 100 Days in Minecraft", "subscribers": 250000, "category_id": "20"}'

docker build -t ytviews .
docker run -p 8000:8000 -v "$PWD/models:/app/models" ytviews
```

Endpoints: `/predict`, `/compare`, `/explain`, `/suggest`, `/track`, `/publish-times`, `/model`,
`/health`. Thumbnails are
sent as `thumbnail_base64`.

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
 thumbnail ─────►│ CLIP image embedding  │ or brightness/contrast/colour/layout statistics  │
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
  channel history removed, and 15% without their thumbnail. That way the model still predicts
  well when you only know a subscriber count. These copies get a lower loss weight and are kept
  out of early stopping and calibration.
* **Blend:** the network's median is mixed with a LightGBM model when an inner split of the
  training data says that helps (the two make different mistakes). The network's calibrated
  range is kept, shifted to the blended median.
* **Early-views forecast:** how much of an early lead or lag persists is measured on held-out
  videos that were fetched at two ages (`ytviews refresh` creates these). Early surprises
  count for less than late ones.
* **Embedding cache:** pretrained text and image embeddings are cached on disk by content hash
  (`~/.cache/ytviews`; `YTVIEWS_CACHE=0` turns it off), so retraining, tuning and backtests
  don't re-encode the same titles and thumbnails.
* **Explanations:** with ~10 feature groups, every on/off combination fits in one batch, so the
  Shapley values are exact. The baseline is an average video *of the same age*.
* **Ensemble + early stopping:** 5 networks trained on different validation splits are
  averaged. Their held-out errors set the conformal widening of the p10–p90 band.
* **Leakage guards:** likes and comments are never inputs, because they're outcomes. The
  channel prior uses only *earlier* uploads, and the channel's average excludes the video being
  predicted.

## Results

`ytviews train` writes `report.md`, `report.png`, `test_predictions.csv` and `meta.json` next to
the model, and `ytviews report --html` turns them into a shareable page. The table below uses
**synthetic data** (`ytviews synth`, time-based split), so it shows the machinery working, not
real-world accuracy. Run `ytviews train` on collected data for real numbers.

| model | MAE (log) | R² (log) | median factor error | within 2× | p10–p90 coverage |
|---|---|---|---|---|---|
| **ytviews (network + LightGBM blend), tuned** | **0.688** | **0.751** | **1.69×** | **61%** | 86% |
| LightGBM (all features) | 0.697 | 0.736 | 1.71× | 60% | – |
| ytviews, default settings | 0.701 | 0.742 | 1.72× | 59% | 88% |
| neural net only, tuned | 0.718 | 0.740 | 1.77× | 57% | 86% |
| ridge (all features) | 0.741 | 0.727 | 1.89× | 54% | – |
| channel heuristic | 0.834 | 0.665 | 2.04× | 49% | – |
| global median | 1.536 | −0.174 | 3.51× | 28% | – |

*Median factor error 1.7× = the typical prediction is within ~1.7× of the true view count.*

The synthetic data includes videos whose views keep growing or fade over months, which no
pre-publication signal can predict; that sets a floor on the error. The network alone and
LightGBM trade places depending on the split, and their blend (50/50 here, chosen on an inner
split) beats both. `ytviews tune` (16 trials, scored on an inner validation split that never
touches the test set) adds a further gain. On real data, compare against the baselines
yourself: they are in every report for that reason.

A single split can be lucky, so `ytviews backtest` repeats it at four points in time (train on
everything before each cutoff, test on the next slice). On the same synthetic data:

| model | mean MAE (log) over 4 folds | spread (std) | within 2× |
|---|---|---|---|
| **blend (50% network)** | **0.779** | 0.085 | **55%** |
| LightGBM | 0.800 | 0.063 | 54% |
| neural net only | 0.812 | 0.118 | 51% |
| channel heuristic | 0.916 | 0.068 | 46% |

The blend wins on average. The network alone is the most variable: it is weakest in the
earliest fold, where it has the least training data, and best in a later one.

![Report](docs/images/report.png)

On the synthetic data the network recovers what was planted: channel size first, then channel
history, then title wording and thumbnail design. It also finds the planted publish-time
effect (weekends +7%, afternoons +30% vs nights, slightly weaker than planted). For comparison, v1 reported a log-scale MAE
of 1.73 on 552 videos, with a random split and no baseline. That number isn't comparable to the
table above (different data and split), but `ytviews train --split random` reproduces v1's
setup if you want a like-for-like run.

## Commands

| command | what it does |
|---|---|
| `ytviews collect` | `--channel ID`, `--seed-channels`, `--trending [--region]`, `--query TEXT` |
| `ytviews refresh` | re-fetch stats for every stored video (adds snapshots) |
| `ytviews import-legacy FILE...` | import v1 JSON caches |
| `ytviews thumbnails` | download thumbnails for stored videos (no quota) |
| `ytviews stats` | dataset summary |
| `ytviews doctor` | check API key, optional libraries, data and model (`--online` tests the key) |
| `ytviews tune` | hyperparameter search → `best_config.json` |
| `ytviews backtest` | rolling-origin evaluation over several time cutoffs |
| `ytviews models` | trained models side by side |
| `ytviews train` | `--split time\|channel\|random`, `--encoder auto\|sbert\|tfidf`, `--image-encoder auto\|clip\|stats\|none`, `--config` |
| `ytviews predict` / `compare` | single prediction + trajectory / rank candidate titles |
| `ytviews explain` | what pushes a prediction up or down |
| `ytviews track` | forecast a live video from its views so far |
| `ytviews when` | best weekday and hour to publish |
| `ytviews compare-thumbnails` | rank alternative thumbnails |
| `ytviews outliers` | stored videos that beat or missed expectations |
| `ytviews batch` | predictions for every row of a CSV |
| `ytviews suggest` | generate and rank title ideas (`--llm` adds Claude's) |
| `ytviews serve` | HTTP API |
| `ytviews report` | print the saved evaluation report, or `--html FILE` for a web page |
| `ytviews synth` / `demo` | synthetic dataset / full offline run |

## Project layout

```
src/ytviews/
  youtube.py     quota-aware YouTube Data API client (batching, retries, parsing)
  collect.py     collection strategies (channels, trending, search, refresh)
  store.py       SQLite storage with per-fetch snapshots; v1 JSON import
  features.py    feature engineering, past-only channel history, FeatureBuilder
  text.py        sentence-transformer / TF-IDF+SVD text encoders
  images.py      CLIP / image-statistics thumbnail encoders
  model.py       ViewsNet (wide & deep, gated fusion, monotone quantile head)
  train.py       ensemble training, early stopping, conformal calibration
  baselines.py   global median, channel heuristic, ridge, LightGBM
  evaluate.py    splits, metrics, permutation importance
  pipeline.py    train → evaluate → report → save
  predictor.py   load a model; predict / compare titles / trajectory
  explain.py     exact group-Shapley explanations
  nowcast.py     early-views forecast and its calibration
  analysis.py    channel outliers, batch CSV predictions
  backtest.py    rolling-origin evaluation
  cache.py       on-disk cache for sentence-transformer / CLIP embeddings
  suggest.py     rule-based + Claude title ideas, ranked by the model
  tune.py        hyperparameter search
  api.py         FastAPI service
  report_html.py shareable HTML report
  synthetic.py   synthetic data with a known generating process
  cli.py         `ytviews` command
app/streamlit_app.py   web UI
Dockerfile             API image (also runs the web UI)
tests/                 offline test-suite (fake API, synthetic data)
legacy/                the original v1 project, untouched
```

## Development

```bash
pip install -e ".[dev,gbm,app,api,llm]"
pytest            # ~1 min, fully offline (the Claude API and YouTube API are faked)
ruff check . && ruff format --check .
```

## Limitations and next steps

* **Selection bias:** API sampling over-represents active, discoverable channels. Use
  `--split channel` for a more pessimistic estimate of performance on new creators.
* **Snapshot timing:** subscriber counts and the channel prior are measured at fetch time,
  not at the moment each historical video was published.
* **Not yet run on real data in this repo:** the numbers above are synthetic. Pretrained
  encoders (sentence-transformers, CLIP) are wired in and tested with stand-ins, but the
  bundled results use the offline fallbacks.
* **Outliers are in-sample** when the model was trained on the same videos; for a cleaner
  audit, train with `--no-refit` so the saved model hasn't seen the newest uploads.
* **Ideas:** fine-tuning the text encoder end to end, thumbnail text (OCR) features,
  per-category growth curves.

## License & acknowledgements

MIT — see [LICENSE](LICENSE). Built on the YouTube Data API v3, PyTorch, scikit-learn,
sentence-transformers, LightGBM and Streamlit. Original v1 by
[Morgan White](https://github.com/morganwhite13).
