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
* for videos already live, a **forecast from early views** ("60K after 24 hours → ~270K after
  a year"),
* a **channel audit** of which past videos beat or missed expectations,
* the same model for **likes or comments** (`ytviews train --target likes`),
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
| **Thumbnails** | not used | CLIP image embeddings, or offline image statistics (brightness, contrast, colourfulness, layout, and with `pip install 'ytviews[faces]'` face count, size and position) |
| **Explanations** | none (README listed "SHAP/LIME" as future work) | exact Shapley values over feature groups for every prediction |
| **Tuning** | hand-picked constants | `ytviews tune`: random search on an inner validation split |
| **Live videos** | not supported | early-views forecast, calibrated on held-out videos fetched at two ages |
| **Engagement** | listed as future work | `--target likes` / `--target comments` train the same model for engagement |
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

Prefer reading to running? [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) is the
same tour with its outputs saved: data, training, prediction, explanation, title comparison,
early-views forecast and the best-time heatmap.

### With real YouTube data

```bash
cp .env.example .env               # then paste your YouTube Data API v3 key into .env

ytviews doctor --online                             # check key, libraries, data, model
ytviews collect --seed-channels                     # v1's 23 channels, ~50 uploads each
ytviews collect --trending --region US --region GB  # trending videos + their back-catalogues
ytviews collect --query "minecraft" --query "cooking" --videos-per-channel 30
ytviews thumbnails                 # download thumbnails (no API quota)
ytviews shorts                     # flag which videos are Shorts (no API quota)
ytviews stats

ytviews tune --trials 20           # optional: find good hyperparameters → best_config.json
ytviews train --config best_config.json   # train, evaluate vs baselines, save models/latest
ytviews backtest --folds 4         # optional: is accuracy stable over time?
ytviews report --html report.html  # shareable results page
ytviews models                     # compare everything you've trained
ytviews experiments                # every training run so far, even overwritten ones
ytviews refresh                    # later: re-fetch stats → growth-curve snapshots → retrain
ytviews monitor                    # how the saved model does on videos newer than it
```

**Automatic collection:** add a `YOUTUBE_API_KEY` repository secret and the
[Collect data](.github/workflows/collect.yml) workflow refreshes the dataset every Monday and
Thursday. It keeps the database on a separate `ytviews-data` branch. Regular refreshes are what
make `ytviews track` calibrated, because they record each video at several ages. You can also
run it by hand from the Actions tab, optionally with trending videos and thumbnails.

The default daily API quota is 10,000 units. With batching, `collect --seed-channels` costs
roughly 50 units, and a `search` query costs 100. `--quota-budget` stops collection before you
hit the limit, and channels collected in the last day are skipped (`--skip-fresh-days`).

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

Or do all of that at once. `plan` ranks title ideas, your thumbnails and every publish slot in
the coming week. It then predicts the best *combination* together, so interactions count. It
never recommends something worse than your draft:

```bash
ytviews plan --title "Minecraft Hardcore Ep. 12" --subscribers 250000 --category 20 \
    --thumbnail a.jpg --alt-thumbnail b.jpg --alt-thumbnail c.jpg \
    --alt-title "I Survived Hardcore Minecraft" --timezone America/New_York --html plan.html
```

For a video that's already live, and for planning:

```bash
ytviews track --title "I Survived 100 Days in Minecraft" --views 60000 --hours 24 \
    --subscribers 250000 --category 20                   # where will it end up?
ytviews when --title "..." --subscribers 250000 --timezone America/New_York   # best slot
ytviews compare-thumbnails a.jpg b.jpg c.jpg --title "..." --subscribers 250000
ytviews outliers --channel-id UCxxxxxxxxxxxx             # your over/under-performers
ytviews channel-report --channel-id UCxxxxxxxxxxxx --timezone Europe/London  # shareable page,
                                                         # incl. where recent uploads are heading
ytviews batch planned_videos.csv --out predictions.csv   # columns = VideoInput fields
```

Output from the synthetic demo model (`ytviews demo`):

```text
"I Survived 100 Days in Minecraft" after 7 days:
  ~209.8K views
  80% range: 72.8K - 520.8K

Expected trajectory:
  day    7:  209.8K  (72.8K - 520.8K)
  day   90:  266.2K  (87.5K - 696.3K)

1.  219.0K  (100% of best)  I Survived 100 Days in Minecraft
2.  187.4K  ( 86% of best)  $1 vs $10,000 Minecraft Build
3.   91.0K  ( 42% of best)  Minecraft Hardcore Ep. 12

An average video after 7 days: ~112.6K views
  title text         x0.59   ----
  channel size       x1.32   ++
  channel history    x1.23   ++
  ...

After 24 hours the model expected ~166.1K views; you have 60.0K (0.36x behind pace).
Forecast:
  day    7:  193.9K  (154.6K - 249.8K)
  day   90:  246.6K  (196.7K - 317.7K)
  day  365:  269.1K  (214.6K - 346.6K)
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

Endpoints: `/plan`, `/predict`, `/compare`, `/explain`, `/suggest`, `/track`, `/publish-times`,
`/model`, `/health`.

Before exposing the API beyond your own machine (the Docker image listens on all interfaces):

* set `YTVIEWS_API_KEY`; every endpoint except `/health` then requires `X-API-Key: <value>`;
* set `YTVIEWS_CORS_ORIGINS=https://your.site` if a web page calls it from the browser;
* leave `YTVIEWS_ALLOW_LLM` unset unless you want callers to spend your Anthropic credits on
  Claude title ideas.

Inputs are size-limited, and uploaded thumbnails must be real images of at most 2 MB. Thumbnails are
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
* **Shorts and languages:** whether a video is a Short comes from `ytviews shorts` when it has
  run (the API doesn't say), otherwise from its length. The video's language is a feature, and
  `--encoder sbert-multi` embeds titles in 50+ languages in one space. Collect non-English
  videos with `ytviews collect --language ''`.
* **Channels without history:** when only a subscriber count is known, the channel's typical
  views are estimated from it, using a relation learned on the training data, instead of being
  filled with the global median. Before this, a subscribers-only prediction barely reacted to
  channel size.
* **Explanations:** with ~10 feature groups, every on/off combination fits in one batch, so the
  Shapley values are exact. The baseline is an average video *of the same age*.
* **Ensemble + early stopping:** 5 networks trained on different validation splits are
  averaged. Their held-out errors set the conformal widening of the p10–p90 band.
* **Leakage guards:** likes and comments are never inputs, because they're outcomes. The
  channel prior uses only *earlier* uploads, and the channel's average excludes the video being
  predicted.

## Results

`ytviews train` writes `report.md`, `report.png`, `MODEL_CARD.md` (intended use, data,
scores, limitations), `test_predictions.csv` and `meta.json` next to the model, and `ytviews report --html` turns them into a shareable page. The table below uses
**synthetic data** (`ytviews synth`, time-based split), so it shows the machinery working, not
real-world accuracy. Run `ytviews train` on collected data for real numbers.

| model | MAE (log) | R² (log) | median factor error | within 2× | p10–p90 coverage |
|---|---|---|---|---|---|
| **ytviews (network + LightGBM blend), tuned** | **0.672** | **0.804** | **1.75×** | **59%** | 81% |
| ytviews, default settings | 0.690 | 0.793 | 1.86× | 56% | 82% |
| neural net only, tuned | 0.694 | 0.791 | 1.82× | 57% | 91% |
| ridge (all features) | 0.698 | 0.786 | 1.88× | 56% | – |
| LightGBM (all features) | 0.701 | 0.785 | 1.84× | 56% | – |
| channel heuristic | 0.811 | 0.719 | 2.02× | 50% | – |
| global median | 1.505 | −0.007 | 3.55× | 29% | – |

*Median factor error 1.75× = the typical prediction is within ~1.75× of the true view count.*

The synthetic data includes videos whose views keep growing or fade over months, which no
pre-publication signal can predict; that sets a floor on the error. The network alone and
LightGBM trade places depending on the split, and their blend (40% network here, chosen on an
inner split) beats both. The blend's p10–p90 range is re-calibrated on that inner split too, so
it covers close to 80% of outcomes. `ytviews tune` (16 trials, scored on an inner validation split that never
touches the test set) adds a further gain. On real data, compare against the baselines
yourself: they are in every report for that reason.

A single split can be lucky, so `ytviews backtest` repeats it at four points in time (train on
everything before each cutoff, test on the next slice). On the same synthetic data:

| model | mean MAE (log) over 4 folds | spread (std) | within 2× |
|---|---|---|---|
| **blend (50% network)** | **0.759** | 0.114 | 53% |
| LightGBM | 0.773 | 0.095 | 53% |
| neural net only | 0.811 | 0.139 | 51% |
| channel heuristic | 0.887 | 0.095 | 46% |

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
| `ytviews shorts` | detect which stored videos are Shorts via youtube.com (no quota) |
| `ytviews export FILE` | dataset to `.csv` / `.parquet`, optionally with engineered features |
| `ytviews stats` | dataset summary |
| `ytviews doctor` | check API key, optional libraries, data and model (`--online` tests the key) |
| `ytviews tune` | hyperparameter search → `best_config.json` |
| `ytviews backtest` | rolling-origin evaluation over several time cutoffs |
| `ytviews models` | trained models side by side |
| `ytviews experiments` | the experiment log: settings and scores of every training run |
| `ytviews train` | `--split time\|channel\|random`, `--encoder auto\|sbert\|sbert-multi\|tfidf`, `--image-encoder auto\|clip\|stats\|none`, `--config` |
| `ytviews plan` | everything at once for a draft: best title, thumbnail and time, predicted together (`--html`) |
| `ytviews predict` / `compare` | single prediction + trajectory / rank candidate titles |
| `ytviews explain` | what pushes a prediction up or down |
| `ytviews track` | forecast a live video from its views so far |
| `ytviews when` | best weekday and hour to publish |
| `ytviews compare-thumbnails` | rank alternative thumbnails |
| `ytviews outliers` | stored videos that beat or missed expectations |
| `ytviews channel-report` | one-page HTML report for a channel: uploads vs expectations, where recent uploads are heading, best times |
| `ytviews update` | routine upkeep: refresh, Shorts + thumbnails, monitor, retrain with the previous settings if degraded (`--retrain auto\|always\|never`), rewrite reports |
| `ytviews monitor` | accuracy on videos published after the model was trained; says when to retrain |
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
  monitor.py     accuracy on videos newer than the model
  channel_report.py  per-channel HTML report
  planner.py     plan a video: titles, thumbnails, publish slot combined
  cache.py       on-disk cache for sentence-transformer / CLIP embeddings
  suggest.py     rule-based + Claude title ideas, ranked by the model
  tune.py        hyperparameter search
  api.py         FastAPI service
  report_html.py shareable HTML report
  synthetic.py   synthetic data with a known generating process
  cli.py         `ytviews` command
app/streamlit_app.py   web UI
notebooks/             executed walkthrough (rebuild: python notebooks/build_walkthrough.py)
.github/workflows/     CI (Python 3.10 + 3.12) and the scheduled data collection
Dockerfile             API image (also runs the web UI)
tests/                 offline test-suite (fake API, synthetic data)
legacy/                the original v1 project, untouched
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the ground rules (offline tests, no leakage,
baselines in every comparison).

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

See [CHANGELOG.md](CHANGELOG.md) for what changed between v1 and v2.

## License & acknowledgements

MIT — see [LICENSE](LICENSE). Built on the YouTube Data API v3, PyTorch, scikit-learn,
sentence-transformers, LightGBM and Streamlit. Original v1 by
[Morgan White](https://github.com/morganwhite13).
