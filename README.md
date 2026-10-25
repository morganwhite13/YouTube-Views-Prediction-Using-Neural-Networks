# YouTube Views Prediction Using Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Predicts how many views a YouTube video will get from its title, description, channel name, category, subscriber count and age. The project collects its own data through the YouTube Data API, preprocesses the text with NLTK, and builds a multi-input Keras model in six stages, adding one input at a time, with a bidirectional LSTM for each text field.

Built for COMP 3106 (Introduction to Artificial Intelligence) at Carleton University. A full write-up is on my portfolio: [YouTube Views Prediction](https://morganwhite13.github.io/projects/youtube-predictor/).

## How it works

### 1. Data collection

`projectFinal.py` samples videos through the YouTube Data API v3 in several ways, so the dataset isn't dominated by one kind of channel:

| Function | What it collects |
| :--- | :--- |
| `getRandomVideos()` | Random search results |
| `getPopularVideos()` | Trending videos |
| `getCategoryVideos()` | Videos from specific categories |
| `getChannelsVideos()` | Videos from a list of channels |
| `getPopularChannelsVideos()` | Videos from popular channels |
| `getRandomChannelsVideos()` | A batch of videos from each of several random channels |
| `getAllVideos()`, `combineDatasets()` | Combined sampling, and a merge of every cached dataset with duplicates removed |

For each video it keeps the title, description, channel name, subscriber count, category ID, publish date and view count. Only English videos with more than 1,000 views are kept.

Every collection function caches its results in a JSON file and reloads it on later runs, which keeps experiments within the API's daily quota.

### 2. Preprocessing

- Text fields are lowercased, stripped of punctuation, tokenized with NLTK and filtered for stop words (`preprocessText()`).
- Each field gets its own Keras `Tokenizer` and is padded to its longest example.
- Subscriber count and days since publication are standardized with `StandardScaler`.

### 3. Models

Six model functions each add one input, so the effect of each can be compared:

| Stage | Function | Inputs |
| :--- | :--- | :--- |
| 1 | `neuralTitleModel` | Title |
| 2 | `neuralTitleSubscriberModel` | + subscriber count |
| 3 | `neuralTSDateModel` | + days since publication |
| 4 | `neuralTSDDescriptionModel` | + description |
| 5 | `neuralTSDDChannelModel` | + channel name |
| 6 | `neuralAllModel` | + category |

The final model (`neuralAllModel`):

```
Title ─────────► Embedding(300) ─► BiLSTM(128) ─► GlobalMaxPool ─┐
Description ───► Embedding(300) ─► BiLSTM(128) ─► GlobalMaxPool ─┤
Channel name ──► Embedding(300) ─► BiLSTM(128) ─► GlobalMaxPool ─┤
Category ──────► Embedding(300) ─► BiLSTM(128) ─► GlobalMaxPool ─┼─► Concatenate
Subscribers (scaled) ────────────────────────────────────────────┤      │
Days since publication (scaled) ─────────────────────────────────┘      ▼
                                              Dense(512, ReLU) ─► Dropout(0.3)
                                              Dense(256, ReLU) ─► Dropout(0.3)
                                              Dense(1, linear) ─► log(1 + views)
```

Every model predicts `log(1 + views)`, since view counts are heavily skewed, and converts its predictions back to view counts. Training uses an 80/20 train/test split, the Adam optimizer, mean squared error loss, 10 epochs and a batch size of 32. Each model prints its test-set MSE and MAE in log space.

`baselineModels` gives the neural models something to beat, on the same target, rows and split: predicting the training mean, and a linear regression on log subscribers and log video age. The script runs it before `neuralAllModel`.

### 4. What-if predictions

After training, `neuralAllModel` predicts views for a few made-up videos (for example "I Survived 100 Days in Canada" and "$1 VS $1,000 Water"), then changes one input at a time (description, title, channel name, subscriber count, category, publish date) to show which inputs the model responds to.

## Running it

```bash
pip install -r requirements.txt
```

1. Get a YouTube Data API v3 key from the [Google Cloud Console](https://console.cloud.google.com/apis/library/youtube.googleapis.com).
2. Set `YOUTUBE_API_KEY` to your key, either as an environment variable (`export YOUTUBE_API_KEY=...`) or in a `.env` file next to the script containing `YOUTUBE_API_KEY=...`.
3. Run `python projectFinal.py`.

On first run the script downloads the NLTK data it needs, collects videos with `getRandomChannelsVideos()` (cached to JSON), prints the baselines, then trains and evaluates `neuralAllModel`. To use a different collection method or model stage, uncomment the matching line at the bottom of the file.

## Limitations and next steps

- **No recorded results yet.** The code has been checked end to end on synthetic data with the same columns, but real results need a run with collected data. Compare each model's log-space MAE with the baselines.
- **Predictions far outside the training data can explode.** Because predictions are converted back from log space, an input well beyond anything seen in training (for example 90× the subscribers) can produce an absurdly large view count.
- **The model is large for the data.** Each embedding table is sized for 100,000 words × 300 dimensions (about 120M parameters across the four), trained on a few hundred videos. Pretrained embeddings such as GloVe, or a much smaller vocabulary, would fit the data better.
- **The split leaks across channels.** The random train/test split puts videos from the same channel on both sides, so a split grouped by channel would give a fairer test score. The tokenizers and scalers are also fit on all rows before the split.
- **Future ideas:** a Transformer encoder or pretrained language model for the text, thumbnail features from a CNN, and more data.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- YouTube Data API by Google
- Inspiration: Title Thumbnail View Predictor (Devpost) and YouTube Views Prediction (Kaggle)
- Libraries: TensorFlow, Keras, NLTK, pandas, NumPy, scikit-learn

## Author

**Morgan White**: [GitHub](https://github.com/morganwhite13) · [LinkedIn](https://www.linkedin.com/in/morgan-white-95b245237/) · [Portfolio](https://morganwhite13.github.io)
