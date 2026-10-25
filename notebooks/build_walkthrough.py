"""Build and execute notebooks/walkthrough.ipynb.

    python notebooks/build_walkthrough.py

The notebook is generated from this script so it stays in sync with the package API; the
committed .ipynb includes the outputs of the last run.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
    md(
        "# ytviews walkthrough\n\n"
        "A tour of the view-count predictor, end to end, on the built-in **synthetic** dataset "
        "(so it runs offline, without a YouTube API key). The numbers show the machinery "
        "working; they are not accuracy on real YouTube videos.\n\n"
        "Steps: generate data → train and evaluate → predict → explain → compare titles → "
        "forecast a live video → pick a publish time."
    ),
    code(
        "import tempfile\nfrom pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\n"
        "from ytviews.images import attach_thumbnails, thumbnail_dir_for\n"
        "from ytviews.pipeline import run_experiment\n"
        "from ytviews.predictor import Predictor, VideoInput\n"
        "from ytviews.store import VideoStore\n"
        "from ytviews.synthetic import write_synthetic_db\n"
        "from ytviews.train import TrainConfig\n\n"
        "work = Path(tempfile.mkdtemp())\n"
        "pd.set_option('display.width', 120)"
    ),
    md(
        "## 1. Data\n\n"
        "The synthetic generator plants known effects (channel size and quality, title wording, "
        "thumbnail design, publish time, growth that differs per video) and stores videos "
        "exactly like the real YouTube collector does, including two snapshots per video."
    ),
    code(
        "db = work / 'synthetic.db'\n"
        "print(write_synthetic_db(db, n_channels=80, seed=3))\n"
        "with VideoStore(db) as store:\n"
        "    df = store.load_dataset(all_snapshots=True)\n"
        "df = attach_thumbnails(df, thumbnail_dir_for(db))\n"
        "df[['title', 'category_id', 'subscribers', 'views', 'published_at', 'fetched_at']]"
        ".head()"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
        "ax.hist(np.log10(df['views'].clip(lower=1)), bins=40, color='#2b59c3')\n"
        "ax.set(xlabel='log10(views)', ylabel='snapshots', title='Views span many orders of "
        "magnitude')\nplt.show()"
    ),
    md(
        "## 2. Train and evaluate\n\n"
        "`run_experiment` splits by time (older uploads train, newest test), trains the network "
        "ensemble, picks a network/LightGBM blend on an inner split, compares against baselines "
        "and saves everything."
    ),
    code(
        "model_dir = work / 'model'\n"
        "meta = run_experiment(df, model_dir, text_encoder='tfidf', image_encoder='stats',\n"
        "                      config=TrainConfig(ensemble=2), log=lambda _: None)\n"
        "metrics = pd.DataFrame(meta['metrics']).T[['mae_log', 'median_factor_error', "
        "'within_2x', 'interval_coverage']]\n"
        "metrics.sort_values('mae_log').round(3)"
    ),
    code(
        "imp = pd.Series(meta['importance']).sort_values()\n"
        "imp[imp > 0.001].plot.barh(figsize=(7, 3.5), color='#2b59c3',\n"
        "    title='What the model relies on (MAE increase when shuffled)')\nplt.show()"
    ),
    md("## 3. Predict a planned video"),
    code(
        "p = Predictor.load(model_dir)\n"
        "video = VideoInput(title='I Survived 100 Days in Minecraft', category_id='20',\n"
        "                   subscribers=250_000, duration_s=900,\n"
        "                   recent_video_views=[180_000, 95_000, 240_000, 130_000])\n"
        "pred = p.predict(video)[0]\n"
        "print(f'7 days: ~{pred.median:,.0f} views (80% range {pred.low:,.0f} - {pred.high:,.0f})')"
    ),
    code(
        "traj = p.trajectory(video)\n"
        "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
        "ax.fill_between(traj['days'], traj['low'], traj['high'], color='#a9bde9', alpha=.6,\n"
        "                label='p10-p90')\n"
        "ax.plot(traj['days'], traj['median'], color='#2b59c3', marker='o', label='median')\n"
        "ax.set(xscale='log', xlabel='days after upload', ylabel='cumulative views',\n"
        "       title='Expected growth')\nax.legend()\nplt.show()"
    ),
    md(
        "## 4. Why that number?\n\n"
        "Exact Shapley values over feature groups, relative to an average video of the same age. "
        "The factors multiply out to the prediction."
    ),
    code(
        "ex = p.explain(video)\n"
        "print(f'average video: {ex.baseline_views:,.0f} -> this video: "
        "{ex.predicted_views:,.0f}')\n"
        "ex.effects.round(3)"
    ),
    md("## 5. Which title?"),
    code(
        "titles = ['I Survived 100 Days in Minecraft', 'Minecraft Hardcore Ep. 12',\n"
        "          '$1 vs $10,000 Minecraft Build', 'Weekly Minecraft Update']\n"
        "p.compare_titles(video, titles).round(2)"
    ),
    code(
        "from ytviews.suggest import suggest_titles\n"
        "suggest_titles(p, VideoInput(title='Minecraft Hardcore Ep. 12', category_id='20',\n"
        "                             subscribers=250_000)).head(6).round(2)"
    ),
    md(
        "## 6. Already live: forecast from early views\n\n"
        "How much an early lead or lag persists was calibrated on held-out videos that were "
        "fetched at two ages."
    ),
    code(
        "nc = p.track(video, observed_views=60_000, observed_age_days=1)\n"
        "print(f'pace vs expectation: {nc.vs_expected:.2f}x (expected ~{nc.expected_views:,.0f} "
        "after 1 day)')\nnc.forecast.round(0)"
    ),
    md("## 7. When to publish"),
    code(
        "grid = p.publish_times(video, 'America/New_York')\n"
        "heat = grid.pivot(index='weekday_index', columns='hour', values='median')\n"
        "days = grid.drop_duplicates('weekday_index').set_index('weekday_index')['weekday']\n"
        "fig, ax = plt.subplots(figsize=(9, 3.2))\n"
        "im = ax.imshow(heat.values, aspect='auto', cmap='Blues')\n"
        "ax.set_yticks(range(7), [days[i] for i in heat.index])\n"
        "ax.set_xticks(range(0, 24, 3), range(0, 24, 3))\n"
        "ax.set(xlabel='hour (New York)', title='Predicted views by publish slot')\n"
        "fig.colorbar(im, ax=ax)\nplt.show()"
    ),
    md(
        "## Next: real data\n\n"
        "```bash\n"
        "cp .env.example .env          # add YOUTUBE_API_KEY\n"
        "ytviews doctor --online\n"
        "ytviews collect --seed-channels && ytviews thumbnails\n"
        "ytviews tune && ytviews train --config best_config.json\n"
        "ytviews report --html report.html\n"
        "```"
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=900, kernel_name="python3").execute()
out = HERE / "walkthrough.ipynb"
nbformat.write(nb, out)
print(f"wrote {out}")
