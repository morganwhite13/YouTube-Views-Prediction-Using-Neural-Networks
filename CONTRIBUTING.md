# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gbm,app,api,llm,youtube]"
pre-commit install        # optional: runs ruff on every commit
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
pytest                    # ~1 min, fully offline
```

CI runs the same checks on Python 3.10 and 3.12.

## Ground rules

- **Tests stay offline.** The YouTube API and the Claude API are faked in `tests/conftest.py`
  and `tests/test_suggest.py`; synthetic data comes from `ytviews.synthetic`. A test that needs
  the network or a downloaded model doesn't belong in the default suite.
- **No leakage.** Anything fitted from data (scalers, vocabularies, encoders, blend weights,
  hyperparameters) is fitted on training rows only. Features must be knowable before a video is
  published, plus its age. New features need a test that proves this, like the ones in
  `tests/test_features.py`.
- **Report against baselines.** A modelling change should show its effect on `ytviews train`
  and, ideally, `ytviews backtest`, with the baseline rows alongside.
- **Keep `legacy/` unchanged.** It is the original v1 project, kept for reference.
- **Regenerate, don't hand-edit**, the notebook: `python notebooks/build_walkthrough.py`.

## Project layout

See the "Project layout" section of the README.
