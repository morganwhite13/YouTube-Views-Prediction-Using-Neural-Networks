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
pytest                    # ~4 min, fully offline
```

`tests/test_properties.py` feeds random valid inputs (any text, timezone, date or count) through
prediction, publish-time grids and the API. For a deeper search before a release run
`YTVIEWS_PROPERTY_EXAMPLES=500 pytest tests/test_properties.py`.

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

## Documentation site

The site (MkDocs) reuses the README, CHANGELOG and this file, generates the command-line
reference from the Typer app and the Python reference from docstrings. Preview it with:

```bash
pip install -e ".[docs]"
typer ytviews.cli utils docs --name ytviews --title "Command line" --output docs/cli.md
mkdocs serve
```

`.github/workflows/docs.yml` builds it with `--strict` on every pull request and publishes it to
GitHub Pages from `main` once Pages is set to "GitHub Actions" and the repository variable
`PAGES_ENABLED` is `true`.

## Releasing

Releases go to PyPI as `ytviews` from `.github/workflows/release.yml`, with trusted publishing
(no API token).

One-time setup:

1. On pypi.org, under *Your projects → Publishing*, add a pending trusted publisher: project
   `ytviews`, owner `morganwhite13`, repository `YouTube-Views-Prediction-Using-Neural-Networks`,
   workflow `release.yml`, environment `pypi`.
2. In the GitHub repository settings, create an environment named `pypi` (optionally with
   required reviewers, so every release needs a click).

Each release:

1. Set `__version__` in `src/ytviews/__init__.py` and move the CHANGELOG's "unreleased" heading
   to that version.
2. Merge to `main`, then tag and push: `git tag v2.0.0 && git push origin v2.0.0`.
3. The workflow builds, checks the tag matches the version, installs the wheel, runs
   `ytviews --help` and publishes.

