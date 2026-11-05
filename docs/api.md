# Python API

Everything the command line does is available from Python. The usual entry point is
`Predictor.load` and a `VideoInput`:

```python
from ytviews.predictor import Predictor, VideoInput

model = Predictor.load("models/latest")
draft = VideoInput(title="I Survived 100 Days in Minecraft", subscribers=250_000)
print(model.predict(draft)[0])  # Prediction(low=..., median=..., high=..., coverage=0.8)
```

## Predicting

::: ytviews.predictor.VideoInput

::: ytviews.predictor.Prediction

::: ytviews.predictor.Predictor

## Planning a video

::: ytviews.planner.make_plan

::: ytviews.planner.schedule

::: ytviews.planner.calendar

::: ytviews.planner.thumbnail_tips

## Live videos

::: ytviews.nowcast.track

## Channels

::: ytviews.analysis.outliers

::: ytviews.digest.weekly_digest

::: ytviews.competitors.compare_channels

## Keeping a model honest

::: ytviews.monitor.monitor

::: ytviews.compare.compare_models

::: ytviews.compare.gate_candidate

## Data

::: ytviews.store.VideoStore

::: ytviews.pipeline.run_experiment
