import numpy as np
import pandas as pd

from ytviews.features import FEATURE_GROUPS, NUMERIC_FEATURES, numeric_frame
from ytviews.predictor import VideoInput


def _frame(*dates: str) -> pd.DataFrame:
    rows = [VideoInput(title="x", published_at=pd.Timestamp(d).to_pydatetime()).to_row()
            for d in dates]  # fmt: skip
    return numeric_frame(pd.DataFrame(rows))


def test_time_of_year_is_continuous_across_new_year():
    num = _frame("2030-12-31T23:00:00Z", "2031-01-01T01:00:00Z", "2030-07-01T00:00:00Z")
    year_end, new_year, summer = num[["doy_sin", "doy_cos"]].to_numpy()
    assert np.linalg.norm(year_end - new_year) < 0.01
    assert np.linalg.norm(year_end - summer) > 1.9  # opposite side of the year


def test_near_holiday_flags_the_days_around_fixed_holidays():
    num = _frame(
        "2030-12-22T12:00:00Z",  # 2 days before Christmas Eve
        "2031-01-03T12:00:00Z",  # 2 days after New Year's Day
        "2030-10-29T12:00:00Z",  # 2 days before Halloween
        "2030-07-15T12:00:00Z",
        "2030-03-20T12:00:00Z",
    )
    assert num["near_holiday"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_season_belongs_to_the_publish_time_group():
    assert {"doy_sin", "doy_cos", "near_holiday"} <= set(FEATURE_GROUPS["publish time"])
    assert {"doy_sin", "doy_cos", "near_holiday"} <= set(NUMERIC_FEATURES)
