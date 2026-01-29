import numpy as np
import pandas as pd


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


def time_features(dates, freq='h'):
    """
    Actually, it is creating the day/week/month/year cycle by the date
    e.g.
    dates length = N ('2016-07-01 00:00:00', '2016-07-01 01:00:00',...)
    freq = "h"
    the output shape is (4, N):
        The columns are [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear], and did normalization for it
        A / max - 0.5. the data range is [-0.5, 0.5]
    freq = "d"
        output [DayOfWeek, DayOfMonth, DayOfYear]

    """
    time_func_list = [DayOfWeek(), DayOfMonth(), DayOfYear()]
    result_list = []
    for time_func in time_func_list:
        time_num = time_func(dates)
        result_list.append(time_num)
    result_list = np.vstack(result_list)  #
    return result_list
