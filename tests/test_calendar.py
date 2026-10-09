import pandas as pd
import pytest

from croprisk import calendar


def _assign(group, uf, t0):
    df = pd.DataFrame({"crop_group": [group], "uf": [uf], "t0": [pd.Timestamp(t0)]})
    return calendar.assign_windows(df).iloc[0]


@pytest.mark.parametrize(
    "group,uf,t0,ws,safra",
    [
        ("soja", "PR", "2023-06-10", "2023-12-01", 2023),  # pre-contracted months before planting
        ("soja", "PR", "2024-01-10", "2023-12-01", 2023),  # late contract, same season
        ("soja", "MT", "2023-09-20", "2023-11-15", 2023),
        ("milho_2", "PR", "2024-02-05", "2024-03-15", 2023),  # second crop, same safra as soy
        ("inverno", "RS", "2024-04-20", "2024-07-01", 2023),  # winter wheat, same safra
        ("cafe", "MG", "2023-09-15", "2024-06-01", 2023),  # 12-month cover: next frost season
        ("fruta_temperada", "RS", "2023-07-01", "2023-09-01", 2023),
    ],
)
def test_window_assignment(group, uf, t0, ws, safra):
    w = _assign(group, uf, t0)
    assert w["ws"] == pd.Timestamp(ws)
    assert w["safra_year"] == safra


def test_relative_window_for_vegetables():
    w = _assign("hortalicas", "SC", "2023-05-01")
    assert w["ws"] == pd.Timestamp("2023-05-31")
    assert w["wlen"] == calendar.RELATIVE_LENGTH


def test_every_crop_group_has_a_calendar():
    assert set(calendar.CROP_GROUP.values()) <= set(calendar.CALENDAR)
