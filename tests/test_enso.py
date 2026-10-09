import pandas as pd

from croprisk.data import enso

ONI_TEXT = """SEAS  YR   TOTAL   ANOM
  OND 2020  25.50  -1.20
  NDJ 2020  25.40  -1.10
  DJF 2021  25.60  -1.00
  JFM 2021  26.00  -0.80
"""


def test_season_end_and_ndj_year():
    oni = enso.load(ONI_TEXT)
    ends = oni["season_end"].dt.strftime("%Y-%m").tolist()
    assert ends == ["2020-12", "2021-01", "2021-02", "2021-03"]


def test_asof_respects_publication_delay():
    oni = enso.load(ONI_TEXT)
    dates = pd.Series(pd.to_datetime(["2021-02-05", "2021-02-15", "2021-04-30"]))
    out = enso.oni_asof(dates, oni)
    # Jan-ending season is published ~Feb 10: not yet known on Feb 5
    assert out["oni"].tolist() == [-1.2, -1.1, -0.8]
