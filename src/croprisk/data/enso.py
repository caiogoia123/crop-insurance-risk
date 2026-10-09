"""Oceanic Nino Index (NOAA CPC): ENSO state known at contract time."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from croprisk import config

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_PATH = config.RAW / "oni.txt"
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
# CPC publishes a season's value early in the following month.
PUBLICATION_DELAY_DAYS = 10


def download(force: bool = False) -> pd.DataFrame:
    config.ensure_dirs()
    if force or not ONI_PATH.exists():
        r = requests.get(ONI_URL, timeout=60)
        r.raise_for_status()
        ONI_PATH.write_text(r.text, encoding="utf-8")
    return load()


def load(text: str | None = None) -> pd.DataFrame:
    """One row per 3-month season with the date its value becomes available."""
    text = ONI_PATH.read_text(encoding="utf-8") if text is None else text
    df = pd.read_csv(io.StringIO(text), sep=r"\s+")
    df.columns = [c.upper() for c in df.columns]
    k = df["SEAS"].map({s: i for i, s in enumerate(SEASONS)})
    last_month = k + 2  # DJF -> Feb ... NDJ -> Jan of the next year (13)
    year = df["YR"] + (last_month > 12)
    month = np.where(last_month > 12, last_month - 12, last_month)
    end = pd.to_datetime({"year": year, "month": month, "day": 1}) + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"season_end": end, "oni": df["ANOM"].astype(float)}).sort_values(
        "season_end"
    )
    out["available"] = (out["season_end"] + pd.Timedelta(days=PUBLICATION_DELAY_DAYS)).astype(
        "datetime64[ns]"
    )
    out["oni_change_3m"] = out["oni"] - out["oni"].shift(3)
    return out.reset_index(drop=True)


def oni_asof(dates: pd.Series, oni: pd.DataFrame) -> pd.DataFrame:
    """Latest ONI value published on or before each date."""
    d = pd.DataFrame(
        {"date": pd.to_datetime(dates).astype("datetime64[ns]").values, "_i": np.arange(len(dates))}
    )
    d = d.sort_values("date")
    m = pd.merge_asof(
        d,
        oni[["available", "oni", "oni_change_3m"]].sort_values("available"),
        left_on="date",
        right_on="available",
        direction="backward",
    )
    m = m.sort_values("_i")
    return pd.DataFrame(
        {"oni": m["oni"].to_numpy(), "oni_change_3m": m["oni_change_3m"].to_numpy()},
        index=dates.index,
    )
