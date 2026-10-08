"""NASA POWER daily climate per MERRA-2 grid cell, with a local parquet cache.

One request per grid cell (not per policy): every municipality is snapped to the
nearest 0.5 x 0.625 degree node, so municipalities that share a cell share a series.
Requests run with low concurrency and exponential backoff to respect the API.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from croprisk import config

log = logging.getLogger(__name__)

POWER_DIR = config.RAW / "power"


def _cell_path(cell_id: str):
    return POWER_DIR / f"{cell_id}.parquet"


def fetch_point(lat: float, lon: float, start: str, end: str, retries: int = 6) -> pd.DataFrame:
    params = {
        "parameters": ",".join(config.POWER_PARAMETERS),
        "community": "AG",
        "latitude": f"{lat:.3f}",
        "longitude": f"{lon:.3f}",
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "LST",
    }
    wait = 5.0
    for attempt in range(retries):
        try:
            r = requests.get(config.POWER_URL, params=params, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            js = r.json()
            par = js["properties"]["parameter"]
            fill = js.get("header", {}).get("fill_value", -999.0)
            df = pd.DataFrame(par)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df = df.replace(fill, np.nan).astype("float32")
            df.index.name = "date"
            return df
        except (requests.RequestException, KeyError, ValueError) as e:
            if attempt == retries - 1:
                raise
            log.warning("POWER %s,%s attempt %d failed: %s", lat, lon, attempt + 1, e)
            time.sleep(wait)
            wait *= 2
    raise RuntimeError("unreachable")


def update_cell(cell_id: str, lat: float, lon: float, end: str) -> str:
    """Download a cell's full series, or only the missing tail if cached."""
    path = _cell_path(cell_id)
    if path.exists():
        old = pd.read_parquet(path)
        last = old.index.max().date()
        end_d = pd.to_datetime(end).date()
        if last >= end_d:
            return "cached"
        start = (last + timedelta(days=1)).strftime("%Y%m%d")
        new = fetch_point(lat, lon, start, end)
        df = pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        status = "extended"
    else:
        df = fetch_point(lat, lon, config.POWER_START, end)
        status = "downloaded"
    # POWER returns fill values for the most recent days it has not processed yet;
    # drop trailing all-missing rows so a later update fetches them again.
    valid = df.notna().any(axis=1)
    if valid.any():
        df = df.loc[: valid[valid].index.max()]
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return status


def update_cells(cells: pd.DataFrame, end: str | None = None, workers: int = 3) -> dict:
    """cells: DataFrame with cell_id, cell_lat, cell_lon (unique)."""
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    end = end or (date.today() - timedelta(days=10)).strftime("%Y%m%d")
    cells = cells.drop_duplicates("cell_id")
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(update_cell, r.cell_id, r.cell_lat, r.cell_lon, end): r.cell_id
            for r in cells.itertuples()
        }
        for i, f in enumerate(as_completed(futs), 1):
            try:
                s = f.result()
            except Exception as e:  # keep going; a later run retries the cell
                log.error("cell %s failed: %s", futs[f], e)
                s = "failed"
            stats[s] = stats.get(s, 0) + 1
            if i % 50 == 0:
                log.info("%d/%d cells %s", i, len(futs), stats)
    log.info("POWER update done: %s", stats)
    return stats


def load_cell(cell_id: str) -> pd.DataFrame:
    return pd.read_parquet(_cell_path(cell_id))
