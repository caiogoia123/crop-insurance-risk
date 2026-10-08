"""Municipality coordinates (municipal seat) and their NASA POWER grid cell.

Source: kelvins/municipios-brasileiros (MIT), IBGE codes + seat coordinates.
Only public, municipality-level coordinates are used; property coordinates are
dropped at ingestion (see croprisk.data.psr).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from croprisk import config

MUNI_PATH = config.RAW / "municipios.csv"


def download_municipalities(force: bool = False) -> pd.DataFrame:
    config.ensure_dirs()
    if force or not MUNI_PATH.exists():
        r = requests.get(config.MUNICIPALITIES_URL, timeout=60)
        r.raise_for_status()
        MUNI_PATH.write_bytes(r.content)
    return load_municipalities()


def load_municipalities() -> pd.DataFrame:
    df = pd.read_csv(MUNI_PATH, dtype={"codigo_ibge": "int64"})
    df = df.rename(columns={"codigo_ibge": "ibge_code", "nome": "muni_name"})
    return df[["ibge_code", "muni_name", "latitude", "longitude", "codigo_uf"]]


def snap_to_grid(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest MERRA-2 grid node (0.5 deg lat x 0.625 deg lon)."""
    clat = np.round(np.asarray(lat) / config.POWER_LAT_STEP) * config.POWER_LAT_STEP
    clon = np.round(np.asarray(lon) / config.POWER_LON_STEP) * config.POWER_LON_STEP
    return np.round(clat, 3), np.round(clon, 3)


def cell_id(clat: float, clon: float) -> str:
    return f"{clat:+07.3f}_{clon:+08.3f}"


def municipality_cells(munis: pd.DataFrame) -> pd.DataFrame:
    """Add grid cell columns (cell_lat, cell_lon, cell_id) to a municipality table."""
    out = munis.copy()
    out["cell_lat"], out["cell_lon"] = snap_to_grid(out["latitude"], out["longitude"])
    out["cell_id"] = [cell_id(a, b) for a, b in zip(out["cell_lat"], out["cell_lon"], strict=True)]
    return out
