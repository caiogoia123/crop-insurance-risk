"""Paths and constants shared by the pipeline.

Every path can be moved with the CROPRISK_HOME environment variable (used on the VM).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CROPRISK_HOME", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

CKAN_PACKAGE_URL = "https://dados.agricultura.gov.br/api/3/action/package_show?id=sisser3"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
MUNICIPALITIES_URL = (
    "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
)

# NASA POWER meteorology comes from MERRA-2 on a 0.5 x 0.625 degree grid.
POWER_LAT_STEP = 0.5
POWER_LON_STEP = 0.625
POWER_PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT"]
POWER_START = "19810101"

# Climatology reference period: entirely before the first policy (2006), so the
# pre-season "normal" never contains information from the years being predicted.
CLIM_START_YEAR = 1981
CLIM_END_YEAR = 2005


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, PROCESSED, MODELS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
