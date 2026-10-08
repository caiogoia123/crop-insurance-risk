"""Crop groups and critical climate windows.

PSR coverage dates are contractual (soy policies often start in June, months before
planting, and many run for 365 days), so they do not mark the crop's phenology.
Climate is therefore measured over a fixed *critical window* per crop group and
macro-region, taken from standard Brazilian crop calendars (CONAB): the period
when drought, frost, heat or excess rain does most of the yield damage.

Window assignment: a policy gets the first window whose start is on or after
``t0 - L/2`` (t0 = coverage start, L = window length), i.e. the first window of
which at least half falls after the contract date.

The agricultural year ("safra") of a policy is the Aug-Jul year that contains the
window start, so a summer crop, the following second-crop corn and the winter
wheat of the same season fall in the same fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SOUTH = {"PR", "SC", "RS"}


@dataclass(frozen=True)
class Window:
    month: int
    day: int
    length: int  # days


# (group) -> {"S": window in the South, "O": window elsewhere}
# A window of None means "relative to the contract": start = t0 + 30 days.
CALENDAR: dict[str, dict[str, Window | None]] = {
    # flowering and grain filling of summer soy
    "soja": {"S": Window(12, 1, 105), "O": Window(11, 15, 105)},
    "milho_1": {"S": Window(11, 15, 105), "O": Window(12, 1, 105)},
    # second-crop corn: drought at flowering (Apr) and frost (Jun)
    "milho_2": {"S": Window(3, 15, 107), "O": Window(3, 15, 107)},
    # winter cereals: frost at flowering, excess rain at harvest
    "inverno": {"S": Window(7, 1, 122), "O": Window(6, 1, 107)},
    "arroz": {"S": Window(12, 1, 105), "O": Window(12, 1, 105)},
    "feijao_1": {"S": Window(11, 1, 92), "O": Window(11, 1, 92)},
    "feijao_2": {"S": Window(3, 1, 92), "O": Window(3, 1, 92)},
    "safrinha_outros": {"S": Window(3, 15, 107), "O": Window(3, 15, 107)},
    "verao_outros": {"S": Window(12, 1, 105), "O": Window(12, 1, 105)},
    # temperate fruit: late frost at budding (Sep-Oct) and hail (Oct-Jan)
    "fruta_temperada": {"S": Window(9, 1, 153), "O": Window(9, 1, 153)},
    "fruta_tropical": {"S": Window(10, 1, 150), "O": Window(10, 1, 150)},
    # coffee: frost (Jun-Aug) and drought before flowering (Aug-Oct)
    "cafe": {"S": Window(6, 1, 153), "O": Window(6, 1, 153)},
    # sugarcane: dry season and frost
    "cana": {"S": Window(5, 1, 153), "O": Window(5, 1, 153)},
    # vegetables are planted all year round: window follows the contract
    "hortalicas": {"S": None, "O": None},
}
RELATIVE_LENGTH = 120
RELATIVE_OFFSET = 30

CROP_GROUP: dict[str, str] = {
    "Soja": "soja",
    "Milho 1ª safra": "milho_1",
    "Milho 2ª safra": "milho_2",
    "Trigo": "inverno",
    "Cevada": "inverno",
    "Aveia": "inverno",
    "Triticale": "inverno",
    "Canola": "inverno",
    "Centeio": "inverno",
    "Arroz": "arroz",
    "Feijão 1ª safra": "feijao_1",
    "Feijão 2ª safra": "feijao_2",
    "Sorgo": "safrinha_outros",
    "Girassol": "safrinha_outros",
    "Algodão": "verao_outros",
    "Amendoim": "verao_outros",
    "Mandioca": "verao_outros",
    "Uva": "fruta_temperada",
    "Maçã": "fruta_temperada",
    "Pêssego": "fruta_temperada",
    "Ameixa": "fruta_temperada",
    "Caqui": "fruta_temperada",
    "Nectarina": "fruta_temperada",
    "Pêra": "fruta_temperada",
    "Kiwi": "fruta_temperada",
    "Figo": "fruta_temperada",
    "Laranja": "fruta_tropical",
    "Tangerina": "fruta_tropical",
    "Limão": "fruta_tropical",
    "Lima": "fruta_tropical",
    "Banana": "fruta_tropical",
    "Goiaba": "fruta_tropical",
    "Manga": "fruta_tropical",
    "Abacate": "fruta_tropical",
    "Maracujá": "fruta_tropical",
    "Atemoia": "fruta_tropical",
    "Abacaxi": "fruta_tropical",
    "Lichia": "fruta_tropical",
    "Mamão": "fruta_tropical",
    "Graviola": "fruta_tropical",
    "Umbu": "fruta_tropical",
    "Cacau": "fruta_tropical",
    "Café": "cafe",
    "Cana-de-açúcar": "cana",
    "Tomate": "hortalicas",
    "Cebola": "hortalicas",
    "Alho": "hortalicas",
    "Batata": "hortalicas",
    "Pimentão": "hortalicas",
    "Melancia": "hortalicas",
    "Melão": "hortalicas",
    "Repolho": "hortalicas",
    "Beterraba": "hortalicas",
    "Cenoura": "hortalicas",
    "Abóbora": "hortalicas",
    "Abobrinha": "hortalicas",
    "Alface": "hortalicas",
    "Chuchu": "hortalicas",
    "Pepino": "hortalicas",
    "Berinjela": "hortalicas",
    "Brócolis": "hortalicas",
    "Couve-flor": "hortalicas",
    "Vagem": "hortalicas",
    "Morango": "hortalicas",
    "Ervilha": "hortalicas",
}
# Out of scope (not crop-yield insurance): livestock, forest, pasture, Carinata (parametric pilot).
OUT_OF_SCOPE_CROPS = {"Pecuário", "Floresta", "Pastagem", "Carinata"}


def region_of(uf: pd.Series) -> pd.Series:
    return np.where(uf.isin(SOUTH), "S", "O")


def assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Return window start (ws), length (wlen) and safra_year for each policy.

    df needs columns: crop_group, uf, t0 (datetime64).
    """
    out = pd.DataFrame(index=df.index)
    ws = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    wlen = pd.Series(np.nan, index=df.index)
    reg = pd.Series(region_of(df["uf"]), index=df.index)
    t0 = df["t0"]
    for group, by_region in CALENDAR.items():
        for r, w in by_region.items():
            m = (df["crop_group"] == group) & (reg == r) & t0.notna()
            if not m.any():
                continue
            if w is None:
                ws[m] = t0[m] + pd.Timedelta(days=RELATIVE_OFFSET)
                wlen[m] = RELATIVE_LENGTH
                continue
            limit = t0[m] - pd.Timedelta(days=w.length // 2)
            year = limit.dt.year
            cand = pd.to_datetime({"year": year, "month": w.month, "day": w.day}, errors="coerce")
            cand = cand.where(
                cand >= limit,
                pd.to_datetime({"year": year + 1, "month": w.month, "day": w.day}, errors="coerce"),
            )
            ws[m] = cand.values
            wlen[m] = w.length
    out["ws"] = ws
    out["wlen"] = wlen.astype("Int64")
    out["region"] = reg
    out["safra_year"] = np.where(ws.dt.month >= 8, ws.dt.year, ws.dt.year - 1)
    out.loc[ws.isna(), "safra_year"] = np.nan
    out["safra_year"] = out["safra_year"].astype("Int64")
    return out


def safra_label(year: int) -> str:
    return f"{year}/{str(year + 1)[-2:]}"
