"""Climate features from NASA POWER daily series.

Two families, both computed per (grid cell, window) and then joined to policies:

* Pre-season climatology: statistics of the policy's critical window over the
  reference years 1981-2005 (all before the first policy, so no information from
  the predicted years). This is what is known when the contract is signed.
* In-season observed weather: the same statistics for the window of the policy's
  own safra, from the window start up to the alert date (a fraction ``h`` of the
  window), plus the 30 days before the window, expressed as anomalies against the
  climatology of exactly the same calendar window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from croprisk import config
from croprisk.data import power

log = logging.getLogger(__name__)

DRY_MM = 1.0  # a day with less than 1 mm counts as dry
HOT_C = 34.0  # Tmax at or above: heat stress at flowering
FROST_C = 3.0  # Tmin at or below: frost risk (grid-cell air temperature, see DECISIONS)
DROUGHT_FRAC = 0.7  # window rain below 70% of normal = drought year
ANTE_DAYS = 30

ORIGIN = pd.Timestamp(config.POWER_START)

STATS = [
    "prec_sum",
    "prec_max5d",
    "cdd",
    "hot_days",
    "frost_days",
    "tmax_mean",
    "tmin_mean",
    "rh_mean",
    "gwet_mean",
    "gwet_min",
]


def load_arrays(cell_id: str) -> dict[str, np.ndarray]:
    df = power.load_cell(cell_id)
    full = pd.date_range(ORIGIN, df.index.max(), freq="D")
    df = df.reindex(full)
    return {
        "prec": df["PRECTOTCORR"].to_numpy("float64"),
        "tmax": df["T2M_MAX"].to_numpy("float64"),
        "tmin": df["T2M_MIN"].to_numpy("float64"),
        "rh": df["RH2M"].to_numpy("float64"),
        "gwet": df["GWETROOT"].to_numpy("float64"),
    }


def _max_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True along axis 1."""
    c = np.cumsum(mask, axis=1)
    reset = np.where(~mask, c, 0)
    reset = np.maximum.accumulate(reset, axis=1)
    return (c - reset).max(axis=1)


def window_stats(arrs: dict[str, np.ndarray], starts: np.ndarray, length: int) -> pd.DataFrame:
    """Statistics of windows [start, start+length) over a cell's daily series.

    Windows that run outside the available series get NaN.
    """
    starts = np.asarray(starts, dtype="int64")
    n_days = len(arrs["prec"])
    ok = (starts >= 0) & (starts + length <= n_days)
    out = {s: np.full(len(starts), np.nan) for s in STATS}
    if not ok.any() or length <= 0:
        return pd.DataFrame(out)
    idx = starts[ok, None] + np.arange(length)[None, :]
    p = arrs["prec"][idx]
    tmax = arrs["tmax"][idx]
    tmin = arrs["tmin"][idx]
    complete = ~np.isnan(p).any(axis=1) & ~np.isnan(tmax).any(axis=1)
    p0 = np.nan_to_num(p)
    cs = np.concatenate([np.zeros((len(p0), 1)), np.cumsum(p0, axis=1)], axis=1)
    k = min(5, length)
    max5 = (cs[:, k:] - cs[:, :-k]).max(axis=1)
    vals = {
        "prec_sum": p0.sum(axis=1),
        "prec_max5d": max5,
        "cdd": _max_run(p0 < DRY_MM).astype("float64"),
        "hot_days": (tmax >= HOT_C).sum(axis=1).astype("float64"),
        "frost_days": (tmin <= FROST_C).sum(axis=1).astype("float64"),
        "tmax_mean": np.nanmean(tmax, axis=1),
        "tmin_mean": np.nanmean(tmin, axis=1),
        "rh_mean": np.nanmean(arrs["rh"][idx], axis=1),
        "gwet_mean": np.nanmean(arrs["gwet"][idx], axis=1),
        "gwet_min": np.nanmin(arrs["gwet"][idx], axis=1),
    }
    for s, v in vals.items():
        v = np.where(complete, v, np.nan)
        out[s][ok] = v
    return pd.DataFrame(out)


def day_index(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return ((pd.DatetimeIndex(dates) - ORIGIN).days).to_numpy()


def climatology(
    arrs: dict[str, np.ndarray], month: int, day: int, offset: int, length: int
) -> dict[str, float]:
    """Reference-period statistics of a calendar window (start month/day + offset)."""
    years = range(config.CLIM_START_YEAR, config.CLIM_END_YEAR + 1)
    day = min(day, 28) if month == 2 else day
    starts = day_index(pd.to_datetime([f"{y}-{month:02d}-{day:02d}" for y in years])) + offset
    st = window_stats(arrs, starts, length)
    ps = st["prec_sum"]
    mean_p = ps.mean()
    res = {f"{s}_mean": st[s].mean() for s in STATS}
    res.update(
        {
            "prec_sum_std": ps.std(ddof=1),
            "prec_sum_p10": ps.quantile(0.10),
            "drought_freq": float((ps < DROUGHT_FRAC * mean_p).mean()),
            "frost_freq": float((st["frost_days"] > 0).mean()),
            "hot_freq": float((st["hot_days"] > 0).mean()),
            "gwet_mean_std": st["gwet_mean"].std(ddof=1),
        }
    )
    return res


def cell_features(
    arrs: dict[str, np.ndarray], wins: pd.DataFrame, horizons: list[float]
) -> pd.DataFrame:
    """Features for the unique windows of one cell.

    wins: unique rows of (ws, wlen) for this cell. Returns one row per input row.
    """
    out = pd.DataFrame(index=wins.index)
    ws = pd.DatetimeIndex(wins["ws"])
    wlen = wins["wlen"].astype(int).to_numpy()
    start_idx = day_index(ws)
    md = pd.DataFrame({"m": ws.month, "d": ws.day, "L": wlen}, index=wins.index)

    specs = [("full", 0, None)] + [(f"h{int(h * 100)}", 0, h) for h in horizons if h < 1.0]
    specs.append(("ante", -ANTE_DAYS, "ante"))

    # climatology, cached by calendar key
    clim_cache: dict[tuple, dict] = {}

    def clim(m, d, off, length):
        key = (m, d, off, length)
        if key not in clim_cache:
            clim_cache[key] = climatology(arrs, m, d, off, length)
        return clim_cache[key]

    for name, off, frac in specs:
        if frac is None:
            lengths = wlen
        elif frac == "ante":
            lengths = np.full(len(wlen), ANTE_DAYS)
        else:
            lengths = np.maximum(1, np.round(wlen * frac).astype(int))
        cl = pd.DataFrame(
            [
                clim(m, d, off, int(length))
                for m, d, length in zip(md["m"], md["d"], lengths, strict=True)
            ],
            index=wins.index,
        )
        # observed stats, grouped by length so windows are vectorized
        obs = pd.DataFrame(index=wins.index, columns=STATS, dtype="float64")
        for length in np.unique(lengths):
            sel = lengths == length
            st = window_stats(arrs, start_idx[sel] + off, int(length))
            obs.loc[wins.index[sel], STATS] = st.to_numpy()
        if name == "full":
            # pre-season climatology of the full critical window
            out["clim_prec_mean"] = cl["prec_sum_mean"]
            out["clim_prec_cv"] = cl["prec_sum_std"] / cl["prec_sum_mean"]
            out["clim_prec_p10_ratio"] = cl["prec_sum_p10"] / cl["prec_sum_mean"]
            out["clim_drought_freq"] = cl["drought_freq"]
            out["clim_cdd"] = cl["cdd_mean"]
            out["clim_max5d"] = cl["prec_max5d_mean"]
            out["clim_hot_days"] = cl["hot_days_mean"]
            out["clim_frost_days"] = cl["frost_days_mean"]
            out["clim_frost_freq"] = cl["frost_freq"]
            out["clim_tmax"] = cl["tmax_mean_mean"]
            out["clim_tmin"] = cl["tmin_mean_mean"]
            out["clim_gwet"] = cl["gwet_mean_mean"]
            out["clim_gwet_min"] = cl["gwet_min_mean"]
        prefix = "ante" if name == "ante" else f"obs_{name}"
        out[f"{prefix}_prec_anom"] = obs["prec_sum"] / cl["prec_sum_mean"].clip(lower=1.0) - 1
        out[f"{prefix}_gwet_anom"] = obs["gwet_mean"] - cl["gwet_mean_mean"]
        if name == "ante":
            continue
        out[f"{prefix}_prec_z"] = (obs["prec_sum"] - cl["prec_sum_mean"]) / cl["prec_sum_std"].clip(
            lower=1.0
        )
        out[f"{prefix}_cdd"] = obs["cdd"]
        out[f"{prefix}_cdd_anom"] = obs["cdd"] - cl["cdd_mean"]
        out[f"{prefix}_max5d_anom"] = obs["prec_max5d"] / cl["prec_max5d_mean"].clip(lower=1.0) - 1
        out[f"{prefix}_hot_days"] = obs["hot_days"]
        out[f"{prefix}_hot_anom"] = obs["hot_days"] - cl["hot_days_mean"]
        out[f"{prefix}_frost_days"] = obs["frost_days"]
        out[f"{prefix}_frost_anom"] = obs["frost_days"] - cl["frost_days_mean"]
        out[f"{prefix}_tmax_anom"] = obs["tmax_mean"] - cl["tmax_mean_mean"]
        out[f"{prefix}_tmin_anom"] = obs["tmin_mean"] - cl["tmin_mean_mean"]
        out[f"{prefix}_gwet_min"] = obs["gwet_min"]
    return out


def build_climate_features(pol: pd.DataFrame, horizons: list[float] = (0.5, 1.0)) -> pd.DataFrame:
    """pol: cell_id, ws, wlen. Returns features aligned to pol.index."""
    horizons = list(horizons)
    keys = pol[["cell_id", "ws", "wlen"]].drop_duplicates().reset_index(drop=True)
    parts = []
    cells = keys["cell_id"].unique()
    for i, cid in enumerate(cells, 1):
        k = keys[keys["cell_id"] == cid]
        try:
            arrs = load_arrays(cid)
        except FileNotFoundError:
            log.warning("no POWER data for cell %s", cid)
            continue
        f = cell_features(arrs, k, horizons)
        parts.append(pd.concat([k, f], axis=1))
        if i % 100 == 0:
            log.info("climate features: %d/%d cells", i, len(cells))
    feats = pd.concat(parts, ignore_index=True)
    merged = pol[["cell_id", "ws", "wlen"]].merge(feats, on=["cell_id", "ws", "wlen"], how="left")
    merged.index = pol.index
    return merged.drop(columns=["cell_id", "ws", "wlen"])
