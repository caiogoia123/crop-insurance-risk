"""Data-quality study: INMET automatic stations vs NASA POWER (South, 2021-2022).

Not the model's climate source (that is NASA POWER, gap-free). This module shows
why: station series have gaps, and a careless daily sum turns a missing day into
"0 mm" - a fake drought. Steps:

1. hourly -> daily with explicit validity rules (a day needs 22+ valid hours);
2. separate real zero rain from missing data: days without enough hours are
   missing, and long runs of exact zeros while neighbours get rain are flagged as
   a stuck sensor;
3. fill gaps by inverse-distance weighting (IDW) from neighbour stations,
   validated leave-one-out on days with real measurements;
4. compare stations with the NASA POWER cell they fall in.
"""

from __future__ import annotations

import io
import json
import logging
import unicodedata
import zipfile

import numpy as np
import pandas as pd
import requests

from croprisk import config, plots
from croprisk.data import geo, power

log = logging.getLogger(__name__)

YEARS = [2021, 2022]
UFS = {"PR", "SC", "RS"}
URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
INMET_DIR = config.RAW / "inmet"
MIN_PREC_HOURS = 22
MIN_TEMP_HOURS = 20
RADIUS_KM = 150
IDW_POWER = 2
MIN_NEIGHBOURS = 2
STUCK_DAYS = 30  # run of exact zeros ...
STUCK_NEIGHBOUR_MM = 60  # ... while neighbours received at least this much
RAIN_DAY_MM = 1.0


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()


def _num(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s.astype("string").str.replace(",", ".", regex=False), errors="coerce")
    return v.where(v > -9990).astype("float64")


def download(years=YEARS) -> list:
    INMET_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for y in years:
        p = INMET_DIR / f"{y}.zip"
        if not p.exists():
            r = requests.get(
                URL.format(year=y), headers={"User-Agent": config.USER_AGENT}, timeout=600
            )
            r.raise_for_status()
            p.write_bytes(r.content)
        paths.append(p)
    return paths


def read_station(raw: bytes) -> tuple[dict, pd.DataFrame]:
    lines = raw.decode("latin1").splitlines()
    meta = {}
    for line in lines[:8]:
        k, _, v = line.partition(":")
        meta[_norm(k).strip()] = v.strip(";").strip()
    df = pd.read_csv(io.StringIO("\n".join(lines[8:])), sep=";", dtype=str)
    names = [_norm(c) for c in df.columns]

    def col(*keys):
        for i, n in enumerate(names):
            if all(k in n for k in keys) and "ORVALHO" not in n:
                return df.iloc[:, i]
        raise KeyError(keys)

    date = df.iloc[:, 0].str.replace("/", "-", regex=False)
    hour = df.iloc[:, 1].str.replace(":", "", regex=False).str[:2]
    ts = pd.to_datetime(date + " " + hour + ":00", errors="coerce")
    out = pd.DataFrame(
        {
            # INMET hours are UTC; local day in the South is UTC-3
            "local": ts - pd.Timedelta(hours=3),
            "prec": _num(col("PRECIPITA")),
            "tmax": _num(col("TEMPERATURA MAXIMA")),
            "tmin": _num(col("TEMPERATURA MINIMA")),
        }
    ).dropna(subset=["local"])
    info = {
        "code": meta.get("CODIGO (WMO)"),
        "name": meta.get("ESTACAO"),
        "uf": meta.get("UF"),
        "lat": float(meta["LATITUDE"].replace(",", ".")),
        "lon": float(meta["LONGITUDE"].replace(",", ".")),
    }
    return info, out


def daily(hourly: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    g = hourly.groupby(hourly["local"].dt.normalize())
    d = pd.DataFrame(
        {
            "prec_hours": g["prec"].count(),
            "prec_sum": g["prec"].sum(min_count=1),
            "naive": g["prec"].sum(),  # missing hours counted as 0 mm (the common mistake)
            "tmax": g["tmax"].max().where(g["tmax"].count() >= MIN_TEMP_HOURS),
            "tmin": g["tmin"].min().where(g["tmin"].count() >= MIN_TEMP_HOURS),
        }
    ).reindex(days)
    d["prec_hours"] = d["prec_hours"].fillna(0)
    d["naive"] = d["naive"].fillna(0.0)  # days with no rows at all also become 0
    d["prec"] = d["prec_sum"].where(d["prec_hours"] >= MIN_PREC_HOURS)
    return d.drop(columns="prec_sum")


def haversine_km(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def idw_estimate(values: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """IDW estimate for every (day, station) from *other* stations (leave-one-out)."""
    dist = haversine_km(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    with np.errstate(divide="ignore"):
        w = np.where((dist > 0) & (dist <= RADIUS_KM), 1.0 / dist**IDW_POWER, 0.0)
    v = values[stations.index].to_numpy()
    m = ~np.isnan(v)
    num = np.nan_to_num(v) @ w.T
    den = m.astype(float) @ w.T
    cnt = m.astype(float) @ (w > 0).T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = np.where(cnt >= MIN_NEIGHBOURS, num / den, np.nan)
    return pd.DataFrame(est, index=values.index, columns=stations.index)


def stuck_zero_mask(obs: pd.Series, neighbours: pd.Series) -> pd.Series:
    """Runs of >= STUCK_DAYS valid days with exactly 0 mm while neighbours got rain."""
    zero = obs.eq(0)
    run_id = (zero != zero.shift()).cumsum()
    mask = pd.Series(False, index=obs.index)
    for _, idx in obs[zero].groupby(run_id[zero]).groups.items():
        if len(idx) >= STUCK_DAYS and neighbours.loc[idx].sum() >= STUCK_NEIGHBOUR_MM:
            mask.loc[idx] = True
    return mask


def rain_detection(obs: pd.Series, ref: pd.Series) -> dict:
    ok = obs.notna() & ref.notna()
    o, r = obs[ok] >= RAIN_DAY_MM, ref[ok] >= RAIN_DAY_MM
    hits = (o & r).sum()
    return {
        "pod": float(hits / max(o.sum(), 1)),  # share of observed rain days the reference also has
        "far": float(((~o) & r).sum() / max(r.sum(), 1)),  # reference rain days that were dry
    }


def run() -> dict:
    zips = download()
    days = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D")
    infos, series = {}, {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                parts = name.split("_")
                if len(parts) < 4 or parts[2] not in UFS:
                    continue
                info, hourly = read_station(zf.read(name))
                code = info["code"]
                infos[code] = info
                series.setdefault(code, []).append(hourly)
    stations = pd.DataFrame.from_dict(infos, orient="index")
    daily_by = {c: daily(pd.concat(h), days) for c, h in series.items()}
    log.info("INMET: %d stations in %s", len(stations), sorted(UFS))

    prec = pd.DataFrame({c: d["prec"] for c, d in daily_by.items()})
    naive = pd.DataFrame({c: d["naive"] for c, d in daily_by.items()})
    est = idw_estimate(prec, stations)

    # stuck sensors: exact zeros while neighbours had rain
    suspect = pd.DataFrame({c: stuck_zero_mask(prec[c], est[c]) for c in prec.columns})
    clean = prec.mask(suspect)
    est = idw_estimate(clean, stations)  # re-estimate without the suspect zeros
    filled = clean.fillna(est)

    # leave-one-out check of IDW on days with a real measurement
    ok = clean.notna() & est.notna()
    # pandas 3 keeps NaN when stacking: drop them explicitly
    a = clean.where(ok).stack().dropna()
    e = est.where(ok).stack().dropna()
    idw_metrics = {
        "days": int(ok.sum().sum()),
        "mae_mm": float((a - e).abs().mean()),
        "r_daily": float(np.corrcoef(a, e)[0, 1]),
        "bias_ratio": float(e.sum() / a.sum()),
        **rain_detection(a, e),
    }

    # NASA POWER at each station's grid cell
    cells = geo.municipality_cells(stations.rename(columns={"lat": "latitude", "lon": "longitude"}))
    rows = []
    pw_prec = {}
    for code, r in cells.iterrows():
        try:
            power.update_cell(r["cell_id"], r["cell_lat"], r["cell_lon"], f"{YEARS[-1]}1231")
            p = power.load_cell(r["cell_id"]).reindex(days)
        except Exception as ex:  # noqa: BLE001
            log.warning("POWER cell %s failed: %s", r["cell_id"], ex)
            continue
        pw_prec[code] = p["PRECTOTCORR"]
        d = daily_by[code]
        obs = clean[code]
        both = obs.notna()
        mon_obs = obs.resample("MS").sum(min_count=1)
        mon_valid = obs.notna().resample("MS").mean() >= 0.9
        mon_pw = p["PRECTOTCORR"].resample("MS").sum()
        mon_ok = mon_valid & mon_obs.notna()
        tmax_ok = d["tmax"].notna()
        rows.append(
            {
                "code": code,
                "name": r["name"],
                "uf": r["uf"],
                "lat": round(r["latitude"], 3),
                "lon": round(r["longitude"], 3),
                "missing_days_pct": round(100 * float(prec[code].isna().mean()), 1),
                "suspect_zero_days": int(suspect[code].sum()),
                "naive_zero_days": int(((naive[code] == 0) & prec[code].isna()).sum()),
                "r_daily_power": float(np.corrcoef(obs[both], p["PRECTOTCORR"][both])[0, 1]),
                "r_monthly_power": float(np.corrcoef(mon_obs[mon_ok], mon_pw[mon_ok])[0, 1])
                if mon_ok.sum() > 3
                else np.nan,
                "power_over_station": float(mon_pw[mon_ok].sum() / mon_obs[mon_ok].sum())
                if mon_ok.any()
                else np.nan,
                **{f"{k}_power": v for k, v in rain_detection(obs, p["PRECTOTCORR"]).items()},
                "tmax_bias_power": float((p["T2M_MAX"] - d["tmax"])[tmax_ok].mean()),
                "tmin_bias_power": float((p["T2M_MIN"] - d["tmin"])[d["tmin"].notna()].mean()),
                "r_tmin_power": float(
                    np.corrcoef(p["T2M_MIN"][d["tmin"].notna()], d["tmin"].dropna())[0, 1]
                ),
            }
        )
    st = pd.DataFrame(rows)
    summary = {
        "years": YEARS,
        "ufs": sorted(UFS),
        "stations": int(len(st)),
        "rules": {
            "valid_rain_day_min_hours": MIN_PREC_HOURS,
            "valid_temp_day_min_hours": MIN_TEMP_HOURS,
            "stuck_zero": f">= {STUCK_DAYS} days of 0 mm while IDW of neighbours >= {STUCK_NEIGHBOUR_MM} mm",
            "idw": f"power {IDW_POWER}, radius {RADIUS_KM} km, >= {MIN_NEIGHBOURS} neighbours",
        },
        "missing_days_pct_median": float(st["missing_days_pct"].median()),
        "stations_over_10pct_missing": int((st["missing_days_pct"] > 10).sum()),
        "station_days_missing": int(prec.isna().sum().sum()),
        "station_days_total": int(prec.size),
        "naive_fake_zero_days": int(st["naive_zero_days"].sum()),
        "suspect_zero_days": int(st["suspect_zero_days"].sum()),
        "stations_with_stuck_runs": int((st["suspect_zero_days"] > 0).sum()),
        "idw_leave_one_out": idw_metrics,
        "gaps_filled_by_idw": int((clean.isna() & filled.notna()).sum().sum()),
        "gaps_left_unfilled": int(filled.isna().sum().sum()),
        "power_vs_station_median": {
            k: float(st[k].median())
            for k in [
                "r_daily_power",
                "r_monthly_power",
                "power_over_station",
                "pod_power",
                "far_power",
                "tmax_bias_power",
                "tmin_bias_power",
                "r_tmin_power",
            ]
        },
    }
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "inmet_quality.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    st.round(3).to_csv(config.REPORTS / "inmet_stations.csv", index=False)
    figures(st, prec, naive, filled, pw_prec, stations)
    log.info("INMET summary: %s", json.dumps(summary, indent=1))
    return summary


def figures(st, prec, naive, filled, pw_prec, stations) -> None:
    import matplotlib.pyplot as plt

    plots.setup()
    # 1) missing days by station
    s = st.sort_values("missing_days_pct", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(s.index, s["missing_days_pct"], width=0.7, color=plots.C[0])
    ax.set_xlabel(f"INMET stations in {', '.join(sorted(UFS))}, sorted")
    ax.set_ylabel("days without a valid rain total (%)")
    ax.set_title(f"Missing daily rainfall, {YEARS[0]}-{YEARS[-1]} ({len(s)} stations)")
    ax.set_xticks([])
    plots.save(fig, config.FIGURES / "inmet_missing.png")

    # 2) gap example: cumulative rain, naive vs cleaned + IDW vs POWER
    # example: the station where IDW filled the most missing days
    n_filled = (prec.isna() & filled.notna()).sum()
    cand = st[st["missing_days_pct"].between(8, 50)].copy()
    cand["filled"] = cand["code"].map(n_filled)
    code = cand.sort_values("filled", ascending=False)["code"].iloc[0]
    name = st.set_index("code").loc[code, "name"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    series = [
        ("station, gaps counted as 0 mm", naive[code].cumsum(), plots.C[1]),
        ("station, gaps filled by IDW", filled[code].cumsum(), plots.C[0]),
        ("NASA POWER cell", pw_prec[code].cumsum(), plots.C[2]),
    ]
    for label, y, c in series:
        y = y.ffill()
        ax.plot(y.index, y.values, color=c, label=label)
        ax.annotate(
            f"{y.iloc[-1]:,.0f} mm",
            (y.index[-1], y.iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=plots.INK2,
        )
    miss = prec[code].isna()
    ax.fill_between(
        prec.index,
        0,
        1,
        where=miss,
        transform=ax.get_xaxis_transform(),
        color=plots.GRAY,
        alpha=0.25,
        linewidth=0,
        label="days without valid data",
    )
    ax.set_ylabel("cumulative rain (mm)")
    ax.set_title(f"Why gaps matter: {name.title()} station ({code})")
    ax.legend(loc="upper left")
    plots.save(fig, config.FIGURES / "inmet_gap_example.png")

    # 3) monthly totals, station vs POWER
    xs, ys = [], []
    for c in pw_prec:
        obs = prec[c]
        mon = obs.resample("MS").sum(min_count=1)
        ok = (obs.notna().resample("MS").mean() >= 0.9) & mon.notna()
        xs.append(mon[ok])
        ys.append(pw_prec[c].resample("MS").sum()[ok])
    x, y = pd.concat(xs), pd.concat(ys)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(x, y, s=10, color=plots.C[0], alpha=0.35, linewidths=0)
    lim = float(max(x.max(), y.max())) * 1.05
    ax.plot([0, lim], [0, lim], color=plots.MUTED, linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    r = np.corrcoef(x, y)[0, 1]
    ax.set_xlabel("station monthly rain (mm, months with 90%+ valid days)")
    ax.set_ylabel("NASA POWER monthly rain (mm)")
    ax.set_title(f"Monthly rain: station vs POWER (r = {r:.2f})")
    plots.save(fig, config.FIGURES / "inmet_vs_power_monthly.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
