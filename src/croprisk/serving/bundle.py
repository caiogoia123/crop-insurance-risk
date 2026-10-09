"""Model bundle: everything the API needs to score a new contract.

Layout of a bundle directory::

    model.txt           LightGBM booster (pre-season model)
    meta.json           version, features, categories, metrics, data snapshot
    rate_calib.json     logistic map log(rate) -> claim probability (for comparison)
    tables/*.parquet    aggregated lookup tables (no policy-level data):
        munis           municipality -> UF, region, grid cell
        clim            climatology features per (cell, window)
        hist_muni       history rates per (municipality, crop group), n >= 10 only
        hist_uf         history rates per (UF, crop group)
        hist_cg         history rate per crop group
        hist_muni_all   history rate per municipality (all crops), n >= 10 only
        yield           median expected yield per (UF, crop)
        oni             ENSO index with publication dates
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from croprisk import calendar
from croprisk.data import enso

TABLES = ["munis", "clim", "hist_muni", "hist_uf", "hist_cg", "hist_muni_all", "yield", "oni"]


def current_dir(models_dir: Path) -> Path:
    """The bundle in use: models/CURRENT holds the version directory name."""
    pointer = models_dir / "CURRENT"
    return models_dir / pointer.read_text().strip()


@dataclass
class Bundle:
    path: Path
    booster: lgb.Booster
    meta: dict
    rate_calib: dict
    tables: dict[str, pd.DataFrame]

    @classmethod
    def load(cls, path: Path) -> Bundle:
        path = Path(path)
        booster = lgb.Booster(model_file=str(path / "model.txt"))
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        rate_calib = json.loads((path / "rate_calib.json").read_text())
        tables = {t: pd.read_parquet(path / "tables" / f"{t}.parquet") for t in TABLES}
        b = cls(path, booster, meta, rate_calib, tables)
        b._index()
        return b

    def _index(self) -> None:
        t = self.tables
        self.munis = t["munis"].set_index("ibge_code")
        self.clim = t["clim"].set_index(["cell_id", "ws_month", "ws_day", "wlen"])
        self.hist_muni = t["hist_muni"].set_index(["ibge_code", "crop_group"])
        self.hist_uf = t["hist_uf"].set_index(["uf", "crop_group"])
        self.hist_cg = t["hist_cg"].set_index("crop_group")
        self.hist_muni_all = t["hist_muni_all"].set_index("ibge_code")
        self.yield_med = t["yield"].set_index(["uf", "crop"])["yld_med"]
        self.oni = t["oni"]
        self.relative_bins = sorted(
            {
                (int(m), int(d))
                for m, d, g in t["clim"][["ws_month", "ws_day", "wlen"]]
                .drop_duplicates()
                .itertuples(index=False)
                if g == calendar.RELATIVE_LENGTH
            }
        )

    # ------------------------------------------------------------------ features
    def features(self, req: dict) -> tuple[pd.DataFrame, dict]:
        """Model input row for one contract, plus context for the response."""
        meta = self.meta
        crop = req["crop"]
        group = calendar.CROP_GROUP[crop]
        muni = self.munis.loc[int(req["ibge_code"])]
        uf = muni["uf"]
        t0 = pd.Timestamp(req["contract_date"])
        pol = pd.DataFrame({"crop_group": [group], "uf": [uf], "t0": [t0]})
        win = calendar.assign_windows(pol).iloc[0]
        ws, wlen = win["ws"], int(win["wlen"])

        row: dict[str, object] = {}
        # categorical (unseen levels become missing inside LightGBM)
        row["crop"] = meta["crop_map"].get(crop, f"{group}_outros")
        row["crop_group"] = group
        row["uf"] = uf
        row["region"] = win["region"]
        ins = req.get("insurer")
        row["insurer"] = ins if ins in meta["categories"]["insurer"] else "other"
        row["product_class"] = req.get("product_class") or "NA"

        area = req["area_ha"]
        si = req["sum_insured"]
        ye = req.get("yield_expected")
        yi = req.get("yield_insured")
        row["log_sum_insured"] = np.log1p(si)
        row["log_area"] = np.log1p(area)
        row["log_si_per_ha"] = np.log1p(si / area)
        row["coverage_level"] = req.get("coverage_level")
        row["yield_expected"] = ye
        row["yield_insured_ratio"] = np.clip(yi / ye, 0, 2) if ye and yi else np.nan
        med = self.yield_med.get((uf, crop), np.nan)
        row["yield_rel"] = ye / med if ye and med == med else np.nan
        row["contract_month"] = t0.month
        row["lead_days"] = (ws - t0).days

        # history (aggregates of matured safras)
        h_cg = self.hist_cg["hist_cg_rate"].get(group, np.nan)
        try:
            h_uf = self.hist_uf.loc[(uf, group)]
            uf_rate, uf_n = h_uf["hist_uf_cg_rate"], h_uf["hist_uf_cg_n"]
        except KeyError:
            uf_rate, uf_n = h_cg, 0.0
        try:
            h_m = self.hist_muni.loc[(int(req["ibge_code"]), group)]
            m_rate, m_n = h_m["hist_muni_cg_rate"], h_m["hist_muni_cg_n"]
        except KeyError:
            m_rate, m_n = uf_rate, 0.0
        row["hist_cg_rate"] = h_cg
        row["hist_uf_cg_rate"] = uf_rate
        row["hist_muni_cg_rate"] = m_rate
        row["hist_muni_rate"] = self.hist_muni_all["hist_muni_rate"].get(
            int(req["ibge_code"]), np.nan
        )
        row["hist_muni_cg_n"] = m_n
        row["hist_uf_cg_n"] = uf_n

        # ENSO known at contract time
        o = enso.oni_asof(pd.Series([t0]), self.oni)
        row["oni"] = float(o["oni"].iloc[0])
        row["oni_change_3m"] = float(o["oni_change_3m"].iloc[0])

        # climatology of the critical window
        m, d = ws.month, ws.day
        key = (muni["cell_id"], m, d, wlen)
        if key not in self.clim.index and wlen == calendar.RELATIVE_LENGTH:
            # vegetables: nearest weekly bin of the window start
            doy = ws.dayofyear
            m, d = min(
                self.relative_bins,
                key=lambda md: abs(pd.Timestamp(2001, md[0], md[1]).dayofyear - doy),
            )
            key = (muni["cell_id"], m, d, wlen)
        clim = self.clim.loc[key]
        for c in meta["clim_features"]:
            row[c] = float(clim[c])

        X = pd.DataFrame([row])[meta["features"]]
        for c in meta["cat_features"]:
            cats = meta["categories"][c]
            X[c] = pd.Categorical(X[c].where(X[c].isin(cats)), categories=cats)
        context = {
            "crop_group": group,
            "uf": uf,
            "municipality": muni["muni_name"],
            "safra": calendar.safra_label(int(win["safra_year"])),
            "critical_window": {
                "start": ws.date().isoformat(),
                "end": (ws + pd.Timedelta(days=wlen)).date().isoformat(),
            },
        }
        return X, context

    # ------------------------------------------------------------------ predict
    def predict(self, req: dict, top_k: int = 5) -> dict:
        X, ctx = self.features(req)
        p = float(self.booster.predict(X)[0])
        contrib = self.booster.predict(X, pred_contrib=True)[0]
        names = self.meta["features"]
        order = np.argsort(-np.abs(contrib[:-1]))[:top_k]
        factors = [
            {
                "feature": names[i],
                "value": None
                if pd.isna(X.iloc[0, i])
                else (
                    str(X.iloc[0, i])
                    if names[i] in self.meta["cat_features"]
                    else float(X.iloc[0, i])
                ),
                "contribution_logit": round(float(contrib[i]), 4),
            }
            for i in order
        ]
        q = self.meta["holdout_score_deciles"]
        decile = int(np.searchsorted(q, p, side="right")) + 1
        out = {
            "probability": round(p, 4),
            "risk_decile": min(decile, 10),
            "holdout_claim_rate": self.meta["holdout"]["metrics"]["pos_rate"],
            **ctx,
            "top_factors": factors,
            "model_version": self.meta["version"],
        }
        rate = req.get("rate")
        if rate:
            a, b = self.rate_calib["intercept"], self.rate_calib["coef"]
            out["rate_implied_probability"] = round(
                float(1 / (1 + np.exp(-(a + b * np.log(rate))))), 4
            )
        return out
