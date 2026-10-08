"""Assemble the modeling table: contract + history + climate features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from croprisk import config, dataset
from croprisk.data import enso, geo
from croprisk.features import climate, history

log = logging.getLogger(__name__)

FEATURES_PATH = config.PROCESSED / "features.parquet"
HORIZONS = (0.5, 1.0)
RARE_CROP = 500  # crops with fewer policies are merged into "<group>_outros"
RARE_INSURER = 1000


def contract_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_sum_insured"] = np.log1p(df["sum_insured"])
    out["log_area"] = np.log1p(df["area_ha"])
    out["log_si_per_ha"] = np.log1p(df["sum_insured"] / df["area_ha"])
    out["coverage_level"] = df["coverage_level"]
    out["yield_expected"] = df["yield_expected"]
    out["yield_insured_ratio"] = (df["yield_insured"] / df["yield_expected"]).clip(0, 2)
    out["contract_month"] = df["t0"].dt.month
    # days between contract and the start of the critical window
    out["lead_days"] = (df["ws"] - df["t0"]).dt.days
    return out


def categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    crop_n = df["crop"].map(df["crop"].value_counts())
    crop = df["crop"].where(crop_n >= RARE_CROP, df["crop_group"] + "_outros")
    ins_n = df["insurer"].map(df["insurer"].value_counts())
    insurer = df["insurer"].where(ins_n >= RARE_INSURER, "other")
    out["crop"] = crop.astype("category")
    out["crop_group"] = df["crop_group"].astype("category")
    out["uf"] = df["uf"].astype("category")
    out["region"] = df["region"].astype("category")
    out["insurer"] = insurer.astype("category")
    out["product_class"] = df["product_class"].fillna("NA").astype("category")
    return out


def build() -> pd.DataFrame:
    pol = pd.read_parquet(dataset.POLICIES_PATH)
    munis = geo.municipality_cells(geo.load_municipalities())
    pol = pol.merge(munis[["ibge_code", "cell_id"]], on="ibge_code", how="left")
    log.info("policies without grid cell: %d", pol["cell_id"].isna().sum())
    pol = pol[pol["cell_id"].notna()].reset_index(drop=True)

    log.info("climate features ...")
    clim = climate.build_climate_features(pol, horizons=HORIZONS)
    oni = enso.oni_asof(pol["t0"], enso.download())
    hist0 = history.history_features(pol, gap=0)
    hist1 = history.history_features(pol, gap=1).add_suffix("_gap1")

    keep = [
        "proposal_id",
        "safra_year",
        "ibge_code",
        "cell_id",
        "t0",
        "ws",
        "wlen",
        "y",
        "indemnity",
        "premium",
        "rate",
        "sum_insured",
        "event",
    ]
    feats = pd.concat(
        [
            pol[keep],
            categorical_features(pol),
            contract_features(pol),
            hist0,
            hist1,
            oni,
            clim,
        ],
        axis=1,
    )
    # float32 halves memory (matters for the monthly retrain on the 12 GB VM)
    money = {"indemnity", "premium", "rate", "sum_insured"}
    for c in feats.columns:
        if feats[c].dtype == "float64" and c not in money:
            feats[c] = feats[c].astype("float32")
    return feats


def main() -> None:
    config.ensure_dirs()
    feats = build()
    feats.to_parquet(FEATURES_PATH, index=False)
    log.info("features: %s -> %s", feats.shape, FEATURES_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
