"""Shared fixtures. Everything here is SYNTHETIC: random features, two fake
municipality entries and random climatology, just enough to exercise the code."""

import numpy as np
import pandas as pd
import pytest

from croprisk import calendar, metrics, models, train
from croprisk.data import enso

ONI_TEXT = "\n".join(
    ["SEAS YR TOTAL ANOM"]
    + [
        f"{s} {y} 26.0 {0.1 * ((i + 3 * y) % 7 - 3):.2f}"
        for y in range(2019, 2027)
        for i, s in enumerate(enso.SEASONS)
    ]
)


def _bundle(tmp_path_factory, algo: str):
    rng = np.random.default_rng(0)
    fs = models.FEATURE_SETS["pre_season"]
    n = 3000
    cats = {
        "crop": ["Soja", "Milho 2ª safra", "Trigo"],
        "crop_group": ["soja", "milho_2", "inverno"],
        "uf": ["PR", "MT"],
        "region": ["S", "O"],
        "insurer": ["SEGURADORA SINTETICA S.A.", "other"],
        "product_class": ["PRODUTIVIDADE", "CUSTEIO", "NA"],
    }
    df = pd.DataFrame({c: pd.Categorical(rng.choice(v, n), categories=v) for c, v in cats.items()})
    for c in fs.num:
        df[c] = rng.normal(size=n)
    df["y"] = (rng.random(n) < 0.2).astype(int)
    df["safra_year"] = rng.choice([2020, 2021, 2022], n)
    df["rate"] = rng.uniform(0.03, 0.15, n)
    if algo == "lgbm":
        model = models.LGBModel(fs, params={"min_child_samples": 20}, rounds=10)
    else:
        model = models.LogRegModel(fs)
    model.fit(df, df["y"].to_numpy())
    p = model.predict(df)
    res = {
        "model": model,
        "rate_model": models.RateModel().fit(df, df["y"].to_numpy()),
        "fs": model.fs,
        "algo": algo,
        "tuning": {"params": {}, "rounds": 10},
        "train": df,
        "hold": df,
        "p_hold": p,
        "metrics": metrics.binary_metrics(df["y"], p),
        "rate_metrics": metrics.binary_metrics(df["y"], p),
        "feature_psi": {},
    }
    munis = pd.DataFrame(
        {
            "ibge_code": [4104808, 5107925],
            "muni_name": ["Cascavel", "Sorriso"],
            "uf": ["PR", "MT"],
            "region": ["S", "O"],
            "cell_id": ["cellA", "cellB"],
        }
    )
    keys = train.clim_keys()
    clim = pd.DataFrame(
        [
            {
                "cell_id": cell,
                "ws_month": m,
                "ws_day": d,
                "wlen": L,
                **{c: rng.random() for c in models.CLIM},
            }
            for cell in ["cellA", "cellB"]
            for m, d, L in keys
        ]
    )
    tables = {
        "munis": munis,
        "clim": clim,
        "hist_muni": pd.DataFrame(
            {
                "ibge_code": [4104808],
                "crop_group": ["soja"],
                "hist_muni_cg_rate": [0.1],
                "hist_muni_cg_n": [3.0],
            }
        ),
        "hist_uf": pd.DataFrame(
            {
                "uf": ["PR", "MT"],
                "crop_group": ["soja", "soja"],
                "hist_uf_cg_rate": [0.12, 0.08],
                "hist_uf_cg_n": [5.0, 5.0],
            }
        ),
        "hist_cg": pd.DataFrame({"crop_group": list(calendar.CALENDAR), "hist_cg_rate": 0.15}),
        "hist_muni_all": pd.DataFrame({"ibge_code": [4104808], "hist_muni_rate": [0.15]}),
        "reference": pd.DataFrame(
            {
                "uf": ["PR", "*", "*"],
                "crop": ["Soja", "Soja", "Tomate"],
                "safra_year": [2024, 2024, 2024],
                "yld_med": [3500.0, 3400.0, 60000.0],
                "siha_med": [5000.0, 4800.0, 30000.0],
                "n": [100, 300, 50],
            }
        ),
        "oni": enso.load(ONI_TEXT),
    }
    root = tmp_path_factory.mktemp(f"models_{algo}")
    out = train.write_bundle(
        res,
        tables,
        2023,
        out_root=root,
        crop_counts=pd.Series({"Soja": 900, "Milho 2ª safra": 900, "Trigo": 900}),
    )
    train.promote(out)
    return root


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory):
    """Bundle with the served algorithm (logistic regression)."""
    return _bundle(tmp_path_factory, "logreg")


@pytest.fixture(scope="session")
def bundle_dir_lgbm(tmp_path_factory):
    return _bundle(tmp_path_factory, "lgbm")
