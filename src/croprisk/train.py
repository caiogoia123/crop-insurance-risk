"""Train the production (pre-season) model and export a bundle for the API.

Protocol (same one the monthly retrain uses on the VM):
* holdout H = most recent matured safra; the model is trained on safras <= H-1
  and evaluated on H. The served model never saw the holdout, so champion and
  challenger can always be compared on the same unseen safra.
* lookup tables use the outcomes of every matured safra (<= H): they are inputs
  (portfolio history), not training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from croprisk import calendar, config, dataset, metrics, models
from croprisk.backtest import TUNED_SETS, TUNING_PATH, load_features
from croprisk.data import enso, geo
from croprisk.features import climate, history
from croprisk.features.build import RARE_CROP

log = logging.getLogger(__name__)

# Served model, chosen on the walk-forward backtest (DECISIONS D20): the logistic
# regression had the best mean AUC of the pre-season candidates and by far the most
# stable ranking (worst safra 0.56 vs 0.39 for LightGBM).
FEATURE_SET = "pre_season"
ALGO = "logreg"
DEFAULT_TUNING = {"params": {"num_leaves": 31, "min_child_samples": 500}, "rounds": 100}


def clim_keys() -> list[tuple[int, int, int]]:
    keys = set()
    for by_region in calendar.CALENDAR.values():
        for w in by_region.values():
            if w is not None:
                keys.add((w.month, w.day, w.length))
    for k in range(53):  # weekly bins for contract-relative windows (vegetables)
        d = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=7 * k)
        keys.add((d.month, d.day, calendar.RELATIVE_LENGTH))
    return sorted(keys)


def tuning_for(fs: str) -> dict:
    """Hyperparameters: env override (retrain reuses the champion's), else the
    backtest tuning file, else the defaults."""
    env = os.environ.get("CROPRISK_TUNING")
    if env:
        return json.loads(env)
    if TUNING_PATH.exists():
        return json.loads(TUNING_PATH.read_text())[TUNED_SETS[fs]]
    return DEFAULT_TUNING


def feature_psi(train: pd.DataFrame, hold: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out = {}
    for c in cols:
        if train[c].dtype.kind == "f":
            out[c] = round(metrics.psi(train[c].to_numpy(), hold[c].to_numpy()), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def fit_and_evaluate(
    df: pd.DataFrame, holdout: int, fs_name: str = FEATURE_SET, algo: str = ALGO
) -> dict:
    fs = models.FEATURE_SETS[fs_name]
    tun = tuning_for(fs_name) if algo == "lgbm" else {"params": {"C": 1.0}, "rounds": None}
    train = df[df["safra_year"] <= holdout - 1]
    hold = df[df["safra_year"] == holdout]
    if algo == "lgbm":
        model = models.LGBModel(fs, params=tun["params"], rounds=tun["rounds"])
    else:
        model = models.LogRegModel(fs)
    model.fit(train, train["y"].to_numpy())
    p_hold = model.predict(hold)
    p_train = model.predict(train.sample(min(len(train), 200_000), random_state=config.SEED))
    met = metrics.binary_metrics(hold["y"], p_hold, base_rate=float(train["y"].mean()))
    met["psi_score"] = metrics.psi(p_train, p_hold)
    rate = models.RateModel().fit(train, train["y"].to_numpy())
    rate_met = metrics.binary_metrics(hold["y"], rate.predict(hold))
    return {
        "model": model,
        "rate_model": rate,
        "fs": model.fs,
        "algo": algo,
        "tuning": tun,
        "train": train,
        "hold": hold,
        "p_hold": p_hold,
        "metrics": met,
        "rate_metrics": rate_met,
        "feature_psi": feature_psi(train, hold, fs.num),
    }


def build_tables(policies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    munis = geo.municipality_cells(geo.load_municipalities())
    uf_by_code = {
        11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO", 21: "MA", 22: "PI",
        23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA", 31: "MG", 32: "ES",
        33: "RJ", 35: "SP", 41: "PR", 42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
    }  # fmt: skip
    munis["uf"] = munis["codigo_uf"].map(uf_by_code)
    munis["region"] = calendar.region_of(munis["uf"])
    munis = munis[["ibge_code", "muni_name", "uf", "region", "cell_id"]]
    log.info("climatology table for %d cells ...", munis["cell_id"].nunique())
    clim = climate.climatology_table(sorted(munis["cell_id"].unique()), clim_keys())
    tabs = history.lookup_tables(policies)
    oni = enso.load()
    ref = pd.read_parquet(dataset.REFERENCE_PATH)
    return {"munis": munis, "clim": clim, **tabs, "reference": ref, "oni": oni}


def write_bundle(
    res: dict,
    tables: dict,
    holdout: int,
    out_root: Path = config.MODELS,
    crop_counts: pd.Series | None = None,
) -> Path:
    fs = res["fs"]
    model = res["model"]
    train = res["train"]
    created = datetime.now(UTC)
    algo = res.get("algo", "lgbm")
    if algo == "lgbm":
        blob = model.booster.model_to_string().encode()
    else:
        lr = model.pipe.named_steps["lr"]
        blob = lr.coef_.tobytes() + lr.intercept_.tobytes()
    digest = hashlib.sha1(blob).hexdigest()[:8]
    version = f"{created:%Y%m%d}-{digest}"
    out = out_root / version
    (out / "tables").mkdir(parents=True, exist_ok=True)
    if algo == "lgbm":
        model.booster.save_model(str(out / "model.txt"))
    else:
        joblib.dump(model, out / "model.joblib")
    lr = res["rate_model"].lr
    (out / "rate_calib.json").write_text(
        json.dumps({"intercept": float(lr.intercept_[0]), "coef": float(lr.coef_[0][0])})
    )
    for name, t in tables.items():
        t.to_parquet(out / "tables" / f"{name}.parquet", index=False)

    if crop_counts is None:
        crop_counts = pd.read_parquet(dataset.POLICIES_PATH, columns=["crop"])[
            "crop"
        ].value_counts()
    n = crop_counts
    crop_map = {
        c: (c if n.get(c, 0) >= RARE_CROP else f"{g}_outros")
        for c, g in calendar.CROP_GROUP.items()
    }
    manifest_path = config.RAW / "psr_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cleaning = json.loads(dataset.REPORT_PATH.read_text()) if dataset.REPORT_PATH.exists() else {}
    meta = {
        "version": version,
        "created_at": created.isoformat(timespec="seconds"),
        "algo": algo,
        "model": ("Logistic regression" if algo == "logreg" else "LightGBM")
        + ", pre-season (contract + history + climatology + ENSO)",
        "feature_set": fs.name,
        "features": fs.columns,
        "cat_features": fs.cat,
        "clim_features": models.CLIM,
        "categories": {c: [str(x) for x in train[c].cat.categories] for c in fs.cat},
        "crop_map": crop_map,
        "insurer_shares": shares(train, "insurer"),
        "product_class_shares": shares(train, "product_class", keep=("PRODUTIVIDADE", "CUSTEIO")),
        "crops": sorted(calendar.CROP_GROUP),
        "tuning": res["tuning"],
        "train_safras": [int(train["safra_year"].min()), int(train["safra_year"].max())],
        "train_rows": int(len(train)),
        "holdout": {
            "safra": calendar.safra_label(holdout),
            "safra_year": holdout,
            "rows": int(len(res["hold"])),
            "metrics": {
                k: round(v, 4) if isinstance(v, float) else v for k, v in res["metrics"].items()
            },
            "insurer_rate_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in res["rate_metrics"].items()
            },
            "top_feature_psi": dict(list(res["feature_psi"].items())[:10]),
        },
        "holdout_score_deciles": np.quantile(res["p_hold"], np.linspace(0.1, 0.9, 9)).tolist(),
        "data_snapshot": {
            "psr_resources": {v["name"]: v.get("last_modified") for v in manifest.values()},
            "claims_cutoff": cleaning.get("claims_cutoff"),
            "safras_kept": cleaning.get("safras_kept"),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def shares(train: pd.DataFrame, col: str, keep=None, top: int = 12) -> dict[str, float]:
    """Market shares in the last training safra (used when a request omits `col`)."""
    last = train[train["safra_year"] == train["safra_year"].max()][col].astype(str)
    if keep:
        last = last[last.isin(keep)]
    vc = last.value_counts(normalize=True).head(top)
    return {k: round(float(v / vc.sum()), 4) for k, v in vc.items()}


def promote(bundle_dir: Path) -> None:
    (bundle_dir.parent / "CURRENT").write_text(bundle_dir.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()
    df = load_features()
    holdout = int(df["safra_year"].max())
    res = fit_and_evaluate(df, holdout)
    log.info("holdout %s: %s", holdout, res["metrics"])
    policies = pd.read_parquet(dataset.POLICIES_PATH)
    tables = build_tables(policies)
    out = write_bundle(res, tables, holdout)
    if not args.no_promote:
        promote(out)
    log.info("bundle written to %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
