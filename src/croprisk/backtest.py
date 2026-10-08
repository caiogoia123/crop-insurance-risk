"""Walk-forward backtest by safra (never a random split).

For each test safra s the models are trained on safras <= s - 1 - gap (expanding
window, or the last `window` safras) and scored on safra s. Hyperparameters are
chosen once, on validation safras that precede every test safra (`tune`).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from croprisk import config, metrics, models, tracking
from croprisk.features.build import FEATURES_PATH

log = logging.getLogger(__name__)

TUNE_SAFRAS = [2010, 2011, 2012]
TEST_SAFRAS = list(range(2013, 2024))  # 2013/14 ... 2023/24
OOF_PATH = config.PROCESSED / "oof.parquet"
FOLD_METRICS_PATH = config.REPORTS / "fold_metrics.csv"
TUNING_PATH = config.REPORTS / "tuning.json"


@dataclass(frozen=True)
class Experiment:
    name: str
    algo: str  # mean | rate | logreg | lgbm
    feature_set: str | None = None
    gap: int = 0
    window: int | None = None  # None = expanding
    group: str = "main"  # main | sensitivity


EXPERIMENTS = [
    Experiment("mean_rate", "mean"),
    Experiment("insurer_rate", "rate"),
    Experiment("logreg_contract", "logreg", "contract"),
    Experiment("logreg_pre_season", "logreg", "pre_season"),
    Experiment("logreg_in_season", "logreg", "in_season"),
    Experiment("lgbm_contract_only", "lgbm", "contract_only"),
    Experiment("lgbm_contract", "lgbm", "contract"),
    Experiment("lgbm_pre_season", "lgbm", "pre_season"),
    Experiment("lgbm_pre_season_rate", "lgbm", "pre_season_rate"),
    Experiment("lgbm_in_season", "lgbm", "in_season"),
    Experiment("lgbm_end_of_window", "lgbm", "end_of_window"),
    Experiment("lgbm_contract_gap1", "lgbm", "contract", gap=1, group="sensitivity"),
    Experiment("lgbm_pre_season_gap1", "lgbm", "pre_season", gap=1, group="sensitivity"),
    Experiment("lgbm_in_season_gap1", "lgbm", "in_season", gap=1, group="sensitivity"),
    Experiment("lgbm_pre_season_roll8", "lgbm", "pre_season", window=8, group="sensitivity"),
]


def load_features() -> pd.DataFrame:
    return models.prepare(pd.read_parquet(FEATURES_PATH))


def train_mask(df: pd.DataFrame, s: int, gap: int = 0, window: int | None = None) -> pd.Series:
    last = s - 1 - gap
    m = df["safra_year"] <= last
    if window is not None:
        m &= df["safra_year"] > last - window
    return m


def make_model(exp: Experiment, params: dict | None = None, rounds: int | None = None):
    if exp.algo == "mean":
        return models.MeanModel()
    if exp.algo == "rate":
        return models.RateModel()
    fs = models.with_gap(models.FEATURE_SETS[exp.feature_set], exp.gap)
    if exp.algo == "logreg":
        return models.LogRegModel(fs)
    return models.LGBModel(fs, params=params, rounds=rounds or models.LGB_ROUNDS)


def tune(df: pd.DataFrame) -> dict:
    """Pick LightGBM capacity and number of rounds on safras 2010-2012 only."""
    grid = [
        {"num_leaves": 31, "min_child_samples": 500},
        {"num_leaves": 63, "min_child_samples": 500},
        {"num_leaves": 127, "min_child_samples": 1000},
        {"num_leaves": 255, "min_child_samples": 2000},
    ]
    fs = models.FEATURE_SETS["pre_season"]
    results = []
    for params in grid:
        losses, rounds, aucs = [], [], []
        for v in TUNE_SAFRAS:
            tr = df[df["safra_year"] < v]
            va = df[df["safra_year"] == v]
            m = models.LGBModel(fs, params=params, rounds=3000).fit(
                tr, tr["y"].to_numpy(), valid=(va, va["y"].to_numpy())
            )
            p = m.predict(va)
            met = metrics.binary_metrics(va["y"], p)
            losses.append(met["logloss"])
            aucs.append(met["auc"])
            rounds.append(m.booster.best_iteration)
        results.append(
            {
                **params,
                "logloss": float(np.mean(losses)),
                "auc": float(np.mean(aucs)),
                "best_rounds": rounds,
            }
        )
        log.info(
            "tune %s -> logloss %.4f auc %.4f rounds %s",
            params,
            np.mean(losses),
            np.mean(aucs),
            rounds,
        )
    best = min(results, key=lambda r: r["logloss"])
    chosen = {
        "params": {
            "num_leaves": best["num_leaves"],
            "min_child_samples": best["min_child_samples"],
        },
        # training sets in the test folds are larger than in tuning: 20% more rounds
        "rounds": int(np.median(best["best_rounds"]) * 1.2),
        "grid": results,
        "validation_safras": TUNE_SAFRAS,
    }
    TUNING_PATH.write_text(json.dumps(chosen, indent=2))
    return chosen


def run(experiments: list[Experiment], df: pd.DataFrame, tuning: dict) -> None:
    tracking.setup("crop-insurance-risk-backtest")
    test = df[df["safra_year"].isin(TEST_SAFRAS)]
    oof = (
        pd.read_parquet(OOF_PATH)
        if OOF_PATH.exists()
        else test[["proposal_id", "safra_year"]].reset_index(drop=True)
    )
    fold_rows = (
        pd.read_csv(FOLD_METRICS_PATH).to_dict("records") if FOLD_METRICS_PATH.exists() else []
    )
    for exp in experiments:
        t_exp = time.time()
        preds = pd.Series(np.nan, index=test.index)
        fold_rows = [r for r in fold_rows if r["experiment"] != exp.name]
        with tracking.run(exp.name, algo=exp.algo, group=exp.group):
            tracking.log_params(
                {
                    "algo": exp.algo,
                    "feature_set": exp.feature_set,
                    "gap": exp.gap,
                    "window": exp.window,
                    **(tuning["params"] if exp.algo == "lgbm" else {}),
                    "rounds": tuning["rounds"] if exp.algo == "lgbm" else None,
                }
            )
            for s in TEST_SAFRAS:
                tr = df[train_mask(df, s, exp.gap, exp.window)]
                te = test[test["safra_year"] == s]
                model = make_model(exp, tuning["params"], tuning["rounds"])
                model.fit(tr, tr["y"].to_numpy())
                p = model.predict(te)
                p_tr = model.predict(tr.sample(min(len(tr), 200_000), random_state=config.SEED))
                preds.loc[te.index] = p
                met = metrics.binary_metrics(te["y"], p, base_rate=float(tr["y"].mean()))
                met["psi_score"] = metrics.psi(p_tr, p)
                met["train_rows"] = int(len(tr))
                fold_rows.append({"experiment": exp.name, "safra_year": s, **met})
                tracking.log_metrics({k: v for k, v in met.items() if k != "n"}, step=s)
                log.info(
                    "%s %s auc=%.4f brier=%.4f", exp.name, s, met.get("auc", np.nan), met["brier"]
                )
            pooled = metrics.binary_metrics(test["y"], preds.to_numpy())
            tracking.log_metrics({f"pooled_{k}": v for k, v in pooled.items()})
        oof[exp.name] = preds.to_numpy()
        oof.to_parquet(OOF_PATH, index=False)
        pd.DataFrame(fold_rows).to_csv(FOLD_METRICS_PATH, index=False)
        log.info("%s done in %.1f min", exp.name, (time.time() - t_exp) / 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="experiment names to (re)run")
    ap.add_argument("--retune", action="store_true")
    args = ap.parse_args()
    config.ensure_dirs()
    df = load_features()
    tuning = (
        json.loads(TUNING_PATH.read_text())
        if TUNING_PATH.exists() and not args.retune
        else tune(df)
    )
    log.info("tuning: %s rounds=%s", tuning["params"], tuning["rounds"])
    exps = [e for e in EXPERIMENTS if not args.only or e.name in args.only]
    run(exps, df, tuning)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
