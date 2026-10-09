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

TUNE_SAFRAS = [2009, 2010, 2011, 2012]
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
    Experiment("logreg_pre_season_gap1", "logreg", "pre_season", gap=1, group="sensitivity"),
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


TUNED_SETS = {
    "contract_only": "contract",
    "contract": "contract",
    "pre_season": "pre_season",
    "pre_season_rate": "pre_season",
    "in_season": "in_season",
    "end_of_window": "end_of_window",
}


def tune_all(df: pd.DataFrame) -> dict:
    out = {fs: tune(df, fs) for fs in sorted(set(TUNED_SETS.values()))}
    TUNING_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def tune(df: pd.DataFrame, fs_name: str = "pre_season") -> dict:
    """Pick LightGBM capacity and number of rounds on safras 2009-2012 only.

    Selection by mean validation AUC over several safras at fixed checkpoints (no
    early stopping). Single-year early stopping is meaningless here: the pre-season
    ranking can invert in a drought year (AUC < 0.5 in 2011/12), and logloss is
    dominated by the year's base rate, which no contract feature can know.
    """
    grid = [
        {"num_leaves": 15, "min_child_samples": 2000},
        {"num_leaves": 31, "min_child_samples": 500},
        {"num_leaves": 63, "min_child_samples": 1000},
        {"num_leaves": 127, "min_child_samples": 2000},
    ]
    checkpoints = [50, 100, 200, 300, 400, 600, 800]
    fs = models.FEATURE_SETS[fs_name]
    results = []
    for params in grid:
        curves = []
        for v in TUNE_SAFRAS:
            tr = df[df["safra_year"] < v]
            va = df[df["safra_year"] == v]
            m = models.LGBModel(fs, params=params, rounds=max(checkpoints)).fit(
                tr, tr["y"].to_numpy()
            )
            curves.append(
                [
                    metrics.binary_metrics(
                        va["y"], m.booster.predict(va[fs.columns], num_iteration=k)
                    )["auc"]
                    for k in checkpoints
                ]
            )
        mean_auc = np.mean(curves, axis=0)
        for k, a in zip(checkpoints, mean_auc, strict=True):
            results.append({**params, "rounds": k, "mean_auc": float(a)})
        log.info(
            "tune %s %s -> mean auc by rounds %s", fs_name, params, np.round(mean_auc, 4).tolist()
        )
    best = max(results, key=lambda r: r["mean_auc"])
    chosen = {
        "params": {
            "num_leaves": best["num_leaves"],
            "min_child_samples": best["min_child_samples"],
        },
        "rounds": best["rounds"],
        "mean_auc": best["mean_auc"],
        "grid": results,
        "validation_safras": TUNE_SAFRAS,
        "criterion": "mean AUC over validation safras",
    }
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
        tun = tuning[TUNED_SETS[exp.feature_set]] if exp.algo == "lgbm" else None
        with tracking.run(exp.name, algo=exp.algo, group=exp.group):
            tracking.log_params(
                {
                    "algo": exp.algo,
                    "feature_set": exp.feature_set,
                    "gap": exp.gap,
                    "window": exp.window,
                    **(tun["params"] if tun else {}),
                    "rounds": tun["rounds"] if tun else None,
                }
            )
            for s in TEST_SAFRAS:
                tr = df[train_mask(df, s, exp.gap, exp.window)]
                te = test[test["safra_year"] == s]
                model = make_model(
                    exp, tun["params"] if tun else None, tun["rounds"] if tun else None
                )
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
    if TUNING_PATH.exists() and not args.retune:
        tuning = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    else:
        tuning = tune_all(df)
    for k, v in tuning.items():
        log.info(
            "tuning %s: %s rounds=%s mean_auc=%.4f", k, v["params"], v["rounds"], v["mean_auc"]
        )
    exps = [e for e in EXPERIMENTS if not args.only or e.name in args.only]
    run(exps, df, tuning)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
