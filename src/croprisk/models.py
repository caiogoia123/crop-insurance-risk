"""Feature sets and model factories (logistic regression, LightGBM, rate baseline)."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from croprisk import config

CAT = ["crop", "crop_group", "uf", "region", "insurer", "product_class"]
CONTRACT_NUM = [
    "log_area",
    "si_per_ha_rel",
    "coverage_level",
    "yield_insured_ratio",
    "yield_rel",
    "contract_month",
    "lead_days",
]
# Cumulative policy counts are left out on purpose: they only grow over time and
# would act as a hidden time index (see DECISIONS D11).
HIST = [
    "hist_cg_rate",
    "hist_uf_cg_rate",
    "hist_muni_cg_rate",
    "hist_muni_rate",
]
CLIM = [
    "clim_prec_mean",
    "clim_prec_cv",
    "clim_prec_p10_ratio",
    "clim_drought_freq",
    "clim_cdd",
    "clim_max5d",
    "clim_hot_days",
    "clim_frost_days",
    "clim_frost_freq",
    "clim_tmax",
    "clim_tmin",
    "clim_gwet",
    "clim_gwet_min",
]
ENSO = ["oni", "oni_change_3m"]
ANTE = ["ante_prec_anom", "ante_gwet_anom"]
_OBS = [
    "prec_anom",
    "prec_z",
    "cdd",
    "cdd_anom",
    "max5d_anom",
    "hot_days",
    "hot_anom",
    "frost_days",
    "frost_anom",
    "tmax_anom",
    "tmin_anom",
    "gwet_anom",
    "gwet_min",
]
OBS_H50 = [f"obs_h50_{c}" for c in _OBS] + ANTE
OBS_FULL = [f"obs_full_{c}" for c in _OBS] + ANTE
RATE = ["log_rate"]

# Features that must never be used: outcome or post-outcome information.
FORBIDDEN = {"y", "indemnity", "event", "loss_ratio_num", "premium", "rate", "subsidy"}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    num: list[str]
    cat: list[str] = field(default_factory=lambda: list(CAT))
    description: str = ""

    @property
    def columns(self) -> list[str]:
        return self.cat + self.num


FEATURE_SETS = {
    "contract_only": FeatureSet(
        "contract_only", CONTRACT_NUM, description="Contract fields, no history, no climate"
    ),
    "contract": FeatureSet(
        "contract", CONTRACT_NUM + HIST, description="Contract fields + as-of portfolio history"
    ),
    "pre_season": FeatureSet(
        "pre_season",
        CONTRACT_NUM + HIST + CLIM + ENSO,
        description="Contract + history + climatology 1981-2005 + ENSO at contract",
    ),
    "pre_season_rate": FeatureSet(
        "pre_season_rate",
        CONTRACT_NUM + HIST + CLIM + ENSO + RATE,
        description="Pre-season + the insurer's rate as a feature",
    ),
    "in_season": FeatureSet(
        "in_season",
        CONTRACT_NUM + HIST + CLIM + ENSO + OBS_H50,
        description="Pre-season + observed weather up to the middle of the critical window",
    ),
    "end_of_window": FeatureSet(
        "end_of_window",
        CONTRACT_NUM + HIST + CLIM + ENSO + OBS_FULL,
        description="Pre-season + observed weather over the whole critical window",
    ),
}

assert not any(set(fs.columns) & FORBIDDEN for fs in FEATURE_SETS.values())

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_lambda": 1.0,
    "max_cat_to_onehot": 8,
    "cat_smooth": 20,
    "min_data_per_group": 200,
    "verbose": -1,
    "seed": config.SEED,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 12,
}
LGB_ROUNDS = 600


def with_gap(fs: FeatureSet, gap: int) -> FeatureSet:
    """History features computed with a label-maturity gap use the *_gap1 columns."""
    if gap == 0:
        return fs
    gapped = set(HIST)
    num = [f"{c}_gap{gap}" if c in gapped else c for c in fs.num]
    return FeatureSet(fs.name, num, fs.cat, fs.description)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    if "log_rate" not in df:
        df["log_rate"] = np.log(df["rate"])
    return df


class LGBModel:
    def __init__(self, fs: FeatureSet, params: dict | None = None, rounds: int = LGB_ROUNDS):
        self.fs = fs
        self.params = {**LGB_PARAMS, **(params or {})}
        self.rounds = rounds
        self.booster: lgb.Booster | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, valid: tuple | None = None):
        dtrain = lgb.Dataset(
            X[self.fs.columns], label=y, categorical_feature=self.fs.cat, free_raw_data=True
        )
        callbacks = []
        valid_sets = []
        if valid is not None:
            dvalid = lgb.Dataset(
                valid[0][self.fs.columns],
                label=valid[1],
                categorical_feature=self.fs.cat,
                reference=dtrain,
            )
            valid_sets = [dvalid]
            callbacks = [lgb.early_stopping(100, verbose=False)]
        params = dict(self.params)
        if valid is not None:
            # Early stopping on AUC, not logloss: a drought year shifts the base rate,
            # which dominates logloss and stops training after a handful of rounds.
            params["metric"] = "auc"
        self.booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.rounds,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(
            X[self.fs.columns], num_iteration=self.booster.best_iteration or None
        )


class LogRegModel:
    def __init__(self, fs: FeatureSet, C: float = 1.0):
        num = list(fs.num)
        self.fs = FeatureSet(fs.name, num, fs.cat)
        pre = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                            (
                                "q",
                                QuantileTransformer(
                                    n_quantiles=200,
                                    output_distribution="normal",
                                    subsample=200_000,
                                    random_state=config.SEED,
                                ),
                            ),
                        ]
                    ),
                    num,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200),
                    fs.cat,
                ),
            ]
        )
        self.pipe = Pipeline(
            [("pre", pre), ("lr", LogisticRegression(C=C, max_iter=1000, random_state=config.SEED))]
        )

    @staticmethod
    def _cat_as_str(X: pd.DataFrame, cols) -> pd.DataFrame:
        X = X.copy()
        for c in cols:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.pipe.fit(self._cat_as_str(X[self.fs.columns], self.fs.cat), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(self._cat_as_str(X[self.fs.columns], self.fs.cat))[:, 1]


class RateModel:
    """The insurer's rate as a score. Ranking uses the raw rate; probabilities come
    from a logistic recalibration of log(rate) fitted on the training folds."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(X[["rate"]].to_numpy()), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(np.log(X[["rate"]].to_numpy()))[:, 1]


class MeanModel:
    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.p)
