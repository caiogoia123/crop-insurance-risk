"""Discrimination, calibration and stability metrics used in credit/insurance risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max distance between score CDFs of positives and negatives."""
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    cum_pos = np.cumsum(ys) / max(ys.sum(), 1)
    cum_neg = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    # evaluate only at the last index of each tied score
    last = np.r_[ps[1:] != ps[:-1], True]
    return float(np.max(np.abs(cum_pos[last] - cum_neg[last])))


def binary_metrics(y, p, base_rate: float | None = None) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "pos_rate": float(y.mean()), "mean_pred": float(p.mean())}
    if 0 < y.sum() < len(y):
        auc = roc_auc_score(y, p)
        out.update(
            auc=float(auc),
            gini=float(2 * auc - 1),
            ks=ks_stat(y, p),
            pr_auc=float(average_precision_score(y, p)),
        )
    out["brier"] = float(brier_score_loss(y, p))
    out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    if base_rate is not None:
        ref = brier_score_loss(y, np.full(len(y), base_rate))
        out["brier_skill"] = float(1 - out["brier"] / ref) if ref > 0 else np.nan
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with bins from the expected (train) quantiles."""
    e = np.asarray(expected, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe = np.clip(pe, 1e-4, None)
    pa = np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def decile_table(y, p, indemnity=None, premium=None, n_bins: int = 10) -> pd.DataFrame:
    """Lift and loss ratio by score decile (decile 10 = highest risk)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    if indemnity is not None:
        df["ind"] = np.asarray(indemnity)
        df["prem"] = np.asarray(premium)
    df["decile"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    g = df.groupby("decile")
    t = pd.DataFrame(
        {
            "n": g.size(),
            "mean_score": g["p"].mean(),
            "claim_rate": g["y"].mean(),
        }
    )
    base = df["y"].mean()
    t["lift"] = t["claim_rate"] / base
    t["capture"] = g["y"].sum() / df["y"].sum()
    if indemnity is not None:
        t["loss_ratio"] = g["ind"].sum() / g["prem"].sum()
    return t.reset_index()


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin")
    return pd.DataFrame({"mean_pred": g["p"].mean(), "obs_rate": g["y"].mean(), "n": g.size()})
