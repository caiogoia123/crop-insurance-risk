"""SHAP explanations for the last walk-forward fold (train <= 2022/23, explain 2023/24).

TreeSHAP values come from LightGBM itself (`pred_contrib=True`); the shap package
is only used to draw the beeswarm.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from croprisk import config, models, plots, tracking
from croprisk.backtest import TUNED_SETS, TUNING_PATH, load_features

log = logging.getLogger(__name__)

N_EXPLAIN = 20_000

LABELS = {
    "hist_muni_cg_rate": "Municipality loss history (crop group)",
    "hist_uf_cg_rate": "State loss history (crop group)",
    "hist_cg_rate": "Crop group loss history",
    "hist_muni_rate": "Municipality loss history (all crops)",
    "hist_muni_cg_n": "Policies in municipality history",
    "hist_uf_cg_n": "Policies in state history",
    "crop": "Crop",
    "crop_group": "Crop group",
    "uf": "State",
    "region": "Region",
    "insurer": "Insurer",
    "product_class": "Product type",
    "log_sum_insured": "Sum insured",
    "log_area": "Area",
    "log_si_per_ha": "Sum insured per ha",
    "coverage_level": "Coverage level",
    "yield_expected": "Expected yield",
    "yield_insured_ratio": "Insured / expected yield",
    "yield_rel": "Expected yield vs state median",
    "contract_month": "Contract month",
    "lead_days": "Days from contract to critical window",
    "oni": "ENSO index (ONI) at contract",
    "oni_change_3m": "ONI change, 3 months",
    "clim_prec_mean": "Normal rain in window",
    "clim_prec_cv": "Rain variability in window",
    "clim_prec_p10_ratio": "Dry-year rain / normal",
    "clim_drought_freq": "Drought frequency 1981-2005",
    "clim_cdd": "Typical longest dry spell",
    "clim_max5d": "Typical max 5-day rain",
    "clim_hot_days": "Typical hot days",
    "clim_frost_days": "Typical frost-risk days",
    "clim_frost_freq": "Years with frost risk",
    "clim_tmax": "Normal max temperature",
    "clim_tmin": "Normal min temperature",
    "clim_gwet": "Normal root-zone soil moisture",
    "clim_gwet_min": "Normal minimum soil moisture",
    "ante_prec_anom": "Rain anomaly, 30 days before window",
    "ante_gwet_anom": "Soil moisture anomaly before window",
}
for _p, _w in [("obs_h50_", "first half of window"), ("obs_full_", "whole window")]:
    LABELS.update(
        {
            f"{_p}prec_anom": f"Rain anomaly, {_w}",
            f"{_p}prec_z": f"Rain z-score, {_w}",
            f"{_p}cdd": f"Longest dry spell, {_w}",
            f"{_p}cdd_anom": f"Dry spell anomaly, {_w}",
            f"{_p}max5d_anom": f"Max 5-day rain anomaly, {_w}",
            f"{_p}hot_days": f"Hot days, {_w}",
            f"{_p}hot_anom": f"Hot days anomaly, {_w}",
            f"{_p}frost_days": f"Frost-risk days, {_w}",
            f"{_p}frost_anom": f"Frost-risk days anomaly, {_w}",
            f"{_p}tmax_anom": f"Max temperature anomaly, {_w}",
            f"{_p}tmin_anom": f"Min temperature anomaly, {_w}",
            f"{_p}gwet_anom": f"Soil moisture anomaly, {_w}",
            f"{_p}gwet_min": f"Minimum soil moisture, {_w}",
        }
    )


def shap_values(fs_name: str, df: pd.DataFrame, test_safra: int):
    tuning = json.loads(TUNING_PATH.read_text())[TUNED_SETS[fs_name]]
    fs = models.FEATURE_SETS[fs_name]
    tr = df[df["safra_year"] <= test_safra - 1]
    te = df[df["safra_year"] == test_safra].sample(N_EXPLAIN, random_state=config.SEED)
    m = models.LGBModel(fs, tuning["params"], tuning["rounds"]).fit(tr, tr["y"].to_numpy())
    contrib = m.booster.predict(te[fs.columns], pred_contrib=True)
    return fs, te, contrib[:, :-1]


def fig_bar(fs, sv, name, title) -> pd.Series:
    import matplotlib.pyplot as plt

    imp = pd.Series(np.abs(sv).mean(axis=0), index=fs.columns).sort_values(ascending=False)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh([LABELS.get(c, c) for c in top.index], top.values, height=0.6, color=plots.C[0])
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    plots.save(fig, config.FIGURES / f"shap_{name}_bar.png")
    return imp


def fig_beeswarm(fs, te, sv, name, title) -> None:
    import matplotlib.pyplot as plt
    import shap

    X = te[fs.columns].copy()
    for c in fs.cat:
        X[c] = X[c].cat.codes.replace(-1, np.nan)
    exp = shap.Explanation(
        values=sv,
        data=X.to_numpy(dtype=float),
        feature_names=[LABELS.get(c, c) for c in fs.columns],
    )
    plt.figure()
    shap.plots.beeswarm(exp, max_display=14, show=False, plot_size=(8, 5.2))
    fig = plt.gcf()
    fig.axes[0].set_title(title, loc="left", fontsize=11, fontweight="semibold")
    plots.save(fig, config.FIGURES / f"shap_{name}_beeswarm.png")


def main() -> None:
    plots.setup()
    df = load_features()
    test = int(df["safra_year"].max())
    out = {}
    for fs_name, title in [
        ("pre_season", "Pre-season model: what drives the score (test safra 2023/24)"),
        ("in_season", "In-season model: what drives the score (test safra 2023/24)"),
    ]:
        fs, te, sv = shap_values(fs_name, df, test)
        imp = fig_bar(fs, sv, fs_name, title)
        fig_beeswarm(fs, te, sv, fs_name, title)
        groups = {
            "contract": [
                c
                for c in fs.columns
                if not c.startswith(("hist_", "clim_", "oni", "obs_", "ante_"))
            ],
            "history": [c for c in fs.columns if c.startswith("hist_")],
            "climatology+ENSO": [c for c in fs.columns if c.startswith(("clim_", "oni"))],
            "season weather": [c for c in fs.columns if c.startswith(("obs_", "ante_"))],
        }
        share = {g: float(imp[cols].sum() / imp.sum()) for g, cols in groups.items() if cols}
        out[fs_name] = {"top10": imp.head(10).round(4).to_dict(), "share_by_block": share}
        log.info("%s SHAP share by block: %s", fs_name, share)
    (config.REPORTS / "shap_summary.json").write_text(json.dumps(out, indent=2))
    if tracking.setup("crop-insurance-risk-evaluation"):
        with tracking.run("explain"):
            for p in config.FIGURES.glob("shap_*.png"):
                tracking.log_artifact(p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
