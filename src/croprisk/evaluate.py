"""Tables and figures from the walk-forward predictions (reports/ and reports/figures/).

Everything here is computed from out-of-time predictions: each policy is scored by
a model trained only on earlier safras.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from croprisk import calendar, config, metrics, plots, tracking
from croprisk.backtest import FOLD_METRICS_PATH, OOF_PATH
from croprisk.features.build import FEATURES_PATH

log = logging.getLogger(__name__)

MAIN = [
    "mean_rate",
    "insurer_rate",
    "logreg_contract",
    "lgbm_contract_only",
    "lgbm_contract",
    "logreg_pre_season",
    "lgbm_pre_season",
    "lgbm_pre_season_rate",
    "logreg_in_season",
    "lgbm_in_season",
    "lgbm_end_of_window",
]
SENS = [
    "lgbm_contract_gap1",
    "lgbm_pre_season_gap1",
    "lgbm_in_season_gap1",
    "lgbm_pre_season_roll8",
]
LABEL = {
    "mean_rate": "Mean claim rate (constant)",
    "insurer_rate": "Insurer's rate (PE_TAXA)",
    "logreg_contract": "Logistic, contract + history",
    "lgbm_contract_only": "LightGBM, contract only",
    "lgbm_contract": "LightGBM, contract + history",
    "logreg_pre_season": "Logistic, pre-season (+ climatology)",
    "lgbm_pre_season": "LightGBM, pre-season (+ climatology, ENSO)",
    "lgbm_pre_season_rate": "LightGBM, pre-season + insurer's rate",
    "logreg_in_season": "Logistic, in-season (mid-window weather)",
    "lgbm_in_season": "LightGBM, in-season (mid-window weather)",
    "lgbm_end_of_window": "LightGBM, end of critical window",
    "lgbm_contract_gap1": "LightGBM contract + history, 1-safra gap",
    "lgbm_pre_season_gap1": "LightGBM pre-season, 1-safra gap",
    "lgbm_in_season_gap1": "LightGBM in-season, 1-safra gap",
    "lgbm_pre_season_roll8": "LightGBM pre-season, rolling 8 safras",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_parquet(OOF_PATH)
    cols = [
        "proposal_id",
        "y",
        "indemnity",
        "premium",
        "rate",
        "crop_group",
        "uf",
        "event",
        "safra_year",
    ]
    f = pd.read_parquet(FEATURES_PATH, columns=cols)
    df = oof.merge(f.drop(columns="safra_year"), on="proposal_id", how="left", validate="1:1")
    df["indemnity"] = df["indemnity"].fillna(0.0)
    folds = pd.read_csv(FOLD_METRICS_PATH)
    return df, folds


def within_group_auc(df: pd.DataFrame, col: str, by: list[str]) -> float:
    """Exposure-weighted mean AUC inside groups (e.g. crop group x safra)."""
    aucs, w = [], []
    for _, g in df.groupby(by, observed=True):
        if 0 < g["y"].sum() < len(g) and len(g) >= 200:
            aucs.append(roc_auc_score(g["y"], g[col]))
            w.append(len(g))
    return float(np.average(aucs, weights=w))


def summary_table(df: pd.DataFrame, folds: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    rate_auc = folds[folds["experiment"] == "insurer_rate"].set_index("safra_year")["auc"]
    for n in names:
        f = folds[folds["experiment"] == n].set_index("safra_year")
        if f.empty:
            continue
        pooled = metrics.binary_metrics(df["y"], df[n])
        rows.append(
            {
                "experiment": n,
                "model": LABEL[n],
                "auc_mean": f["auc"].mean(),
                "auc_sd": f["auc"].std(),
                "auc_min": f["auc"].min(),
                "ks_mean": f["ks"].mean(),
                "gini_mean": f["gini"].mean(),
                "pr_auc_mean": f["pr_auc"].mean(),
                "brier_mean": f["brier"].mean(),
                "brier_skill_mean": f["brier_skill"].mean(),
                "auc_within_crop": within_group_auc(df, n, ["safra_year", "crop_group"]),
                "pooled_auc": pooled.get("auc", np.nan),
                "safras_beating_rate": int((f["auc"] > rate_auc.reindex(f.index)).sum()),
                "n_safras": len(f),
                "psi_score_mean": f["psi_score"].mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_delta(folds: pd.DataFrame, a: str, b: str, metric: str = "auc") -> dict:
    fa = folds[folds["experiment"] == a].set_index("safra_year")[metric]
    fb = folds[folds["experiment"] == b].set_index("safra_year")[metric]
    d = (fa - fb).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "mean_delta": float(d.mean()),
        "ci95": [float(d.mean() - 2.228 * se), float(d.mean() + 2.228 * se)],  # t(10) for 11 safras
        "safras_a_better": int((d > 0).sum()),
        "n": int(len(d)),
        "by_safra": {int(k): round(float(v), 4) for k, v in d.items()},
    }


def deciles_within_safra(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("safra_year")[col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1
    )


def decile_report(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.assign(decile=deciles_within_safra(df, col))
    g = d.groupby("decile")
    t = pd.DataFrame(
        {
            "policies": g.size(),
            "claim_rate": g["y"].mean(),
            "mean_score": g[col].mean(),
            "loss_ratio": g["indemnity"].sum() / g["premium"].sum(),
            "share_of_claims": g["y"].sum() / d["y"].sum(),
            "share_of_indemnity": g["indemnity"].sum() / d["indemnity"].sum(),
        }
    )
    t["lift"] = t["claim_rate"] / d["y"].mean()
    return t.reset_index()


def double_lift(df: pd.DataFrame, model: str, ref: str = "insurer_rate") -> pd.DataFrame:
    """Sort by model/ref ratio (within safra): if the model adds information the
    insurer's price does not have, the loss ratio rises across the deciles."""
    d = df.copy()
    d["ratio"] = d[model] / d[ref]
    d["decile"] = deciles_within_safra(d, "ratio")
    g = d.groupby("decile")
    return pd.DataFrame(
        {
            "claim_rate": g["y"].mean(),
            "model_mean": g[model].mean(),
            "rate_implied_mean": g[ref].mean(),
            "loss_ratio": g["indemnity"].sum() / g["premium"].sum(),
            "mean_rate_charged": g["rate"].mean(),
        }
    ).reset_index()


def fmt_table(t: pd.DataFrame) -> str:
    out = [
        "| Model | AUC mean (sd) | worst safra | KS | Gini | PR-AUC | Brier | AUC within crop | safras > insurer rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t.itertuples():
        out.append(
            f"| {r.model} | {r.auc_mean:.3f} ({r.auc_sd:.3f}) | {r.auc_min:.3f} | {r.ks_mean:.3f} | "
            f"{r.gini_mean:.3f} | {r.pr_auc_mean:.3f} | {r.brier_mean:.4f} | {r.auc_within_crop:.3f} | "
            f"{r.safras_beating_rate}/{r.n_safras} |"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------- figures
def fig_auc_by_safra(folds: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    series = [
        ("insurer_rate", plots.C[1]),
        ("lgbm_contract", plots.C[3]),
        ("lgbm_pre_season", plots.C[0]),
        ("lgbm_in_season", plots.C[2]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for n, c in series:
        f = folds[folds["experiment"] == n].sort_values("safra_year")
        x = [calendar.safra_label(s) for s in f["safra_year"]]
        ax.plot(
            x,
            f["auc"],
            color=c,
            label=LABEL[n],
            marker="o",
            markersize=4,
            markeredgecolor=plots.SURFACE,
            markeredgewidth=1,
        )
    ax.axhline(0.5, color=plots.MUTED, linewidth=1)
    ax.text(0, 0.505, "random ranking", color=plots.MUTED, fontsize=8)
    ax.set_ylabel("AUC (test safra)")
    ax.set_title("Out-of-time AUC by safra: every model is trained only on earlier safras")
    ax.legend(loc="lower left", ncols=2)
    ax.tick_params(axis="x", rotation=45)
    plots.save(fig, config.FIGURES / "auc_by_safra.png")


def fig_calibration(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    ax = axes[0]
    for n, c in [
        ("insurer_rate", plots.C[1]),
        ("lgbm_pre_season", plots.C[0]),
        ("lgbm_in_season", plots.C[2]),
    ]:
        t = metrics.calibration_table(df["y"], df[n], n_bins=20)
        ax.plot(
            t["mean_pred"],
            t["obs_rate"],
            color=c,
            marker="o",
            markersize=4,
            label=LABEL[n],
            markeredgecolor=plots.SURFACE,
            markeredgewidth=1,
        )
    lim = 0.75
    ax.plot([0, lim], [0, lim], color=plots.MUTED, linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mean predicted probability (20 bins)")
    ax.set_ylabel("observed claim rate")
    ax.set_title("Calibration, all test safras pooled")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    g = df.groupby("safra_year")
    x = [calendar.safra_label(s) for s in g.size().index]
    ax.plot(x, g["y"].mean(), color=plots.INK, label="observed")
    ax.plot(x, g["lgbm_pre_season"].mean(), color=plots.C[0], label="pre-season model")
    ax.plot(x, g["lgbm_in_season"].mean(), color=plots.C[2], label="in-season model")
    ax.set_ylabel("claim rate")
    ax.set_title("Claim rate per safra: observed vs predicted")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    plots.save(fig, config.FIGURES / "calibration.png")


def fig_ks(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, n in zip(axes, ["lgbm_pre_season", "lgbm_in_season"], strict=True):
        # score as a within-safra percentile so the safras can be pooled
        pct = df.groupby("safra_year")[n].rank(pct=True)
        q = np.linspace(0, 1, 201)
        pos = np.searchsorted(np.sort(pct[df["y"] == 1]), q, side="right") / (df["y"] == 1).sum()
        neg = np.searchsorted(np.sort(pct[df["y"] == 0]), q, side="right") / (df["y"] == 0).sum()
        i = int(np.argmax(np.abs(neg - pos)))
        ax.plot(q, neg, color=plots.C[0], label="no claim")
        ax.plot(q, pos, color=plots.C[1], label="claim")
        ax.vlines(q[i], pos[i], neg[i], color=plots.INK2, linewidth=1)
        ax.text(q[i] + 0.02, (pos[i] + neg[i]) / 2, f"KS = {neg[i] - pos[i]:.2f}", fontsize=9)
        ax.set_xlabel("score percentile within safra")
        ax.set_title(LABEL[n])
    axes[0].set_ylabel("cumulative share of policies")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    plots.save(fig, config.FIGURES / "ks.png")


def fig_deciles(dec: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    names = [
        ("insurer_rate", plots.C[1]),
        ("lgbm_pre_season", plots.C[0]),
        ("lgbm_in_season", plots.C[2]),
    ]
    w = 0.26
    for k, (n, c) in enumerate(names):
        t = dec[n]
        x = t["decile"] + (k - 1) * w
        axes[0].bar(x, t["claim_rate"], width=w * 0.92, color=c, label=LABEL[n])
        axes[1].bar(x, t["loss_ratio"], width=w * 0.92, color=c, label=LABEL[n])
    axes[0].set_title("Claim rate by score decile")
    axes[0].set_ylabel("claim rate")
    axes[1].set_title("Loss ratio (indemnity / premium) by score decile")
    axes[1].axhline(1.0, color=plots.MUTED, linewidth=1)
    for ax in axes:
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("decile within safra (10 = highest risk)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    plots.save(fig, config.FIGURES / "deciles.png")


def fig_double_lift(dl: pd.DataFrame, model: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(
        dl["decile"],
        dl["loss_ratio"],
        color=plots.C[0],
        marker="o",
        markersize=5,
        markeredgecolor=plots.SURFACE,
        markeredgewidth=1,
    )
    ax.axhline(1.0, color=plots.MUTED, linewidth=1)
    for i in (0, len(dl) - 1):
        ax.annotate(
            f"{dl['loss_ratio'].iloc[i]:.2f}",
            (dl["decile"].iloc[i], dl["loss_ratio"].iloc[i]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=plots.INK2,
        )
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("decile of model risk / rate-implied risk (within safra)")
    ax.set_ylabel("loss ratio (indemnity / premium)")
    ax.set_title(f"Double lift: {LABEL[model]} vs the insurer's rate")
    plots.save(fig, config.FIGURES / "double_lift.png")


def main() -> None:
    plots.setup()
    df, folds = load()
    main_t = summary_table(df, folds, MAIN)
    sens_t = summary_table(df, folds, ["lgbm_contract", "lgbm_pre_season", "lgbm_in_season"] + SENS)
    deltas = {
        "climatology_vs_contract": paired_delta(folds, "lgbm_pre_season", "lgbm_contract"),
        "in_season_vs_pre_season": paired_delta(folds, "lgbm_in_season", "lgbm_pre_season"),
        "pre_season_vs_insurer_rate": paired_delta(folds, "lgbm_pre_season", "insurer_rate"),
        "contract_vs_insurer_rate": paired_delta(folds, "lgbm_contract", "insurer_rate"),
        "in_season_vs_insurer_rate": paired_delta(folds, "lgbm_in_season", "insurer_rate"),
        "rate_feature_vs_pre_season": paired_delta(
            folds, "lgbm_pre_season_rate", "lgbm_pre_season"
        ),
        "lgbm_vs_logreg_pre_season": paired_delta(folds, "lgbm_pre_season", "logreg_pre_season"),
        "history_vs_contract_only": paired_delta(folds, "lgbm_contract", "lgbm_contract_only"),
        "gap1_cost_pre_season": paired_delta(folds, "lgbm_pre_season_gap1", "lgbm_pre_season"),
    }
    dec = {n: decile_report(df, n) for n in ["insurer_rate", "lgbm_pre_season", "lgbm_in_season"]}
    dl = double_lift(df, "lgbm_pre_season")
    by_crop = (
        pd.DataFrame(
            {
                n: {
                    cg: within_group_auc(g, n, ["safra_year"])
                    for cg, g in df.groupby("crop_group")
                    if g["y"].sum() > 500
                }
                for n in ["insurer_rate", "lgbm_contract", "lgbm_pre_season", "lgbm_in_season"]
            }
        )
        .round(3)
        .sort_index()
    )
    top = df.assign(dec=deciles_within_safra(df, "lgbm_in_season"))
    by_event = (
        top[top["y"] == 1]
        .groupby("event")
        .agg(claims=("y", "size"), share_top2_deciles=("dec", lambda s: float((s >= 9).mean())))
        .sort_values("claims", ascending=False)
        .head(8)
    )
    psi_t = folds[folds["experiment"].isin(["lgbm_pre_season", "lgbm_in_season"])].pivot(
        index="safra_year", columns="experiment", values="psi_score"
    )
    summary = {
        "test_safras": [calendar.safra_label(int(s)) for s in sorted(df["safra_year"].unique())],
        "test_policies": int(len(df)),
        "test_claim_rate": float(df["y"].mean()),
        "main": main_t.round(4).to_dict("records"),
        "sensitivity": sens_t.round(4).to_dict("records"),
        "deltas": deltas,
        "deciles": {k: v.round(4).to_dict("records") for k, v in dec.items()},
        "double_lift_pre_season_vs_rate": dl.round(4).to_dict("records"),
        "auc_by_crop_group": by_crop.to_dict(),
        "in_season_claims_by_event": by_event.round(3).reset_index().to_dict("records"),
        "psi_score_by_safra": psi_t.round(4).reset_index().to_dict("records"),
    }
    (config.REPORTS / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    md = [
        "# Backtest results (generated by `make evaluate`)",
        "",
        f"Test safras {summary['test_safras'][0]} to {summary['test_safras'][-1]}, "
        f"{len(df):,} policies, claim rate {df['y'].mean():.1%}.",
        "",
        "## Main models",
        "",
        fmt_table(main_t),
        "",
        "## Sensitivity (label gap, rolling window)",
        "",
        fmt_table(sens_t),
        "",
        "## Paired differences in AUC across safras",
        "",
        "| Comparison | mean delta AUC | 95% CI | safras better |",
        "|---|---|---|---|",
    ]
    for k, d in deltas.items():
        md.append(
            f"| {k} | {d['mean_delta']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
            f"{d['safras_a_better']}/{d['n']} |"
        )
    md += ["", "## AUC within crop group (mean over safras)", "", by_crop.to_markdown(), ""]
    md += ["## Deciles (within safra)", ""]
    for k, v in dec.items():
        md += [f"### {LABEL[k]}", "", v.round(3).to_markdown(index=False), ""]
    md += [
        "## Double lift: pre-season model vs insurer's rate",
        "",
        dl.round(3).to_markdown(index=False),
        "",
    ]
    md += [
        "## In-season model: claims in the top two deciles, by cause",
        "",
        by_event.to_markdown(),
        "",
    ]
    md += ["## Score PSI, train vs test safra", "", psi_t.round(3).to_markdown(), ""]
    (config.REPORTS / "results.md").write_text("\n".join(md), encoding="utf-8")

    fig_auc_by_safra(folds)
    fig_calibration(df)
    fig_ks(df)
    fig_deciles(dec)
    fig_double_lift(dl, "lgbm_pre_season")

    if tracking.setup("crop-insurance-risk-evaluation"):
        with tracking.run("evaluate"):
            for r in main_t.itertuples():
                tracking.log_metrics(
                    {f"{r.experiment}_auc_mean": r.auc_mean, f"{r.experiment}_brier": r.brier_mean}
                )
            for p in sorted(config.FIGURES.glob("*.png")):
                tracking.log_artifact(p)
            tracking.log_artifact(config.REPORTS / "metrics_summary.json")
    log.info("\n%s", fmt_table(main_t))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
