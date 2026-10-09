"""Do insurance claim waves precede rural-credit defaults? (optional study)

SCR.data (BCB) gives the rural credit portfolio by UF and month (Jul 2012 on).
Default rate = loans more than 90 days overdue / active portfolio, for crop
financing ("custeio"). For each UF and safra we compare the PSR claim rate with
the change in the custeio default rate between the start of the safra (August)
and twelve months after the harvest season (the following August), when unpaid
custeio loans of that safra are already more than 90 days overdue.

Aggregated public data only; no policy-level information is used here beyond the
UF x safra claim rate.
"""

from __future__ import annotations

import json
import logging
import zipfile

import pandas as pd
import requests
from scipy.stats import spearmanr

from croprisk import calendar, config, dataset, plots

log = logging.getLogger(__name__)

URL = "https://www.bcb.gov.br/pda/desig/scrdata_{year}.zip"
SCR_DIR = config.RAW / "scr"
AGG_PATH = config.PROCESSED / "scr_rural_uf_month.parquet"
YEARS = range(2012, 2026)
USECOLS = [
    "data_base",
    "uf",
    "modalidade",
    "submodalidade",
    "carteira_ativa",
    "carteira_inadimplencia",
]
FOCUS_UFS = ["RS", "PR", "MS", "MT"]
MIN_POLICIES = 1000  # UF x safra cells with fewer policies are too noisy


def download() -> None:
    SCR_DIR.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        p = SCR_DIR / f"scrdata_{y}.zip"
        if not p.exists():
            r = requests.get(
                URL.format(year=y), headers={"User-Agent": config.USER_AGENT}, timeout=900
            )
            r.raise_for_status()
            p.write_bytes(r.content)


def aggregate() -> pd.DataFrame:
    parts = []
    for y in YEARS:
        with zipfile.ZipFile(SCR_DIR / f"scrdata_{y}.zip") as z:
            for name in sorted(z.namelist()):
                df = pd.read_csv(
                    z.open(name), sep=";", dtype=str, usecols=USECOLS, encoding="utf-8-sig"
                )
                df = df[df["modalidade"].str.contains("rurais", case=False, na=False)]
                for c in ("carteira_ativa", "carteira_inadimplencia"):
                    df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
                g = df.groupby(["data_base", "uf", "submodalidade"], as_index=False)[
                    ["carteira_ativa", "carteira_inadimplencia"]
                ].sum()
                parts.append(g)
        log.info("SCR %d aggregated", y)
    out = pd.concat(parts, ignore_index=True)
    out["month"] = pd.to_datetime(out["data_base"]).dt.to_period("M").dt.to_timestamp()
    out["submodalidade"] = (
        out["submodalidade"].str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    )
    out = out.drop(columns="data_base")
    out.to_parquet(AGG_PATH, index=False)
    return out


def run() -> dict:
    download()
    scr = pd.read_parquet(AGG_PATH) if AGG_PATH.exists() else aggregate()
    cust = scr[scr["submodalidade"] == "Custeio"]
    m = cust.groupby(["uf", "month"])[["carteira_ativa", "carteira_inadimplencia"]].sum()
    m["default_rate"] = m["carteira_inadimplencia"] / m["carteira_ativa"]
    m = m.reset_index()

    pol = pd.read_parquet(dataset.POLICIES_PATH, columns=["uf", "safra_year", "y"])
    cr = (
        pol.groupby(["uf", "safra_year"])
        .agg(policies=("y", "size"), claim_rate=("y", "mean"))
        .reset_index()
    )
    cr["uf"] = cr["uf"].astype(str)

    rate = m.set_index(["uf", "month"])["default_rate"]
    rows = []
    for r in cr.itertuples():
        before = pd.Timestamp(int(r.safra_year), 8, 1)
        after = pd.Timestamp(int(r.safra_year) + 2, 8, 1)  # 12 months after the Aug-Jul safra ends
        try:
            d0, d1 = rate.loc[(r.uf, before)], rate.loc[(r.uf, after)]
        except KeyError:
            continue
        rows.append(
            {**r._asdict(), "default_before": d0, "default_after": d1, "delta_pp": 100 * (d1 - d0)}
        )
    t = pd.DataFrame(rows).drop(columns="Index")
    t = t[t["policies"] >= MIN_POLICIES]
    rho, p = spearmanr(t["claim_rate"], t["delta_pp"])
    # within-UF version: is a UF's bad safra followed by a larger rise than its usual?
    t["claim_dev"] = t["claim_rate"] - t.groupby("uf")["claim_rate"].transform("mean")
    t["delta_dev"] = t["delta_pp"] - t.groupby("uf")["delta_pp"].transform("mean")
    rho_w, p_w = spearmanr(t["claim_dev"], t["delta_dev"])
    top = t.sort_values("claim_rate", ascending=False).head(8)
    summary = {
        "uf_safra_cells": int(len(t)),
        "min_policies_per_cell": MIN_POLICIES,
        "spearman_claims_vs_default_change": {"rho": float(rho), "p_value": float(p)},
        "spearman_within_uf": {"rho": float(rho_w), "p_value": float(p_w)},
        "worst_claim_safras": [
            {
                "uf": x.uf,
                "safra": calendar.safra_label(int(x.safra_year)),
                "claim_rate": round(float(x.claim_rate), 3),
                "default_change_pp": round(float(x.delta_pp), 2),
            }
            for x in top.itertuples()
        ],
        "median_default_change_pp": float(t["delta_pp"].median()),
    }
    (config.REPORTS / "scr_credit.json").write_text(json.dumps(summary, indent=2))
    figure(m, cr, t, rho_w)
    log.info("SCR summary %s", summary)
    return summary


def figure(m: pd.DataFrame, cr: pd.DataFrame, t: pd.DataFrame, rho_w: float) -> None:
    import matplotlib.pyplot as plt

    plots.setup()
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
    for ax, uf in zip(axes.flat, FOCUS_UFS, strict=True):
        s = m[m["uf"] == uf]
        c = cr[cr["uf"] == uf]
        # background: claim rate of each safra (Aug-Jul), darker = more claims
        for r in c.itertuples():
            x0 = pd.Timestamp(int(r.safra_year), 8, 1)
            ax.axvspan(
                x0,
                x0 + pd.DateOffset(years=1),
                color=plots.C[1],
                alpha=min(0.6, 1.2 * r.claim_rate),
                linewidth=0,
            )
        ax.plot(s["month"], 100 * s["default_rate"], color=plots.C[0])
        worst = c.sort_values("claim_rate").iloc[-1]
        ax.set_title(
            f"{uf}: worst safra {calendar.safra_label(int(worst.safra_year))} "
            f"({worst.claim_rate:.0%} of policies paid)",
            fontsize=10,
        )
        ax.set_ylabel("custeio > 90 days overdue (%)")
        ax.set_xlim(pd.Timestamp("2012-08-01"), s["month"].max())
    fig.suptitle(
        "Rural crop-loan default (line) and PSR claim rate by safra (shading: darker = more claims)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.tight_layout()
    plots.save(fig, config.FIGURES / "scr_default_vs_claims.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
