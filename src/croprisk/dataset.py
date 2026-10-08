"""Cleaning, scope, target and censoring rules -> one modeling table.

Every rule is logged in a cleaning report (rows removed per step), which feeds
DATA_CARD.md. See DECISIONS.md for the reasoning behind each rule.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from croprisk import calendar, config
from croprisk.data.psr import load_policies

log = logging.getLogger(__name__)

OUT_OF_SCOPE_CLASSES = {"PECUÁRIO", "FLORESTAS", "RECEITA", "PARAMÉTRICO"}
TRUNCATED_INSURER = "Too Seguros S.A."  # 2016-2024 file is cut at the Excel row limit
# Claims of a policy are considered fully registered when the critical window ended
# at least this many days before the claims cutoff.
MATURITY_LAG_DAYS = 90
POLICIES_PATH = config.PROCESSED / "policies.parquet"
REPORT_PATH = config.REPORTS / "cleaning_report.json"


def claims_cutoff(df: pd.DataFrame) -> pd.Timestamp:
    """Data-driven estimate of the date up to which claims were registered.

    The public files carry no claim date. The latest critical-window end among
    policies that *have* an indemnity is a lower bound for the claims cutoff; we
    take the 99.5th percentile to be robust to a few odd records.
    """
    paid = df.loc[df["y"] == 1, "wend"].dropna()
    return paid.quantile(0.995)


def build(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    df = load_policies() if df is None else df
    steps: list[dict] = []

    def step(name: str, keep: pd.Series) -> None:
        nonlocal df
        removed = int((~keep).sum())
        steps.append({"step": name, "removed": removed, "remaining": int(keep.sum())})
        df = df[keep].copy()

    steps.append({"step": "raw rows (3 files)", "removed": 0, "remaining": len(df)})
    step("duplicated proposal id", ~df["proposal_id"].duplicated(keep="first"))

    # In the 2006-2015 file start/end dates are a placeholder (2016-07-22/23); the
    # contract date is the best available start. In 2016+ the median gap between
    # contract and coverage start is 0 days.
    old = df["source_file"].str.contains("2006")
    df["t0"] = df["dt_start"].where(~old, df["dt_proposal"])
    df["t0"] = df["t0"].fillna(df["dt_proposal"]).fillna(df["dt_policy"])
    df.loc[old, ["dt_start", "dt_end"]] = pd.NaT

    step(
        "out-of-scope crop (livestock, forest, pasture)",
        ~df["crop"].isin(calendar.OUT_OF_SCOPE_CROPS),
    )
    step(
        "out-of-scope product class (livestock, forest, revenue, parametric)",
        ~df["product_class"].isin(OUT_OF_SCOPE_CLASSES),
    )
    step("crop without calendar", df["crop"].isin(calendar.CROP_GROUP.keys()))
    step("missing municipality code or start date", df["ibge_code"].notna() & df["t0"].notna())
    step(
        "invalid money values (sum insured, premium or rate <= 0, rate >= 1)",
        (df["sum_insured"] > 0) & (df["premium"] > 0) & (df["rate"] > 0) & (df["rate"] < 1),
    )
    implied = df["premium"] / df["sum_insured"]
    step(
        "rate inconsistent with premium / sum insured (>25% apart)",
        (implied / df["rate"]).between(0.75, 1.25),
    )
    # Zero area (mostly fruit) and zero coverage are missing-value codes, not
    # invalid policies: set to missing instead of dropping the row.
    df.loc[~(df["area_ha"] > 0), "area_ha"] = np.nan
    cov = df["coverage_level"]
    cov = cov.where(~cov.between(5, 100), cov / 100)  # stored as percent (55 -> 0.55)
    df["coverage_level"] = cov.where(cov.between(0.05, 1.0))
    for c in ("yield_expected", "yield_insured"):
        df.loc[~(df[c] > 0), c] = np.nan
    steps.append(
        {
            "step": "zero area / coverage / yield set to missing (row kept)",
            "removed": 0,
            "remaining": len(df),
        }
    )
    dur = (df["dt_end"] - df["dt_start"]).dt.days
    step(
        "coverage shorter than 30 days or longer than 2 years (2016+)",
        dur.isna() | dur.between(30, 730),
    )
    step(
        "Too Seguros 2020+ (file truncated inside this insurer's block)",
        ~((df["insurer"] == TRUNCATED_INSURER) & (df["policy_year"] >= 2020)),
    )

    df["crop_group"] = df["crop"].map(calendar.CROP_GROUP)
    win = calendar.assign_windows(df)
    df = df.join(win)
    df["wend"] = df["ws"] + pd.to_timedelta(df["wlen"].astype("float"), unit="D")
    df["y"] = (df["indemnity"].fillna(0) > 0).astype("int8")
    df["loss_ratio_num"] = df["indemnity"].fillna(0.0)

    cutoff = claims_cutoff(df)
    matured = df["wend"] + pd.Timedelta(days=MATURITY_LAG_DAYS) <= cutoff
    # A safra is kept only if nearly all of it is matured; a partially observed
    # safra would look like an unusually good year.
    share = matured.groupby(df["safra_year"]).mean()
    complete = share[share >= 0.98].index
    steps.append(
        {"step": f"claims cutoff estimated at {cutoff.date()}", "removed": 0, "remaining": len(df)}
    )
    step("safra not fully matured (censored)", df["safra_year"].isin(complete))
    step("policy not matured inside a kept safra", matured.loc[df.index])

    df["uf"] = df["uf"].astype("string")
    df = df.drop(columns=["source_file"]).reset_index(drop=True)
    report = {
        "steps": steps,
        "claims_cutoff": str(cutoff.date()),
        "maturity_lag_days": MATURITY_LAG_DAYS,
        "matured_share_by_safra": {int(k): round(float(v), 4) for k, v in share.items()},
        "safras_kept": [int(s) for s in sorted(complete)],
        "rows": len(df),
        "claim_rate": round(float(df["y"].mean()), 4),
    }
    return df, report


def main() -> None:
    config.ensure_dirs()
    df, report = build()
    df.to_parquet(POLICIES_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("policies: %d rows, claim rate %.3f", len(df), df["y"].mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
