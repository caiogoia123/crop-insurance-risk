"""As-of portfolio history: past loss rates by municipality and crop group.

For a policy in safra s, only outcomes of safras <= s - 1 - gap are used, so the
feature is the same whether the row is in a training or in a test fold.
Rates are shrunk toward the next level up (municipality -> UF -> crop group ->
all), and municipality-level rates based on fewer than MIN_N policies fall back to
the UF rate (this is also the privacy rule for the lookup table served by the API).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHRINK = 50.0  # pseudo-policies of the prior
MIN_N = 10  # municipality cells with fewer policies use the UF rate


def _asof_counts(df: pd.DataFrame, keys: list[str], gap: int) -> pd.DataFrame:
    """Cumulative (n, k) per key using safras strictly before s - gap."""
    g = df.groupby(keys + ["safra_year"], observed=True)["y"].agg(n="size", k="sum").reset_index()
    safras = np.sort(df["safra_year"].unique())
    full = []
    for s in safras:
        past = g[g["safra_year"] <= s - 1 - gap]
        agg = past.groupby(keys, observed=True)[["n", "k"]].sum().reset_index()
        agg["safra_year"] = s
        full.append(agg)
    return pd.concat(full, ignore_index=True)


def _rate(n, k, prior):
    return (k + SHRINK * prior) / (n + SHRINK)


def history_features(df: pd.DataFrame, gap: int = 0) -> pd.DataFrame:
    """df: ibge_code, uf, crop_group, crop, safra_year, y."""
    base = df[["ibge_code", "uf", "crop_group", "crop", "safra_year"]].copy()
    out = pd.DataFrame(index=df.index)

    allc = _asof_counts(df.assign(_all=1), ["_all"], gap)
    cg = _asof_counts(df, ["crop_group"], gap)
    ufcg = _asof_counts(df, ["uf", "crop_group"], gap)
    mcg = _asof_counts(df, ["ibge_code", "crop_group"], gap)
    m_all = _asof_counts(df, ["ibge_code"], gap)

    b = base.assign(_all=1).merge(
        allc.rename(columns={"n": "n_all", "k": "k_all"}), on=["_all", "safra_year"], how="left"
    )
    # no history at all (first safra): missing, never a mean that includes the future
    b["p_all"] = b["k_all"] / b["n_all"]
    b = b.merge(
        cg.rename(columns={"n": "n_cg", "k": "k_cg"}), on=["crop_group", "safra_year"], how="left"
    )
    b = b.merge(
        ufcg.rename(columns={"n": "n_uf", "k": "k_uf"}),
        on=["uf", "crop_group", "safra_year"],
        how="left",
    )
    b = b.merge(
        mcg.rename(columns={"n": "n_m", "k": "k_m"}),
        on=["ibge_code", "crop_group", "safra_year"],
        how="left",
    )
    b = b.merge(
        m_all.rename(columns={"n": "n_ma", "k": "k_ma"}), on=["ibge_code", "safra_year"], how="left"
    )
    for c in ["n_cg", "k_cg", "n_uf", "k_uf", "n_m", "k_m", "n_ma", "k_ma"]:
        b[c] = b[c].fillna(0)
    p_cg = _rate(b["n_cg"], b["k_cg"], b["p_all"])
    p_uf = _rate(b["n_uf"], b["k_uf"], p_cg)
    p_m = _rate(b["n_m"], b["k_m"], p_uf)
    p_m = np.where(b["n_m"] >= MIN_N, p_m, p_uf)
    p_ma = _rate(b["n_ma"], b["k_ma"], b["p_all"])
    p_ma = np.where(b["n_ma"] >= MIN_N, p_ma, np.nan)

    out["hist_cg_rate"] = p_cg.to_numpy()
    out["hist_uf_cg_rate"] = p_uf.to_numpy()
    out["hist_muni_cg_rate"] = np.asarray(p_m)
    out["hist_muni_rate"] = np.asarray(p_ma)
    out["hist_muni_cg_n"] = np.log1p(b["n_m"].to_numpy())
    out["hist_uf_cg_n"] = np.log1p(b["n_uf"].to_numpy())
    out.index = df.index

    return out


def lookup_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """History tables for scoring new contracts (API), as of every matured safra.

    Built by scoring a "probe" row for each known (municipality, crop) at the
    next safra with `history_features`, so the API uses exactly the training math.
    Only aggregates leave this function; municipality rates with n < MIN_N have
    already fallen back to the UF rate.
    """
    nxt = int(df["safra_year"].max()) + 1
    cols = ["ibge_code", "uf", "crop_group", "crop", "safra_year", "y"]
    probe = (
        df.drop_duplicates(["ibge_code", "crop_group", "crop"])[
            ["ibge_code", "uf", "crop_group", "crop"]
        ]
        .assign(safra_year=nxt, y=0)
        .reset_index(drop=True)
    )
    both = pd.concat([df[cols], probe[cols]], ignore_index=True)
    feats = history_features(both, gap=0).iloc[len(df) :].reset_index(drop=True)
    pr = pd.concat([probe, feats], axis=1)
    hist_muni = pr.drop_duplicates(["ibge_code", "crop_group"])[
        ["ibge_code", "crop_group", "hist_muni_cg_rate", "hist_muni_cg_n"]
    ]
    hist_muni = hist_muni[hist_muni["hist_muni_cg_n"] > 0]
    hist_uf = pr.drop_duplicates(["uf", "crop_group"])[
        ["uf", "crop_group", "hist_uf_cg_rate", "hist_uf_cg_n"]
    ]
    hist_cg = pr.drop_duplicates("crop_group")[["crop_group", "hist_cg_rate"]]
    hist_muni_all = pr.drop_duplicates("ibge_code")[["ibge_code", "hist_muni_rate"]].dropna()
    return {
        "hist_muni": hist_muni.reset_index(drop=True),
        "hist_uf": hist_uf.reset_index(drop=True),
        "hist_cg": hist_cg.reset_index(drop=True),
        "hist_muni_all": hist_muni_all.reset_index(drop=True),
    }
