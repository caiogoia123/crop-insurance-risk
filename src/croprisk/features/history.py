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
    """df: ibge_code, uf, crop_group, crop, safra_year, y, yield_expected."""
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
    b["p_all"] = (b["k_all"] / b["n_all"]).fillna(df["y"].mean() if len(df) else 0.15)
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

    # expected yield relative to the past median for the same crop and UF
    med = []
    for s in np.sort(df["safra_year"].unique()):
        past = df[df["safra_year"] <= s - 1 - gap]
        m = past.groupby(["uf", "crop"], observed=True)["yield_expected"].median().rename("yld_med")
        med.append(m.reset_index().assign(safra_year=s))
    med = pd.concat(med, ignore_index=True)
    y = base.merge(med, on=["uf", "crop", "safra_year"], how="left")
    out["yield_rel"] = df["yield_expected"].to_numpy() / y["yld_med"].to_numpy()
    return out


def lookup_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """History rates using every matured safra, for scoring new contracts (API)."""
    nxt = int(df["safra_year"].max()) + 1
    tmp = df.copy()
    probe = df.drop_duplicates(["ibge_code", "crop_group"])[
        ["ibge_code", "uf", "crop_group", "crop"]
    ].assign(safra_year=nxt, y=0, yield_expected=np.nan)
    feats = history_features(pd.concat([tmp, probe], ignore_index=True), gap=0).iloc[len(tmp) :]
    probe = probe.reset_index(drop=True)
    feats = feats.reset_index(drop=True)
    muni = pd.concat([probe[["ibge_code", "uf", "crop_group"]], feats], axis=1)
    muni = muni.drop(columns=["yield_rel"])
    yld = (
        df.groupby(["uf", "crop"], observed=True)["yield_expected"]
        .median()
        .rename("yld_med")
        .reset_index()
    )
    return {"muni": muni, "yield": yld}
