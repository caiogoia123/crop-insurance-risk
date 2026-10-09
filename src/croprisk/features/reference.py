"""Contract values relative to recent market references (no labels involved).

Sum insured per hectare grew 8x in nominal R$ between 2006 and 2023 (soy: R$695 to
R$5,476), and expected yields trend upward. Raw values drift out of the range the
model was trained on, so contracts are described relative to the median of the
same crop and UF in the previous REF_WINDOW safras. These medians use contract
fields only (no claims), so even censored safras can serve as a reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REF_WINDOW = 3


def reference_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Per (uf, crop, safra_year) medians, plus national rows with uf='*'."""
    d = df.assign(siha=df["sum_insured"] / df["area_ha"])
    agg = {
        "yld_med": ("yield_expected", "median"),
        "siha_med": ("siha", "median"),
        "n": ("siha", "size"),
    }
    by_uf = d.groupby(["uf", "crop", "safra_year"], observed=True).agg(**agg).reset_index()
    nat = d.groupby(["crop", "safra_year"], observed=True).agg(**agg).reset_index().assign(uf="*")
    out = pd.concat([by_uf, nat], ignore_index=True)
    out["uf"] = out["uf"].astype(str)
    out["crop"] = out["crop"].astype(str)
    return out


def reference_for(med: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Reference medians for rows of keys (uf, crop, safra_year): median of the
    per-safra medians over the REF_WINDOW safras before safra_year (UF level when
    available, national otherwise)."""
    k = keys[["uf", "crop", "safra_year"]].astype({"uf": str, "crop": str})
    out = pd.DataFrame(index=keys.index, columns=["yld_ref", "siha_ref"], dtype="float64")
    for s in np.unique(k["safra_year"]):
        past = med[(med["safra_year"] >= s - REF_WINDOW) & (med["safra_year"] <= s - 1)]
        ref = past.groupby(["uf", "crop"])[["yld_med", "siha_med"]].median()
        rows = k["safra_year"] == s
        sub = k[rows]
        uf_ref = ref.reindex(pd.MultiIndex.from_frame(sub[["uf", "crop"]]))
        nat_ref = ref.reindex(pd.MultiIndex.from_arrays([["*"] * len(sub), sub["crop"]]))
        vals = np.where(np.isnan(uf_ref.to_numpy()), nat_ref.to_numpy(), uf_ref.to_numpy())
        out.loc[rows, ["yld_ref", "siha_ref"]] = vals
    return out


def relative_features(df: pd.DataFrame, med: pd.DataFrame) -> pd.DataFrame:
    """yield_rel and si_per_ha_rel for policies (uf, crop, safra_year, values)."""
    ref = reference_for(med, df)
    return pd.DataFrame(
        {
            "yield_rel": (df["yield_expected"] / ref["yld_ref"]).clip(0, 5),
            "si_per_ha_rel": (df["sum_insured"] / df["area_ha"] / ref["siha_ref"]).clip(0, 10),
        },
        index=df.index,
    )
