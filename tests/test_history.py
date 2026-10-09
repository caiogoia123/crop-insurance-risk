import numpy as np
import pandas as pd

from croprisk.features import history


def _policies(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ibge_code": rng.choice([1, 2, 3, 4], n),
            "uf": "PR",
            "crop_group": rng.choice(["soja", "milho_2"], n),
            "crop": "Soja",
            "safra_year": rng.choice(range(2010, 2016), n),
            "y": (rng.random(n) < 0.2).astype(int),
            "yield_expected": rng.uniform(2000, 4000, n),
        }
    )


def test_features_do_not_use_current_or_future_safras():
    df = _policies()
    f0 = history.history_features(df)
    changed = df.copy()
    late = changed["safra_year"] >= 2013
    changed.loc[late, "y"] = 1 - changed.loc[late, "y"]
    f1 = history.history_features(changed)
    early = df["safra_year"] <= 2013  # 2013 rows only see <= 2012
    pd.testing.assert_frame_equal(f0[early], f1[early])
    assert not f0[~early].equals(f1[~early])


def test_gap_skips_the_previous_safra():
    df = _policies()
    changed = df.copy()
    changed.loc[changed["safra_year"] == 2014, "y"] = 1
    rows = df["safra_year"] == 2015
    g0 = history.history_features(df, gap=1)[rows]
    g1 = history.history_features(changed, gap=1)[rows]
    pd.testing.assert_frame_equal(g0, g1)


def test_first_safra_has_only_the_prior():
    df = _policies()
    f = history.history_features(df)
    first = df["safra_year"] == df["safra_year"].min()
    assert (f.loc[first, "hist_muni_cg_n"] == 0).all()
    assert f.loc[first, "hist_muni_rate"].isna().all()


def test_small_municipality_cells_fall_back_to_uf():
    df = _policies()
    df.loc[df["ibge_code"] == 4, "safra_year"] = 2015  # municipality 4 only in last safra
    df = pd.concat([df, df[df["ibge_code"] == 4].head(3).assign(safra_year=2014)])
    f = history.history_features(df.reset_index(drop=True))
    rows = (df.reset_index(drop=True)["ibge_code"] == 4) & (
        df.reset_index(drop=True)["safra_year"] == 2015
    )
    assert np.allclose(f.loc[rows, "hist_muni_cg_rate"], f.loc[rows, "hist_uf_cg_rate"])


def test_lookup_tables_match_training_math():
    df = _policies()
    tabs = history.lookup_tables(df)
    nxt = df.assign(safra_year=2016).drop_duplicates(["ibge_code", "crop_group"]).assign(y=0)
    f = history.history_features(pd.concat([df, nxt], ignore_index=True)).iloc[len(df) :]
    f = pd.concat([nxt.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    m = f.merge(tabs["hist_muni"], on=["ibge_code", "crop_group"], suffixes=("", "_tab"))
    assert np.allclose(m["hist_muni_cg_rate"], m["hist_muni_cg_rate_tab"])


def test_reference_uses_only_previous_safras():
    from croprisk.features import reference

    rows = []
    for s in range(2015, 2021):
        for v in (1.0, 2.0, 3.0):
            rows.append(
                {
                    "uf": "PR",
                    "crop": "Soja",
                    "safra_year": s,
                    "sum_insured": 1000.0 * (s - 2014) * v,
                    "area_ha": 1.0,
                    "yield_expected": 3000.0 + s,
                }
            )
    df = pd.DataFrame(rows)
    med = reference.reference_medians(df)
    keys = pd.DataFrame({"uf": ["PR", "MT"], "crop": ["Soja", "Soja"], "safra_year": [2020, 2020]})
    ref = reference.reference_for(med, keys)
    # safras 2017-2019 have medians 6000, 8000, 10000 -> 8000; MT falls back to national
    assert ref["siha_ref"].tolist() == [8000.0, 8000.0]
    assert ref["yld_ref"].iloc[0] == 3000 + 2018
