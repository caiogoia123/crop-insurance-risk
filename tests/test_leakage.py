"""Leakage audit as tests: outcome fields and post-contract information."""

import pandas as pd
import pytest

from croprisk import backtest, models


@pytest.mark.parametrize("fs", list(models.FEATURE_SETS.values()), ids=lambda f: f.name)
def test_no_outcome_fields_in_any_feature_set(fs):
    assert not set(fs.columns) & models.FORBIDDEN
    assert not any(c.lower().startswith(("indemn", "event", "valor_indeniz")) for c in fs.columns)


@pytest.mark.parametrize("name", ["contract_only", "contract", "pre_season", "pre_season_rate"])
def test_contract_time_models_see_no_season_weather(name):
    cols = models.FEATURE_SETS[name].columns
    assert not [c for c in cols if c.startswith(("obs_", "ante_"))]


def test_contract_models_see_no_climate():
    for name in ("contract_only", "contract"):
        cols = models.FEATURE_SETS[name].columns
        assert not [c for c in cols if c.startswith(("clim_", "oni"))]


def test_rate_only_where_intended():
    with_rate = [n for n, fs in models.FEATURE_SETS.items() if "log_rate" in fs.columns]
    assert with_rate == ["pre_season_rate"]


@pytest.mark.parametrize("gap", [0, 1])
@pytest.mark.parametrize("window", [None, 8])
def test_train_mask_is_strictly_in_the_past(gap, window):
    df = pd.DataFrame({"safra_year": list(range(2006, 2024))})
    for s in backtest.TEST_SAFRAS:
        tr = df[backtest.train_mask(df, s, gap, window)]["safra_year"]
        assert tr.max() <= s - 1 - gap
        if window:
            assert tr.nunique() <= window


def test_tuning_safras_precede_test_safras():
    assert max(backtest.TUNE_SAFRAS) < min(backtest.TEST_SAFRAS)
