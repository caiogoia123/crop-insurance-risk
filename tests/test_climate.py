import numpy as np
import pandas as pd

from croprisk import config
from croprisk.features import climate


def _series(seed=0):
    dates = pd.date_range(config.POWER_START, "2024-12-31", freq="D")
    rng = np.random.default_rng(seed)
    n = len(dates)
    return dates, {
        "prec": rng.gamma(0.5, 8, n),
        "tmax": 28 + rng.normal(0, 3, n),
        "tmin": 15 + rng.normal(0, 4, n),
        "rh": np.full(n, 70.0),
        "gwet": np.clip(rng.normal(0.6, 0.1, n), 0, 1),
    }


def test_window_stats_known_values():
    arrs = {
        "prec": np.array([0, 0, 0, 5, 0, 0, 10, 0, 0, 0], dtype=float),
        "tmax": np.array([30, 35, 36, 30, 30, 30, 30, 30, 30, 30], dtype=float),
        "tmin": np.array([10, 2, 3, 10, 10, 10, 10, 10, 10, 1], dtype=float),
        "rh": np.full(10, 50.0),
        "gwet": np.linspace(0.5, 0.9, 10),
    }
    st = climate.window_stats(arrs, np.array([0]), 10).iloc[0]
    assert st["prec_sum"] == 15
    assert st["cdd"] == 3  # days 0-2 and 7-9 are the longest dry runs
    assert st["prec_max5d"] == 15  # days 3..7
    assert st["hot_days"] == 2  # >= 34 C
    assert st["frost_days"] == 3  # <= 3 C
    assert np.isclose(st["gwet_min"], 0.5)


def test_window_outside_series_is_nan():
    _, arrs = _series()
    st = climate.window_stats(arrs, np.array([-5, len(arrs["prec"]) - 3]), 10)
    assert st.isna().all().all()


def test_climatology_ignores_years_after_reference_period():
    """Leakage guard: changing weather after 2005 must not change the climatology."""
    dates, arrs = _series()
    base = climate.climatology(arrs, 12, 1, 0, 105)
    late = dates >= "2006-06-01"
    arrs2 = {k: v.copy() for k, v in arrs.items()}
    arrs2["prec"][late] *= 5
    arrs2["tmin"][late] -= 10
    assert climate.climatology(arrs2, 12, 1, 0, 105) == base


def test_observed_anomaly_sign():
    dates, arrs = _series()
    wins = pd.DataFrame({"ws": [pd.Timestamp("2021-12-01")], "wlen": [105]})
    dry = {k: v.copy() for k, v in arrs.items()}
    i0 = int((pd.Timestamp("2021-12-01") - dates[0]).days)
    dry["prec"][i0 : i0 + 105] = 0.0
    f = climate.cell_features(dry, wins, [0.5])
    assert f["obs_h50_prec_anom"].iloc[0] == -1.0
    assert f["obs_full_cdd"].iloc[0] == 105
