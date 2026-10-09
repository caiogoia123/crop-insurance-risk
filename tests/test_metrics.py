import numpy as np

from croprisk import metrics


def test_ks_perfect_and_random():
    y = np.r_[np.zeros(500), np.ones(500)]
    assert metrics.ks_stat(y, y + 0.1) == 1.0
    rng = np.random.default_rng(0)
    assert metrics.ks_stat(y, rng.random(1000)) < 0.1


def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.2
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    m = metrics.binary_metrics(y, p, base_rate=0.2)
    assert np.isclose(m["gini"], 2 * m["auc"] - 1)
    assert 0 < m["brier"] < 0.25
    assert "brier_skill" in m


def test_psi():
    rng = np.random.default_rng(2)
    a = rng.normal(size=20000)
    assert metrics.psi(a, a) < 1e-3
    assert metrics.psi(a, a + 1.0) > 0.25


def test_decile_table():
    rng = np.random.default_rng(3)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = metrics.decile_table(y, p, indemnity=y * 100.0, premium=np.full(1000, 10.0))
    assert len(t) == 10
    assert np.isclose(t["capture"].sum(), 1.0)
    assert t["claim_rate"].iloc[-1] > t["claim_rate"].iloc[0]
    assert t["loss_ratio"].iloc[-1] > t["loss_ratio"].iloc[0]
