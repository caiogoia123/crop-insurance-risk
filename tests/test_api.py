"""API contract tests against a SYNTHETIC model bundle (see conftest)."""

import pytest
from fastapi.testclient import TestClient

from croprisk.serving import app as app_module

VALID = {
    "crop": "Soja",
    "ibge_code": 4104808,
    "contract_date": "2025-09-15",
    "coverage_level": 0.7,
    "area_ha": 120,
    "sum_insured": 350000,
    "yield_expected": 3600,
    "yield_insured": 2520,
    "insurer": "SEGURADORA SINTETICA S.A.",
    "product_class": "PRODUTIVIDADE",
    "rate": 0.065,
}


@pytest.fixture()
def client(bundle_dir, monkeypatch):
    monkeypatch.setattr(app_module, "MODELS_DIR", bundle_dir)
    app_module.get_bundle.cache_clear()
    yield TestClient(app_module.app)
    app_module.get_bundle.cache_clear()


def test_health_and_model_info(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["version"] == h["model_version"]
    assert "auc" in info["holdout"]["metrics"]
    assert info["limitations"]


def test_index_and_docs(client):
    assert "Crop insurance claim risk" in client.get("/").text
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200, r.text
    out = r.json()
    assert 0 < out["probability"] < 1
    assert 1 <= out["risk_decile"] <= 10
    assert out["safra"] == "2025/26"
    assert out["critical_window"]["start"] == "2025-12-01"
    assert len(out["top_factors"]) == 5
    assert 0 < out["rate_implied_probability"] < 1


def test_missing_insurer_is_averaged_over_market_shares(client):
    base = {k: v for k, v in VALID.items() if k not in ("insurer", "product_class")}
    out = client.post("/predict", json=base).json()
    named = [
        client.post("/predict", json={**base, "insurer": i, "product_class": pc}).json()[
            "probability"
        ]
        for i in ("SEGURADORA SINTETICA S.A.", "other")
        for pc in ("PRODUTIVIDADE", "CUSTEIO")
    ]
    assert min(named) - 1e-4 <= out["probability"] <= max(named) + 1e-4


def test_predict_unknown_insurer_and_vegetable(client):
    r = client.post(
        "/predict",
        json={**VALID, "insurer": "NOVA SEGURADORA", "crop": "Tomate", "ibge_code": 5107925},
    )
    assert r.status_code == 200, r.text
    assert r.json()["crop_group"] == "hortalicas"


@pytest.mark.parametrize(
    "patch",
    [
        {"crop": "Banana-da-terra"},
        {"ibge_code": 4199999},
        {"contract_date": "1999-01-01"},
        {"area_ha": -3},
        {"coverage_level": 1.5},
        {"product_class": "PECUARIO"},
        {"rate": 2},
    ],
)
def test_predict_rejects_bad_input(client, patch):
    assert client.post("/predict", json={**VALID, **patch}).status_code == 422


def test_meta_endpoints(client):
    opt = client.get("/meta/options").json()
    assert "Soja" in opt["crops"]
    mun = client.get("/meta/municipalities", params={"uf": "pr"}).json()
    assert mun == [{"ibge_code": 4104808, "name": "Cascavel"}]
    assert client.get("/meta/municipalities", params={"uf": "XX"}).status_code == 404


def test_contributions_explain_the_logit(bundle_dir):
    """Linear SHAP: intercept + coef . mean(z) + sum(contributions) = logit(p)."""
    import numpy as np

    from croprisk.serving.bundle import Bundle, current_dir

    b = Bundle.load(current_dir(bundle_dir))
    X, _ = b.features({**VALID})
    lr = b.model.pipe.named_steps["lr"]
    base = lr.intercept_[0] + float(np.dot(lr.coef_[0], b.model.z_mean))
    p = b.model.predict(X)[0]
    assert np.isclose(base + b.model.contributions(X).sum(), np.log(p / (1 - p)), atol=1e-6)


def test_lightgbm_bundle_still_supported(bundle_dir_lgbm):
    from croprisk.serving.bundle import Bundle, current_dir

    b = Bundle.load(current_dir(bundle_dir_lgbm))
    out = b.predict({**VALID})
    assert 0 < out["probability"] < 1
    assert len(out["top_factors"]) == 5
